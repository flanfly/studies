"""Binance spot exchange adapter.

The public spot market endpoints (``/api/v3/exchangeInfo`` and
``/api/v3/klines``) are unauthenticated. Borrow rates are fetched
from the authenticated ``/sapi/v1/margin/interestRateHistory`` endpoint
and signed with HMAC-SHA256 over the query string.

Binance borrow rates are per-asset (not per-pair): the same rate
applies to a given asset regardless of the pair it is borrowed
through, and the rate is the same for cross and isolated margin. We
cache the rates per ``base`` asset on the first ``pairs()`` call and
populate both ``cross_rate`` and ``isolated_rate`` with the same
annualized value.

Binance klines return both ``open_time`` and ``close_time`` per candle,
so the ``close_ts`` column is taken from the API (inclusive last instant
of the daily candle, in millisecond resolution).
"""

import asyncio
import datetime as dt
import hashlib
import hmac
import logging as l
import random
import time
from typing import Iterable
from urllib.parse import urlencode

import polars as pl
import httpx
from httpx import AsyncClient

from . import _split_symbol, Exchange, TransientError


__all__ = ["Binance"]


# Binance sapi error codes that mean "this asset doesn't have margin
# trading enabled" rather than "this request was malformed / your
# credentials are bad". ``-11027`` is the canonical one. We treat
# 400 / 404 responses with this code as expected and silent.
_BINANCE_ASSET_NOT_SUPPORTED_CODES = frozenset({-11027})


def _parse_binance_error_code(response: httpx.Response) -> int | None:
    """Return the ``code`` field from a Binance JSON error envelope,
    or ``None`` if the body isn't a valid ``{"code": ..., "msg": ...}``
    dict.

    Used by both the silent-skip path (``-11027``) and the
    transient-retry path (``-1003``/``-1015``/``-1021``) to classify
    responses without re-parsing them.
    """
    try:
        body = response.json()
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    code = body.get("code")
    return code if isinstance(code, int) else None


def _is_asset_not_supported(response: httpx.Response, asset: str) -> bool:
    """Return ``True`` if ``response`` is Binance's "asset not
    supported for margin" envelope.

    Binance returns a JSON body of the form
    ``{"code": -11027, "msg": "asset X is not supported"}`` for any
    asset that doesn't have margin trading enabled. The status
    code is 400 in practice, but we also accept 404 to be defensive
    against future API changes (and to silence the test-suite
    "asset not found" response the user reported).

    Malformed / non-JSON bodies return ``False``; we err on the
    side of letting the warning fire for those, since they
    indicate a real failure mode (auth, network, etc.) rather
    than a per-asset not-supported case.
    """
    try:
        body = response.json()
    except Exception:
        return False
    if not isinstance(body, dict):
        return False
    code = body.get("code")
    msg = body.get("msg", "")
    if code not in _BINANCE_ASSET_NOT_SUPPORTED_CODES:
        return False
    # The ``msg`` always mentions the asset name; check the asset
    # so we don't accidentally silence a different ``-11027`` error
    # that happens to share the code.
    return asset.upper() in str(msg).upper()


async def _fetch_with_semaphore(
    sem: asyncio.Semaphore,
    adapter: "Binance",
    client: AsyncClient,
    asset: str,
) -> float | None:
    """Wrap ``adapter._fetch_borrow_rate(client, asset)`` with
    ``sem`` so ``_load_borrow_rates`` can cap in-flight requests.

    Module-level (not a method) so ``asyncio.gather`` can be called
    with it as a free coroutine.
    """
    async with sem:
        return await adapter._fetch_borrow_rate(client, asset)


class Binance(Exchange):
    """Binance spot exchange adapter."""

    HOST = "api.binance.com"
    SAPI_HOST = "api.binance.com"  # no separate demo sapi
    # USDⓈ-M futures (USDT- and USDC-margined perpetuals) live on a
    # separate host. The public funding endpoint here is used for
    # the ``funding_rate`` column on ``pairs()``.
    FAPI_HOST = "fapi.binance.com"
    NAME = "binance"
    # Max daily candles returned by a single ``klines()`` call. The
    # ``/api/v3/klines`` endpoint caps the response at 1000 candles
    # per request, and the actual historical retention for major
    # pairs (e.g. BTCUSDT) goes back to 2017-08-17 -- about 8.75
    # years. Use ``klines_paged()`` to fetch a wider range.
    MAX_KLINES = 1000

    # Binance error codes that warrant a retry (rate limiting,
    # service-side throttling). Other error codes are propagated
    # immediately.
    TRANSIENT_CODES = frozenset({
        -1003,  # TOO_MANY_REQUESTS
        -1015,  # too many new orders / rate limit
        -1007,  # timeout waiting for response from backend
        -1021,  # timestamp outside recvWindow (transient clock issues)
    })

    # Error code returned by the futures premiumIndex endpoint when
    # the symbol doesn't exist. We treat this as a silent per-asset
    # "null" rather than a failure (most spot pairs simply have no
    # perpetual contract on Binance).
    _BINANCE_INVALID_SYMBOL_CODE = -1121

    # All Binance USDⓈ-M perpetuals use an 8h funding interval. The
    # ``fundingInfo`` endpoint confirms this for every contract.
    _FUNDING_INTERVAL_HOURS: float = 8.0
    # Cap on in-flight ``_fetch_funding_rate`` calls. The public
    # ``/fapi/v1/premiumIndex`` endpoint is rate-limited at 2400
    # calls/min/IP, but each call is also gated by the order-book
    # weight (typically 1-2), so 30-wide is a safe default that
    # finishes ~430 bases in <5s.
    _FUNDING_CONCURRENCY: int = 30
    # Backoff schedule for per-call retries on transient errors
    # (429, 5xx, transport). Inherits the base-class default
    # (100ms → 10s, infinite attempts); the per-adapter constants
    # are gone -- ``_funding_request`` uses ``self.RETRY_BASE_DELAY``
    # and ``self.RETRY_MAX_DELAY`` directly.

    # Max in-flight borrow-rate fetches. The sapi endpoint is much
    # more rate-limited than the public endpoints, and firing 400+
    # concurrent requests causes a wave of ``-1021`` ("timestamp
    # outside recvWindow") errors that swamp the log. 50 is well
    # within sapi's per-minute budget and keeps the log clean.
    BORROW_RATE_CONCURRENCY: int = 50

    def __init__(self, api_key: str, api_secret: str):
        self._api_key = api_key
        self._api_secret = api_secret
        # Populated by ``_load_borrow_rates`` on first ``pairs()`` call:
        # ``{asset_upper: annualized_rate}``. None when credentials are
        # missing or the sapi call fails.
        self._borrow_rate_cache: dict[str, float] | None = None
        # Cap on in-flight ``_fetch_borrow_rate`` calls. Created
        # lazily so it binds to the running event loop on first use.
        self._borrow_rate_sem: asyncio.Semaphore | None = None

    # ------------------------------------------------------------------
    # signing helpers
    # ------------------------------------------------------------------
    def _sign(self, params: dict) -> dict:
        """Add ``timestamp`` and ``signature`` to a query-string param dict.
        Binance uses HMAC-SHA256 over the canonical query string built
        from the same params.
        """
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        query = urlencode(params)
        sig = hmac.new(
            self._api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = sig
        return params

    async def _sapi_get(self, client: AsyncClient, path: str, params: dict) -> object:
        """Make an authenticated GET against the sapi (signed) host."""
        signed = self._sign(params)
        headers = {"X-MBX-APIKEY": self._api_key}
        url = f"https://{self.SAPI_HOST}{path}"
        resp = await client.get(url, params=signed, headers=headers, timeout=30.0)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # borrow rate fetchers
    # ------------------------------------------------------------------
    async def _fetch_borrow_rate(self, client: AsyncClient, asset: str) -> float | None:
        """Return the most recent daily interest rate for ``asset``,
        annualized as ``daily * 365``. Returns None on auth / parse
        failure so ``pairs()`` degrades to null rates gracefully.

        Response classification:

        * ``400`` / ``404`` with ``code == -11027`` ("asset X is not
          supported") is the expected response for any asset that
          doesn't have margin trading enabled. We return ``None``
          silently -- it's not a real error, just noise in the log.

        * 4xx / 5xx with a code in ``TRANSIENT_CODES`` (``-1003``
          TOO_MANY_REQUESTS, ``-1015`` rate limit, ``-1007``
          backend timeout, ``-1021`` "timestamp outside recvWindow")
          is retried with exponential backoff + jitter. ``-1021`` in
          particular fires sporadically when 50+ concurrent requests
          get queued by httpx and the signed timestamp goes stale
          by the time the server sees it -- a fresh sign + retry
          fixes it.

        * Any other 4xx / 5xx is a real failure (auth, signature,
          programming error). We log a warning and return ``None``
          so the rest of the batch can proceed.
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                data = await self._sapi_get(
                    client,
                    "/sapi/v1/margin/interestRateHistory",
                    {"asset": asset.upper()},
                )
                # Success.
                if not isinstance(data, list) or not data:
                    return None
                try:
                    daily = float(data[0]["dailyInterestRate"])
                except (KeyError, TypeError, ValueError):
                    return None
                return daily * 365
            except httpx.HTTPStatusError as e:
                resp = e.response
                code = _parse_binance_error_code(resp)
                # ``-11027`` "asset X is not supported" is the expected
                # response for any asset that isn't marginable. Silent.
                if (
                    resp.status_code in (400, 404)
                    and code in _BINANCE_ASSET_NOT_SUPPORTED_CODES
                    and asset.upper() in str(resp.json().get("msg", "")).upper()
                ):
                    return None
                # Transient: rate limit, recvWindow, backend timeout.
                if resp.status_code in (400, 429) and code in self.TRANSIENT_CODES:
                    delay = min(
                        self.RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                        self.RETRY_MAX_DELAY,
                    )
                    delay = delay * (0.75 + random.random() * 0.5)
                    l.warning(
                        "Binance borrowRate(%s) transient error "
                        "(attempt %d): %s; retrying in %.1fs "
                        "(infinite retries)",
                        asset,
                        attempt,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                # Real failure: log once, return None.
                l.warning(f"Binance borrowRate({asset}) failed: {e}")
                return None
            except Exception as e:
                # Transport / DNS / TLS / etc. The base ``is_transient_error``
                # recognises the common httpx types.
                if not self.is_transient_error(e):
                    l.warning(f"Binance borrowRate({asset}) failed: {e}")
                    return None
                delay = min(
                    self.RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                    self.RETRY_MAX_DELAY,
                )
                delay = delay * (0.75 + random.random() * 0.5)
                l.warning(
                    "Binance borrowRate(%s) transient error "
                    "(attempt %d): %s; retrying in %.1fs "
                    "(infinite retries)",
                    asset,
                    attempt,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)

    async def _load_borrow_rates(
        self, client: AsyncClient, bases: Iterable[str]
    ) -> dict[str, float]:
        """``{base_lower: annual_rate}`` for each base asset, fetched
        concurrently and cached on the instance.

        Concurrency is capped at ``BORROW_RATE_CONCURRENCY`` via an
        ``asyncio.Semaphore`` so we don't trip Binance's sapi rate
        limit (which manifests as ``-1021`` "timestamp outside
        recvWindow" -- a side effect of the signed timestamp on a
        request going stale while hundreds of them are queued). The
        actual fetch count for ``quote_assets=USDT`` is ~430.
        """
        if self._borrow_rate_cache is not None:
            return self._borrow_rate_cache
        bases = sorted({b for b in bases})
        rates: dict[str, float] = {}
        if self._borrow_rate_sem is None:
            self._borrow_rate_sem = asyncio.Semaphore(
                self.BORROW_RATE_CONCURRENCY
            )
        sem = self._borrow_rate_sem
        results = await asyncio.gather(
            *(_fetch_with_semaphore(sem, self, client, b) for b in bases)
        )
        for b, r in zip(bases, results):
            if r is not None:
                rates[b.lower()] = r
        self._borrow_rate_cache = rates
        return rates

    # ------------------------------------------------------------------
    # funding rate fetcher
    # ------------------------------------------------------------------
    async def _funding_request(
        self, client: AsyncClient, symbol: str
    ) -> dict | None:
        """Make a single Binance funding-rate HTTP request with
        infinite per-call retry on transient errors (429, 5xx,
        transport). Returns the parsed JSON payload on success,
        ``None`` on a permanent failure.
        """
        url = (
            f"https://{self.FAPI_HOST}/fapi/v1/premiumIndex"
            f"?symbol={symbol}"
        )
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = await client.get(url, timeout=30.0)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    delay = min(
                        self.RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                        self.RETRY_MAX_DELAY,
                    )
                    delay = delay * (0.75 + random.random() * 0.5)
                    l.warning(
                        "Binance funding_rate(%s) transient error "
                        "(attempt %d): %s; retrying in %.1fs "
                        "(infinite retries)",
                        symbol,
                        attempt,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                # Non-transient HTTP error: log once, return None.
                l.warning(f"Binance funding_rate({symbol}) HTTP error: {e}")
                return None
            except Exception as e:
                if not self.is_transient_error(e):
                    raise RuntimeError(
                        f"Binance funding_rate request failed: {e}"
                    ) from e
                delay = min(
                    self.RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                    self.RETRY_MAX_DELAY,
                )
                delay = delay * (0.75 + random.random() * 0.5)
                l.warning(
                    "Binance funding_rate(%s) transient error "
                    "(attempt %d): %s; retrying in %.1fs "
                    "(infinite retries)",
                    symbol,
                    attempt,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)

    async def _fetch_funding_rate(
        self, client: AsyncClient, base: str, quote: str
    ) -> float | None:
        """Return the most recent funding rate (annualised APR) for the
        Binance USDⓈ-M perpetual on ``base`` / ``quote``, or ``None``
        if no such contract exists or the request fails for a
        non-transient reason.

        Endpoint: ``GET /fapi/v1/premiumIndex?symbol={base}{quote}``
        on ``fapi.binance.com`` (unauthenticated, public). Response
        shape::

            {
              "symbol": "BTCUSDT",
              "markPrice": "...",
              "lastFundingRate": "-0.00003376",
              "nextFundingTime": ...,
              "time": ...
            }

        We probe USDT-settled perps first; if Binance returns
        ``-1121`` ("Invalid symbol"), we fall back to USDC-
        settled (``BTCUSDC``) and accept whatever the response
        reports. We deliberately skip the COIN-M (inverse-
        margined) endpoint on ``dapi.binance.com`` -- the user
        wants stablecoin-settled perpetuals only.

        A response with ``code == -1121`` for both USDT and USDC
        means the base has no perpetual contract at all; we
        return ``None`` silently. Transient errors (429, 5xx,
        transport) are retried with per-call backoff in
        :meth:`_funding_request`.
        """
        for settle in (quote.upper(), "USDC" if quote.upper() == "USDT" else "USDT"):
            symbol = f"{base.upper()}{settle}"
            payload = await self._funding_request(client, symbol)
            if payload is None:
                return None

            if not isinstance(payload, dict):
                continue
            code = payload.get("code")
            # ``code`` is absent on success. If present and an int,
            # it follows the Binance error envelope convention.
            if isinstance(code, int) and code == self._BINANCE_INVALID_SYMBOL_CODE:
                continue
            if isinstance(code, int) and code in self.TRANSIENT_CODES:
                # ``_funding_request`` would have retried these
                # already; if we still get one, give up on this
                # settle-currency.
                return None
            if isinstance(code, int) and code != 0:
                # Real error: log once, move on.
                l.warning(
                    f"Binance funding_rate({symbol}) error: {payload}"
                )
                return None
            try:
                rate = float(payload["lastFundingRate"])
            except (KeyError, TypeError, ValueError):
                return None
            return self.annualize_funding_rate(
                rate, self._FUNDING_INTERVAL_HOURS
            )

        # All settle currencies tried returned -1121: no perpetual
        # contract for this base on Binance.
        return None

    async def _fetch_funding_rates(
        self, client: AsyncClient, bases: Iterable[str]
    ) -> dict[str, float]:
        """``{base_lower: annual_funding_rate}`` for each base asset
        that has a Binance USDⓈ-M perpetual. Concurrent fetch via
        ``asyncio.gather``, bounded by ``_FUNDING_CONCURRENCY`` to
        stay under the per-IP rate limit on the public
        ``/fapi/v1/premiumIndex`` endpoint.
        """
        bases = sorted({b.lower() for b in bases})
        if not bases:
            return {}

        semaphore = asyncio.Semaphore(self._FUNDING_CONCURRENCY)

        async def _one(b: str) -> tuple[str, float | None]:
            async with semaphore:
                try:
                    rate = await self._fetch_funding_rate(
                        client, base=b, quote="USDT"
                    )
                except Exception as e:
                    l.warning(f"Binance funding_rate({b}) failed: {e}")
                    return b, None
            return b, rate

        results = await asyncio.gather(*(_one(b) for b in bases))
        return {b: r for b, r in results if r is not None}

    # ------------------------------------------------------------------
    # pairs
    # ------------------------------------------------------------------
    async def pairs(self, client: AsyncClient, quote_assets: set[str]) -> pl.DataFrame:
        """Returns active USDT/USDC spot pairs with cross/isolated
        margin borrow rates and the current perpetual-futures
        funding rate APR.

        Columns: ts, symbol, exchange, base, quote, cross_rate,
        isolated_rate, funding_rate

        ``cross_rate`` and ``isolated_rate`` are populated from the
        per-asset ``interestRateHistory`` endpoint (the same rate
        applies to both cross and isolated margin on Binance).
        ``funding_rate`` is the most recent Binance USDⓈ-M
        perpetual funding payment, annualised as
        ``rate * 3 * 365`` (Binance pays funding every 8h on the
        standard linear perp). It's ``None`` for bases that have
        no USDT- or USDC-margined perpetual on Binance.
        """
        now = dt.datetime.now(dt.timezone.utc)
        try:
            resp = await client.get(
                f"https://{self.HOST}/api/v3/exchangeInfo?permissions=SPOT",
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            raise RuntimeError(f"Binance exchangeInfo request failed: {e}") from e

        quote_set = {q.upper() for q in quote_assets}
        active_bases: set[str] = set()
        rows: list[dict] = []
        for s in data.get("symbols", []):
            if s.get("status") != "TRADING":
                continue
            base = s.get("baseAsset", "")
            quote = s.get("quoteAsset", "")
            if not base or quote not in quote_set:
                continue
            active_bases.add(base)
            rows.append(
                {
                    "ts": now,
                    "symbol": s["symbol"],
                    "exchange": self.NAME,
                    "base": base.lower(),
                    "quote": quote.lower(),
                    "cross_rate": None,
                    "isolated_rate": None,
                }
            )

        if rows:
            rates = await self._load_borrow_rates(client, active_bases)
            funding_rates = await self._fetch_funding_rates(
                client, active_bases
            )
            for r in rows:
                rate = rates.get(r["base"])
                r["cross_rate"] = rate
                r["isolated_rate"] = rate
                r["funding_rate"] = funding_rates.get(r["base"])

        if not rows:
            return self.empty_pairs_df()

        df = pl.DataFrame(rows)
        df = df.select(
            pl.col("ts").cast(pl.Datetime("us", time_zone="UTC")),
            "symbol",
            "exchange",
            "base",
            "quote",
            pl.col("cross_rate").cast(pl.Float64),
            pl.col("isolated_rate").cast(pl.Float64),
            pl.col("funding_rate").cast(pl.Float64),
        )
        self.validate_pairs_df(df)
        return df

    # ------------------------------------------------------------------
    # klines
    # ------------------------------------------------------------------
    async def klines(
        self,
        client: AsyncClient,
        symbol: str,
        start_time: dt.datetime,
        end_time: dt.datetime,
    ) -> pl.DataFrame:
        """Fetch at most ``MAX_KLINES`` daily klines for ``symbol`` in the
        half-open range ``[start_time, end_time)``.

        Columns: open_ts, close_ts, symbol, exchange, base, quote, open, high,
        low, close, base_volume, quote_volume

        ``start_time`` is inclusive, ``end_time`` is exclusive: only candles
        whose open time ``open_ts`` satisfies ``start_time <= open_ts < end_time``
        are returned. ``close_ts`` is taken straight from the API (inclusive
        last instant of the daily candle, in millisecond resolution; Binance
        reports ``close_time = open_time + 24h - 1ms``). The result is
        capped at ``MAX_KLINES`` daily candles; use ``klines_paged()`` for
        wider ranges.
        """
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=dt.timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=dt.timezone.utc)

        # Binance symbols are concatenated upper case (e.g. ``BTCUSDT``).
        symbol = symbol.upper()
        base, quote = _split_symbol(symbol)
        end_ms = int(end_time.timestamp() * 1000)
        # The per-page cap is 1000 candles. Clip the request window to
        # ``MAX_KLINES`` days from the end so the response always fits in
        # one call.
        window_start = end_time - dt.timedelta(
            seconds=self.MAX_KLINES * self.DAILY_SECONDS
        )
        start_ms = int(max(start_time, window_start).timestamp() * 1000)

        url = f"https://{self.HOST}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": start_ms,
            "endTime": end_ms - 1,
            "limit": 1000,
        }
        try:
            resp = await client.get(url, params=params, timeout=30.0)
            resp.raise_for_status()
            batch = resp.json()
        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            raise RuntimeError(f"Binance klines request failed: {e}") from e
        if not isinstance(batch, list):
            # Binance returns a ``{"code": ..., "msg": ...}`` envelope on
            # application errors. Classify the well-known transient
            # codes; everything else is a permanent failure.
            if isinstance(batch, dict) and batch.get("code") in self.TRANSIENT_CODES:
                raise TransientError(
                    f"Binance klines transient: {batch}"
                )
            raise RuntimeError(f"Binance klines error: {batch}")

        # Cap at MAX_KLINES rows (the API may return a touch more on the
        # edge of a window).
        batch = batch[: self.MAX_KLINES]

        rows: list[dict] = []
        for k in batch:
            # [open_time, open, high, low, close, base_vol, close_time,
            #  quote_vol, trades, taker_buy_base, taker_buy_quote, _]
            k_open_ms = int(k[0])
            if k_open_ms < int(start_time.timestamp() * 1000) or k_open_ms >= end_ms:
                continue
            open_ts = dt.datetime.fromtimestamp(k_open_ms / 1000.0, tz=dt.timezone.utc)
            # ``close_time`` is the inclusive last instant of the daily
            # candle; cast from ms-resolution to microsecond.
            close_ts = dt.datetime.fromtimestamp(int(k[6]) / 1000.0, tz=dt.timezone.utc)
            rows.append(
                {
                    "open_ts": open_ts,
                    "close_ts": close_ts,
                    "symbol": symbol,
                    "base": base,
                    "quote": quote,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "base_volume": float(k[5]),
                    "quote_volume": float(k[7]),
                }
            )

        if not rows:
            return self.empty_klines_df()

        df = (
            pl.DataFrame(rows)
            .unique(subset=["open_ts", "symbol"], keep="last")
            .sort("open_ts")
            .select(
                pl.col("open_ts").cast(pl.Datetime("us", time_zone="UTC")),
                pl.col("close_ts").cast(pl.Datetime("us", time_zone="UTC")),
                "symbol",
                pl.lit(self.NAME).alias("exchange"),
                "base",
                "quote",
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("base_volume").cast(pl.Float64),
                pl.col("quote_volume").cast(pl.Float64),
            )
        )
        self.validate_klines_df(df)
        return df
