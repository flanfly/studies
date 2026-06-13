"""KuCoin spot exchange adapter.

Authenticated endpoints (``/api/v3/margin/borrowRate``) use the
KC-API-* v2 signing scheme (HmacSHA256 over
``timestamp + method + endpoint + body``).
"""

import asyncio
import base64
import datetime as dt
import hashlib
import hmac
import logging as l
import random
import time

from typing import Iterable

import httpx
import polars as pl
from httpx import AsyncClient

from . import _split_symbol, Exchange, TransientError


__all__ = ["KuCoin"]


class KuCoin(Exchange):
    """KuCoin spot exchange adapter."""

    HOST = "api.kucoin.com"
    # KuCoin's contract (perpetual futures) API lives on a separate
    # host. Public, unauthenticated, well-behaved -- the funding
    # endpoint is rate-limited at ~2 calls / second per IP.
    FUTURES_HOST = "api-futures.kucoin.com"
    NAME = "kucoin"
    # Max daily candles returned by a single ``klines()`` call. The
    # ``/api/v1/market/candles`` endpoint uses ``startAt``/``endAt``
    # (in seconds) to scope the response and returns at most 1500
    # daily candles per request. The actual historical depth (verified
    # for BTC-USDT) goes back to 2017-10-19 -- about 8.6 years; use
    # ``klines_paged()`` to fetch a wider range.
    MAX_KLINES = 1500

    # KuCoin error codes that warrant a retry (rate limiting,
    # service-side throttling). Other error codes are propagated
    # immediately.
    TRANSIENT_CODES = frozenset({
        "429000",  # too many requests
        "300012",  # service busy
        "500000",  # internal server error
    })

    # KuCoin error code returned by the funding endpoint when the
    # contract doesn't have a funding rate (e.g. inverse-margined
    # contracts, or a symbol that doesn't exist). We treat it as a
    # silent per-asset "null" rather than a failure.
    _KUCOIN_FUNDING_NOT_SUPPORTED = "415000"

    # Spot ticker -> KuCoin contract base code. Most coins map 1:1
    # (e.g. ETH spot -> ETH contract), but BTC uses KuCoin's
    # internal ``XBT`` code on the contract side. Anything not in
    # this map is assumed to map to itself (upper-cased).
    _SPOT_TO_CONTRACT_BASE: dict[str, str] = {
        "btc": "XBT",
    }

    # KuCoin's standard funding interval for linear perps is 8h
    # (granularity=28800000ms). The response also reports it in
    # milliseconds; we use the constant as the default and trust
    # the response if it differs (e.g. for coins KuCoin has moved
    # to a 4h schedule). This matches the "stablecoin-settled
    # perpetuals only" convention used across all exchanges in
    # this codebase: USDT-margined or USDC-margined perps, never
    # inverse-coin-margined ones.
    _FUNDING_INTERVAL_HOURS: float = 8.0
    # Cap on in-flight ``_fetch_funding_rate`` calls. KuCoin's
    # futures public API is rate-limited at 2000 calls per 30s
    # per IP -- that's ~67/s. Firing 8-wide with no per-call
    # pacing exhausts the window in ~12s for our 945-base batch.
    # A 4-wide cap keeps us at ~25/s and well under the limit.
    _FUNDING_CONCURRENCY: int = 4
    # Backoff schedule for per-call retries on 429 / transient
    # errors. Inherits the base-class default (100ms → 10s,
    # infinite attempts); the per-adapter constants are gone --
    # ``_funding_request`` uses ``self.RETRY_BASE_DELAY`` and
    # ``self.RETRY_MAX_DELAY`` directly.

    def __init__(self, api_key: str, api_secret: str, api_password: str):
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_password = api_password

    # ------------------------------------------------------------------
    # signing helpers
    # ------------------------------------------------------------------
    def _sign(self, timestamp: str, method: str, endpoint: str, body: str) -> str:
        payload = timestamp + method.upper() + endpoint + body
        return base64.b64encode(
            hmac.new(
                self._api_secret.encode(), payload.encode(), hashlib.sha256
            ).digest()
        ).decode()

    def _passphrase_sign(self) -> str:
        return base64.b64encode(
            hmac.new(
                self._api_secret.encode(),
                self._api_password.encode(),
                hashlib.sha256,
            ).digest()
        ).decode()

    def _headers(self, method: str, endpoint: str, body: str = "") -> dict:
        now = str(int(time.time() * 1000))
        return {
            "KC-API-KEY": self._api_key,
            "KC-API-SIGN": self._sign(now, method, endpoint, body),
            "KC-API-TIMESTAMP": now,
            "KC-API-PASSPHRASE": self._passphrase_sign(),
            "KC-API-KEY-VERSION": "2",
        }

    # ------------------------------------------------------------------
    # borrow rate fetchers
    # ------------------------------------------------------------------
    async def _fetch_borrow_rates(
        self, client: AsyncClient, currencies: list[str]
    ) -> dict[str, float]:
        """{base_lower: annual_rate} for a list of currencies, batched."""
        rates: dict[str, float] = {}
        batch_size = 50
        for i in range(0, len(currencies), batch_size):
            batch = currencies[i : i + batch_size]
            endpoint = f"/api/v3/margin/borrowRate?currency={','.join(batch)}"
            headers = self._headers("GET", endpoint)
            try:
                resp = await client.get(
                    f"https://{self.HOST}{endpoint}", headers=headers, timeout=30.0
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                l.warning(f"KuCoin borrowRate batch failed: {e}")
                continue
            if data.get("code") != "200000":
                l.warning(f"KuCoin borrowRate API error: {data}")
                continue
            for item in data.get("data", {}).get("items", []):
                rates[item["currency"].lower()] = float(item["annualizedBorrowRate"])
        return rates

    async def _fetch_cross_pairs(
        self, client: AsyncClient
    ) -> list[tuple[str, str, str]]:
        """[(base, quote, symbol)] for cross margin USDT/USDC pairs."""
        try:
            resp = await client.get(
                f"https://{self.HOST}/api/v3/margin/symbols", timeout=30.0
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            l.warning(f"KuCoin cross margin symbols failed: {e}")
            return []
        if data.get("code") != "200000":
            l.warning(f"KuCoin cross margin symbols error: {data}")
            return []

        items = data.get("data", [])
        if isinstance(items, dict):
            items = items.get("items", [])

        pairs: list[tuple[str, str, str]] = []
        for s in items:
            sym = s.get("symbol", "")
            if not sym.upper().endswith("-USDT") and not sym.upper().endswith("-USDC"):
                continue
            if not s.get("enableTrading", False):
                continue
            base, _, quote = sym.partition("-")
            pairs.append((base, quote, sym))
        return pairs

    async def _fetch_isolated_pairs(
        self, client: AsyncClient
    ) -> list[tuple[str, str, str]]:
        """[(base, quote, symbol)] for isolated margin USDT/USDC pairs."""
        try:
            resp = await client.get(
                f"https://{self.HOST}/api/v1/isolated/symbols", timeout=30.0
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            l.warning(f"KuCoin isolated margin symbols failed: {e}")
            return []
        if data.get("code") != "200000":
            l.warning(f"KuCoin isolated margin symbols error: {data}")
            return []

        items = data.get("data", [])
        if isinstance(items, dict):
            items = items.get("items", [])

        pairs: list[tuple[str, str, str]] = []
        for s in items:
            sym = s.get("symbol", "")
            if not sym.upper().endswith("-USDT") and not sym.upper().endswith("-USDC"):
                continue
            if not s.get("tradeEnable", False):
                continue
            base, _, quote = sym.partition("-")
            pairs.append((base, quote, sym))
        return pairs

    # ------------------------------------------------------------------
    # funding rate fetcher
    # ------------------------------------------------------------------
    def _contract_symbol(self, base: str, quote: str = "USDT") -> str:
        """Spot ticker -> KuCoin contract symbol (e.g. ``btc`` /
        ``USDT`` -> ``XBTUSDTM``). The ``M`` suffix marks
        USDT/USDC-margined linear perps; inverse-margined contracts
        (settled in the base currency) use a different suffix and
        are deliberately skipped here.

        We probe USDT-margined first; if the contract doesn't
        exist, fall back to USDC-margined. This keeps the lookup
        robust as KuCoin adds new contracts.
        """
        contract_base = self._SPOT_TO_CONTRACT_BASE.get(
            base.lower(), base.upper()
        )
        return f"{contract_base}{quote.upper()}M"

    async def _funding_request(
        self, client: AsyncClient, contract: str
    ) -> dict | None:
        """Make a single KuCoin funding-rate HTTP request with
        infinite per-call retry on transient errors (429, 5xx,
        transport errors). Returns the parsed JSON payload on
        success, ``None`` on a permanent failure.

        429s come with a ``gw-ratelimit-reset`` header (ms until
        the per-30s window resets) which we honour directly
        rather than blind-backoff. That keeps the per-call
        latency low for transient blips while still respecting
        the rate limit on sustained traffic. The hint is capped
        at ``RETRY_MAX_DELAY`` (10s by default) so a buggy header
        value can't hang us.
        """
        url = (
            f"https://{self.FUTURES_HOST}/api/v1/funding-rate/"
            f"{contract}/current"
        )
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = await client.get(url, timeout=30.0)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Honour the gateway's reset hint, but cap it
                    # at ``RETRY_MAX_DELAY`` so a buggy header
                    # value can't hang us for the full window.
                    reset_ms = 0
                    try:
                        reset_ms = int(
                            e.response.headers.get(
                                "gw-ratelimit-reset", "0"
                            )
                        )
                    except (TypeError, ValueError):
                        pass
                    delay = max(
                        self.RETRY_BASE_DELAY,
                        min(
                            reset_ms / 1000.0,
                            self.RETRY_MAX_DELAY,
                        ),
                    )
                    delay = delay * (0.75 + random.random() * 0.5)
                    l.warning(
                        "KuCoin funding_rate(%s) 429 "
                        "(attempt %d): reset hint %dms, "
                        "sleeping %.1fs (infinite retries)",
                        contract,
                        attempt,
                        reset_ms,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                # Non-transient HTTP error: log once, return None.
                l.warning(f"KuCoin funding_rate({contract}) HTTP error: {e}")
                return None
            except Exception as e:
                # Transport / DNS / TLS / etc. Defer to the
                # base-class transient classifier.
                if not self.is_transient_error(e):
                    raise RuntimeError(
                        f"KuCoin funding_rate request failed: {e}"
                    ) from e
                delay = min(
                    self.RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                    self.RETRY_MAX_DELAY,
                )
                delay = delay * (0.75 + random.random() * 0.5)
                l.warning(
                    "KuCoin funding_rate(%s) transient error "
                    "(attempt %d): %s; retrying in %.1fs "
                    "(infinite retries)",
                    contract,
                    attempt,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)

    async def _fetch_funding_rate(
        self, client: AsyncClient, base: str, quote: str
    ) -> float | None:
        """Return the most recent funding rate (annualised APR) for the
        KuCoin USDT- or USDC-margined linear swap on ``base`` /
        ``quote``, or ``None`` if no such contract exists or the
        request fails for a non-transient reason.

        Endpoint: ``GET /api/v1/funding-rate/{CONTRACT}/current``
        on ``api-futures.kucoin.com`` (unauthenticated, public;
        different host from spot). Response shape::

            {
              "code": "200000",
              "data": {
                  "symbol": ".XBTUSDTMFPI8H",
                  "granularity": 28800000,    // ms
                  "timePoint": 1781049600000,
                  "value": -3.0e-5,           // per-interval rate
                  "fundingTime": 1781078400000
              }
            }

        An envelope with ``code == "415000"`` ("funding rate is
        not supported") is the expected response for: (a) a base
        with no contract at all on KuCoin, (b) a base whose only
        contract is inverse-margined, (c) a USDC contract when
        the caller is probing USDT. We try the next settle
        currency in that case and only return ``None`` if all
        attempts 415000.

        Transient errors (429, 5xx, transport) are retried with
        per-call backoff in :meth:`_funding_request`.
        """
        for settle in (quote.upper(), "USDC" if quote.upper() == "USDT" else "USDT"):
            contract = self._contract_symbol(base, settle)
            payload = await self._funding_request(client, contract)
            if payload is None:
                return None

            if not isinstance(payload, dict):
                continue
            code = payload.get("code")
            if code in self.TRANSIENT_CODES:
                # ``_funding_request`` would have retried these
                # already; if we still get one, give up on this
                # settle-currency.
                return None
            if code == self._KUCOIN_FUNDING_NOT_SUPPORTED:
                continue
            if code != "200000":
                l.warning(
                    f"KuCoin funding_rate({contract}) error: {payload}"
                )
                return None

            data = payload.get("data") or {}
            try:
                rate = float(data["value"])
            except (KeyError, TypeError, ValueError):
                return None
            # Use the response's ``granularity`` if present (KuCoin
            # occasionally moves individual contracts to a different
            # funding cadence), else fall back to the class default.
            granularity_ms = data.get("granularity")
            if isinstance(granularity_ms, (int, float)) and granularity_ms > 0:
                interval_hours = granularity_ms / 3_600_000.0
            else:
                interval_hours = self._FUNDING_INTERVAL_HOURS
            return self.annualize_funding_rate(rate, interval_hours)

        # All settle currencies tried returned 415000 (no contract
        # for this base). Return None silently -- the spot pair
        # simply has no stablecoin-margined perpetual on KuCoin.
        return None

    async def _fetch_funding_rates(
        self, client: AsyncClient, bases: Iterable[str]
    ) -> dict[str, float]:
        """``{base_lower: annual_funding_rate}`` for each base asset
        that has a KuCoin USDT- or USDC-margined linear swap.
        Concurrent fetch via ``asyncio.gather``, bounded by
        ``_FUNDING_CONCURRENCY`` to stay under KuCoin's per-IP
        rate limit (~2 calls/sec for the funding endpoint).
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
                    l.warning(f"KuCoin funding_rate({b}) failed: {e}")
                    return b, None
            return b, rate

        results = await asyncio.gather(*(_one(b) for b in bases))
        return {b: r for b, r in results if r is not None}

    # ------------------------------------------------------------------
    # pairs
    # ------------------------------------------------------------------
    async def pairs(self, client: AsyncClient, quote_assets: set[str]) -> pl.DataFrame:
        """Returns active USDT/USDC spot pairs with cross/isolated
        margin rates and the current perpetual-futures funding rate
        APR.

        Columns: ts, symbol, exchange, base, quote, cross_rate,
        isolated_rate, funding_rate

        ``funding_rate`` is the most recent KuCoin stablecoin-
        margined linear swap (USDT- or USDC-settled) funding
        payment, annualised as ``rate * 3 * 365`` (KuCoin pays
        funding every 8h on the standard linear perp). It's
        ``None`` for bases that have no USDT- or USDC-margined
        perpetual on KuCoin.
        """
        now = dt.datetime.now(dt.timezone.utc)
        # Active spot symbols (public). Filter by enableTrading + quote asset.
        try:
            resp = await client.get(f"https://{self.HOST}/api/v2/symbols", timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            raise RuntimeError(f"KuCoin symbols request failed: {e}") from e
        if data.get("code") != "200000":
            if data.get("code") in self.TRANSIENT_CODES:
                raise TransientError(
                    f"KuCoin symbols transient: {data}"
                )
            raise RuntimeError(f"KuCoin symbols API error: {data}")

        # Cross and isolated margin pairs (used purely to fetch borrow
        # rates; we no longer filter spot pairs to those that have margin
        # trading – pairs without margin trading get null rates).
        _cross_pairs, iso_pairs = await asyncio.gather(
            self._fetch_cross_pairs(client),
            self._fetch_isolated_pairs(client),
        )

        # The borrowRate endpoint is case-sensitive – currencies must be
        # upper case. Keep separate upper-case lists for the API call while
        # using lower-case keys for the response lookup.
        cross_bases_upper = sorted({b.upper() for b, _, _ in _cross_pairs})
        iso_bases_upper = sorted({b.upper() for b, _, _ in iso_pairs})

        async def _empty_rates() -> dict[str, float]:
            return {}

        # Borrow rates are currency-level, not symbol-level.
        cross_rates_raw, iso_rates_raw = await asyncio.gather(
            (
                self._fetch_borrow_rates(client, cross_bases_upper)
                if cross_bases_upper
                else _empty_rates()
            ),
            (
                self._fetch_borrow_rates(client, iso_bases_upper)
                if iso_bases_upper
                else _empty_rates()
            ),
        )
        # Normalize to lower-case keys for joining with `base`.
        cross_rates = {k.lower(): v for k, v in cross_rates_raw.items()}
        iso_rates = {k.lower(): v for k, v in iso_rates_raw.items()}

        quote_set = {q.lower() for q in quote_assets}
        rows: list[dict] = []
        for s in data.get("data", []):
            sym = s.get("symbol", "")
            base = s.get("baseCurrency", "").lower()
            quote = s.get("quoteCurrency", "").lower()
            if not base or not quote or quote not in quote_set:
                continue
            if not s.get("enableTrading", False):
                continue
            # Return every active spot pair; cross_rate / isolated_rate
            # stay null for pairs that don't have margin trading.
            rows.append(
                {
                    "ts": now,
                    "symbol": sym,
                    "exchange": self.NAME,
                    "base": base,
                    "quote": quote,
                    "cross_rate": cross_rates.get(base),
                    "isolated_rate": iso_rates.get(base),
                }
            )

        if not rows:
            return self.empty_pairs_df()

        # Funding rates are per-asset and independent of margin
        # rates; fetch them concurrently with the borrow rates.
        funding_rates = await self._fetch_funding_rates(
            client, {r["base"] for r in rows}
        )

        for r in rows:
            r["funding_rate"] = funding_rates.get(r["base"])

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
        are returned. ``close_ts`` is the inclusive last instant of the daily
        candle (``open_ts + 24h - 1us``). The result is capped at
        ``MAX_KLINES`` daily candles; use ``klines_paged()`` for wider
        ranges.
        """
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=dt.timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=dt.timezone.utc)

        # KuCoin expects BASE-QUOTE in upper case (e.g. BTC-USDT).
        if "-" not in symbol:
            upper = symbol.upper()
            if upper.endswith("USDT"):
                ku_symbol = upper[:-4] + "-USDT"
            elif upper.endswith("USDC"):
                ku_symbol = upper[:-4] + "-USDC"
            else:
                raise ValueError(
                    f"KuCoin symbol must look like BTC-USDT, got {symbol!r}"
                )
        else:
            ku_symbol = symbol.upper()

        base, quote = _split_symbol(symbol)
        # KuCoin's ``startAt``/``endAt`` are integer seconds, not ms. The
        # per-page cap is 1500 candles, so the requested window is
        # clipped to ``MAX_KLINES * DAILY_SECONDS`` seconds from the end.
        end_s = int(end_time.timestamp())
        window_start = end_time - dt.timedelta(
            seconds=self.MAX_KLINES * self.DAILY_SECONDS
        )
        start_s = int(max(start_time, window_start).timestamp())

        url = f"https://{self.HOST}/api/v1/market/candles"
        params = {
            "type": "1day",
            "symbol": ku_symbol,
            "startAt": start_s,
            "endAt": end_s - 1,
        }
        try:
            resp = await client.get(url, params=params, timeout=30.0)
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            raise RuntimeError(f"KuCoin klines request failed: {e}") from e
        if payload.get("code") != "200000":
            if payload.get("code") in self.TRANSIENT_CODES:
                raise TransientError(
                    f"KuCoin klines transient: {payload}"
                )
            raise RuntimeError(f"KuCoin klines error: {payload}")
        batch = payload.get("data", []) or []

        # Cap at MAX_KLINES rows (the API may return a touch more on the
        # edge of a window).
        batch = batch[: self.MAX_KLINES]

        rows: list[dict] = []
        for k in batch:
            # [time_seconds, open, close, high, low, base_volume, quote_volume]
            k_ts_s = int(k[0])
            if k_ts_s < int(start_time.timestamp()) or k_ts_s >= end_s:
                continue
            open_ts = dt.datetime.fromtimestamp(k_ts_s, tz=dt.timezone.utc)
            rows.append(
                {
                    "open_ts": open_ts,
                    # KuCoin doesn't return an explicit close timestamp;
                    # derive the inclusive last instant of the daily
                    # candle as ``open_ts + 24h - 1us``.
                    "close_ts": open_ts
                    + dt.timedelta(seconds=self.DAILY_SECONDS)
                    - dt.timedelta(microseconds=1),
                    "symbol": symbol,
                    "base": base,
                    "quote": quote,
                    "open": float(k[1]),
                    "close": float(k[2]),
                    "high": float(k[3]),
                    "low": float(k[4]),
                    "base_volume": float(k[5]),
                    "quote_volume": float(k[6]),
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
