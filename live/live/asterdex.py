"""AsterDex exchange adapter (perpetual-futures only).

AsterDex is a Binance-style perpetuals DEX on BNB Chain. Its
fapi endpoints mirror the Binance fapi API verbatim -- same
``/fapi/v1/klines`` shape (12-element array), same
``/fapi/v1/premiumIndex`` response, same ``/fapi/v1/exchangeInfo``
metadata, same 1000-bar per-call cap on klines. Every public
endpoint is unauthenticated, so the adapter takes no
constructor arguments.

The adapter deviates from the rest of the framework in the
same way :class:`live.hyperliquid.Hyperliquid` does:

* ``pairs()`` emits one row per **perp contract** (AsterDex has
  no spot market, no margin-borrow endpoint, no isolated
  margin). ``base`` is the lower-cased symbol, ``quote`` is
  ``"usdt"`` (AsterDex only lists USDT-margined perps as far
  as the ``exchangeInfo`` response shows).
* ``klines()`` returns **perp OHLCV** (the only kind AsterDex
  exposes). The 12-element response is parsed in the same
  order as Binance fapi, including the ``trades`` field at
  index 8 -- but we deliberately do NOT filter on
  ``trades > 0`` the way the Hyperliquid adapter does for the
  ``n > 0`` filter. AsterDex does not back-fill historical
  bars: a direct probe of any window before the venue's data
  retention boundary returns ``[]``, not synthetic OHLC. The
  boundary itself is the venue's actual history limit, not a
  backfill region.
* ``funding_rate`` is annualised with the standard 8h
  schedule (``rate * 3 * 365``). The funding interval is
  verified to be 8h for every contract (delta of
  ``fundingTime`` is exactly 8.0h between consecutive
  payments) -- the constant is hardcoded as
  ``_FUNDING_INTERVAL_HOURS = 8.0`` and not read from the
  response.

Public endpoints used (all unauthenticated):

* ``GET https://fapi.asterdex.com/fapi/v1/exchangeInfo`` --
  returns the perp symbol list with status, base/quote
  asset, precision, and per-symbol metadata. The
  ``symbols`` field is filtered to ``status == "TRADING"``
  and ``contractType == "PERPETUAL"``.
* ``GET https://fapi.asterdex.com/fapi/v1/klines`` --
  returns up to 1000 daily candles for one symbol in the
  ``[startTime, endTime)`` ms window. No auth, public.
  Historical retention for BTCUSDT goes back to 2021-09-01
  (~4.8 years); newer pairs are shorter.
* ``GET https://fapi.asterdex.com/fapi/v1/premiumIndex`` --
  returns the current funding rate + mark + index for one
  symbol. The ``lastFundingRate`` field is the most recent
  per-interval rate, used to populate the ``funding_rate``
  column on ``pairs()``.

AsterDex has no borrow-rate concept, so ``cross_rate`` and
``isolated_rate`` are always ``None`` for every row. Funding
rates are populated for the entire perp universe in a single
``exchangeInfo`` walk, NOT via a one-call batched
``fundingRate`` endpoint (we follow the Binance pattern: probe
``premiumIndex`` for the bases we care about).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import random
from typing import Iterable

import httpx
import polars as pl
from httpx import AsyncClient

from . import Exchange


__all__ = ["AsterDex"]


logger = logging.getLogger(__name__)


# Single base URL -- all public endpoints live here.
_ASTER_HOST = "fapi.asterdex.com"


# Max daily candles returned by a single ``klines()`` call.
# Mirrors Binance fapi: the endpoint caps responses at 1000
# bars per request. The base-class ``klines_paged()`` walks
# wider ranges in 1000-bar chunks.
_KLINE_CAP = 1000

# AsterDex perpetuals all pay funding every 8 hours (the
# standard Binance-compatible schedule). Confirmed by direct
# probe: consecutive ``fundingTime`` deltas on BTCUSDT are
# exactly 8.0h. If a per-symbol override ever shows up in
# ``exchangeInfo`` (a ``fundingIntervalHours`` field), this
# constant becomes a fallback.
_FUNDING_INTERVAL_HOURS: float = 8.0

# Cap on in-flight ``_fetch_funding_rate`` calls. AsterDex's
# public premiumIndex endpoint is rate-limited per IP, so we
# keep concurrency well under the limit. 30-wide finishes a
# few hundred perps in a few seconds.
_FUNDING_CONCURRENCY: int = 30


def _drop_zero_trade_candles(rows: list[dict]) -> list[dict]:
    """Drop candles with ``trades == 0`` from ``rows``.

    **Currently a no-op safety net.** AsterDex does not
    back-fill historical bars -- a direct probe of any window
    before the venue's data retention boundary returns
    ``[]`` rather than synthetic OHLC. We retain the filter
    for symmetry with :class:`live.hyperliquid.Hyperliquid`
    and as a defensive measure if a future venue change ever
    introduces back-fill. The check is on the parsed
    ``trades`` field; bars with ``trades`` missing are kept
    (a live bar that omits ``trades`` should not be
    silently dropped).
    """
    return [
        r for r in rows
        if r.get("trades") is None or int(r["trades"]) > 0
    ]


class AsterDex(Exchange):
    """AsterDex perpetual-futures adapter.

    No constructor arguments: every public endpoint is
    unauthenticated. The adapter is stateless; concurrent
    callers can share a single instance across requests.
    """

    HOST = _ASTER_HOST
    NAME = "asterdex"
    # Native daily candles open at 00:00 UTC (same alignment
    # as Binance / KuCoin / Kraken / Hyperliquid).
    DAILY_ALIGN_HOUR_UTC = 0
    # Per-call cap on ``klines()``; ``klines_paged()`` does
    # the chunking for wider ranges.
    MAX_KLINES = _KLINE_CAP
    # Hardcoded 8h funding schedule (confirmed by direct
    # probe of the ``fundingRate`` endpoint). Used by
    # ``_fetch_funding_rates`` to annualise per-interval
    # rates.
    _FUNDING_INTERVAL_HOURS: float = _FUNDING_INTERVAL_HOURS

    # ------------------------------------------------------------------
    # low-level HTTP helpers
    # ------------------------------------------------------------------
    async def _fapi_get(
        self,
        client: AsyncClient,
        path: str,
        params: dict | None = None,
    ) -> object:
        """GET against ``https://fapi.asterdex.com{path}``.

        Returns the parsed JSON body. HTTP and transport
        errors are propagated; the base-class
        ``is_transient_error`` + ``_retry`` decides whether
        to retry.
        """
        url = f"https://{self.HOST}{path}"
        try:
            resp = await client.get(url, params=params, timeout=30.0)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"AsterDex {path} request failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    # funding rate fetcher (overrides the base default)
    # ------------------------------------------------------------------
    async def _funding_request(
        self, client: AsyncClient, symbol: str
    ) -> dict | None:
        """Make a single AsterDex funding-rate HTTP request
        with infinite per-call retry on transient errors
        (429, 5xx, transport). Returns the parsed JSON
        payload on success, ``None`` on a permanent failure.
        """
        url = (
            f"https://{self.HOST}/fapi/v1/premiumIndex"
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
                    delay = delay * (0.75 + random.random() * 0.5)  # noqa: F821 -- defined in base via import
                    logger.warning(
                        "AsterDex funding_rate(%s) transient error "
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
                logger.warning(
                    f"AsterDex funding_rate({symbol}) HTTP error: {e}"
                )
                return None
            except Exception as e:
                if not self.is_transient_error(e):
                    raise RuntimeError(
                        f"AsterDex funding_rate request failed: {e}"
                    ) from e
                delay = min(
                    self.RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                    self.RETRY_MAX_DELAY,
                )
                delay = delay * (0.75 + random.random() * 0.5)
                logger.warning(
                    "AsterDex funding_rate(%s) transient error "
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
        """Return the most recent funding rate (annualised
        APR) for the AsterDex USDT-margined perpetual on
        ``base``/``quote``, or ``None`` if no contract
        exists or the request fails for a non-transient
        reason.

        Endpoint: ``GET /fapi/v1/premiumIndex?symbol={SYMBOL}``
        (unauthenticated, public). Response shape::

            {
              "symbol": "BTCUSDT",
              "markPrice": "...",
              "indexPrice": "...",
              "estimatedSettlePrice": "...",
              "lastFundingRate": "0.00001234",
              "interestRate": "0.00010000",
              "nextFundingTime": ...,
              "time": ...
            }

        AsterDex only lists USDT-margined perps (no USDC
        fallback to probe, no COIN-M inverse-margined
        products), so we do a single ``premiumIndex`` call
        per base. A non-transient HTTP error or a
        non-dict response is treated as a per-asset miss
        and returns ``None`` silently -- the spot pair
        simply has no perpetual contract on AsterDex.

        Transient errors (429, 5xx, transport) are retried
        with per-call backoff in :meth:`_funding_request`.
        """
        symbol = f"{base.upper()}USDT"
        payload = await self._funding_request(client, symbol)
        if payload is None:
            return None
        if not isinstance(payload, dict):
            return None
        try:
            rate = float(payload["lastFundingRate"])
        except (KeyError, TypeError, ValueError):
            return None
        return self.annualize_funding_rate(
            rate, self._FUNDING_INTERVAL_HOURS
        )

    async def _fetch_funding_rates(
        self, client: AsyncClient, bases: Iterable[str]
    ) -> dict[str, float]:
        """``{base_lower: annual_funding_rate}`` for every
        perp in ``bases`` that has a contract on AsterDex.

        Concurrent fetch via ``asyncio.gather``, bounded by
        ``_FUNDING_CONCURRENCY`` to stay under the public
        ``/fapi/v1/premiumIndex`` per-IP rate limit. AsterDex
        only lists USDT-margined perps, so we probe the
        ``{BASE}USDT`` symbol and accept whatever the
        response reports.
        """
        bases = sorted({b.lower() for b in bases})
        if not bases:
            return {}

        semaphore = asyncio.Semaphore(_FUNDING_CONCURRENCY)

        async def _one(b: str) -> tuple[str, float | None]:
            async with semaphore:
                try:
                    rate = await self._fetch_funding_rate(
                        client, base=b, quote="USDT"
                    )
                except Exception as e:
                    logger.warning(
                        f"AsterDex funding_rate({b}) failed: {e}"
                    )
                    return b, None
            return b, rate

        results = await asyncio.gather(*(_one(b) for b in bases))
        return {b: r for b, r in results if r is not None}

    # ------------------------------------------------------------------
    # pairs
    # ------------------------------------------------------------------
    async def pairs(
        self, client: AsyncClient, quote_assets: set[str], limit: int | None = None
    ) -> pl.DataFrame:
        """Return one row per active AsterDex perp contract.

        Columns: ts, symbol, exchange, base, quote, cross_rate,
        isolated_rate, funding_rate

        ``symbol`` is the AsterDex perp ticker
        (``BTCUSDT``, ``ETHUSDT``, ...), ``base`` is the
        lower-cased base asset, ``quote`` is the literal
        string ``"usdt"`` (AsterDex only lists USDT-margined
        perps). ``quote_assets`` is **ignored** for the same
        reason as :meth:`live.hyperliquid.Hyperliquid.pairs`:
        there is no spot market to filter by.

        ``cross_rate`` and ``isolated_rate`` are always
        ``None`` (no borrow-rate concept on a perp-only
        venue). ``funding_rate`` is the per-8h rate from
        ``premiumIndex`` annualised as ``rate * 3 * 365``.

        Filtering: only perps with ``status == "TRADING"``
        and ``contractType == "PERPETUAL"`` are included.
        AsterDex does not currently list inverse-margined or
        USDC-settled perps in its ``exchangeInfo`` response,
        so a fallback to the second settle currency (as the
        Binance adapter does) is unnecessary.
        """
        now = dt.datetime.now(dt.timezone.utc)
        try:
            data = await self._fapi_get(
                client, "/fapi/v1/exchangeInfo"
            )
        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"AsterDex exchangeInfo request failed: {e}"
            ) from e
        if not isinstance(data, dict):
            raise RuntimeError(
                f"AsterDex exchangeInfo: expected dict, got "
                f"{type(data).__name__}"
            )

        # Quote filter is intentional no-op: we always emit
        # every USDT perp. The argument is kept in the
        # signature for framework conformance.
        _ = quote_assets  # noqa: F841 -- silence unused-arg

        rows: list[dict] = []
        active_bases: set[str] = set()
        for s in data.get("symbols", []):
            if s.get("status") != "TRADING":
                continue
            if s.get("contractType") != "PERPETUAL":
                continue
            base = s.get("baseAsset", "")
            quote = s.get("quoteAsset", "")
            if not base or quote != "USDT":
                # AsterDex only lists USDT-margined perps;
                # skip anything else (defensive -- we don't
                # expect a non-USDT perp to be in the list).
                continue
            rows.append(
                {
                    "ts": now,
                    "symbol": s["symbol"],
                    "exchange": self.NAME,
                    "base": base.lower(),
                    "quote": "usdt",
                    "cross_rate": None,
                    "isolated_rate": None,
                    "funding_rate": None,  # populated below
                }
            )
            active_bases.add(base)

        if not rows:
            return self.empty_pairs_df()

        # Funding rates are independent of the symbols
        # endpoint -- fetch them in a single batched gather.
        funding_rates = await self._fetch_funding_rates(
            client, active_bases
        )
        for r in rows:
            r["funding_rate"] = funding_rates.get(r["base"])

        df = pl.DataFrame(rows).select(
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
        """Return at most ``MAX_KLINES`` daily perp candles
        for ``symbol`` in the half-open range ``[start_time,
        end_time)``.

        Columns: open_ts, close_ts, symbol, exchange, base,
        quote, open, high, low, close, base_volume, quote_volume

        ``base`` is the lower-cased symbol minus the ``USDT``
        suffix, ``quote`` is the literal string ``"usdt"``
        (matching :meth:`pairs`). ``close_ts`` is the
        inclusive last instant of the daily candle in
        microsecond resolution (taken from the API's
        ``close_time`` field at index 6, converted from
        ms-resolution to µs).

        The AsterDex ``/fapi/v1/klines`` response is a
        12-element array (Binance fapi shape)::

            [open_time, o, h, l, c, base_vol, close_time,
             quote_vol, trades, taker_buy_base,
             taker_buy_quote, _]

        We parse the same fields as the Binance adapter.
        ``trades`` (index 8) is read but not used for the
        ``n > 0`` filter: AsterDex does not back-fill
        historical bars, so a venue-history boundary is
        always reflected as a missing-window (0 rows
        returned), not as zero-trade bars. The
        ``_drop_zero_trade_candles`` filter is retained as
        a defensive no-op for symmetry with
        :class:`live.hyperliquid.Hyperliquid`.

        The per-call cap is 1000 candles (server-side
        limit). Use ``klines_paged()`` for wider ranges; it
        walks the range backward in 1000-bar chunks, the
        same pattern used by the Binance adapter.
        """
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=dt.timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=dt.timezone.utc)

        symbol = symbol.upper()
        # AsterDex symbols are concatenated upper case
        # (``BTCUSDT``); strip the USDT suffix to get the
        # base, exactly as ``_split_symbol`` does for
        # Binance-style symbols.
        if not symbol.endswith("USDT"):
            raise ValueError(
                f"AsterDex symbol {symbol!r} is not a USDT "
                f"perp; this adapter only handles USDT-"
                f"margined contracts"
            )
        base = symbol[:-4].lower()
        quote = "usdt"

        # Clip the request window to ``MAX_KLINES`` days
        # from the end. The server returns at most 1000
        # bars per call; for a wider range the caller
        # must use ``klines_paged()`` (which is the
        # standard path through ``main.py``).
        window_start = end_time - dt.timedelta(
            seconds=self.MAX_KLINES * self.DAILY_SECONDS
        )
        clipped_start = max(start_time, window_start)

        start_ms = int(clipped_start.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        try:
            batch = await self._fapi_get(
                client,
                "/fapi/v1/klines",
                params={
                    "symbol": symbol,
                    "interval": "1d",
                    "startTime": start_ms,
                    "endTime": end_ms - 1,
                    "limit": 1000,
                },
            )
        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"AsterDex klines request failed: {e}"
            ) from e
        if not isinstance(batch, list):
            # AsterDex returns a JSON error envelope (e.g.
            # ``{"code":-1121,"msg":"Invalid symbol."}``) on
            # application errors. Surface a clear RuntimeError
            # so the caller can decide what to do; the
            # ``_retry`` wrapper above has already retried on
            # transient HTTP errors.
            if isinstance(batch, dict):
                raise RuntimeError(
                    f"AsterDex klines error: {batch}"
                )
            raise RuntimeError(
                f"AsterDex klines: expected list, got "
                f"{type(batch).__name__}"
            )

        # Defensive: cap at MAX_KLINES rows in case the
        # server ever returns a touch more on the edge of a
        # window.
        batch = batch[: self.MAX_KLINES]

        rows: list[dict] = []
        for k in batch:
            try:
                k_open_ms = int(k[0])
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            # Half-open range: ``start_time <= open_ts < end_time``.
            if k_open_ms < start_ms or k_open_ms >= end_ms:
                continue
            open_ts = dt.datetime.fromtimestamp(
                k_open_ms / 1000.0, tz=dt.timezone.utc
            )
            try:
                # ``close_time`` is the inclusive last
                # instant of the daily candle; cast from
                # ms-resolution to microsecond.
                close_ts = dt.datetime.fromtimestamp(
                    int(k[6]) / 1000.0, tz=dt.timezone.utc
                )
                # ``trades`` is read for the defensive
                # ``_drop_zero_trade_candles`` filter (a
                # no-op for AsterDex today; see module
                # docstring).
                trades = int(k[8]) if k[8] is not None else None
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
                        # Preserved on the row so the
                        # post-parse filter can see it.
                        # Stripped from the final frame
                        # because it is not part of
                        # ``KLINES_SCHEMA``.
                        "trades": trades,
                    }
                )
            except (KeyError, TypeError, ValueError, IndexError):
                # Malformed bar -- skip and continue.
                continue

        rows = _drop_zero_trade_candles(rows)
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
