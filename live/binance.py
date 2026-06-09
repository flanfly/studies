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
import time
from typing import Iterable
from urllib.parse import urlencode

import polars as pl
from httpx import AsyncClient

from . import _split_symbol, Exchange


__all__ = ["Binance"]


class Binance(Exchange):
    """Binance spot exchange adapter."""

    HOST = "api.binance.com"
    SAPI_HOST = "api.binance.com"  # no separate demo sapi
    NAME = "binance"
    # Max daily candles returned by a single ``klines()`` call. The
    # ``/api/v3/klines`` endpoint caps the response at 1000 candles
    # per request, and the actual historical retention for major
    # pairs (e.g. BTCUSDT) goes back to 2017-08-17 -- about 8.75
    # years. Use ``klines_paged()`` to fetch a wider range.
    MAX_KLINES = 1000

    def __init__(self, api_key: str, api_secret: str):
        self._api_key = api_key
        self._api_secret = api_secret
        # Populated by ``_load_borrow_rates`` on first ``pairs()`` call:
        # ``{asset_upper: annualized_rate}``. None when credentials are
        # missing or the sapi call fails.
        self._borrow_rate_cache: dict[str, float] | None = None

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
        """
        try:
            data = await self._sapi_get(
                client,
                "/sapi/v1/margin/interestRateHistory",
                {"asset": asset.upper()},
            )
        except Exception as e:
            l.warning(f"Binance borrowRate({asset}) failed: {e}")
            return None
        if not isinstance(data, list) or not data:
            return None
        # Response is sorted newest-first; take the most recent rate.
        try:
            daily = float(data[0]["dailyInterestRate"])
        except (KeyError, TypeError, ValueError):
            return None
        return daily * 365

    async def _load_borrow_rates(
        self, client: AsyncClient, bases: Iterable[str]
    ) -> dict[str, float]:
        """``{base_lower: annual_rate}`` for each base asset, fetched
        concurrently and cached on the instance.
        """
        if self._borrow_rate_cache is not None:
            return self._borrow_rate_cache
        bases = sorted({b for b in bases})
        rates: dict[str, float] = {}
        results = await asyncio.gather(
            *(self._fetch_borrow_rate(client, b) for b in bases)
        )
        for b, r in zip(bases, results):
            if r is not None:
                rates[b.lower()] = r
        self._borrow_rate_cache = rates
        return rates

    # ------------------------------------------------------------------
    # pairs
    # ------------------------------------------------------------------
    async def pairs(self, client: AsyncClient, quote_assets: set[str]) -> pl.DataFrame:
        """Returns active USDT/USDC spot pairs.

        Columns: ts, symbol, exchange, base, quote, cross_rate, isolated_rate
        ``cross_rate`` and ``isolated_rate`` are populated from the per-asset
        ``interestRateHistory`` endpoint (the same rate applies to both
        cross and isolated margin on Binance).
        """
        now = dt.datetime.now(dt.timezone.utc)
        try:
            resp = await client.get(
                f"https://{self.HOST}/api/v3/exchangeInfo?permissions=SPOT",
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Binance exchangeInfo request failed: {e}")

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
            for r in rows:
                rate = rates.get(r["base"])
                r["cross_rate"] = rate
                r["isolated_rate"] = rate

        if not rows:
            return pl.DataFrame(
                schema={
                    "ts": pl.Datetime("us", time_zone="UTC"),
                    "symbol": pl.Utf8,
                    "exchange": pl.Utf8,
                    "base": pl.Utf8,
                    "quote": pl.Utf8,
                    "cross_rate": pl.Float64,
                    "isolated_rate": pl.Float64,
                }
            )

        df = pl.DataFrame(rows).with_columns(
            pl.col("ts").cast(pl.Datetime("us", time_zone="UTC")),
        )
        return df.select(
            "ts", "symbol", "exchange", "base", "quote", "cross_rate", "isolated_rate"
        )

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
        except Exception as e:
            raise RuntimeError(f"Binance klines request failed: {e}")
        if not isinstance(batch, list):
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
            return pl.DataFrame(
                schema={
                    "open_ts": pl.Datetime("us", time_zone="UTC"),
                    "close_ts": pl.Datetime("us", time_zone="UTC"),
                    "symbol": pl.Utf8,
                    "exchange": pl.Utf8,
                    "base": pl.Utf8,
                    "quote": pl.Utf8,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "base_volume": pl.Float64,
                    "quote_volume": pl.Float64,
                }
            )

        df = (
            pl.DataFrame(rows)
            .unique(subset=["open_ts", "symbol"], keep="last")
            .sort("open_ts")
            .with_columns(
                pl.col("open_ts").cast(pl.Datetime("us", time_zone="UTC")),
                pl.col("close_ts").cast(pl.Datetime("us", time_zone="UTC")),
                pl.lit(self.NAME).alias("exchange"),
            )
        )
        return df.select(
            "open_ts",
            "close_ts",
            "symbol",
            "exchange",
            "base",
            "quote",
            "open",
            "high",
            "low",
            "close",
            "base_volume",
            "quote_volume",
        )
