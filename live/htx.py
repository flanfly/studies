"""HTX (formerly Huobi) spot exchange adapter.

Authenticated endpoints (loan-info) are signed using HTX's v2 signing
scheme (HmacSHA256 over ``METHOD\nhost\npath\nquery_string``).
"""

import asyncio
import base64
import datetime as dt
import hashlib
import hmac
import logging as l
from typing import Optional
from urllib.parse import urlencode

import polars as pl
from httpx import AsyncClient

from . import _split_symbol, Exchange


__all__ = ["HTX"]


class HTX(Exchange):
    """HTX spot exchange adapter."""

    HOST = "api.huobi.pro"
    NAME = "htx"
    # Max daily candles returned by a single ``klines()`` call. With
    # ``size=2000`` the endpoint returns the most recent 2000 daily
    # candles; the ``from``/``to`` parameters do not constrain the
    # result, so the historical depth accessible through ``klines()``
    # is also 2000 days. Use ``klines_paged()`` for wider ranges.
    MAX_KLINES = 2000

    def __init__(self, access_key: str, secret_key: str):
        self._access_key = access_key
        self._secret_key = secret_key

    # ------------------------------------------------------------------
    # signing helpers
    # ------------------------------------------------------------------
    def _sign(self, method: str, host: str, path: str, params: dict) -> str:
        sorted_keys = sorted(params.keys())
        from requests.utils import quote

        encoded = "&".join(f"{k}={quote(str(params[k]), safe='')}" for k in sorted_keys)
        payload = f"{method}\n{host}\n{path}\n{encoded}"
        sig = hmac.new(
            self._secret_key.encode(), payload.encode(), hashlib.sha256
        ).digest()
        return base64.b64encode(sig).decode()

    async def _private_get(
        self, client: AsyncClient, path: str, extra: Optional[dict] = None
    ) -> Optional[dict]:
        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        params: dict = {
            "AccessKeyId": self._access_key,
            "SignatureMethod": "HmacSHA256",
            "SignatureVersion": "2",
            "Timestamp": timestamp,
        }
        if extra:
            params.update(extra)
        params["Signature"] = self._sign("GET", self.HOST, path, params)

        url = f"https://{self.HOST}{path}?{urlencode(params)}"
        resp = await client.get(url, timeout=30.0)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # borrow rate fetchers
    # ------------------------------------------------------------------
    async def _fetch_isolated_rates(self, client: AsyncClient) -> dict[str, float]:
        """{base_lower: annual_rate} from HTX isolated margin."""
        try:
            data = await self._private_get(client, "/v1/margin/loan-info")
        except Exception as e:
            l.warning(f"HTX isolated loan-info failed: {e}")
            return {}
        if data is None or data.get("status") != "ok":
            l.warning(f"HTX isolated loan-info error: {data}")
            return {}

        rates: dict[str, float] = {}
        for item in data.get("data", []):
            symbol: str = item.get("symbol", "")
            if not symbol.upper().endswith("USDT"):
                continue
            base = symbol[:-4].lower()
            for cur in item.get("currencies", []):
                if cur.get("currency", "").upper() == base.upper():
                    rates[base] = float(cur["interest-rate"]) * 365
        return rates

    async def _fetch_cross_symbols(self, client: AsyncClient) -> set[str]:
        """{base_lower} of USDT-quoted cross margin symbols."""
        try:
            resp = await client.get(
                f"https://{self.HOST}/v1/margin/symbols", timeout=30.0
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            l.warning(f"HTX cross margin symbols failed: {e}")
            return set()
        if data.get("status") != "ok":
            return set()

        bases: set[str] = set()
        for sym in data.get("data", []):
            s = sym.get("symbol", "")
            if s.upper().endswith("USDT"):
                bases.add(s[:-4].lower())
        return bases

    async def _fetch_cross_rates(self, client: AsyncClient) -> dict[str, float]:
        """{base_lower: annual_rate} from HTX cross margin, restricted to USDT pairs."""
        try:
            data = await self._private_get(client, "/v1/cross-margin/loan-info")
        except Exception as e:
            l.warning(f"HTX cross-margin loan-info failed: {e}")
            return {}
        if data is None or data.get("status") != "ok":
            l.warning(f"HTX cross-margin loan-info error: {data}")
            return {}

        daily_rates: dict[str, float] = {}
        for item in data.get("data", []):
            cur = item.get("currency", "").upper()
            daily_rates[cur] = float(item["interest-rate"])

        cross_bases = await self._fetch_cross_symbols(client)
        rates: dict[str, float] = {}
        for base in cross_bases:
            up = base.upper()
            if up in daily_rates:
                rates[base] = daily_rates[up] * 365
        return rates

    # ------------------------------------------------------------------
    # pairs
    # ------------------------------------------------------------------
    async def pairs(self, client: AsyncClient, quote_assets: set[str]) -> pl.DataFrame:
        """Returns active spot pairs with cross/isolated margin borrow rates.

        Columns: ts, symbol, exchange, base, quote, cross_rate, isolated_rate
        """
        now = dt.datetime.now(dt.timezone.utc)
        url = f"https://{self.HOST}/v1/common/symbols"
        resp = await client.get(url, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok":
            raise RuntimeError(f"HTX symbols error: {data}")

        rows: list[dict] = []
        for s in data.get("data", []):
            if s.get("state") != "online":
                continue
            base = s.get("base-currency", "").lower()
            quote = s.get("quote-currency", "").lower()
            if not base or not quote or quote not in quote_assets:
                continue
            rows.append(
                {
                    "ts": now,
                    "symbol": s["symbol"],
                    "exchange": self.NAME,
                    "base": base,
                    "quote": quote,
                }
            )

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

        cross_rates, isolated_rates = await asyncio.gather(
            self._fetch_cross_rates(client),
            self._fetch_isolated_rates(client),
        )

        for r in rows:
            base = r["base"]
            r["cross_rate"] = cross_rates.get(base)
            r["isolated_rate"] = isolated_rates.get(base)

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
        are returned. ``close_ts`` is the inclusive last instant of the daily
        candle (``open_ts + 24h - 1us``). The result is capped at
        ``MAX_KLINES`` daily candles; use ``klines_paged()`` for wider
        ranges.
        """
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=dt.timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=dt.timezone.utc)

        base, quote = _split_symbol(symbol)
        start_s = int(start_time.timestamp())
        end_s = int(end_time.timestamp())

        url = f"https://{self.HOST}/market/history/kline"
        params = {
            "symbol": symbol.lower(),
            "period": "1day",
            "size": self.MAX_KLINES,
        }
        try:
            resp = await client.get(url, params=params, timeout=30.0)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            raise RuntimeError(f"HTX klines request failed: {e}")
        if payload.get("status") != "ok":
            raise RuntimeError(f"HTX klines error: {payload}")
        batch = payload.get("data", []) or []

        rows: list[dict] = []
        for k in batch:
            k_ts = int(k["id"])
            # Inclusive start, exclusive end.
            if k_ts < start_s or k_ts >= end_s:
                continue
            open_ts = dt.datetime.fromtimestamp(k_ts, tz=dt.timezone.utc)
            rows.append(
                {
                    "open_ts": open_ts,
                    # HTX doesn't return an explicit close timestamp;
                    # derive the inclusive last instant of the daily
                    # candle as ``open_ts + 24h - 1us``.
                    "close_ts": open_ts
                    + dt.timedelta(seconds=self.DAILY_SECONDS)
                    - dt.timedelta(microseconds=1),
                    "symbol": symbol,
                    "base": base,
                    "quote": quote,
                    "open": float(k["open"]),
                    "high": float(k["high"]),
                    "low": float(k["low"]),
                    "close": float(k["close"]),
                    "base_volume": float(k["amount"]),
                    "quote_volume": float(k["vol"]),
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
