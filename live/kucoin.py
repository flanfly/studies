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
import time

import polars as pl
from httpx import AsyncClient

from . import _split_symbol, Exchange


__all__ = ["KuCoin"]


class KuCoin(Exchange):
    """KuCoin spot exchange adapter."""

    HOST = "api.kucoin.com"
    NAME = "kucoin"
    # Max daily candles returned by a single ``klines()`` call. The
    # ``/api/v1/market/candles`` endpoint uses ``startAt``/``endAt``
    # (in seconds) to scope the response and returns at most 1500
    # daily candles per request. The actual historical depth (verified
    # for BTC-USDT) goes back to 2017-10-19 -- about 8.6 years; use
    # ``klines_paged()`` to fetch a wider range.
    MAX_KLINES = 1500

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
    # pairs
    # ------------------------------------------------------------------
    async def pairs(self, client: AsyncClient, quote_assets: set[str]) -> pl.DataFrame:
        """Returns active USDT/USDC spot pairs with cross/isolated margin rates.

        Columns: ts, symbol, exchange, base, quote, cross_rate, isolated_rate
        """
        now = dt.datetime.now(dt.timezone.utc)
        # Active spot symbols (public). Filter by enableTrading + quote asset.
        try:
            resp = await client.get(f"https://{self.HOST}/api/v2/symbols", timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"KuCoin symbols request failed: {e}")
        if data.get("code") != "200000":
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
        except Exception as e:
            raise RuntimeError(f"KuCoin klines request failed: {e}")
        if payload.get("code") != "200000":
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
