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

import httpx
import polars as pl
from httpx import AsyncClient

from . import _split_symbol, Exchange, TransientError


__all__ = ["HTX"]


class HTX(Exchange):
    """HTX spot exchange adapter."""

    HOST = "api.huobi.pro"
    NAME = "htx"
    # HTX native daily candles are aligned to 16:00 UTC. To produce
    # midnight-UTC 1d candles (matching Binance / KuCoin / Kraken), we
    # pull 4h candles and reassemble them. The 4h endpoint is capped at
    # 2000 candles per call, i.e. ``2000 / 6 = 333`` 1d candles.
    MAX_KLINES = 333
    # The output 1d candles open at midnight UTC; the 4h candles we
    # aggregate are also aligned to multiples of 4h (00, 04, 08, ...).
    DAILY_ALIGN_HOUR_UTC = 0
    # HTX 4h-candle period in seconds. Used by the 1d aggregation to
    # group 4h candles by their containing calendar day in UTC.
    _FOURH_SECONDS = 4 * 60 * 60
    # Number of 4h candles that make up a calendar day. Each 1d
    # candle (00:00 to 00:00 next) is the union of 6 of these.
    _FOURH_PER_DAY = 6

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
            err_msg = (data.get("err-msg") or "").lower()
            if "request limit" in err_msg or "rate limit" in err_msg:
                raise TransientError(f"HTX symbols rate limited: {data}")
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
            return self.empty_pairs_df()

        cross_rates, isolated_rates = await asyncio.gather(
            self._fetch_cross_rates(client),
            self._fetch_isolated_rates(client),
        )

        for r in rows:
            base = r["base"]
            r["cross_rate"] = cross_rates.get(base)
            r["isolated_rate"] = isolated_rates.get(base)

        df = pl.DataFrame(rows)
        # Project onto the canonical schema (this also fixes the order
        # and triggers the dtypes we declared in ``PAIRS_SCHEMA``).
        df = df.select(
            pl.col("ts").cast(pl.Datetime("us", time_zone="UTC")),
            "symbol",
            "exchange",
            "base",
            "quote",
            pl.col("cross_rate").cast(pl.Float64),
            pl.col("isolated_rate").cast(pl.Float64),
        )
        self.validate_pairs_df(df)
        return df

    # ------------------------------------------------------------------
    # klines
    # ------------------------------------------------------------------
    async def _fetch_4h_klines(
        self, client: AsyncClient, symbol: str
    ) -> pl.DataFrame:
        """Return the most recent ``MAX_KLINES * 6`` 4h candles for
        ``symbol`` from HTX as a typed polars frame.

        Columns: open_ts, open, high, low, close, base_volume, quote_volume
        (all timestamps ``Datetime(time_zone='UTC')``).

        The 4h endpoint is capped at 2000 candles per call, so this
        method fetches ``MAX_KLINES * 6`` (= 1998) 4h candles -- the
        source of the cap on 1d coverage in :meth:`klines`.

        HTX's ``from``/``to`` parameters are advisory; the endpoint
        always returns the most recent ``size`` candles. Callers
        filter on the client side using ``open_ts``.
        """
        url = f"https://{self.HOST}/market/history/kline"
        params = {
            "symbol": symbol.lower(),
            "period": "4hour",
            "size": self.MAX_KLINES * self._FOURH_PER_DAY,
        }
        try:
            resp = await client.get(url, params=params, timeout=30.0)
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            raise RuntimeError(f"HTX klines request failed: {e}") from e
        if payload.get("status") != "ok":
            err_msg = (payload.get("err-msg") or "").lower()
            if "request limit" in err_msg or "rate limit" in err_msg:
                raise TransientError(f"HTX klines rate limited: {payload}")
            raise RuntimeError(f"HTX klines error: {payload}")
        batch = payload.get("data", []) or []
        if not batch:
            return pl.DataFrame(
                schema={
                    "open_ts": pl.Datetime("us", time_zone="UTC"),
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "base_volume": pl.Float64,
                    "quote_volume": pl.Float64,
                }
            )
        rows: list[dict] = []
        for k in batch:
            rows.append(
                {
                    "open_ts": dt.datetime.fromtimestamp(
                        int(k["id"]), tz=dt.timezone.utc
                    ),
                    "open": float(k["open"]),
                    "high": float(k["high"]),
                    "low": float(k["low"]),
                    "close": float(k["close"]),
                    "base_volume": float(k["amount"]),
                    "quote_volume": float(k["vol"]),
                }
            )
        return (
            pl.DataFrame(rows)
            .with_columns(pl.col("open_ts").cast(pl.Datetime("us", time_zone="UTC")))
        )

    @staticmethod
    def _aggregate_4h_to_1d(df_4h: pl.DataFrame) -> pl.DataFrame:
        """Group 4h candles into UTC calendar days and aggregate to 1d.

        Each output row covers ``[day 00:00, day+1 00:00)`` and carries:

          * ``open``           -- the open of the 00:00 4h candle
          * ``close``          -- the close of the 20:00 4h candle
          * ``high`` / ``low`` -- max / min of the day's 4h highs / lows
          * ``base_volume``    -- sum of 4h amounts
          * ``quote_volume``   -- sum of 4h quote volumes

        ``open_ts`` is set to the calendar-day boundary in UTC. Days
        with fewer than 6 4h candles (incomplete days, e.g. the
        in-progress current day) are kept as-is -- the caller is
        responsible for filtering them out via the half-open
        ``[start, end)`` range.

        Returns a polars frame with columns:
        ``open_ts, open, high, low, close, base_volume, quote_volume``.
        """
        if df_4h.height == 0:
            return df_4h
        # The 4h endpoint returns candles in descending ``id`` order
        # (most recent first). Sort ascending so the ``first()`` /
        # ``last()`` aggregations below are well-defined
        # (``first()`` = 00:00 candle, ``last()`` = 20:00 candle for a
        # full day). We ``sort`` after the ``truncate`` so the
        # aggregation key (``day_start``) is computed on the original
        # timestamps.
        df_4h = df_4h.sort("open_ts")
        # ``truncate`` snaps each 4h ``open_ts`` down to the start of
        # its UTC day. We use ``"1d"`` which is wall-clock-day in the
        # timestamp's own timezone (UTC here).
        return (
            df_4h.with_columns(
                pl.col("open_ts").dt.truncate("1d").alias("day_start")
            )
            .group_by("day_start", maintain_order=True)
            .agg(
                # The 00:00 4h candle is the first 4h candle of the
                # day, so its ``open`` is the day's open.
                pl.col("open").first().alias("open"),
                # The 20:00 4h candle is the last 4h candle of the
                # day, so its ``close`` is the day's close.
                pl.col("close").last().alias("close"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("base_volume").sum().alias("base_volume"),
                pl.col("quote_volume").sum().alias("quote_volume"),
            )
            .rename({"day_start": "open_ts"})
            .sort("open_ts")
        )

    async def klines(
        self,
        client: AsyncClient,
        symbol: str,
        start_time: dt.datetime,
        end_time: dt.datetime,
    ) -> pl.DataFrame:
        """Fetch at most ``MAX_KLINES`` 1d klines for ``symbol`` in the
        half-open range ``[start_time, end_time)``.

        HTX native daily candles are aligned to 16:00 UTC, but the
        rest of the pipeline expects midnight-UTC alignment (matching
        Binance / KuCoin / Kraken). We pull 4h candles (which are
        themselves aligned to multiples of 4h, including 00:00 and
        16:00 UTC) and aggregate them into 1d candles keyed on the
        UTC calendar day. Each output 1d candle spans
        ``[day 00:00, day+1 00:00)`` and the OHLC / volume columns
        are computed from the 6 4h candles inside that window.

        Columns: open_ts, close_ts, symbol, exchange, base, quote, open, high,
        low, close, base_volume, quote_volume

        ``start_time`` is inclusive, ``end_time`` is exclusive: only
        candles whose open time ``open_ts`` satisfies
        ``start_time <= open_ts < end_time`` are returned.
        ``close_ts`` is the inclusive last instant of the daily
        candle (``open_ts + 24h - 1us``). The result is capped at
        ``MAX_KLINES`` daily candles; use ``klines_paged()`` for
        wider ranges.
        """
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=dt.timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=dt.timezone.utc)

        base, quote = _split_symbol(symbol)

        # Fetch the raw 4h candles. The 4h endpoint ignores
        # ``from``/``to`` and just returns the most recent
        # ``MAX_KLINES * 6`` candles, so we filter on the client
        # side.
        df_4h = await self._fetch_4h_klines(client, symbol)
        if df_4h.height == 0:
            return self.empty_klines_df()

        df_1d = self._aggregate_4h_to_1d(df_4h)
        if df_1d.height == 0:
            return self.empty_klines_df()

        # Half-open filter ``[start_time, end_time)`` on the 1d
        # ``open_ts``. This is what excludes the in-progress today
        # candle when ``end_time`` is today's 00:00 UTC.
        df_1d = df_1d.filter(
            (pl.col("open_ts") >= start_time) & (pl.col("open_ts") < end_time)
        )
        if df_1d.height == 0:
            return self.empty_klines_df()

        # Materialize the per-row fields (``close_ts``, ``symbol``,
        # etc.) and project onto the canonical schema.
        # ``close_ts`` is the inclusive last instant of the 1d
        # candle: ``open_ts + 24h - 1us``. Polars ``dt.offset_by``
        # takes a string like ``"86399999999us"`` (= 24h - 1us).
        df = df_1d.with_columns(
            pl.col("open_ts")
            .dt.offset_by(f"{self.DAILY_SECONDS * 1_000_000 - 1}us")
            .alias("close_ts"),
        ).select(
            pl.col("open_ts").cast(pl.Datetime("us", time_zone="UTC")),
            pl.col("close_ts").cast(pl.Datetime("us", time_zone="UTC")),
            pl.lit(symbol).alias("symbol"),
            pl.lit(self.NAME).alias("exchange"),
            pl.lit(base).alias("base"),
            pl.lit(quote).alias("quote"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("base_volume").cast(pl.Float64),
            pl.col("quote_volume").cast(pl.Float64),
        )
        self.validate_klines_df(df)
        return df

