"""Kraken spot exchange adapter (no authentication required for the public
endpoints we use: ``/0/public/AssetPairs``, ``/0/public/Assets`` and
``/0/public/OHLC``).

Kraken only supports **cross margin** (no isolated margin), so
``cross_rate`` is populated from the per-asset ``margin_rate`` field of
``/0/public/Assets`` and ``isolated_rate`` is always null. ``AssetPairs``
includes per-asset Kraken codes (e.g. ``XXBT`` for BTC) which are
normalized via the ``/Assets`` endpoint to the common ticker (``XBT``),
then mapped to the conventional ticker (``BTC``) for the ``base`` column.
"""

import datetime as dt
import logging as l

import polars as pl
from httpx import AsyncClient

from . import _split_symbol, Exchange


__all__ = ["Kraken"]


class Kraken(Exchange):
    """Kraken spot exchange adapter."""

    HOST = "api.kraken.com"
    NAME = "kraken"
    OHLC_INTERVAL = 1440  # daily candles
    # Max daily klines the Kraken API retains. ``/0/public/OHLC`` returns
    # at most 720 daily candles, and that's also the historical depth --
    # older data is not available on the public endpoint.
    MAX_KLINES = 720

    # Kraken altname -> common ticker for the assets whose Kraken altname
    # still carries a leading letter. ``/Assets`` returns ``XBT`` for BTC,
    # ``XLM`` for XLM, etc. — the rest of the codebase uses ``btc``, so we
    # override a small set of well-known codes.
    _ALTNAME_OVERRIDES = {
        "XBT": "BTC",
        "XDG": "DOGE",
    }

    def __init__(self) -> None:
        # No credentials needed for the public endpoints we use.
        # Populated by ``_load_altnames`` on first call. Two parallel dicts
        # keyed by Kraken asset code (``XXBT``, ``USDT``, ...):
        #   * ``_altname_cache[code]`` -> Kraken altname (``XBT``, ``USDT``)
        #   * ``_margin_cache[code]`` -> annual borrow rate (e.g. ``0.01``)
        # The small set of altnames whose Kraken form still carries a
        # non-conventional leading letter (``XBT`` for BTC, ``XDG`` for
        # DOGE) is normalized only at emit time, not in the cache, so the
        # reverse lookup ``altname -> kraken_code`` is lossless.
        self._altname_cache: dict[str, str] | None = None
        self._margin_cache: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    async def _load_altnames(self, client: AsyncClient) -> dict[str, str]:
        """``{kraken_code: altname}`` for all Kraken assets.

        The first lookup hits ``/Assets``; subsequent calls reuse the cache.
        Also populates ``_margin_cache`` with the per-asset annual borrow
        rate (``margin_rate`` field; absent assets are simply missing from
        the cache, so the lookup returns null at the call site).
        """
        if self._altname_cache is not None:
            return self._altname_cache
        try:
            resp = await client.get(
                f"https://{self.HOST}/0/public/Assets", timeout=30.0
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            l.warning(f"Kraken assets request failed: {e}")
            self._altname_cache = {}
            self._margin_cache = {}
            return self._altname_cache
        if data.get("error"):
            l.warning(f"Kraken assets API error: {data['error']}")
            self._altname_cache = {}
            self._margin_cache = {}
            return self._altname_cache
        altnames: dict[str, str] = {}
        margins: dict[str, float] = {}
        for code, info in data.get("result", {}).items():
            altnames[code] = info.get("altname", code)
            mr = info.get("margin_rate")
            if mr is not None:
                try:
                    margins[code] = float(mr)
                except (TypeError, ValueError):
                    pass
        self._altname_cache = altnames
        self._margin_cache = margins
        return altnames

    def _common_ticker(self, code_or_altname: str) -> str:
        """Lower-case an asset code/altname, mapping to the conventional
        ticker (e.g. XBT -> BTC, XDG -> DOGE) for the ``base`` column.
        """
        upper = code_or_altname.upper()
        return self._ALTNAME_OVERRIDES.get(upper, upper).lower()

    # ------------------------------------------------------------------
    # pairs
    # ------------------------------------------------------------------
    async def pairs(self, client: AsyncClient, quote_assets: set[str]) -> pl.DataFrame:
        """Returns active USDT/USDC spot pairs.

        Columns: ts, symbol, exchange, base, quote, cross_rate, isolated_rate
        ``cross_rate`` is the per-asset annual borrow rate from
        ``/0/public/Assets`` (Kraken only supports cross margin).
        ``isolated_rate`` is always null.
        """
        now = dt.datetime.now(dt.timezone.utc)
        altnames = await self._load_altnames(client)
        # ``_load_altnames`` populates ``_margin_cache`` in the same call.
        margins = self._margin_cache or {}

        try:
            resp = await client.get(
                f"https://{self.HOST}/0/public/AssetPairs", timeout=30.0
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Kraken AssetPairs request failed: {e}")
        if data.get("error"):
            raise RuntimeError(f"Kraken AssetPairs API error: {data['error']}")

        quote_set = {q.upper() for q in quote_assets}
        rows: list[dict] = []
        for altname, info in data.get("result", {}).items():
            if info.get("status") != "online":
                continue
            base_raw = info.get("base", "")
            quote_raw = info.get("quote", "")
            if not base_raw or quote_raw not in quote_set:
                continue
            # The ``base`` from AssetPairs is the Kraken asset code (e.g.
            # ``XXBT``); the altname map gives ``XBT``, which is then
            # overridden to ``BTC`` for the common ticker.
            base_altname = altnames.get(base_raw, base_raw)
            rows.append(
                {
                    "ts": now,
                    "symbol": altname,
                    "exchange": self.NAME,
                    "base": self._common_ticker(base_altname),
                    "quote": quote_raw.lower(),
                    # Kraken is cross-margin only: the borrow rate is a
                    # per-asset field on /Assets, not per-pair.
                    "cross_rate": margins.get(base_raw),
                    "isolated_rate": None,
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

        # Kraken OHLC takes the pair altname (e.g. XBTUSDT, ADAUSDT).
        symbol = symbol.upper()
        # Apply the same base-asset normalization as ``pairs()`` so the
        # ``base`` column matches (XBT -> BTC, XDG -> DOGE, etc.).
        altnames = await self._load_altnames(client)
        base_raw, quote = _split_symbol(symbol)
        # The altname map is keyed on the Kraken code (e.g. ``XXBT``); the
        # ``base_raw`` we got from splitting the symbol is the altname
        # (``XBT``). Build a reverse lookup to translate back to the code.
        reverse = {v.upper(): k for k, v in altnames.items()}
        kraken_code = reverse.get(base_raw.upper(), base_raw.upper())
        base = self._common_ticker(altnames.get(kraken_code, base_raw))

        # Kraken returns at most ``MAX_KLINES`` (720) candles with
        # ``ts >= since``. Set ``since = start_time`` and filter the
        # response to the requested half-open window.
        start_s = int(start_time.timestamp())
        end_s = int(end_time.timestamp())
        url = f"https://{self.HOST}/0/public/OHLC"
        params = {
            "pair": symbol,
            "interval": self.OHLC_INTERVAL,
            "since": start_s,
        }
        try:
            resp = await client.get(url, params=params, timeout=30.0)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            raise RuntimeError(f"Kraken OHLC request failed: {e}")
        if payload.get("error"):
            raise RuntimeError(f"Kraken OHLC API error: {payload['error']}")

        batch = payload.get("result", {}).get(symbol, []) or []
        # Cap at MAX_KLINES rows (the API may return a touch more on
        # the edge of a window).
        batch = batch[: self.MAX_KLINES]

        rows: list[dict] = []
        for k in batch:
            # [time, open, high, low, close, vwap, volume, count]
            k_ts = int(k[0])
            # Inclusive start, exclusive end.
            if k_ts < start_s or k_ts >= end_s:
                continue
            open_ts = dt.datetime.fromtimestamp(k_ts, tz=dt.timezone.utc)
            rows.append(
                {
                    "open_ts": open_ts,
                    # Kraken doesn't return an explicit close timestamp;
                    # derive the inclusive last instant of the daily
                    # candle as ``open_ts + 24h - 1us``.
                    "close_ts": open_ts
                    + dt.timedelta(seconds=self.DAILY_SECONDS)
                    - dt.timedelta(microseconds=1),
                    "symbol": symbol,
                    "base": base,
                    "quote": quote,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "base_volume": float(k[6]),
                    # Kraken OHLC doesn't return a quote-asset volume;
                    # approximate it as ``vwap * volume``.
                    "quote_volume": float(k[5]) * float(k[6]),
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
