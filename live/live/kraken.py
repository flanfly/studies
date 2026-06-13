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

import httpx
import polars as pl
from httpx import AsyncClient

from . import Exchange, TransientError


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

    # Kraken error strings that warrant a retry (rate limiting,
    # temporary service errors). Other errors are propagated
    # immediately.
    TRANSIENT_ERRORS = frozenset({
        "EGeneral:Temporary",
        "EService:Temporary",
        "EOrder:Rate limit exceeded",
        "EDesk:Rate limit exceeded",
        "Rate limit exceeded",
    })

    # Kraken altname -> common ticker for the assets whose Kraken altname
    # still carries a leading letter. ``/Assets`` returns ``XBT`` for BTC,
    # ``XLM`` for XLM, etc. — the rest of the codebase uses ``btc``, so we
    # override a small set of well-known codes.
    _ALTNAME_OVERRIDES = {
        "XBT": "BTC",
        "XDG": "DOGE",
        "ZUSD": "USD",
        "ZEUR": "EUR",
        "ZGBP": "GBP",
        "ZJPY": "JPY",
        "ZCAD": "CAD",
        "ZAUD": "AUD",
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
        # Cache of ``/0/public/AssetPairs`` ``result`` dict, keyed by
        # pair altname (``XBTUSDT``, ``BGBUSD``, ``AUSDUSD``, ...).
        # ``klines()`` needs this to recover the (base, quote) of a
        # symbol without relying on a string-split heuristic that
        # breaks for short or ambiguous tickers.
        self._pairs_cache: dict[str, dict] | None = None
        # Set of Kraken asset codes that are **actually** margin-enabled
        # for at least one online USDT/USDC/USD pair. Derived from
        # ``/AssetPairs`` by collecting the ``base`` of any online pair
        # with non-empty ``leverage_buy`` (Kraken uses cross margin
        # only, so isolated marginability is irrelevant). An asset that
        # has a non-null ``margin_rate`` on ``/Assets`` but no marginable
        # pair is **not** included -- the per-asset field exists for
        # informational purposes but the asset cannot actually be
        # borrowed against, so we emit ``cross_rate=null`` for those
        # pairs.
        self._marginable_bases: set[str] | None = None

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
            self._marginable_bases = set()
            return self._altname_cache
        if data.get("error"):
            l.warning(f"Kraken assets API error: {data['error']}")
            self._altname_cache = {}
            self._margin_cache = {}
            self._marginable_bases = set()
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

    async def _load_pairs(self, client: AsyncClient) -> dict[str, dict]:
        """``{pair_altname: pair_info}`` for all Kraken pairs.

        The first call hits ``/0/public/AssetPairs``; subsequent calls
        reuse the cache. ``pair_info`` is the raw dict from the
        ``result`` field, which includes ``base`` (Kraken asset code)
        and ``quote`` (Kraken asset code). ``klines()`` uses this to
        recover the correct base/quote of a symbol without relying
        on a string-split heuristic.
        """
        if self._pairs_cache is not None:
            return self._pairs_cache
        try:
            resp = await client.get(
                f"https://{self.HOST}/0/public/AssetPairs", timeout=30.0
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            raise RuntimeError(f"Kraken AssetPairs request failed: {e}") from e
        errs = data.get("error") or []
        if errs:
            if any(e in self.TRANSIENT_ERRORS for e in errs):
                raise TransientError(
                    f"Kraken AssetPairs transient: {errs}"
                )
            raise RuntimeError(f"Kraken AssetPairs API error: {errs}")
        self._pairs_cache = data.get("result", {}) or {}
        # Derive the set of margin-enabled base assets from
        # ``/AssetPairs``. A pair is marginable if it has at least one
        # buy-side leverage level (``leverage_buy`` non-empty).
        # ``leverage_sell`` is a redundant signal for cross-margin
        # only pairs (Kraken has no isolated margin), so we just
        # check ``leverage_buy``. We restrict to online pairs so a
        # recently delisted pair doesn't leak into the set.
        marginable: set[str] = set()
        for info in self._pairs_cache.values():
            if info.get("status") != "online":
                continue
            lev_buy = info.get("leverage_buy") or []
            if lev_buy:
                base_code = info.get("base", "")
                if base_code:
                    marginable.add(base_code)
        self._marginable_bases = marginable
        return self._pairs_cache

    # ------------------------------------------------------------------
    # pairs
    # ------------------------------------------------------------------
    async def pairs(self, client: AsyncClient, quote_assets: set[str]) -> pl.DataFrame:
        """Returns active USDT/USDC spot pairs.

        Columns: ts, symbol, exchange, base, quote, cross_rate,
        isolated_rate, funding_rate
        ``cross_rate`` is the per-asset annual borrow rate from
        ``/0/public/Assets`` (Kraken only supports cross margin).
        ``isolated_rate`` is always null. ``funding_rate`` is
        always null because Kraken has no native perpetuals --
        only dated futures, which settle and don't have a
        funding rate.

        A pair's ``cross_rate`` is ``null`` if the base asset is not
        actually margin-enabled, even if ``/Assets`` reports a
        non-null ``margin_rate`` for that asset. We use the
        ``/AssetPairs`` ``leverage_buy`` field (non-empty means at
        least one leverage level is offered) as the authoritative
        signal that the pair can actually be margin-traded.
        """
        now = dt.datetime.now(dt.timezone.utc)
        altnames = await self._load_altnames(client)
        # ``_load_altnames`` populates ``_margin_cache`` and
        # ``_load_pairs`` populates ``_marginable_bases`` -- snapshot
        # the value **after** awaiting ``_load_pairs`` so we read the
        # freshly-populated cache rather than the pre-call ``None``.
        margins = self._margin_cache or {}
        pairs_map = await self._load_pairs(client)
        marginable_bases = self._marginable_bases or set()

        quote_set = {self._common_ticker(q).upper() for q in quote_assets}
        rows: list[dict] = []
        for altname, info in pairs_map.items():
            if info.get("status") != "online":
                continue
            base_raw = info.get("base", "")
            quote_raw = info.get("quote", "")
            if not base_raw or not quote_raw:
                continue

            # Map Kraken codes (e.g. ``XXBT``, ``ZUSD``) to common altnames
            # (``XBT``, ``USD``) then to conventional tickers (``BTC``, ``USD``).
            base_altname = altnames.get(base_raw, base_raw)
            quote_altname = altnames.get(quote_raw, quote_raw)

            base_common = self._common_ticker(base_altname)
            quote_common = self._common_ticker(quote_altname)

            if quote_common.upper() not in quote_set:
                continue

            # Kraken is cross-margin only: the borrow rate is a
            # per-asset field on /Assets, not per-pair. Emit ``null``
            # if the base asset is not actually margin-enabled for any
            # online pair, even if /Assets reports a rate for it.
            cross_rate = (
                margins.get(base_raw) if base_raw in marginable_bases else None
            )
            rows.append(
                {
                    "ts": now,
                    "symbol": altname,
                    "exchange": self.NAME,
                    "base": base_common.lower(),
                    "quote": quote_common.lower(),
                    "cross_rate": cross_rate,
                    "isolated_rate": None,
                    # Kraken has no native perpetuals (only dated
                    # futures, which settle and don't have a
                    # funding rate). The base-class default
                    # ``_fetch_funding_rates`` returns ``{}``,
                    # so this column is ``null`` for every Kraken
                    # row.
                    "funding_rate": None,
                }
            )

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
        are returned. ``close_ts`` is the inclusive last instant of the daily
        candle (``open_ts + 24h - 1us``). The result is capped at
        ``MAX_KLINES`` daily candles; use ``klines_paged()`` for wider
        ranges.
        """
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=dt.timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=dt.timezone.utc)

        # Kraken OHLC takes the pair altname (e.g. XBTUSDT, BGBUSD).
        symbol = symbol.upper()
        # Look the pair up in the cached ``/AssetPairs`` data so we
        # get the canonical ``base``/``quote`` Kraken codes instead
        # of relying on a string-split heuristic that breaks for
        # short or ambiguous tickers (e.g. ``BGBUSD`` -> base=``BGB``
        # not ``BGBU``, ``AUSDUSD`` -> base=``AUSD`` not ``AUSD``).
        altnames = await self._load_altnames(client)
        pairs_map = await self._load_pairs(client)
        pair_info = pairs_map.get(symbol)
        if pair_info is None:
            # Symbol not in /AssetPairs: keep the same failure mode
            # the caller would see on the OHLC endpoint, but with a
            # clearer message.
            raise RuntimeError(
                f"Kraken unknown symbol {symbol!r}: not in /AssetPairs"
            )
        base_raw = pair_info.get("base", "")
        quote_raw = pair_info.get("quote", "")
        # Translate Kraken codes (e.g. ``XXBT``, ``ZUSD``) to the
        # common altname (``XBT``, ``USD``) and then to the
        # conventional ticker (``BTC``, ``USD``) so the result
        # matches the ``base``/``quote`` columns emitted by
        # ``pairs()``.
        base = self._common_ticker(altnames.get(base_raw, base_raw))
        quote = self._common_ticker(altnames.get(quote_raw, quote_raw))

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
        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            raise RuntimeError(f"Kraken OHLC request failed: {e}") from e
        errs = payload.get("error") or []
        if errs:
            if any(e in self.TRANSIENT_ERRORS for e in errs):
                raise TransientError(
                    f"Kraken OHLC transient: {errs}"
                )
            raise RuntimeError(f"Kraken OHLC API error: {errs}")

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
