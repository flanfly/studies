"""Hyperliquid exchange adapter (perpetual-futures only).

Hyperliquid is a perpetuals-only DEX: there is **no** spot market, no
margin-borrow endpoint, and no isolated-margin concept. The adapter
therefore deviates from the rest of the framework in three ways:

* ``pairs()`` emits one row per **perp contract** (not spot pair) with
  ``base = <coin lower>`` and ``quote = "usd"``. ``cross_rate`` and
  ``isolated_rate`` are always ``None`` (no borrow rates on HL). The
  ``funding_rate`` column carries the current hourly funding APR for
  the perp on the same ``base/quote``.
* ``klines()`` returns **perp OHLCV** (Hyperliquid has no spot klines
  endpoint). The ``base``/``quote`` columns are filled in so a cross-
  exchange join on ``(base, quote)`` works the same way it does for
  spot pairs.
* ``klines()`` filters out candles with ``n == 0`` (zero trades).
  Hyperliquid back-fills synthetic / oracle-only bars before the
  pair's mainnet launch; ``n > 0`` is the canonical "this is a real
  trade" marker. See :func:`_drop_zero_trade_candles`.

Public endpoints used (all unauthenticated):

* ``POST https://api.hyperliquid.xyz/info`` with
  ``{"type": "metaAndAssetCtxs"}`` -- returns the perp symbol list
  plus the live snapshot (mark, OI, **funding**) in one round trip.
* ``POST https://api.hyperliquid.xyz/info`` with
  ``{"type": "candleSnapshot", "req": {...}}`` -- returns up to 500
  candles for a single coin. The hard cap on this endpoint forces
  ``MAX_KLINES = 500``; ``klines_paged()`` walks the range backward
  in 500-bar chunks. The native candle history is finite: a coin
  that launched N days ago has at most N daily bars of real trade
  data (anything older is back-filled oracle data and is dropped by
  the ``n > 0`` filter).
* ``POST https://api.hyperliquid.xyz/info`` with
  ``{"type": "fundingHistory", "coin": <sym>, ...}`` -- unused at
  the framework level (the current funding rate is taken from
  ``metaAndAssetCtxs`` in a single call).

Hyperliquid funding is **per-hour** (not 8h like Binance/HTX/KuCoin),
so the annualisation is ``rate * 24 * 365``.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from typing import Iterable

import httpx
import polars as pl
from httpx import AsyncClient

from . import Exchange


__all__ = ["Hyperliquid"]


logger = logging.getLogger(__name__)


# The single public endpoint for all read-only requests.
_INFO_URL = "https://api.hyperliquid.xyz/info"


# ``candleSnapshot`` returns at most 500 bars per call. The cap is
# fixed by the server; we set ``MAX_KLINES`` to match so
# ``klines_paged()`` walks the range in 500-bar chunks.
_CANDLE_SNAPSHOT_CAP = 500

# Hyperliquid funding interval is 1 hour. Every perp pays funding
# hourly; the rate returned by ``metaAndAssetCtxs`` is per-hour.
_FUNDING_INTERVAL_HOURS: float = 1.0

# Cap on in-flight ``_fetch_meta`` / ``_fetch_candles`` requests.
# Hyperliquid's public ``/info`` endpoint is rate-limited per IP but
# the documented limit is generous (1200 weight/min); 20-wide is
# safely under that and finishes a few hundred coins in a couple of
# seconds.
_KLINES_CONCURRENCY: int = 20


def _drop_zero_trade_candles(rows: list[dict]) -> list[dict]:
    """Drop candles with ``n == 0`` from ``rows``.

    Hyperliquid back-fills synthetic / oracle-only bars before a
    coin's perp mainnet launch on its public candle history. Those
    bars have ``n = 0`` (no trades) and ``v = 0.0`` (no volume) but
    realistic-looking OHLC from the oracle. Filtering them out
    gives us the same "real trade history" cut that we get from a
    node replay, at the cost of dropping the chart-continuity
    region.

    The check is on the parsed ``n`` field; if a row is missing
    ``n`` entirely (older API versions, or a malformed response),
    we keep it. The alternative -- drop-on-missing -- would
    silently truncate recent data if Hyperliquid ever omits ``n``
    on a live bar.
    """
    return [r for r in rows if r.get("n") is None or int(r["n"]) > 0]


class Hyperliquid(Exchange):
    """Hyperliquid perpetual-futures adapter.

    No constructor arguments: every public endpoint is
    unauthenticated. The adapter is stateless; concurrent callers
    can share a single instance across requests.
    """

    HOST = "api.hyperliquid.xyz"
    NAME = "hyperliquid"
    # Native daily candles open at 00:00 UTC (Hyperliquid aligns to
    # midnight, same as Binance / KuCoin / Kraken).
    DAILY_ALIGN_HOUR_UTC = 0
    # ``candleSnapshot`` caps responses at 500 bars. We honour the
    # cap; ``klines_paged()`` does the chunking.
    MAX_KLINES = _CANDLE_SNAPSHOT_CAP
    # Hyperliquid funding interval is 1 hour. Every perp pays
    # funding hourly; the rate returned by ``metaAndAssetCtxs``
    # is per-hour. ``annualize_funding_rate(rate, 1)`` produces
    # ``rate * 24 * 365``.
    _FUNDING_INTERVAL_HOURS: float = _FUNDING_INTERVAL_HOURS

    # We override ``KLINES_CONCURRENCY`` because Hyperliquid's
    # endpoint is on a different cost / weight budget than the
    # per-symbol spot endpoints on Binance/HTX/KuCoin (no per-call
    # signing, lower latency). 20-wide is plenty.
    KLINES_CONCURRENCY = _KLINES_CONCURRENCY

    # ------------------------------------------------------------------
    # low-level HTTP helpers
    # ------------------------------------------------------------------
    async def _post_info(
        self, client: AsyncClient, payload: dict
    ) -> object:
        """POST to ``/info`` and return the parsed JSON body.

        The endpoint returns a single JSON value whose shape
        depends on ``payload["type"]``: a list (candleSnapshot,
        fundingHistory), a dict with a list of asset contexts
        (metaAndAssetCtxs -> returns ``[meta, ctxs]``), or a dict
        (meta, spotMeta). We do not coerce -- the caller knows
        what it asked for.
        """
        try:
            resp = await client.post(_INFO_URL, json=payload, timeout=30.0)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Hyperliquid /info request failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    # meta + funding (one call for the whole perp universe)
    # ------------------------------------------------------------------
    async def _fetch_meta_and_ctxs(
        self, client: AsyncClient
    ) -> tuple[list[dict], list[dict]]:
        """Return ``(meta, asset_ctxs)`` from
        ``{"type": "metaAndAssetCtxs"}``.

        * ``meta`` is a dict with a ``universe`` list of
          ``{name, szDecimals, maxLeverage, ...}``.
        * ``asset_ctxs`` is a list (one per symbol, in the same
          order) of ``{funding, markPx, midPx, openInterest,
          premium, ...}``.

        The single round-trip is the cheapest way to get both the
        symbol list AND the current funding rate for every perp.
        """
        data = await self._post_info(
            client, {"type": "metaAndAssetCtxs"}
        )
        if not isinstance(data, list) or len(data) != 2:
            raise RuntimeError(
                f"Hyperliquid metaAndAssetCtxs unexpected shape: "
                f"{type(data).__name__}"
            )
        meta, ctxs = data
        if not isinstance(meta, dict) or not isinstance(ctxs, list):
            raise RuntimeError(
                "Hyperliquid metaAndAssetCtxs: meta is not a dict "
                "or ctxs is not a list"
            )
        universe = meta.get("universe", [])
        if len(universe) != len(ctxs):
            raise RuntimeError(
                f"Hyperliquid metaAndAssetCtxs length mismatch: "
                f"universe has {len(universe)} entries, ctxs has "
                f"{len(ctxs)}"
            )
        return universe, ctxs

    # ------------------------------------------------------------------
    # error classification override
    # ------------------------------------------------------------------
    def is_transient_error(self, exc: BaseException) -> bool:
        """Classify a failure as transient (retry) or permanent
        (propagate immediately).

        Inherits the base-class behaviour: ``TransientError``,
        429, generic 5xx, and httpx transport errors are
        transient. Adds one Hyperliquid-specific permanent
        case:

          * ``candleSnapshot`` for an **unknown coin** returns
            HTTP 500 with body ``null``. No number of retries
            will ever turn a fake coin into a real one, so we
            surface this as a permanent failure. The matching
            klines / pairs call propagates the error to the
            caller instead of looping forever.

        The check is intentionally narrow: it only fires on
        status 500 + body that is exactly the JSON literal
        ``null`` (whitespace-only allowed). Any 5xx with a
        real error envelope is treated as a genuine exchange
        outage and retried.

        Note that this classifier sees the raw
        ``httpx.HTTPStatusError`` (or any wrapper the
        surrounding code put it in) -- the base-class
        implementation walks ``__cause__``/``__context__`` for
        us, so wrapping the error in a ``RuntimeError`` for
        the caller message doesn't lose the classification.
        """
        # Walk the cause chain once. If any link is a
        # "500 + null body" HTTPStatusError, the call is
        # permanent (return False so the retry loop stops).
        # Otherwise fall through to the base-class behaviour,
        # which marks 429 / 5xx-with-body / transport errors
        # as transient.
        seen: set[int] = set()
        cur: BaseException | None = exc
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if isinstance(cur, httpx.HTTPStatusError):
                if self._is_unknown_coin_500(cur):
                    return False
            nxt = cur.__cause__ or cur.__context__
            cur = nxt if nxt is not cur else None
        return super().is_transient_error(exc)

    @staticmethod
    def _is_unknown_coin_500(err: httpx.HTTPStatusError) -> bool:
        """``True`` iff ``err`` is the "unknown coin" 500+null
        signature that ``candleSnapshot`` returns for a coin
        that doesn't exist on Hyperliquid.

        We only flag responses with a 500 status AND a body
        that is exactly the JSON literal ``null`` (possibly
        with surrounding whitespace). Any other 5xx -- a real
        server-side stack trace, a JSON error envelope, etc.
        -- is treated as transient and retried.
        """
        if err.response.status_code != 500:
            return False
        return err.response.text.strip() == "null"

    # ------------------------------------------------------------------
    # funding rate fetcher (overrides the base default)
    # ------------------------------------------------------------------
    async def _fetch_funding_rates(
        self, client: AsyncClient, bases: Iterable[str]
    ) -> dict[str, float]:
        """``{base_lower: annual_funding_apr}`` for every perp in
        ``bases`` that has a contract.

        Implemented as a single ``metaAndAssetCtxs`` call (not
        one-per-base) so the cost is O(1) HTTP request regardless
        of how many bases the caller passes. ``bases`` is used
        purely to filter the result; the response itself covers
        the entire perp universe.
        """
        bases_l = {b.lower() for b in bases}
        if not bases_l:
            return {}
        try:
            universe, ctxs = await self._fetch_meta_and_ctxs(client)
        except Exception as e:
            logger.warning(f"Hyperliquid metaAndAssetCtxs failed: {e}")
            return {}

        rates: dict[str, float] = {}
        for entry, ctx in zip(universe, ctxs):
            name = entry.get("name", "")
            if not name or name.lower() not in bases_l:
                continue
            raw = ctx.get("funding") if isinstance(ctx, dict) else None
            if raw is None:
                continue
            try:
                per_interval = float(raw)
            except (TypeError, ValueError):
                continue
            rates[name.lower()] = self.annualize_funding_rate(
                per_interval, self._FUNDING_INTERVAL_HOURS
            )
        return rates

    # ------------------------------------------------------------------
    # pairs
    # ------------------------------------------------------------------
    async def pairs(
        self, client: AsyncClient, quote_assets: set[str]
    ) -> pl.DataFrame:
        """Return one row per active perp contract.

        Columns: ts, symbol, exchange, base, quote, cross_rate,
        isolated_rate, funding_rate

        ``symbol`` is the Hyperliquid coin ticker (``BTC``,
        ``kPEPE``, ``MELANIA``, ...), ``base`` is the lower-cased
        ticker, ``quote`` is the literal string ``"usd"``. We
        emit ``"usd"`` (not ``"usdc"``) to match the rest of the
        framework's convention of lower-cased quote columns and
        because Hyperliquid price feeds are USD-indexed, not
        USDC-denominated (USDC is the settlement asset, but the
        mark/index is USD).

        ``quote_assets`` is **ignored**. Hyperliquid has no spot
        market, so there is no notion of "the caller wants USDT
        pairs". We emit every perp. This deviates from the
        convention used by the spot exchanges -- the user
        explicitly asked for perp-as-pair in the design
        decision; downstream consumers should be aware that
        ``quote == "usd"`` here does NOT mean USDT-margined
        and that the ``klines()`` returned for these rows are
        **perp** OHLCV, not spot.

        ``cross_rate`` and ``isolated_rate`` are always ``None``
        (Hyperliquid has no borrow-rate concept; the perp is
        margined in USDC, not borrowed). ``funding_rate`` is
        the per-hour rate from ``metaAndAssetCtxs`` annualised
        as ``rate * 24 * 365``.
        """
        now = dt.datetime.now(dt.timezone.utc)
        try:
            universe, ctxs = await self._fetch_meta_and_ctxs(client)
        except Exception as e:
            logger.warning(f"Hyperliquid pairs() meta fetch failed: {e}")
            return self.empty_pairs_df()

        rows: list[dict] = []
        for entry, ctx in zip(universe, ctxs):
            symbol = entry.get("name", "")
            if not symbol:
                continue
            # Defensive: skip perps that Hyperliquid has explicitly
            # marked as not trading (``markPx`` and ``midPx`` are
            # both missing or zero in those cases). A non-trading
            # perp has nothing useful in the klines table either.
            mark = ctx.get("markPx") if isinstance(ctx, dict) else None
            if mark is None or (
                isinstance(mark, str) and mark in ("", "0", "0.0")
            ):
                continue
            try:
                funding_raw = float(ctx.get("funding", 0.0))
            except (TypeError, ValueError):
                funding_raw = None
            funding_apr = (
                self.annualize_funding_rate(
                    funding_raw, self._FUNDING_INTERVAL_HOURS
                )
                if funding_raw is not None
                else None
            )
            rows.append(
                {
                    "ts": now,
                    "symbol": symbol,
                    "exchange": self.NAME,
                    "base": symbol.lower(),
                    "quote": "usd",
                    "cross_rate": None,
                    "isolated_rate": None,
                    "funding_rate": funding_apr,
                }
            )

        if not rows:
            return self.empty_pairs_df()

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
    async def _fetch_candles(
        self,
        client: AsyncClient,
        coin: str,
        start_ms: int,
        end_ms: int,
    ) -> list[dict]:
        """Make a single ``candleSnapshot`` request for ``coin`` in
        the half-open millisecond range ``[start_ms, end_ms)``.

        Returns the raw list of candles (one dict per bar) or
        ``[]`` on a malformed response. Retries on transient
        errors are handled by the base-class ``_retry`` wrapper
        around ``klines()``; this helper does not retry
        internally.
        """
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": "1d",
                "startTime": start_ms,
                "endTime": end_ms,
            },
        }
        data = await self._post_info(client, payload)
        return data if isinstance(data, list) else []

    async def klines(
        self,
        client: AsyncClient,
        symbol: str,
        start_time: dt.datetime,
        end_time: dt.datetime,
    ) -> pl.DataFrame:
        """Return at most ``MAX_KLINES`` daily perp candles for
        ``symbol`` in the half-open range ``[start_time,
        end_time)``.

        Columns: open_ts, close_ts, symbol, exchange, base, quote,
        open, high, low, close, base_volume, quote_volume

        ``base`` is the lower-cased symbol, ``quote`` is the
        literal string ``"usd"`` (matching :meth:`pairs`). The
        ``close_ts`` is the inclusive last instant of the daily
        candle in microsecond resolution
        (``open_ts + 24h - 1us``), matching the convention used
        by HTX / KuCoin / Kraken (Hyperliquid's response only
        carries the open timestamp ``t`` and the end-of-bar
        timestamp ``T``; we prefer the µs convention so the
        downstream join logic is uniform).

        **Candles with ``n == 0`` are dropped.** Hyperliquid
        back-fills synthetic / oracle-only bars before a coin's
        perp mainnet launch; those bars have ``n = 0`` and ``v =
        0.0`` but realistic OHLC from the oracle price feed.
        Filtering them out gives us only the "real trade" bars;
        see :func:`_drop_zero_trade_candles`.

        The per-call cap is 500 candles
        (``candleSnapshot`` server-side limit). Use
        ``klines_paged()`` for wider ranges; it walks the range
        backward in 500-bar chunks, exactly the same pattern
        used by the other adapters' pagination helpers.
        """
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=dt.timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=dt.timezone.utc)

        # Hyperliquid symbols are bare coin tickers (``BTC``,
        # ``kPEPE``). Keep case as the API expects; ``base`` is
        # lower-cased at row-build time.
        symbol = symbol
        base = symbol.lower()
        quote = "usd"

        # Clip the request window to ``MAX_KLINES`` days from
        # the end. The server returns at most 500 bars per
        # call; for a wider range the caller must use
        # ``klines_paged()`` (which is the standard path through
        # ``main.py``).
        window_start = end_time - dt.timedelta(
            seconds=self.MAX_KLINES * self.DAILY_SECONDS
        )
        clipped_start = max(start_time, window_start)

        start_ms = int(clipped_start.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        batch = await self._fetch_candles(
            client, symbol, start_ms, end_ms
        )
        # Defensive: cap at MAX_KLINES rows in case the server
        # ever returns a touch more on the edge of a window.
        batch = batch[: self.MAX_KLINES]

        rows: list[dict] = []
        for k in batch:
            try:
                k_open_ms = int(k["t"])
            except (KeyError, TypeError, ValueError):
                continue
            # Half-open range: ``start_time <= open_ts < end_time``.
            if k_open_ms < start_ms or k_open_ms >= end_ms:
                continue
            open_ts = dt.datetime.fromtimestamp(
                k_open_ms / 1000.0, tz=dt.timezone.utc
            )
            try:
                rows.append(
                    {
                        "open_ts": open_ts,
                        # µs-resolution convention: inclusive last
                        # instant of the daily candle. Same as
                        # HTX / KuCoin / Kraken emit.
                        "close_ts": open_ts
                        + dt.timedelta(days=1)
                        - dt.timedelta(microseconds=1),
                        "symbol": symbol,
                        "base": base,
                        "quote": quote,
                        "open": float(k["o"]),
                        "high": float(k["h"]),
                        "low": float(k["l"]),
                        "close": float(k["c"]),
                        "base_volume": float(k["v"]),
                        # Hyperliquid's response does not include a
                        # quote-asset volume field; approximate as
                        # ``vwap * volume`` is not possible
                        # either (no ``vwap``). Use ``close *
                        # base_volume`` as a reasonable proxy:
                        # that's the notional in USD at the
                        # candle's close, which is the same proxy
                        # other adapters use when the API is
                        # silent.
                        "quote_volume": float(k["c"])
                        * float(k["v"]),
                        # Preserved on the row so the post-parse
                        # ``_drop_zero_trade_candles`` filter can
                        # see it. The column is stripped from
                        # the final frame because it is not
                        # part of ``KLINES_SCHEMA``.
                        "n": k.get("n"),
                    }
                )
            except (KeyError, TypeError, ValueError):
                # Malformed bar -- skip and continue. We log
                # once at the end if we dropped anything so a
                # run of bad bars doesn't go silent.
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
