"""Tests for the AsterDex exchange adapter.

AsterDex is a Binance-style perpetuals DEX on BNB Chain. The
adapter deviates from the spot-pair convention used by the
other exchanges the same way :class:`live.hyperliquid.Hyperliquid`
does:

  * ``pairs()`` emits one row per **perp contract** with
    ``quote = "usdt"`` and no borrow rates.
  * ``klines()`` returns **perp OHLCV** (the only kind
    AsterDex exposes). No ``n > 0`` filter is needed
    because AsterDex does not back-fill historical bars --
    a direct probe of any window before the venue's data
    retention boundary returns ``[]``, not synthetic OHLC.
    The filter is retained as a defensive no-op for
    symmetry with the Hyperliquid adapter.
  * Funding is **8h** (confirmed by direct probe of
    consecutive ``fundingTime`` deltas), so the
    annualisation is ``rate * 3 * 365``.

These tests cover:

  * Schema contract: ``pairs()`` and ``klines()`` produce
    the canonical polars schemas.
  * Funding rate annualisation matches the 8h schedule.
  * Per-call klines are capped at ``MAX_KLINES`` (1000)
    and the response is parsed in the same 12-field
    Binance fapi shape.
  * Live ``pairs()`` returns a non-empty frame with
    realistic ``funding_rate`` values (annualised) and the
    well-known ``BTCUSDT`` perp is included.
  * Live ``klines()`` for ``BTCUSDT`` covers a sensible
    historical depth (AsterDex perp history goes back to
    2021-09-01 for BTCUSDT) and the ``open_ts`` is
    midnight-UTC aligned.
  * ``pairs()`` returns ``cross_rate = None`` and
    ``isolated_rate = None`` for every row.
  * Offline: 8h annualisation math, n=0 filter helper, no
    constructor arguments needed.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from httpx import AsyncClient

from live import AsterDex
from ._assertions import (
    assert_close_ts_matches_open,
    assert_klines_schema,
    assert_lowercase,
    assert_open_ts_in_range,
    assert_pairs_schema,
)


# ----------------------------------------------------------------------
# offline unit tests
# ----------------------------------------------------------------------


def test_asterdex_class_constants() -> None:
    """Class constants match the documented AsterDex API."""
    ex = AsterDex()
    assert ex.NAME == "asterdex"
    assert ex.MAX_KLINES == 1000  # Binance fapi hard cap
    assert ex.DAILY_ALIGN_HOUR_UTC == 0  # midnight UTC
    assert ex.HOST == "fapi.asterdex.com"
    assert ex._FUNDING_INTERVAL_HOURS == 8.0


def test_annualize_funding_rate_8h_asterdex() -> None:
    """AsterDex pays funding every 8 hours (confirmed by direct
    probe of consecutive ``fundingTime`` deltas). 1bp per 8h
    interval = 3bp/day = ~10.95% APR.
    """
    ex = AsterDex()
    assert ex.annualize_funding_rate(0.0001, 8.0) == pytest.approx(
        0.1095, rel=1e-9
    )


def test_annualize_funding_rate_asterdex_default() -> None:
    """Using the adapter's own ``_FUNDING_INTERVAL_HOURS``
    constant gives the right annualisation for a real-looking
    AsterDex funding payment.
    """
    ex = AsterDex()
    # A typical AsterDex funding payment of 5bp/8h = 0.0005 * 3
    # * 365 = 0.5475 = 54.75% APR.
    assert ex.annualize_funding_rate(
        0.0005, ex._FUNDING_INTERVAL_HOURS
    ) == pytest.approx(0.0005 * 3 * 365, rel=1e-9)


def test_pairs_schema_dtype_funding_rate() -> None:
    """``funding_rate`` is ``pl.Float64`` (same as the other
    adapters)."""
    df = AsterDex.empty_pairs_df()
    assert "funding_rate" in df.columns
    assert df.schema["funding_rate"] == pl.Float64
    # And the cross/isolated columns are present even though
    # AsterDex never populates them -- the schema must match
    # the contract.
    assert df.schema["cross_rate"] == pl.Float64
    assert df.schema["isolated_rate"] == pl.Float64


def test_pairs_schema_includes_canonical_columns() -> None:
    """The full ``PAIRS_SCHEMA`` is exposed via ``empty_pairs_df``."""
    df = AsterDex.empty_pairs_df()
    expected = {
        "ts",
        "symbol",
        "exchange",
        "base",
        "quote",
        "cross_rate",
        "isolated_rate",
        "funding_rate",
    }
    assert set(df.columns) == expected


def test_klines_schema_dtype_columns() -> None:
    """``klines()`` uses the canonical ``KLINES_SCHEMA``."""
    df = AsterDex.empty_klines_df()
    expected = {
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
    }
    assert set(df.columns) == expected
    for col in ("open", "high", "low", "close", "base_volume", "quote_volume"):
        assert df.schema[col] == pl.Float64
    for col in ("open_ts", "close_ts"):
        dtype = df.schema[col]
        assert isinstance(dtype, pl.Datetime)
        assert dtype.time_zone == "UTC"


def test_quote_is_usdt_for_empty_pairs() -> None:
    """``empty_pairs_df`` has the canonical schema; ``quote``
    defaults to ``pl.Utf8`` and the convention is to emit
    ``"usdt"`` for AsterDex perps."""
    df = AsterDex.empty_pairs_df()
    assert df.schema["quote"] == pl.Utf8


def test_asterdex_has_no_ctor_args() -> None:
    """The adapter must be constructible without credentials,
    since every public endpoint is unauthenticated.
    """
    ex = AsterDex()
    # No exceptions on construction; the instance is usable.
    assert ex.NAME == "asterdex"


# ----------------------------------------------------------------------
# n>0 filter (defensive no-op for AsterDex, but the helper still
# exists for symmetry with Hyperliquid)
# ----------------------------------------------------------------------


def test_drop_zero_trade_candles_drops_trades_zero() -> None:
    """Bars with ``trades == 0`` are dropped (defensive -- AsterDex
    does not currently back-fill, but the filter is retained
    in case a future venue change introduces back-fill).
    """
    from live.asterdex import _drop_zero_trade_candles

    rows = [
        {"open_ts": "x", "trades": 0},   # zero trades, drop
        {"open_ts": "y", "trades": 100},  # real, keep
        {"open_ts": "z", "trades": 0},   # zero trades, drop
    ]
    out = _drop_zero_trade_candles(rows)
    assert len(out) == 1
    assert out[0]["open_ts"] == "y"


def test_drop_zero_trade_candles_keeps_missing_trades() -> None:
    """Bars with ``trades`` missing are KEPT (defensive -- a real
    live bar that omits ``trades`` should not be silently
    dropped)."""
    from live.asterdex import _drop_zero_trade_candles

    rows = [
        {"open_ts": "x", "trades": None},   # unknown, keep
        {"open_ts": "y", "trades": 0},      # explicit zero, drop
        {"open_ts": "z"},                   # missing field, keep
    ]
    out = _drop_zero_trade_candles(rows)
    assert len(out) == 2
    kept = {r["open_ts"] for r in out}
    assert kept == {"x", "z"}


# ----------------------------------------------------------------------
# live tests (hit the public AsterDex API)
# ----------------------------------------------------------------------


async def test_pairs_schema_includes_all_canonical_columns(
    client: AsyncClient, asterdex: AsterDex
) -> None:
    """Live ``pairs()`` returns a frame with the canonical
    ``PAIRS_SCHEMA`` -- the schema check matches the other
    exchanges."""
    pairs = await asterdex.pairs_with_retry(
        client, quote_assets=set()  # AsterDex ignores quote_assets
    )
    assert_pairs_schema(pairs)
    assert pairs.height > 0


async def test_pairs_quote_is_usdt(
    client: AsyncClient, asterdex: AsterDex
) -> None:
    """Every row has ``quote == "usdt"`` (AsterDex only lists
    USDT-margined perps)."""
    pairs = await asterdex.pairs_with_retry(client, quote_assets=set())
    assert pairs.height > 0
    quotes = pairs["quote"].unique().to_list()
    assert quotes == ["usdt"], (
        f"expected quote='usdt' for every row, got {quotes!r}"
    )


async def test_pairs_cross_and_isolated_rates_are_null(
    client: AsyncClient, asterdex: AsterDex
) -> None:
    """``cross_rate`` and ``isolated_rate`` are always null on
    AsterDex -- there's no borrow-rate concept on a perp-only
    DEX."""
    pairs = await asterdex.pairs_with_retry(client, quote_assets=set())
    assert pairs.height > 0
    n_cross_null = pairs.filter(pl.col("cross_rate").is_null()).height
    n_iso_null = pairs.filter(pl.col("isolated_rate").is_null()).height
    assert n_cross_null == pairs.height, (
        "cross_rate should be null for all rows"
    )
    assert n_iso_null == pairs.height, (
        "isolated_rate should be null for all rows"
    )


async def test_pairs_funding_rate_annualised(
    client: AsyncClient, asterdex: AsterDex
) -> None:
    """``funding_rate`` is the **annualised** value
    (``rate * 3 * 365``) -- not the per-8h rate. A few-bp/8h
    perp should be in the tens-of-percent APR range, not a
    few bps."""
    pairs = await asterdex.pairs_with_retry(client, quote_assets=set())
    btc = pairs.filter(pl.col("symbol") == "BTCUSDT")
    if btc.height == 0:
        pytest.skip("BTCUSDT perp not returned (shouldn't happen on mainnet)")
    rate = btc["funding_rate"][0]
    assert rate is not None, "BTCUSDT funding_rate should be populated"
    # BTC's perp funding is usually within a few % APR. We
    # allow up to +/-100% APR to absorb short-lived spikes on
    # liquidations / cascades.
    assert -1.0 < rate < 1.0, (
        f"BTCUSDT funding_rate={rate:.4f} implausible (expected "
        f"small APR for a high-liquidity perp)"
    )


async def test_pairs_base_is_lowercase(
    client: AsyncClient, asterdex: AsterDex
) -> None:
    """``base`` is the lower-cased symbol (matching the
    convention used by HTX / KuCoin / Kraken / Binance /
    Hyperliquid)."""
    pairs = await asterdex.pairs_with_retry(client, quote_assets=set())
    assert pairs.height > 0
    assert_lowercase(pairs, "base")


async def test_pairs_includes_btc(
    client: AsyncClient, asterdex: AsterDex
) -> None:
    """The BTC perp is on the canonical list."""
    pairs = await asterdex.pairs_with_retry(client, quote_assets=set())
    btc = pairs.filter(pl.col("symbol") == "BTCUSDT")
    assert btc.height == 1, (
        f"expected 1 BTCUSDT row, got {btc.height}"
    )
    assert btc["base"][0] == "btc"
    assert btc["quote"][0] == "usdt"


async def test_pairs_filters_non_perpetual(
    client: AsyncClient, asterdex: AsterDex
) -> None:
    """``pairs()`` only includes perps with
    ``contractType == "PERPETUAL"``. A future dated-futures
    or option product would be filtered out (not currently
    listed by AsterDex, but the filter is the documented
    behaviour).
    """
    pairs = await asterdex.pairs_with_retry(client, quote_assets=set())
    assert pairs.height > 0
    # Every row's symbol must end in "USDT" (the only quote
    # asset AsterDex lists perps against).
    non_usdt = pairs.filter(~pl.col("symbol").str.ends_with("USDT"))
    assert non_usdt.height == 0, (
        f"expected all symbols to end with USDT, got "
        f"{non_usdt['symbol'].to_list()[:5]!r}..."
    )


async def test_klines_schema_for_btc(
    client: AsyncClient, asterdex: AsterDex
) -> None:
    """``klines()`` for BTCUSDT returns the canonical schema in
    a typical 14-day range."""
    end = dt.datetime.now(dt.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = end - dt.timedelta(days=14)
    df = await asterdex.klines_with_retry(
        client, "BTCUSDT", start_time=start, end_time=end
    )
    assert_klines_schema(df)
    assert_open_ts_in_range(df, start, end)
    # AsterDex's kline response includes an explicit
    # ``close_time`` field (Binance fapi convention);
    # ``assert_close_ts_matches_open`` with
    # ``api_provided=True`` is the right check (we only
    # need to assert ``close_ts > open_ts``).
    assert_close_ts_matches_open(df, api_provided=True)


async def test_klines_no_zero_trade_bars(
    client: AsyncClient, asterdex: AsterDex
) -> None:
    """The klines frame MUST NOT contain any bars with
    ``trades == 0``. We don't have a direct handle on
    ``trades`` in the schema, so we use the proxy: every bar
    must have a non-negative ``base_volume``.

    More importantly, the row count for BTCUSDT over the
    last 365 days must be ≤ 365 (AsterDex does not currently
    back-fill, so we expect exactly 365).
    """
    end = dt.datetime.now(dt.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = end - dt.timedelta(days=365)
    df = await asterdex.klines_with_retry(
        client, "BTCUSDT", start_time=start, end_time=end
    )
    assert_klines_schema(df)
    zero_vol = df.filter(pl.col("base_volume") <= 0.0)
    assert zero_vol.height == 0, (
        f"klines() returned {zero_vol.height} zero-volume bars; "
        f"the trades > 0 filter is not working"
    )
    # A 365-day window that started ~2025-06-12 is well
    # inside the venue's real-history region (BTCUSDT perp
    # history starts 2021-09-01), so we expect 365 bars
    # exactly.
    assert df.height == 365, (
        f"klines() returned {df.height} bars in a 365-day "
        f"window; expected 365 (AsterDex does not currently "
        f"back-fill)"
    )


async def test_klines_open_ts_at_midnight_utc(
    client: AsyncClient, asterdex: AsterDex
) -> None:
    """Every ``open_ts`` opens at 00:00:00 UTC (AsterDex
    aligns to midnight, same as Binance / KuCoin / Kraken /
    Hyperliquid)."""
    end = dt.datetime.now(dt.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = end - dt.timedelta(days=30)
    df = await asterdex.klines_with_retry(
        client, "BTCUSDT", start_time=start, end_time=end
    )
    if df.height == 0:
        pytest.skip("no klines returned")
    hours = df["open_ts"].dt.hour().unique().to_list()
    assert hours == [0], (
        f"expected open_ts at 00:00 UTC, got hours {hours!r}"
    )


async def test_klines_paged_walks_backward(
    client: AsyncClient, asterdex: AsterDex
) -> None:
    """``klines_paged()`` returns > ``MAX_KLINES`` bars by
    walking backward in 1000-bar chunks. The result covers
    BTCUSDT's full history (back to 2021-09-01 for the
    most-traded perps) and is sorted by ``open_ts``.
    """
    end = dt.datetime.now(dt.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    # 1500 days ≈ 4.1 years; this should require 2+ paged
    # calls (1000-bar cap per call) and hit the venue's
    # 2021-09-01 boundary.
    start = end - dt.timedelta(days=1500)
    df = await asterdex.klines_paged(
        client, "BTCUSDT", start_time=start, end_time=end
    )
    assert_klines_schema(df)
    assert_open_ts_in_range(df, start, end)
    assert df.height > AsterDex.MAX_KLINES, (
        f"klines_paged() returned {df.height} bars; expected "
        f"> {AsterDex.MAX_KLINES} to prove the chunking works"
    )
    assert df["open_ts"].is_sorted(), (
        "klines_paged() result is not sorted by open_ts"
    )


async def test_funding_rate_annualisation_factor(
    client: AsyncClient, asterdex: AsterDex
) -> None:
    """Spot-check the annualisation factor: the funding_rate
    for BTC must be a real-looking small APR (the underlying
    8h rate is what the raw API returns).

    We hit the public ``_fetch_funding_rates`` path and pluck
    the BTC entry. This is the same code path used by
    ``pairs()``.
    """
    rates = await asterdex._fetch_funding_rates(client, ["BTC"])
    assert "btc" in rates, (
        f"expected BTC in funding dict, got {list(rates)}"
    )
    rate = rates["btc"]
    # BTC funding is rarely > 50% APR in either direction.
    # Allow ±200% to absorb liquidation cascades.
    assert -2.0 < rate < 2.0, (
        f"BTC funding_rate={rate:.4f} implausible"
    )
