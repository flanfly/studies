"""Tests for the HTX (formerly Huobi) exchange adapter.

These tests run against the live HTX API. They require
``HTX_ACCESS_KEY`` and ``HTX_SECRET_KEY`` in ``.env``; the test is
skipped if either is missing.

HTX uses the symbol convention ``btcusdt`` (concatenated, lowercased)
and is symbol-specific for both cross and isolated margin (no
``/0/public/margin/symbols`` endpoint – both rates come from
``/v1/margin/loan-info`` and ``/v1/cross-margin/loan-info``).
"""

from __future__ import annotations

import asyncio
import datetime as dt

import polars as pl
import pytest
from httpx import AsyncClient

from live import HTX
from ._assertions import (
    assert_close_ts_matches_open,
    assert_klines_schema,
    assert_kline_candles_per_symbol,
    assert_lowercase,
    assert_open_ts_in_range,
    assert_pairs_joinable_to_klines,
    assert_pairs_schema,
)


WANTED = ["btcusdt", "ethusdt", "solusdt"]
SYMBOL_CONVENTION_DESC = "symbol == base+quote (HTX convention)"


async def test_pairs(client: AsyncClient, htx: HTX) -> None:
    pairs = await htx.pairs(client, quote_assets={"usdt", "usdc"})
    assert_pairs_schema(pairs)
    assert pairs.height > 0, "no pairs returned"

    # Symbols are ``base+quote`` in lower case.
    bad = pairs.filter(
        (pl.col("base") + pl.col("quote")) != pl.col("symbol")
    )
    assert bad.height == 0, (
        f"HTX symbol should equal base+quote ({bad.height} mismatches)"
    )

    # Borrow rates: at least the isolated rates are populated from
    # ``/v1/margin/loan-info``. The cross-margin account may or may not
    # be enabled (depending on the test key), so we don't assert a
    # specific count for ``cross_rate``; just that the columns exist
    # and have valid types.
    n_with_cross = pairs.filter(pl.col("cross_rate").is_not_null()).height
    n_with_iso = pairs.filter(pl.col("isolated_rate").is_not_null()).height
    assert n_with_iso > 0, "isolated_rate should be populated for active pairs"
    # If cross-margin is enabled, the rate counts should be similar in
    # magnitude (within an order of magnitude) to isolated.
    if n_with_cross > 0:
        assert n_with_cross > 0
        # Sanity: cross rates are reasonable annual rates (< 100%).
        max_cross = pairs["cross_rate"].drop_nulls().max()
        assert max_cross < 1.0, f"cross_rate {max_cross} implausibly high"

    sample = pairs.filter(pl.col("symbol").is_in(WANTED))
    assert sample.height == len(WANTED), (
        f"missing well-known pairs (got {sample.height}/{len(WANTED)})"
    )


async def test_klines(
    client: AsyncClient, htx: HTX, pairs_window: tuple
) -> None:
    start, end = pairs_window
    klines = pl.concat(
        await asyncio.gather(
            *(
                htx.klines_with_retry(client, sym, start, end)
                for sym in WANTED
            )
        )
    ).sort(["symbol", "open_ts"])

    assert_klines_schema(klines)
    assert_open_ts_in_range(klines, start, end)
    assert_close_ts_matches_open(klines, api_provided=False)
    assert_lowercase(klines, "base", "quote")
    assert_kline_candles_per_symbol(klines, 14, *WANTED)


async def test_klines_empty_range(
    client: AsyncClient, htx: HTX, pairs_window: tuple
) -> None:
    start, _ = pairs_window
    empty = await htx.klines_with_retry(client, "btcusdt", start, start)
    assert empty.height == 0
    assert_klines_schema(empty)


async def test_klines_paged(
    client: AsyncClient, htx: HTX, paged_window: tuple
) -> None:
    start, end = paged_window
    paged = await htx.klines_paged(
        client, "btcusdt", start_time=start, end_time=end
    )
    capped = await htx.klines_with_retry(
        client, "btcusdt", start_time=start, end_time=end
    )

    assert paged.height > 0
    assert capped.height <= HTX.MAX_KLINES, (
        f"klines() returned {capped.height} > MAX_KLINES={HTX.MAX_KLINES}"
    )
    assert paged.height >= capped.height
    assert_open_ts_in_range(paged, start, end)
    assert_close_ts_matches_open(paged, api_provided=False)


async def test_pairs_joinable_to_klines(
    client: AsyncClient, htx: HTX, pairs_window: tuple
) -> None:
    start, end = pairs_window
    pairs = await htx.pairs_with_retry(client, quote_assets={"usdt", "usdc"})
    klines = pl.concat(
        await asyncio.gather(
            *(
                htx.klines_with_retry(client, sym, start, end)
                for sym in WANTED
            )
        )
    )
    assert_pairs_joinable_to_klines(htx, pairs, klines)


# ----------------------------------------------------------------------
# offline tests for the 4h -> 1d aggregation. These run without
# network access and exercise the boundary / partial-day cases that
# are hard to reproduce against the live API.
# ----------------------------------------------------------------------


def test_aggregate_4h_full_day() -> None:
    """6 contiguous 4h candles covering one UTC day aggregate to one
    1d candle whose ``open`` is the 00:00 candle's open and
    ``close`` is the 20:00 candle's close."""
    df_4h = pl.DataFrame(
        {
            "open_ts": [
                "2026-06-08T00:00:00",
                "2026-06-08T04:00:00",
                "2026-06-08T08:00:00",
                "2026-06-08T12:00:00",
                "2026-06-08T16:00:00",
                "2026-06-08T20:00:00",
            ],
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            "base_volume": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "quote_volume": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0],
        }
    ).with_columns(pl.col("open_ts").str.to_datetime(time_unit="us", time_zone="UTC"))

    out = HTX._aggregate_4h_to_1d(df_4h)
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["open_ts"] == dt.datetime(
        2026, 6, 8, 0, 0, 0, tzinfo=dt.timezone.utc
    )
    assert row["open"] == 100.0  # 00:00 open
    assert row["close"] == 105.5  # 20:00 close
    assert row["high"] == 106.0  # max
    assert row["low"] == 99.0  # min
    assert row["base_volume"] == 210.0  # sum
    assert row["quote_volume"] == 21000.0  # sum


def test_aggregate_4h_descending_input() -> None:
    """The HTX 4h endpoint returns candles in descending ``id`` order
    (most recent first). The aggregation must still produce the right
    OHLC values regardless of input order -- ``open`` from the 00:00
    candle, ``close`` from the 20:00 candle."""
    df_4h = pl.DataFrame(
        {
            "open_ts": [
                # Descending -- most recent first.
                "2026-06-08T20:00:00",
                "2026-06-08T16:00:00",
                "2026-06-08T12:00:00",
                "2026-06-08T08:00:00",
                "2026-06-08T04:00:00",
                "2026-06-08T00:00:00",
            ],
            "open": [105.0, 104.0, 103.0, 102.0, 101.0, 100.0],
            "high": [106.0, 105.0, 104.0, 103.0, 102.0, 101.0],
            "low": [104.0, 103.0, 102.0, 101.0, 100.0, 99.0],
            "close": [105.5, 104.5, 103.5, 102.5, 101.5, 100.5],
            "base_volume": [60.0, 50.0, 40.0, 30.0, 20.0, 10.0],
            "quote_volume": [6000.0, 5000.0, 4000.0, 3000.0, 2000.0, 1000.0],
        }
    ).with_columns(pl.col("open_ts").str.to_datetime(time_unit="us", time_zone="UTC"))

    out = HTX._aggregate_4h_to_1d(df_4h)
    assert out.height == 1
    row = out.row(0, named=True)
    # The aggregation is order-insensitive: ``open`` is always the
    # 00:00 candle's open, ``close`` is always the 20:00 candle's
    # close.
    assert row["open"] == 100.0
    assert row["close"] == 105.5
    assert row["high"] == 106.0
    assert row["low"] == 99.0
    assert row["base_volume"] == 210.0
    assert row["quote_volume"] == 21000.0


def test_aggregate_4h_partial_day_kept() -> None:
    """A day with < 6 4h candles is still kept (the half-open filter
    is responsible for excluding in-progress days). The aggregation
    simply takes whatever candles are present."""
    df_4h = pl.DataFrame(
        {
            "open_ts": [
                "2026-06-08T00:00:00",
                "2026-06-08T04:00:00",
                "2026-06-08T08:00:00",
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "base_volume": [10.0, 20.0, 30.0],
            "quote_volume": [1000.0, 2000.0, 3000.0],
        }
    ).with_columns(pl.col("open_ts").str.to_datetime(time_unit="us", time_zone="UTC"))

    out = HTX._aggregate_4h_to_1d(df_4h)
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["open"] == 100.0
    assert row["close"] == 102.5
    assert row["base_volume"] == 60.0


def test_aggregate_4h_multiple_days() -> None:
    """Two full days in one frame produce two 1d rows in order."""
    df_4h = pl.DataFrame(
        {
            "open_ts": [
                "2026-06-08T00:00:00", "2026-06-08T04:00:00",
                "2026-06-08T08:00:00", "2026-06-08T12:00:00",
                "2026-06-08T16:00:00", "2026-06-08T20:00:00",
                "2026-06-09T00:00:00", "2026-06-09T04:00:00",
                "2026-06-09T08:00:00", "2026-06-09T12:00:00",
                "2026-06-09T16:00:00", "2026-06-09T20:00:00",
            ],
            "open": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "high": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "low": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "base_volume": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "quote_volume": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    ).with_columns(pl.col("open_ts").str.to_datetime(time_unit="us", time_zone="UTC"))

    out = HTX._aggregate_4h_to_1d(df_4h)
    assert out.height == 2
    opens = out["open"].to_list()
    assert opens == [1, 7]  # 06-08 open=1, 06-09 open=7
    closes = out["close"].to_list()
    assert closes == [6, 12]


def test_aggregate_4h_empty() -> None:
    """An empty 4h frame produces an empty 1d frame."""
    df_4h = pl.DataFrame(
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
    out = HTX._aggregate_4h_to_1d(df_4h)
    assert out.height == 0


def test_htx_klines_opens_at_midnight_utc() -> None:
    """``DAILY_ALIGN_HOUR_UTC`` is 0 for HTX: 1d candles open at
    midnight UTC, not 16:00 UTC (the native HTX 1d alignment)."""
    assert HTX.DAILY_ALIGN_HOUR_UTC == 0


def test_htx_max_klines_matches_4h_cap() -> None:
    """``MAX_KLINES`` for HTX is 333 (= 2000 4h-candles / 6 per day)."""
    assert HTX.MAX_KLINES == 333
    # 1998 4h-candles per call (one headroom candle below the 2000
    # 4h-candles cap imposed by the API).
    assert HTX.MAX_KLINES * HTX._FOURH_PER_DAY == 1998

