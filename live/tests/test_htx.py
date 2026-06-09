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
            *(htx.klines(client, sym, start, end) for sym in WANTED)
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
    empty = await htx.klines(client, "btcusdt", start, start)
    assert empty.height == 0
    assert_klines_schema(empty)


async def test_klines_paged(
    client: AsyncClient, htx: HTX, paged_window: tuple
) -> None:
    start, end = paged_window
    paged = await htx.klines_paged(
        client, "btcusdt", start_time=start, end_time=end
    )
    capped = await htx.klines(
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
    pairs = await htx.pairs(client, quote_assets={"usdt", "usdc"})
    klines = pl.concat(
        await asyncio.gather(
            *(htx.klines(client, sym, start, end) for sym in WANTED)
        )
    )
    assert_pairs_joinable_to_klines(htx, pairs, klines)
