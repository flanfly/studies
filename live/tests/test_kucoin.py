"""Tests for the KuCoin exchange adapter.

These tests run against the live KuCoin API. They require
``KUCOIN_API_KEY``, ``KUCOIN_API_SECRET``, and ``KUCOIN_API_PASSWORD``
in ``.env``; the test is skipped if any are missing.

KuCoin uses the symbol convention ``BTC-USDT`` (dash-separated, upper
case). Borrow rates are currency-level (not pair-level) and the
``borrowRate`` endpoint is case-sensitive.
"""

from __future__ import annotations

import asyncio

import polars as pl
import pytest
from httpx import AsyncClient

from live import KuCoin
from ._assertions import (
    assert_close_ts_matches_open,
    assert_klines_schema,
    assert_kline_candles_per_symbol,
    assert_lowercase,
    assert_open_ts_in_range,
    assert_pairs_joinable_to_klines,
    assert_pairs_schema,
)


WANTED = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]


async def test_pairs(client: AsyncClient, kucoin: KuCoin) -> None:
    pairs = await kucoin.pairs(client, quote_assets={"usdt", "usdc"})
    assert_pairs_schema(pairs)
    assert pairs.height > 0, "no pairs returned"

    # Symbols are ``UPPER(base)-UPPER(quote)``.
    bad = pairs.filter(
        (pl.col("base").str.to_uppercase() + "-" + pl.col("quote").str.to_uppercase())
        != pl.col("symbol")
    )
    assert bad.height == 0, (
        f"KuCoin symbol should be UPPER(base)-UPPER(quote) "
        f"({bad.height} mismatches)"
    )

    # KuCoin's borrowRate endpoint typically returns a rate for most
    # active assets, but a few exotic bases may not have one. We don't
    # make hard assertions on the counts – just that the columns
    # exist and have valid types.
    n_with_cross = pairs.filter(pl.col("cross_rate").is_not_null()).height
    n_with_iso = pairs.filter(pl.col("isolated_rate").is_not_null()).height
    assert n_with_cross > 0, "no cross_rate populated"
    assert n_with_iso > 0, "no isolated_rate populated"

    sample = pairs.filter(pl.col("symbol").is_in(WANTED))
    assert sample.height == len(WANTED), (
        f"missing well-known pairs (got {sample.height}/{len(WANTED)})"
    )


async def test_klines(
    client: AsyncClient, kucoin: KuCoin, pairs_window: tuple
) -> None:
    start, end = pairs_window
    klines = pl.concat(
        await asyncio.gather(
            *(kucoin.klines(client, sym, start, end) for sym in WANTED)
        )
    ).sort(["symbol", "open_ts"])

    assert_klines_schema(klines)
    assert_open_ts_in_range(klines, start, end)
    assert_close_ts_matches_open(klines, api_provided=False)
    assert_lowercase(klines, "base", "quote")
    assert_kline_candles_per_symbol(klines, 14, *WANTED)


async def test_klines_empty_range(
    client: AsyncClient, kucoin: KuCoin, pairs_window: tuple
) -> None:
    start, _ = pairs_window
    empty = await kucoin.klines(client, "BTC-USDT", start, start)
    assert empty.height == 0
    assert_klines_schema(empty)


async def test_klines_paged(
    client: AsyncClient, kucoin: KuCoin, paged_window: tuple
) -> None:
    start, end = paged_window
    paged = await kucoin.klines_paged(
        client, "BTC-USDT", start_time=start, end_time=end
    )
    capped = await kucoin.klines(
        client, "BTC-USDT", start_time=start, end_time=end
    )

    assert paged.height > 0
    assert capped.height <= KuCoin.MAX_KLINES, (
        f"klines() returned {capped.height} > MAX_KLINES={KuCoin.MAX_KLINES}"
    )
    assert paged.height >= capped.height
    assert_open_ts_in_range(paged, start, end)
    assert_close_ts_matches_open(paged, api_provided=False)


async def test_pairs_joinable_to_klines(
    client: AsyncClient, kucoin: KuCoin, pairs_window: tuple
) -> None:
    start, end = pairs_window
    pairs = await kucoin.pairs(client, quote_assets={"usdt", "usdc"})
    klines = pl.concat(
        await asyncio.gather(
            *(kucoin.klines(client, sym, start, end) for sym in WANTED)
        )
    )
    assert_pairs_joinable_to_klines(kucoin, pairs, klines)
