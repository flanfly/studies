"""Tests for the Binance exchange adapter.

These tests run against the live Binance API. They require
``BINANCE_API_KEY`` and ``BINANCE_API_SECRET`` in ``.env``; the test
is skipped if either is missing.

Binance uses the symbol convention ``BTCUSDT`` (concatenated, upper
case) and the klines endpoint returns an explicit ``close_time`` per
candle, which we forward as ``close_ts`` (in microsecond resolution).

Binance borrow rates are per-asset (the same rate applies regardless
of pair, and the same for cross and isolated margin) and require the
API key to have margin permissions. If the key doesn't have that
permission, the rates come back as null and the test prints a note
but does not fail.
"""

from __future__ import annotations

import asyncio

import polars as pl
import pytest
from httpx import AsyncClient

from live import Binance
from ._assertions import (
    assert_close_ts_matches_open,
    assert_klines_schema,
    assert_kline_candles_per_symbol,
    assert_lowercase,
    assert_open_ts_in_range,
    assert_pairs_joinable_to_klines,
    assert_pairs_schema,
)


WANTED = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


async def test_pairs(client: AsyncClient, binance: Binance) -> None:
    pairs = await binance.pairs_with_retry(client, quote_assets={"usdt", "usdc"})
    assert_pairs_schema(pairs)
    assert pairs.height > 0, "no pairs returned"

    # Symbols are ``UPPER(base)UPPER(quote)``.
    bad = pairs.filter(
        (pl.col("base").str.to_uppercase() + pl.col("quote").str.to_uppercase())
        != pl.col("symbol")
    )
    assert bad.height == 0, (
        f"Binance symbol should be UPPER(base)UPPER(quote) "
        f"({bad.height} mismatches)"
    )

    # ``cross_rate`` and ``isolated_rate`` are populated from the same
    # per-asset endpoint, so they should match in count.
    n_with_cross = pairs.filter(pl.col("cross_rate").is_not_null()).height
    n_with_iso = pairs.filter(pl.col("isolated_rate").is_not_null()).height
    assert n_with_cross == n_with_iso, (
        f"cross_rate ({n_with_cross}) and isolated_rate ({n_with_iso}) "
        f"counts differ"
    )

    sample = pairs.filter(pl.col("symbol").is_in(WANTED))
    assert sample.height == len(WANTED), (
        f"missing well-known pairs (got {sample.height}/{len(WANTED)})"
    )


async def test_klines(
    client: AsyncClient, binance: Binance, pairs_window: tuple
) -> None:
    start, end = pairs_window
    klines = pl.concat(
        await asyncio.gather(
            *(
                binance.klines_with_retry(client, sym, start, end)
                for sym in WANTED
            )
        )
    ).sort(["symbol", "open_ts"])

    assert_klines_schema(klines)
    assert_open_ts_in_range(klines, start, end)
    # Binance returns the close timestamp itself (ms resolution), so we
    # only assert that ``close_ts > open_ts`` -- not the exact ``+24h-1us``
    # relation.
    assert_close_ts_matches_open(klines, api_provided=True)
    assert_lowercase(klines, "base", "quote")
    assert_kline_candles_per_symbol(klines, 14, *WANTED)


async def test_klines_empty_range(
    client: AsyncClient, binance: Binance, pairs_window: tuple
) -> None:
    start, _ = pairs_window
    empty = await binance.klines_with_retry(client, "BTCUSDT", start, start)
    assert empty.height == 0
    assert_klines_schema(empty)


async def test_klines_paged(
    client: AsyncClient, binance: Binance, paged_window: tuple
) -> None:
    start, end = paged_window
    paged = await binance.klines_paged(
        client, "BTCUSDT", start_time=start, end_time=end
    )
    capped = await binance.klines_with_retry(
        client, "BTCUSDT", start_time=start, end_time=end
    )

    assert paged.height > 0
    assert capped.height <= Binance.MAX_KLINES, (
        f"klines() returned {capped.height} > MAX_KLINES={Binance.MAX_KLINES}"
    )
    assert paged.height >= capped.height
    assert_open_ts_in_range(paged, start, end)
    assert_close_ts_matches_open(paged, api_provided=True)


async def test_pairs_joinable_to_klines(
    client: AsyncClient, binance: Binance, pairs_window: tuple
) -> None:
    start, end = pairs_window
    pairs = await binance.pairs_with_retry(client, quote_assets={"usdt", "usdc"})
    klines = pl.concat(
        await asyncio.gather(
            *(
                binance.klines_with_retry(client, sym, start, end)
                for sym in WANTED
            )
        )
    )
    assert_pairs_joinable_to_klines(binance, pairs, klines)
