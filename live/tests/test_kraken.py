"""Tests for the Kraken exchange adapter.

No credentials are required: Kraken's public ``/0/public/*`` endpoints
are unauthenticated. Kraken is **cross-margin only** (no isolated
margin) and uses Kraken-specific altnames (``XBT`` for BTC, ``XDG``
for DOGE, etc.) which we normalize to the common ticker in the
``base`` column.
"""

from __future__ import annotations

import asyncio

import polars as pl
import pytest
from httpx import AsyncClient

from live import Kraken
from ._assertions import (
    assert_close_ts_matches_open,
    assert_klines_schema,
    assert_kline_candles_per_symbol,
    assert_lowercase,
    assert_open_ts_in_range,
    assert_pairs_joinable_to_klines,
    assert_pairs_schema,
)


# Common-ticker symbols to look for. We use the Kraken altname form
# (``XBTUSDT`` for BTC) which is what ``/AssetPairs`` returns.
WANTED = ["XBTUSDT", "ETHUSDT", "SOLUSDT"]


async def test_pairs(client: AsyncClient, kraken: Kraken) -> None:
    pairs = await kraken.pairs(client, quote_assets={"usdt", "usdc"})
    assert_pairs_schema(pairs)
    assert pairs.height > 0, "no pairs returned"

    # Kraken is cross-margin only: every active USDT/USDC pair has a
    # ``cross_rate`` and ``isolated_rate`` is always null.
    n_with_cross = pairs.filter(pl.col("cross_rate").is_not_null()).height
    n_with_iso = pairs.filter(pl.col("isolated_rate").is_not_null()).height
    assert n_with_cross == pairs.height, (
        f"{pairs.height - n_with_cross} pairs missing cross_rate"
    )
    assert n_with_iso == 0, (
        f"{n_with_iso} pairs have isolated_rate, expected 0 "
        f"(Kraken is cross-margin only)"
    )

    # ``base`` must be the common ticker (XBT -> BTC, XDG -> DOGE, etc.).
    xbt_usdt = pairs.filter(pl.col("symbol") == "XBTUSDT")
    assert xbt_usdt.height == 1
    assert xbt_usdt["base"][0] == "btc", (
        f"XBTUSDT base is {xbt_usdt['base'][0]!r}, expected 'btc'"
    )

    sample = pairs.filter(pl.col("symbol").is_in(WANTED))
    assert sample.height == len(WANTED), (
        f"missing well-known pairs (got {sample.height}/{len(WANTED)})"
    )


async def test_klines(
    client: AsyncClient, kraken: Kraken, pairs_window: tuple
) -> None:
    start, end = pairs_window
    klines = pl.concat(
        await asyncio.gather(
            *(kraken.klines(client, sym, start, end) for sym in WANTED)
        )
    ).sort(["symbol", "open_ts"])

    assert_klines_schema(klines)
    assert_open_ts_in_range(klines, start, end)
    assert_close_ts_matches_open(klines, api_provided=False)
    assert_lowercase(klines, "base", "quote")
    assert_kline_candles_per_symbol(klines, 14, *WANTED)

    # Cross-check: the ``base`` column from ``klines()`` must match the
    # ``base`` column from ``pairs()`` for the same symbol (in particular,
    # XBTUSDT must show ``base=btc`` in both, not ``xbt`` in klines).
    pairs = await kraken.pairs(client, quote_assets={"usdt", "usdc"})
    klines_base = (
        klines.group_by("symbol")
        .agg(pl.col("base").first())
        .rename({"base": "klines_base"})
    )
    merged = pairs.join(klines_base, on="symbol", how="inner")
    bad = merged.filter(pl.col("base") != pl.col("klines_base"))
    assert bad.height == 0, (
        f"pairs/klines base mismatch on {bad.height} symbols: "
        f"{bad.select('symbol', 'base', 'klines_base')}"
    )


async def test_klines_empty_range(
    client: AsyncClient, kraken: Kraken, pairs_window: tuple
) -> None:
    start, _ = pairs_window
    empty = await kraken.klines(client, "XBTUSDT", start, start)
    assert empty.height == 0
    assert_klines_schema(empty)


async def test_klines_paged(
    client: AsyncClient, kraken: Kraken, paged_window: tuple
) -> None:
    start, end = paged_window
    paged = await kraken.klines_paged(
        client, "XBTUSDT", start_time=start, end_time=end
    )
    capped = await kraken.klines(
        client, "XBTUSDT", start_time=start, end_time=end
    )

    assert paged.height > 0
    assert capped.height <= Kraken.MAX_KLINES, (
        f"klines() returned {capped.height} > MAX_KLINES={Kraken.MAX_KLINES}"
    )
    assert paged.height >= capped.height
    assert_open_ts_in_range(paged, start, end)
    assert_close_ts_matches_open(paged, api_provided=False)


async def test_pairs_joinable_to_klines(
    client: AsyncClient, kraken: Kraken, pairs_window: tuple
) -> None:
    start, end = pairs_window
    pairs = await kraken.pairs(client, quote_assets={"usdt", "usdc"})
    klines = pl.concat(
        await asyncio.gather(
            *(kraken.klines(client, sym, start, end) for sym in WANTED)
        )
    )
    assert_pairs_joinable_to_klines(kraken, pairs, klines)
