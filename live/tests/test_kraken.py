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
    pairs = await kraken.pairs_with_retry(client, quote_assets={"usdt", "usdc"})
    assert_pairs_schema(pairs)
    assert pairs.height > 0, "no pairs returned"

    # Kraken is cross-margin only: ``isolated_rate`` is always null.
    # ``cross_rate`` may be null for pairs whose base asset is not
    # actually margin-enabled (no online pair with non-empty
    # ``leverage_buy``), even if /Assets reports a non-null rate for
    # the asset -- the per-asset field is informational and doesn't
    # imply the pair is marginable.
    assert pairs.filter(pl.col("isolated_rate").is_not_null()).height == 0, (
        "isolated_rate is set, expected all-null (Kraken is cross-margin only)"
    )
    assert pairs.filter(pl.col("cross_rate") < 0).height == 0, (
        "negative cross_rate (sanity check)"
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
            *(
                kraken.klines_with_retry(client, sym, start, end)
                for sym in WANTED
            )
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
    empty = await kraken.klines_with_retry(client, "XBTUSDT", start, start)
    assert empty.height == 0
    assert_klines_schema(empty)


async def test_klines_paged(
    client: AsyncClient, kraken: Kraken, paged_window: tuple
) -> None:
    start, end = paged_window
    paged = await kraken.klines_paged(
        client, "XBTUSDT", start_time=start, end_time=end
    )
    capped = await kraken.klines_with_retry(
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
    pairs = await kraken.pairs_with_retry(client, quote_assets={"usdt", "usdc"})
    klines = pl.concat(
        await asyncio.gather(
            *(
                kraken.klines_with_retry(client, sym, start, end)
                for sym in WANTED
            )
        )
    )
    assert_pairs_joinable_to_klines(kraken, pairs, klines)


async def test_bgb_usd_pair_and_klines(
    client: AsyncClient, kraken: Kraken, pairs_window: tuple
) -> None:
    """Regression test: symbols like ``BGBUSD`` (3+3) and ``AUSDUSD``
    (4+3) used to break the string-split heuristic in ``klines()``
    because the base/quote can't be recovered by chopping a fixed
    suffix. ``BGBUSD`` has base=``BGB`` and quote=``ZUSD`` on Kraken;
    the adapter now looks the pair up in cached ``/AssetPairs``
    data instead of splitting the symbol.
    """
    start, end = pairs_window
    pairs = await kraken.pairs_with_retry(
        client, quote_assets={"usd", "usdt", "usdc"}
    )

    # BGBUSD must be present in the pairs() output with conventional
    # tickers in base/quote (``bgb``/``usd``), not the raw Kraken
    # codes (``BGB``/``ZUSD``).
    bgb = pairs.filter(pl.col("symbol") == "BGBUSD")
    assert bgb.height == 1, (
        f"BGBUSD not in Kraken pairs() output. "
        f"Got {pairs.height} pairs total."
    )
    assert bgb["base"][0] == "bgb"
    assert bgb["quote"][0] == "usd"

    # klines() must not crash and must return a valid row.
    klines = await kraken.klines_with_retry(client, "BGBUSD", start, end)
    assert_klines_schema(klines)
    if klines.height > 0:
        # If data is available, base/quote must be the conventional
        # tickers, matching ``pairs()``.
        assert klines["base"][0] == "bgb"
        assert klines["quote"][0] == "usd"
        assert_open_ts_in_range(klines, start, end)
        assert_close_ts_matches_open(klines, api_provided=False)


async def test_bgb_usd_cross_rate_is_null(
    client: AsyncClient, kraken: Kraken
) -> None:
    """Regression test: Kraken's ``/Assets`` endpoint reports a
    ``margin_rate`` for ``BGB`` (``0``), but the ``BGBUSD`` pair has
    empty ``leverage_buy``/``leverage_sell`` on ``/AssetPairs`` --
    meaning the pair is not actually margin-tradeable. The previous
    implementation emitted ``cross_rate=0.0`` for BGBUSD, conflating
    "asset has zero margin rate" with "asset has no margin trading".
    The correct value is ``null``.
    """
    pairs = await kraken.pairs_with_retry(
        client, quote_assets={"usd", "usdt", "usdc"}
    )
    bgb = pairs.filter(pl.col("symbol") == "BGBUSD")
    assert bgb.height == 1
    assert bgb["cross_rate"][0] is None, (
        f"BGBUSD cross_rate={bgb['cross_rate'][0]!r}, expected null "
        f"(BGBUSD has empty leverage_buy/leverage_sell on Kraken)"
    )

    # Sanity check: pairs that ARE marginable (XBTUSDT) must still
    # have a non-null ``cross_rate``.
    xbt = pairs.filter(pl.col("symbol") == "XBTUSDT")
    assert xbt.height == 1
    assert xbt["cross_rate"][0] is not None, (
        "XBTUSDT cross_rate is null, expected a non-null annual rate"
    )

    # Cross-check: for the non-null cross_rate pairs, every distinct
    # base must have the same cross_rate (Kraken's rate is per-asset,
    # not per-pair) -- the value comes from /Assets.
    rates = pairs.filter(pl.col("cross_rate").is_not_null())
    by_base = rates.group_by("base").agg(pl.col("cross_rate").n_unique())
    inconsistent = by_base.filter(pl.col("cross_rate") > 1)
    assert inconsistent.height == 0, (
        f"per-asset rate not stable across pairs: "
        f"{inconsistent.select('base', 'cross_rate')}"
    )

