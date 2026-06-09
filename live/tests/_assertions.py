"""Shared assertion helpers for the ``live`` exchange adapter tests.

These functions encode the contract of ``Exchange.pairs()`` and
``Exchange.klines()``:

  * ``pairs()`` returns active spot pairs with cross/isolated margin
    borrow rates; the ``base``/``quote`` columns are lower-cased
    common tickers.
  * ``klines()`` returns at most ``MAX_KLINES`` daily ohlcv candles in
    the half-open range ``[start, end)``; ``start`` is inclusive,
    ``end`` is exclusive; ``close_ts`` is the inclusive last instant
    of the daily candle.
  * The per-exchange proprietary ``symbol`` is opaque to consumers:
    pass it straight from ``pairs()`` to ``klines()`` and join on
    ``(symbol, base, quote)``.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from live import (
    Exchange,
    KLINES_SCHEMA,
    PAIRS_SCHEMA,
    empty_klines_df,
    empty_pairs_df,
    validate_klines_df,
    validate_pairs_df,
)


# ``KLINES_SCHEMA`` / ``PAIRS_SCHEMA`` are dicts of ``{col: dtype}``;
# pulling just the column names is enough for the schema-shape checks
# below.
KLINES_COLUMNS = set(KLINES_SCHEMA)
PAIRS_COLUMNS = set(PAIRS_SCHEMA)


def assert_klines_schema(df: pl.DataFrame) -> None:
    # ``validate_klines_df`` raises ``ValueError`` on schema mismatch;
    # surface it through pytest's assertion machinery instead.
    try:
        validate_klines_df(df)
    except ValueError as e:
        pytest.fail(str(e))


def assert_pairs_schema(df: pl.DataFrame) -> None:
    try:
        validate_pairs_df(df)
    except ValueError as e:
        pytest.fail(str(e))


def assert_open_ts_in_range(
    df: pl.DataFrame, start: dt.datetime, end: dt.datetime
) -> None:
    """Every ``open_ts`` must satisfy ``start <= open_ts < end``."""
    if df.height == 0:
        return
    min_ts, max_ts = df["open_ts"].min(), df["open_ts"].max()
    assert min_ts >= start, f"open_ts {min_ts} is before start {start}"
    assert max_ts < end, f"open_ts {max_ts} is not before end {end}"


def assert_close_ts_matches_open(
    df: pl.DataFrame, *, api_provided: bool
) -> None:
    """``close_ts`` is the inclusive last instant of the daily candle.

    ``api_provided=True`` is the Binance convention (close_ts from the
    API, in millisecond resolution); we only need to assert that
    ``close_ts > open_ts``.

    ``api_provided=False`` is the µs-resolution convention (HTX, KuCoin,
    Kraken); we assert ``close_ts == open_ts + 24h - 1us``.
    """
    if df.height == 0:
        return
    day = dt.timedelta(days=1)
    us = dt.timedelta(microseconds=1)
    if api_provided:
        bad = df.filter(pl.col("close_ts") <= pl.col("open_ts"))
        assert bad.height == 0, f"close_ts <= open_ts on {bad.height} rows"
    else:
        bad = df.filter(pl.col("close_ts") != pl.col("open_ts") + day - us)
        assert bad.height == 0, f"close_ts mismatch on {bad.height} rows"


def assert_lowercase(df: pl.DataFrame, *cols: str) -> None:
    for col in cols:
        bad = df.filter(pl.col(col) != pl.col(col).str.to_lowercase())
        assert bad.height == 0, f"{col} not lowercase on {bad.height} rows"


def assert_kline_candles_per_symbol(
    df: pl.DataFrame, expected: int, *symbols: str
) -> None:
    """Each symbol in ``symbols`` must have exactly ``expected`` candles."""
    for sym in symbols:
        n = df.filter(pl.col("symbol") == sym).height
        assert n == expected, f"{sym} returned {n} klines, expected {expected}"


def assert_pairs_joinable_to_klines(
    exchange: Exchange,
    pairs: pl.DataFrame,
    klines: pl.DataFrame,
    *,
    base: str = "btc",
    quote: str = "usdt",
) -> None:
    """Round-trip a btc/usdt row from ``pairs()`` through ``klines()``
    and assert that:

      * ``pairs()`` returns exactly one row for ``base/quote``.
      * The opaque ``symbol`` it returns appears in the klines column.
      * ``pairs ⨝ klines`` on ``(symbol, base, quote)`` produces one
        row per kline.
      * The ``base``/``quote`` columns are the common tickers, usable
        for cross-exchange joins.
    """
    pair = pairs.filter((pl.col("base") == base) & (pl.col("quote") == quote))
    assert pair.height == 1, (
        f"expected 1 {base}/{quote} pair, got {pair.height}"
    )
    pair_symbol = pair["symbol"][0]

    sym_klines = klines.filter(pl.col("symbol") == pair_symbol)
    assert sym_klines.height > 0, f"no klines for {pair_symbol!r}"
    assert sym_klines["symbol"].unique().to_list() == [pair_symbol], (
        f"klines symbol {sym_klines['symbol'].unique().to_list()!r} != "
        f"pairs symbol {pair_symbol!r}"
    )

    joined = pair.join(sym_klines, on=["symbol", "base", "quote"], how="inner")
    assert joined.height == sym_klines.height, (
        f"pairs⨝klines join produced {joined.height} rows, "
        f"expected {sym_klines.height}"
    )

    assert pair["base"][0] == base
    assert pair["quote"][0] == quote
    assert sym_klines["base"][0] == base
    assert sym_klines["quote"][0] == quote
