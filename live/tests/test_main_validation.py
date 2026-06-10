"""Tests for the timestamp-validation guard in ``live/main.py``.

The CLI runs ``_validate_timestamps_utc`` on the concatenated klines
and pairs dataframes right before writing them to parquet. The
function must:

  1. Reject dataframes whose timestamp columns are not
     ``Datetime(time_zone='UTC')``.
  2. Reject klines whose ``open_ts`` is not at a daily alignment
     boundary (00:00 UTC for midnight-aligned exchanges, 16:00 UTC
     for HTX).
  3. Pass on the empty dataframe (no work to do).
  4. Pass on a well-formed dataframe.
"""

from __future__ import annotations

import polars as pl
import pytest

from live import empty_klines_df
from main import _validate_timestamps_utc


def _klines_with_typed_ts(rows: list[dict]) -> pl.DataFrame:
    """Build a klines frame and cast the timestamp columns to
    ``Datetime(time_zone='UTC')`` from ISO strings."""
    df = pl.DataFrame(rows)
    return df.with_columns(
        pl.col("open_ts").str.to_datetime(time_unit="us", time_zone="UTC"),
        pl.col("close_ts").str.to_datetime(time_unit="us", time_zone="UTC"),
    )


def test_empty_klines_passes() -> None:
    _validate_timestamps_utc(empty_klines_df(), name="klines")


def test_empty_pairs_passes() -> None:
    _validate_timestamps_utc(pl.DataFrame(), name="pairs")


def test_midnight_klines_pass() -> None:
    """Binance / KuCoin / Kraken open at 00:00 UTC."""
    df = _klines_with_typed_ts(
        [
            {
                "open_ts": "2026-06-08T00:00:00.000000",
                "close_ts": "2026-06-08T23:59:59.999999",
                "symbol": "BTCUSDT",
                "exchange": "binance",
                "base": "btc",
                "quote": "usdt",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "base_volume": 0.0,
                "quote_volume": 0.0,
            },
            {
                "open_ts": "2026-06-08T00:00:00.000000",
                "close_ts": "2026-06-08T23:59:59.999999",
                "symbol": "ETHUSDT",
                "exchange": "kucoin",
                "base": "eth",
                "quote": "usdt",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "base_volume": 0.0,
                "quote_volume": 0.0,
            },
        ]
    )
    # Schema is UTC by default, strings auto-parse to UTC.
    assert df.schema["open_ts"] == pl.Datetime("us", time_zone="UTC")
    _validate_timestamps_utc(df, name="klines")


def test_htx_16utc_klines_pass() -> None:
    """HTX daily candles open at 16:00 UTC, not 00:00."""
    df = _klines_with_typed_ts(
        [
            {
                "open_ts": "2026-06-08T16:00:00.000000",
                "close_ts": "2026-06-09T15:59:59.999999",
                "symbol": "BTCUSDT",
                "exchange": "htx",
                "base": "btc",
                "quote": "usdt",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "base_volume": 0.0,
                "quote_volume": 0.0,
            }
        ]
    )
    _validate_timestamps_utc(df, name="klines")


def test_non_aligned_klines_rejected() -> None:
    """A kline whose open_ts is at 03:00 UTC is not on a daily boundary
    and must be flagged."""
    df = _klines_with_typed_ts(
        [
            {
                "open_ts": "2026-06-08T03:00:00.000000",
                "close_ts": "2026-06-09T02:59:59.999999",
                "symbol": "BTCUSDT",
                "exchange": "binance",
                "base": "btc",
                "quote": "usdt",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "base_volume": 0.0,
                "quote_volume": 0.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="alignment boundary"):
        _validate_timestamps_utc(df, name="klines")


def test_naive_datetime_rejected() -> None:
    """A kline with a naive (no-tz) datetime must be rejected -- the
    schema requires UTC."""
    df = pl.DataFrame(
        [
            {
                "open_ts": "2026-06-08T00:00:00.000000",
                "close_ts": "2026-06-08T23:59:59.999999",
                "symbol": "BTCUSDT",
                "exchange": "binance",
                "base": "btc",
                "quote": "usdt",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "base_volume": 0.0,
                "quote_volume": 0.0,
            }
        ],
    ).with_columns(
        # Force naive datetimes (no time_zone).
        pl.col("open_ts").str.to_datetime(time_unit="us"),
        pl.col("close_ts").str.to_datetime(time_unit="us"),
    )
    with pytest.raises(ValueError, match="time_zone='UTC'"):
        _validate_timestamps_utc(df, name="klines")


def test_non_utc_timezone_rejected() -> None:
    """A kline stamped with a non-UTC timezone (e.g. Europe/Berlin) must
    be rejected. Adapters are required to convert to UTC explicitly."""
    df = pl.DataFrame(
        [
            {
                "open_ts": "2026-06-08T00:00:00.000000",
                "close_ts": "2026-06-08T23:59:59.999999",
                "symbol": "BTCUSDT",
                "exchange": "binance",
                "base": "btc",
                "quote": "usdt",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "base_volume": 0.0,
                "quote_volume": 0.0,
            }
        ],
    ).with_columns(
        pl.col("open_ts").str.to_datetime(time_unit="us", time_zone="Europe/Berlin"),
        pl.col("close_ts").str.to_datetime(time_unit="us", time_zone="Europe/Berlin"),
    )
    with pytest.raises(ValueError, match="time_zone='UTC'"):
        _validate_timestamps_utc(df, name="klines")


def test_pairs_naive_rejected() -> None:
    """Pairs with a naive ``ts`` column must be rejected."""
    df = (
        pl.DataFrame(
            [
                {
                    "ts": "2026-06-08T00:00:00.000000",
                    "symbol": "BTCUSDT",
                    "exchange": "binance",
                    "base": "btc",
                    "quote": "usdt",
                    "cross_rate": 0.01,
                    "isolated_rate": None,
                }
            ],
        )
        # Strip the time_zone declared in PAIRS_SCHEMA.
        .with_columns(pl.col("ts").str.to_datetime(time_unit="us"))
    )
    with pytest.raises(ValueError, match="time_zone='UTC'"):
        _validate_timestamps_utc(df, name="pairs")
