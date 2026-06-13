"""Tests for ``Universe.valid_period`` semantics.

The check guards the assertion in ``Backtest.__init__`` that the
data is granular enough to support a rebalance every ``period``.
A rebalance is "supported" iff the rebalance lands exactly on a
bar boundary, i.e. ``period`` is a multiple of the data's bar
size (the GCD of all consecutive diffs).
"""

import datetime as dt

import polars as pl
import pytest

from backtest_ng import Manual


def _df(rows: list[tuple[str, dt.datetime]]) -> pl.DataFrame:
    """Build a minimal OHLCV-shaped DataFrame from ``(symbol, ts)`` rows.

    Other columns are filled with placeholders so the Manual universe's
    default column-name lookups (``close``, ``volume``) succeed.
    """
    return pl.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "ts": pl.Series([r[1] for r in rows]).cast(pl.Datetime("us")),
            "close": [0.0] * len(rows),
            "volume": [0.0] * len(rows),
        }
    )


# ---------------------------------------------------------------------------
# basic cadence checks
# ---------------------------------------------------------------------------


def test_valid_period_1d_bars_accept_1d():
    """1-day bars support a 1-day rebalance."""
    rows = [
        (s, dt.datetime(2024, 1, d))
        for s in ("A",)
        for d in range(1, 8)
    ]
    u = Manual(_df(rows))
    assert u.valid_period(dt.timedelta(days=1))


def test_valid_period_1d_bars_accept_2d():
    """1-day bars support a 2-day rebalance (rebalance lands on every
    other bar).  This is the user's case from volatility-spread."""
    rows = [
        (s, dt.datetime(2024, 1, d))
        for s in ("A",)
        for d in range(1, 8)
    ]
    u = Manual(_df(rows))
    assert u.valid_period(dt.timedelta(days=2))


def test_valid_period_1d_bars_accept_7d():
    """1-day bars support a 7-day rebalance."""
    rows = [
        (s, dt.datetime(2024, 1, d))
        for s in ("A",)
        for d in range(1, 15)
    ]
    u = Manual(_df(rows))
    assert u.valid_period(dt.timedelta(days=7))


def test_valid_period_1d_bars_reject_12h():
    """1-day bars do NOT support a 12-hour rebalance — a rebalance
    would land mid-bar."""
    rows = [
        (s, dt.datetime(2024, 1, d))
        for s in ("A",)
        for d in range(1, 8)
    ]
    u = Manual(_df(rows))
    assert not u.valid_period(dt.timedelta(hours=12))


def test_valid_period_2d_bars_reject_1d():
    """2-day bars do NOT support a 1-day rebalance — there's no
    bar on alternating days."""
    rows = [
        (s, dt.datetime(2024, 1, d))
        for s in ("A",)
        for d in (1, 3, 5, 7, 9)
    ]
    u = Manual(_df(rows))
    assert not u.valid_period(dt.timedelta(days=1))


def test_valid_period_2d_bars_accept_2d():
    """2-day bars support a 2-day rebalance."""
    rows = [
        (s, dt.datetime(2024, 1, d))
        for s in ("A",)
        for d in (1, 3, 5, 7, 9)
    ]
    u = Manual(_df(rows))
    assert u.valid_period(dt.timedelta(days=2))


# ---------------------------------------------------------------------------
# mixed-cadence data
# ---------------------------------------------------------------------------


def test_valid_period_mixed_1d_and_2d_bars():
    """A symbol with both 1d and 2d gaps has bar size 1d.  Every
    whole-day rebalance is supported, but a sub-day one isn't."""
    rows = [
        ("A", dt.datetime(2024, 1, 1)),
        ("A", dt.datetime(2024, 1, 2)),  # 1d
        ("A", dt.datetime(2024, 1, 3)),
        ("A", dt.datetime(2024, 1, 5)),  # 2d gap
        ("A", dt.datetime(2024, 1, 6)),
        ("A", dt.datetime(2024, 1, 7)),
    ]
    u = Manual(_df(rows))
    assert u.valid_period(dt.timedelta(days=1))
    assert u.valid_period(dt.timedelta(days=2))
    assert u.valid_period(dt.timedelta(days=3))
    assert u.valid_period(dt.timedelta(days=7))
    assert not u.valid_period(dt.timedelta(hours=12))


def test_valid_period_gcd_across_symbols():
    """The bar size is the GCD across all symbols.  If one symbol
    has 2d bars and another has 1d bars, the bar size is 1d."""
    rows = [
        ("A", dt.datetime(2024, 1, 1)),
        ("A", dt.datetime(2024, 1, 2)),
        ("A", dt.datetime(2024, 1, 3)),
        ("B", dt.datetime(2024, 1, 1)),
        ("B", dt.datetime(2024, 1, 3)),  # 2d bar
        ("B", dt.datetime(2024, 1, 5)),
    ]
    u = Manual(_df(rows))
    assert u.valid_period(dt.timedelta(days=1))


def test_valid_period_8d_gap_in_1d_bars():
    """1d bars with a single 8d gap (e.g. a long holiday) — GCD is
    1d, so 1d, 2d, 4d, 7d rebalances all work."""
    rows = [
        ("A", dt.datetime(2024, 1, d))
        for d in (1, 2, 3, 4, 12, 13, 14, 15, 16)
    ]
    u = Manual(_df(rows))
    assert u.valid_period(dt.timedelta(days=1))
    assert u.valid_period(dt.timedelta(days=4))
    assert u.valid_period(dt.timedelta(days=7))


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------


def test_valid_period_rejects_zero():
    """``period=0`` is degenerate — no real cadence to check, so we
    reject it to surface the misconfiguration in the caller's code."""
    rows = [("A", dt.datetime(2024, 1, d)) for d in range(1, 4)]
    u = Manual(_df(rows))
    assert not u.valid_period(dt.timedelta(0))


def test_valid_period_rejects_negative():
    rows = [("A", dt.datetime(2024, 1, d)) for d in range(1, 4)]
    u = Manual(_df(rows))
    assert not u.valid_period(dt.timedelta(seconds=-1))


def test_valid_period_empty_dataframe():
    """Empty DataFrame → no diffs, vacuously valid."""
    empty = pl.DataFrame(schema={"symbol": pl.Utf8, "ts": pl.Datetime("us"),
                                 "close": pl.Float64, "volume": pl.Float64})
    u = Manual(empty)
    assert u.valid_period(dt.timedelta(days=1))


def test_valid_period_single_row_per_symbol():
    """Single-row symbol has no diffs — period is moot, treated as
    valid.  Avoids false negatives on sparse universes."""
    rows = [("A", dt.datetime(2024, 1, 1)), ("B", dt.datetime(2024, 1, 1))]
    u = Manual(_df(rows))
    assert u.valid_period(dt.timedelta(days=1))


def test_valid_period_sub_millisecond_diffs_ignored():
    """A diff of 0.5ms rounds to 0ms in integer milliseconds.  We
    filter out non-positive diffs so they don't poison the GCD."""
    rows = [
        ("A", dt.datetime(2024, 1, 1, 0, 0, 0, 0)),
        ("A", dt.datetime(2024, 1, 1, 0, 0, 0, 500)),  # 0.5ms
        ("A", dt.datetime(2024, 1, 2)),
        ("A", dt.datetime(2024, 1, 3)),
    ]
    u = Manual(_df(rows))
    # 1-day spacing dominates.  A 1d period is valid.
    assert u.valid_period(dt.timedelta(days=1))
    assert u.valid_period(dt.timedelta(days=2))


# ---------------------------------------------------------------------------
# timezone-aware timestamps
# ---------------------------------------------------------------------------


def test_valid_period_timezone_aware_utc():
    """A timezone-aware UTC timestamp column works the same as a naive one."""
    df = pl.DataFrame(
        {
            "symbol": ["A"] * 5,
            "ts": pl.Series(
                [dt.datetime(2024, 1, d, tzinfo=dt.timezone.utc) for d in range(1, 6)]
            ).cast(pl.Datetime("us", time_zone="UTC")),
            "close": [0.0] * 5,
            "volume": [0.0] * 5,
        }
    )
    u = Manual(df)
    assert u.valid_period(dt.timedelta(days=1))
    assert u.valid_period(dt.timedelta(days=2))
    assert not u.valid_period(dt.timedelta(hours=12))


# ---------------------------------------------------------------------------
# the user's case (regression)
# ---------------------------------------------------------------------------


def test_valid_period_user_volatility_spread_repro():
    """The user's volatility-spread notebook uses a 2-day rebalance
    on 1-day bars and was failing on the old (broken)
    ``valid_period`` check.  This regression test pins the fix."""
    rows = [
        (f"sym{i}", dt.datetime(2024, 1, d))
        for i in range(3)
        for d in range(1, 8)
    ]
    u = Manual(_df(rows))
    assert u.valid_period(dt.timedelta(days=2))
