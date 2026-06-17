"""Tests for the ``VolatilityWeighted`` portfolio model.

The model is the framework equivalent of the manual inverse-vol
construction in ``sector-rotation-prod-v1.py``:

    inv_vol[s]  = 1 / sqrt(var[s])
    weight_long [s]  = (inv_vol[s] / sum_long) * leverage
    weight_short[s]  = -(inv_vol[s] / sum_short) * leverage

Per-direction normalisation keeps a vol-heavy short basket from
shrinking the long basket (and vice versa).  Symbols whose volatility
is missing or non-positive are dropped from their side of the basket.
"""

import datetime as dt

import polars as pl
import pytest

import backtest_ng as bt
from backtest_ng.interface import Portfolio, Signal, Target


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


START = dt.datetime(2024, 1, 1)


def _df(rows: list[tuple[str, float]]) -> pl.DataFrame:
    """Two-symbol, one-bar universe with custom var values."""
    return pl.DataFrame(
        [
            (START, "A", 100.0, 100.0, 100.0, 100.0, 1_000.0, var)
            for sym, var in rows
            for var in [rows[0][1] if sym == "A" else rows[1][1]]
        ],
        schema=["ts", "symbol", "open", "high", "low", "close", "volume", "var"],
    )


def _build(rows: list[tuple[str, float]], leverage: float = 2.0) -> list[Target]:
    """Helper: build the universe and run the model with a long+short set.

    ``rows`` is a list of (symbol, var) tuples.  The first row is used
    twice — once as a long, once as a short — so a typical call is
    ``_build([('A', 0.01), ('B', 0.04)])`` which yields two longs
    (A, B) and one short (A again).
    """
    sym_a, var_a = rows[0]
    var_b = rows[1][1]
    df = pl.DataFrame(
        {
            "ts": [START, START],
            "symbol": [sym_a, rows[1][0]],
            "open": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [100.0, 100.0],
            "close": [100.0, 100.0],
            "volume": [1_000.0, 1_000.0],
            "var": [var_a, var_b],
        }
    )
    u = bt.Manual(df)
    signals = [Signal(rows[0][0], True, 1.0), Signal(rows[1][0], True, 1.0), Signal(sym_a, False, 1.0)]
    vw = bt.VolatilityWeighted(volatility_col="var", leverage=leverage)
    return vw(pl.DataFrame(), u, signals, Portfolio(cash=1.0, positions=[], working=[]))


# ---------------------------------------------------------------------------
# core construction
# ---------------------------------------------------------------------------


def test_inverse_vol_weights_low_vol_symbol_higher():
    """A (var=0.01) is half as volatile as B (var=0.04) → A gets 2x B's weight.

    inv_vol_A = 1/sqrt(0.01) = 10, inv_vol_B = 1/sqrt(0.04) = 5.  With
    the v1-style shared tiv, A and B share the budget with the short
    (A again).  tiv = 10 + 5 + 10 = 25.
    A's long weight  = 10/25 * leverage
    B's long weight  =  5/25 * leverage
    A's short weight = -10/25 * leverage
    """
    targets = _build([("A", 0.01), ("B", 0.04)], leverage=3.0)
    by_side = {(t.symbol, t.weight > 0): t.weight for t in targets}
    assert by_side[("A", True)] == pytest.approx(1.2)  # long
    assert by_side[("B", True)] == pytest.approx(0.6)  # long
    assert by_side[("A", False)] == pytest.approx(-1.2)  # short
    # Gross exposure is exactly leverage.
    assert sum(abs(t.weight) for t in targets) == pytest.approx(3.0)


def test_long_and_short_share_one_tiv():
    """A and B are longs, C is the only short.  The v1 reference
    normalises the *combined* basket so the gross is `leverage` (≈2.0),
    not 2.0 per side.  The long:short ratio is determined by the
    count and the relative vol.

    inv_vols: A=10 (var=0.01), B=5 (var=0.04), C=10 (var=0.01).
    tiv = 25, weights = inv_vol/25 * leverage.  A=0.8, B=0.4, C=0.8.
    """
    df = pl.DataFrame(
        {
            "ts": [START] * 3,
            "symbol": ["A", "B", "C"],
            "open": [100.0] * 3,
            "high": [100.0] * 3,
            "low": [100.0] * 3,
            "close": [100.0] * 3,
            "volume": [1_000.0] * 3,
            "var": [0.01, 0.04, 0.01],
        }
    )
    u = bt.Manual(df)
    signals = [
        Signal("A", True, 1.0),
        Signal("B", True, 1.0),
        Signal("C", False, 1.0),
    ]
    vw = bt.VolatilityWeighted(volatility_col="var", leverage=2.0)
    targets = vw(pl.DataFrame(), u, signals, Portfolio(cash=1.0, positions=[], working=[]))

    by_sym = {t.symbol: t.weight for t in targets}
    # All share one tiv, so a 2.0 leverage gives a 2.0-gross basket.
    assert by_sym["A"] == pytest.approx(0.8)
    assert by_sym["B"] == pytest.approx(0.4)
    assert by_sym["C"] == pytest.approx(-0.8)
    assert sum(abs(t.weight) for t in targets) == pytest.approx(2.0)


def test_short_signs_are_negative():
    """Every short-side target must have a negative weight."""
    targets = _build([("A", 0.01), ("B", 0.04)])
    short_targets = [t for t in targets if t.weight < 0]
    # A is the duplicated name used as the short side; there's exactly
    # one negative target even though A appears twice in the basket.
    assert len(short_targets) == 1
    assert short_targets[0].symbol == "A"
    assert short_targets[0].weight < 0


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------


def test_zero_vol_symbol_dropped_from_basket():
    """A signal whose ``var`` is 0 contributes 0 to the basket and is dropped."""
    df = pl.DataFrame(
        {
            "ts": [START, START],
            "symbol": ["A", "B"],
            "open": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [100.0, 100.0],
            "close": [100.0, 100.0],
            "volume": [1_000.0, 1_000.0],
            "var": [0.0, 0.04],
        }
    )
    u = bt.Manual(df)
    signals = [Signal("A", True, 1.0), Signal("B", True, 1.0)]
    vw = bt.VolatilityWeighted(volatility_col="var", leverage=2.0)
    targets = vw(pl.DataFrame(), u, signals, Portfolio(cash=1.0, positions=[], working=[]))

    by_sym = {t.symbol: t.weight for t in targets}
    # A has no vol → dropped; B gets the full long leverage.
    assert by_sym["A"] == 0.0
    assert by_sym["B"] == pytest.approx(2.0)


def test_null_vol_symbol_dropped_from_basket():
    """A signal whose ``var`` is null in the universe is treated the same as 0."""
    df = pl.DataFrame(
        {
            "ts": [START, START],
            "symbol": ["A", "B"],
            "open": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [100.0, 100.0],
            "close": [100.0, 100.0],
            "volume": [1_000.0, 1_000.0],
            "var": [None, 0.04],
        },
        schema={
            "ts": pl.Datetime,
            "symbol": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "var": pl.Float64,
        },
    )
    u = bt.Manual(df)
    signals = [Signal("A", True, 1.0), Signal("B", True, 1.0)]
    vw = bt.VolatilityWeighted(volatility_col="var", leverage=2.0)
    targets = vw(pl.DataFrame(), u, signals, Portfolio(cash=1.0, positions=[], working=[]))

    by_sym = {t.symbol: t.weight for t in targets}
    assert by_sym["A"] == 0.0
    assert by_sym["B"] == pytest.approx(2.0)


def test_zero_vol_on_one_side_leaves_other_side_unchanged():
    """If all the longs have missing vol but the short has a real var,
    the short gets the full leverage on its side and the longs are
    zeroed — the long:short imbalance is the natural consequence of
    the shared tiv dropping to the short's inv_vol only."""
    df = pl.DataFrame(
        {
            "ts": [START, START, START],
            "symbol": ["A", "B", "C"],
            "open": [100.0] * 3,
            "high": [100.0] * 3,
            "low": [100.0] * 3,
            "close": [100.0] * 3,
            "volume": [1_000.0] * 3,
            "var": [None, None, 0.01],
        },
        schema={
            "ts": pl.Datetime,
            "symbol": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "var": pl.Float64,
        },
    )
    u = bt.Manual(df)
    signals = [
        Signal("A", True, 1.0),
        Signal("B", True, 1.0),
        Signal("C", False, 1.0),
    ]
    vw = bt.VolatilityWeighted(volatility_col="var", leverage=2.0)
    targets = vw(pl.DataFrame(), u, signals, Portfolio(cash=1.0, positions=[], working=[]))

    by_sym = {t.symbol: t.weight for t in targets}
    # A and B are dropped (no vol); C carries the full basket weight.
    assert by_sym["A"] == 0.0
    assert by_sym["B"] == 0.0
    assert by_sym["C"] == pytest.approx(-2.0)


def test_fallback_equal_weight_when_column_missing():
    """If the universe has no ``var`` column, fall back to per-basket
    equal weight (i.e. ``leverage / n`` per name, signed by side)."""
    df = pl.DataFrame(
        {
            "ts": [START, START],
            "symbol": ["A", "B"],
            "open": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [100.0, 100.0],
            "close": [100.0, 100.0],
            "volume": [1_000.0, 1_000.0],
            # no ``var`` column on purpose
        }
    )
    u = bt.Manual(df)
    signals = [Signal("A", True, 1.0), Signal("B", True, 1.0)]
    vw = bt.VolatilityWeighted(volatility_col="var", leverage=2.0)
    targets = vw(pl.DataFrame(), u, signals, Portfolio(cash=1.0, positions=[], working=[]))

    by_sym = {t.symbol: t.weight for t in targets}
    # No var anywhere → tiv=0 → equal split of leverage across signals.
    assert by_sym["A"] == pytest.approx(1.0)
    assert by_sym["B"] == pytest.approx(1.0)


def test_empty_signals_returns_empty():
    """No signals → no targets (matches the rest of the model)."""
    df = pl.DataFrame(
        {
            "ts": [START],
            "symbol": ["A"],
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [1_000.0],
            "var": [0.01],
        }
    )
    u = bt.Manual(df)
    vw = bt.VolatilityWeighted()
    assert vw(pl.DataFrame(), u, [], Portfolio(cash=1.0, positions=[], working=[])) == []


def test_leverage_default_is_one():
    """Default ``leverage=1.0`` so a unit-vol portfolio is dollar-neutral
    on each side without further configuration."""
    vw = bt.VolatilityWeighted()
    assert vw.leverage == 1.0
    assert vw.volatility_col == "var"
