"""Integration tests for the stop-loss flow through the engine.

* ``Simple`` emits a ``stop-loss`` working order for every target with
  ``max_risk`` set, *only* when the universe exposes ``high``/``low``
  channels.
* The engine evaluates working stops **before** market orders on every
  bar; a triggered stop is filled at its stop price and the resulting
  trade is recorded.
"""

import datetime as dt

import polars as pl
import pytest

import backtest_ng as bt
from backtest_ng.interface import (
    AlphaModel,
    PortfolioModel,
    RiskModel,
    Signal,
    Target,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


START = dt.datetime(2024, 1, 1)


def _bars(closes: list[float]) -> pl.DataFrame:
    """Build a 1-symbol, ``len(closes)``-bar DataFrame with high/low tracks
    0.5 above/below close."""
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    n = len(closes)
    return pl.DataFrame(
        {
            "ts": pl.datetime_range(START, START + dt.timedelta(days=n - 1), "1d", eager=True),
            "symbol": ["A"] * n,
            "close": closes,
            "high": highs,
            "low": lows,
            "volume": [1_000.0] * n,
        }
    )


# ---------------------------------------------------------------------------
# ``Simple`` execution model
# ---------------------------------------------------------------------------


class AlwaysLong(AlphaModel):
    def __call__(self, history, u):
        return [Signal(symbol="A", bullish=True, confidence=1.0)]


class LongWithRisk(PortfolioModel):
    """``max_risk=0.1`` on a long target ⇒ stop at 0.9 × price."""

    def __call__(self, history, u, signals, portfolio):
        if not signals:
            return []
        return [Target(symbol="A", weight=1.0, max_risk=0.1)]


class ShortWithRisk(PortfolioModel):
    """``max_risk=0.1`` on a short target ⇒ stop at 1.1 × price."""

    def __call__(self, history, u, signals, portfolio):
        if not signals:
            return []
        return [Target(symbol="A", weight=-1.0, max_risk=0.1)]


class NoRisk(PortfolioModel):
    """Same as LongWithRisk but ``max_risk=None`` — no stop should be emitted."""

    def __call__(self, history, u, signals, portfolio):
        if not signals:
            return []
        return [Target(symbol="A", weight=1.0)]


def test_simple_emits_stop_loss_for_max_risk_target():
    """When the universe has high/low and ``max_risk`` is set, ``Simple``
    must emit a ``stop-loss`` working order before the market order."""
    u = bt.Manual(_bars([100.0, 100.0]), high_col="high", low_col="low")
    folio = bt.Portfolio(cash=10_000.0, positions=[], working=[])
    orders = bt.Simple()(
        history=pl.DataFrame(), u=u, targets=[Target("A", 1.0, max_risk=0.1)], portfolio=folio
    )

    # One market order + one stop-loss working order.
    types = [o.type for o in orders]
    assert types.count("market") == 1
    assert types.count("stop-loss") == 1

    stop = next(o for o in orders if o.type == "stop-loss")
    # Stop is for selling the long we just bought — shares is negative
    # of the target notional / price.
    assert stop.shares < 0
    # Trigger price = current price * (1 - max_risk) = 100 * 0.9 = 90.
    assert stop.stop_loss == pytest.approx(90.0)
    assert stop.symbol == "A"


def test_simple_emits_stop_loss_for_short_target():
    """A short target with ``max_risk`` produces a buy-stop (positive shares)
    that triggers when the bar's high touches ``price * (1 + max_risk)``."""
    u = bt.Manual(_bars([100.0, 100.0]), high_col="high", low_col="low")
    folio = bt.Portfolio(cash=10_000.0, positions=[], working=[])
    orders = bt.Simple()(
        history=pl.DataFrame(), u=u, targets=[Target("A", -1.0, max_risk=0.1)], portfolio=folio
    )

    stop = next(o for o in orders if o.type == "stop-loss")
    # Stop is for buying back the short — shares is positive.
    assert stop.shares > 0
    # Trigger price = 100 * 1.1 = 110.
    assert stop.stop_loss == pytest.approx(110.0)


def test_simple_skips_stop_when_no_high_low():
    """No high/low → no stop order, even with ``max_risk`` set."""
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(START, START + dt.timedelta(days=1), "1d", eager=True),
            "symbol": ["A", "A"],
            "close": [100.0, 100.0],
            "volume": [1_000.0, 1_000.0],
        }
    )
    u = bt.Manual(df)  # no high_col / low_col
    folio = bt.Portfolio(cash=10_000.0, positions=[], working=[])
    orders = bt.Simple()(
        history=pl.DataFrame(), u=u, targets=[Target("A", 1.0, max_risk=0.1)], portfolio=folio
    )
    assert all(o.type != "stop-loss" for o in orders)


def test_simple_skips_stop_when_max_risk_is_none():
    """``max_risk=None`` (the default) means no stop."""
    u = bt.Manual(_bars([100.0, 100.0]), high_col="high", low_col="low")
    folio = bt.Portfolio(cash=10_000.0, positions=[], working=[])
    orders = bt.Simple()(
        history=pl.DataFrame(), u=u, targets=[Target("A", 1.0)], portfolio=folio
    )
    assert all(o.type != "stop-loss" for o in orders)


def test_simple_skips_stop_when_max_risk_is_zero_or_negative():
    """``max_risk=0`` (or negative) is a degenerate request — skip the stop."""
    u = bt.Manual(_bars([100.0, 100.0]), high_col="high", low_col="low")
    folio = bt.Portfolio(cash=10_000.0, positions=[], working=[])
    orders = bt.Simple()(
        history=pl.DataFrame(), u=u, targets=[Target("A", 1.0, max_risk=0.0)], portfolio=folio
    )
    assert all(o.type != "stop-loss" for o in orders)


# ---------------------------------------------------------------------------
# engine: stop-loss fires before market orders, fills at stop price
# ---------------------------------------------------------------------------


def test_engine_long_stop_triggers_and_fills_at_stop_price():
    """End-to-end: a long with ``max_risk=0.1`` enters at 100; on day 2 the
    low touches 90, so the sell-stop fills at 90 and the trade is recorded.

    Bar setup (close, low):
        day 0: 100 (entry)        → stop placed at 90
        day 1:  95 (low = 94.5)    → stop still alive
        day 2:  85 (low = 84.5)    → stop fires at 90, closes long
    """
    # 4 bars to give the engine a bar to re-buy after the stop fires
    # (so we can also assert no extra phantom trades are generated).
    df = _bars([100.0, 95.0, 85.0, 100.0])
    u = bt.Manual(df, high_col="high", low_col="low")
    bt_ = bt.Backtest(
        universe=u, alpha=AlwaysLong(), portfolio=LongWithRisk(), period=1
    )
    bt_.run(initial_equity=1_000.0)

    # Exactly one trade: entry at 100, exit at 90 (the stop price).
    assert bt_.trades.height == 1
    t = bt_.trades.row(0, named=True)
    assert t["symbol"] == "A"
    assert t["price"] == pytest.approx(100.0)  # entry price

    # Compute the expected exit price from pnl/ret algebra to verify the
    # stop fired at 90, not at the bar's close (85) or the low (84.5).
    # ret = (exit - entry) / entry  ⇒  exit = entry * (1 + ret) for longs.
    expected_exit = 100.0 * (1.0 + t["ret"])
    assert expected_exit == pytest.approx(90.0, rel=1e-6), (
        f"expected exit at stop price 90, got {expected_exit:.4f}"
    )


def test_engine_short_stop_triggers_and_fills_at_stop_price():
    """End-to-end: short with ``max_risk=0.1`` enters at 100; the stop
    sits at 110 (price * 1.1) and triggers when the bar's high touches it.

    Bar setup (close, high):
        day 0: 100 (entry)        → stop placed at 110
        day 1: 105 (high = 105.5) → stop still alive
        day 2: 115 (high = 115.5) → stop fires at 110
    """
    df = _bars([100.0, 105.0, 115.0, 100.0])
    u = bt.Manual(df, high_col="high", low_col="low")
    bt_ = bt.Backtest(
        universe=u, alpha=AlwaysLong(), portfolio=ShortWithRisk(), period=1
    )
    bt_.run(initial_equity=1_000.0)

    assert bt_.trades.height == 1
    t = bt_.trades.row(0, named=True)
    assert t["symbol"] == "A"
    # Shares are negative (short closed).
    # Note: the trades DataFrame is lossy on the schema side — exit price
    # is not a column, but we can derive it from ret for a short:
    # ret = (entry - exit) / entry  ⇒  exit = entry * (1 - ret).
    expected_exit = 100.0 * (1.0 - t["ret"])
    assert expected_exit == pytest.approx(110.0, rel=1e-6), (
        f"expected short stop fill at 110, got {expected_exit:.4f}"
    )


def test_engine_no_stop_trigger_when_price_stays_above_long_stop():
    """If the bar's low never reaches the stop, the working order lingers
    and no trade is recorded."""
    df = _bars([100.0, 105.0, 110.0, 115.0])  # all rising
    u = bt.Manual(df, high_col="high", low_col="low")
    bt_ = bt.Backtest(
        universe=u, alpha=AlwaysLong(), portfolio=LongWithRisk(), period=1
    )
    bt_.run(initial_equity=1_000.0)
    # No closes — stop never fired.
    assert bt_.trades.height == 0


def test_engine_stops_run_before_market_orders_same_bar():
    """On the bar where the stop fires, the engine must close via the stop
    *first*, then rebuy via the market order — so the recorded trade's
    exit price is the stop price (90), not the close (85).

    We assert this by checking the ret: if the stop ran after the market
    order, the recorded exit would be 85 → ret = -0.15, not -0.10.
    """
    df = _bars([100.0, 95.0, 85.0, 100.0])
    u = bt.Manual(df, high_col="high", low_col="low")
    bt_ = bt.Backtest(
        universe=u, alpha=AlwaysLong(), portfolio=LongWithRisk(), period=1
    )
    bt_.run(initial_equity=1_000.0)

    t = bt_.trades.row(0, named=True)
    # The trade's ret must be ~ -10% (stop at 90), not -15% (close 85).
    # Allow a small tolerance for fees eating into ret.
    assert t["ret"] > -0.13, (
        f"ret = {t['ret']:.4f} suggests the stop ran AFTER the market "
        f"order (or the stop was filled at the close, not the stop price)"
    )
