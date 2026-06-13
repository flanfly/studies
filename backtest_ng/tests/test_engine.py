"""Tests for ``backtest_ng.engine.Backtest.run``.

Exercises the rebalance loop end-to-end against fake data and a skeleton
``AlphaModel`` so we can assert the new ``history: pl.DataFrame`` argument
is plumbed through to every model call.
"""

import datetime as dt

import polars as pl
import pytest

import backtest_ng as bt
from backtest_ng.interface import (
    AlphaModel,
    ExecutionModel,
    PortfolioModel,
    RiskModel,
    Signal,
    Target,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

START = dt.datetime(2024, 1, 1)
BARS = 5  # 2024-01-01 .. 2024-01-05


def _synthetic_df() -> pl.DataFrame:
    """One-symbol universe with 5 daily bars of constant-volume data."""
    return pl.DataFrame(
        {
            "ts": pl.datetime_range(START, START + dt.timedelta(days=BARS - 1), "1d", eager=True),
            "symbol": ["A"] * BARS,
            "close": [100.0 + i for i in range(BARS)],
            "volume": [1_000.0] * BARS,
        }
    )


# ---------------------------------------------------------------------------
# Skeleton models
# ---------------------------------------------------------------------------


class SkeletonAlpha(AlphaModel):
    """Generates one long signal for ``A`` on the first rebalance, then stops.

    Captures every call's ``history`` argument so the test can assert that
    the engine actually passes it through.
    """

    def __init__(self):
        self.history_calls: list[pl.DataFrame] = []

    def __call__(self, history: pl.DataFrame, u: bt.Universe) -> list[Signal]:
        self.history_calls.append(history.clone())
        # Only fire on the very first rebalance: history is empty then.
        if len(self.history_calls) == 1:
            return [Signal(symbol="A", bullish=True, confidence=1.0)]
        return []


class SkeletonPortfolio(PortfolioModel):
    """One-bar target → 100% long A; otherwise liquidate."""

    def __init__(self):
        self.history_calls: list[pl.DataFrame] = []

    def __call__(
        self,
        history: pl.DataFrame,
        u: bt.Universe,
        signals: list[Signal],
        portfolio: bt.Portfolio,
    ) -> list[Target]:
        self.history_calls.append(history.clone())
        if not signals:
            return []
        return [Target(symbol="A", weight=1.0)]


class SkeletonRisk(RiskModel):
    """Pass-through risk model — just records the call."""

    def __init__(self):
        self.history_calls: list[pl.DataFrame] = []

    def __call__(
        self,
        history: pl.DataFrame,
        u: bt.Universe,
        targets: list[Target],
        portfolio: bt.Portfolio,
    ) -> list[Target]:
        self.history_calls.append(history.clone())
        return targets


class SkeletonExecution(ExecutionModel):
    """Records calls and emits a market order to hit the target weight."""

    def __init__(self):
        self.history_calls: list[pl.DataFrame] = []

    def __call__(
        self,
        history: pl.DataFrame,
        u: bt.Universe,
        targets: list[Target],
        portfolio: bt.Portfolio,
    ) -> list[bt.Order]:
        import ulid

        self.history_calls.append(history.clone())
        klines = u.klines()
        orders: list[bt.Order] = []
        for t in targets:
            kline = klines.get(t.symbol)
            if kline is None:
                continue
            price = kline.price
            # 100% of current equity → 1 share at ~$100 → notional ≈ equity.
            # We don't need it to be exact, just a non-zero market order so
            # the engine exercises execute_orders().
            orders.append(
                bt.Order(
                    type="market",
                    id=str(ulid.new()),
                    cancel=None,
                    symbol=t.symbol,
                    shares=portfolio.cash / price * t.weight,
                    fee=0.0,
                )
            )
        return orders


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_passes_history_to_every_model():
    """Every model call inside ``run()`` receives a non-None ``history`` DataFrame."""
    alpha = SkeletonAlpha()
    portfolio = SkeletonPortfolio()
    risk = SkeletonRisk()
    execution = SkeletonExecution()

    bt_ = bt.Backtest(
        universe=bt.Manual(_synthetic_df()),
        alpha=alpha,
        portfolio=portfolio,
        risk=risk,
        execution=execution,
        period=1,  # 1 day → rebalance on every bar
    )
    bt_.run()

    # The engine runs one rebalance per bar (5 bars) — every model must
    # be invoked exactly once per rebalance.
    assert len(alpha.history_calls) == BARS
    assert len(portfolio.history_calls) == BARS
    assert len(risk.history_calls) == BARS
    assert len(execution.history_calls) == BARS

    # All calls must have received a real (non-None) DataFrame.
    for calls in (alpha, portfolio, risk, execution):
        for h in calls.history_calls:
            assert isinstance(h, pl.DataFrame)
            # Engine writes a fixed schema, including ``ts`` and ``cash``.
            assert "ts" in h.columns
            assert "cash" in h.columns


def test_run_grows_history_bar_by_bar():
    """The history DataFrame passed to a model grows by one row per bar."""
    alpha = SkeletonAlpha()

    bt_ = bt.Backtest(
        universe=bt.Manual(_synthetic_df()),
        alpha=alpha,
        portfolio=SkeletonPortfolio(),
        risk=SkeletonRisk(),
        execution=SkeletonExecution(),
        period=1,
    )
    bt_.run()

    # First call → history is empty (this is the bar *before* the
    # current one gets appended).  Each subsequent call sees one more row.
    heights = [h.height for h in alpha.history_calls]
    assert heights[0] == 0
    for prev, curr in zip(heights, heights[1:]):
        assert curr == prev + 1, f"history did not grow monotonically: {heights}"


def test_run_executes_trades_and_records_history():
    """A real run should produce a non-empty ``history`` and an initial cash drop."""
    bt_ = bt.Backtest(
        universe=bt.Manual(_synthetic_df()),
        alpha=SkeletonAlpha(),
        portfolio=SkeletonPortfolio(),
        risk=SkeletonRisk(),
        execution=SkeletonExecution(),
        period=1,
    )
    initial = 1_000.0
    bt_.run(initial_equity=initial)

    # The engine writes one row per bar.
    assert bt_.history.height == BARS
    assert bt_.history["ts"].to_list() == sorted(bt_.history["ts"].to_list())

    # First bar rebalances: cash drops from initial because the skeleton
    # execution spends cash to buy shares of A.
    cash_series = bt_.history["cash"].to_list()
    assert cash_series[0] < initial

    # The signal/target/order counts reflect the skeleton models' logic.
    rebalance_rows = bt_.history.filter(pl.col("rebalance"))
    assert rebalance_rows["signals"].to_list()[0] == 1
    assert rebalance_rows["targets"].to_list()[0] == 1
    assert rebalance_rows["orders"].to_list()[0] == 1
    # Remaining rebalances have no signal → no targets → no orders.
    assert rebalance_rows["signals"].to_list()[1:] == [0] * (BARS - 1)


def test_live_passes_history_to_every_model():
    """The ``live()`` method must also pass ``self.history`` to every model it calls.

    Note: ``live()`` currently only invokes the alpha, portfolio, and risk
    models — the execution model is not part of the live trading path.
    """
    # ``live()`` calls the chain alpha → portfolio → risk in order.  If
    # alpha returns no signals the chain stops at portfolio, so use an
    # alpha that always signals.
    class AlwaysSignal(AlphaModel):
        def __init__(self):
            self.history_calls: list[pl.DataFrame] = []

        def __call__(self, history: pl.DataFrame, u: bt.Universe) -> list[Signal]:
            self.history_calls.append(history.clone())
            return [Signal(symbol="A", bullish=True, confidence=1.0)]

    alpha = AlwaysSignal()
    portfolio = SkeletonPortfolio()
    risk = SkeletonRisk()
    execution = SkeletonExecution()

    bt_ = bt.Backtest(
        universe=bt.Manual(_synthetic_df()),
        alpha=alpha,
        portfolio=portfolio,
        risk=risk,
        execution=execution,
        period=1,
    )
    # ``live()`` doesn't depend on a prior ``run()``; it just needs the
    # backtest to be configured.
    bt_.live(equity=10_000.0)

    # Exactly one invocation per model in the live() chain.
    assert len(alpha.history_calls) == 1
    assert len(portfolio.history_calls) == 1
    assert len(risk.history_calls) == 1

    for h in (
        alpha.history_calls[0],
        portfolio.history_calls[0],
        risk.history_calls[0],
    ):
        assert isinstance(h, pl.DataFrame)
        assert "ts" in h.columns
        assert "cash" in h.columns
