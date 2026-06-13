"""Tests for stop-loss working orders.

Covers ``Portfolio.check_working_against_klines`` (trigger evaluation) and
``Portfolio.execute_orders`` filling a triggered stop at its stop price.
Both directions are tested:

* **positive ``shares``** working order = buy-stop, hedging a short
  position.  Triggers when ``kline.high >= stop_loss``.
* **negative ``shares``** working order = sell-stop, hedging a long
  position.  Triggers when ``kline.low <= stop_loss``.
"""

import datetime as dt

from backtest_ng.interface import Kline, Order, Portfolio, Position


NOW = dt.datetime(2024, 1, 15)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pos(symbol: str, shares: float, price: float) -> Position:
    return Position(
        symbol=symbol, shares=shares, entry=NOW, price=price, fee=0.0
    )


def _stop(symbol: str, shares: float, stop_loss: float) -> Order:
    """A sell-stop (``shares < 0``) or buy-stop (``shares > 0``) working order."""
    return Order(
        type="stop-loss",
        id=f"sl-{symbol}",
        cancel=None,
        symbol=symbol,
        shares=shares,
        fee=0.0,
        stop_loss=stop_loss,
    )


# ---------------------------------------------------------------------------
# check_working_against_klines — trigger evaluation
# ---------------------------------------------------------------------------


def test_sell_stop_triggers_when_low_touches_stop():
    """Long position's sell-stop fires when the bar's low reaches the stop."""
    folio = Portfolio(
        cash=0.0,
        positions=[_pos("AAPL", 100.0, 10.0)],
        working=[_stop("AAPL", -100.0, 9.0)],  # sell-stop at 9
    )
    klines = {"AAPL": Kline(price=9.5, high=10.0, low=8.5)}  # low=8.5 <= 9

    triggered, new_folio = folio.check_working_against_klines(klines)
    assert len(triggered) == 1
    assert triggered[0].symbol == "AAPL"
    assert triggered[0].shares == -100.0
    assert triggered[0].stop_loss == 9.0
    assert new_folio.working == []  # removed from working


def test_sell_stop_does_not_trigger_when_low_stays_above_stop():
    """If the bar's low doesn't reach the stop, the order is held."""
    folio = Portfolio(
        cash=0.0,
        positions=[_pos("AAPL", 100.0, 10.0)],
        working=[_stop("AAPL", -100.0, 9.0)],  # sell-stop at 9
    )
    klines = {"AAPL": Kline(price=9.5, high=10.0, low=9.1)}  # low=9.1 > 9

    triggered, new_folio = folio.check_working_against_klines(klines)
    assert triggered == []
    assert len(new_folio.working) == 1
    assert new_folio.working[0].stop_loss == 9.0


def test_buy_stop_triggers_when_high_touches_stop():
    """Short position's buy-stop fires when the bar's high reaches the stop."""
    folio = Portfolio(
        cash=0.0,
        positions=[_pos("AAPL", -100.0, 10.0)],
        working=[_stop("AAPL", 100.0, 11.0)],  # buy-stop at 11
    )
    klines = {"AAPL": Kline(price=10.5, high=11.5, low=10.0)}  # high=11.5 >= 11

    triggered, new_folio = folio.check_working_against_klines(klines)
    assert len(triggered) == 1
    assert triggered[0].symbol == "AAPL"
    assert triggered[0].shares == 100.0
    assert triggered[0].stop_loss == 11.0
    assert new_folio.working == []


def test_buy_stop_does_not_trigger_when_high_stays_below_stop():
    """If the bar's high doesn't reach the stop, the order is held."""
    folio = Portfolio(
        cash=0.0,
        positions=[_pos("AAPL", -100.0, 10.0)],
        working=[_stop("AAPL", 100.0, 11.0)],  # buy-stop at 11
    )
    klines = {"AAPL": Kline(price=10.5, high=10.8, low=10.0)}  # high=10.8 < 11

    triggered, new_folio = folio.check_working_against_klines(klines)
    assert triggered == []
    assert len(new_folio.working) == 1
    assert new_folio.working[0].stop_loss == 11.0


def test_sell_stop_boundary_inclusive():
    """``low == stop`` triggers — the stop fires on touch, not just breach."""
    folio = Portfolio(
        cash=0.0,
        positions=[_pos("AAPL", 100.0, 10.0)],
        working=[_stop("AAPL", -100.0, 9.0)],
    )
    klines = {"AAPL": Kline(price=9.0, high=9.5, low=9.0)}  # low == stop

    triggered, _ = folio.check_working_against_klines(klines)
    assert len(triggered) == 1


def test_buy_stop_boundary_inclusive():
    """``high == stop`` triggers — the stop fires on touch, not just breach."""
    folio = Portfolio(
        cash=0.0,
        positions=[_pos("AAPL", -100.0, 10.0)],
        working=[_stop("AAPL", 100.0, 11.0)],
    )
    klines = {"AAPL": Kline(price=11.0, high=11.0, low=10.5)}  # high == stop

    triggered, _ = folio.check_working_against_klines(klines)
    assert len(triggered) == 1


def test_stop_held_when_intrabar_channels_missing():
    """If high/low are unavailable, the order is held (we can't tell)."""
    folio = Portfolio(
        cash=0.0,
        positions=[_pos("AAPL", 100.0, 10.0)],
        working=[_stop("AAPL", -100.0, 9.0)],
    )
    klines = {"AAPL": Kline(price=9.0, high=None, low=None)}

    triggered, new_folio = folio.check_working_against_klines(klines)
    assert triggered == []
    assert len(new_folio.working) == 1


def test_non_stop_working_orders_passthrough():
    """Non-stop working orders (e.g. type='cancel') are not evaluated and
    remain in ``working``."""
    cancel = Order(type="cancel", id="c1", cancel="abc", symbol="AAPL", shares=0.0, fee=0.0)
    folio = Portfolio(
        cash=0.0,
        positions=[],
        working=[cancel, _stop("AAPL", -100.0, 9.0)],
    )
    klines = {"AAPL": Kline(price=8.0, high=10.0, low=8.0)}  # stop triggers

    triggered, new_folio = folio.check_working_against_klines(klines)
    assert len(triggered) == 1
    assert triggered[0].id == "sl-AAPL"
    assert len(new_folio.working) == 1
    assert new_folio.working[0].id == "c1"  # cancel is preserved


# ---------------------------------------------------------------------------
# execute_orders — stop fills at the stop price, not the kline close
# ---------------------------------------------------------------------------


def test_triggered_sell_stop_fills_at_stop_price_for_long():
    """A triggered sell-stop closes a long at the stop price, not the bar's
    close — verified via the resulting ``cash`` and recorded ``Trade``."""
    folio = Portfolio(
        cash=0.0,
        positions=[_pos("AAPL", 100.0, 10.0)],
        working=[_stop("AAPL", -100.0, 9.0)],
    )
    # Kline closes at 8.5 (well below the stop) — but the fill should be at
    # the stop price 9.0 because the user spec says "assume the stop loss
    # was executed at its stop price".
    klines = {"AAPL": Kline(price=8.5, high=10.0, low=8.5)}
    price_overrides = {"AAPL": 9.0}

    triggered, folio = folio.check_working_against_klines(klines)
    new_folio, closed = folio.execute_orders(
        NOW, klines, triggered, price_overrides=price_overrides
    )

    # Position is fully closed; cash = 100 * 9.0 (proceeds at stop price).
    assert new_folio.positions == []
    assert new_folio.cash == 900.0

    # Trade records the stop price as the exit price.
    assert len(closed) == 1
    t = closed[0]
    assert t.symbol == "AAPL"
    assert t.entry_price == 10.0
    assert t.exit_price == 9.0
    assert t.shares == 100.0


def test_triggered_buy_stop_fills_at_stop_price_for_short():
    """A triggered buy-stop closes a short at the stop price, not the close."""
    folio = Portfolio(
        cash=0.0,
        positions=[_pos("AAPL", -100.0, 10.0)],
        working=[_stop("AAPL", 100.0, 11.0)],
    )
    # Kline closes at 12.0 (well above the stop) — fill should still be at 11.
    klines = {"AAPL": Kline(price=12.0, high=12.0, low=10.0)}
    price_overrides = {"AAPL": 11.0}

    triggered, folio = folio.check_working_against_klines(klines)
    new_folio, closed = folio.execute_orders(
        NOW, klines, triggered, price_overrides=price_overrides
    )

    # Short closed at 11.0: cash spent = 100 * 11.0 = 1100.
    assert new_folio.positions == []
    assert new_folio.cash == -1100.0

    assert len(closed) == 1
    t = closed[0]
    assert t.symbol == "AAPL"
    assert t.entry_price == 10.0
    assert t.exit_price == 11.0
    assert t.shares == -100.0


def test_stop_preserves_existing_cash():
    """The stop's cash effect stacks on top of the prior cash balance."""
    folio = Portfolio(
        cash=500.0,
        positions=[_pos("AAPL", 100.0, 10.0)],
        working=[_stop("AAPL", -100.0, 9.0)],
    )
    klines = {"AAPL": Kline(price=8.5, high=10.0, low=8.5)}

    triggered, folio = folio.check_working_against_klines(klines)
    new_folio, closed = folio.execute_orders(
        NOW, klines, triggered, price_overrides={"AAPL": 9.0}
    )

    assert new_folio.cash == 500.0 + 100 * 9.0
    assert closed[0].exit_price == 9.0


def test_working_passed_through_when_no_stop_triggers():
    """If no stop triggers, the working list is preserved unchanged."""
    folio = Portfolio(
        cash=0.0,
        positions=[_pos("AAPL", 100.0, 10.0)],
        working=[_stop("AAPL", -100.0, 9.0)],
    )
    klines = {"AAPL": Kline(price=9.5, high=10.0, low=9.4)}  # no trigger

    triggered, new_folio = folio.check_working_against_klines(klines)
    assert triggered == []
    assert len(new_folio.working) == 1
    # execute_orders should be a no-op (no orders passed) but should
    # return a portfolio whose ``working`` is preserved.
    final, closed = new_folio.execute_orders(NOW, klines, [])
    assert final.working == new_folio.working
    assert closed == []
