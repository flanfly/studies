"""Tests for Portfolio.execute_orders."""

import datetime as dt

from backtest_ng.interface import Kline, Order, Portfolio, Position, Trade

NOW = dt.datetime(2024, 1, 15)
# ``klines`` carries both the close (``price``) and intrabar ``high``/``low``.
# Stop-loss tests use the high/low channels; FIFO fill tests only need ``price``.
PRICES = {"AAPL": Kline(price=10.0, high=10.0, low=10.0)}
EPS = 1e-9


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pos(symbol: str, shares: float, entry: dt.datetime, price: float, fee: float) -> Position:
    return Position(symbol=symbol, shares=shares, entry=entry, price=price, fee=fee)


def _order(symbol: str, shares: float, fee: float) -> Order:
    return Order(type="market", id="x", cancel=None, symbol=symbol, shares=shares, fee=fee)


def _trade(
    symbol: str,
    entry: dt.datetime,
    exit: dt.datetime,
    entry_price: float,
    exit_price: float,
    entry_fee: float,
    exit_fee: float,
    shares: float,
) -> Trade:
    return Trade(
        symbol=symbol,
        entry_ts=entry,
        exit_ts=exit,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_fee=entry_fee,
        exit_fee=exit_fee,
        shares=shares,
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_buy_into_empty_portfolio():
    """Single buy order in an empty portfolio."""
    folio = Portfolio(cash=10_000.0, positions=[], working=[])
    new_folio, closed = folio.execute_orders(NOW, PRICES, [_order("AAPL", 100.0, 1.0)])

    assert new_folio.cash == 10_000.0 - 100 * 10.0 - 1.0  # 9899.0
    assert len(new_folio.positions) == 1
    p = new_folio.positions[0]
    assert p.symbol == "AAPL"
    assert p.shares == 100.0
    assert p.entry == NOW
    assert p.price == 10.0
    assert p.fee == 1.0
    assert closed == []


def test_sell_into_empty_portfolio():
    """Single sell order in an empty portfolio → opens a short position."""
    folio = Portfolio(cash=10_000.0, positions=[], working=[])
    new_folio, closed = folio.execute_orders(NOW, PRICES, [_order("AAPL", -100.0, 1.0)])

    # proceeds = 100 * 10 = 1000, less 1.0 fee → cash increases by 999
    assert new_folio.cash == 10_000.0 - (-100) * 10.0 - 1.0  # 10999.0
    assert len(new_folio.positions) == 1
    p = new_folio.positions[0]
    assert p.symbol == "AAPL"
    assert p.shares == -100.0
    assert p.entry == NOW
    assert p.price == 10.0
    assert p.fee == 1.0
    assert closed == []


def test_reduce_long_position():
    """Long 100, sell 50 → position reduced to 50, no trade closed."""
    ts0 = dt.datetime(2024, 1, 1)
    folio = Portfolio(
        cash=10_000.0,
        positions=[_pos("AAPL", 100.0, ts0, 10.0, 0.5)],
        working=[],
    )
    new_folio, closed = folio.execute_orders(NOW, PRICES, [_order("AAPL", -50.0, 0.5)])

    # proceeds = 50 * 10 = 500 less fee 0.5 → +499.5
    assert new_folio.cash == 10_000.0 + 499.5
    assert len(new_folio.positions) == 1
    p = new_folio.positions[0]
    assert p.symbol == "AAPL"
    assert p.shares == 50.0
    assert p.entry == ts0  # original entry preserved
    assert p.price == 10.0  # original price preserved
    assert p.fee == 0.5  # original entry fee preserved
    assert closed == []


def test_reduce_short_position():
    """Short 100, buy 50 → position reduced to -50, no trade closed."""
    ts0 = dt.datetime(2024, 1, 1)
    folio = Portfolio(
        cash=10_000.0,
        positions=[_pos("AAPL", -100.0, ts0, 10.0, 0.5)],
        working=[],
    )
    new_folio, closed = folio.execute_orders(NOW, PRICES, [_order("AAPL", 50.0, 0.5)])

    # buying back 50 shares costs 500 + 0.5 fee
    assert new_folio.cash == 10_000.0 - 500.5
    assert len(new_folio.positions) == 1
    p = new_folio.positions[0]
    assert p.symbol == "AAPL"
    assert p.shares == -50.0
    assert p.entry == ts0
    assert p.price == 10.0
    assert p.fee == 0.5
    assert closed == []


def test_long_reversed_to_short():
    """Long 100, sell 150 → long closed (trade recorded), short 50 opened."""
    ts0 = dt.datetime(2024, 1, 1)
    folio = Portfolio(
        cash=10_000.0,
        positions=[_pos("AAPL", 100.0, ts0, 10.0, 0.5)],
        working=[],
    )
    new_folio, closed = folio.execute_orders(NOW, PRICES, [_order("AAPL", -150.0, 1.5)])

    # batch for closing 100: fee 1.5 * 100/150 = 1.0
    # proceed 100 * 10 - 1.0 = +999
    # residual for opening 50 short: fee 1.5 * 50/150 = 0.5
    # short opening: receive 50 * 10 - 0.5 = +499.5
    expected_cash = 10_000.0 + 999.0 + 499.5

    assert new_folio.cash == expected_cash
    assert len(new_folio.positions) == 1
    p = new_folio.positions[0]
    assert p.symbol == "AAPL"
    assert p.shares == -50.0
    assert p.entry == NOW
    assert p.price == 10.0
    assert p.fee == 0.5

    assert len(closed) == 1
    t = closed[0]
    assert t.symbol == "AAPL"
    assert t.entry_ts == ts0
    assert t.exit_ts == NOW
    assert t.entry_price == 10.0
    assert t.exit_price == 10.0
    assert t.entry_fee == 0.5
    assert t.exit_fee == 1.0
    assert t.shares == 100.0


def test_short_reversed_to_long():
    """Short 100, buy 150 → short closed (trade recorded), long 50 opened."""
    ts0 = dt.datetime(2024, 1, 1)
    folio = Portfolio(
        cash=10_000.0,
        positions=[_pos("AAPL", -100.0, ts0, 10.0, 0.5)],
        working=[],
    )
    new_folio, closed = folio.execute_orders(NOW, PRICES, [_order("AAPL", 150.0, 1.5)])

    # closing 100 short: pay 100 * 10 + 1.0 = 1001
    # opening 50 long: pay 50 * 10 + 0.5 = 500.5
    expected_cash = 10_000.0 - 1001.0 - 500.5

    assert new_folio.cash == expected_cash
    assert len(new_folio.positions) == 1
    p = new_folio.positions[0]
    assert p.symbol == "AAPL"
    assert p.shares == 50.0
    assert p.entry == NOW
    assert p.price == 10.0
    assert p.fee == 0.5

    assert len(closed) == 1
    t = closed[0]
    assert t.symbol == "AAPL"
    assert t.entry_ts == ts0
    assert t.exit_ts == NOW
    assert t.entry_price == 10.0
    assert t.exit_price == 10.0
    assert t.entry_fee == 0.5
    assert t.exit_fee == 1.0
    assert t.shares == -100.0


def test_two_longs_close_one_reduce_second():
    """Two long positions: sell 120 closes 100-lot, reduces 50-lot to 30."""
    ts1 = dt.datetime(2024, 1, 1)
    ts2 = dt.datetime(2024, 1, 8)
    folio = Portfolio(
        cash=10_000.0,
        positions=[
            _pos("AAPL", 100.0, ts1, 10.0, 0.5),
            _pos("AAPL", 50.0, ts2, 11.0, 0.25),
        ],
        working=[],
    )
    new_folio, closed = folio.execute_orders(NOW, PRICES, [_order("AAPL", -120.0, 1.2)])

    # close 100-lot: fee 1.2 * 100/120 = 1.0,  proceed +999
    # reduce 50-lot: fee 1.2 *  20/120 = 0.2,  proceed +199.8
    expected_cash = 10_000.0 + 999.0 + 199.8

    assert new_folio.cash == expected_cash
    assert len(new_folio.positions) == 1
    p = new_folio.positions[0]
    assert p.symbol == "AAPL"
    assert p.shares == 30.0
    assert p.entry == ts2  # second position preserved
    assert p.price == 11.0
    assert p.fee == 0.25

    assert len(closed) == 1
    t = closed[0]
    assert t.symbol == "AAPL"
    assert t.entry_ts == ts1
    assert t.exit_ts == NOW
    assert t.entry_price == 10.0
    assert t.exit_price == 10.0
    assert t.entry_fee == 0.5
    assert t.exit_fee == 1.0
    assert t.shares == 100.0


def test_two_shorts_close_one_reduce_second():
    """Two short positions: buy 120 closes -100-lot, reduces -50-lot to -30."""
    ts1 = dt.datetime(2024, 1, 1)
    ts2 = dt.datetime(2024, 1, 8)
    folio = Portfolio(
        cash=10_000.0,
        positions=[
            _pos("AAPL", -100.0, ts1, 10.0, 0.5),
            _pos("AAPL", -50.0, ts2, 11.0, 0.25),
        ],
        working=[],
    )
    new_folio, closed = folio.execute_orders(NOW, PRICES, [_order("AAPL", 120.0, 1.2)])

    # close -100 lot: fee 1.0, cost 100 * 10 + 1.0 = 1001
    # reduce -50 lot: fee 0.2, cost  20 * 10 + 0.2 =  200.2
    expected_cash = 10_000.0 - 1001.0 - 200.2

    assert new_folio.cash == expected_cash
    assert len(new_folio.positions) == 1
    p = new_folio.positions[0]
    assert p.symbol == "AAPL"
    assert p.shares == -30.0
    assert p.entry == ts2
    assert p.price == 11.0
    assert p.fee == 0.25

    assert len(closed) == 1
    t = closed[0]
    assert t.symbol == "AAPL"
    assert t.entry_ts == ts1
    assert t.exit_ts == NOW
    assert t.entry_price == 10.0
    assert t.exit_price == 10.0
    assert t.entry_fee == 0.5
    assert t.exit_fee == 1.0
    assert t.shares == -100.0
