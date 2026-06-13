from . import ExecutionModel, Order, Target, Universe, Portfolio, Position, Kline
from typing import Dict
import logging as l
import polars as pl
import ulid

eps = 1e-12


class Simple(ExecutionModel):
    def __init__(self, fee_bps: float = 20):
        self.fee_bps = fee_bps

    def __call__(
        self,
        history: pl.DataFrame,
        u: Universe,
        targets: list[Target],
        portfolio: Portfolio,
    ) -> list[Order]:

        if abs(sum([t.weight for t in targets])) > 1 + eps:
            l.warning("target weights sum to >1")

        by_symbol: Dict[str, float] = {}
        for t in targets:
            by_symbol.setdefault(t.symbol, 0.0)
            by_symbol[t.symbol] += t.weight

        orders: list[Order] = []
        klines = u.klines()

        # index positions by symbol, liquidate unwanted positions
        shares: Dict[str, float] = {}
        for pos in portfolio.positions:
            kline = klines.get(pos.symbol)
            if kline is None:
                l.warning(f"no price for {pos.symbol}, skipping")
                continue

            if pos.symbol in by_symbol:
                shares.setdefault(pos.symbol, 0.0)
                shares[pos.symbol] += pos.shares

            else:
                value = pos.shares * kline.price
                fee = self.fee_bps / 10_000.0 * abs(value)
                orders.append(
                    Order(
                        type="market",
                        id=str(ulid.new()),
                        cancel=None,
                        symbol=pos.symbol,
                        shares=-pos.shares,
                        fee=fee,
                    )
                )

        equity = portfolio.cash + sum(
            pos.shares
            * klines.get(pos.symbol, Kline(price=0.0, high=None, low=None)).price
            for pos in portfolio.positions
        )

        # --- stop-loss working orders (must precede market orders) ----------
        # For each target with ``max_risk`` set, the universe must expose
        # ``high``/``low`` channels to be useful — otherwise we'd be checking
        # the same close the market order fills at, and the stop would only
        # ever trigger exactly when the close equals the stop (a degenerate
        # case).  When high/low are missing we skip the stop.
        has_intrabar = all(
            k.high is not None and k.low is not None for k in klines.values()
        )
        if has_intrabar:
            for t in targets:
                if t.max_risk is not None:
                    assert t.max_risk > 0.0 and t.max_risk < 1.0
                else:
                    continue

                kline = klines.get(t.symbol)
                if kline is None:
                    continue
                notional = t.weight * equity
                target_shares = notional / kline.price
                if abs(target_shares) <= eps:
                    continue
                if t.weight > 0:
                    # Long: stop sits below the entry, close by selling.
                    stop_price = kline.price * (1.0 - t.max_risk)
                    working_shares = -target_shares
                else:
                    # Short: stop sits above the entry, close by buying.
                    stop_price = kline.price * (1.0 + t.max_risk)
                    working_shares = -target_shares
                fee = self.fee_bps / 10_000.0 * abs(working_shares * stop_price)
                orders.append(
                    Order(
                        type="stop-loss",
                        id=str(ulid.new()),
                        cancel=None,
                        symbol=t.symbol,
                        shares=working_shares,
                        fee=fee,
                        stop_loss=stop_price,
                    )
                )

        for symbol, weight in by_symbol.items():
            notw = weight * equity

            kline = klines.get(symbol)
            if kline is None:
                l.warning(f"no price for {symbol}, skipping")
                continue

            s = shares.get(symbol)
            if s is not None:
                notw -= s * kline.price

            if abs(notw) <= eps:
                l.warning(f"position on {symbol} too small")
                continue

            orders.append(
                Order(
                    type="market",
                    id=str(ulid.new()),
                    cancel=None,
                    symbol=symbol,
                    shares=notw / kline.price,
                    fee=self.fee_bps / 10_000.0 * abs(notw),
                )
            )

        return orders
