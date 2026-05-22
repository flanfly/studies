from . import ExecutionModel, Order, Target, Universe, Portfolio, Position
from typing import Dict
import logging as l
import ulid

eps = 1e-12


class Simple(ExecutionModel):
    def __init__(self, fee_bps: float = 20):
        self.fee_bps = fee_bps

    def __call__(
        self, u: Universe, targets: list[Target], portfolio: Portfolio
    ) -> list[Order]:

        if abs(sum([t.weight for t in targets])) > 1 + eps:
            l.warning("target weights sum to >1")

        by_symbol: Dict[str, float] = {}
        for t in targets:
            by_symbol.setdefault(t.symbol, 0.0)
            by_symbol[t.symbol] += t.weight

        orders: list[Order] = []
        prices = u.prices()

        # index positions by symbol, liquidate unwanted positions
        shares: Dict[str, float] = {}
        for pos in portfolio.positions:
            price = prices.get(pos.symbol)
            if price is None:
                l.warning(f"no price for {pos.symbol}, skipping")
                continue

            if pos.symbol in by_symbol:
                shares.setdefault(pos.symbol, 0.0)
                shares[pos.symbol] += pos.shares

            else:
                value = pos.shares * price
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
            pos.shares * prices.get(pos.symbol, 0.0) for pos in portfolio.positions
        )

        for symbol, weight in by_symbol.items():
            notw = weight * equity

            price = prices.get(symbol)
            if price is None:
                l.warning(f"no price for {symbol}, skipping")
                continue

            s = shares.get(symbol)
            if s is not None:
                notw -= s * price

            if abs(notw) <= eps:
                l.warning(f"position on {symbol} too small")
                continue

            orders.append(
                Order(
                    type="market",
                    id=str(ulid.new()),
                    cancel=None,
                    symbol=symbol,
                    shares=notw / price,
                    fee=self.fee_bps / 10_000.0 * abs(notw),
                )
            )

        return orders
