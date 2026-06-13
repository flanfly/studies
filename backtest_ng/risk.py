from . import RiskModel, Target, Universe, Portfolio
import polars as pl


class NoRisk(RiskModel):
    def __call__(
        self,
        history: pl.DataFrame,
        u: Universe,
        targets: list[Target],
        portfolio: Portfolio,
    ) -> list[Target]:
        return targets


class MaxRisk(RiskModel):
    def __init__(self, maxdd: float):
        self.maxdd = maxdd

    def __call__(
        self,
        history: pl.DataFrame,
        u: Universe,
        targets: list[Target],
        portfolio: Portfolio,
    ) -> list[Target]:
        return [
            Target(symbol=t.symbol, weight=t.weight, max_risk=self.maxdd)
            for t in targets
        ]


# class MaxDrawdown(RiskModel):
#    def __init__(
#        self,
#        absolute=0.2,  # longs
#        trailing=0.1,  # longs
#        absolute_short=None,  # equal absolute if none
#        trailing_short=None,  # equal trailing if none
#        timestamp_col="ts",
#        symbol_col="symbol",
#        price_col="close",
#    ):
#        self.absolute = absolute
#        self.trailing = trailing
#        self.absolute_short = absolute_short if absolute_short is not None else absolute
#        self.trailing_short = trailing_short if trailing_short is not None else trailing
#        self.ts_col = timestamp_col
#        self.symbol_col = symbol_col
#        self.price_col = price_col
#
#    def __call__(self, df: pl.DataFrame, folio: list[Position]) -> list[Order]:
#        if not folio:
#            return []
#
#        today = df[self.ts_col].max()
#        day_data = df.filter(pl.col(self.ts_col) == today)
#        prices = dict(day_data.select([self.symbol_col, self.price_col]).iter_rows())
#        # Use low/high columns if available for stop trigger logic
#        lows = (
#            dict(day_data.select([self.symbol_col, "low"]).iter_rows())
#            if "low" in day_data.columns
#            else prices
#        )
#        highs = (
#            dict(day_data.select([self.symbol_col, "high"]).iter_rows())
#            if "high" in day_data.columns
#            else prices
#        )
#
#        orders = []
#        for pos in folio:
#            curr_price = prices.get(pos.symbol)
#            if curr_price is None:
#                continue
#
#            if pos.shares > 0:  # Long
#                low_price = lows.get(pos.symbol, curr_price)
#                if low_price / pos.open - 1 < -self.absolute:
#                    orders.append(Order(pos.symbol, -pos.shares))
#                elif low_price / pos.high - 1 < -self.trailing:
#                    orders.append(Order(pos.symbol, -pos.shares))
#            else:  # Short
#                high_price = highs.get(pos.symbol, curr_price)
#                if 1 - high_price / pos.open < -self.absolute_short:
#                    orders.append(Order(pos.symbol, -pos.shares))
#                elif 1 - high_price / pos.low < -self.trailing_short:
#                    orders.append(Order(pos.symbol, -pos.shares))
#        return orders
