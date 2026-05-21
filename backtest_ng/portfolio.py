from abc import ABC, abstractmethod
import polars as pl
import datetime as dt

from typing import Dict

from . import PortfolioModel, Signal, Portfolio, Universe, Target


class EqualWeight(PortfolioModel):
    def __init__(self, timestamp_col="ts", symbol_col="symbol", price_col="close"):
        self.ts_col = timestamp_col
        self.symbol_col = symbol_col
        self.price_col = price_col

    def __call__(
        self,
        u: Universe,
        signals: list[Signal],
        portfolio: Portfolio,
    ) -> list[Target]:
        prices = u.prices()

        if len(signals) == 0:
            return []

        w = 1.0 / len(signals)
        by_symbol: Dict[str, float] = {}
        for s in signals:
            by_symbol.setdefault(s.symbol, 0.0)
            by_symbol[s.symbol] += w * (1 if s.bullish else -1)

        return [Target(symbol=k, weight=v) for k, v in by_symbol.items()]


# class VolumeWeighted(PortfolioModel):
#    def __init__(
#        self,
#        timestamp_col="ts",
#        symbol_col="symbol",
#        price_col="close",
#        volume_col="volume",
#    ):
#        self.ts_col = timestamp_col
#        self.symbol_col = symbol_col
#        self.price_col = price_col
#        self.volume_col = volume_col
#
#    def __call__(
#        self,
#        df: pl.DataFrame,
#        signals: list[Signal],
#        folio: list[Position],
#        equity: float,
#    ) -> list[Order]:
#        today = df[self.ts_col].max()
#        prices = dict(
#            df.filter(pl.col(self.ts_col) == today)
#            .select([self.symbol_col, self.price_col])
#            .iter_rows()
#        )
#        vols = dict(
#            df.filter(pl.col(self.ts_col) == today)
#            .select([self.symbol_col, self.volume_col])
#            .iter_rows()
#        )
#
#        if not signals or equity <= 0:
#            return [Order(pos.symbol, -pos.shares) for pos in folio]
#
#        total_vol = sum(vols.get(signal.symbol, 0) for signal in signals)
#        if total_vol <= 0:
#            return [Order(pos.symbol, -pos.shares) for pos in folio]
#
#        target_shares = {}
#        for signal in signals:
#            price = prices.get(signal.symbol)
#            if price is not None and price > 0:
#                target_shares[signal.symbol] = (
#                    (vols.get(signal.symbol, 0) / total_vol)
#                    * equity
#                    / price
#                    * (1 if signal.bullish else -1)
#                )
#
#        orders = []
#        current_positions = {pos.symbol: pos.shares for pos in folio}
#        all_symbols = set(current_positions.keys()) | set(target_shares.keys())
#
#        for sym in all_symbols:
#            current = current_positions.get(sym, 0.0)
#            target = target_shares.get(sym, 0.0)
#            delta = target - current
#            if abs(delta) > EPSILON:
#                orders.append(Order(sym, delta))
#
#        return orders
#
#
# class SimpleLeverage(PortfolioModel):
#    def __init__(self, leverage: float, inner: PortfolioModel = EqualWeight()):
#        self.inner = inner
#        self.leverage = leverage
#
#    def __call__(
#        self,
#        df: pl.DataFrame,
#        signals: list[Signal],
#        folio: list[Position],
#        equity: float,
#    ) -> list[Order]:
#        return self.inner(df, signals, folio, equity * self.leverage)
#
#
# class TopN(PortfolioModel):
#    def __init__(self, max_long=5, max_short=5, portfolio_model=None):
#        self.max_long = max_long
#        self.max_short = max_short
#        self.portfolio_model = portfolio_model or EqualWeight()
#
#    def __call__(
#        self,
#        df: pl.DataFrame,
#        signals: list[Signal],
#        folio: list[Position],
#        equity: float,
#    ) -> list[Order]:
#        if not signals:
#            return []
#
#        # Sort signals by confidence for long and short separately
#        long_signals = sorted(
#            [s for s in signals if s.bullish], key=lambda s: s.confidence, reverse=True
#        )
#        short_signals = sorted(
#            [s for s in signals if not s.bullish],
#            key=lambda s: s.confidence,
#            reverse=True,
#        )
#
#        top_long = long_signals[: self.max_long]
#        top_short = short_signals[: self.max_short]
#
#        selected_signals = top_long + top_short
#        return self.portfolio_model(df, selected_signals, folio, equity)
