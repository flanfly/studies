"""Public interfaces.

AlphaModels --[ Signals ]-> PortfolioModel --[ Targets ]-> RiskModel(s) --[ adj. Targets ]-> ExecutionModel --[ Orders ]-> Backtest
"""

from abc import ABC, abstractmethod
import polars as pl
import datetime as dt

import numpy as np

from dataclasses import dataclass

from copy import copy
from typing import Tuple, Dict

import logging as l

eps = 1e-12


class Universe(ABC):
    @abstractmethod
    def df(self) -> pl.DataFrame:
        pass

    @abstractmethod
    def timestamp_col(self) -> str:
        pass

    @abstractmethod
    def symbol_col(self) -> str:
        pass

    @abstractmethod
    def price_col(self) -> str:
        pass

    @abstractmethod
    def volume_col(self) -> str:
        pass

    @abstractmethod
    def until(self, now: dt.datetime) -> "Universe":
        pass

    def prices(self, now: dt.datetime | None = None) -> Dict[str, float]:
        if now is None:
            now = self.df()[self.timestamp_col()].max()

        df = (
            self.df()
            .filter(pl.col(self.timestamp_col()) <= now)
            .sort([self.symbol_col(), self.timestamp_col()])
            .group_by(self.symbol_col())
            .agg(pl.col(self.price_col()).last())
        )

        return {
            r[self.symbol_col()]: r[self.price_col()]
            for r in (df.iter_rows(named=True))
        }

    def valid_period(self, period: dt.timedelta) -> bool:
        m = pl.duration(
            days=period.days, seconds=period.seconds, microseconds=period.microseconds
        ).dt.total_milliseconds()
        return (
            self.df()
            .filter(
                (
                    pl.col(self.timestamp_col())
                    .diff()
                    .over(self.symbol_col())
                    .dt.total_milliseconds()
                    % m
                )
                != 0
            )
            .height
            > 0
        )

    def timestamps(self) -> list[dt.datetime]:
        return sorted(self.df()[self.timestamp_col()].unique().to_list())


@dataclass(frozen=True)
class Position:
    symbol: str
    """Ticker"""
    shares: float
    """Held net shared, negative for short positions"""
    entry: dt.datetime
    """Entry time"""
    price: float
    """Entry price"""
    fee: float
    """Entry fee"""


@dataclass
class Trade:
    symbol: str
    entry_ts: dt.datetime
    exit_ts: dt.datetime
    entry_price: float
    exit_price: float
    entry_fee: float
    exit_fee: float
    shares: float


@dataclass(frozen=True)
class Order:
    """Orders send by the execution model."""

    symbol: str
    shares: float
    """negative for sell orders."""
    fee: float


@dataclass(frozen=True)
class Portfolio:
    positions: list[Position]
    """symbol -> Position"""
    cash: float
    working: list[Order]
    """working orders (stops)"""

    def get_avg(self, symbol) -> Position | None:
        cost = 0.0
        shares = 0.0
        fee = 0.0
        entry = None

        for p in self.positions:
            if p.symbol != symbol:
                continue
            shares += p.shares
            fee += p.fee
            cost += p.shares * p.price
            entry = min(entry, p.entry) if entry is not None else p.entry

        if entry is None:
            return None
        else:
            return Position(
                symbol=symbol, shares=shares, entry=entry, price=cost / shares, fee=fee
            )

    def longs(self) -> list[Position]:
        return [p for p in self.positions if p.shares > 0]

    def shorts(self) -> list[Position]:
        return [p for p in self.positions if p.shares < 0]

    def execute_orders(
        self, now: dt.datetime, prices: Dict[str, float], orders: list[Order]
    ) -> Tuple["Portfolio", list[Trade]]:
        """Returns a new portfolio with the orders executed against. Retuns closed trades"""

        positions = sorted([copy(p) for p in self.positions], key=lambda p: p.entry)
        cash = self.cash
        closed = []

        for o in orders:
            shares = o.shares  # mutable remaining-to-fill quantity
            sym = o.symbol

            price = prices.get(sym)
            if price is None:
                l.warning(f"{now}: no price for {sym}")
                continue

            # --- FIFO matching against existing opposite-side positions ---
            for p in positions[:]:
                if abs(shares) < eps:          # order fully matched — stop early
                    break
                if np.sign(p.shares) == np.sign(shares):  # same direction — skip
                    continue
                if p.symbol != sym:
                    continue

                idx = positions.index(p)
                batch = min(abs(shares), abs(p.shares))

                # Fee is split proportionally between the closed batch and any
                # residual that remains open, so neither side double-counts.
                batch_fee_ratio = batch / abs(o.shares)
                batch_fee    = o.fee * batch_fee_ratio       # fee for this batch
                residual_fee = o.fee * (1.0 - batch_fee_ratio)  # fee for remainder

                pshares = p.shares + np.sign(shares) * batch
                shares -= np.sign(shares) * batch

                if abs(pshares) <= eps:
                    # Position fully closed
                    closed.append(
                        Trade(
                            symbol=sym,
                            entry_ts=p.entry,
                            exit_ts=now,
                            entry_price=p.price,
                            exit_price=price,
                            entry_fee=p.fee,
                            exit_fee=batch_fee,
                            shares=p.shares,
                        )
                    )
                    del positions[idx]
                else:
                    # Position partially reduced — record no trade yet; carry
                    # only the entry fee of the original lot forward.  The
                    # exit fee for this batch is accounted for via cash below.
                    positions[idx] = Position(
                        symbol=sym,
                        shares=pshares,
                        entry=p.entry,
                        price=p.price,
                        fee=p.fee,          # do NOT accumulate the exit fee here
                    )

                # Cash settlement for the closed batch (proceeds or cost)
                # plus the proportional exit fee for this batch.
                cash -= np.sign(o.shares) * batch * price + batch_fee

            if abs(shares) < eps:
                # Order fully consumed by FIFO — no new position to open.
                # Residual fee (if any rounding) already charged per-batch above.
                continue

            # --- Open a new position with the remaining unfilled shares ---
            # Use only the fee attributable to this remaining (opening) portion.
            open_fee_ratio = abs(shares) / abs(o.shares)
            open_fee = o.fee * open_fee_ratio
            positions.append(
                Position(
                    symbol=o.symbol,
                    shares=shares,
                    entry=now,
                    price=price,
                    fee=open_fee,
                )
            )
            # BUG-FIX: use `shares` (remaining), NOT `o.shares` (full order).
            # Using o.shares here double-charges the already-settled FIFO batches.
            cash -= shares * price + open_fee

        return Portfolio(positions=positions, cash=cash, working=[]), closed


# def _execute_orders(
#    orders: list[Order], prices: dict, ts: dt.datetime
# ) -> Tuple[Portfolio, float, float, float]:
#    new_folio_dict = {pos.symbol: pos for pos in folio}
#    total_fees = 0.0
#
#    # Sort orders to prioritize liquidations/reductions to free up cash
#    # Orders that reduce exposure (opposite sign of current position)
#    def order_priority(order):
#        pos = new_folio_dict.get(order.symbol)
#        if pos is None:
#            return 1  # Opening new position
#        # If same sign, we are increasing
#        if (pos.shares > 0 and order.shares > 0) or (
#            pos.shares < 0 and order.shares < 0
#        ):
#            return 2
#        return 0  # Reducing or closing
#
#    sorted_orders = sorted(orders, key=order_priority)
#
#    for order in sorted_orders:
#        price = prices.get(order.symbol)
#        if price is None:
#            continue
#
#        fee = abs(order.shares * price) * self.fee
#        total_fees += fee
#        self.cash -= order.shares * price + fee
#
#        if order.symbol in new_folio_dict:
#            pos = new_folio_dict[order.symbol]
#            new_shares = pos.shares + order.shares
#
#            # Check if position is closed or direction reversed
#            if abs(new_shares) < EPSILON:
#                # closed
#                self.trades.append(
#                    Trade(
#                        symbol=pos.symbol,
#                        entry_ts=pos.ts,
#                        exit_ts=ts,
#                        entry_price=pos.open,
#                        exit_price=price,
#                        shares=pos.shares,
#                        high=pos.high,
#                        low=pos.low,
#                    )
#                )
#                del new_folio_dict[order.symbol]
#            elif (pos.shares > 0 and new_shares < 0) or (
#                pos.shares < 0 and new_shares > 0
#            ):
#                # reversed
#                self.trades.append(
#                    Trade(
#                        symbol=pos.symbol,
#                        entry_ts=pos.ts,
#                        exit_ts=ts,
#                        entry_price=pos.open,
#                        exit_price=price,
#                        shares=pos.shares,
#                        high=pos.high,
#                        low=pos.low,
#                    )
#                )
#                new_folio_dict[order.symbol] = Position(
#                    order.symbol, new_shares, price, ts, price, price
#                )
#            else:
#                # scaling
#                if (order.shares > 0 and pos.shares > 0) or (
#                    order.shares < 0 and pos.shares < 0
#                ):
#                    # scaling in: update average price
#                    total_shares = pos.shares + order.shares
#                    avg_price = (
#                        pos.shares * pos.open + order.shares * price
#                    ) / total_shares
#                    new_folio_dict[order.symbol] = Position(
#                        order.symbol,
#                        total_shares,
#                        avg_price,
#                        pos.ts,
#                        max(pos.high, price),
#                        min(pos.low, price),
#                    )
#                else:
#                    # scaling out: keep original entry price and ts
#                    self.trades.append(
#                        Trade(
#                            symbol=pos.symbol,
#                            entry_ts=pos.ts,
#                            exit_ts=ts,
#                            entry_price=pos.open,
#                            exit_price=price,
#                            shares=-order.shares,
#                            high=pos.high,
#                            low=pos.low,
#                        )
#                    )
#                    new_folio_dict[order.symbol] = Position(
#                        order.symbol,
#                        new_shares,
#                        pos.open,
#                        pos.ts,
#                        pos.high,
#                        pos.low,
#                    )
#        else:
#            if abs(order.shares) > EPSILON:
#                new_folio_dict[order.symbol] = Position(
#                    order.symbol, order.shares, price, ts, price, price
#                )
#    return list(new_folio_dict.values()), total_fees


@dataclass(frozen=True)
class Signal:
    """Insights generated by alpha models."""

    symbol: str
    bullish: bool
    confidence: float


@dataclass(frozen=True)
class Target:
    """Portfolio targets derived from Signals by the portfolio model."""

    symbol: str
    weight: float
    """Percentage of margin, negative for short."""


class AlphaModel(ABC):
    @abstractmethod
    def __call__(self, u: Universe) -> list[Signal]:
        pass


class PortfolioModel(ABC):
    @abstractmethod
    def __call__(
        self,
        u: Universe,
        signals: list[Signal],
        portfolio: Portfolio,
    ) -> list[Target]:
        pass


class RiskModel(ABC):
    @abstractmethod
    def __call__(
        self, u: Universe, targets: list[Target], portfolio: Portfolio
    ) -> list[Target]:
        pass


class ExecutionModel(ABC):
    @abstractmethod
    def __call__(
        self, u: Universe, targets: list[Target], portfolio: Portfolio
    ) -> list[Order]:
        pass
