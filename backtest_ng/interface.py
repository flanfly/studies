"""Public interfaces.

AlphaModels --[ Signals ]-> PortfolioModel --[ Targets ]-> RiskModel(s) --[ adj. Targets ]-> ExecutionModel --[ Orders ]-> Backtest
"""

from abc import ABC, abstractmethod
import polars as pl
import datetime as dt

import numpy as np

from dataclasses import dataclass

from copy import copy
from typing import Tuple, Dict, Literal

import logging as l

eps = 1e-12


@dataclass(frozen=True)
class Kline:
    high: float | None
    low: float | None
    price: float


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
    def high_col(self) -> str | None:
        pass

    @abstractmethod
    def low_col(self) -> str | None:
        pass

    @abstractmethod
    def volume_col(self) -> str:
        pass

    @abstractmethod
    def until(self, now: dt.datetime) -> "Universe":
        pass

    def klines(self, now: dt.datetime | None = None) -> Dict[str, Kline]:
        if now is None:
            now = self.df()[self.timestamp_col()].max()

        # Pull the last bar per symbol so we get the close, high, and low
        # for the as-of timestamp in one shot.  The previous implementation
        # only aggregated the price column, which caused a KeyError when a
        # high/low column was configured on the universe.
        last_bar = (
            self.df()
            .filter(pl.col(self.timestamp_col()) <= now)
            .sort([self.symbol_col(), self.timestamp_col()])
            .group_by(self.symbol_col())
            .tail(1)
        )

        return {
            r[self.symbol_col()]: Kline(
                high=r[self.high_col()] if self.high_col() else None,
                low=r[self.low_col()] if self.low_col() else None,
                price=r[self.price_col()],
            )
            for r in last_bar.iter_rows(named=True)
        }

    def valid_period(self, period: dt.timedelta) -> bool:
        """Check that ``period`` is achievable on the data.

        The data's bar size is the GCD of all positive consecutive
        diffs within each symbol (taken across all symbols — the
        tightest constraint wins). ``period`` is achievable iff
        ``period`` is a multiple of the bar size, i.e. a rebalance
        lands exactly on a bar boundary.

        Examples
        --------
        * 1d bars, period 2d: ``2d % 1d == 0`` → valid.
        * 1d bars, period 3h: ``3h % 1d != 0`` → invalid.
        * 2d bars, period 1d: ``1d % 2d != 0`` → invalid.
        * 1d bars with a single 8d gap, period 4d: GCD is 1d,
          ``4d % 1d == 0`` → valid (rebalance lands on every 4th
          1d bar, ignoring the gap row).
        """
        if period <= dt.timedelta(0):
            return False

        # Convert period to integer milliseconds.
        period_ms = (
            period.days * 86_400_000
            + period.seconds * 1_000
            + period.microseconds // 1_000
        )

        diffs = (
            self.df()
            .sort([self.symbol_col(), self.timestamp_col()])
            .select(
                pl.col(self.timestamp_col())
                .diff()
                .over(self.symbol_col())
                .dt.total_milliseconds()
                .alias("diff_ms")
            )["diff_ms"]
            .drop_nulls()
            .unique()
            .to_list()
        )
        # Skip zero / negative diffs (duplicate rows, out-of-order).
        diffs = [d for d in diffs if d > 0]
        if not diffs:
            # Single-row symbol, or every row is a duplicate — there's
            # no cadence to check; period is moot.  Treat as valid.
            return True

        def _gcd(a: int, b: int) -> int:
            while b:
                a, b = b, a % b
            return a

        bar_ms = diffs[0]
        for d in diffs[1:]:
            bar_ms = _gcd(bar_ms, d)
            if bar_ms == 1:
                # Smallest possible bar size; the only way ``period``
                # fails to divide it is if period < 1ms, which we've
                # already rejected via the ``<= 0`` check above.
                break

        return period_ms % bar_ms == 0

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

    type: Literal["market", "limit", "stop-loss", "trailing-stop", "cancel"]
    id: str
    """random, unique order id"""
    cancel: str | None
    """order id to cancel, only valid for type == 'cancel'"""
    symbol: str
    shares: float
    """negative for sell orders."""
    fee: float
    stop_loss: float | None = None
    """trigger price for stop-loss / trailing-stop orders (None otherwise)."""


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
        self,
        now: dt.datetime,
        klines: Dict[str, Kline],
        orders: list[Order],
        price_overrides: Dict[str, float] | None = None,
    ) -> Tuple["Portfolio", list[Trade]]:
        """Returns a new portfolio with the orders executed against. Retuns closed trades

        ``price_overrides`` lets the caller fill an order at a price other
        than the kline close (e.g. a triggered stop loss fills at its
        stop price, not the bar's close).
        """

        positions = sorted([copy(p) for p in self.positions], key=lambda p: p.entry)
        cash = self.cash
        closed = []
        price_overrides = price_overrides or {}

        for o in orders:
            shares = o.shares  # mutable remaining-to-fill quantity
            sym = o.symbol

            kline = klines.get(sym)
            if kline is None:
                l.warning(f"{now}: no price for {sym}")
                continue

            fill_price = price_overrides.get(sym, kline.price)

            # --- FIFO matching against existing opposite-side positions ---
            for p in positions[:]:
                if abs(shares) < eps:  # order fully matched — stop early
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
                batch_fee = o.fee * batch_fee_ratio  # fee for this batch
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
                            exit_price=fill_price,
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
                        fee=p.fee,  # do NOT accumulate the exit fee here
                    )

                # Cash settlement for the closed batch (proceeds or cost)
                # plus the proportional exit fee for this batch.
                cash -= np.sign(o.shares) * batch * fill_price + batch_fee

            if abs(shares) < eps:
                # Order fully consumed by FIFO — no new position to open.
                # Residual fee (if any rounding) already charged per-batch above.
                continue

            # --- Merge into existing same-direction position, or open new ---
            fee = o.fee * abs(shares) / abs(o.shares)
            merged = False
            for i, p in enumerate(positions):
                if p.symbol == sym and np.sign(p.shares) == np.sign(shares):
                    # Merge: weighted-average entry price, keep earliest entry.
                    total_shares = p.shares + shares
                    avg_price = (
                        p.shares * p.price + shares * fill_price
                    ) / total_shares
                    positions[i] = Position(
                        symbol=sym,
                        shares=total_shares,
                        entry=p.entry,
                        price=avg_price,
                        fee=p.fee + fee,
                    )
                    cash -= shares * fill_price + fee
                    merged = True
                    break

            if not merged:
                positions.append(
                    Position(
                        symbol=o.symbol,
                        shares=shares,
                        entry=now,
                        price=fill_price,
                        fee=fee,
                    )
                )
                cash -= shares * fill_price + fee

        return Portfolio(positions=positions, cash=cash, working=list(self.working)), closed

    def check_working_against_klines(
        self, klines: Dict[str, Kline]
    ) -> Tuple[list[Order], "Portfolio"]:
        """Return triggered stop-loss orders, plus a new portfolio with
        them removed from ``working``.

        Trigger rules (per ``Order.stop_loss`` and the sign of ``shares``):

        * Negative ``shares`` (a sell-stop, hedging a long position) triggers
          when ``kline.low <= stop_loss`` — price fell to or past the stop.
        * Positive ``shares`` (a buy-stop, hedging a short position) triggers
          when ``kline.high >= stop_loss`` — price rose to or past the stop.

        Stops whose kline is missing or whose high/low are unavailable are
        silently held (cannot tell whether they triggered) so the engine
        can retry on the next bar.
        """
        triggered: list[Order] = []
        remaining: list[Order] = []

        for o in self.working:
            if o.type != "stop-loss" or o.stop_loss is None:
                # Non-stop working orders are not evaluated here.
                remaining.append(o)
                continue

            kline = klines.get(o.symbol)
            if kline is None or kline.high is None or kline.low is None:
                # No way to know — hold the order for the next bar.
                remaining.append(o)
                continue

            if o.shares < 0:
                # Sell-stop: trigger when the bar's low touched the stop.
                if kline.low <= o.stop_loss:
                    triggered.append(o)
                else:
                    remaining.append(o)
            else:
                # Buy-stop: trigger when the bar's high touched the stop.
                if kline.high >= o.stop_loss:
                    triggered.append(o)
                else:
                    remaining.append(o)

        return triggered, Portfolio(
            positions=self.positions, cash=self.cash, working=remaining
        )


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

    max_risk: float | None = None
    """Maximum drawdown, implemented as stop loss. ``None`` ⇒ no stop attached."""


class AlphaModel(ABC):
    @abstractmethod
    def __call__(self, history: pl.DataFrame, u: Universe) -> list[Signal]:
        pass


class PortfolioModel(ABC):
    @abstractmethod
    def __call__(
        self,
        history: pl.DataFrame,
        u: Universe,
        signals: list[Signal],
        portfolio: Portfolio,
    ) -> list[Target]:
        pass


class RiskModel(ABC):
    @abstractmethod
    def __call__(
        self,
        history: pl.DataFrame,
        u: Universe,
        targets: list[Target],
        portfolio: Portfolio,
    ) -> list[Target]:
        pass


class ExecutionModel(ABC):
    @abstractmethod
    def __call__(
        self,
        history: pl.DataFrame,
        u: Universe,
        targets: list[Target],
        portfolio: Portfolio,
    ) -> list[Order]:
        pass
