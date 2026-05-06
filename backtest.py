from abc import ABC, abstractmethod
import polars as pl
import datetime as dt

import sys

from tqdm import tqdm
from dataclasses import dataclass

from typing import Tuple

EPSILON = sys.float_info.epsilon


@dataclass(frozen=True)
class Position:
    symbol: str
    shares: float  # negative for short
    open: float
    ts: dt.datetime
    high: float
    low: float


@dataclass(frozen=True)
class Signal:
    symbol: str
    bullish: bool
    confidence: float


@dataclass(frozen=True)
class Order:
    symbol: str
    shares: float  # negative for sell


EPSILON = 1e-9

class AlphaModel(ABC):
    @abstractmethod
    def __call__(self, df: pl.DataFrame) -> list[Signal]:
        pass


class Rank(AlphaModel):
    def __init__(self, signal: pl.Expr, gates: Tuple[pl.Expr, pl.Expr]):
        self.signal = signal
        self.gates = gates

    def __call__(self, df: pl.DataFrame) -> list[Signal]:
        today = df["ts"].max()
        long = df.filter(self.gates[0] & (pl.col("ts") == today)).sort(
            self.signal, descending=True
        )
        short = df.filter(self.gates[1] & (pl.col("ts") == today)).sort(
            self.signal, descending=False
        )

        return [
            *[
                Signal(
                    symbol=row["symbol"],
                    bullish=True,
                    confidence=(len(long) - i) / len(long),
                )
                for i, row in enumerate(long.iter_rows(named=True))
            ],
            *[
                Signal(
                    symbol=row["symbol"],
                    bullish=False,
                    confidence=(len(short) - i) / len(short),
                )
                for (i, row) in enumerate(short.iter_rows(named=True))
            ],
        ]


class PortfolioModel(ABC):
    @abstractmethod
    def __call__(
        self,
        df: pl.DataFrame,
        signals: list[Signal],
        folio: list[Position],
        equity: float,
    ) -> list[Order]:
        pass


class EqualWeight(PortfolioModel):
    def __init__(self, timestamp_col="ts", symbol_col="symbol", price_col="close"):
        self.ts_col = timestamp_col
        self.symbol_col = symbol_col
        self.price_col = price_col

    def __call__(
        self,
        df: pl.DataFrame,
        signals: list[Signal],
        folio: list[Position],
        equity: float,
    ) -> list[Order]:
        today = df[self.ts_col].max()
        prices = dict(
            df.filter(pl.col(self.ts_col) == today)
            .select([self.symbol_col, self.price_col])
            .iter_rows()
        )

        if not signals or equity <= 0:
            # liquidate all positions if no signals or no equity
            return [Order(pos.symbol, -pos.shares) for pos in folio]

        target_shares = {}
        target_weight_value = equity / len(signals)
        for signal in signals:
            price = prices.get(signal.symbol)
            if price is not None and price > 0:
                target_shares[signal.symbol] = (
                    target_weight_value / price * (1 if signal.bullish else -1)
                )

        orders = []
        # Current positions
        current_positions = {pos.symbol: pos.shares for pos in folio}
        all_symbols = set(current_positions.keys()) | set(target_shares.keys())

        for sym in all_symbols:
            current = current_positions.get(sym, 0.0)
            target = target_shares.get(sym, 0.0)
            delta = target - current
            if abs(delta) > EPSILON:  # Avoid tiny precision orders
                orders.append(Order(sym, delta))

        return orders


class VolumeWeighted(PortfolioModel):
    def __init__(
        self,
        timestamp_col="ts",
        symbol_col="symbol",
        price_col="close",
        volume_col="volume",
    ):
        self.ts_col = timestamp_col
        self.symbol_col = symbol_col
        self.price_col = price_col
        self.volume_col = volume_col

    def __call__(
        self,
        df: pl.DataFrame,
        signals: list[Signal],
        folio: list[Position],
        equity: float,
    ) -> list[Order]:
        today = df[self.ts_col].max()
        prices = dict(
            df.filter(pl.col(self.ts_col) == today)
            .select([self.symbol_col, self.price_col])
            .iter_rows()
        )
        vols = dict(
            df.filter(pl.col(self.ts_col) == today)
            .select([self.symbol_col, self.volume_col])
            .iter_rows()
        )

        if not signals or equity <= 0:
            return [Order(pos.symbol, -pos.shares) for pos in folio]

        total_vol = sum(vols.get(signal.symbol, 0) for signal in signals)
        if total_vol <= 0:
            return [Order(pos.symbol, -pos.shares) for pos in folio]

        target_shares = {}
        for signal in signals:
            price = prices.get(signal.symbol)
            if price is not None and price > 0:
                target_shares[signal.symbol] = (
                    (vols.get(signal.symbol, 0) / total_vol)
                    * equity
                    / price
                    * (1 if signal.bullish else -1)
                )

        orders = []
        current_positions = {pos.symbol: pos.shares for pos in folio}
        all_symbols = set(current_positions.keys()) | set(target_shares.keys())

        for sym in all_symbols:
            current = current_positions.get(sym, 0.0)
            target = target_shares.get(sym, 0.0)
            delta = target - current
            if abs(delta) > EPSILON:
                orders.append(Order(sym, delta))

        return orders


class SimpleLeverage(PortfolioModel):
    def __init__(self, leverage: float, inner: PortfolioModel = EqualWeight()):
        self.inner = inner
        self.leverage = leverage

    def __call__(
        self,
        df: pl.DataFrame,
        signals: list[Signal],
        folio: list[Position],
        equity: float,
    ) -> list[Order]:
        return self.inner(df, signals, folio, equity * self.leverage)


class TopN(PortfolioModel):
    def __init__(self, max_long=5, max_short=5, portfolio_model=None):
        self.max_long = max_long
        self.max_short = max_short
        self.portfolio_model = portfolio_model or EqualWeight()

    def __call__(
        self,
        df: pl.DataFrame,
        signals: list[Signal],
        folio: list[Position],
        equity: float,
    ) -> list[Order]:
        if not signals:
            return []

        # Sort signals by confidence for long and short separately
        long_signals = sorted(
            [s for s in signals if s.bullish], key=lambda s: s.confidence, reverse=True
        )
        short_signals = sorted(
            [s for s in signals if not s.bullish],
            key=lambda s: s.confidence,
            reverse=True,
        )

        top_long = long_signals[: self.max_long]
        top_short = short_signals[: self.max_short]

        selected_signals = top_long + top_short
        return self.portfolio_model(df, selected_signals, folio, equity)


class RiskModel(ABC):
    @abstractmethod
    def __call__(self, df: pl.DataFrame, folio: list[Position]) -> list[Order]:
        pass


class MaxDrawdown(RiskModel):
    def __init__(
        self,
        absolute=0.2,  # longs
        trailing=0.1,  # longs
        absolute_short=None,  # equal absolute if none
        trailing_short=None,  # equal trailing if none
        timestamp_col="ts",
        symbol_col="symbol",
        price_col="close",
    ):
        self.absolute = absolute
        self.trailing = trailing
        self.absolute_short = absolute_short if absolute_short is not None else absolute
        self.trailing_short = trailing_short if trailing_short is not None else trailing
        self.ts_col = timestamp_col
        self.symbol_col = symbol_col
        self.price_col = price_col

    def __call__(self, df: pl.DataFrame, folio: list[Position]) -> list[Order]:
        if not folio:
            return []

        today = df[self.ts_col].max()
        day_data = df.filter(pl.col(self.ts_col) == today)
        prices = dict(
            day_data.select([self.symbol_col, self.price_col]).iter_rows()
        )
        # Use low/high columns if available for stop trigger logic
        lows = dict(day_data.select([self.symbol_col, "low"]).iter_rows()) if "low" in day_data.columns else prices
        highs = dict(day_data.select([self.symbol_col, "high"]).iter_rows()) if "high" in day_data.columns else prices

        orders = []
        for pos in folio:
            curr_price = prices.get(pos.symbol)
            if curr_price is None:
                continue

            if pos.shares > 0:  # Long
                low_price = lows.get(pos.symbol, curr_price)
                if low_price / pos.open - 1 < -self.absolute:
                    orders.append(Order(pos.symbol, -pos.shares))
                elif low_price / pos.high - 1 < -self.trailing:
                    orders.append(Order(pos.symbol, -pos.shares))
            else:  # Short
                high_price = highs.get(pos.symbol, curr_price)
                if 1 - high_price / pos.open < -self.absolute_short:
                    orders.append(Order(pos.symbol, -pos.shares))
                elif 1 - high_price / pos.low < -self.trailing_short:
                    orders.append(Order(pos.symbol, -pos.shares))
        return orders


class NoRisk(RiskModel):
    def __call__(self, df: pl.DataFrame, folio: list[Position]) -> list[Order]:
        return []


@dataclass
class Trade:
    symbol: str
    entry_ts: dt.datetime
    exit_ts: dt.datetime
    entry_price: float
    exit_price: float
    shares: float
    high: float
    low: float


class Backtest:
    def __init__(
        self,
        df: pl.DataFrame,
        alpha: AlphaModel,
        portfolio: PortfolioModel = EqualWeight(),
        risk: RiskModel = NoRisk(),
        period=30,
        fee=0.002,
        benchmark=None,
        timestamp_col="ts",
        symbol_col="symbol",
        price_col="close",
        eager_rebalance=False,
    ):
        if not all(col in df.columns for col in [timestamp_col, symbol_col, price_col]):
            raise ValueError(
                f"DataFrame must contain '{timestamp_col}', '{symbol_col}', and '{price_col}' columns"
            )

        self.df = df
        self.alpha = alpha
        self.portfolio = portfolio
        self.risk = risk
        self.period = period
        self.benchmark = benchmark
        self.ts_col = timestamp_col
        self.symbol_col = symbol_col
        self.price_col = price_col
        self.fee = fee
        self.eager_rebalance = eager_rebalance

        self.trades: list[Trade] = []

    def run(self, initial_equity=1):
        self.initial_equity = initial_equity
        days = self.df.select(pl.col(self.ts_col).unique().sort()).to_series().to_list()

        self.folio = []
        self.cash = float(initial_equity)
        last_rebalance = None
        self.history = []

        for day in tqdm(days):
            # get the data for the current day
            dfnow = self.df.filter(pl.col(self.ts_col) <= day).sort(self.ts_col)
            day_data = self.df.filter(pl.col(self.ts_col) == day)
            prices = dict(
                day_data.select([self.symbol_col, self.price_col]).iter_rows()
            )
            highs = dict(day_data.select([self.symbol_col, "high"]).iter_rows()) if "high" in day_data.columns else prices
            lows = dict(day_data.select([self.symbol_col, "low"]).iter_rows()) if "low" in day_data.columns else prices

            # 1. update high/low for positions using current day's extremes
            new_folio = []
            for pos in self.folio:
                c_high = highs.get(pos.symbol, prices.get(pos.symbol, pos.open))
                c_low = lows.get(pos.symbol, prices.get(pos.symbol, pos.open))
                new_folio.append(
                    Position(
                        pos.symbol,
                        pos.shares,
                        pos.open,
                        pos.ts,
                        max(pos.high, c_high),
                        min(pos.low, c_low),
                    )
                )
            self.folio = new_folio

            day_fees = 0.0

            # 2. Risk Model First (Stops)
            orders = self.risk(dfnow, self.folio)
            self.folio, fees = self._execute_orders(self.folio, orders, prices, day)
            day_fees += fees

            # 3. Portfolio Rebalance
            equity = self.cash + sum(
                pos.shares * prices.get(pos.symbol, pos.open) for pos in self.folio
            )

            if equity <= EPSILON:
                print(f"Broke at {day} due to low equity: {equity}")
                break

            signals = self.alpha(dfnow)
            
            should_rebalance = False
            if last_rebalance is None or (day - last_rebalance).days >= self.period:
                should_rebalance = True
            elif self.eager_rebalance and self.cash > (equity * 0.05):
                should_rebalance = True

            if should_rebalance:
                orders = self.portfolio(dfnow, signals, self.folio, equity)
                self.folio, fees = self._execute_orders(self.folio, orders, prices, day)
                day_fees += fees
                last_rebalance = day

            # Final equity for the day after all trades and fees
            equity = self.cash + sum(
                pos.shares * prices.get(pos.symbol, pos.open) for pos in self.folio
            )
            
            n_long = sum(1 for pos in self.folio if pos.shares > 0)
            n_short = sum(1 for pos in self.folio if pos.shares < 0)
            
            self.history.append({
                "ts": day, 
                "equity": equity, 
                "fees": day_fees,
                "n_long": n_long,
                "n_short": n_short
            })

        return self

    # columns: entry ts, symbol, shares (negative for short), exit ts
    def live(self, equity: float) -> pl.DataFrame:
        """
        Compute the portfolio for paper trading using the last day's data.
        Runs alpha and portfolio construction models with an empty starting
        folio and returns the resulting positions as a DataFrame.

        Returns
        -------
        pl.DataFrame with columns: symbol, shares, entry_ts, exit_ts, entry_price
        """
        today = self.df[self.ts_col].max()
        dfnow = self.df.filter(pl.col(self.ts_col) <= today).sort(self.ts_col)
        day_data = self.df.filter(pl.col(self.ts_col) == today)
        prices = dict(
            day_data.select([self.symbol_col, self.price_col]).iter_rows()
        )

        signals = self.alpha(dfnow)
        orders = self.portfolio(dfnow, signals, [], equity)

        rows = []
        for order in orders:
            price = prices.get(order.symbol)
            if price is not None and abs(order.shares) > EPSILON:
                exit_ts = today + dt.timedelta(days=self.period)
                rows.append(
                    {
                        "symbol": order.symbol,
                        "shares": order.shares,
                        "entry_ts": today,
                        "exit_ts": exit_ts,
                        "entry_price": price,
                    }
                )

        if not rows:
            return pl.DataFrame(
                schema={
                    "symbol": pl.Utf8,
                    "shares": pl.Float64,
                    "entry_ts": pl.Datetime,
                    "exit_ts": pl.Datetime,
                    "entry_price": pl.Float64,
                }
            )

        return pl.DataFrame(rows)

    def _execute_orders(
        self, folio: list[Position], orders: list[Order], prices: dict, ts: dt.datetime
    ) -> Tuple[list[Position], float]:
        new_folio_dict = {pos.symbol: pos for pos in folio}
        total_fees = 0.0

        # Sort orders to prioritize liquidations/reductions to free up cash
        # Orders that reduce exposure (opposite sign of current position)
        def order_priority(order):
            pos = new_folio_dict.get(order.symbol)
            if pos is None:
                return 1 # Opening new position
            # If same sign, we are increasing
            if (pos.shares > 0 and order.shares > 0) or (pos.shares < 0 and order.shares < 0):
                return 2
            return 0 # Reducing or closing

        sorted_orders = sorted(orders, key=order_priority)

        for order in sorted_orders:
            price = prices.get(order.symbol)
            if price is None:
                continue

            fee = abs(order.shares * price) * self.fee
            total_fees += fee
            self.cash -= (order.shares * price + fee)

            if order.symbol in new_folio_dict:
                pos = new_folio_dict[order.symbol]
                new_shares = pos.shares + order.shares

                # Check if position is closed or direction reversed
                if abs(new_shares) < EPSILON:
                    # closed
                    self.trades.append(
                        Trade(
                            symbol=pos.symbol,
                            entry_ts=pos.ts,
                            exit_ts=ts,
                            entry_price=pos.open,
                            exit_price=price,
                            shares=pos.shares,
                            high=pos.high,
                            low=pos.low,
                        )
                    )
                    del new_folio_dict[order.symbol]
                elif (pos.shares > 0 and new_shares < 0) or (
                    pos.shares < 0 and new_shares > 0
                ):
                    # reversed
                    self.trades.append(
                        Trade(
                            symbol=pos.symbol,
                            entry_ts=pos.ts,
                            exit_ts=ts,
                            entry_price=pos.open,
                            exit_price=price,
                            shares=pos.shares,
                            high=pos.high,
                            low=pos.low,
                        )
                    )
                    new_folio_dict[order.symbol] = Position(
                        order.symbol, new_shares, price, ts, price, price
                    )
                else:
                    # scaling
                    if (order.shares > 0 and pos.shares > 0) or (
                        order.shares < 0 and pos.shares < 0
                    ):
                        # scaling in: update average price
                        total_shares = pos.shares + order.shares
                        avg_price = (
                            pos.shares * pos.open + order.shares * price
                        ) / total_shares
                        new_folio_dict[order.symbol] = Position(
                            order.symbol,
                            total_shares,
                            avg_price,
                            pos.ts,
                            max(pos.high, price),
                            min(pos.low, price),
                        )
                    else:
                        # scaling out: keep original entry price and ts
                        self.trades.append(
                            Trade(
                                symbol=pos.symbol,
                                entry_ts=pos.ts,
                                exit_ts=ts,
                                entry_price=pos.open,
                                exit_price=price,
                                shares=-order.shares,
                                high=pos.high,
                                low=pos.low,
                            )
                        )
                        new_folio_dict[order.symbol] = Position(
                            order.symbol,
                            new_shares,
                            pos.open,
                            pos.ts,
                            pos.high,
                            pos.low,
                        )
            else:
                if abs(order.shares) > EPSILON:
                    new_folio_dict[order.symbol] = Position(
                        order.symbol, order.shares, price, ts, price, price
                    )
        return list(new_folio_dict.values()), total_fees

    def report(self, plot='brief') -> pl.DataFrame:
        if not self.history:
            return pl.DataFrame()

        history_df = pl.DataFrame(self.history)
        history_df = history_df.with_columns(
            returns=pl.col("equity").pct_change().fill_null(0),
        )

        # Benchmark returns for IR calculation
        if self.benchmark is not None:
            if isinstance(self.benchmark, str):
                bench_df = self.df.filter(pl.col(self.symbol_col) == self.benchmark)
            else:
                bench_df = self.benchmark

            bench_df = bench_df.select(
                [
                    pl.col(self.ts_col).alias("ts"),
                    pl.col(self.price_col).alias("bench_price"),
                ]
            ).sort("ts")

            history_df = history_df.join(bench_df, on="ts", how="left")
            history_df = history_df.with_columns(
                bench_returns=pl.col("bench_price").pct_change().fill_null(0)
            ).with_columns(active_returns=pl.col("returns") - pl.col("bench_returns"))
        else:
            history_df = history_df.with_columns(
                active_returns=pl.lit(None).cast(pl.Float64)
            )

        # Trades stats
        if self.trades:
            trades_df = pl.from_dicts([t.__dict__ for t in self.trades])
            trades_df = trades_df.with_columns(
                year=pl.col("exit_ts").dt.year(),
                pnl=(pl.col("exit_price") - pl.col("entry_price")) * pl.col("shares"),
                pnl_pct=pl.when(pl.col("shares") > 0)
                .then(pl.col("exit_price") / pl.col("entry_price") - 1)
                .otherwise(1 - pl.col("exit_price") / pl.col("entry_price")),
                mfe=pl.when(pl.col("shares") > 0)
                .then(pl.col("high") / pl.col("entry_price") - 1)
                .otherwise(1 - pl.col("low") / pl.col("entry_price")),
                mae=pl.when(pl.col("shares") > 0)
                .then(pl.col("low") / pl.col("entry_price") - 1)
                .otherwise(1 - pl.col("high") / pl.col("entry_price")),
            )
            trade_yearly = trades_df.group_by("year").agg(
                [
                    pl.col("mfe").mean().alias("mfe"),
                    pl.col("mfe").std().alias("mfe_std"),
                    pl.col("mae").mean().alias("mae"),
                    pl.col("mae").std().alias("mae_std"),
                    ((pl.col("pnl") > 0).sum() / pl.col("pnl").count()).alias(
                        "win_rate"
                    ),
                ]
            )
        else:
            trade_yearly = pl.DataFrame(
                schema={
                    "year": pl.Int32,
                    "mfe": pl.Float64,
                    "mfe_std": pl.Float64,
                    "mae": pl.Float64,
                    "mae_std": pl.Float64,
                    "win_rate": pl.Float64,
                }
            )

        # Yearly equity stats
        history_df = history_df.with_columns(
            year=pl.col("ts").dt.year(),
            neg_returns=pl.when(pl.col("returns") < 0)
            .then(pl.col("returns"))
            .otherwise(0),
        )

        if self.benchmark is not None:
            history_df = history_df.with_columns(
                bench_neg_returns=pl.when(pl.col("bench_returns") < 0)
                .then(pl.col("bench_returns"))
                .otherwise(0),
            )

        agg_exprs = [
            # CAGR for the year
            (
                (pl.col("equity").last() / pl.col("equity").first())
                ** (
                    1
                    / (
                        (pl.col("ts").last() - pl.col("ts").first()).dt.total_days()
                        / 365.25
                    )
                )
                - 1
            ).alias("cagr"),
            (pl.col("returns").mean() * 252).alias("ann_return"),
            (pl.col("returns").std() * (252**0.5)).alias("ann_std"),
            (pl.col("neg_returns").std() * (252**0.5)).alias("downside_std"),
            (pl.col("active_returns").mean() * 252).alias("active_return_ann"),
            (pl.col("active_returns").std() * (252**0.5)).alias("tracking_error"),
            ((pl.col("equity") / pl.col("equity").cum_max() - 1).min()).alias("maxdd"),
            pl.col("fees").sum().alias("fees"),
        ]

        if self.benchmark is not None:
            agg_exprs.extend(
                [
                    (
                        (pl.col("bench_price").last() / pl.col("bench_price").first())
                        ** (
                            1
                            / (
                                (
                                    pl.col("ts").last() - pl.col("ts").first()
                                ).dt.total_days()
                                / 365.25
                            )
                        )
                        - 1
                    ).alias("bench_cagr"),
                    (pl.col("bench_returns").mean() * 252).alias("bench_ann_return"),
                    (pl.col("bench_returns").std() * (252**0.5)).alias("bench_ann_std"),
                    (pl.col("bench_neg_returns").std() * (252**0.5)).alias(
                        "bench_downside_std"
                    ),
                    (
                        (pl.col("bench_price") / pl.col("bench_price").cum_max() - 1).min()
                    ).alias("bench_maxdd"),
                ]
            )

        yearly_stats = (
            history_df.group_by("year")
            .agg(agg_exprs)
            .with_columns(
                sharpe=pl.when(pl.col("ann_std") > EPSILON)
                .then(pl.col("ann_return") / pl.col("ann_std"))
                .otherwise(pl.lit(None).cast(pl.Float64)),
                sortino=pl.when(pl.col("downside_std") > EPSILON)
                .then(pl.col("ann_return") / pl.col("downside_std"))
                .otherwise(pl.lit(None).cast(pl.Float64)),
                ir=pl.when(pl.col("tracking_error") > EPSILON)
                .then(pl.col("active_return_ann") / pl.col("tracking_error"))
                .otherwise(pl.lit(None).cast(pl.Float64)),
            )
        )

        if self.benchmark is not None:
            yearly_stats = yearly_stats.with_columns(
                bench_sharpe=pl.when(pl.col("bench_ann_std") > EPSILON)
                .then(pl.col("bench_ann_return") / pl.col("bench_ann_std"))
                .otherwise(pl.lit(None).cast(pl.Float64)),
                bench_sortino=pl.when(pl.col("bench_downside_std") > EPSILON)
                .then(pl.col("bench_ann_return") / pl.col("bench_downside_std"))
                .otherwise(pl.lit(None).cast(pl.Float64)),
            )

        final_report = (
            yearly_stats.join(trade_yearly, on="year", how="left")
            .sort("year")
        )

        if self.benchmark is not None:
            # Create two separate dataframes: one for Strategy and one for Benchmark
            # Metrics we want to compare
            comparison_cols = ["cagr", "ann_return", "ann_std", "maxdd", "sharpe", "sortino"]
            
            # 1. Strategy DataFrame
            strat_report = final_report.select(
                [pl.col("year"), pl.lit("Strategy").alias("src")] +
                [pl.col(c) for c in comparison_cols if c in final_report.columns] +
                [pl.col(c) for c in ["ir", "fees", "mfe", "mfe_std", "mae", "mae_std", "win_rate"] if c in final_report.columns]
            )
            
            # 2. Benchmark DataFrame
            bench_report = final_report.select(
                [pl.col("year"), pl.lit("Benchmark").alias("src")] +
                [pl.col(f"bench_{c}").alias(c) for c in comparison_cols if f"bench_{c}" in final_report.columns]
            )
            
            # 3. Combine and sort
            final_report = pl.concat([strat_report, bench_report], how="diagonal").sort(["year", "src"], descending=[False, True])
        else:
            final_report = final_report.with_columns(src=pl.lit("Strategy"))
            # Reorder columns to put src first after year
            cols = final_report.columns
            if "year" in cols:
                cols.remove("year")
                cols.remove("src")
                final_report = final_report.select(["year", "src"] + cols)

        if plot:
            import matplotlib.pyplot as plt

            show_symbols = (plot == 'full')

            # 1. Identify symbols to plot first
            symbols_to_plot = []
            if show_symbols and self.trades:
                trades_df = pl.from_dicts([t.__dict__ for t in self.trades])
                trades_df = trades_df.with_columns(
                    pnl=(pl.col("exit_price") - pl.col("entry_price")) * pl.col("shares")
                )

                symbol_stats = trades_df.group_by("symbol").agg(
                    [
                        pl.col("pnl").sum().alias("total_pnl"),
                        pl.col("pnl").count().alias("trade_count"),
                    ]
                )

                profitable = (
                    symbol_stats.filter(pl.col("total_pnl") > 0)
                    .sort("trade_count", descending=True)
                    .head(3)
                )
                unprofitable = (
                    symbol_stats.filter(pl.col("total_pnl") <= 0)
                    .sort("trade_count", descending=True)
                    .head(3)
                )
                
                for row in profitable.iter_rows(named=True):
                    symbols_to_plot.append((row["symbol"], "Profitable"))
                for row in unprofitable.iter_rows(named=True):
                    symbols_to_plot.append((row["symbol"], "Unprofitable"))

            num_plots = 2 + len(symbols_to_plot)
            fig, axes = plt.subplots(
                num_plots, 
                1, 
                figsize=(12, 4 * num_plots), 
                sharex=True, 
                gridspec_kw={'height_ratios': [3, 1] + [3] * len(symbols_to_plot)}
            )
            
            if num_plots == 1:
                axes = [axes]

            ax_eq = axes[0]
            ax_exp = axes[1]
            
            # Equity Plot
            ax_eq.plot(history_df["ts"], history_df["equity"], label="Strategy Equity")
            ax_eq.set_ylabel("Equity")
            ax_eq.set_yscale("log")
            
            if self.benchmark is not None:
                bench_prices = history_df["bench_price"].drop_nulls()
                if not bench_prices.is_empty():
                    bench_norm = bench_prices / bench_prices[0] * self.initial_equity
                    ax_eq.plot(
                        history_df.filter(pl.col("bench_price").is_not_null())["ts"],
                        bench_norm,
                        label="Benchmark",
                        linestyle="--",
                        color="gray",
                    )
            
            ax_eq.set_title("Strategy Performance & Symbol Details")
            ax_eq.legend(loc="upper left")

            # Exposure Plot (Long/Short/Net)
            if "n_long" in history_df.columns:
                ax_exp.fill_between(history_df["ts"], history_df["n_long"], color="green", alpha=0.3, label="Longs")
                ax_exp.fill_between(history_df["ts"], -history_df["n_short"], color="red", alpha=0.3, label="Shorts")
                net_bias = history_df["n_long"] - history_df["n_short"]
                ax_exp.plot(history_df["ts"], net_bias, color="black", linewidth=1, label="Net Bias")
                ax_exp.axhline(0, color='black', linestyle='-', alpha=0.2)
                ax_exp.set_ylabel("Exposure")
                ax_exp.legend(loc="upper left")

            # Symbol Plots
            for i, (sym, group_name) in enumerate(symbols_to_plot):
                ax = axes[i + 2]
                sym_df = self.df.filter(pl.col(self.symbol_col) == sym).sort(self.ts_col)
                if sym_df.is_empty():
                    continue

                st = trades_df.filter(pl.col("symbol") == sym)

                # Position status for shading
                sym_df = sym_df.with_columns(pos_type=pl.lit(0))
                for t in st.iter_rows(named=True):
                    mask = (sym_df[self.ts_col] >= t["entry_ts"]) & (sym_df[self.ts_col] <= t["exit_ts"])
                    if t["shares"] > 0:
                        sym_df = sym_df.with_columns(pos_type=pl.when(mask).then(pl.lit(1)).otherwise(pl.col("pos_type")))
                    else:
                        sym_df = sym_df.with_columns(pos_type=pl.when(mask).then(pl.lit(-1)).otherwise(pl.col("pos_type")))

                # Cumulative PnL for this symbol
                sym_trades = st.sort("exit_ts")
                pnl_series = sym_trades.select([
                    pl.col("exit_ts").alias("ts"),
                    pl.col("pnl")
                ]).group_by("ts").agg(pl.col("pnl").sum()).sort("ts")
                
                sym_df = sym_df.join(pnl_series, left_on=self.ts_col, right_on="ts", how="left")
                sym_df = sym_df.with_columns(cum_pnl = pl.col("pnl").fill_null(0).cum_sum())

                ts_vals = sym_df[self.ts_col].to_numpy()
                price_vals = sym_df[self.price_col].to_numpy()
                pos_vals = sym_df["pos_type"].to_numpy()
                cum_pnl_vals = sym_df["cum_pnl"].to_numpy()

                ax.plot(ts_vals, price_vals, color="black", alpha=0.3, label=f"{sym} Price")
                ax.set_yscale("log")
                ax.set_ylabel(f"{sym} Price")

                # Overlay PnL
                ax_pnl = ax.twinx()
                ax_pnl.plot(ts_vals, cum_pnl_vals, color="blue", linewidth=1.5, alpha=0.5, label="Cum PnL")
                ax_pnl.set_ylabel("PnL")

                # Markers
                long_t = st.filter(pl.col("shares") > 0)
                if not long_t.is_empty():
                    ax.scatter(long_t["entry_ts"], long_t["entry_price"], marker="^", color="green", s=40, zorder=5)
                    ax.scatter(long_t["exit_ts"], long_t["exit_price"], marker="v", color="darkred", s=40, zorder=5)

                short_t = st.filter(pl.col("shares") < 0)
                if not short_t.is_empty():
                    ax.scatter(short_t["entry_ts"], short_t["entry_price"], marker="v", color="orange", s=40, zorder=5)
                    ax.scatter(short_t["exit_ts"], short_t["exit_price"], marker="^", color="blue", s=40, zorder=5)

                # Shading
                for j in range(len(sym_df) - 1):
                    color = None
                    if pos_vals[j] == 1: color = "green"
                    elif pos_vals[j] == -1: color = "red"
                    if color:
                        ax.axvspan(ts_vals[j], ts_vals[j+1], color=color, alpha=0.1)

                ax.set_title(f"{sym} ({group_name})")

            plt.tight_layout()
            plt.show()

        return final_report
