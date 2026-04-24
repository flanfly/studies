from abc import ABC, abstractmethod
import polars as pl
import datetime as dt

from tqdm import tqdm
from dataclasses import dataclass

from typing import Tuple


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
        if not signals:
            return []

        today = df[self.ts_col].max()
        prices = dict(
            df.filter(pl.col(self.ts_col) == today)
            .select([self.symbol_col, self.price_col])
            .iter_rows()
        )

        # liquidate all positions
        orders = [Order(pos.symbol, -pos.shares) for pos in folio]

        if equity <= 0:
            return orders

        target_weight = equity / len(signals)
        for signal in signals:
            price = prices.get(signal.symbol)
            if price is not None and price > 0:
                orders.append(
                    Order(
                        signal.symbol,
                        target_weight / price * (1 if signal.bullish else -1),
                    )
                )
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
        if not signals:
            return []

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

        # liquidate all positions
        orders = [Order(pos.symbol, -pos.shares) for pos in folio]

        total_vol = sum(vols.get(signal.symbol, 0) for signal in signals)
        if total_vol <= 0:
            return orders

        orders += [
            Order(
                signal.symbol,
                (vols.get(signal.symbol, 0) / total_vol)
                * equity
                / prices.get(signal.symbol, 1)
                * (1 if signal.bullish else -1),
            )
            for signal in signals
            if prices.get(signal.symbol, 0) > 0
        ]
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
        initial_equity=1,
        benchmark=None,
        timestamp_col="ts",
        symbol_col="symbol",
        price_col="close",
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
        self.initial_equity = initial_equity
        self.benchmark = benchmark
        self.ts_col = timestamp_col
        self.symbol_col = symbol_col
        self.price_col = price_col
        self.fee = fee

        self.trades: list[Trade] = []

    def run(self):
        days = self.df.select(pl.col(self.ts_col).unique().sort()).to_series().to_list()

        self.folio = []
        self.cash = float(self.initial_equity)
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

            if equity <= 1e-6:
                print(f"Broke at {day} due to low equity: {equity}")
                break

            signals = self.alpha(dfnow)
            if last_rebalance is None or (day - last_rebalance).days >= self.period:
                orders = self.portfolio(dfnow, signals, self.folio, equity)
                self.folio, fees = self._execute_orders(self.folio, orders, prices, day)
                day_fees += fees
                last_rebalance = day

            # Final equity for the day after all trades and fees
            equity = self.cash + sum(
                pos.shares * prices.get(pos.symbol, pos.open) for pos in self.folio
            )
            self.history.append({"ts": day, "equity": equity, "fees": day_fees})

        return self

    def _execute_orders(
        self, folio: list[Position], orders: list[Order], prices: dict, ts: dt.datetime
    ) -> Tuple[list[Position], float]:
        new_folio_dict = {pos.symbol: pos for pos in folio}
        total_fees = 0.0

        for order in orders:
            price = prices.get(order.symbol)
            if price is None:
                continue

            fee = abs(order.shares * price) * self.fee
            total_fees += fee
            self.cash -= (order.shares * price + fee)

            if order.symbol in new_folio_dict:
                pos = new_folio_dict[order.symbol]
                new_shares = pos.shares + order.shares

                if (pos.shares > 0 and new_shares <= 0) or (
                    pos.shares < 0 and new_shares >= 0
                ):
                    # closed or reversed
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
                    if new_shares != 0:
                        new_folio_dict[order.symbol] = Position(
                            order.symbol, new_shares, price, ts, price, price
                        )
                    else:
                        del new_folio_dict[order.symbol]
                else:
                    # scaling
                    if (order.shares > 0 and pos.shares > 0) or (
                        order.shares < 0 and pos.shares < 0
                    ):
                        # scaling in
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
                        # scaling out
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
                if order.shares != 0:
                    new_folio_dict[order.symbol] = Position(
                        order.symbol, order.shares, price, ts, price, price
                    )
        return list(new_folio_dict.values()), total_fees

    def report(self, plot=False) -> pl.DataFrame:
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
                    pl.col("mae").mean().alias("mae"),
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
                    "mae": pl.Float64,
                    "win_rate": pl.Float64,
                }
            )

        # Yearly equity stats
        yearly_stats = (
            history_df.with_columns(
                year=pl.col("ts").dt.year(),
                neg_returns=pl.when(pl.col("returns") < 0)
                .then(pl.col("returns"))
                .otherwise(0),
            )
            .group_by("year")
            .agg(
                [
                    # CAGR for the year
                    (
                        (pl.col("equity").last() / pl.col("equity").first())
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
                    ).alias("cagr"),
                    (pl.col("returns").mean() * 252).alias("ann_return"),
                    (pl.col("returns").std() * (252**0.5)).alias("ann_std"),
                    (pl.col("neg_returns").std() * (252**0.5)).alias("downside_std"),
                    (pl.col("active_returns").mean() * 252).alias("active_return_ann"),
                    (pl.col("active_returns").std() * (252**0.5)).alias(
                        "tracking_error"
                    ),
                    ((pl.col("equity") / pl.col("equity").cum_max() - 1).min()).alias(
                        "maxdd"
                    ),
                    pl.col("fees").sum().alias("fees"),
                ]
            )
            .with_columns(
                sharpe=pl.col("ann_return") / pl.col("ann_std"),
                sortino=pl.col("ann_return") / pl.col("downside_std"),
                ir=pl.col("active_return_ann") / pl.col("tracking_error"),
            )
        )

        cols = [
            "year",
            "sortino",
            "sharpe",
            "ir",
            "cagr",
            "ann_std",
            "maxdd",
            "fees",
            "mfe",
            "mae",
            "win_rate",
        ]

        final_report = (
            yearly_stats.join(trade_yearly, on="year", how="left")
            .select(
                [
                    c
                    for c in cols
                    if c in yearly_stats.columns or c in trade_yearly.columns
                ]
            )
            .sort("year")
        )

        if plot:
            import matplotlib.pyplot as plt

            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(history_df["ts"], history_df["equity"], label="Strategy Equity")
            ax1.set_ylabel("Equity")
            ax1.legend(loc="upper left")

            if self.benchmark is not None:
                bench_prices = history_df["bench_price"].drop_nulls()
                if not bench_prices.is_empty():
                    bench_norm = bench_prices / bench_prices[0] * self.initial_equity
                    ax1.plot(
                        history_df.filter(pl.col("bench_price").is_not_null())["ts"],
                        bench_norm,
                        label="Benchmark",
                        linestyle="--",
                        color="gray",
                    )
                    ax1.legend(loc="upper left")

            plt.title("Equity Curve")
            plt.show()

        return final_report
