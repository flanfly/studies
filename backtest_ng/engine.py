"""Back test engine implementation. backtest() and associated type and functions."""

from abc import ABC, abstractmethod
import polars as pl
import datetime as dt
import numpy as np

import matplotlib.pyplot as plt

import sys

from tqdm import tqdm
from dataclasses import dataclass
import functools as fc
import itertools as it

from typing import Tuple, Dict

from . import (
    AlphaModel,
    RiskModel,
    PortfolioModel,
    ExecutionModel,
    Universe,
    Position,
    Order,
    Trade,
    NoRisk,
    EqualWeight,
    Simple,
    Portfolio,
)

eps = sys.float_info.epsilon


class Backtest:
    def __init__(
        self,
        universe: Universe,
        alpha: AlphaModel | list[AlphaModel],
        risk: RiskModel | list[RiskModel] = NoRisk(),
        portfolio: PortfolioModel = EqualWeight(),
        execution: ExecutionModel = Simple(),
        period: int | str = "30d",
        benchmark=None,
        title="Strategy",
    ):
        from pytimeparse.timeparse import timeparse

        self.universe = universe
        self.alpha = alpha if isinstance(alpha, list) else [alpha]
        self.portfolio = portfolio
        self.risk = risk if isinstance(risk, list) else [risk]
        self.execution = execution
        self.period = (
            timeparse(period.lower().strip())
            if isinstance(period, str)
            else dt.timedelta(days=period)
        )
        self.benchmark = benchmark
        self.title = title

        self.trades: list[Trade] = []

        assert universe.valid_period(self.period)

    def run(self, initial_equity: float = 1.0) -> None:
        self.initial_equity = initial_equity
        self.history = []

        folio = Portfolio(cash=float(initial_equity), positions=[], working=[])
        stamps = self.universe.timestamps()

        last_rebalance = None
        history_frag = []
        trades_frag = []
        signals: list = []
        targets: list = []
        orders: list = []

        for now in tqdm(stamps):
            u = self.universe.until(now)
            prices = u.prices(now)

            rebalance = last_rebalance is None or now - last_rebalance >= self.period
            if rebalance:
                # NOTE — same-bar assumption: `u` includes the current bar
                # (universe.until uses <=).  Alpha signals are therefore
                # generated from the same close price used for execution.
                # This is only realistic if you can trade at the closing
                # auction after the signal is known.  To avoid the bias,
                # pass universe.until(prev_bar) to alpha models and
                # universe.until(now) to the execution model.

                # alpha generation
                signals = list(it.chain.from_iterable([a(u) for a in self.alpha]))

                # portfolio construction
                targets = self.portfolio(u, signals, folio)

                # risk management
                adj = fc.reduce(lambda t, r: r(u, t, folio), self.risk, targets)

                # execution
                orders = self.execution(u, adj, folio)

                # trade recording
                folio, closed = folio.execute_orders(now, prices, orders)
                for t in closed:
                    pnl = (
                        (t.exit_price - t.entry_price) * t.shares
                        - t.entry_fee
                        - t.exit_fee
                    )
                    ret = (
                        (t.exit_price - t.entry_price) / t.entry_price
                        if t.shares > 0
                        else (t.entry_price - t.exit_price) / t.entry_price
                    )
                    trades_frag.append(
                        {
                            "entry": t.entry_ts,
                            "exit": t.exit_ts,
                            "symbol": t.symbol,
                            "price": t.entry_price,
                            "shares": t.shares,
                            "pnl": pnl,
                            "ret": ret,
                        }
                    )
                last_rebalance = now

            # mark portfolio to market
            l = [prices.get(p.symbol, 0.0) * p.shares for p in folio.longs()]
            s = [prices.get(p.symbol, 0.0) * p.shares for p in folio.shorts()]

            history_frag.append(
                {
                    "ts": now,
                    "cash": folio.cash,
                    "long_notational": sum(l),
                    "short_notational": sum(s),
                    "long_positions": len(l),
                    "short_positions": len(s),
                    "fees": sum([o.fee for o in orders]),
                    "rebalance": rebalance,
                    "orders": len(orders),
                    "signals": len(signals),
                    "targets": len(targets),
                }
            )

            equity = folio.cash + sum(l) + sum(s)
            if equity < eps:
                print(f"Broke at {now} due to low equity: {equity}")
                break

        self.history = pl.DataFrame(
            history_frag,
            schema={
                "ts": pl.Datetime,
                "cash": pl.Float64,
                "long_notational": pl.Float64,
                "short_notational": pl.Float64,
                "long_positions": pl.Float64,
                "short_positions": pl.Float64,
                "fees": pl.Float64,
                "rebalance": pl.Boolean,
                "orders": pl.Int32,
                "signals": pl.Int32,
                "targets": pl.Int32,
            },
        )
        self.trades = pl.DataFrame(
            trades_frag,
            schema={
                "entry": pl.Datetime,
                "exit": pl.Datetime,
                "symbol": pl.String,
                "price": pl.Float64,
                "shares": pl.Float64,
                "pnl": pl.Float64,
                "ret": pl.Float64,
            },
        )

    # columns: entry ts, symbol, shares (negative for short), exit ts
    def live(self, equity: float) -> pl.DataFrame:
        """
        Compute the portfolio for paper trading using the last day's data.
        Runs alpha and portfolio construction models with an empty starting
        folio and returns the resulting positions as a DataFrame.

        Returns
        -------
        pl.DataFrame with columns: symbol, shares, position_value, entry_ts, exit_ts, entry_price
        """

        today = self.universe.df()[self.universe.timestamp_col()].max()
        df = self.universe.df().filter(pl.col(self.universe.timestamp_col()) == today)
        prices = self.universe.prices(today)

        folio = Portfolio(cash=equity, positions=[], working=[])
        signals = it.chain.from_iterable([a(self.universe) for a in self.alpha])
        targets = fc.reduce(
            lambda t, r: r(self.universe, t, folio),
            self.risk,
            self.portfolio(self.universe, list(signals), folio),
        )

        schema = {
            "symbol": pl.Utf8,
            "shares": pl.Float64,
            "position_value": pl.Float64,
            "entry_ts": pl.Datetime,
            "exit_ts": pl.Datetime,
            "entry_price": pl.Float64,
        }
        frag = []
        for target in targets:
            price = prices.get(target.symbol)
            if price is None or abs(target.weight) <= eps:
                continue

            exit_ts = today + self.period
            position_value = target.weight * equity
            frag.append(
                {
                    "symbol": target.symbol,
                    "shares": position_value / price,
                    "position_value": position_value,
                    "entry_ts": today,
                    "exit_ts": exit_ts,
                    "entry_price": price,
                }
            )

        if len(frag) == 0:
            return pl.DataFrame(schema=schema)
        else:
            return pl.DataFrame(frag, schema=schema)

    def report(self, plot="brief") -> pl.DataFrame:
        bench = (
            self.universe.df()
            .filter(pl.col(self.universe.symbol_col()) == self.benchmark)
            .select(
                ts=self.universe.timestamp_col(),
                price=self.universe.price_col(),
            )
        )

        if plot == "brief" or plot == True:
            _plot_timeseries(self.history, bench, self.title)
            _plot_trades(self.trades, self.title)

        return _trade_statistics(
            self.trades, self.history, bench, self.title, self.universe
        )


def _trade_statistics(
    trades: pl.DataFrame,
    history: pl.DataFrame,
    benchmark: pl.DataFrame,
    title: str,
    universe,
) -> pl.DataFrame:
    import great_tables as gt

    eq_df = history.select(
        ts=pl.col("ts"),
        equity=pl.col("cash") + pl.col("long_notational") + pl.col("short_notational"),
        fees=pl.col("fees"),
    )

    has_bench = benchmark.height > 0
    if has_bench:
        bm = benchmark.select(ts=pl.col("ts"), bench_price=pl.col("price"))
        combined = eq_df.join(bm, on="ts", how="inner").sort("ts")
        combined = combined.with_columns(
            s_ret=pl.col("equity").pct_change(),
            b_ret=pl.col("bench_price").pct_change(),
        ).drop_nulls()
        s_ret = combined["s_ret"].to_numpy()
        b_ret = combined["b_ret"].to_numpy()
        excess = s_ret - b_ret
        eq = combined["equity"].to_numpy()
        bp = combined["bench_price"].to_numpy()
    else:
        eq_df = eq_df.with_columns(s_ret=pl.col("equity").pct_change()).drop_nulls()
        s_ret = eq_df["s_ret"].to_numpy()
        eq = eq_df["equity"].to_numpy()
        b_ret = None
        bp = None

    n_days = len(s_ret)
    n_years = n_days / 365.25

    def _sharpe(r):
        return float(np.sqrt(365.25) * np.mean(r) / np.std(r)) if np.std(r) > 0 else 0.0

    def _sortino(r):
        d = r[r < 0]
        if len(d) < 2 or np.std(d) == 0:
            return 0.0
        return float(np.sqrt(365.25) * np.mean(r) / np.std(d))

    def _cagr(v):
        if v[0] <= 0 or v[-1] <= 0:
            return float("nan")
        return float((v[-1] / v[0]) ** (1.0 / n_years) - 1.0)

    def _maxdd(v):
        peak = np.maximum.accumulate(v)
        return float(np.min(v / peak - 1.0))

    def _ir(exc):
        te = np.std(exc)
        return float(np.sqrt(365.25) * np.mean(exc) / te) if te > 0 else 0.0

    s_sharpe = _sharpe(s_ret)
    s_sortino = _sortino(s_ret)
    s_cagr = _cagr(eq)
    s_mdd = _maxdd(eq)
    total_fees = history["fees"].sum()
    fees_pct = float(total_fees / eq[0]) if eq[0] > 0 else float("nan")

    if has_bench:
        b_sharpe = _sharpe(b_ret)
        b_sortino = _sortino(b_ret)
        b_cagr = _cagr(bp)
        b_mdd = _maxdd(bp)
        s_ir = _ir(excess)
    else:
        b_sharpe = b_sortino = b_cagr = b_mdd = float("nan")
        s_ir = float("nan")

    # Half-Kelly criterion: μ / (2σ²) on daily returns
    s_var = float(np.var(s_ret))
    s_half_kelly = float(np.mean(s_ret) / (2 * s_var)) if s_var > 0 else 0.0
    if has_bench:
        b_var = float(np.var(b_ret))
        b_half_kelly = float(np.mean(b_ret) / (2 * b_var)) if b_var > 0 else 0.0
    else:
        b_half_kelly = float("nan")

    if trades.height > 0:
        win_rate = float((trades["ret"] > 0).mean())
        avg_ret = float(trades["ret"].mean())
        ret_std = float(trades["ret"].std())

        # MAE / MFE from daily close prices between entry and exit.
        prices_df = universe.df().select(
            ts=universe.timestamp_col(),
            symbol=universe.symbol_col(),
            close=universe.price_col(),
        )
        # Pre-index by symbol for fast lookup.
        prices_by_sym = {}
        for sym in prices_df["symbol"].unique().to_list():
            pdf = prices_df.filter(pl.col("symbol") == sym).sort("ts")
            prices_by_sym[sym] = (
                pdf["ts"].to_numpy(),
                pdf["close"].to_numpy(),
            )

        mae_vals = []
        mfe_vals = []
        for t in trades.iter_rows(named=True):
            sym = t["symbol"]
            if sym not in prices_by_sym:
                continue
            tss, closes = prices_by_sym[sym]
            entry_ts_ns = np.datetime64(t["entry"])
            exit_ts_ns = np.datetime64(t["exit"])
            mask = (tss > entry_ts_ns) & (tss <= exit_ts_ns)
            path = closes[mask]
            if len(path) == 0:
                continue
            is_short = t["shares"] < 0
            entry_px = t["price"]
            if is_short:
                # Adverse = price rallies → entry - max is negative.
                # Favorable = price drops → entry - min is positive.
                mae = (entry_px - float(path.max())) / entry_px
                mfe = (entry_px - float(path.min())) / entry_px
            else:
                # Adverse = price drops → min - entry is negative.
                # Favorable = price rallies → max - entry is positive.
                mae = (float(path.min()) - entry_px) / entry_px
                mfe = (float(path.max()) - entry_px) / entry_px
            mae_vals.append(mae)
            mfe_vals.append(mfe)

        mae_mean = float(np.mean(mae_vals)) if mae_vals else float("nan")
        mae_std = float(np.std(mae_vals)) if mae_vals else float("nan")
        mfe_mean = float(np.mean(mfe_vals)) if mfe_vals else float("nan")
        mfe_std = float(np.std(mfe_vals)) if mfe_vals else float("nan")
    else:
        win_rate = avg_ret = ret_std = float("nan")
        mae_mean = mae_std = mfe_mean = mfe_std = float("nan")

    def _v(s, b):
        return [s, b]

    rows = {
        "Metric": [
            "CAGR",
            "Sharpe",
            "Sortino",
            "IR",
            "Half Kelly",
            "Max DD",
            "Fees",
            "return_mean",
            "return_stdev",
            "mae_mean",
            "mae_stdev",
            "mfe_mean",
            "mfe_stdev",
        ],
        "Strategy": [
            s_cagr,
            s_sharpe,
            s_sortino,
            s_ir,
            s_half_kelly,
            s_mdd,
            fees_pct,
            avg_ret,
            ret_std,
            mae_mean,
            mae_std,
            mfe_mean,
            mfe_std,
        ],
        "Benchmark": [
            b_cagr if has_bench else None,
            b_sharpe if has_bench else None,
            b_sortino if has_bench else None,
            None,
            b_half_kelly if has_bench else None,
            b_mdd if has_bench else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
    }

    stats = pl.DataFrame(rows)

    # Transpose: Strategy / Benchmark become rows, metrics become columns.
    metric_order = rows["Metric"]
    stats = stats.drop("Metric").transpose(
        include_header=True,
        header_name="Statistic",
        column_names=metric_order,
    )

    # Columns that hold percentages vs ratios.
    pct_cols = [
        "CAGR",
        "Max DD",
        "Fees",
        "return_mean",
        "return_stdev",
        "mae_mean",
        "mae_stdev",
        "mfe_mean",
        "mfe_stdev",
    ]
    ratio_cols = ["Sharpe", "Sortino", "IR", "Half Kelly"]

    tbl = (
        gt.GT(stats, rowname_col="Statistic")
        .tab_header(title=title, subtitle="Strategy Statistics")
        .tab_options(
            table_background_color="#FFFFFF",
            table_font_color="#222222",
            heading_background_color="#FFFFFF",
            heading_title_font_weight="bold",
            column_labels_background_color="#F8F8F8",
            column_labels_font_weight="bold",
            stub_background_color="#FFFFFF",
            stub_font_weight="bold",
            row_striping_background_color="#FFFFFF",
            row_striping_include_table_body=True,
            row_striping_include_stub=True,
            grand_summary_row_background_color="#FFFFFF",
            source_notes_background_color="#FFFFFF",
            data_row_padding="6px",
            data_row_padding_horizontal="8px",
            table_border_top_color="#CCCCCC",
            table_border_bottom_color="#CCCCCC",
            heading_border_bottom_color="#CCCCCC",
            column_labels_border_top_color="#CCCCCC",
            column_labels_border_bottom_color="#CCCCCC",
            table_body_hlines_color="#E0E0E0",
        )
        .tab_spanner(label="Performance", columns=["CAGR", "Max DD", "Fees"])
        .tab_spanner(label="Risk", columns=["Sharpe", "Sortino", "IR", "Half Kelly"])
        .tab_spanner(label="Return", columns=["return_mean", "return_stdev"])
        .tab_spanner(label="Adverse", columns=["mae_mean", "mae_stdev"])
        .tab_spanner(label="Favorable", columns=["mfe_mean", "mfe_stdev"])
        .cols_label(
            return_mean="Mean",
            return_stdev="St. dev.",
            mae_mean="Mean",
            mae_stdev="St. dev.",
            mfe_mean="Mean",
            mfe_stdev="St. dev.",
        )
        .fmt_percent(
            columns=pct_cols,
            decimals=1,
        )
        .fmt_number(
            columns=ratio_cols,
            decimals=2,
            n_sigfig=0,
        )
        .tab_style(
            style=gt.style.fill("#FFFFFF"),
            locations=gt.loc.body(),
        )
        .sub_missing(
            missing_text="—",
        )
    )

    # Bold the better value in each column (higher-is-better unless noted).
    # Only CAGR, Sharpe, Sortino, and Max DD get the bold treatment.
    higher_better = {"CAGR", "Sharpe", "Sortino", "Half Kelly"}
    lower_better = {"Max DD"}  # closer to zero = better → compare |v|

    strategy_vals = stats.row(0, named=True)
    benchmark_vals = stats.row(1, named=True)

    for col in stats.columns:
        if col == "Statistic":
            continue
        sv = strategy_vals[col]
        bv = benchmark_vals[col]
        if sv is None and bv is None:
            continue
        if col in higher_better:
            best = (
                "Strategy"
                if (sv is not None and (bv is None or sv > bv))
                else "Benchmark"
            )
        elif col in lower_better:
            best = (
                "Strategy"
                if (sv is not None and (bv is None or abs(sv) < abs(bv)))
                else "Benchmark"
            )
        else:
            continue
        tbl = tbl.tab_style(
            style=gt.style.text(weight="bold"),
            locations=gt.loc.body(columns=col, rows=best),
        )

    # Green / red fill for positive / negative values in return / ratio columns.
    green_cols = {"CAGR", "Sharpe", "Sortino", "Half Kelly"}
    for row_name in ("Strategy", "Benchmark"):
        vals = strategy_vals if row_name == "Strategy" else benchmark_vals
        for col in green_cols:
            v = vals.get(col)
            if v is not None:
                if v > 0:
                    tbl = tbl.tab_style(
                        style=gt.style.fill("#E6F4EA"),
                        locations=gt.loc.body(columns=col, rows=row_name),
                    )
                elif v < 0:
                    tbl = tbl.tab_style(
                        style=gt.style.fill("#FCE8E8"),
                        locations=gt.loc.body(columns=col, rows=row_name),
                    )

    # Display the table (works in Jupyter / IPython; in plain scripts prints HTML)
    try:
        from IPython.display import display as ipy_display

        ipy_display(tbl)
    except ImportError:
        print(tbl._repr_html_())

    return stats


def _plot_timeseries(df: pl.DataFrame, benchmark: pl.DataFrame, title: str):
    num_plots = 3
    fig, axes = plt.subplots(
        num_plots,
        1,
        figsize=(12, 4 * num_plots),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    ax_eq = axes[0]
    ax_dd = axes[1]
    ax_exp = axes[2]

    df = df.with_columns(
        equity=pl.col("cash") + pl.col("long_notational") + pl.col("short_notational")
    ).with_columns(
        equity_sma60=pl.col("equity").rolling_mean(window_size=60, min_periods=1)
    )

    initial_equity = base = df["equity"].drop_nulls().first()

    # Equity Plot
    ax_eq.plot(df["ts"], df["equity"], label="Strategy Equity")

    # 60-day SMA of equity
    ax_eq.plot(
        df["ts"],
        df["equity_sma60"],
        linestyle=":",
        color=ax_eq.get_lines()[0].get_color(),
        linewidth=1.0,
        label="60-day SMA",
    )

    ax_eq.set_ylabel("Equity")
    ax_eq.set_yscale("log")

    if benchmark.height > 0:
        # Normalisation factor — first non-null benchmark price
        base = benchmark["price"].drop_nulls().first()
        dfhist = benchmark.with_columns(
            bench_norm=pl.col("price") / base * initial_equity,
        ).with_columns(
            bench_sma60=pl.col("bench_norm").rolling_mean(window_size=60, min_periods=1)
        )

        ax_eq.plot(
            dfhist["ts"],
            dfhist["bench_norm"],
            color="dimgray",
            linestyle="-",
            linewidth=1.0,
            label="Benchmark",
        )
        # 60-day SMA of benchmark
        ax_eq.plot(
            dfhist["ts"],
            dfhist["bench_sma60"],
            linestyle=":",
            color="dimgray",
            linewidth=1.0,
            label="Benchmark 60-day SMA",
        )

    ax_eq.set_title(f"{title} — Performance & Symbol Details")
    ax_eq.legend(loc="upper left")

    # Drawdown Plot
    drawdown = (df["equity"] / df["equity"].cum_max() - 1).to_numpy()
    ax_dd.fill_between(df["ts"], drawdown, 0, color="crimson", alpha=0.4)
    ax_dd.plot(df["ts"], drawdown, color="crimson", linewidth=0.6)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.yaxis.set_major_formatter(
        plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=0)
    )
    ax_dd.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    # Exposure Plot (Long/Short/Net)
    ax_exp.fill_between(
        df["ts"],
        df["long_positions"],
        color="green",
        alpha=0.3,
        label="Longs",
    )
    ax_exp.fill_between(
        df["ts"],
        -df["short_positions"],
        color="red",
        alpha=0.3,
        label="Shorts",
    )
    net_bias = df["long_positions"] - df["short_positions"]
    ax_exp.plot(
        df["ts"],
        net_bias,
        color="black",
        linewidth=1,
        label="Net Bias",
    )
    ax_exp.axhline(0, color="black", linestyle="-", alpha=0.2)
    ax_exp.set_ylabel("Exposure")
    ax_exp.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


# Vertical markers — positions in log space, labels in simple returns.
# Each line gets an inline annotation at the top of the axes.
def _vline(ax, xpos, color, lw, ls, top_label, val_label):
    ax.axvline(xpos, color=color, linewidth=lw, linestyle=ls)
    # two-line label: descriptor on top, value below
    ax.text(
        xpos,
        1.015,
        f"{top_label}\n{val_label}",
        transform=ax.get_xaxis_transform(),  # x in data, y in axes coords
        ha="center",
        va="bottom",
        fontsize=7,
        color="black",
        clip_on=False,
        linespacing=1.3,
    )


def _plot_trades(df: pl.DataFrame, title: str):
    import matplotlib.ticker as mticker
    from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis
    from scipy.stats import johnsonsu

    if df.height == 0:
        return

    # Use all non-null trade returns — axis caps keep the plot readable.
    filtered = df.filter(pl.col("ret").is_not_null())
    # Clip at -99.9999 % so log1p never hits -inf for complete-loss trades.
    # The clipped trades still contribute to the histogram and dollar-weighted
    # stats; the axis cap at -85 % makes the left tail readable.
    ret_clipped = np.clip(filtered["ret"].to_numpy(), -1 + eps, None)
    log_returns = np.log1p(ret_clipped)
    # Dollar weights: |shares| * entry_price (position notional)
    pos_value = filtered["price"].to_numpy() * filtered["shares"].abs().to_numpy()

    w = pos_value / pos_value.sum()  # normalised dollar weights
    # Dollar-weighted moments in log space
    mu_log = float(np.average(log_returns, weights=w))
    sig_log = float(np.sqrt(np.average((log_returns - mu_log) ** 2, weights=w)))
    skewness = float(sp_skew(log_returns))
    kurt_excess = float(sp_kurtosis(log_returns, fisher=True))

    # Fit Johnson SU in log-return space — 4 parameters to match
    # all four moments while staying a valid PDF.
    js_params = johnsonsu.fit(log_returns)

    # X grid — use clipped log-return range, then cap the view.
    data_lo = log_returns.min()
    data_hi = log_returns.max()
    x_lo = max(mu_log - 4 * sig_log, data_lo)
    x_hi = min(mu_log + 4 * sig_log, data_hi)
    x_log = np.linspace(x_lo, x_hi, 800)

    fig2, ax_ret = plt.subplots(figsize=(12, 5))

    # Johnson SU — filled gray area drawn first so histogram sits on top
    fitted_pdf = johnsonsu.pdf(x_log, *js_params)
    mu_simple = float(np.expm1(mu_log))
    lo_simple = float(np.expm1(mu_log - sig_log))
    hi_simple = float(np.expm1(mu_log + sig_log))
    ax_ret.fill_between(
        x_log,
        fitted_pdf,
        color="lightgray",
        alpha=0.8,
        linewidth=0,
        label=f"Johnson SU fit (equal-wt)  μ={mu_simple:.2%}  σ⁻={lo_simple:.2%}  σ⁺={hi_simple:.2%}",
    )

    # Histogram in log-return space — dollar-weighted bars
    # sit on top of the (equal-weighted) Johnson SU reference curve.
    ax_ret.hist(
        log_returns,
        bins=min(400, max(160, len(log_returns) // 2)),
        weights=pos_value,
        density=True,
        alpha=0.5,
        color="steelblue",
        label="Trades ($-weighted)",
    )

    ax_ret.axvline(np.log1p(0), color="dimgray", linewidth=0.9, linestyle="-")
    _vline(
        ax_ret,
        mu_log,
        "royalblue",
        1.5,
        "--",
        "mean",
        f"{mu_simple:.2%}",
    )
    # Weighted median — silent line, no label (often overlaps mean)
    order = np.argsort(log_returns)
    cum_w = np.cumsum(w[order])
    idx = np.searchsorted(cum_w, 0.5)
    median_log = float(log_returns[order][idx])
    ax_ret.axvline(median_log, color="teal", linewidth=1.0, linestyle="--")
    _vline(
        ax_ret,
        mu_log - sig_log,
        "darkorange",
        1.2,
        ":",
        "−1σ",
        f"{lo_simple:.2%}",
    )
    _vline(
        ax_ret,
        mu_log + sig_log,
        "darkorange",
        1.2,
        ":",
        "+1σ",
        f"{hi_simple:.2%}",
    )
    _vline(
        ax_ret,
        mu_log - 2 * sig_log,
        "firebrick",
        1.0,
        ":",
        "−2σ",
        f"{float(np.expm1(mu_log - 2 * sig_log)):.2%}",
    )
    _vline(
        ax_ret,
        mu_log + 2 * sig_log,
        "firebrick",
        1.0,
        ":",
        "+2σ",
        f"{float(np.expm1(mu_log + 2 * sig_log)):.2%}",
    )

    # Skew / kurtosis annotation (computed on log returns)
    # Cap x-axis symmetric around 0 so that ±2σ are inside the plot.
    half_span = max(
        abs(mu_log - 2 * sig_log),
        abs(mu_log + 2 * sig_log),
    )
    # Floor in case sigma is degenerate; add 5 % padding.
    half_span = max(half_span, data_lo * 0.5, -data_hi * 0.5, 0.05) * 1.05
    ax_ret.set_xlim(-half_span, half_span)
    ax_ret.text(
        0.98,
        0.97,
        f"skew = {skewness:.3f}\nexcess kurt = {kurt_excess:.3f}",
        transform=ax_ret.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    # X-axis ticks: choose nice simple-return levels, place them
    # at their log-return positions, label as simple percentages.
    nice_simple = [
        -0.9,
        -0.75,
        -0.5,
        -0.25,
        0.0,
        0.25,
        0.5,
        1.0,
        2.0,
        3.0,
        5.0,
    ]
    tick_log = [np.log1p(s) for s in nice_simple if x_lo <= np.log1p(s) <= x_hi]
    ax_ret.set_xticks(tick_log)
    ax_ret.set_xticklabels([f"{np.expm1(t):.0%}" for t in tick_log])

    ax_ret.set_xlabel(
        "Trade return, simple (net of fees · $-weighted · axis centred at 0, ±2σ inside)"
    )
    ax_ret.set_ylabel("Density (log-return space)")
    ax_ret.set_title(
        f"{title} — Trade Returns Distribution (n={len(log_returns)} · $-weighted)",
        pad=28,
    )
    plt.tight_layout()
    plt.show()


#        df = self.history.with_columns(
#            equity=pl.col("cash")
#            + pl.col("long_notational")
#            - pl.col("short_notational"),
#        ).with_columns(
#            returns=pl.col("equity").pct_change().fill_null(0),
#        )
#
#        # Benchmark returns for IR calculation
#        if self.benchmark is not None:
#            df = df.join(
#                self.universe.df.filter(
#                    pl.col(self.universe.symbol_col) == self.benchmark
#                )
#                .select(
#                    ts=pl.col(self.universe.timestamp_col),
#                    bench_price=pl.col(self.universe.price_col),
#                )
#                .sort("ts"),
#                on="ts",
#                how="left",
#            )
#
#        else:
#            df = df.with_columns(
#                bench_price=pl.lit(None).cast(pl.Float64),
#            )
#
#        df = df.with_columns(
#            bench_returns=pl.col("bench_price").pct_change().fill_null(0)
#        ).with_columns(active_returns=pl.col("returns") - pl.col("bench_returns"))
#
#        # Trades stats
#        if self.trades:
#            trades_df = pl.from_dicts([t.__dict__ for t in self.trades])
#            trades_df = trades_df.with_columns(
#                year=pl.col("exit_ts").dt.year(),
#                pnl=(pl.col("exit_price") - pl.col("entry_price")) * pl.col("shares"),
#                pnl_pct=pl.when(pl.col("shares") > 0)
#                .then(pl.col("exit_price") / pl.col("entry_price") - 1)
#                .otherwise(1 - pl.col("exit_price") / pl.col("entry_price")),
#                mfe=pl.when(pl.col("shares") > 0)
#                .then(pl.col("high") / pl.col("entry_price") - 1)
#                .otherwise(1 - pl.col("low") / pl.col("entry_price")),
#                mae=pl.when(pl.col("shares") > 0)
#                .then(pl.col("low") / pl.col("entry_price") - 1)
#                .otherwise(1 - pl.col("high") / pl.col("entry_price")),
#            )
#            trade_yearly = trades_df.group_by("year").agg(
#                [
#                    pl.col("mfe").mean().alias("mfe"),
#                    pl.col("mfe").std().alias("mfe_std"),
#                    pl.col("mae").mean().alias("mae"),
#                    pl.col("mae").std().alias("mae_std"),
#                    ((pl.col("pnl") > 0).sum() / pl.col("pnl").count()).alias(
#                        "win_rate"
#                    ),
#                ]
#            )
#        else:
#            trade_yearly = pl.DataFrame(
#                schema={
#                    "year": pl.Int32,
#                    "mfe": pl.Float64,
#                    "mfe_std": pl.Float64,
#                    "mae": pl.Float64,
#                    "mae_std": pl.Float64,
#                    "win_rate": pl.Float64,
#                }
#            )
#
#        # Yearly equity stats
#        history_df = history_df.with_columns(
#            year=pl.col("ts").dt.year(),
#            neg_returns=pl.when(pl.col("returns") < 0)
#            .then(pl.col("returns"))
#            .otherwise(0),
#            # Drawdown from the global all-time high (not reset per year)
#            drawdown=(pl.col("equity") / pl.col("equity").cum_max() - 1),
#        )
#
#        if self.benchmark is not None:
#            history_df = history_df.with_columns(
#                bench_neg_returns=pl.when(pl.col("bench_returns") < 0)
#                .then(pl.col("bench_returns"))
#                .otherwise(0),
#                bench_drawdown=(
#                    pl.col("bench_price") / pl.col("bench_price").cum_max() - 1
#                ),
#            )
#
#        agg_exprs = [
#            # CAGR for the year
#            (
#                (pl.col("equity").last() / pl.col("equity").first())
#                ** (
#                    1
#                    / (
#                        (pl.col("ts").last() - pl.col("ts").first()).dt.total_days()
#                        / 365.25
#                    )
#                )
#                - 1
#            ).alias("cagr"),
#            (pl.col("returns").mean() * 252).alias("ann_return"),
#            (pl.col("returns").std() * (252**0.5)).alias("ann_std"),
#            (pl.col("neg_returns").std() * (252**0.5)).alias("downside_std"),
#            (pl.col("active_returns").mean() * 252).alias("active_return_ann"),
#            (pl.col("active_returns").std() * (252**0.5)).alias("tracking_error"),
#            pl.col("drawdown").min().alias("maxdd"),
#            pl.col("fees").sum().alias("fees"),
#        ]
#
#        if self.benchmark is not None:
#            agg_exprs.extend(
#                [
#                    (
#                        (pl.col("bench_price").last() / pl.col("bench_price").first())
#                        ** (
#                            1
#                            / (
#                                (
#                                    pl.col("ts").last() - pl.col("ts").first()
#                                ).dt.total_days()
#                                / 365.25
#                            )
#                        )
#                        - 1
#                    ).alias("bench_cagr"),
#                    (pl.col("bench_returns").mean() * 252).alias("bench_ann_return"),
#                    (pl.col("bench_returns").std() * (252**0.5)).alias("bench_ann_std"),
#                    (pl.col("bench_neg_returns").std() * (252**0.5)).alias(
#                        "bench_downside_std"
#                    ),
#                    pl.col("bench_drawdown").min().alias("bench_maxdd"),
#                ]
#            )
#
#        yearly_stats = (
#            history_df.group_by("year")
#            .agg(agg_exprs)
#            .with_columns(
#                sharpe=pl.when(pl.col("ann_std") > EPSILON)
#                .then(pl.col("ann_return") / pl.col("ann_std"))
#                .otherwise(pl.lit(None).cast(pl.Float64)),
#                sortino=pl.when(pl.col("downside_std") > EPSILON)
#                .then(pl.col("ann_return") / pl.col("downside_std"))
#                .otherwise(pl.lit(None).cast(pl.Float64)),
#                ir=pl.when(pl.col("tracking_error") > EPSILON)
#                .then(pl.col("active_return_ann") / pl.col("tracking_error"))
#                .otherwise(pl.lit(None).cast(pl.Float64)),
#            )
#        )
#
#        if self.benchmark is not None:
#            yearly_stats = yearly_stats.with_columns(
#                bench_sharpe=pl.when(pl.col("bench_ann_std") > EPSILON)
#                .then(pl.col("bench_ann_return") / pl.col("bench_ann_std"))
#                .otherwise(pl.lit(None).cast(pl.Float64)),
#                bench_sortino=pl.when(pl.col("bench_downside_std") > EPSILON)
#                .then(pl.col("bench_ann_return") / pl.col("bench_downside_std"))
#                .otherwise(pl.lit(None).cast(pl.Float64)),
#            )
#
#        final_report = yearly_stats.join(trade_yearly, on="year", how="left").sort(
#            "year"
#        )
#
#        if self.benchmark is not None:
#            # Create two separate dataframes: one for Strategy and one for Benchmark
#            # Metrics we want to compare
#            comparison_cols = [
#                "cagr",
#                "ann_return",
#                "ann_std",
#                "maxdd",
#                "sharpe",
#                "sortino",
#            ]
#
#            # 1. Strategy DataFrame
#            strat_report = final_report.select(
#                [pl.col("year"), pl.lit("Strategy").alias("src")]
#                + [pl.col(c) for c in comparison_cols if c in final_report.columns]
#                + [
#                    pl.col(c)
#                    for c in [
#                        "ir",
#                        "fees",
#                        "mfe",
#                        "mfe_std",
#                        "mae",
#                        "mae_std",
#                        "win_rate",
#                    ]
#                    if c in final_report.columns
#                ]
#            )
#
#            # 2. Benchmark DataFrame
#            bench_report = final_report.select(
#                [pl.col("year"), pl.lit("Benchmark").alias("src")]
#                + [
#                    pl.col(f"bench_{c}").alias(c)
#                    for c in comparison_cols
#                    if f"bench_{c}" in final_report.columns
#                ]
#            )
#
#            # 3. Combine and sort
#            final_report = pl.concat([strat_report, bench_report], how="diagonal").sort(
#                ["year", "src"], descending=[False, True]
#            )
#        else:
#            final_report = final_report.with_columns(src=pl.lit("Strategy"))
#            # Reorder columns to put src first after year
#            cols = final_report.columns
#            if "year" in cols:
#                cols.remove("year")
#                cols.remove("src")
#                final_report = final_report.select(["year", "src"] + cols)
#
#        if plot:
#
#
#        return final_report
#
#
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
