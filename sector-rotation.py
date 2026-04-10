# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% editable=true slideshow={"slide_type": ""}
# derive spx membership

import polars as pl
import numpy as np
import pandas as pd
from typing import Callable, Literal
from tqdm import tqdm

sector_etfs = [
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]


def zscore(window: int) -> Callable[[pl.Expr], pl.Expr]:
    def _zscore(val: pl.Expr) -> pl.Expr:
        return (val - val.rolling_mean(window)) / val.rolling_std(window)

    return _zscore


# yang-zhang variance estimation parameters
yz_k = 0.34
yz_win = 25

parameter_sets = [
    {"max_long": 2, "max_short": 1, "period": 30, "stop_long": .5, "stop_short": 0.3},

]

for param in parameter_sets:
    max_long = param["max_long"]
    max_short = param["max_short"]
    rebalance_period = param["period"]
    stop_long = param.get("stop_long")
    stop_short = param.get("stop_short")

    signals = {
        "mom12m": pl.col("mom12m"),
        "mom6m": pl.col("mom6m"),
        # "mom3m": pl.col("mom3m"),
        # "mom2m": pl.col("mom2m"),
        # "mom1m": pl.col("mom1m"),
        # "mom3+6+12m": pl.col("mom12m") + pl.col("mom6m") + pl.col("mom3m"),
        "mom12-1m": pl.col("mom12m") - pl.col("mom1m"),
    }

    perf = None

    for signal, expr in signals.items():
        df = (
            # uv run yf.py SPY XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY --output yf.parquet
            pl.read_parquet("yf.parquet")
            .filter(pl.col("symbol").is_in(sector_etfs))
            .select(
                date=pl.col("ts").dt.date(),
                symbol=pl.col("symbol"),
                open=pl.col("open"),
                high=pl.col("high"),
                low=pl.col("low"),
                close=pl.col("close"),
                vol=pl.col("volume"),
            )
            .sort(["symbol", "date"])
            # yz-variance estimation
            .with_columns(
                o=pl.col("open").log() - pl.col("close").shift(1).over("symbol").log(),
                u=pl.col("high").log() - pl.col("open").log(),
                d=pl.col("low").log() - pl.col("open").log(),
                c=pl.col("close").log() - pl.col("open").log(),
            )
            .with_columns(
                rs=pl.col("u") * (pl.col("u") - pl.col("c"))
                + pl.col("d") * (pl.col("d") - pl.col("c"))
            )
            .with_columns(
                var=(
                    pl.col("o").rolling_var(yz_win)
                    + yz_k * pl.col("c").rolling_var(yz_win)
                    + ((1 - yz_k) * pl.col("rs").rolling_mean(yz_win))
                ).over("symbol")
            )
            .select(["date", "symbol", "open", "high", "low", "close", "vol", "var"])
            .with_columns(
                **{
                    f"mom{n}m": pl.col("close").pct_change(21 * n).over("symbol")
                    for n in [1, 2, 3, 6, 12]
                },
                sma50d=pl.col("close").rolling_mean(50).over("symbol"),
            )
            .with_columns(score=expr)
            .with_columns(
                long_rank=pl.when(
                    # (pl.col("sma50d") < pl.col("close")) & (pl.col("score") > 0)
                    (pl.col("mom12m") > 0)
                    | (pl.col("mom6m") > 0)
                )
                .then(
                    pl.col("score").rank(descending=True).over("date")
                    / pl.len().over("date")
                )
                .otherwise(None),
                short_rank=pl.when(
                    (pl.col("mom6m") < 0)
                    & (pl.col("mom12m") < 0)
                    # (pl.col("sma50d") > pl.col("close")) & (pl.col("score") < 0)
                )
                .then(
                    pl.col("score").rank(descending=False).over("date")
                    / pl.len().over("date")
                )
                .otherwise(None),
            )
        )

        portfolio = (
            []
        )  # will store dicts: {'symbol': s, 'shares': x, 'last_close': y, 'type': 'long'|'short'}
        days_since_rebalance = rebalance_period
        perf_frag = []

        # We track cash and shares to exactly calculate daily NAV
        cash = 1.0
        portfolio_equity = 1.0

        # Load the whole parquet once outside to get Treasury prices if needed
        # Or simply read it inside if we want to ensure we have it:
        df_tlt = (
            pl.read_parquet("yf.parquet")
            .filter(pl.col("symbol") == "IEF")
            .select(
                date=pl.col("ts").dt.date(),
                symbol=pl.col("symbol"),
                close=pl.col("close"),
            )
        )

        for day in tqdm(df["date"].unique().sort().to_list()):
            df_now = df.filter(pl.col("date") == day)

            # Compute today's portfolio NAV
            # Process trailing stops for the day
            new_portfolio = []
            for pos in portfolio:
                sym = pos["symbol"]
                shares = pos["shares"]
                ptype = pos["type"]

                row = df_now.filter(pl.col("symbol") == sym)
                if len(row) == 0:
                    if sym == "IEF":
                        row_tlt = df_tlt.filter(pl.col("date") == day)
                        if len(row_tlt) > 0:
                            pos["last_close"] = row_tlt["close"][0]
                    new_portfolio.append(pos)
                    continue

                if sym != "IEF":
                    high = row["high"][0]
                    low = row["low"][0]
                    open_p = row["open"][0]
                    close = row["close"][0]

                    if ptype == "long":
                        entry_high = max(pos.get("entry_high", pos["last_close"]), high)
                        pos["entry_high"] = entry_high
                        if stop_long is not None and stop_long > 0:
                            stop_price = entry_high * (1.0 - stop_long)
                            if low <= stop_price:
                                exit_price = min(open_p, stop_price)
                                cash += shares * exit_price
                                continue
                    elif ptype == "short":
                        entry_low = min(pos.get("entry_low", pos["last_close"]), low)
                        pos["entry_low"] = entry_low
                        if stop_short is not None and stop_short > 0:
                            stop_price = entry_low * (1.0 + stop_short)
                            if high >= stop_price:
                                exit_price = max(open_p, stop_price)
                                cash += shares * exit_price
                                continue

                    pos["last_close"] = close
                    new_portfolio.append(pos)
                else:
                    pos["last_close"] = row["close"][0]
                    new_portfolio.append(pos)

            portfolio = new_portfolio

            today_value = cash
            for pos in portfolio:
                today_value += pos["shares"] * pos["last_close"]

            # Daily geometric return based on exact portfolio equity
            if portfolio_equity > 0:
                ret = (today_value / portfolio_equity) - 1.0
            else:
                ret = 0.0

            portfolio_equity = today_value

            perf_frag.append(
                pl.DataFrame(
                    {
                        "date": [day],
                        signal: [ret],
                    }
                )
            )

            # rebalance portfolio
            long_bkt = (
                df_now.filter(pl.col("long_rank").is_null().not_())
                .sort("long_rank")["symbol"]
                .to_list()
            )
            short_bkt = (
                df_now.filter(pl.col("short_rank").is_null().not_())
                .sort("short_rank", descending=False)["symbol"]
                .to_list()
            )

            bkt_len = len(long_bkt[:max_long]) + len(short_bkt[:max_short])
            yd_folio = portfolio.copy()
            portfolio = []

            if days_since_rebalance < rebalance_period:
                for pos in yd_folio:
                    sym = pos["symbol"]
                    shares = pos["shares"]
                    ptype = pos["type"]

                    if (
                        sym == "IEF"
                        or (ptype == "long" and sym in long_bkt)
                        or (ptype == "short" and sym in short_bkt)
                        or True
                    ):
                        portfolio.append(pos)
                    else:
                        # Position dropped from top ranks mid-period, liquidate it into cash
                        # Then put that cash into IEF
                        row = df_now.filter(pl.col("symbol") == sym)
                        if len(row) > 0:
                            liq_val = shares * row["close"][0]
                        else:
                            liq_val = shares * pos["last_close"]

                        cash += liq_val

                        row_tlt = df_tlt.filter(pl.col("date") == day)
                        if len(row_tlt) > 0 and cash > 0:
                            ief_close = row_tlt["close"][0]
                            ief_shares = cash / ief_close
                            portfolio.append(
                                {
                                    "symbol": "IEF",
                                    "shares": ief_shares,
                                    "last_close": ief_close,
                                    "type": "long",
                                }
                            )
                            cash = 0.0

                days_since_rebalance += 1

            elif bkt_len > 0:
                # Full rebalance
                # Target weight for each position using inverse volatility weighting
                close_dict = dict(df_now[["symbol", "close"]].iter_rows(named=False))
                var_dict = dict(df_now[["symbol", "var"]].iter_rows(named=False))

                selected_syms = long_bkt[:max_long] + short_bkt[:max_short]
                inv_vols = {}
                for s in selected_syms:
                    v = var_dict.get(s)
                    if v is not None and v > 0:
                        inv_vols[s] = 1.0 / (v**0.5)
                    else:
                        inv_vols[s] = 0.0

                total_inv_vol = sum(inv_vols.values())

                # Allocate from current equity
                # sum of absolute weights is 1.0
                # cash = equity - sum(shares * close)
                # For longs, shares = weight * equity / close
                # For shorts, shares = -weight * equity / close
                cash = portfolio_equity

                for s in long_bkt[:max_long]:
                    w = (
                        inv_vols[s] / total_inv_vol
                        if total_inv_vol > 0
                        else 1.0 / bkt_len
                    )
                    shares = (w * portfolio_equity) / close_dict[s]
                    portfolio.append(
                        {
                            "symbol": s,
                            "shares": shares,
                            "last_close": close_dict[s],
                            "type": "long",
                            "entry_high": close_dict[s],
                        }
                    )
                    cash -= shares * close_dict[s]

                for s in short_bkt[:max_short]:
                    w = (
                        inv_vols[s] / total_inv_vol
                        if total_inv_vol > 0
                        else 1.0 / bkt_len
                    )
                    shares = -(w * portfolio_equity) / close_dict[s]
                    portfolio.append(
                        {
                            "symbol": s,
                            "shares": shares,
                            "last_close": close_dict[s],
                            "type": "short",
                            "entry_low": close_dict[s],
                        }
                    )
                    cash -= (
                        shares * close_dict[s]
                    )  # subtracting a negative adds to cash

                # any remaining cash goes to IEF
                if cash > 0:
                    row_tlt = df_tlt.filter(pl.col("date") == day)
                    if len(row_tlt) > 0:
                        ief_close = row_tlt["close"][0]
                        ief_shares = cash / ief_close
                        portfolio.append(
                            {
                                "symbol": "IEF",
                                "shares": ief_shares,
                                "last_close": ief_close,
                                "type": "long",
                            }
                        )
                        cash = 0.0

                days_since_rebalance = 0
            else:
                # all to IEF
                cash = portfolio_equity
                if cash > 0:
                    row_tlt = df_tlt.filter(pl.col("date") == day)
                    if len(row_tlt) > 0:
                        ief_close = row_tlt["close"][0]
                        ief_shares = cash / ief_close
                        portfolio.append(
                            {
                                "symbol": "IEF",
                                "shares": ief_shares,
                                "last_close": ief_close,
                                "type": "long",
                            }
                        )
                        cash = 0.0
                days_since_rebalance = 0

        if perf is None:
            perf = pl.concat(perf_frag)
        else:
            perf = perf.join(pl.concat(perf_frag), on="date", how="inner")

    df_perf = perf.join(
        pl.read_parquet("yf.parquet")
        .filter(pl.col("symbol") == "SPY")
        .sort("ts")
        .select(
            date=pl.col("ts").dt.date(),
            spy_ret=pl.col("close").pct_change(),
        ),
        on="date",
        how="inner",
    ).with_columns(
        [(pl.col(c) + 1).cum_prod().alias(c) for c in signals.keys()]
        + [(pl.col("spy_ret") + 1).cum_prod().alias("spy")]
    )

    pdf = df_perf.to_pandas()
    pdf.plot(
        x="date",
        y=[*signals.keys(), "spy"],
        figsize=(15, 7),
        title=f"rebalance every {rebalance_period}d long {max_long}, short {max_short} stop {stop_long * 100}/{stop_short * 100}%",
    )
    import matplotlib.pyplot as plt

    plt.savefig(f"plot_long{max_long}_short{max_short}.png")
    plt.show()

    # Compute Summary Statistics
    print(f"\n--- Summary for rebalance {rebalance_period}d | long {max_long} | short {max_short} ---")
    
    def calc_cagr(series_cum):
        if len(series_cum) < 2: return 0.0
        n_years = len(series_cum) / 252.0
        tot_ret = series_cum.iloc[-1] / series_cum.iloc[0]
        if tot_ret <= 0: return 0.0
        return (tot_ret ** (1 / n_years)) - 1.0

    def calc_mdd(series_cum):
        roll_max = np.maximum.accumulate(series_cum)
        drawdown = series_cum / roll_max - 1.0
        return drawdown.min()

    def calc_sharpe(series_ret):
        mean = series_ret.mean() * 252.0
        std = series_ret.std() * np.sqrt(252.0)
        return mean / std if std > 0 else 0.0

    def calc_ir(series_ret, spy_ret):
        active_ret = series_ret - spy_ret
        mean = active_ret.mean() * 252.0
        std = active_ret.std() * np.sqrt(252.0)
        return mean / std if std > 0 else 0.0

    stats = []
    for sig in [*signals.keys(), 'spy']:
        s_cum = pdf[sig]
        s_ret = pdf[sig].pct_change().fillna(0.0) if sig != 'spy' else pdf['spy_ret']

        cagr = calc_cagr(s_cum)
        mdd = calc_mdd(s_cum.values)
        sharpe = calc_sharpe(s_ret)
        ir = calc_ir(s_ret, pdf['spy_ret']) if sig != 'spy' else 0.0
        
        stats.append({
            "Signal": sig,
            "Total CAGR": f"{cagr*100:.2f}%",
            "Max DD": f"{mdd*100:.2f}%",
            "Sharpe": f"{sharpe:.2f}",
            "IR vs SPY": f"{ir:.2f}" if sig != 'spy' else "-"
        })
        
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))

    print("\nYearly Return (%):")
    pdf['Year'] = pd.to_datetime(pdf['date']).dt.year
    yearly_ret = pdf.groupby('Year').apply(lambda x: (
        pd.Series({sig: (((x[sig].iloc[-1] / x[sig].iloc[0]) - 1.0) * 100) for sig in [*signals.keys(), 'spy']})
    ))
    print(yearly_ret.round(2).to_string())
    print("\n")
