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

lev_2x = {
    "XLB": "UYM", # Ultra Basic Materials 2x
    "XLC": "XLC", # No 2x
    "XLE": "DIG", # Ultra Oil & Gas 2x
    "XLF": "UYG", # Ultra Financials 2x
    "XLI": "UXI", # Ultra Industrials 2x
    "XLK": "ROM", # Ultra Technology 2x
    "XLP": "UGE", # Ultra Consumer Goods 2x
    "XLRE": "URE", # Ultra Real Estate 2x
    "XLU": "UPW", # Ultra Utilities 2x
    "XLV": "RXL", # Ultra Health Care 2x
    "XLY": "UCC", # Ultra Consumer Services 2x
}

lev_3x = {
    "XLB": "UYM", # Fallback to 2x
    "XLC": "XLC", # No 3x
    "XLE": "ERX", # Daily Energy Bull 2X (was 3x, now 2x)
    "XLF": "FAS", # Daily Financial Bull 3X
    "XLI": "DUSL", # Daily Industrials Bull 3X
    "XLK": "TECL", # Daily Technology Bull 3X
    "XLP": "UGE", # Fallback to 2x
    "XLRE": "DRN", # Daily Real Estate Bull 3X
    "XLU": "UTSL", # Daily Utilities Bull 3X
    "XLV": "CURE", # Daily Healthcare Bull 3X
    "XLY": "WANT", # Daily Consumer Discretionary Bull 3X
}

parameter_sets = [
    {
        "max_long": 2, "max_short": 1, "period": 20, "stop_long": .5, "stop_short": 0.3, "hard_stop_long": 1, "hard_stop_short": 1, "leverage": 1.0, 
    },
    {
        "max_long": 2, "max_short": 1, "period": 22, "stop_long": .5, "stop_short": 0.3, "hard_stop_long": 1, "hard_stop_short": 1, "leverage": 1.0, 
    },
    {
        "max_long": 2, "max_short": 1, "period": 24, "stop_long": .5, "stop_short": 0.3, "hard_stop_long": 1, "hard_stop_short": 1, "leverage": 1.0, 
    },
    {
        "max_long": 2, "max_short": 1, "period": 26, "stop_long": .5, "stop_short": 0.3, "hard_stop_long": 1, "hard_stop_short": 1, "leverage": 1.0, 
    },
     {
        "max_long": 2, "max_short": 1, "period": 28, "stop_long": .5, "stop_short": 0.3, "hard_stop_long": 1, "hard_stop_short": 1, "leverage": 1.0, 
    },
    {
        "max_long": 2, "max_short": 1, "period": 30, "stop_long": .5, "stop_short": 0.3, "hard_stop_long": 1, "hard_stop_short": 1, "leverage": 1.0, 
    },
    {
        "max_long": 2, "max_short": 1, "period": 32, "stop_long": .5, "stop_short": 0.3, "hard_stop_long": 1, "hard_stop_short": 1, "leverage": 1.0, 
    },
    {
        "max_long": 2, "max_short": 1, "period": 34, "stop_long": .5, "stop_short": 0.3, "hard_stop_long": 1, "hard_stop_short": 1, "leverage": 1.0, 
    }
]

all_run_stats = []

for param in parameter_sets:
    max_long = param["max_long"]
    max_short = param["max_short"]
    rebalance_period = param["period"]
    stop_long = param.get("stop_long")
    stop_short = param.get("stop_short")
    hard_stop_long = param.get("hard_stop_long", 0.05)
    hard_stop_short = param.get("hard_stop_short", 0.05)
    leverage = param.get("leverage", 1.0)
    etf_mapping = param.get("etf_mapping", {})
    name = param.get("name", "run")

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
    trades_per_signal = {}

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
        trades = []
        trades_per_signal[signal] = trades

        # Load the whole parquet once outside to get Treasury prices if needed
        # Or simply read it inside if we want to ensure we have it:
        all_needed_syms = ["IEF"] + list(etf_mapping.values())
        df_supp = (
            pl.read_parquet("yf.parquet")
            .filter(pl.col("symbol").is_in(all_needed_syms))
            .select(
                date=pl.col("ts").dt.date(),
                symbol=pl.col("symbol"),
                open=pl.col("open"),
                high=pl.col("high"),
                low=pl.col("low"),
                close=pl.col("close"),
            )
        )
        df_tlt = df_supp.filter(pl.col("symbol") == "IEF")

        for day in tqdm(df["date"].unique().sort().to_list()):
            df_now = df.filter(pl.col("date") == day)
            df_supp_now = df_supp.filter(pl.col("date") == day)

            # Compute today's portfolio NAV
            # Process trailing stops for the day
            new_portfolio = []
            for pos in portfolio:
                sym = pos["symbol"]
                shares = pos["shares"]
                ptype = pos["type"]

                # If this position is an ETF replacement or IEF, check df_supp_now instead
                if sym in all_needed_syms:
                    row = df_supp_now.filter(pl.col("symbol") == sym)
                else:
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
                        entry_low_ex = min(pos.get("entry_low_ex", pos["last_close"]), low)
                        pos["entry_high"] = entry_high
                        pos["entry_low_ex"] = entry_low_ex
                        
                        entry_price = pos["entry_price"]
                        
                        stop_price_trail = entry_high * (1.0 - stop_long) if (stop_long is not None and stop_long > 0) else -1
                        stop_price_hard = entry_price * (1.0 - hard_stop_long) if (hard_stop_long is not None and hard_stop_long > 0) else -1
                        stop_price = max(stop_price_trail, stop_price_hard)
                        
                        if stop_price > 0 and low <= stop_price:
                            exit_price = min(open_p, stop_price)
                            cash += shares * exit_price
                            trades.append({
                                "open_date": pos["open_date"],
                                "close_date": day,
                                "etf": sym,
                                "direction": "long",
                                "shares": shares,
                                "entry_price": entry_price,
                                "close_price": exit_price,
                                "profit": (exit_price / entry_price) - 1.0,
                                "mfe": (entry_high / entry_price) - 1.0,
                                "mae": (entry_low_ex / entry_price) - 1.0,
                                "leverage": leverage,
                                "reason": "stop"
                            })
                            continue
                            
                    elif ptype == "short":
                        entry_low = min(pos.get("entry_low", pos["last_close"]), low)
                        entry_high_ex = max(pos.get("entry_high_ex", pos["last_close"]), high)
                        pos["entry_low"] = entry_low
                        pos["entry_high_ex"] = entry_high_ex
                        
                        entry_price = pos["entry_price"]
                        
                        stop_price_trail = entry_low * (1.0 + stop_short) if (stop_short is not None and stop_short > 0) else float('inf')
                        stop_price_hard = entry_price * (1.0 + hard_stop_short) if (hard_stop_short is not None and hard_stop_short > 0) else float('inf')
                        stop_price = min(stop_price_trail, stop_price_hard)
                        
                        if stop_price < float('inf') and high >= stop_price:
                            exit_price = max(open_p, stop_price)
                            cash += shares * exit_price
                            trades.append({
                                "open_date": pos["open_date"],
                                "close_date": day,
                                "etf": sym,
                                "direction": "short",
                                "shares": shares,
                                "entry_price": entry_price,
                                "close_price": exit_price,
                                "profit": 1.0 - (exit_price / entry_price),
                                "mfe": 1.0 - (entry_low / entry_price),
                                "mae": 1.0 - (entry_high_ex / entry_price),
                                "leverage": leverage,
                                "reason": "stop"
                            })
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

                    # To determine if we keep it, we need to know the original underlying symbol
                    # If this is a mapped ETF, we want to know what it mapped *from*
                    # However, it's easier to just assume we hold everything mid-period unless explicitly liquidating.
                    # Wait, the logic before was: only hold if still in top rankings. But now we have trailing stops.
                    # For a simple trailing stop momentum strategy, we usually just hold until rebalance or stop.
                    # I will let it hold until rebalance day or stopout!
                    portfolio.append(pos)

                days_since_rebalance += 1

            elif bkt_len > 0:
                # Full rebalance
                # Record trades for positions being closed out at rebalance
                for pos in yd_folio:
                    sym = pos["symbol"]
                    if sym == "IEF": continue
                    
                    row = df_now.filter(pl.col("symbol") == sym)
                    if len(row) == 0 and sym in all_needed_syms:
                        row = df_supp_now.filter(pl.col("symbol") == sym)
                        
                    close_price = row["close"][0] if len(row) > 0 else pos["last_close"]
                    
                    trades.append({
                        "open_date": pos["open_date"],
                        "close_date": day,
                        "etf": sym,
                        "direction": pos["type"],
                        "shares": pos["shares"],
                        "entry_price": pos["entry_price"],
                        "close_price": close_price,
                        "profit": (close_price / pos["entry_price"]) - 1.0 if pos["type"] == "long" else 1.0 - (close_price / pos["entry_price"]),
                        "mfe": (pos.get("entry_high", close_price) / pos["entry_price"]) - 1.0 if pos["type"] == "long" else 1.0 - (pos.get("entry_low", close_price) / pos["entry_price"]),
                        "mae": (pos.get("entry_low_ex", close_price) / pos["entry_price"]) - 1.0 if pos["type"] == "long" else 1.0 - (pos.get("entry_high_ex", close_price) / pos["entry_price"]),
                        "leverage": leverage,
                        "reason": "rebalance"
                    })
                
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
                    ) * leverage
                    
                    buy_sym = etf_mapping.get(s, s)
                    if buy_sym == s:
                        buy_close = close_dict[s]
                    else:
                        row_supp = df_supp_now.filter(pl.col("symbol") == buy_sym)
                        if len(row_supp) > 0:
                            buy_close = row_supp["close"][0]
                        else:
                            # fallback if mapped ETF data not available yet
                            buy_sym = s
                            buy_close = close_dict[s]

                    shares = (w * portfolio_equity) / buy_close
                    portfolio.append(
                        {
                            "symbol": buy_sym,
                            "shares": shares,
                            "last_close": buy_close,
                            "type": "long",
                            "entry_high": buy_close,
                            "entry_low_ex": buy_close,
                            "entry_price": buy_close,
                            "open_date": day,
                        }
                    )
                    cash -= shares * buy_close

                for s in short_bkt[:max_short]:
                    w = (
                        inv_vols[s] / total_inv_vol
                        if total_inv_vol > 0
                        else 1.0 / bkt_len
                    ) * leverage
                    
                    buy_sym = etf_mapping.get(s, s)
                    if buy_sym == s:
                        buy_close = close_dict[s]
                    else:
                        row_supp = df_supp_now.filter(pl.col("symbol") == buy_sym)
                        if len(row_supp) > 0:
                            buy_close = row_supp["close"][0]
                        else:
                            buy_sym = s
                            buy_close = close_dict[s]

                    shares = -(w * portfolio_equity) / buy_close
                    portfolio.append(
                        {
                            "symbol": buy_sym,
                            "shares": shares,
                            "last_close": buy_close,
                            "type": "short",
                            "entry_low": buy_close,
                            "entry_high_ex": buy_close,
                            "entry_price": buy_close,
                            "open_date": day,
                        }
                    )
                    cash -= (
                        shares * buy_close
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
                                "entry_high": ief_close,
                                "entry_low_ex": ief_close,
                                "entry_price": ief_close,
                                "open_date": day,
                            }
                        )
                        cash = 0.0

                days_since_rebalance = 0
            else:
                # all to IEF
                cash = portfolio_equity
                
                # Record trades for positions being closed out due to no signal
                for pos in yd_folio:
                    sym = pos["symbol"]
                    if sym == "IEF": continue
                    
                    row = df_now.filter(pl.col("symbol") == sym)
                    if len(row) == 0 and sym in all_needed_syms:
                        row = df_supp_now.filter(pl.col("symbol") == sym)
                        
                    close_price = row["close"][0] if len(row) > 0 else pos["last_close"]
                    
                    trades.append({
                        "open_date": pos["open_date"],
                        "close_date": day,
                        "etf": sym,
                        "direction": pos["type"],
                        "shares": pos["shares"],
                        "entry_price": pos["entry_price"],
                        "close_price": close_price,
                        "profit": (close_price / pos["entry_price"]) - 1.0 if pos["type"] == "long" else 1.0 - (close_price / pos["entry_price"]),
                        "mfe": (pos.get("entry_high", close_price) / pos["entry_price"]) - 1.0 if pos["type"] == "long" else 1.0 - (pos.get("entry_low", close_price) / pos["entry_price"]),
                        "mae": (pos.get("entry_low_ex", close_price) / pos["entry_price"]) - 1.0 if pos["type"] == "long" else 1.0 - (pos.get("entry_high_ex", close_price) / pos["entry_price"]),
                        "leverage": leverage,
                        "reason": "flat_market"
                    })
                
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
                                "entry_high": ief_close,
                                "entry_low_ex": ief_close,
                                "entry_price": ief_close,
                                "open_date": day,
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
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15, 7))
    for sig in signals.keys():
        ax.plot(pdf["date"], pdf[sig], label=sig)
    ax.plot(pdf["date"], pdf["spy"], label="spy", color="black", linestyle="--")
    ax.set_title(f"rebalance every {rebalance_period}d long {max_long}, short {max_short} stop {stop_long * 100}/{stop_short * 100}% {name}")
    ax.legend()
    plt.savefig(f"plot_long{max_long}_short{max_short}_{name}.png")
    plt.show()

    print(f"\n--- Summary for rebalance {rebalance_period}d | long {max_long} | short {max_short} | variant: {name} ---")
    
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

    pdf['Year'] = pd.to_datetime(pdf['date']).dt.year
    yearly_ret = pdf.groupby('Year').apply(lambda x: (
        pd.Series({sig: (((x[sig].iloc[-1] / x[sig].iloc[0]) - 1.0) * 100) for sig in [*signals.keys(), 'spy']})
    ))
    
    ax = yearly_ret.plot(kind='bar', figsize=(15, 7), title=f"Yearly Return (%) - {name}")
    ax.set_ylabel("Return (%)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"yearly_ret_long{max_long}_short{max_short}_{name}.png")
    plt.show()

    for sig in [*signals.keys(), 'spy']:
        s_cum = pdf[sig]
        s_ret = pdf[sig].pct_change().fillna(0.0) if sig != 'spy' else pdf['spy_ret']

        cagr = calc_cagr(s_cum)
        mdd = calc_mdd(s_cum.values)
        sharpe = calc_sharpe(s_ret)
        ir = calc_ir(s_ret, pdf['spy_ret']) if sig != 'spy' else 0.0
        
        y_mean = yearly_ret[sig].mean() / 100.0
        y_std = yearly_ret[sig].std() / 100.0
        
        tdf = pd.DataFrame(trades_per_signal.get(sig, []))
        if not tdf.empty:
            wins = tdf[tdf['profit'] > 0]
            losses = tdf[tdf['profit'] <= 0]
            win_rate = len(wins) / len(tdf)
            avg_win = wins['profit'].mean() if len(wins) > 0 else 0.0
            avg_loss = losses['profit'].mean() if len(losses) > 0 else 0.0
            
            if avg_loss < 0:
                reward_risk = abs(avg_win / avg_loss)
                kelly = win_rate - ((1.0 - win_rate) / reward_risk)
            else:
                kelly = 0.0
                
            mfe_mean = tdf['mfe'].mean() if 'mfe' in tdf.columns else 0.0
            mfe_std = tdf['mfe'].std() if 'mfe' in tdf.columns else 0.0
            mae_mean = tdf['mae'].mean() if 'mae' in tdf.columns else 0.0
            mae_std = tdf['mae'].std() if 'mae' in tdf.columns else 0.0
            
            tdf.to_csv(f"trades_long{max_long}_short{max_short}_{name}_{sig}.csv", index=False)
        else:
            win_rate = avg_win = avg_loss = kelly = mfe_mean = mfe_std = mae_mean = mae_std = 0.0

        # Prevent duplicate spy lines in the global summary
        if sig == 'spy':
            if not any(r['Variant'] == 'SPY Baseline' for r in all_run_stats):
                all_run_stats.append({
                    "Variant": "SPY Baseline",
                    "Signal": "spy",
                    "CAGR": f"{cagr*100:.2f}%",
                    "Yrly Mean": f"{y_mean*100:.2f}%",
                    "Yrly Std": f"{y_std*100:.2f}%",
                    "Sharpe": f"{sharpe:.2f}",
                    "IR": "-",
                    "Max DD": f"{mdd*100:.2f}%",
                    "Win Rate": "-",
                    "Avg Win": "-",
                    "Avg Loss": "-",
                    "Kelly": "-",
                    "Half K.": "-",
                    "MAE": "-",
                    "MAE Std": "-",
                    "MFE": "-",
                    "MFE Std": "-"
                })
        else:
            all_run_stats.append({
                "Variant": name,
                "Signal": sig,
                "CAGR": f"{cagr*100:.2f}%",
                "Yrly Mean": f"{y_mean*100:.2f}%",
                "Yrly Std": f"{y_std*100:.2f}%",
                "Sharpe": f"{sharpe:.2f}",
                "IR": f"{ir:.2f}",
                "Max DD": f"{mdd*100:.2f}%",
                "Win Rate": f"{win_rate*100:.2f}%",
                "Avg Win": f"{avg_win*100:.2f}%",
                "Avg Loss": f"{avg_loss*100:.2f}%",
                "Kelly": f"{kelly*100:.2f}%",
                "Half K.": f"{(kelly/2.0)*100:.2f}%",
                "MAE": f"{mae_mean*100:.2f}%",
                "MAE Std": f"{mae_std*100:.2f}%",
                "MFE": f"{mfe_mean*100:.2f}%",
                "MFE Std": f"{mfe_std*100:.2f}%"
            })

print("\n" + "="*80)
print("=== GLOBAL SUMMARY ===")
print("="*80)
summary_df = pd.DataFrame(all_run_stats)

# Move SPY to the bottom
spy_row = summary_df[summary_df['Signal'] == 'spy']
other_rows = summary_df[summary_df['Signal'] != 'spy']
summary_df = pd.concat([other_rows, spy_row])

print(summary_df.to_string(index=False))
print("\n")

# %%
