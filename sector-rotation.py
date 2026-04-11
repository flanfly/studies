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

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
max_long = "2"
max_short = "1"
period = "30"
stop_long = "0.5"
stop_short = "0.3"
hard_stop_long = "0.05"
hard_stop_short = "0.05"
leverage = "1.0"
# mom1m, mom2m, mom3m, mom6m, mom12m, mom12-1m-a, mom12-1m-b
signal = "mom12-1m-a"
variant_name = "default"

# %%
import polars as pl
import numpy as np
import pandas as pd
from typing import Callable, Literal
from tqdm import tqdm
import matplotlib.pyplot as plt
import scrapbook as sb

max_long = int(max_long)
max_short = int(max_short)
period = int(period)
stop_long = float(stop_long)
stop_short = float(stop_short)
hard_stop_long = float(hard_stop_long)
hard_stop_short = float(hard_stop_short)
leverage = float(leverage)

print(
    f"Params: L={max_long} S={max_short} P={period} SL={stop_long} SS={stop_short} HL={hard_stop_long} HS={hard_stop_short} Lev={leverage} Sig={signal}"
)

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
lev_2x = {
    "XLB": "UYM",
    "XLC": "XLC",
    "XLE": "DIG",
    "XLF": "UYG",
    "XLI": "UXI",
    "XLK": "ROM",
    "XLP": "UGE",
    "XLRE": "URE",
    "XLU": "UPW",
    "XLV": "RXL",
    "XLY": "UCC",
}
lev_3x = {
    "XLB": "UYM",
    "XLC": "XLC",
    "XLE": "ERX",
    "XLF": "FAS",
    "XLI": "DUSL",
    "XLK": "TECL",
    "XLP": "UGE",
    "XLRE": "DRN",
    "XLU": "UTSL",
    "XLV": "CURE",
    "XLY": "WANT",
}

etf_mapping = {}
if variant_name == "2x":
    etf_mapping = lev_2x
elif variant_name == "3x":
    etf_mapping = lev_3x

yz_k, yz_win = 0.34, 25

signals_map = {
    "mom12m": pl.col("mom12m"),
    "mom1m": pl.col("mom1m"),
    "mom2m": pl.col("mom2m"),
    "mom3m": pl.col("mom3m"),
    "mom6m": pl.col("mom6m"),
    "mom12-1m-a": pl.col("mom12m") - pl.col("mom1m"),
    "mom12-1m-b": pl.col("mom11m").shift(21).over("symbol"),
}
expr = signals_map[signal]

df = (
    # uv run yf.py SPY XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY XLB IEF --output yf.parquet
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
            for n in [1, 2, 3, 6, 11, 12]
        },
        sma50d=pl.col("close").rolling_mean(50).over("symbol"),
    )
    .with_columns(score=expr)
    .with_columns(
        long_rank=pl.when((pl.col("mom12m") > 0) | (pl.col("mom6m") > 0))
        .then(
            pl.col("score").rank(descending=True).over("date") / pl.len().over("date")
        )
        .otherwise(None),
        short_rank=pl.when((pl.col("mom6m") < 0) & (pl.col("mom12m") < 0))
        .then(
            pl.col("score").rank(descending=False).over("date") / pl.len().over("date")
        )
        .otherwise(None),
    )
)

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

portfolio, trades, perf_frag = [], [], []
cash, portfolio_equity, days_since_rebalance = 1.0, 1.0, period

for day in tqdm(df["date"].unique().sort().to_list()):
    df_now, df_supp_now = df.filter(pl.col("date") == day), df_supp.filter(
        pl.col("date") == day
    )
    new_portfolio = []
    for pos in portfolio:
        sym, shares, ptype = pos["symbol"], pos["shares"], pos["type"]
        row = (
            df_supp_now.filter(pl.col("symbol") == sym)
            if sym in all_needed_syms
            else df_now.filter(pl.col("symbol") == sym)
        )
        if len(row) == 0:
            if sym == "IEF":
                row_tlt = df_tlt.filter(pl.col("date") == day)
                if len(row_tlt) > 0:
                    pos["last_close"] = row_tlt["close"][0]
            new_portfolio.append(pos)
            continue
        if sym != "IEF":
            high, low, open_p, close = (
                row["high"][0],
                row["low"][0],
                row["open"][0],
                row["close"][0],
            )
            entry_price = pos["entry_price"]
            if ptype == "long":
                entry_high = max(pos.get("entry_high", pos["last_close"]), high)
                pos["entry_high"], pos["entry_low_ex"] = entry_high, min(
                    pos.get("entry_low_ex", pos["last_close"]), low
                )
                stop_p = max(
                    entry_high * (1 - stop_long) if stop_long > 0 else -1,
                    entry_price * (1 - hard_stop_long) if hard_stop_long > 0 else -1,
                )
                if stop_p > 0 and low <= stop_p:
                    exit_p = min(open_p, stop_p)
                    cash += shares * exit_p
                    trades.append(
                        {
                            "open_date": pos["open_date"],
                            "close_date": day,
                            "etf": sym,
                            "direction": "long",
                            "shares": shares,
                            "entry_price": entry_price,
                            "close_price": exit_p,
                            "profit": (exit_p / entry_price) - 1,
                            "mfe": (entry_high / entry_price) - 1,
                            "mae": (pos["entry_low_ex"] / entry_price) - 1,
                            "leverage": leverage,
                            "reason": "stop",
                        }
                    )
                    continue
            else:
                entry_low = min(pos.get("entry_low", pos["last_close"]), low)
                pos["entry_low"], pos["entry_high_ex"] = entry_low, max(
                    pos.get("entry_high_ex", pos["last_close"]), high
                )
                stop_p = min(
                    entry_low * (1 + stop_short) if stop_short > 0 else float("inf"),
                    (
                        entry_price * (1 + hard_stop_short)
                        if hard_stop_short > 0
                        else float("inf")
                    ),
                )
                if stop_p < float("inf") and high >= stop_p:
                    exit_p = max(open_p, stop_p)
                    cash += shares * exit_p
                    trades.append(
                        {
                            "open_date": pos["open_date"],
                            "close_date": day,
                            "etf": sym,
                            "direction": "short",
                            "shares": shares,
                            "entry_price": entry_price,
                            "close_price": exit_p,
                            "profit": 1 - (exit_p / entry_price),
                            "mfe": 1 - (entry_low / entry_price),
                            "mae": 1 - (pos["entry_high_ex"] / entry_price),
                            "leverage": leverage,
                            "reason": "stop",
                        }
                    )
                    continue
            pos["last_close"] = close
            new_portfolio.append(pos)
        else:
            pos["last_close"] = row["close"][0]
            new_portfolio.append(pos)
    portfolio = new_portfolio
    today_value = cash + sum(p["shares"] * p["last_close"] for p in portfolio)
    ret = (today_value / portfolio_equity) - 1.0 if portfolio_equity > 0 else 0.0
    portfolio_equity = today_value
    perf_frag.append(pl.DataFrame({"date": [day], "ret": [ret]}))

    long_bkt = (
        df_now.filter(pl.col("long_rank").is_null().not_())
        .sort("long_rank")["symbol"]
        .to_list()
    )
    short_bkt = (
        df_now.filter(pl.col("short_rank").is_null().not_())
        .sort("short_rank")["symbol"]
        .to_list()
    )
    bkt_len = len(long_bkt[:max_long]) + len(short_bkt[:max_short])
    yd_folio = portfolio.copy()

    if days_since_rebalance < period:
        for pos in yd_folio:
            pass  # kept
        days_since_rebalance += 1
    else:
        for pos in yd_folio:
            if pos["symbol"] == "IEF":
                continue
            row = df_now.filter(pl.col("symbol") == pos["symbol"])
            if len(row) == 0 and pos["symbol"] in all_needed_syms:
                row = df_supp_now.filter(pl.col("symbol") == pos["symbol"])
            cp = row["close"][0] if len(row) > 0 else pos["last_close"]
            trades.append(
                {
                    "open_date": pos["open_date"],
                    "close_date": day,
                    "etf": pos["symbol"],
                    "direction": pos["type"],
                    "shares": pos["shares"],
                    "entry_price": pos["entry_price"],
                    "close_price": cp,
                    "profit": (
                        (cp / pos["entry_price"]) - 1
                        if pos["type"] == "long"
                        else 1 - (cp / pos["entry_price"])
                    ),
                    "mfe": (
                        (pos.get("entry_high", cp) / pos["entry_price"]) - 1
                        if pos["type"] == "long"
                        else 1 - (pos.get("entry_low", cp) / pos["entry_price"])
                    ),
                    "mae": (
                        (pos.get("entry_low_ex", cp) / pos["entry_price"]) - 1
                        if pos["type"] == "long"
                        else 1 - (pos.get("entry_high_ex", cp) / pos["entry_price"])
                    ),
                    "leverage": leverage,
                    "reason": "rebalance",
                }
            )

        portfolio = []
        if bkt_len > 0:
            close_dict = dict(df_now[["symbol", "close"]].iter_rows(named=False))
            var_dict = dict(df_now[["symbol", "var"]].iter_rows(named=False))
            selected = long_bkt[:max_long] + short_bkt[:max_short]
            inv_vols = {
                s: 1.0 / (var_dict.get(s, 1e-6) ** 0.5) if var_dict.get(s, 0) > 0 else 0
                for s in selected
            }
            tiv = sum(inv_vols.values())
            cash = portfolio_equity
            for s in long_bkt[:max_long]:
                w = (inv_vols[s] / tiv if tiv > 0 else 1.0 / bkt_len) * leverage
                bs = etf_mapping.get(s, s)
                row_s = df_supp_now.filter(pl.col("symbol") == bs)
                bc = row_s["close"][0] if len(row_s) > 0 else close_dict[s]
                sh = (w * portfolio_equity) / bc
                portfolio.append(
                    {
                        "symbol": bs,
                        "shares": sh,
                        "last_close": bc,
                        "type": "long",
                        "entry_high": bc,
                        "entry_low_ex": bc,
                        "entry_price": bc,
                        "open_date": day,
                    }
                )
                cash -= sh * bc
            for s in short_bkt[:max_short]:
                w = (inv_vols[s] / tiv if tiv > 0 else 1.0 / bkt_len) * leverage
                bs = etf_mapping.get(s, s)
                row_s = df_supp_now.filter(pl.col("symbol") == bs)
                bc = row_s["close"][0] if len(row_s) > 0 else close_dict[s]
                sh = -(w * portfolio_equity) / bc
                portfolio.append(
                    {
                        "symbol": bs,
                        "shares": sh,
                        "last_close": bc,
                        "type": "short",
                        "entry_low": bc,
                        "entry_high_ex": bc,
                        "entry_price": bc,
                        "open_date": day,
                    }
                )
                cash -= sh * bc
            if cash > 0:
                row_t = df_tlt.filter(pl.col("date") == day)
                if len(row_t) > 0:
                    ic = row_t["close"][0]
                    ish = cash / ic
                    portfolio.append(
                        {
                            "symbol": "IEF",
                            "shares": ish,
                            "last_close": ic,
                            "type": "long",
                            "entry_high": ic,
                            "entry_low_ex": ic,
                            "entry_price": ic,
                            "open_date": day,
                        }
                    )
                    cash = 0
            days_since_rebalance = 0
        else:
            cash = portfolio_equity
            if cash > 0:
                row_t = df_tlt.filter(pl.col("date") == day)
                if len(row_t) > 0:
                    ic = row_t["close"][0]
                    ish = cash / ic
                    portfolio.append(
                        {
                            "symbol": "IEF",
                            "shares": ish,
                            "last_close": ic,
                            "type": "long",
                            "entry_high": ic,
                            "entry_low_ex": ic,
                            "entry_price": ic,
                            "open_date": day,
                        }
                    )
                    cash = 0
            days_since_rebalance = 0

perf = pl.concat(perf_frag)
df_perf = perf.join(
    pl.read_parquet("yf.parquet")
    .filter(pl.col("symbol") == "SPY")
    .sort("ts")
    .select(date=pl.col("ts").dt.date(), spy_ret=pl.col("close").pct_change()),
    on="date",
    how="inner",
)
df_perf = df_perf.with_columns(
    cum=(pl.col("ret") + 1).cum_prod(), spy=(pl.col("spy_ret") + 1).cum_prod()
)
pdf = df_perf.to_pandas()
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(pdf["date"], pdf["cum"], label="Strategy")
ax.plot(pdf["date"], pdf["spy"], label="SPY", color="black", linestyle="--")
plt.savefig(f"result_{variant_name}.png")
plt.close()


def calc_stats(cum, rets, spy_rets):
    ny = len(cum) / 252.0
    cagr = (cum.iloc[-1] / cum.iloc[0]) ** (1 / ny) - 1
    mdd = (cum / np.maximum.accumulate(cum) - 1).min()
    sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252))
    ir = ((rets - spy_rets).mean() * 252) / ((rets - spy_rets).std() * np.sqrt(252))

    # Sortino Ratio
    downside_rets = rets[rets < 0]
    downside_std = downside_rets.std() * np.sqrt(252)
    sortino = (rets.mean() * 252) / downside_std if downside_std > 0 else 0

    return cagr, mdd, sharpe, ir, sortino


c, m, s, i, sortino = calc_stats(pdf["cum"], pdf["ret"], pdf["spy_ret"])
tdf = pd.DataFrame(trades)
wr = len(tdf[tdf["profit"] > 0]) / len(tdf) if len(tdf) > 0 else 0
print(
    f"SUMMARY: CAGR={c:.2%}, MDD={m:.2%}, Sharpe={s:.2f}, Sortino={sortino:.2f}, IR={i:.2f}, WinRate={wr:.2%}"
)

sb.glue("cagr", float(c))
sb.glue("maxdd", float(m))
sb.glue("sortino", float(sortino))
sb.glue("sharpe", float(s))
sb.glue("win_rate", float(wr))
