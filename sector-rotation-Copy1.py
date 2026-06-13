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
max_long = "4"
max_short = "1"
period = "21"
stop_long = "0.5"
stop_short = "0.05"
hard_stop_long = "1"
hard_stop_short = "1"
leverage = "2"
# mom1m, mom2m, mom3m, mom6m, mom12m, mom12-1m-a, mom12-1m-b
signal = "mom12-1m-a"
gate = "mom12m-andor-6m"
variant_name = "default"
daily_exit = "False"
show_figs = "True"
use_live_data = "False"

# %%
import polars as pl
import numpy as np
import pandas as pd
from typing import Callable, Literal
from tqdm import tqdm
import matplotlib.pyplot as plt
import scrapbook as sb

max_long_param = int(max_long)
max_short_param = int(max_short)
period_param = int(period)
stop_long_param = float(stop_long)
stop_short_param = float(stop_short)
hard_stop_long_param = float(hard_stop_long)
hard_stop_short_param = float(hard_stop_short)
leverage_param = float(leverage)
daily_exit_param = daily_exit.lower() == "true"
show_figs_param = show_figs.lower() == "true"
use_live_data_param = use_live_data.lower() == "true"

print(
    f"Params: L={max_long_param} S={max_short_param} P={period_param} SL={stop_long_param} SS={stop_short_param} HL={hard_stop_long_param} HS={hard_stop_short_param} Lev={leverage_param} Sig={signal} DX={daily_exit_param}"
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
signal_expr = signals_map[signal]

gates_map = {
    "mom12m-andor-6m": [
        (pl.col("mom12m") > 0) | (pl.col("mom6m") > 0),
        (pl.col("mom12m") < 0) & (pl.col("mom6m") < 0),
    ],
    "ema50d": [
        pl.col("ema50d") < pl.col("close"),
        pl.col("ema50d") > pl.col("close"),
    ],
}
gate_expr = gates_map[gate]

if use_live_data_param:
    from yf import yf_download
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile() as f:
        yf_download(
            [
                "SPY",
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
                "XLB",
                "IEF",
            ],
            f.name,
            "2y",
        )
        df = pl.read_parquet(f.name)
else:
    # uv run yf.py SPY XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY XLB IEF --output yf.parquet
    df = pl.read_parquet("yf.parquet")


df = (
    df.filter(pl.col("symbol").is_in(sector_etfs + ["SPY"]))
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
        ema50d=pl.col("close").ewm_mean(span=50, adjust=True).over("symbol"),
    )
    .with_columns(score=signal_expr)
    .with_columns(
        long_rank=pl.when(gate_expr[0])
        .then(
            pl.col("score").rank(descending=True).over("date") / pl.len().over("date")
        )
        .otherwise(None),
        short_rank=pl.when(gate_expr[1])
        .then(
            pl.col("score").rank(descending=False).over("date") / pl.len().over("date")
        )
        .otherwise(None),
    )
)

df

# %%
import backtest as bt

stats = bt.Backtest(
    df.rename({'date':'ts'}),
    bt.Rank(signal_expr, gate_expr),
    bt.SimpleLeverage(2, bt.TopN(4,2, bt.EqualWeight('ts','symbol','close'))),
    bt.MaxDrawdown(1,.3,1,.05),
    21,
    1,
    'SPY',
).run().report(True).drop_nans('sortino')
print(stats)

# %%
print(stats['maxdd'].min())
print(stats['sortino'].mean())

# %%
