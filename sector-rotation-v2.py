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
period = "21d"
stop_long = "0.5"
stop_short = "0.05"

# %%
import backtest_ng as bt
import polars as pl

from typing import Dict


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
start_year = 2020

class Alpha(bt.AlphaModel):
    def __init__(self, momentum_expr: pl.Expr, long_gate_expr: pl.Expr, short_gate_expr: pl.Expr,
                 indicators: Dict[str, pl.Expr]):
        self.momentum_expr = momentum_expr
        self.long_gate_expr = long_gate_expr
        self.short_gate_expr = short_gate_expr
        self.indicators = indicators
        self.max_long = 4
        self.max_short = 1
         

    def __call__(self, history: pl.DataFrame, u: bt.Universe) -> list[bt.Signal]:
        tscol = u.timestamp_col()
        symcol = u.symbol_col()
        df = (
            u.df()
            .with_columns(**self.indicators)
            .with_columns(
                score=self.momentum_expr.over(symcol)
            )
            .with_columns(
                lrank=pl.when(self.long_gate_expr)
                    .then((pl.col("score").rank(descending=True) / pl.len()).over(tscol))
                    .otherwise(None),
                srank=pl.when(self.short_gate_expr)
                    .then((pl.col("score").rank(descending=False) / pl.len()).over(tscol))
                    .otherwise(None),
            )
        )
        
        today = df[tscol].max()
        assert today is not None
        dfnow = df.filter(pl.col(tscol) == today)
        l = dfnow.sort("lrank")
        s = dfnow.sort("srank")
        
        ls = [
            bt.Signal(r[symcol], True, 1.0)
            for r in l.iter_rows(named=True)
        ]
        ss = [
            bt.Signal(r[symcol], False, -1.0)
            for r in s.iter_rows(named=True)
        ]

        return ls[:self.max_long] + ss[:self.max_short]

test = bt.Backtest(
    universe=bt.YFinance(
        tickers=sector_etfs,
        start=start_year,
    ),
    alpha=Alpha(
        momentum_expr=pl.col("mom12m") - pl.col("mom1m"),
        long_gate_expr=(pl.col("mom12m") > 0) | (pl.col("mom6m") > 0),
        short_gate_expr=(pl.col("mom12m") < 0) & (pl.col("mom6m") < 0),
        indicators={
            "mom1m": pl.col("close").pct_change(21).over("symbol"),
            "mom6m": pl.col("close").pct_change(21*6).over("symbol"),
            "mom12m": pl.col("close").pct_change(21*12).over("symbol"),
        }   
    ),
    #portfolio=bt.VolatilityWeighted(),
    #risk=bt.Fixed(
    #    long_trailing_pct=stop_long_P,
    #    short_trailing_pct=stop_short_P,
    #),
    #execution=bt.Schwab(
    #    base_rate_pct=10,
    #    maintainance_pct=30,
    #),
    benchmark='SPY',
    period=21,
)

test.run()
test.report(plot=True)

# %%
test.live(equity=10_000)
