---
name: backtest-design
description: A guide how to author trading strategy backtests.
---

Backtesting Framework and Best Practices
========================================

This doc details how to backtest trading strategies. Each strategy is a Jupyter
notebook in the Jupytext percent format. The notebooks should have the
following structure.

Parameters
----------

A cell that defines the strategy hyperparameters like holding periods, lookback
windows and quantile thresholds. The parameters should make sense for the
strategy being tested, the goal is to run a hyperparameter search. The
parameter cell must be tagged with "parameters" and all parameters should be
strings. We're using papermill to parameterize the notebooks.

A cell that parses the parameters into the correct types and sanitiy checks
them if necessary. Things like negative holding periods or top percentiles
smaller than bottom percentiles should be caught here. The parsed parameter
names should end with "_P" to distinguish them from the raw string parameters.
The parsed parameters should be printed out for easy debugging and record
keeping.

Data Sources
------------

A cell that loads the necessary data for the strategy and derives features. We
use Polars and Parquet files for this. For crypto currencies, use
`live/klines.parquet` and `live/symbols.parquet`. `live/symbols.parquet` contains
funding and borrowing rates across multiple exchanges, while `live/klines.parquet`
provides OHLCV market data. For equities, you can use yf.parquet which contains daily OHLCV
klines. Add a comment on top of the read_parquet line listing the required
tickers in the format: "uv run yf.py TICKER [TICKER ...]". The parquet file has
the following columns: ts (timestamp), symbol (ticker), open, high, low, close,
volume. For all other data needs, add a comment detailing what data is expected
and what the columns should be.

Backtest
--------

A cell that uses the framework in `backtest_ng/` to run the actual backtest.
The framework is modular and accepts custom alpha source, risk models and
portfolio construction logic. You'll likely need at least one alpha model to
generate trading signals. Try to use the existing risk and portfolio models as
much as possible.

Reporting
---------

The last cell takes the result data frame from the backtest and exports the
relevant metrics using scrapbook glue and prints those metrics for debugging.
Metrics should include at least the CAGR, Sortino ratio and max drawdown. In
general all metrics returned by the backtest should be exported.

For crypto currencies start the backtest at 2020. For equities start at 2000.

Example
-------

This is an example notebook. Imitate the coding style and structure as much as
possible.

```python
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
days_holding = "7"
top_percentile = "0.99"
bottom_percentile = "0.00"
days_momentum_score = "14"
days_momentum_gate = "21,30"

# %%
days_holding_P = int(days_holding)
top_percentile_P = float(top_percentile)
bottom_percentile_P = (
    float(bottom_percentile) if bottom_percentile != "None" else None
)
days_momentum_score_P = int(days_momentum_score)
days_momentum_gate_P = [int(n) for n in days_momentum_gate.split(",") if n != ""]

print(f"""
days_holding: {days_holding_P}
days_momentum_score: {days_momentum_score_P}
days_momentum_gate: {days_momentum_gate_P}

top_percentile: {top_percentile_P}
bottom_percentile: {bottom_percentile_P}
""")


# %%
import datetime as dt
import polars as pl
import scrapbook as sb
import backtest as bt
import functools as fc
import operator

df = (
    pl.read_parquet("polarity/latest-data.parquet")
    .rename(
        {
            "asset": "symbol",
            "price": "close",
            "timestamp": "ts",
        }
    )
    .with_columns(ts=pl.col("ts").dt.cast_time_unit("us"))
    .join(
        pl.read_parquet('stables-1d.parquet')
        .with_columns(
            ts=pl.col("ts").dt.cast_time_unit("us").dt.replace_time_zone(None),
            symbol=pl.col('symbol').str.to_lowercase().str.strip_suffix('usdt'),
        ),
        on=['ts','symbol'],
        how='inner'
    )
    .sort(["symbol", "ts"])
    .with_columns(**{
        f'mom{n}': pl.col("close").pct_change(n).over("symbol")
        for n in set([days_momentum_score_P, *days_momentum_gate_P])
    })
    .with_columns(
        rank=(
            pl.col(f"mom{days_momentum_score_P}").rank(method="ordinal").over("ts")
            / pl.col(f"mom{days_momentum_score_P}").count().over("ts")
        ),
        #volume=pl.col('quote_volume'),
    )
    .sort("ts")
    .filter(pl.col("ts").dt.year() >= 2023)
    .drop_nulls()
)

class Alpha(bt.AlphaModel):
    def __init__(self, long_expr: pl.Expr, short_expr: pl.Expr):
        self.long_expr = long_expr
        self.short_expr = short_expr

    def __call__(self, df: pl.DataFrame) -> list[bt.Signal]:
        today = df["ts"].max()
        dfnow = df.filter(pl.col("ts") == today)
        l = dfnow.filter(self.long_expr)
        s = dfnow.filter(self.short_expr)

        return [
            bt.Signal(r['symbol'], True, r['rank'])
            for r in l.iter_rows(named=True)
        ] + [
            bt.Signal(r['symbol'], False, 1-r['rank'])
            for r in s.iter_rows(named=True)
        ]

if len(days_momentum_gate_P) > 0:
    long_gate_expr = fc.reduce(operator.or_, [pl.col(f'mom{n}') > 0 for n in days_momentum_gate_P]) & (pl.col('mkt') > -0.0)
    short_gate_expr = fc.reduce(operator.and_, [pl.col(f'mom{n}') < 0 for n in days_momentum_gate_P]) & (pl.col('mkt') < -0.15)
else:
    long_gate_expr = pl.lit(True)
    short_gate_expr = pl.lit(True)


test = bt.Backtest(
    df,
    alpha=Alpha(
        long_expr=(pl.col("rank") >= top_percentile_P) & long_gate_expr,
        short_expr=(
            (pl.col("rank") <= bottom_percentile_P) & short_gate_expr
            if bottom_percentile_P is not None
            else pl.lit(False)
        ),
    ),
    #portfolio=bt.VolumeWeighted(),
    #risk=bt.MaxDrawdown(.3, .3),
    benchmark="btc",
    period=days_holding_P,
    eager_rebalance=True,
)

test.run()
res = test.report(plot=True)

for col in set(res.columns) - {'year', 'src'}:
    s = res.filter(pl.col('src') == 'Strategy')
    b = res.filter(pl.col('src') == 'Benchmark')

    print(f"{col}: {s[col].mean()} ({b[col].mean()})")
    sb.glue(col,s[col].mean())

# %%
```
