# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     cell_metadata_json: true
#     formats: py:percent,ipynb
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

# %% [markdown]
# # MAX Momentum Strategy Backtest
# This notebook implements the MAX(N) metric backtest, which ranks assets based on the
# average of their N highest daily returns over a lookback window.
# Based on the research: "MAX Momentum in the Cryptocurrency Market" (Source 2).

# %% {"tags": ["parameters"]}
# 1. PARAMETERS
n_max = "1"  # N in MAX(N): average of top N daily returns
lookback_days = "14"  # Lookback window for metric calculation
holding_days = "1"  # Holding/Rebalance period
universe_top_n = "50"  # Top n coins by dollar volume (proxy for MCAP)
start_year = "2023"  # Backtest start year

# %%
# 2. PARAMETER PARSING
import polars as pl
import numpy as np
import scrapbook as sb
import datetime as dt

N_P = int(n_max)
lookback_P = int(lookback_days)
holding_P = int(holding_days)
universe_P = int(universe_top_n)
start_date_P = dt.datetime.strptime(f"{start_year}-01-01", "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)

# Sanity Checks
assert N_P > 0, "N must be a positive integer"
assert lookback_P >= N_P, "Lookback window must be at least as large as N"
assert holding_P > 0, "Holding period must be positive"
assert universe_P > 0, "Universe size must be positive"

print(f"Backtesting MAX({N_P}) Momentum")
print(f"Lookback: {lookback_P}d, Holding: {holding_P}d, Universe: Top {universe_P}")

# %%
# 3. DATA LOADING & FEATURE ENGINEERING
# Using stables-1d.parquet as specified in TEMPLATE.md (Source 3)
df = pl.read_parquet("stables-1d.parquet")

# Filter by start date
df = df.filter(pl.col("ts") >= start_date_P)

# Feature Engineering
# Calculate daily returns for MAX calculation
df = df.with_columns(
    (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("daily_ret")
)


# Calculate MAX(N): Average of N highest daily returns in the lookback window
def get_max_n(rets, n):
    # Returns the average of the n largest values in the window
    return np.mean(np.partition(rets, -n)[-n:]) if len(rets) >= n else np.nan


df = df.with_columns(
    pl.col("daily_ret")
    .rolling_map(lambda window: get_max_n(window, N_P), window_size=lookback_P)
    .over("symbol")
    .alias("max_metric")
)

# Identify Universe: Top N assets by dollar volume (quote_volume)
# This serves as the proxy for market capitalization filters used in the paper [1]
df = df.with_columns(
    pl.col("quote_volume").rank(descending=True).over("ts").alias("volume_rank")
)

# %%
# 4. BACKTEST (Signal Generation & Portfolio Construction)
# Define rebalance days based on holding period
# Using a simple modulo on the daily sequence to simulate rebalancing intervals
df = df.sort(["symbol", "ts"]).with_columns(
    (
        pl.col("ts").dt.truncate("1d").rank("dense") % holding_P == 0
    ).alias("rebalance_day")
)

# Filter for liquid universe and rank on rebalance days
signals = (
    df.filter((pl.col("volume_rank") <= universe_P) & (pl.col("rebalance_day")))
    .with_columns(
        pl.col("max_metric")
        .qcut(10, labels=[str(i) for i in range(10)], allow_duplicates=True)  # Decile sorting [2]
        .over("ts")
        .alias("decile")
    )
    .select(["ts", "symbol", "decile"])
)

# Join signals back and forward fill for the holding period
df = df.join(signals, on=["ts", "symbol"], how="left").with_columns(
    pl.col("decile").forward_fill().over("symbol")
)

# Portfolio logic: Long the High MAX decile (9) and Short the Low MAX decile (0) [3]
# Calculate decile-level equal-weighted returns
portfolio_rets = (
    df.filter(pl.col("decile").is_in(["0", "9"]))
    .group_by(["ts", "decile"])
    .agg(pl.col("daily_ret").mean().alias("decile_ret"))
    .pivot(index="ts", on="decile", values="decile_ret")
    .sort("ts")
)

# Strategy Return: Long High (9) - Short Low (0)
# Note: Source 2 highlights significant High-Low excess returns [3]
portfolio_rets = portfolio_rets.with_columns(
    (pl.col("9") - pl.col("0")).fill_null(0).alias("strategy_ret")
)

# %%
# 5. REPORTING
rets = portfolio_rets.select("strategy_ret").to_series()
cum_rets = (1 + rets).cum_prod()

# CAGR
total_days = (portfolio_rets["ts"].max() - portfolio_rets["ts"].min()).days
cagr = (cum_rets.tail(1).item() ** (365 / total_days)) - 1

# Max Drawdown
peak = cum_rets.cum_max()
drawdown = (cum_rets - peak) / peak
mdd = drawdown.min()

# Sortino Ratio (Daily to Annualized)
downside_std = rets.filter(rets < 0).std()
sortino = (rets.mean() / downside_std) * np.sqrt(365) if downside_std > 0 else 0

# Exporting metrics via Scrapbook
sb.glue("cagr", float(cagr))
sb.glue("sortino", float(sortino))
sb.glue("max_drawdown", float(mdd))

print(f"--- Strategy Performance ---")
print(f"CAGR: {cagr:.2%}")
print(f"Sortino Ratio: {sortino:.2f}")
print(f"Max Drawdown: {mdd:.2%}")

# %%
