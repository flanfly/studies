# %% [markdown]
# # Starkiller Capital: Cross-sectional Momentum with Trend Overlay
# This notebook backtests a strategy that selects assets based on 30-day cumulative returns
# from a liquid universe, with a Bitcoin EMA trend filter to manage market beta [4, 5].

# %% {"tags": ["parameters"]}
# 1. PARAMETERS
lookback = "30"  # Window for cumulative return ranking
holding_period = "7"  # Rebalance frequency in days
n_buckets = "5"  # 5 = Quintiles
ema_fast = "5"  # Fast EMA for BTC trend filter
ema_slow = "50"  # Slow EMA for BTC trend filter
volume_cutoff = "5_000_000"  # $5M consistent volume filter
start_date = "2020-01-01"  # Required start for crypto [3]

# %%
# 2. PARAMETER PARSING
import polars as pl
from datetime import datetime, timezone

lookback_P = int(lookback)
holding_period_P = int(holding_period)
n_buckets_P = int(n_buckets)
ema_fast_P = int(ema_fast)
ema_slow_P = int(ema_slow)
volume_cutoff_P = float(volume_cutoff)
start_date_P = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

# Sanity Checks [2]
assert lookback_P > 0, "Lookback must be positive"
assert holding_period_P > 0, "Holding period must be positive"
assert n_buckets_P > 1, "Must have at least 2 buckets for cross-sectional ranking"
assert ema_fast_P < ema_slow_P, "Fast EMA must be smaller than Slow EMA"

print(
    f"Running backtest with: Lookback={lookback_P}, Rebalance={holding_period_P}d, Buckets={n_buckets_P}"
)
print(f"Liquidity Filter: ${volume_cutoff_P:,.0f} consistent volume")
print(f"Trend Overlay: BTC EMA({ema_fast_P}/{ema_slow_P})")

# %%
# 3. DATA LOADING & FEATURE ENGINEERING
import numpy as np
import scrapbook as sb

# Load Binance USDT daily data [2]
# Sort ensures rolling windows and shifts work correctly
df = pl.read_parquet("stables-1d.parquet").sort(["symbol", "ts"])

# Calculate Features BEFORE filtering (allows for EMA/Momentum warm-up)
df = df.with_columns(
    [
        # Daily return for portfolio construction
        (pl.col("close").pct_change().over("symbol")).alias("daily_return"),
        # Cumulative return for momentum ranking [6]
        (pl.col("close").pct_change(lookback_P).over("symbol")).alias("momentum_score"),
        # 30-day Volume Consistency Filter
        (pl.col("quote_volume") >= volume_cutoff_P)
        .cast(pl.Int32)
        .rolling_mean(window_size=30)
        .over("symbol")
        .alias("liquidity_pass"),
    ]
)

# Generate Bitcoin Trend Overlay Signal [5]
btc_prices = df.filter(pl.col("symbol") == "BTCUSDT").select(["ts", "close"])
btc_trend = (
    btc_prices.with_columns(
        [
            pl.col("close").ewm_mean(span=ema_fast_P).alias("ema_f"),
            pl.col("close").ewm_mean(span=ema_slow_P).alias("ema_s"),
        ]
    )
    .with_columns(
        (pl.col("ema_f") > pl.col("ema_s")).cast(pl.Int32).alias("trend_signal")
    )
    .select(["ts", "trend_signal"])
)

# Join trend signal back to main dataframe
df = df.join(btc_trend, on="ts", how="left")

# Now filter by start date for the backtest
df = df.filter(pl.col("ts") >= start_date_P)

# %%
# 4. BACKTEST (Signal Generation & Portfolio Construction)
# Rebalance on a fixed schedule (e.g., weekly) [7]
df = df.with_columns(
    (pl.col("ts").dt.weekday() == 4).alias("rebalance_day")  # Thursday at midnight [7]
)

# Apply liquidity filter and manual exclusions (Stablecoins/Wrapped)
# Note: stables-1d.parquet usually requires a manual list of exclusions [8, 9]
df = df.filter(pl.col("liquidity_pass") >= 0.5)

# Signal: Sort assets into buckets on rebalance days
# We use rank() before qcut to handle ties deterministically
signals = (
    df.filter(
        pl.col("rebalance_day")
        & pl.col("momentum_score").is_not_null()
        & pl.col("momentum_score").is_finite()
    )
    .with_columns(
        pl.col("momentum_score")
        .rank("dense")
        .qcut(n_buckets_P, labels=[str(i) for i in range(n_buckets_P)])
        .over("ts")
        .alias("bucket")
    )
    .select(["ts", "symbol", "bucket"])
)

# Carry signals forward through the holding period
df = (
    df.join(signals, on=["ts", "symbol"], how="left")
    .sort(["symbol", "ts"])
    .with_columns(pl.col("bucket").forward_fill().over("symbol"))
    .with_columns(pl.col("bucket").shift(1).over("symbol"))
)

# Portfolio Strategy: Top Quintile (Winner) Portfolio [4, 10]
winner_bucket = str(n_buckets_P - 1)

# Equal weight calculation within the winner bucket
portfolio = (
    df.filter(pl.col("bucket") == winner_bucket)
    .group_by("ts", maintain_order=True)
    .agg(
        [
            pl.col("daily_return").mean().alias("raw_return"),
            pl.col("trend_signal").first().alias("market_trend"),
        ]
    )
)

# Apply Trend Following Overlay: If BTC trend is down, move to cash (0 return) [5, 11]
portfolio = portfolio.with_columns(
    (pl.col("raw_return") * pl.col("market_trend").shift(1))
    .fill_null(0)
    .alias("strategy_return")
).sort("ts")

# %%
# 5. REPORTING
# Calculate Performance Metrics [2]
returns = portfolio.select("strategy_return").to_series()
cum_rets = (1 + returns).cum_prod()

# CAGR
days = (portfolio["ts"].max() - portfolio["ts"].min()).days
cagr = (cum_rets.tail(1).item()) ** (365 / days) - 1

# Max Drawdown
running_max = cum_rets.cum_max()
drawdown = (cum_rets / running_max) - 1
mdd = drawdown.min()

# Sortino Ratio (Standard downside deviation)
downside_deviation = np.sqrt((np.minimum(0, returns) ** 2).mean())
sortino = (
    (returns.mean() / downside_deviation) * np.sqrt(365)
    if downside_deviation > 0
    else 0
)

# Scrapbook Glue for Hyperparameter Search [2]
sb.glue("cagr", float(cagr))
sb.glue("sortino", float(sortino))
sb.glue("max_drawdown", float(mdd))

print(f"CAGR: {cagr:.2%}")
print(f"Sortino Ratio: {sortino:.2f}")
print(f"Max Drawdown: {mdd:.2%}")

# %%
