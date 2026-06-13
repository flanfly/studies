# %% [markdown]
# # Cross-sectional Momentum (J/K) Backtest
# This notebook backtests a standard cross-sectional momentum strategy.
# Assets are ranked by cumulative returns over a lookback window (J).
# Portfolios are formed using parameterized winner and loser bucket sizes
# and held for a fixed duration (K).

# %% {"tags": ["parameters"]}
# 1. PARAMETERS
lookback_days = "30"  # J: Formation period in days
holding_days = "7"  # K: Holding/Rebalance period in days
winner_percentile = "0.2"  # Size of the winner bucket (0.2 = top 20%)
loser_percentile = "0.1"  # Size of the loser bucket (0.2 = bottom 20%)
min_volume = "1_000_000"  # Minimum daily dollar volume filter
start_date = "2025-01-01"  # Backtest start date
reference_pair = "BTCUSDT"  # Benchmark pair for equity curve comparison

# %%
# 2. PARAMETER PARSING
import polars as pl
import numpy as np
import scrapbook as sb

lookback_P = int(lookback_days)
holding_P = int(holding_days)
winner_P = float(winner_percentile)
loser_P = float(loser_percentile)
volume_P = float(min_volume)

# Sanity Checks
assert lookback_P > 0, "Lookback period must be positive"
assert holding_P > 0, "Holding period must be positive"
assert 0 <= winner_P <= 1, "Winner percentile must be between 0 and 1"
assert 0 <= loser_P <= 1, "Loser percentile must be between 0 and 1"
assert (winner_P + loser_P) <= 1.0, "Sum of buckets cannot exceed 100% of universe"

print(f"Backtesting J={lookback_P}d / K={holding_P}d Momentum")
print(
    f"Buckets: Top {winner_P*100:.0f}% (Winners) | Bottom {loser_P*100:.0f}% (Losers)"
)

# %%
# 3. DATA LOADING & FEATURE ENGINEERING
# Using stables-1d.parquet: ts, symbol, open, high, low, close, volume, quote_volume
df = pl.read_parquet("stables-1d.parquet")

# Filter by start date
df = df.filter(
    pl.col("ts") >= pl.lit(start_date).str.to_datetime(time_unit="ms", time_zone="UTC")
)

# Calculate Features
df = df.sort(["symbol", "ts"]).with_columns(
    [
        pl.col("close").rolling_mean(window_size=30).over("symbol").alias("ma_30d"),
        # Cumulative return over J days (Momentum Score)
        (pl.col("close").pct_change(lookback_P).over("symbol")).alias("mom_score"),
        # Simple daily return for PnL calculation
        (pl.col("close").pct_change(1).over("symbol")).alias("daily_ret"),
        # 7-day moving average of volume for liquidity filtering (Source 5 logic)
        (pl.col("quote_volume").rolling_mean(window_size=7).over("symbol")).alias(
            "avg_vol"
        ),
    ]
)

# %%
# 4. BACKTEST (Signal Generation & Portfolio Construction)
# Identify rebalance days (every K days)
df = df.with_columns(
    (pl.col("ts").dt.truncate("1d").rank("dense") % holding_P == 0).alias(
        "rebalance_day"
    )
)

# Signal Generation
# Rank assets on rebalance days within the liquid universe
signals = (
    df.filter((pl.col("avg_vol") >= volume_P) & (pl.col("rebalance_day")))
    .with_columns(
        [
            pl.col("mom_score").rank(descending=True).over("ts").alias("rank_desc"),
            pl.len().over("ts").alias("universe_size"),
        ]
    )
    .with_columns(
        [
            # Assign to Winner bucket (top X%)
            (
                (pl.col("rank_desc") <= (pl.col("universe_size") * winner_P))
                & (pl.col("mom_score") > 0)
                & (pl.col("ma_30d") < pl.col("close"))
            ).alias("is_winner"),
            # Assign to Loser bucket (bottom X%)
            (
                (pl.col("rank_desc") > (pl.col("universe_size") * (1 - loser_P)))
                & (pl.col("mom_score") < 0)
                & (pl.col("ma_30d") > pl.col("close"))
            ).alias("is_loser"),
        ]
    )
    .select(["ts", "symbol", "is_winner", "is_loser"])
)

# Join signals back to main dataframe, forward fill for the holding period,
# then lag by 1 day: today's return belongs to yesterday's portfolio.
# This avoids lookahead bias on rebalance days where the new signal
# would otherwise be credited with the day's already-earned return.
df = df.join(signals, on=["ts", "symbol"], how="left").with_columns(
    [
        pl.col("is_winner")
        .forward_fill()
        .over("symbol")
        .shift(1)
        .over("symbol")
        .fill_null(False),
        pl.col("is_loser")
        .forward_fill()
        .over("symbol")
        .shift(1)
        .over("symbol")
        .fill_null(False),
    ]
)

# Portfolio Logic: Equal-weighted Winner - Equal-weighted Loser (Source 1, 11)
portfolio_rets = (
    df.group_by("ts")
    .agg(
        [
            pl.col("daily_ret").filter(pl.col("is_winner")).mean().alias("winner_ret"),
            pl.col("daily_ret").filter(pl.col("is_loser")).mean().alias("loser_ret"),
        ]
    )
    .with_columns(
        (pl.col("winner_ret").fill_null(0) - pl.col("loser_ret").fill_null(0)).alias(
            "strategy_ret"
        )
    )
    .sort("ts")
)

# %%
# 5. REPORTING
returns = portfolio_rets.select("strategy_ret").to_series()
cum_rets = (1 + returns).cum_prod()

# CAGR calculation
total_days = (portfolio_rets["ts"].max() - portfolio_rets["ts"].min()).days
final_value = cum_rets.tail(1).item()
if final_value > 0:
    cagr = final_value ** (365 / total_days) - 1
else:
    # Negative terminal wealth: use arithmetic annualization as fallback
    cagr = returns.mean() * 365

# Max Drawdown
peak = cum_rets.cum_max()
drawdown = (cum_rets - peak) / peak
mdd = drawdown.min()

# Sortino Ratio (Annualized)
downside_std = returns.filter(returns < 0).std()
sortino = (
    (returns.mean() / downside_std) * np.sqrt(365)
    if downside_std and downside_std > 0
    else 0
)

# Glue metrics for papermill/hyperparameter tracking
sb.glue("cagr", float(cagr))
sb.glue("sortino", float(sortino))
sb.glue("max_drawdown", float(mdd))

print(f"--- Strategy Metrics ---")
print(f"CAGR: {cagr:.2%}")
print(f"Sortino Ratio: {sortino:.2f}")
print(f"Max Drawdown: {mdd:.2%}")

# %%
# 6. EQUITY CURVE vs BTC BUY & HOLD
import matplotlib.pyplot as plt

# Extract reference pair buy-and-hold returns from the same filtered dataframe
btc_rets = (
    df.filter(pl.col("symbol") == reference_pair)
    .select(["ts", "daily_ret"])
    .sort("ts")
    .with_columns((1 + pl.col("daily_ret").fill_null(0)).cum_prod().alias("btc_equity"))
)

# Align dates: join BTC equity curve onto the portfolio dates for plotting
btc_equity_map = dict(zip(btc_rets["ts"].to_list(), btc_rets["btc_equity"].to_list()))
portfolio_dates = portfolio_rets.select("ts").to_series().to_list()
btc_aligned = [btc_equity_map.get(d, None) for d in portfolio_dates]

# Forward-fill any missing BTC dates (e.g., if strategy has dates BTC doesn't)
for i in range(1, len(btc_aligned)):
    if btc_aligned[i] is None:
        btc_aligned[i] = btc_aligned[i - 1]

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax.plot(
    portfolio_dates,
    cum_rets.to_list(),
    label="Strategy (Winners − Losers)",
    linewidth=1.5,
)
ax.plot(
    portfolio_dates,
    btc_aligned,
    label=f"{reference_pair} Buy & Hold",
    linewidth=1,
    linestyle="--",
    color="orange",
)
ax.set_title(
    f"Cross-sectional Momentum (J={lookback_P}d / K={holding_P}d) vs {reference_pair}"
)
ax.set_xlabel("Date")
ax.set_ylabel("Equity (log scale)")
ax.set_yscale("log")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# Long vs Short leg cumulative returns
long_equity = (
    1 + portfolio_rets.select("winner_ret").to_series().fill_null(0)
).cum_prod()
short_equity = (
    1 + portfolio_rets.select("loser_ret").to_series().fill_null(0)
).cum_prod()
ax2.plot(
    portfolio_dates,
    long_equity.to_list(),
    label="Long (Winners)",
    linewidth=1.5,
    color="green",
)
ax2.plot(
    portfolio_dates,
    short_equity.to_list(),
    label="Short (Losers)",
    linewidth=1,
    linestyle="--",
    color="red",
)
ax2.set_title("Long Leg vs Short Leg")
ax2.set_xlabel("Date")
ax2.set_ylabel("Equity (log scale)")
ax2.set_yscale("log")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()
