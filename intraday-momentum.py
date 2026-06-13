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
# Intraday Cross-Sectional Momentum
# ===================================
#
# Cross-sectional intraday momentum strategy on crypto perpetuals:
#
# - For each symbol independently, compute momentum return over the last `momentum_minutes` minutes
# - Each minute, rank all symbols into `n_quantiles` quantiles based on momentum
# - Long the top quantile, short the bottom quantile
# - Equal-weight allocation to all selected symbols
# - Rebalance every `holding_minutes` minutes

# %% {"tags": ["parameters"]}
momentum_minutes = "60"
holding_minutes = "10"
n_quantiles = "10"
data_path = "stables-1m.parquet"
start_date = "2023-01-01"

# %%
import polars as pl
import numpy as np
import datetime as dt
import scrapbook as sb
import backtest as bt

momentum_minutes_P = int(momentum_minutes)
holding_minutes_P = int(holding_minutes)
n_quantiles_P = int(n_quantiles)
start_date_P = dt.datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)

assert momentum_minutes_P > 0, "Momentum lookback must be positive"
assert holding_minutes_P > 0, "Holding period must be positive"
assert n_quantiles_P > 1, "Must have at least 2 quantiles for cross-sectional ranking"

print(
    f"""
Running backtest with: Momentum={momentum_minutes_P}m, Rebalance={holding_minutes_P}m, Quantiles={n_quantiles_P}
"""
)

# %%
# Load 1-minute Binance USDT perpetual data
df = (
    pl.read_parquet(data_path)
    .sort(["symbol", "ts"])
    .with_columns(pl.col("ts").dt.cast_time_unit("us"))
)

# Compute momentum feature (pre-computed so the AlphaModel can read it directly)
df = df.with_columns(
    (pl.col("close") / pl.col("close").shift(momentum_minutes_P) - 1)
    .over("symbol")
    .alias("momentum")
)

# Filter by start date (after computing momentum to allow warm-up)
df = df.filter(pl.col("ts") >= start_date_P)

print(f"Data loaded: {df.shape[0]:,} rows, {df['symbol'].n_unique()} symbols")
print(f"Date range: {df['ts'].min()} to {df['ts'].max()}")


# %%
class MomentumAlpha(bt.AlphaModel):
    """Cross-sectional momentum: rank symbols into quantiles by momentum each minute.
    Long the top quantile, short the bottom quantile.
    """

    def __init__(self, n_quantiles: int):
        self.n_quantiles = n_quantiles

    def __call__(self, df: pl.DataFrame) -> list[bt.Signal]:
        today = df["ts"].max()
        dfnow = df.filter(pl.col("ts") == today)

        if dfnow.is_empty():
            return []

        candidates = dfnow.filter(
            pl.col("momentum").is_not_null() & pl.col("momentum").is_finite()
        )

        if candidates.is_empty():
            return []

        # Rank into quantiles per minute
        quantile_labels = [str(i) for i in range(self.n_quantiles)]
        ranked = candidates.with_columns(
            pl.col("momentum")
            .rank("dense")
            .qcut(
                self.n_quantiles,
                labels=quantile_labels,
                allow_duplicates=True,
            )
            .alias("quantile")
        )

        top_quantile = str(self.n_quantiles - 1)
        bottom_quantile = "0"

        signals = []
        for row in ranked.iter_rows(named=True):
            if row["quantile"] == top_quantile:
                signals.append(bt.Signal(row["symbol"], bullish=True, confidence=1.0))
            elif row["quantile"] == bottom_quantile:
                signals.append(bt.Signal(row["symbol"], bullish=False, confidence=1.0))

        return signals


# %%
test = bt.Backtest(
    df,
    alpha=MomentumAlpha(n_quantiles_P),
    portfolio=bt.EqualWeight(),
    risk=bt.NoRisk(),
    fee=0.001,
    period=holding_minutes_P,
    freq="intraday",
)

test.run()
res = test.report(plot=True)

# %%
for col in set(res.columns) - {"year", "src"}:
    s = res.filter(pl.col("src") == "Strategy")

    val = s[col].mean() if not s.is_empty() else None

    # Handle NaN/Inf for JSON compliance in sb.glue
    if val is not None and (np.isnan(val) or np.isinf(val)):
        val = None

    print(f"{col}: {val}")
    sb.glue(col, float(val) if val is not None else None)

# %%
