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

# %%
import polars as pl
import datetime as dt

df = (
    pl.read_csv('glbx-mdp3-20100606-20260412.ohlcv-1d.csv.zst')
    .with_columns(
        ts=pl.col('ts_event').str.to_datetime(format="%Y-%m-%dT%H:%M:%S.000000000Z"),
    )
    .select(
        ts=pl.col('ts'),
        close=pl.col('close'),
        open=pl.col("open"),
        high=pl.col("high"),
        low=pl.col("low"),
        volume=pl.col('volume'),
        symbol=pl.col('symbol').str.slice(0, pl.col('symbol').str.len_chars()-2),
        expy=pl.col('symbol').str.slice(-1).str.to_integer() + pl.col('ts').dt.year().floordiv(10) * 10,
        expm=pl.col('symbol').str.slice(-2,1),
    )
    .sort('ts', 'expy', 'expm')
    .group_by('ts')
    .agg(
        pl.col('symbol').first(),
        pl.col('open').first().cast(pl.Float32),
        pl.col('high').first().cast(pl.Float32),
        pl.col('low').first().cast(pl.Float32),
        pl.col('close').first().cast(pl.Float32),
        pl.col('volume').first(),
    )
    .select(['open','high','low','close','volume','ts','symbol'])
)

print(df)
df.write_parquet('spdr-sector-futures.parquet')

# %%
import itertools as it

mom_lookback = [1,3,7,14,21,30,60,90,120]
input_features = [
    # original
    "udpil",
    "udpim",
    "udpis",
    "upprob",
    "mbi",
    "tci",
    # derived
    "tcidelta",
    "mdcdelta",
    *[f"mom{n}d" for n in mom_lookback],
]

features = [
    *input_features,
    *[f"{f}roc" for f in input_features],
]

deriv_win = 7
zscore_win = 30

def rolling_zscore(expr, window):
    return (
        (expr - expr.rolling_mean(window, min_samples=1))
        / expr.rolling_std(window, min_samples=1)
    ).over("symbol")



df = (
     pl.read_parquet('polarity/data/*parquet')
    .rename(
        {
            "asset": "symbol",
            "price": "close",
            "timestamp": "ts",
        }
    )
    .with_columns(ts=pl.col("ts").dt.cast_time_unit("us"))
    .sort(["symbol", "ts"])
    # derive pct distance from *cv
    .with_columns(
        tcidelta=(pl.col("tcicv") - pl.col("close")) / pl.col("close"),
        mdcdelta=(pl.col("mdccv") - pl.col("close")) / pl.col("close"),
    )
    .with_columns(**{
        f'mom{n}d': pl.col('close') / pl.col('close').shift(n).over('symbol')
        for n in mom_lookback
    })  
    # derive 1st deriv, shift at most deriv_win, less if not enough history
    .with_columns(
        **{
            f"{col}roc": rolling_zscore(
                pl.col(col)
                - pl.coalesce(
                    [
                        pl.col(col).shift(i).over("symbol")
                        for i in range(deriv_win, 0, -1)
                    ]
                ),
                zscore_win,
            )
            for col in input_features
        }
    )
    # normalize
    .with_columns(**{col: rolling_zscore(pl.col(col), zscore_win) for col in features})
    .select(
        [
            "ts",
            "symbol",
            "close",
            *features,
        ]
    )
    .with_columns(**{
        f'fwd{n}d': pl.col('close').log().shift(-n).over('symbol') - pl.col('close').log()
        for n in [1,3,7,14,21,30]
    })
    .drop_nulls()
    .drop_nans()
    .group_by('symbol')
    .agg(**{
        f'{s}{n}d': pl.corr(pl.col(s), pl.col(f'fwd{n}d'), method='spearman')
        f'{s}{n}d': pl.corr(pl.col(s), pl.col(f'fwd{n}d'), method='spearman')
        for s,n in it.product(features, [1,3,7,14,21,30])
    })
    .mean()
    .unpivot()
    .drop_nulls()
    .with_columns(pl.col('value').cast(pl.Float32))
    .sort('value')
)

df.tail(10)

# %%
dtype = pl.Struct([
    pl.Field("signal", pl.String),
    pl.Field("gate", pl.String),
    pl.Field("interval_days", pl.String),
    pl.Field("max_long", pl.String),
    pl.Field("max_short", pl.String),
])

(
    pl.read_csv("results/*.csv")
    .with_columns(
        pl.col("parameters").str.json_decode(dtype)
    )
    .unnest('parameters')
    .with_columns(
        pl.col('interval_days').cast(pl.Int32),
        pl.col('max_long').cast(pl.Int32),
        pl.col('max_short').cast(pl.Int32),
    )
    .pivot(on='name', values='data')
    .select(['sortino','max_drawdown','cagr','final_equity','signal','gate','interval_days','max_long','max_short'])
    .filter(pl.col('sortino') > 1)
    .sort('sortino')
    
)

# %% [markdown]
# # Crypto momentum spreads (momentum-research)

# %%
dtype = pl.Struct([
    pl.Field("days_holding", pl.String),
    pl.Field("top_percentile", pl.String),
    pl.Field("bottom_percentile", pl.String),
    pl.Field("days_momentum", pl.String),
])

df = (
    pl.read_csv("momentum-research-backtest-results-20260422-185439/*.csv")
    .with_columns(
        pl.col("parameters").str.json_decode(dtype)
    )
    .unnest('parameters')
    .select(
        pl.col('days_holding').cast(pl.Int32),
        pl.col('top_percentile').cast(pl.Float32),
        pl.col('bottom_percentile').cast(pl.Float32),
        pl.col('days_momentum').cast(pl.Int32),
        pl.col('name'),
        pl.col('data'),
    )
    .pivot(on='name', values='data')
    .filter(pl.col('sortino') > 1)
    .sort('sortino')
)

df

# %% [markdown]
# Short holding periods combined with long momentum lookbacks perform the best. Short holding periods with short momentum lookbacks seem to perform worst.

# %%
(
    df.group_by('days_holding')
    .agg(
        ir=pl.col('ir').max()
    )
    .sort('ir')
)

# %% [markdown]
# The maximum IR per holding period shows again short periods are better.

# %%
(
    df
    .filter(pl.col('sortino') > 1)
    .sort('sortino')
)

# %% [markdown]
# A backtest without fees supports the idea that the shortest holding periods have the best risk adjusted returns.

# %%
import seaborn as sns

hm = (
    df.sort(['days_holding','days_momentum','sortino'])
    .group_by(['days_holding','days_momentum']).last()
    .pivot(
        index="days_holding",
        on="days_momentum",
        values="sortino"
    )
)
pddf = hm.to_pandas().set_index("days_holding").sort_index(ascending=False)
pddf.columns = pddf.columns.astype(int)
pddf = pddf.sort_index(axis=1)
pddf.columns.name = "days_momentum"
sns.heatmap(
    pddf,
    annot=True,
    fmt=".1f",
)
pddf

# %% [markdown]
# One day holding periods outperform.

# %% jupyter={"source_hidden": true}
(
    df.sort(['days_holding','sortino'])
    .group_by('days_holding')
    .last()
    .sort('days_holding')
    .to_pandas()
    .plot(x='days_holding', y='sortino', kind='bar')
)

# %% [markdown]
# Sortino falls as holding periods go up.

# %%
dtype = pl.Struct([
    pl.Field("days_holding", pl.String),
    pl.Field("top_percentile", pl.String),
    pl.Field("bottom_percentile", pl.String),
    pl.Field("days_momentum", pl.String),
])

df = (
    pl.read_csv("momentum-research-backtest-results-20260423-095445/*.csv")
    .with_columns(
        pl.col("parameters").str.json_decode(dtype)
    )
    .unnest('parameters')
    .select(
        pl.col('days_holding').cast(pl.Int32),
        pl.col('top_percentile').cast(pl.Float32),
        pl.col('bottom_percentile').cast(pl.Float32),
        pl.col('days_momentum').cast(pl.Int32),
        pl.col('name'),
        pl.col('data'),
    )
    .pivot(on='name', values='data')
    .filter(pl.col('sortino') > 1)
    .sort('sortino')
)

import seaborn as sns

hm = (
    df.sort(['days_holding','days_momentum','sortino'])
    .group_by(['days_holding','days_momentum']).last()
    .pivot(
        index="days_holding",
        on="days_momentum",
        values="sortino"
    )
)
pddf = hm.to_pandas().set_index("days_holding").sort_index(ascending=False)
pddf.columns = pddf.columns.astype(int)
pddf = pddf.sort_index(axis=1)
pddf.columns.name = "days_momentum"
sns.heatmap(
    pddf,
    annot=True,
    fmt=".1f",
)
pddf
df

# %%
dtype = pl.Struct([
    pl.Field("days_holding", pl.String),
    pl.Field("top_percentile", pl.String),
    pl.Field("bottom_percentile", pl.String),
    pl.Field("days_momentum", pl.String),
])

df = (
    pl.read_csv("momentum-research-backtest-results-20260423-101049/*.csv")
    .with_columns(
        pl.col("parameters").str.json_decode(dtype)
    )
    .unnest('parameters')
    .select(
        pl.col('days_holding').cast(pl.Int32),
        pl.col('top_percentile').cast(pl.Float32),
        pl.col('bottom_percentile').cast(pl.Float32),
        pl.col('days_momentum').cast(pl.Int32),
        pl.col('name'),
        pl.col('data'),
    )
    .pivot(on='name', values='data')
    .filter(pl.col('sortino') > 1)
    .sort('sortino')
)

import seaborn as sns

hm = (
    df.sort(['days_holding','days_momentum','sortino'])
    .group_by(['days_holding','days_momentum']).last()
    .pivot(
        index="days_holding",
        on="days_momentum",
        values="sortino"
    )
)
pddf = hm.to_pandas().set_index("days_holding").sort_index(ascending=False)
pddf.columns = pddf.columns.astype(int)
pddf = pddf.sort_index(axis=1)
pddf.columns.name = "days_momentum"
sns.heatmap(
    pddf,
    annot=True,
    fmt=".1f",
)
pddf
df

# %%
dtype = pl.Struct([
    pl.Field("days_holding", pl.String),
    pl.Field("top_percentile", pl.String),
    pl.Field("bottom_percentile", pl.String),
    pl.Field("days_momentum", pl.String),
])

df = (
    pl.read_csv("momentum-research-backtest-results-20260423-104824/*.csv")
    .with_columns(
        pl.col("parameters").str.json_decode(dtype)
    )
    .unnest('parameters')
    .select(
        pl.col('days_holding').cast(pl.Int32),
        pl.col('top_percentile').cast(pl.Float32),
        (
            pl.when(pl.col('bottom_percentile') == 'None')
            .then(pl.lit(None).alias('bottom_percentile'))
            .otherwise(pl.col('bottom_percentile'))
            .cast(pl.Float32)
        ),
        pl.col('days_momentum').cast(pl.Int32),
        pl.col('name'),
        pl.col('data'),
    )
    .pivot(on='name', values='data')
    .filter((pl.col('sortino') > 1) & (pl.col('days_holding') < pl.col('days_momentum')))
    .sort('sortino')
)

import seaborn as sns

hm = (
    df.sort(['days_holding','days_momentum','sortino'])
    .group_by(['days_holding','days_momentum']).last()
    .pivot(
        index="days_holding",
        on="days_momentum",
        values="sortino"
    )
)
pddf = hm.to_pandas().set_index("days_holding").sort_index(ascending=False)
pddf.columns = pddf.columns.astype(int)
pddf = pddf.sort_index(axis=1)
pddf.columns.name = "days_momentum"
sns.heatmap(
    pddf,
    annot=True,
    fmt=".1f",
)
pddf
df

# %% [markdown]
# **Todo**
#
# - [x] add fees
# - [ ] experiment with stops
# - [ ] check for lookahead bias
# - [ ] spread analysis for pd metrics
# - [ ] volume weight portfolio
# - [x] explore more around 30d holding period and 50d momentum
# - [ ] above mdc as gate
# - [ ] eval mean reversion vs trend following

# %%
pl.read_parquet('yf.parquet')

# %%
dtype = pl.Struct([
    pl.Field("days_holding", pl.String),
    pl.Field("n_buckets", pl.String),
    pl.Field("days_momentum", pl.String),
    pl.Field("ema_slow", pl.String),
    pl.Field("volume_cutoff", pl.String),
])

(
    pl.read_csv("cs-momentum-trail-resilts-1/*.csv")
    .with_columns(
        pl.col("parameters").str.json_decode(dtype)
    )
    .unnest('parameters')
    .with_columns(
        pl.col('days_holding').cast(pl.Int32),
        pl.col('n_buckets').cast(pl.Int32),
        pl.col('days_momentum').cast(pl.Int32),
        pl.col('ema_slow').cast(pl.Int32),
    )
    .pivot(on='name', values='data')
    .select(['sortino','max_drawdown','cagr','final_equity','days_holding','n_buckets','ema_slow','days_momentum','volume_cutoff'])
    .filter(pl.col('sortino') > 1)
    .sort('sortino')
    
).write_csv('cs-momentum-results-1.csv')

# %%
dtype = pl.Struct([
    pl.Field("days_holding", pl.String),
    pl.Field("n_buckets", pl.String),
    pl.Field("days_momentum", pl.String),
    pl.Field("ema_slow", pl.String),
    pl.Field("volume_cutoff", pl.String),
])

(
    pl.read_csv("cs-momentum-results-2/results/*.csv")
    .with_columns(
        pl.col("parameters").str.json_decode(dtype)
    )
    .unnest('parameters')
    .with_columns(
        pl.col('days_holding').cast(pl.Int32),
        pl.col('n_buckets').cast(pl.Int32),
        pl.col('days_momentum').cast(pl.Int32),
        pl.col('ema_slow').cast(pl.Int32),
    )
    .pivot(on='name', values='data')
    #.filter(pl.col('sortino') > 1)
    .sort('ir')
    
)

# %%
import polars as pl
from dotenv import load_dotenv
import os

load_dotenv()

so = {
    "aws_access_key_id": os.getenv("R2_ACCESS_KEY"),
    "aws_secret_access_key": os.getenv("R2_SECRET_KEY"),
    "aws_endpoint_url": f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
    "aws_region": "auto",  # R2 requires 'auto' or 'us-east-1' (aliased to auto)
}


frag = []
for yr in range(2023,2027):
    u = f"s3://studies-binance-store/spot-1m/year={yr}/"
    frag.append(pl.scan_parquet(u, storage_options=so, hive_partitioning=True))

(
    pl.concat(frag)
    .filter(pl.col('symbol').str.ends_with('USDC'))
    .sink_parquet('stables-1m.parquet')
)

# %%
import polars as pl
    
(
    pl.scan_parquet('stables-1m.parquet')
    .sort(['symbol','ts'])
    .with_columns(
        vol=(pl.col('quote_volume') - pl.col('quote_volume').rolling_mean(24*60).over('symbol')) / pl.col('quote_volume').rolling_std(24*60).over('symbol'),
        fwd15=pl.col('close').pct_change(-1).over('symbol'),
        fwd15=pl.col('close').pct_change(-15).over('symbol'),
        fwd30=pl.col('close').pct_change(-30).over('symbol'),
        fwd45=pl.col('close').pct_change(-45).over('symbol'),
        fwd60=pl.col('close').pct_change(-60).over('symbol'),
        fwd90=pl.col('close').pct_change(-90).over('symbol'),
        fwd120=pl.col('close').pct_change(-120).over('symbol'),
        fwd240=pl.col('close').pct_change(-240).over('symbol'),
        fwd360=pl.col('close').pct_change(-360).over('symbol'),
        fwd480=pl.col('close').pct_change(-480).over('symbol'),
        fwd600=pl.col('close').pct_change(-600).over('symbol'),
        fwd1200=pl.col('close').pct_change(-1200).over('symbol'),
        fwd2400=pl.col('close').pct_change(-2400).over('symbol'),
        fwd3600=pl.col('close').pct_change(-3600).over('symbol'),
        fwd4800=pl.col('close').pct_change(-4800).over('symbol'),
        fwd6000=pl.col('close').pct_change(-6000).over('symbol'),
        mom=pl.col('close').pct_change(15).over('symbol')
    )
    .filter((pl.col('mom') >= 0.2) & (pl.col('vol') > 2))
).sink_csv('stats.csv')

# %%
import polars as pl
import matplotlib.pyplot as plt

mins = [1,15,30,45,60,90,120,240,360,480,600,1200,2400,3600,4800,6000]
df = pl.read_csv('stats.csv', try_parse_dates=True).group_by(pl.col('ts').dt.year())
avg = df.agg([pl.col(f'fwd{n}').mean() for n in mins]).unpivot(index="ts", on=[f"fwd{n}" for n in mins])
std = df.agg([pl.col(f'fwd{n}').std() for n in mins]).unpivot(index="ts", on=[f"fwd{n}" for n in mins])

df = (
    avg
    .rename({'value':'mean'})
    .join(
        std.rename({'value':'std'}),
        on=['ts','variable']
    )
    .with_columns(
        minutes=pl.col('variable').str.extract(r"(\d+)", 1).cast(pl.Int32)
    )
)
print(df)

plt.figure(figsize=(18, 8))

for year in df['ts'].unique().to_list():
    # Filter data for the specific year
    year_data = df.filter(pl.col("ts") == year)
    
    plt.plot(
        year_data["minutes"], 
        year_data["mean"], 
        marker='o', 
        markersize=4,
        label=f"Year {year}"
    )
plt.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
# 3. Formatting
plt.xscale('log')  # Log scale makes 15 vs 6000 readable
plt.xlabel("Forward Minutes (Log Scale)")
plt.ylabel("Mean Value")
plt.title("Mean Stats by Year")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.show()

# %%
pl.read_parquet('stables-1d.parquet')
df = (
    pl.read_parquet('polarity/data/*.parquet')
    .rename({'asset': 'symbol'})
    .sort(['symbol','ts'])
    .with_columns(
        fwd=pl.col('price').pct_change(-30).over('symbol'),
        mom=pl.col('price').pct_change(30*3).over('symbol'),
    )
    .filter(pl.col('mom').is_not_null() & pl.col('fwd').is_not_null())
    .with_columns(
        rank=pl.col('mom').qcut(
            10,
            labels=[str(i) for i in range(10)],
            allow_duplicates=True
        ).over('ts'),
    )
    .filter((pl.col('rank') == '9') & (pl.col('mom') > 0) & (pl.col('ts').dt.year() >= 2023))
    .select(['ts','symbol','rank','mom','fwd'])
    .group_by('ts')
    .agg(
        pl.col('fwd').mean()
    )
)

print(df.mean(), df.std())

# %%
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# 1. Your original backtest logic
df = (
    pl.scan_parquet('polarity/data/*.parquet')  # Using scan for better memory management
    .rename({'asset': 'symbol'})
    .sort(['symbol','ts'])
    .with_columns(
        fwd=pl.col('price').pct_change(-30).over('symbol'),
        mom=pl.col('price').pct_change(30*3).over('symbol'),
    )
    .filter(pl.col('mom').is_not_null() & pl.col('fwd').is_not_null())
    .with_columns(
        rank=pl.col('mom').qcut(
            10,
            labels=[str(i) for i in range(10)],
            allow_duplicates=True
        ).over('ts'),
    )
    .filter((pl.col('rank') == '9') & (pl.col('mom') > 0) & (pl.col('ts').dt.year() >= 2023))
    .select(['ts','symbol','rank','mom','fwd'])
    .group_by('ts')
    .agg(
        pl.col('fwd').mean()
    )
    .collect() # Execute the query
)

# ... [Keep Step 1 exactly the same to get your df] ...

# 2. Extract returns
returns = df.drop_nulls('fwd')['fwd'].to_numpy()

# NEW: Filter out extreme outliers using percentiles
# This removes the craziest 1% of spikes and drops
lower_bound = np.percentile(returns, 1)
upper_bound = np.percentile(returns, 99)
clean_returns = returns[(returns >= lower_bound) & (returns <= upper_bound)]

# Calculate moments on the CLEANED data
mu = np.mean(clean_returns)
sigma = np.std(clean_returns)

print(f"Cleaned Mean: {mu:.4f}, Cleaned Std Dev: {sigma:.4f}")

# 3. Build the Plot (using clean_returns)
plt.figure(figsize=(10, 6))

# Histogram using the cleaned data
plt.hist(clean_returns, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')

# Generate x-values for the bell curve based on the new, sane limits
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

# Calculate and plot the PDF
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution Fit')

# 4. Mark the specific lines
plt.axvline(0, color='black', linestyle='-', linewidth=1.5, label='Zero')
plt.axvline(mu, color='red', linestyle='--', linewidth=2, label=f'Mean ({mu:.4f})')

# Plus 1 and 2 Sigma
plt.axvline(mu + sigma, color='green', linestyle='-.', linewidth=2, label='+1 Sigma')
plt.axvline(mu + 2*sigma, color='purple', linestyle=':', linewidth=2, label='+2 Sigma')

# (Optional but recommended) Minus 1 and 2 Sigma for visual symmetry
plt.axvline(mu - sigma, color='green', linestyle='-.', linewidth=1, alpha=0.5)
plt.axvline(mu - 2*sigma, color='purple', linestyle=':', linewidth=1, alpha=0.5)

# 5. Formatting
plt.title('Histogram of Mean Forward Returns (Top Decile, Positive Momentum)')
plt.xlabel('30-Day Forward Return')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Display the plot
plt.show()

# %%
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# 1. Define the holding periods we want to look forward
horizons = [1, 3, 7, 14, 21, 30, 40, 50]

# Generate a list of Polars expressions for each horizon
fwd_exprs = [
    pl.col('price').pct_change(-h).over('symbol').alias(f'fwd_{h}d') 
    for h in horizons
]

# 2. Polars Data Pipeline
df = (
    pl.scan_parquet('polarity/data/*.parquet')
    .rename({'asset': 'symbol'})
    .sort(['symbol','ts'])
    .with_columns(
        mom = pl.col('price').pct_change(90).over('symbol'), # 90-day momentum
        *fwd_exprs # Unpack the list of forward return expressions
    )
    .filter(pl.col('mom').is_not_null())
    .with_columns(
        rank=pl.col('mom').qcut(
            10,
            labels=[str(i) for i in range(10)],
            allow_duplicates=True
        ).over('ts'),
    )
    .filter((pl.col('rank') == '9') & (pl.col('mom') > 0))
    # Select only the columns we need for plotting to save memory
    .select([f'fwd_{h}d' for h in horizons]) 
    .collect()
)

# 3. Calculate Cleaned Means and Standard Deviations
means = []
stds = []

for h in horizons:
    col_name = f'fwd_{h}d'
    
    # Drop nulls for this specific horizon (e.g., the last 50 days of the dataset)
    returns = df.drop_nulls(col_name)[col_name].to_numpy()
    
    # Apply the outlier filter (1st and 99th percentiles) to clean the data
    if len(returns) > 0:
        lower_bound = np.percentile(returns, 1)
        upper_bound = np.percentile(returns, 99)
        clean_returns = returns[(returns >= lower_bound) & (returns <= upper_bound)]
        
        means.append(np.mean(clean_returns))
        
        # NOTE: We are plotting Standard Deviation. If you want Standard Error of the Mean (SEM) 
        # to show confidence intervals instead, use: np.std(clean_returns) / np.sqrt(len(clean_returns))
        stds.append(np.std(clean_returns)) 
    else:
        means.append(0)
        stds.append(0)

# 4. Build the Error Bar Plot
plt.figure(figsize=(10, 6))

# plt.errorbar handles both the line plot and the vertical bars
plt.errorbar(
    x=horizons, 
    y=means, 
    yerr=stds, 
    fmt='-o',           # Line with circle markers
    color='steelblue', 
    ecolor='black',     # Color of the error bars
    capsize=5,          # Width of the caps on the error bars
    capthick=1.5,
    markersize=6,
    label='Mean Return ± 1 Std Dev'
)

# Add a reference line at zero
plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Return')

# Formatting
plt.title('Term Structure of Forward Returns (Top Decile, Positive Momentum)')
plt.xlabel('Holding Period (Days)')
plt.ylabel('Return')
plt.xticks(horizons) # Force x-axis to show our specific horizons
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Display the plot
plt.show()

# %%
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# 1. Base Data Pipeline
# Assuming 'mcap' is available in your raw data
df_base = (
    pl.scan_parquet('polarity/latest-data/*.parquet')
    .rename({'asset': 'symbol'})
    .sort(['symbol','ts'])
    .with_columns(
        fwd=pl.col('price').pct_change(-30).over('symbol'),
        mom=pl.col('price').pct_change(90).over('symbol'), # 90-day momentum
    )
    .filter(pl.col('mom').is_not_null() & pl.col('fwd').is_not_null())
    .with_columns(
        rank=pl.col('mom').qcut(
            10,
            labels=[str(i) for i in range(10)],
            allow_duplicates=True
        ).over('ts'),
    )
    .filter((pl.col('rank') == '9') & (pl.col('mom') > 0) & (pl.col('ts').dt.year() >= 2023))
    # We must explicitly select 'mcap' so it is available for filtering below
    .select(['ts', 'symbol', 'market_cap', 'fwd']) 
    .collect()
)

# 2. Define Market Cap Buckets (Assuming mcap is in raw dollars)
# If your data stores mcap in millions, adjust these thresholds (e.g., 10000 instead of 10_000_000_000)
buckets = {
    "Mega Cap (>= $10B)": df_base.filter(pl.col('market_cap') >= 10_000_000_000),
    "Mid Cap ($1B - $10B)": df_base.filter((pl.col('market_cap') >= 1_000_000_000) & (pl.col('market_cap') < 10_000_000_000)),
    "All Coins": df_base
}

# 3. Setup the Subplots
# sharex=True and sharey=True are critical so the visual scale matches across all 3 charts
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
fig.suptitle('30-Day Forward Returns (Top Decile Momentum) by Market Cap', fontsize=16)

colors = ['forestgreen', 'darkorange', 'steelblue']

# 4. Process and Plot Each Bucket
for ax, (title, subset_df), color in zip(axes, buckets.items(), colors):
    
    # Calculate daily mean returns for this specific mcap bucket
    daily_returns_df = (
        subset_df.group_by('ts')
        .agg(pl.col('fwd').mean())
        .drop_nulls('fwd')
    )
    
    returns = daily_returns_df['fwd'].to_numpy()
    
    # Outlier handling (1st and 99th percentile) to ensure clean bell curves
    if len(returns) > 0:
        lower_bound = np.percentile(returns, 1)
        upper_bound = np.percentile(returns, 99)
        clean_returns = returns[(returns >= lower_bound) & (returns <= upper_bound)]
        
        mu = np.mean(clean_returns)
        sigma = np.std(clean_returns)
        
        # Histogram
        ax.hist(clean_returns, bins=40, density=True, alpha=0.6, color=color, edgecolor='black')
        
        # Bell Curve
        xmin, xmax = ax.get_xlim() # Get axis limits to draw the line
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, sigma)
        ax.plot(x, p, 'k', linewidth=2)
        
        # Markings
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.axvline(mu, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mu:.4f}')
        ax.axvline(mu + sigma, color='green', linestyle=':', linewidth=1.5, label='+1 Sigma')
        ax.axvline(mu - sigma, color='green', linestyle=':', linewidth=1.5)
        
        ax.set_title(f'{title}\nStd Dev: {sigma:.4f}')
        ax.legend()
    else:
        ax.set_title(f'{title}\n(Not Enough Data)')
    
    ax.set_xlabel('30-Day Forward Return')
    ax.grid(True, alpha=0.3)

# Formatting the Y-axis only on the first chart to reduce clutter
axes[0].set_ylabel('Density')
plt.tight_layout()
plt.show()

# %%
parameters = [
    "holding_days",
    "long_decile",
    "short_decile",
    "momentum_days",
    "min_mcap",
]

dtype = pl.Struct([pl.Field(p, pl.String) for p in parameters])

df = (
    pl.read_csv("pd-momentum-results-20260505-152721/*.csv")
    .with_columns(
        pl.col("parameters").str.json_decode(dtype)
    )
    .unnest('parameters')
    .select([
        *[pl.col(p).cast(pl.Float32) for p in parameters],
        pl.col('name'),
        pl.col('data'),
    ])
    .pivot(on='name', values='data')
    .filter(pl.col('sortino') > 1)
    .sort('sortino')
)

df


# %%
import polars as pl

pl.read_csv('cmc_historical_data_2020.csv')

# %%
import polars as pl

(
    pl.read_csv('cmc_historical_data_2020.csv')
    .join(pl.read_csv('cmc_listings.csv'), on=['slug'])
    .select(
        ts=pl.col('time_open').str.to_datetime(format="%Y-%m-%d %H:%M:%S+00:00").dt.replace_time_zone("UTC"),
        ticker=pl.col('symbol').str.to_lowercase(),
        open=pl.col('open'),
        high=pl.col('high'),
        low=pl.col('low'),
        close=pl.col('close'),
        volume=pl.col('volume'),
        market_cap=pl.col('market_cap'),
        circulating_supply=pl.col('circulating_supply'),
        symbol=pl.col('slug'),
        listed=pl.col('date_added').str.to_datetime(format="%Y-%m-%dT%H:%M:%S.000Z").dt.replace_time_zone("UTC"),
        is_active=pl.col('is_active') > 0,
        market_pairs=pl.col('market_pair_count'),
    )
    .sort(['symbol','ts'])
    .unique(['symbol','ts']) # bug in cmc.py?
    .write_parquet('cmc-usd-1d-2020-2026.parquet')
)


# %%
pl.read_csv('cmc_listings.csv')


# %%
days_momentum = 30
days_skip = 4
days_holding = 14
days_trend = 60

blacklist = ['terra-luna','ftx-token']
benchmark = 'bitcoin'
winsorize = [-1, 5]
mom_buckets = 10
mom_ranks = ['9']
min_volume = 1_000_000
min_mcap = 1_000_000
start_year = 2020

df = (
    pl.read_parquet('cmc-usd-1d-2020-2026.parquet')
    .filter(
        (pl.col('symbol').is_in(blacklist).not_()) &
        (pl.col('market_cap') > 0) &
        (pl.col('is_active')) &
        (pl.col('market_pairs') > 1)
    )
    .join(
        pl.read_parquet('cmc-usd-1d-2020-2026.parquet')
        .filter(pl.col('symbol') == benchmark)
        .select(
            ts=pl.col('ts'),
            ref=pl.col('close')
        ),
        on='ts'
    )
    .sort(['symbol','ts'])
    .with_columns(
        fwd=(pl.col('close').shift(-1).over('symbol') / pl.col('close') - 1).clip(*winsorize),
        mom=pl.col('close').shift(days_skip) / pl.col('close').shift(days_momentum).over('symbol') - 1,
        momdays=(pl.col('ts').shift(days_skip) - pl.col('ts').shift(days_momentum).over('symbol')).dt.total_days(),
        trend=pl.col('ref').rolling_mean(days_trend).over('symbol'),
    )
    .filter(
        (pl.col('mom').is_not_null()) &
        (pl.col('momdays') == days_momentum - days_skip)
    )
    .with_columns(
        rank=pl.col('mom').qcut(
            mom_buckets,
            labels=[str(i) for i in range(mom_buckets)],
            allow_duplicates=True
        ).over('ts'),
    )
    .filter(
        (pl.col('rank').is_in(mom_ranks)) &
        (pl.col('mom') > 0) &
        (pl.col('ts').dt.year() >= start_year) &
        (pl.col('volume') > min_volume) &
        (pl.col('market_cap') > min_mcap) &
        (pl.col('trend') < pl.col('close'))
    )
    .select(['ts','symbol','rank','mom','fwd','market_cap'])
    .group_by('ts')
    .agg(
        pl.col('fwd').mean(),
        pl.col('market_cap').mean(),
    )
)

print(f'''
Average return per position for {days_holding}d: {df.mean()['fwd'][0] * 100}%
Annualized                           {((df.mean()['fwd'][0] + 1)**(365/days_holding) - 1) * 100}%
Median                               {df.median()['fwd'][0] * 100}%
Sigma                                {df.std()['fwd'][0] * 100}%
''')

# %%
pl.read_parquet('cmc-usd-1d-2020-2026.parquet')

# %%
import polars as pl
import numpy as np

# --- Configuration ---
HOLDING_PERIOD = 30
POSITIONS_PER_SUB_PORTFOLIO = 20
FEE_PER_LEG = 0.0005 
ANNUALIZATION_FACTOR = 365
MIN_MARKET_CAP_RANK = 300
YEAR = 2026

df = (
    pl.read_parquet('cmc-usd-1d-2020-2026.parquet')
    .sort(['symbol', 'ts'])
    
    # 1. Base defensive filter (prevents division by zero on corrupted prices)
    #.filter(pl.col('close') > 0.0001) 
    
    .with_columns([
        # 2. 90-day momentum (Needs continuous history, computed for ALL coins)
        (pl.col('close') / pl.col('close').shift(90).over('symbol') - 1).alias('mom'),

        # 3. 1-day forward return with Smart Delisting Protection
        (pl.col('close').shift(-1).over('symbol') / pl.col('close') - 1)
        .fill_null(
            pl.when(pl.col('ts') == pl.col('ts').max())
            .then(0.0)  # The dataset ended, so the return is flat
            .otherwise(-1.0) # The coin died before the dataset ended, 100% loss
        )
        .clip(lower_bound=-1.0, upper_bound=4.0)
        .alias('fwd_ret_1d')
    ])
    
    # 4. UNIVERSE SELECTION: Rank Market Cap per day
    .with_columns(
        mc_rank = pl.col('market_cap').rank(descending=True, method='ordinal').over('ts')
    )
    
    # 5. MASKING: Isolate the momentum of ONLY the top coins
    .with_columns(
        top_mom = pl.when(pl.col('mc_rank') <= MIN_MARKET_CAP_RANK)
        .then(pl.col('mom'))
        .otherwise(None) # Coins outside top get None, making them invisible to the next ranking step
    )
    
    # 6. MOMENTUM RANKING: Rank momentum only within the Top universe
    .with_columns(
        mom_rank = pl.col('top_mom').rank(descending=True, method='ordinal').over('ts')
    )
    
    # 7. SELECTION: Binary mask for the Top 20 momentum coins (from the Top universe)
    .with_columns(
        is_selected = pl.when(
            (pl.col('mom_rank') <= POSITIONS_PER_SUB_PORTFOLIO) & 
            (pl.col('top_mom') > 0) # Ensure momentum is positive
        ).then(1).otherwise(0)
    )
    
    # 8. SUB-WEIGHTING
    .with_columns(
        sub_weight = pl.col('is_selected') / POSITIONS_PER_SUB_PORTFOLIO
    )
    
    # 9. J&T OVERLAP: Aggregate weight across 30 active sub-portfolios
    .with_columns(
        total_weight = (
            pl.col('sub_weight').rolling_sum(window_size=HOLDING_PERIOD).over('symbol') / HOLDING_PERIOD
        )
    )
)

# --- Performance Calculation ---
strategy_returns = (
    df.filter(pl.col('ts').dt.year() >= YEAR)
    .group_by('ts')
    .agg(
        strategy_ret_1d = (pl.col('total_weight') * pl.col('fwd_ret_1d')).sum()
    )
    .sort('ts')
)

daily_fee_drag = (FEE_PER_LEG * 2) / HOLDING_PERIOD
strategy_results = strategy_returns.with_columns(
    net_daily_ret = pl.col('strategy_ret_1d') - daily_fee_drag
)

total_days = strategy_results.height
cumulative_return = (strategy_results['net_daily_ret'] + 1).product()
cagr = (cumulative_return ** (ANNUALIZATION_FACTOR / total_days)) - 1

mean_net_ret = strategy_results['net_daily_ret'].mean()
downside_deviation = strategy_results.filter(pl.col('net_daily_ret') < 0)['net_daily_ret'].std()
sortino_ratio = (mean_net_ret / downside_deviation) * np.sqrt(ANNUALIZATION_FACTOR)
# --- 10. Advanced Risk Metrics & Benchmark Comparison ---

# Extract BTC's 1-day forward returns from the main DataFrame to use as the benchmark
btc_benchmark = (
    df.filter((pl.col('symbol') == 'bitcoin') & (pl.col('ts').dt.year() >= YEAR))
    .select(
        ts=pl.col('ts'),
        btc_ret_1d=pl.col('fwd_ret_1d')
    )
)

# Join BTC returns to strategy returns
strategy_results = strategy_results.join(btc_benchmark, on='ts', how='left')

# Calculate Active Returns (Strategy Net Return - BTC Return)
strategy_results = strategy_results.with_columns(
    active_ret = pl.col('net_daily_ret') - pl.col('btc_ret_1d')
)

# --- MAX DRAWDOWN CALCULATION ---
# Strategy Drawdown
strategy_results = strategy_results.with_columns(
    cum_return = (pl.col('net_daily_ret') + 1).cum_prod()
).with_columns(
    rolling_peak = pl.col('cum_return').cum_max()
).with_columns(
    drawdown = (pl.col('cum_return') / pl.col('rolling_peak')) - 1
)
strat_mdd = strategy_results['drawdown'].min()

# BTC Drawdown
strategy_results = strategy_results.with_columns(
    btc_cum_return = (pl.col('btc_ret_1d') + 1).cum_prod()
).with_columns(
    btc_rolling_peak = pl.col('btc_cum_return').cum_max()
).with_columns(
    btc_drawdown = (pl.col('btc_cum_return') / pl.col('btc_rolling_peak')) - 1
)
btc_mdd = strategy_results['btc_drawdown'].min()

# --- INFORMATION RATIO & BENCHMARK CAGR ---
# BTC CAGR
btc_total_days = strategy_results['btc_ret_1d'].drop_nulls().len()
btc_cum_return_final = (strategy_results['btc_ret_1d'].drop_nulls() + 1).product()
btc_cagr = (btc_cum_return_final ** (ANNUALIZATION_FACTOR / btc_total_days)) - 1

# Information Ratio
mean_active_ret = strategy_results['active_ret'].mean()
tracking_error = strategy_results['active_ret'].std()

information_ratio = (mean_active_ret / tracking_error) * np.sqrt(ANNUALIZATION_FACTOR)
# --- 11. Position Sizing: The Kelly Criterion ---

# Using the continuous approximation for Kelly: f = Expected Excess Return / Variance
# Assuming the Risk-Free Rate = 0 (standard in native crypto strategies)
daily_variance = strategy_results['net_daily_ret'].var()

# We annualize both the mean and the variance
annualized_mean_ret = mean_net_ret * ANNUALIZATION_FACTOR
annualized_variance = daily_variance * ANNUALIZATION_FACTOR

# Full Kelly Fraction
kelly_fraction = annualized_mean_ret / annualized_variance

# Half Kelly (The institutional standard for real-world trading)
half_kelly = kelly_fraction / 2

# --- UPDATED FINAL PRINT OUT ---
print(f"--- Strategy vs Buy-and-Hold BTC (Since {YEAR}) ---")
print(f"Strategy CAGR:     {cagr:.2%}")
print(f"BTC BnH CAGR:      {btc_cagr:.2%}\n")

print(f"Strategy MDD:      {strat_mdd:.2%}")
print(f"BTC BnH MDD:       {btc_mdd:.2%}\n")

print(f"Information Ratio: {information_ratio:.2f}")
print(f"Sortino Ratio:     {sortino_ratio:.2f}\n")

print(f"--- Recommended Allocation ---")
print(f"Full Kelly:        {kelly_fraction:.2%}")
print(f"Half Kelly:        {half_kelly:.2%}")

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

# --- 1. Prepare Data ---
dates = strategy_results['ts'].to_numpy()
strat_equity = (strategy_results['net_daily_ret'].fill_null(0) + 1).cum_prod().to_numpy()
btc_equity = (strategy_results['btc_ret_1d'].fill_null(0) + 1).cum_prod().to_numpy()
strat_rets = strategy_results['net_daily_ret'].drop_nulls().to_numpy()

# --- 2. Calculate Statistical Moments ---
mean_ret = np.mean(strat_rets)
std_ret = np.std(strat_rets)
variance = np.var(strat_rets)          # 2nd Moment
skewness = stats.skew(strat_rets)      # 3rd Moment
# We use Fisher’s definition (Excess Kurtosis), where Normal = 0
excess_kurtosis = stats.kurtosis(strat_rets)  # 4th Moment

# --- 3. Fit Johnson SU Distribution (Matches all 4 Moments) ---
# This is the "Gold Standard" for fitting financial return distributions
# a, b are shape parameters (skew/kurtosis), loc is center, scale is width
a_fit, b_fit, loc_fit, scale_fit = stats.johnsonsu.fit(strat_rets)

# --- 4. Plotting ---
sns.set_theme(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1.5, 1]})

# --- Plot 1: Log Equity Curve ---
ax1.plot(dates, strat_equity, label='Strategy (Top 20)', color='#1f77b4', linewidth=2)
ax1.plot(dates, btc_equity, label='BTC Buy & Hold', color='#ff7f0e', linestyle='--', alpha=0.8)
ax1.set_yscale('log')
ax1.set_title(f'Equity Growth of $1 Since {YEAR} (Log Scale)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Cumulative Value (USD)')
ax1.legend(loc='upper left')

# --- Plot 2: Distribution with 4-Moment Fitted Curve ---
# Using more bins to show the "peaky" nature (Kurtosis)
sns.histplot(strat_rets, bins=150, stat='density', color='royalblue', alpha=0.3, ax=ax2)

# Generate fitted Johnson SU PDF
x_axis = np.linspace(strat_rets.min(), strat_rets.max(), 1000)
johnson_pdf = stats.johnsonsu.pdf(x_axis, a_fit, b_fit, loc_fit, scale_fit)
ax2.plot(x_axis, johnson_pdf, color='darkred', linewidth=2.5, label='Fitted Johnson SU (4-Moment)')

# Reference lines
ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax2.axvline(mean_ret, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ret:.2%}')
ax2.axvline(mean_ret + std_ret, color='green', linestyle=':', linewidth=1.5, label='+1$\sigma$')
ax2.axvline(mean_ret - std_ret, color='green', linestyle=':', linewidth=1.5, label='-1$\sigma$')

# Stats box with all 4 moments
stats_text = (
    f"1st (Mean): {mean_ret:.4f}\n"
    f"2nd (Var):  {variance:.6f}\n"
    f"3rd (Skew): {skewness:.3f}\n"
    f"4th (Kurt): {excess_kurtosis:.3f}"
)
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes, fontsize=11, 
         verticalalignment='top', bbox=props, family='monospace')

ax2.set_title('Return Distribution vs. 4-Moment Fitted Curve (Johnson SU)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Daily Net Return')
ax2.set_ylabel('Probability Density')
ax2.legend(loc='upper right')

# Zoom in slightly on the center to see the fit better, while keeping tails visible
ax2.set_xlim(mean_ret - 4*std_ret, mean_ret + 6*std_ret)

plt.tight_layout()
plt.show()

# %%
bps_fee = 5
days_holding = 14
num_buckets = 4
days_delta = 7
start_year = 2023

df = (
    pl.read_parquet('polarity/latest-data/*.parquet',missing_columns='insert')
    .sort(['asset','ts'])
    .with_columns(
        fwd=(pl.col('price').shift(-days_holding).over('asset') / pl.col('price') - 1) - (bps_fee / 100.0 / 100.0),
        tcidelta=pl.col('tci') - pl.col('tci').shift(days_delta).over('asset'),

    )
    .filter(
        (pl.col('ts').dt.year() >= start_year) &
        (pl.col('total_volume') >= 1_000_000)
    )
    #.with_columns(**{
    #    c: pl.col(c).qcut(
    #        num_buckets,
    #        labels=[str(i) for i in range(num_buckets)],
    #        allow_duplicates=True
    #    ).over('ts') for c in ['udpil','udpim','udpis','mdccv','mbi','tci','mtm','mcm','tcicv','upprob']
    #})

    # 4.32%
    #.filter(
    #    (pl.col('upprob') > 0.75) &
    #    (pl.col('mdccv') < pl.col('price')) &
    #    (pl.col('tcidelta') > 0)
    #)

    .with_columns(
        ret=pl.when(
            (pl.col('udpim') < pl.col('udpis')) &
            (pl.col('upprob') > 0.75) &
            (pl.col('mdccv') < pl.col('price')) &
            (pl.col('tcidelta') > 0)
        ).then(pl.col('fwd')).otherwise(None)
    )
    .group_by('ts')
    .agg(
        pl.col('fwd').mean(),
        pl.col('ret').mean(),
    )
)

print(f'''
strategy / market
mean: {df.mean()['ret'][0]} / {df.mean()['fwd'][0]}
annualized: {np.pow(df.mean()['ret'][0] + 1, 365/days_holding) - 1} / {np.pow(df.mean()['fwd'][0] + 1, 365/days_holding) - 1}
stdev: {df.std()['ret'][0]} / {df.std()['fwd'][0]}
''')

#df.group_by(pl.col('ts').dt.strftime('%Y-%m')).mean().sort('ts').to_pandas().plot(x='ts',y='fwd',kind='bar')
df.sort('ts').to_pandas().plot(x='ts',y=['ret','fwd'])

# %%
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# 1. Base Data Pipeline
# Assuming 'mcap' is available in your raw data
df_base = (
    #pl.scan_parquet('polarity/latest-data/*.parquet')
    #.rename({'asset': 'symbol'})
    pl.scan_parquet("cmc-usd-1d-2020-2026.parquet")
    .with_columns(
        price=pl.col('close'),
    )
    .sort(['symbol','ts'])
    .with_columns(
        fwd=pl.col('price').pct_change(-30).over('symbol'),
        mom=pl.col('price').pct_change(90).over('symbol'), # 90-day momentum
    )
    .filter(pl.col('mom').is_not_null() & pl.col('fwd').is_not_null())
    .with_columns(
        rank=pl.col('mom').qcut(
            10,
            labels=[str(i) for i in range(10)],
            allow_duplicates=True
        ).over('ts'),
    )
    .filter((pl.col('rank') == '9') & (pl.col('mom') > 0) & (pl.col('ts').dt.year() >= 2023))
    # We must explicitly select 'mcap' so it is available for filtering below
    .select(['ts', 'symbol', 'market_cap', 'fwd']) 
    .collect()
)

# 2. Define Market Cap Buckets (Assuming mcap is in raw dollars)
# If your data stores mcap in millions, adjust these thresholds (e.g., 10000 instead of 10_000_000_000)
buckets = {
    "Mega Cap (>= $500B)": df_base.filter(pl.col('market_cap') >= 500_000_000_000),
    "Mid Cap ($100B - $500B)": df_base.filter((pl.col('market_cap') >= 100_000_000_000) & (pl.col('market_cap') < 500_000_000_000)),
    "All Coins": df_base
}

# 3. Setup the Subplots
# sharex=True and sharey=True are critical so the visual scale matches across all 3 charts
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
fig.suptitle('30-Day Forward Returns (Top Decile Momentum) by Market Cap', fontsize=16)

colors = ['forestgreen', 'darkorange', 'steelblue']

# 4. Process and Plot Each Bucket
for ax, (title, subset_df), color in zip(axes, buckets.items(), colors):
    
    # Calculate daily mean returns for this specific mcap bucket
    daily_returns_df = (
        subset_df.group_by('ts')
        .agg(pl.col('fwd').mean())
        .drop_nulls('fwd')
    )
    
    returns = daily_returns_df['fwd'].to_numpy()
    
    # Outlier handling (1st and 99th percentile) to ensure clean bell curves
    if len(returns) > 0:
        lower_bound = np.percentile(returns, 1)
        upper_bound = np.percentile(returns, 99)
        clean_returns = returns[(returns >= lower_bound) & (returns <= upper_bound)]
        
        mu = np.mean(clean_returns)
        sigma = np.std(clean_returns)
        
        # Histogram
        ax.hist(clean_returns, bins=40, density=True, alpha=0.6, color=color, edgecolor='black')
        
        # Bell Curve
        xmin, xmax = ax.get_xlim() # Get axis limits to draw the line
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, sigma)
        ax.plot(x, p, 'k', linewidth=2)
        
        # Markings
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.axvline(mu, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mu:.4f}')
        ax.axvline(mu + sigma, color='green', linestyle=':', linewidth=1.5, label='+1 Sigma')
        ax.axvline(mu - sigma, color='green', linestyle=':', linewidth=1.5)
        
        ax.set_title(f'{title}\nStd Dev: {sigma:.4f}')
        ax.legend()
    else:
        ax.set_title(f'{title}\n(Not Enough Data)')
    
    ax.set_xlabel('30-Day Forward Return')
    ax.grid(True, alpha=0.3)

# Formatting the Y-axis only on the first chart to reduce clutter
axes[0].set_ylabel('Density')
plt.tight_layout()
plt.show()

# %%
dtype = pl.Struct([
    pl.Field("days_holding", pl.String),
    pl.Field("top_percentile", pl.String),
    pl.Field("bottom_percentile", pl.String),
    pl.Field("days_momentum", pl.String),
])

(
    pl.read_csv("momentum-research-backtest-results-20260512-134132/*.csv")
    .with_columns(
        pl.col("parameters").str.json_decode(dtype)
    )
    .unnest('parameters')
    .with_columns(
        pl.col('days_holding').cast(pl.Int32),
        pl.col('top_percentile').cast(pl.Float32),
        pl.col('days_momentum').cast(pl.Int32),
    )
    .pivot(on='name', values='data')
    #.select(['sortino','maxdd','cagr','final_equity','signal','gate','interval_days','max_long','max_short'])
    #.filter((pl.col('sortino') > 1) & (pl.col('top_percentile') < 0.99)) 
    .sort('sortino')
    
).write_csv('res.csv')
