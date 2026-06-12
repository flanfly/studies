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

# %% [markdown]
# Performance based on cross-sectional volatility
# ===============================================
#
# Expected Returns, Figure 19.5 (a) documents
# the consistent outperformance of low-volatility stocks over high-volatility stocks in the
# Ang et al. (2006 ) study, based on a broad cross-section of U.S. stocks from 1963 to 2000.
# Figure 19.5 ( b ) shows the returns of high-volatility and low-volatility quintiles separately
# and adds more recent evidence from Robeco's David Blitz and UBS's Giuliano De Rossi. Blitz's
# data are on the large-cap universe since 1986; the results are weaker but low-volatility
# stocks still outperform. De Rossi's data are for a broad universe ( Russell 3000 stocks )
# and cover the period from 2000 through 2009; over this period, high-volatility stocks
# outperform ( especially high-volatility small caps, which rallied during the 2003 and 2009
# recoveries ). The inferences appear sensitive to research design issues; high-volatility
# stocks' relative performance in the 2000s would be clearly worse if we exclude small caps,
# use value-weighted rather than equal-weighted quintile portfolios, or control for other
# factors including size. As noted, trading costs and other limits to arbitrage may make
# this regularity hard to exploit in practice. 

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
days_holding = "2"
min_volume = "5_000_000"
start_year = "2025"
num_buckets = "10"
days_volatility_sma = "30"

# %%
days_holding_P = int(days_holding)
days_volatility_sma_P = int(days_volatility_sma)
num_buckets_P = int(num_buckets)
min_volume_P = int(min_volume)
start_year_P = int(start_year)

print(f"""
days_holding: {days_holding_P}
num_buckets: {num_buckets_P}
min_volume: {min_volume_P}

start_year: {start_year_P}
""")


# %%
import datetime as dt
import polars as pl
import scrapbook as sb
import backtest_ng as bt
import functools as fc
import json


stables = [c.lower() for c in [
    "USDT", "BUSD", "USDC", "PAX", "PAXG", "TUSD", "DAI", "USDP", "UST",
    "FDUSD", "USD1", "XUSD", "USD", "EURC", "EURI", "AEUR", "EUR", "GBP",
    "JPY", "AUD", "CAD", "CHF", "NZD", "KRW", "RLUSD"
]]

df = (
    # select exchanges as source
    pl.read_parquet('live/symbols.parquet')

    # select by lowest borrow rate
    #.filter(pl.col('exchange') != 'binance')
    .rename({
        "cross_rate": "cross",
        "isolated_rate": "isolated",
        # comment out the following line
        "funding_rate":"perp",
    })
    .unpivot(
        on=[
            "cross",
            "isolated",
            # comment out the following line
            "perp",
        ],
        index=["symbol",'exchange','base'],
        variable_name="type", 
        value_name="rate"
    )
    .filter(pl.col('rate').is_not_null())
    .sort('rate', descending=True)
    .group_by('base')
    .last()
 
    # join ohlcv data
    .join(pl.read_parquet('live/klines.parquet'),on=['exchange','symbol'])
    .with_columns(
        symbol=pl.col('exchange') + ":" + pl.col('symbol'),
        ts=pl.col('open_ts'),
        volume=pl.col("quote_volume"),
    )
    .filter(pl.col('base').is_in(stables).not_())

    # compute variance
    .sort(['symbol','ts'])
    .with_columns(
        ts=pl.col('ts').cast(pl.Datetime("us")),
        ho = (pl.col('high') / pl.col('open')).log(),
        hc = (pl.col('high') / pl.col('close')).log(),
        lo = (pl.col('low') / pl.col('open')).log(),
        lc = (pl.col('low') / pl.col('close')).log(),
    )
    .with_columns(
        var=(pl.col('ho') * pl.col('hc')) + (pl.col('lo') * pl.col('lc'))
    )
    .select([
        pl.col('ts'),
        pl.col('symbol'),
        pl.col('close'),
        pl.col('high'),
        pl.col('low'),
        pl.col('volume'),
        pl.col('var').rolling_mean(window_size=days_volatility_sma_P).over('symbol').mul(365).sqrt().alias('vol'),
        pl.col('exchange'),
        pl.col('rate'),
        pl.col('type'),
    ])
        
    .drop_nulls(subset=['vol']) 
    .filter(pl.col('ts').dt.year() >= start_year_P)
    .with_columns(
        bucket=pl.when(
            (pl.col('volume') > min_volume_P)
        )
        .then(pl.col("vol"))
        .qcut(num_buckets_P, labels=[str(i) for i in range(num_buckets_P)], allow_duplicates=True).over('ts'),
    )
)

class Alpha(bt.AlphaModel):
    def __call__(self, history: pl.DataFrame, u: bt.Universe) -> list[bt.Signal]:
        df = u.df()
        today = df["ts"].max()
        s = df.filter((pl.col('ts') == today) & (pl.col("bucket") == str(num_buckets_P - 1)))
        return [
            bt.Signal(r['symbol'], False, -1.0)
            for r in s.iter_rows(named=True)
        ]

test = bt.Backtest(
    bt.Manual(df,high_col='high',low_col='low'),
    title='vol spread',
    alpha=Alpha(),
    risk=bt.MaxRisk(.4),
    benchmark="BTCUSDT",
    period=days_holding_P,
)

test.run()
res = test.report(plot=True)

# %%
print(
    test.live(equity=10_000)
    .join(df.select(
        entry_ts=pl.col('ts'),
        symbol=pl.col('symbol'), 
        exchange=pl.col('exchange'),
        rate=pl.col('rate'),
        margin=pl.col('type'),
    ), on=['entry_ts','symbol'])
    .select(
        pl.col('entry_ts').dt.strftime('%Y-%m-%d'),
        pl.col('exchange'),
        pl.col('margin'),
        pl.col('symbol'),
        pl.col('entry_price'),
        pl.col('rate'),
        (pl.col('entry_price') * pl.col('shares')).abs().alias('cost'),
    )
    .sort('symbol')

    .write_csv()
)

# %%
