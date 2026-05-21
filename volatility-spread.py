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
min_volume = "1_000_000"
start_year = "2025"
num_buckets = "10"
days_volatility_sma = "14"

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

stables = [f'{c}USDT' for c in [
    "USDT", "BUSD", "USDC", "PAX", "PAXG", "TUSD", "DAI", "USDP", "UST",
    "FDUSD", "USD1", "XUSD", "USD", "EURC", "EURI", "AEUR", "EUR", "GBP",
    "JPY", "AUD", "CAD", "CHF", "NZD", "KRW", "RLUSD"
]]

with open('kucoin-margin.json') as fd:
    kucoin_margin = [s.replace("-","") for s in json.load(fd)]

with open('htx-margin.json') as fd:
    htx_margin = [s.replace("-","") for s in json.load(fd)]

df = (
    pl.read_parquet('stables-1d.parquet')
    .rename({"quote_volume":"volume",})
    .filter(pl.col('symbol').is_in(stables).not_())

    # limit to KuCoin and HTX marginable coins. remove to see the pure performance
    .filter((pl.col('symbol').is_in(kucoin_margin)) | (pl.col('symbol').str.to_lowercase().is_in(htx_margin)))
    
    .sort(['symbol','ts'])
    .join(pl.read_parquet('stables-1d.parquet').filter(pl.col('symbol') == 'BTCUSDT').select(ts=pl.col('ts'), btc=pl.col('close')), on=['ts'])
    .with_columns(
        ts=pl.col('ts').cast(pl.Datetime("us")),
        sma=pl.col('btc').rolling_mean(60).over('symbol'),
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
        pl.col('volume'),
        pl.col('var').rolling_mean(window_size=days_volatility_sma_P).over('symbol').mul(365).sqrt().alias('vol'),
        pl.col('btc'),
        pl.col('sma'),
    ])
        
    .drop_nulls(subset=['vol']) 
    .filter(pl.col('ts').dt.year() >= start_year_P)
    .with_columns(
        bucket=pl.col('vol').qcut(num_buckets_P, labels=[str(i) for i in range(num_buckets_P)], allow_duplicates=True).over('ts'),
    )
)

class Alpha(bt.AlphaModel):
    def __init__(self, long_expr: pl.Expr, short_expr: pl.Expr):
        self.long_expr = long_expr
        self.short_expr = short_expr

    def __call__(self, u: bt.Universe) -> list[bt.Signal]:
        df = u.df()
        today = df["ts"].max()
        dfnow = df.filter(pl.col("ts") == today)
        l = dfnow.filter(self.long_expr)
        s = dfnow.filter(self.short_expr)

        return [
            bt.Signal(r['symbol'], True, 1.0)
            for r in l.iter_rows(named=True)
        ] + [
            bt.Signal(r['symbol'], False, -1.0)
            for r in s.iter_rows(named=True)
        ]

test = bt.Backtest(
    bt.Manual(df),
    title='vol spread',
    alpha=Alpha(
        long_expr=pl.lit(False),#(pl.col("bucket") == '0') & (pl.col('volume') > min_volume_P),# & (pl.col('btc') > pl.col('sma')),
        short_expr=(pl.col("bucket") == str(num_buckets_P - 1)) & (pl.col('volume') > min_volume_P),# & (pl.col('btc') / pl.col('sma') <= .95),
    ),
    benchmark="BTCUSDT",
    period=days_holding_P,
)

test.run()
res = test.report(plot=True)

#for col in sorted([c for c in res.columns if c not in ['year', 'src']]):
#    s = res.filter(pl.col('src') == 'Strategy')
#    b = res.filter(pl.col('src') == 'Benchmark')
#
#    print(f"{col}: {s[col].mean()} ({b[col].mean()})")
#    sb.glue(col,s[col].mean())

# %%
print(
    test.live(equity=10_000)
    .select(
        pl.col('entry_ts').dt.strftime('%Y-%m-%d'),
        pl.col('symbol').str.to_uppercase(),
        pl.lit(''),
        pl.col('entry_price'),
        pl.col('shares'),
        (pl.col('entry_price') * pl.col('shares')).alias('cost'),
    )
    .sort('symbol')
    .write_csv()
)

# %%
