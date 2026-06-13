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
days_holding = "3"
top_percentile = "0.95"
bottom_percentile = "0.00"
days_momentum_score = "40"
min_volume = "1_000_000"

# %%
days_holding_P = int(days_holding)
top_percentile_P = float(top_percentile)
bottom_percentile_P = (
    float(bottom_percentile) if bottom_percentile != "None" else None
)
days_momentum_score_P = int(days_momentum_score)
min_volume_P = int(min_volume)

print(f"""
days_holding: {days_holding_P}
days_momentum_score: {days_momentum_score_P}
min_volume: {min_volume_P}

top_percentile: {top_percentile_P}
bottom_percentile: {bottom_percentile_P}
""")


# %%
import datetime as dt
import polars as pl
import scrapbook as sb
import backtest_ng as bt
import functools as fc
import operator

stables = [
    "USDT", "BUSD", "USDC", "PAX", "PAXG", "TUSD", "DAI", "USDP", "UST",
    "FDUSD", "USD1", "XUSD", "USD", "EURC", "EURI", "AEUR", "EUR", "GBP",
    "JPY", "AUD", "CAD", "CHF", "NZD", "KRW", "RLUSD", "USD0", "EURS",
    "USDAI", "USDA", "USDM", "FRXUSD", "APYUSD", "A7A5", "NUSD", "FDIT",
    "SATUSD", "YLDS", "GHO", "BFUSD", "CRVUSD", "GUSD", "USX", "AUSD",
    "USDTB", "APXUSD", "RUSD",
]
credit = [
    "REUSD", "FIGR_HELOC", "USDY", "JAAA", "SAFO", "USYC", "OUSG", "EURSAFO",
    "JTRSY", "USTBL", "USTB", "THBILL", "BUIDL", "ONYC",
]
syms = {
    f'{c.upper()}USDT': c for c
    in pl.read_parquet("polarity/latest-data.parquet")['asset'].unique().to_list()
    if not (c in stables or c in credit)
}
cols = ['kucoin','kraken','htx','binance']

cmcdf = (
    pl.read_parquet('data/dataset/all.parquet')
    .with_columns(
        exchange=pl.coalesce([pl.when(pl.col(c).is_in(syms.keys())).then(pl.lit(c)).otherwise(None) for c in cols]),
        pair=pl.coalesce([pl.when(pl.col(c).is_in(syms.keys())).then(pl.col(c)).otherwise(None) for c in cols]),
    )
    .drop_nulls(subset=['exchange'])
    .with_columns(
        symbol=pl.col('pair').replace_strict(syms, default=None),
    )
    .join(
       pl.read_parquet("polarity/latest-data.parquet")
        .select(
            ts=pl.col('timestamp').dt.replace_time_zone('UTC'),
            symbol=pl.col('asset'),
            price=pl.col('price'),
        ),
        on=['ts','symbol']
    )
    #.select(
    #    ts=pl.col('ts'),
    #    symbol=pl.col('kucoin'),
    #    close=pl.col('close'),
    #    volume=pl.col('volume'),
    #)
    #.with_columns(
    #    age=pl.col("ts") - pl.col("ts").min().over("symbol"),
    #)
    #.drop_nulls(subset=['symbol'])
)

geckodf = (
    pl.read_parquet('metrics.parquet')
    #.filter((pl.col('coingecko_slug').is_in(pl.read_parquet('polarity/lala.parquet')['coingecko_slug'].unique().to_list())))
    .with_columns(symbol=pl.col('coingecko_slug'))
)

# change this to cmcdf for the issue to occur
df = (geckodf
    .with_columns(ts=pl.col("ts").dt.cast_time_unit("us"))
    .sort(["symbol", "ts"])
    .drop_nulls(subset=['volume'])
    .with_columns(**{
        f'mom{n}': pl.col("close").pct_change(n).over("symbol")
        for n in set([days_momentum_score_P])
    })
    .with_columns(
        volume=pl.col('volume').rolling_mean(days_holding_P).over('symbol'),
    )
    .with_columns(
        rank=(
            pl.col(f"mom{days_momentum_score_P}").rank(method="ordinal").over("ts")
            / pl.col(f"mom{days_momentum_score_P}").count().over("ts")
        ),
    )
    .filter((pl.col("ts") > dt.datetime(2023, 1, 1,tzinfo=dt.timezone.utc)))# & (pl.col("ts") <= dt.datetime(2026, 5, 16)))
)

df = (
    pl.read_parquet("polarity/latest-data.parquet")
    .rename(
        {
            "asset": "symbol",
            "price": "close",
            "timestamp":"ts",
        }
    )
    .with_columns(ts=pl.col("ts").dt.cast_time_unit("us").dt.replace_time_zone('UTC'))
    .sort(["symbol", "ts"])
    .with_columns(**{
        f'mom{n}': pl.col("close").pct_change(n).over("symbol")
        for n in set([days_momentum_score_P])
    })
    .with_columns(
        rank=(
            pl.col(f"mom{days_momentum_score_P}").rank(method="ordinal").over("ts")
            / pl.col(f"mom{days_momentum_score_P}").count().over("ts")
        ),
        volume=pl.col('total_volume').rolling_mean(days_holding_P).over('symbol'),
    )
    .sort("ts")
    .filter((pl.col("ts") > dt.datetime(2023, 1, 1,tzinfo=dt.timezone.utc)))# & (pl.col("ts") <= dt.datetime(2026, 5, 16)))
    .drop_nulls(['volume'])
)

class Alpha(bt.AlphaModel):
    def __init__(self, long_expr: pl.Expr, short_expr: pl.Expr):
        self.long_expr = long_expr
        self.short_expr = short_expr

    def __call__(self, history: pl.DataFrame, u: bt.Universe) -> list[bt.Signal]:
        today = u.df()["ts"].max()
        dfnow = u.df().filter(pl.col("ts") == today)
        l = dfnow.filter(self.long_expr)
        s = dfnow.filter(self.short_expr)

        return [
            bt.Signal(symbol=r['symbol'], bullish=True, confidence=r['rank'])
            for r in l.iter_rows(named=True)
        ] + [
            bt.Signal(symbol=r['symbol'], bullish=False, confidence=1-r['rank'])
            for r in s.iter_rows(named=True)
        ]

test = bt.Backtest(
    title="polarity momentum (95%)",
    universe=bt.Manual(df),
    alpha=Alpha(
        long_expr=(pl.col("rank") >= top_percentile_P) & (pl.col('volume') > min_volume_P),
        short_expr=(
            (pl.col("rank") <= bottom_percentile_P) & (pl.col('volume') > min_volume_P)
            if bottom_percentile_P is not None
            else pl.lit(False)
        ),
    ),
    benchmark="btc",
    period=days_holding_P,
)

test.run()
res = test.report(plot=True)

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
print(test.positions.filter(pl.col('ts') == dt.datetime(2026,5,13)).write_csv())

# %%
