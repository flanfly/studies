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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
import datetime as dt
from scipy.stats import spearmanr
from tqdm import tqdm
import itertools as it

# %%
deriv_win = 7
zscore_win = 30

features = [
    "udpil",
    "udpim",
    "udpis",
    "upprob",
    "mbi",
    "tci",
    "tcidelta",
    "mdcdelta",
    "tcicvroc",
    "mdccvroc",
    "udpisroc",
    "udpimroc",
    "udpilroc",
    "upprobroc",
    "mbiroc",
    "tciroc",
    "tcideltaroc",
    "mdcdeltaroc",
]

def rolling_zscore(expr, window):
    return ((expr - expr.rolling_mean(window)) / expr.rolling_std(window)).over(
        "symbol"
    )


df = (
    pl.read_parquet("polarity/data/*.parquet")
    .rename(
        {
            "asset": "symbol",
            "price": "close",
            "timestamp": "ts",
        }
    )
        .with_columns(ts=pl.col('ts').dt.cast_time_unit("us"))
    .sort(["symbol", "ts"])
    # derive pct distance from *cv
    .with_columns(
        tcidelta=rolling_zscore(
            (pl.col("tcicv") - pl.col("close")) / pl.col("close"), zscore_win
        ),
        mdcdelta=rolling_zscore(
            (pl.col("mdccv") - pl.col("close")) / pl.col("close"), zscore_win
        ),
    )
    # derive 1st deriv
    .with_columns(
        **{
            f"{col}roc": rolling_zscore(
                pl.col(col) - pl.col(col).shift(deriv_win).over("symbol"), zscore_win
            )
            for col in [
                "tcicv",
                "mdccv",
                "udpis",
                "udpim",
                "udpil",
                "upprob",
                "mbi",
                "tci",
                "tcidelta",
                "mdcdelta",
            ]
        }
    )
    .select(
        [
            "ts",
            "symbol",
            "close",
            *features,
        ]
    )
)

start_year = 2020
horizons = {"1d": 1, "3d": 3, "1w": 7, "2w": 14, "1m": 30}

symbols = df["symbol"].unique().sort().to_list()
days = df.filter(pl.col('ts').dt.year() >= start_year)['ts'].unique().sort().to_list()

pred_frag = []

for tup in tqdm(it.product(symbols, days, horizons.items()), total=len(symbols)*len(days)*len(horizons)):
    sym, day, hori = tup
    t, v = hori
    
    # derive the target column after we cut off the training data
    train_df = (df
        .filter((pl.col("symbol") == sym) & (pl.col("ts") <= day) & (pl.col('ts').dt.year() >= start_year))
        .with_columns(
            target=(pl.col("close").shift(-v).log() - pl.col("close").log()).over("symbol")
        )
        .sort("ts")
        .drop_nulls(subset=features)
        .drop_nans(subset=features)
    )
    test_df = train_df.filter(pl.col('ts') == day)
    train_h = train_df.filter(pl.col('target').is_null().not_())
    
    if train_h.height < 10 or test_df.height == 0:
        continue

    X_train, y_train = train_h[features], train_h["target"]
    X_test = test_df[features]

    model_lin = LinearRegression().fit(X_train.to_pandas(), y_train.to_pandas())
    model_gbt = HistGradientBoostingRegressor(
        max_iter=50, max_depth=3, random_state=42
    ).fit(X_train, y_train)
    model_com = HistGradientBoostingRegressor(
        max_iter=50, max_depth=3, random_state=42
    ).fit(X_train, y_train - model_lin.predict(X_train))

    pred_frag.append(pl.DataFrame({
        "ts": test_df["ts"].dt.cast_time_unit("us"),
        "symbol": test_df["symbol"],
        f"lin{t}": model_lin.predict(X_test),
        f"gbt{t}": model_gbt.predict(X_test),
        f"com{t}": model_lin.predict(X_test) + model_com.predict(X_test),
    }))

pred_df = (
    pl.concat(pred_frag,how="diagonal")
        .sort(['ts','symbol'])
        .group_by(['ts','symbol'])
        .agg(pl.all().first(ignore_nulls=True))
)
pred_df.write_parquet('predictions.parquet')
df = df.join(pred_df, on=["ts", "symbol"], how="inner")
df

# %%
pred_df

# %%
(
    df
        .sort(['symbol','ts'])
        .group_by('symbol')
        .agg(
            lin1d=pl.corr(pl.col('close').log().diff().shift(1).over('symbol'),pl.col('lin1d')),
            gbt1d=pl.corr(pl.col('close').log().diff().shift(1).over('symbol'),pl.col('gbt1d')),
            com1d=pl.corr(pl.col('close').log().diff().shift(1).over('symbol'),pl.col('com1d')),
            
            lin3d=pl.corr(pl.col('close').log().diff(n=3).shift(1).over('symbol'),pl.col('lin3d')),
            gbt3d=pl.corr(pl.col('close').log().diff(n=3).shift(1).over('symbol'),pl.col('gbt3d')),
            com3d=pl.corr(pl.col('close').log().diff(n=3).shift(1).over('symbol'),pl.col('com3d')),
            
            lin1w=pl.corr(pl.col('close').log().diff(n=7).shift(1).over('symbol'),pl.col('lin1w')),
            gbt1w=pl.corr(pl.col('close').log().diff(n=7).shift(1).over('symbol'),pl.col('gbt1w')),
            com1w=pl.corr(pl.col('close').log().diff(n=7).shift(1).over('symbol'),pl.col('com1w')),
            
            #lin2w=pl.corr(pl.col('close').log().diff(n=14).shift(1).over('symbol'),pl.col('lin2w')),
            #gbt2w=pl.corr(pl.col('close').log().diff(n=14).shift(1).over('symbol'),pl.col('gbt2w')),
            #com2w=pl.corr(pl.col('close').log().diff(n=14).shift(1).over('symbol'),pl.col('com2w')),

            #lin1m=pl.corr(pl.col('close').log().diff(n=30).shift(1).over('symbol'),pl.col('lin1m')),
            #gbt1m=pl.corr(pl.col('close').log().diff(n=30).shift(1).over('symbol'),pl.col('gbt1m')),
            #com1m=pl.corr(pl.col('close').log().diff(n=30).shift(1).over('symbol'),pl.col('com1m')),
        )
        .mean()
        .unpivot()
        .filter(pl.col('variable') != 'symbol')
        .sort('value')
)

# %%
from math import log

models = {
    "lin1d": 1,
    "gbt1d": 1,
    "com1d": 1,
    "lin1w": 7,
    "gbt1w": 7,
    "com1w": 7,
    "lin2w": 14,
    "gbt2w": 14,
    "com2w": 14,
    "lin1m": 30,
    "gbt1m": 30,
    "com1m": 30,
}

prices_df = df.pivot(index="ts", on="symbol", values="close").sort("ts")
prices_df = prices_df.fill_null(strategy="forward")
prices = prices_df.to_pandas().set_index("ts")

res_frag = []
trades_frag = []
leverage = 1

for model, hold_win in models.items():
    portfolio = []
    last_rebalance_date = None
    
    capital = 1.0
    prev_port_value = 1.0
    
    days = (
        df.filter(pl.col("ts").dt.year() >= start_year)["ts"].unique().sort().to_list()
    )

    for today in tqdm(days):
        # 1. Update portfolio value
        port_value = 0
        if not portfolio:
            port_value = capital
        else:
            for pos in portfolio:
                close = prices.at[today, pos['symbol']]
                if pd.isna(close):
                    close = pos['open_price']
                
                if pos['direction'] == 1:
                    pos_value = pos['capital'] * (close / pos['open_price'])
                else:
                    pos_value = pos['capital'] * (2 - close / pos['open_price'])
                port_value += pos_value
                
        if port_value <= 0:
            daily_logret = np.nan
        else:
            try:
                daily_logret = log(port_value / prev_port_value)
            except:
                daily_logret = np.nan
            
        prev_port_value = port_value

        res_frag.append(
            pl.DataFrame(
                {
                    "ts": today,
                    "logret": daily_logret,
                    "model": model,
                },
                schema={"ts": pl.Datetime, "logret": pl.Float32, "model": pl.Utf8},
            )
        )

        # 2. Rebalance logic
        if last_rebalance_date is None or (today - last_rebalance_date).days >= hold_win or today == days[-1]:
            # Close existing portfolio if any
            if portfolio:
                for pos in portfolio:
                    close = prices.at[today, pos['symbol']]
                    if pd.isna(close): close = pos['open_price']
                    if pos['direction'] == 1:
                        ret = close / pos['open_price']
                    else:
                        ret = 2 - close / pos['open_price']
                    trades_frag.append(
                        pl.DataFrame(
                            {
                                "symbol": [pos['symbol']],
                                "open": [pos['open_date']],
                                "close": [today],
                                "logret": [log(max(ret, 1e-6))],
                                "direction": [pos['direction']],
                                "model": model,
                            },
                            schema={
                                "symbol": pl.Utf8,
                                "open": pl.Datetime,
                                "close": pl.Datetime,
                                "logret": pl.Float32,
                                "direction": pl.Int8,
                                "model": pl.Utf8,
                            },
                        )
                    )
            
            # Realize capital
            capital = port_value
            portfolio = []
            
            # Open new positions immediately at today's close
            cs_df = df.filter((pl.col("ts") == today) & pl.col(model).is_not_null())
            if cs_df.height > 0:
                cs_df = cs_df.with_columns(
                    rank=pl.col(model)
                    .qcut(10, allow_duplicates=True, labels=[str(i) for i in range(10)])
                )
                long = cs_df.filter(
                    (pl.col("rank").is_in(["8", "9"])) & (pl.col(model) > 0)
                )
                short = cs_df.filter(
                    (pl.col("rank").is_in(["0", "1"])) & (pl.col(model) < 0)
                )
                
                num_positions = long.height + short.height
                if num_positions > 0:
                    w = leverage / num_positions
                    for row in long.iter_rows(named=True):
                        portfolio.append({
                            "symbol": row['symbol'],
                            "open_date": today,
                            "open_price": row['close'],
                            "direction": 1,
                            "capital": capital * w,
                            "model": model,
                        })
                    for row in short.iter_rows(named=True):
                        portfolio.append({
                            "symbol": row['symbol'],
                            "open_date": today,
                            "open_price": row['close'],
                            "direction": -1,
                            "capital": capital * w,
                            "model": model,
                        })
            
            last_rebalance_date = today

if len(trades_frag) > 0:
    trades_df = pl.concat(trades_frag)
else:
    trades_df = pl.DataFrame(schema={
        "symbol": pl.Utf8,
        "open": pl.Datetime,
        "close": pl.Datetime,
        "logret": pl.Float32,
        "direction": pl.Int8,
        "model": pl.Utf8,
    })
res_df = pl.concat(res_frag)

# %%
res_df

# %%
for model in models:
    (
        res_df
        .filter(pl.col('model') == model)    
        .join(
            df.filter(pl.col("symbol") == "btc").select(
                ts=pl.col("ts").dt.cast_time_unit("us"),
                btc=pl.col("close").log().diff(),
            ),
            on="ts",
            how="inner",
        )
        .drop_nans()
        .with_columns(
            btc=pl.col('btc').cum_sum(),
            value=pl.col('logret').cum_sum(),
        )
        .filter(pl.col("ts").dt.year() >= start_year)
        .to_pandas()
    ).plot(y=["value", "btc"], x="ts", title=model)

# %%
