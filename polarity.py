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

# %%
horizons = {"1d": 1, "3d": 3, "1w": 7, "2w": 14, "1m": 30}
deriv_win = 7
zscore_win = 30


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
    .sort(["symbol", "ts"])
    .with_columns(
        **{
            f"target{k}": (
                pl.col("close").shift(-v).log() - pl.col("close").log()
            ).over("symbol")
            for k, v in horizons.items()
        }
    )
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
            *[f"target{h}" for h in horizons],
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
    )
)

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
symbols = df["symbol"].unique().sort().to_list()
cutoff = 2024

res_frag = []

for sym in tqdm(symbols):
    train_df = (
        df.filter((pl.col("symbol") == sym) & (pl.col("ts").dt.year() <= cutoff))
        .sort("ts")
        .drop_nulls(subset=features)
        .drop_nans(subset=features)
    )
    test_df = (
        df.filter(pl.col("symbol") == sym)
        .sort("ts")
        .drop_nulls(subset=features)
        .drop_nans(subset=features)
    )

    if train_df.height < 10:
        print(f"skip {sym}")
        continue

    res_df = pl.DataFrame(
        {
            "ts": test_df["ts"],
            "symbol": test_df["symbol"],
        }
    )

    for t, v in horizons.items():
        train_h = train_df.drop_nulls(subset=[f"target{t}"]).drop_nans(
            subset=[f"target{t}"]
        )
        X_train, y_train = train_h[features], train_h[f"target{t}"]
        X_test = test_df[features]

        model_lin = LinearRegression().fit(X_train.to_pandas(), y_train.to_pandas())
        model_gbt = HistGradientBoostingRegressor(
            max_iter=50, max_depth=3, random_state=42
        ).fit(X_train, y_train)
        model_com = HistGradientBoostingRegressor(
            max_iter=50, max_depth=3, random_state=42
        ).fit(X_train, y_train - model_lin.predict(X_train))

        res_df = res_df.with_columns(
            **{
                f"lin{t}": model_lin.predict(X_test),
                f"gbt{t}": model_gbt.predict(X_test),
                f"com{t}": model_lin.predict(X_test) + model_com.predict(X_test),
            }
        )

    res_frag.append(res_df)

df = df.join(pl.concat(res_frag), on=["ts", "symbol"], how="inner")

# %%
from math import log

start_year = 2023

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

for model, hold_win in models.items():
    res_frag = []
    trades_frag = []
    leverage = 1

    portfolio = []
    last_rebalance = None
    days = (
        df.filter(pl.col("ts").dt.year() >= start_year)["ts"].unique().sort().to_list()
    )

    for today in tqdm(days):
        # fetch current prices
        latest_close = {}
        for pos in portfolio:
            close = df.filter(
                (pl.col("symbol") == pos["symbol"]) & (pl.col("ts") <= today)
            ).sort("ts")
            if close["ts"][-1] < today:
                print(f"{pos['symbol']} was delisted")
            latest_close[pos["symbol"]] = close["close"][-1]

        # compute portfolio values
        ret = 0
        for pos in portfolio:
            close = latest_close[pos["symbol"]]
            initial = pos["price"] * abs(pos["weight"])
            if pos["weight"] > 0:
                initial *= close / pos["price"] * abs(pos["weight"])
            else:
                initial *= pos["price"] / close * abs(pos["weight"])
            ret += initial
        ret = 0 if ret == 0 else 1 / ret

        res_frag.append(
            pl.DataFrame(
                {
                    "ts": today,
                    "logret": np.nan if ret == 0 else log(ret),
                },
                schema={
                    "ts": pl.Datetime,
                    "logret": pl.Float32,
                },
            )
        )

        if not (
            last_rebalance is None
            or today - last_rebalance >= dt.timedelta(days=hold_win)
            or today == days[-1]
        ):
            continue

        # rebalance
        latest_rebalance = today
        closed_syms = []
        closed_logret = []
        closed_open_ts = []
        closed_dir = []

        # close open positions
        for pos in portfolio:
            # print(pos)
            close = latest_close[pos["symbol"]]
            # print('close',close)
            if pos["weight"] > 0:
                ret = close / pos["price"]
            else:
                ret = pos["price"] / close
            # print('ret',ret)

            closed_logret.append(log(ret))
            closed_syms.append(pos["symbol"])
            closed_open_ts.append(pos["open"])
            closed_dir.append(1 if pos["weight"] > 0 else -1)
            # print('logret',closed_logret[-1])
            # print('dir',closed_dir[-1])

        trades_frag.append(
            pl.DataFrame(
                {
                    "symbol": closed_syms,
                    "open": closed_open_ts,
                    "close": today,
                    "logret": closed_logret,
                    "direction": closed_dir,
                },
                schema={
                    "symbol": pl.Utf8,
                    "open": pl.Datetime,
                    "close": pl.Datetime,
                    "logret": pl.Float32,
                    "direction": pl.Int8,
                },
            )
        )
        portfolio = []

        cs_df = df.filter(pl.col("ts") == pl.lit(today)).with_columns(
            rank=pl.col(model)
            .qcut(10, allow_duplicates=True, labels=[str(i) for i in range(10)])
            .over("ts"),
        )

        long = cs_df.filter(
            ((pl.col("rank") == "9") | (pl.col("rank") == "9")) & (pl.col(model) > 0)
        )
        short = cs_df.filter(
            ((pl.col("rank") == "0") | (pl.col("rank") == "0")) & (pl.col(model) < 0)
        )
        syms = long["symbol"].to_list() + short["symbol"].to_list()

        if long.height + short.height == 0:
            continue

        w = leverage / (long.height + short.height)
        new_positions = (
            df.filter((pl.col("ts") == today) & (pl.col("symbol").is_in(syms)))
            .sort(["symbol", "ts"])
            .with_columns(
                direction=pl.when(pl.col("symbol").is_in(long["symbol"].to_list()))
                .then(1)
                .otherwise(-1),
            )
        )

        for row in new_positions.iter_rows(named=True):
            portfolio.append(
                {
                    "symbol": row["symbol"],
                    "open": today,
                    "price": row["close"],
                    "weight": w * row["direction"],
                }
            )

    trades_df = pl.concat(trades_frag)
    res_df = pl.concat(res_frag)
    print(res_df)
    print(trades_df)
    (
        res_df.join(
            df.filter(pl.col("symbol") == "btc").select(
                ts=pl.col("ts").dt.cast_time_unit("us"),
                btc=pl.col("close").log().diff(),
            ),
            on="ts",
            how="inner",
        )
        .drop_nans()
        .with_columns(
            btc=pl.col("btc").cum_sum(),
            value=pl.col("logret").cum_sum(),
        )
        .filter(pl.col("ts").dt.year() >= 2023)
        .to_pandas()
    ).plot(y=["value", "btc"], x="ts", title=model)
    break

# %%
