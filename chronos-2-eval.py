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
# papermill parameters

# i/o
input_ohlcv_file = "usdt1d.parquet"
input_gjr_file = "garch.parquet"

output_feat_file = "chronos-2-features.parquet"
output_pred_file = "pred.parquet"
output_eval_file = "chronos-2-eval.parquet"

# chronos-2
device_map = "cpu"
cross_learning = True
lookback = 60
lookforward = 1
norm_axis = "rows"  # rows: time series, columns: cross-sectional

reference_pair = "BTCUSDT"

# derived, forecasted metrics
gjr_garch_dist = "skewt"

# %% editable=true slideshow={"slide_type": ""}
print(
    f"""
input_ohlcv_file = {input_ohlcv_file}
input_gjr_file = {input_gjr_file}

output_feat_file = {output_feat_file}
output_pred_file = {output_pred_file}
output_eval_file = {output_eval_file}

device_map = {device_map}
cross_learning = {cross_learning}
lookback = {lookback}
lookforward = {lookforward}
norm_axis = {norm_axis}

reference_pair = {reference_pair}

# derived, forecasted metrics
gjr_garch_dist = {gjr_garch_dist}
"""
)

if lookforward != 1:
    raise ValueError("BUG: code can't handle lookforwards other than 1")

# %% editable=true slideshow={"slide_type": ""}
# volatility, taker buy/sell ratio, momentum, auto correlation, level

import polars as pl
from arch import arch_model
import numpy as np
from tqdm import tqdm

ohlcv = pl.read_parquet(input_ohlcv_file).sort(["symbol", "ts"])
gjr = pl.read_parquet(input_gjr_file).sort(["symbol", "ts"])

df = (
    ohlcv.join(gjr, on=["symbol", "ts"])
    .filter(
        (pl.col("open") > 0)
        & (pl.col("high") > 0)
        & (pl.col("low") > 0)
        & (pl.col("close") > 0)
        & (pl.col("base_volume") > 0)
        & (pl.col("quote_volume") > 0)
    )
    # todays log return (target)
    .with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().over("symbol").alias("ret")
    )
    # log wick symmetry
    .with_columns(
        (
            (pl.col("high") - pl.col("close") + pl.lit(1e-6))
            / (pl.col("close") - pl.col("low") + pl.lit(1e-6))
        )
        .log()
        .over("symbol")
        .alias("sym"),
    )
    # rogers-satchell volatility
    .with_columns(
        (
            (
                (pl.col("high") / pl.col("close")).log()
                * (pl.col("high") / pl.col("open")).log()
            )
            + (
                (pl.col("low") / pl.col("close")).log()
                * (pl.col("low") / pl.col("open")).log()
            )
        )
        .sqrt()
        .alias("sigma_rs"),
    )
    # vwap
    .with_columns(
        (pl.col("quote_volume") / pl.col("base_volume")).alias("vwap"),
    )
    # cumulative volume delta
    .with_columns(
        (pl.col("taker_buy_base_volume") - pl.col("taker_buy_quote_volume")).alias(
            "cvd"
        ),
    )
    # n-day momentum
    .with_columns(
        (pl.col("ret").rolling_sum(window_size=120).over("symbol").alias("m120")),
        (pl.col("ret").rolling_sum(window_size=30).over("symbol").alias("m30")),
    )
    .select(
        # base columns
        pl.col("ts"),
        pl.col("symbol"),
        pl.col("ret"),
        pl.col("quote_volume"),
        # derived features
        pl.col("sym"),
        pl.col("sigma_rs"),
        pl.col("vwap"),
        pl.col("cvd"),
        pl.col("m120"),
        pl.col("m30"),
        # gjr-garch
        pl.col("forecast").alias("sigma_forecast"),
        pl.col("mu"),
        pl.col("omega"),
        pl.col("alpha[1]"),
        pl.col("gamma[1]"),
        pl.col("beta[1]"),
        pl.col("eta"),
        pl.col("lambda"),
    )
    .drop_nulls()
    .write_parquet(output_feat_file)
)

# %% editable=true slideshow={"slide_type": ""} tags=["dev_only"]
# inference with chronos-2

import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline
import datetime as dt
from tqdm import tqdm

pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device_map)
pred = []
df = pl.read_parquet(output_feat_file)

for td in tqdm(df["ts"].unique().sort()):
    lb = td - dt.timedelta(days=lookback)
    yd = td - dt.timedelta(days=1)

    # extract batch
    win = df.filter((pl.col("ts") >= lb) & (pl.col("ts") < td)).filter(
        (pl.len().over("symbol") >= lookback)
        & (pl.col("ts").max().over("symbol") == yd)
    )

    if len(win) < 3:
        continue

    # fill in gaps
    win = (
        win.drop_nulls()
        .upsample(time_column="ts", every="1d", group_by="symbol")
        .with_columns(pl.all().forward_fill())
    )

    future_df = pl.DataFrame()

    # forecast volatility using yesterday's parameters
    for sym in win["symbol"].unique():
        row = df.filter((pl.col("ts") == yd) & (pl.col("symbol") == sym))
        if row.height != 1:
            continue
        if row["sigma_forecast"].is_nan().all():
            var = np.full(lookforward, np.nan)
        else:
            params = np.array(
                [
                    row["mu"][0],
                    row["omega"][0],
                    row["alpha[1]"][0],
                    row["gamma[1]"][0],
                    row["beta[1]"][0],
                    row["eta"][0],
                    row["lambda"][0],
                ]
            )
            ret = win.filter(pl.col("symbol") == sym).select("ret").to_numpy().flatten()
            fc = arch_model(
                ret * 100,
                vol="Garch",
                p=1,
                o=1,
                q=1,
                dist=gjr_garch_dist,
                rescale=False,
            ).forecast(params=params, horizon=lookforward, reindex=False)
            var = np.sqrt(fc.variance.values[-1, :]) / 100

        future_df = pl.concat(
            [
                future_df,
                pl.DataFrame(
                    {
                        "ts": [
                            yd + dt.timedelta(days=d + 1) for d in range(lookforward)
                        ],
                        "sigma_forecast": var,
                        "symbol": sym,
                    }
                ),
            ]
        )

    # remove garch parameters
    win = win.drop(["mu", "omega", "alpha[1]", "gamma[1]", "beta[1]", "eta", "lambda"])

    # normalize batch
    features = [c for c in win.columns if c not in ["ts", "symbol", "ret"]]
    if norm_axis == "rows":
        norm = win.with_columns(
            [
                (
                    (pl.col(c) - pl.col(c).mean().over("symbol"))
                    / (pl.col(c).std().over("symbol") + 1e-9)
                ).alias(c)
                for c in features
            ]
        )
        future_norm = future_df.with_columns(
            [
                (
                    (pl.col(c) - pl.col(c).mean().over("symbol"))
                    / (pl.col(c).std().over("symbol") + 1e-9)
                ).alias(c)
                for c in ["sigma_forecast"]
            ]
        )
    elif norm_axis == "columns":
        norm = win.with_columns(
            [
                (
                    (pl.col(c) - pl.col(c).mean().over("ts"))
                    / (pl.col(c).std().over("ts") + 1e-9)
                ).alias(c)
                for c in features
            ]
        )
        future_norm = win.with_columns(
            [
                (
                    (pl.col(c) - pl.col(c).mean().over("ts"))
                    / (pl.col(c).std().over("ts") + 1e-9)
                ).alias(c)
                for c in ["sigma_forecast"]
            ]
        )
    else:
        raise Exception(f"unknown normalization direction {norm_axis}")

    # Generate predictions with covariates
    pdf = pipeline.predict_df(
        norm.sort(["ts", "symbol"]).to_pandas(),
        future_norm.sort(["ts", "symbol"]).to_pandas(),
        prediction_length=lookforward,
        quantile_levels=[0.1587, 0.5, 0.8413],
        id_column="symbol",
        timestamp_column="ts",
        target="ret",
        validate_inputs=True,
        cross_learning=cross_learning,
    )

    if pdf["0.5"].isna().any():
        raise Exception(f"NaN prediction for batch {td}")

    pdf = pl.from_pandas(pdf).select(
        [
            pl.col("ts"),
            pl.col("symbol"),
            pl.col("0.5").alias("pred"),
            pl.col("0.1587").alias("low_sigma"),
            pl.col("0.8413").alias("high_sigma"),
        ]
    )
    pred.append(pdf)

pl.concat(pred).write_parquet(output_pred_file)
