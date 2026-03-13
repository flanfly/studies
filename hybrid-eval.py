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

# %% tags=["parameters"]
input_ohlcv_file = "stables1d.parquet"
output_features_file = "hybrid-features.parquet"
output_forecast_file = "hybrid-forecast.parquet"
market_symbol = "BTCUSDT"

device = "cpu"
ts2vec_dim = 64
ts2vec_epochs = 10
ts2vec_lr = 0.003

cb_iterations = 10000
cb_lr = 0.2
cb_depth = 6

forecast_lookback = 120

training_days = 365 * 5
validation_days = 365 * 1
test_days = 30

# %%
# prepare data
import polars as pl

df = (
    pl.read_parquet(input_ohlcv_file)
    .sort(["ts", "symbol"])
    .filter(
        (pl.col("open") > 0)
        & (pl.col("high") > 0)
        & (pl.col("low") > 0)
        & (pl.col("close") > 0)
        & (pl.col("base_volume") > 0)
        & (pl.col("quote_volume") > 0)
    )
    .with_columns(
        [
            # todays log return (target)
            (pl.col("close") / pl.col("open"))
            .log()
            .over("symbol")
            .alias("ret")
        ]
    )
    .select(
        [
            # base columns
            pl.col("ts"),
            pl.col("symbol"),
            pl.col("ret"),
        ]
    )
    .drop_nulls()
)

df.write_parquet(output_features_file)
df.head()

# %%
# forecast with catboost using ts2vec features

from ts2vec.ts2vec import TS2Vec
from catboost import CatBoostRegressor
import datetime as dt
import numpy as np
import polars as pl
from typing import List, Tuple

# we optimize the information ratio
df = (
    pl.read_parquet(output_features_file)
    .sort(["symbol", "ts"])
    .join(
        pl.read_parquet(output_features_file)
        .filter(pl.col("symbol") == market_symbol)
        .select(["ts", "ret"])
        .rename({"ret": "ref"}),
        on=["ts"],
        how="inner",
    )
    # .with_columns(target=(pl.col("ret") - pl.col("ref")).shift(-1).over("symbol"))
    .with_columns(target=(pl.col("ret")).shift(-1).over("symbol"))
    .drop("ref")
    .drop_nulls()
    # .filter(
    #    pl.col("symbol").is_in(
    #        [
    #            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT'
    #        ]
    #    )
    # )
    # .filter(pl.col("ts").dt.year() >= 2020)
)

featcols = sorted([col for col in df.columns if col not in ["ts", "symbol", "target"]])
print(f"feature columns: {featcols} ({len(featcols)})")

symbols = df["symbol"].unique().sort().to_list()
print(f"{len(symbols)} unique symbols")

ts = df["ts"].unique().sort()
print(f"{len(ts)} unique timestamps, from {ts[0]} to {ts[-1]}")

# time ranges: 5 years of past data, of that 9 months for validation, used for one month ahead.
input_ts = ts[-(training_days + validation_days + test_days) : -test_days]
train_ts, val_ts = input_ts[:training_days], input_ts[training_days:]
test_ts = ts[-test_days:]

print(f"train    {train_ts[0]}-{train_ts[-1]} ({len(train_ts)})")
print(f"validate {val_ts[0]}-{val_ts[-1]} ({len(val_ts)})")
print(f"test_ts {test_ts[0]}-{test_ts[-1]} ({len(test_ts)})")

# fit TS2Vec to training data, restart if it doesn't converge


def padded_normalized_feature_cube(
    in_ts: pl.Series,
    df: pl.DataFrame,
    stats: pl.DataFrame,
    symbols: List[str],
    featcols: List[str],
    tgtcol: str = "target",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    (symbols, timestamps (padded), features) feature z-score cube
    (symbols, timestamp) mask matrix for missing data
    """

    assert in_ts.is_sorted()

    xcube = np.zeros((len(symbols), len(in_ts), len(featcols)))
    ymat = np.zeros((len(symbols), len(in_ts)))
    mask = np.zeros((len(symbols), len(in_ts)), dtype=bool)
    ts = np.empty((len(symbols), len(in_ts)), dtype=dt.datetime)
    syms = np.empty((len(symbols), len(in_ts)), dtype=object)

    tsdf = pl.DataFrame({"ts": in_ts})

    for i, sym in enumerate(symbols):
        aligned = (
            tsdf.join(
                df.filter(pl.col("symbol") == sym).join(stats, on="symbol", how="inner")
                # z-score normalization per symbol using training set stats (train_stats)
                .with_columns(
                    [
                        (
                            (pl.col(c) - pl.col(f"{c}_mean"))
                            / (pl.col(f"{c}_std") + 1e-9)
                        ).alias(c)
                        for c in featcols
                    ]
                ),
                on="ts",
                how="left",
            )
            # pad missing timestamps with non-zero, avoids nan and inf when fitting
            # track filled values in a mask
            .with_columns(
                [
                    pl.col("symbol").is_null().alias("mask"),
                    *[pl.col(c).fill_null(1e-5) for c in featcols],
                ]
            )
            # clip target to +/- 50%
            .with_columns(
                pl.col(tgtcol).clip(lower_bound=np.log(0.5), upper_bound=np.log(1.5))
            )
            .select(["ts", "mask", "symbol", tgtcol, *featcols])
            .sort("ts")
        )

        assert len(aligned["mask"]) == len(in_ts)

        xcube[i, :, :] = aligned[featcols].to_numpy()
        ymat[i, :] = aligned[tgtcol].to_numpy()
        mask[i, :] = aligned["mask"].to_numpy()
        ts[i, :] = aligned["ts"].to_numpy()
        syms[i, :] = np.full(len(aligned["ts"]), sym, dtype=object)

    assert not np.isnan(xcube).any()
    assert not np.isinf(xcube).any()
    assert xcube.shape == (len(symbols), len(in_ts), len(featcols))

    assert ymat.shape == (len(symbols), len(in_ts))
    assert mask.shape == (len(symbols), len(in_ts))
    assert ts.shape == (len(symbols), len(in_ts))
    assert syms.shape == (len(symbols), len(in_ts))

    return (xcube, ymat, ts, syms, mask)


# summary statistics for normalization
train_stats = (
    df.filter((pl.col("ts") >= train_ts.min()) & (pl.col("ts") <= train_ts.max()))
    .group_by("symbol")
    .agg(
        *[pl.col(c).mean().alias(f"{c}_mean") for c in featcols],
        *[pl.col(c).std().alias(f"{c}_std") for c in featcols],
    )
)

# (symbols, timestamps (padded), features) cube,
# (symbols, timestamps) targets
# (symbols, timestamps) timestamps
# (symbols, timestamps) symbols
# (symbols, timestamps) mask
train_pnfc, train_tm, train_tsm, train_sm, train_fm = padded_normalized_feature_cube(
    train_ts, df, train_stats, symbols, featcols
)

loss_log = [np.nan]
for i in range(5):  # max 5 attempts
    ts2vec_model = TS2Vec(
        input_dims=len(featcols), device=device, output_dims=ts2vec_dim, lr=0.003
    )
    loss_log = ts2vec_model.fit(train_pnfc, n_epochs=ts2vec_epochs, verbose=True)
    if not np.isnan(loss_log).any() and not np.isinf(loss_log).any():
        break


# %%
def extract_gbt_matrix(
    emb: np.ndarray, tgt: np.ndarray, ts: np.ndarray, syms: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, pl.DataFrame]:
    mat_val = emb.reshape(-1, ts2vec_dim)
    vec_mask = mask.flatten()
    vec_tgt = tgt.flatten()
    vec_syms = syms.flatten()
    vec_ts = ts.flatten()

    assert mat_val.shape[0] == vec_mask.shape[0]
    assert mat_val.shape[0] == vec_tgt.shape[0]
    assert mat_val.shape[0] == vec_syms.shape[0]
    assert mat_val.shape[0] == vec_ts.shape[0]

    mask = (~vec_mask) & (~np.isnan(vec_tgt))

    df = pl.DataFrame(
        {
            "target": vec_tgt[mask],
            "ts": pl.Series(vec_ts[mask].astype("datetime64[us]")).dt.replace_time_zone(
                "UTC"
            ),
            "symbol": vec_syms[mask],
        }
    )

    return mat_val[mask], df


X_train, df_train = extract_gbt_matrix(
    ts2vec_model.encode(train_pnfc, causal=True),
    train_tm,
    train_tsm,
    train_sm,
    train_fm,
)
y_train = df_train["target"]

# validation set
val_pnfc, val_tm, val_tsm, val_sm, val_fm = padded_normalized_feature_cube(
    val_ts, df, train_stats, symbols, featcols
)
X_val, df_val = extract_gbt_matrix(
    ts2vec_model.encode(val_pnfc, causal=True),
    val_tm,
    val_tsm,
    val_sm,
    val_fm,
)
y_val = df_val["target"]

# fit gbt regressor
gbt_model = CatBoostRegressor(
    iterations=cb_iterations,
    learning_rate=cb_lr,
    depth=cb_depth,
    loss_function="RMSE",
    verbose=100,
    early_stopping_rounds=50,
    use_best_model=True,
    task_type="CPU" if device == "cpu" else "GPU",
)
gbt_model.fit(X_train, y_train, eval_set=(X_val, y_val))

# %%
warm_test_ts = pl.concat([val_ts, test_ts])
res_ary = []

for td in test_ts:
    yd = td - dt.timedelta(days=1)
    my_ts = warm_test_ts.filter(warm_test_ts < td).sort()[-forecast_lookback:]
    my_pnfc, my_tm, my_tsm, my_sm, my_fm = padded_normalized_feature_cube(
        my_ts, df, train_stats, symbols, featcols
    )
    X_my, df_my = extract_gbt_matrix(
        ts2vec_model.encode(my_pnfc, causal=True),
        my_tm,
        my_tsm,
        my_sm,
        my_fm,
    )

    # (symbols, timestamps)
    df_my = (
        df_my.with_columns(pred=gbt_model.predict(X_my))
        .filter(pl.col("ts") == yd)
        .with_columns(ts=pl.lit(td).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"))
        .sort(["symbol"])
    )
    res_ary.append(df_my)

pl.concat(res_ary).select(["ts", "symbol", "pred"]).write_parquet(output_forecast_file)
# %%
# portfolio evaluation with transaction costs

fee_rate = 0.002  # 0.2% entry and exit fee

eval = (
    pl.read_parquet(output_forecast_file)
    .sort(["symbol", "ts"])
    .join(
        pl.read_parquet(input_ohlcv_file)
        .with_columns(ret=(pl.col("close") / pl.col("open")).log())
        .select(["ts", "symbol", "ret"]),
        on=["ts", "symbol"],
        how="inner",
    )
    # Long (+1) if pred > 0, Short (-1) if pred < 0
    .with_columns([pl.when(pl.col("pred") > 0).then(1).otherwise(-1).alias("position")])
    # 2. Calculate Position Changes
    # Shift position down by 1 to see what we held yesterday.
    # fill_null(0) assumes we start in cash (0) before our first trade.
    .with_columns(
        [pl.col("position").shift(1).fill_null(0).over("symbol").alias("prev_position")]
    )
    # 3. Calculate Transaction Costs
    # Math: If we go from +1 to -1, the absolute difference is 2.
    # 2 * 0.002 = 0.004 (0.4% total fee to flip the position).
    .with_columns(
        [
            ((pl.col("position") - pl.col("prev_position")).abs() * fee_rate).alias(
                "tx_fees"
            )
        ]
    )
    # 4. Calculate Net Returns
    # Strategy Return = Gross Return - Transaction Fees
    .with_columns(
        [
            (pl.col("position") * pl.col("ret") - pl.col("tx_fees")).alias(
                "strategy_ret"
            ),
            pl.col("ret").alias("bnh_ret"),
        ]
    )
)

# 5. Aggregate portfolio performance per day across all symbols
portfolio_daily = (
    eval.group_by("ts")
    .agg([pl.col("strategy_ret").mean(), pl.col("bnh_ret").mean()])
    .sort("ts")
)

# 6. Calculate Cumulative Returns
portfolio_daily = portfolio_daily.with_columns(
    [
        (1 + pl.col("strategy_ret")).cum_prod().alias("cum_strategy"),
        (1 + pl.col("bnh_ret")).cum_prod().alias("cum_bnh"),
    ]
)

final_strat = portfolio_daily["cum_strategy"][-1] - 1
final_bnh = portfolio_daily["cum_bnh"][-1] - 1

print(f"\n--- Test Set Cumulative Performance (Net of 0.2% Fees) ---")
print(f"CatBoost + TS2Vec L/S Strategy : {final_strat * 100:.2f}%")
print(f"Buy & Hold Baseline            : {final_bnh * 100:.2f}%")

import matplotlib.pyplot as plt

# ==========================================
# 8. PLOTTING
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(
    portfolio_daily["ts"],
    portfolio_daily["cum_strategy"],
    label="TS2Vec Strategy",
    color="blue",
    linewidth=2,
)
plt.plot(
    portfolio_daily["ts"],
    portfolio_daily["cum_bnh"],
    label="Buy & Hold",
    color="gray",
    alpha=0.7,
)
plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
plt.title("TS2Vec Embeddings vs. Buy & Hold")
plt.ylabel("Cumulative Growth")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
