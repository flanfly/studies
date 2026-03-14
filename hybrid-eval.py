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

seed = 1337

# %%
# prepare data
import polars as pl
import random
import torch
import numpy as np

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

eps = 1e-8

market_df = (
    pl.read_parquet(input_ohlcv_file)
    .filter(pl.col("symbol") == market_symbol)
    .sort("ts")
    .filter(
        (pl.col("open") > 0)
        & (pl.col("high") > 0)
        & (pl.col("low") > 0)
        & (pl.col("close") > 0)
        & (pl.col("base_volume") > 0)
        & (pl.col("quote_volume") > 0)
    )
    .with_columns(
        ref=(pl.col("close") / pl.col("open")).log(),
        ref_cc=(pl.col("close") / pl.col("close").shift(1)).log(),
        ref_range=(pl.col("high") / pl.col("low")).log(),
    )
    .with_columns(
        ref_mom_3=pl.col("ref").rolling_sum(window_size=3),
        ref_mom_7=pl.col("ref").rolling_sum(window_size=7),
        ref_mom_30=pl.col("ref").rolling_sum(window_size=30),
        ref_vol_7=pl.col("ref").rolling_std(window_size=7),
        ref_vol_30=pl.col("ref").rolling_std(window_size=30),
    )
    .select(
        [
            "ts",
            "ref",
            "ref_cc",
            "ref_range",
            "ref_mom_3",
            "ref_mom_7",
            "ref_mom_30",
            "ref_vol_7",
            "ref_vol_30",
        ]
    )
)

df = (
    pl.read_parquet(input_ohlcv_file)
    .sort(["symbol", "ts"])
    .filter(
        (pl.col("open") > 0)
        & (pl.col("high") > 0)
        & (pl.col("low") > 0)
        & (pl.col("close") > 0)
        & (pl.col("base_volume") > 0)
        & (pl.col("quote_volume") > 0)
    )
    .join(market_df, on="ts", how="inner")
    .sort(["symbol", "ts"])
    .with_columns(
        # base returns
        ret=(pl.col("close") / pl.col("open")).log(),
        cc_ret=(pl.col("close") / pl.col("close").shift(1)).log().over("symbol"),
        gap=(pl.col("open") / pl.col("close").shift(1)).log().over("symbol"),
        target=(pl.col("close").shift(-1) / pl.col("open").shift(-1))
        .log()
        .over("symbol"),
        # bar structure
        hl_range=(pl.col("high") / pl.col("low")).log(),
        close_loc=(
            2.0
            * (
                (pl.col("close") - pl.col("low"))
                / (pl.col("high") - pl.col("low") + pl.lit(eps))
            )
            - 1.0
        ),
        body_frac=(
            (pl.col("close") - pl.col("open"))
            / (pl.col("high") - pl.col("low") + pl.lit(eps))
        ),
        upper_wick_frac=(
            (pl.col("high") - pl.max_horizontal("open", "close"))
            / (pl.col("high") - pl.col("low") + pl.lit(eps))
        ),
        lower_wick_frac=(
            (pl.min_horizontal("open", "close") - pl.col("low"))
            / (pl.col("high") - pl.col("low") + pl.lit(eps))
        ),
        wick_asym=(
            (pl.col("high") - pl.max_horizontal("open", "close") + pl.lit(eps))
            / (pl.min_horizontal("open", "close") - pl.col("low") + pl.lit(eps))
        ).log(),
        # range-based vol
        sigma_rs=(
            (
                (pl.col("high") / pl.col("close")).log()
                * (pl.col("high") / pl.col("open")).log()
            )
            + (
                (pl.col("low") / pl.col("close")).log()
                * (pl.col("low") / pl.col("open")).log()
            )
        ).sqrt(),
        # volume / participation
        vwap=pl.col("quote_volume") / (pl.col("base_volume") + pl.lit(eps)),
        vwap_close=(
            pl.col("close")
            / (pl.col("quote_volume") / (pl.col("base_volume") + pl.lit(eps)))
        ).log(),
        vwap_open=(
            (pl.col("quote_volume") / (pl.col("base_volume") + pl.lit(eps)))
            / pl.col("open")
        ).log(),
        log_quote_vol=(pl.col("quote_volume") + pl.lit(eps)).log(),
        log_base_vol=(pl.col("base_volume") + pl.lit(eps)).log(),
        # order-flow proxy in quote units
        flow_imb=(2.0 * pl.col("taker_buy_quote_volume") - pl.col("quote_volume"))
        / (pl.col("quote_volume") + pl.lit(eps)),
    )
    .with_columns(
        # market-relative features
        ret_rel=pl.col("ret") - pl.col("ref"),
        cc_ret_rel=pl.col("cc_ret") - pl.col("ref_cc"),
        range_rel=pl.col("hl_range") - pl.col("ref_range"),
    )
    .with_columns(
        # momentum
        mom_3=pl.col("ret").rolling_sum(window_size=3).over("symbol"),
        mom_7=pl.col("ret").rolling_sum(window_size=7).over("symbol"),
        mom_14=pl.col("ret").rolling_sum(window_size=14).over("symbol"),
        mom_30=pl.col("ret").rolling_sum(window_size=30).over("symbol"),
        mom_90=pl.col("ret").rolling_sum(window_size=90).over("symbol"),
        # residual / relative momentum
        rel_mom_3=pl.col("ret_rel").rolling_sum(window_size=3).over("symbol"),
        rel_mom_7=pl.col("ret_rel").rolling_sum(window_size=7).over("symbol"),
        rel_mom_30=pl.col("ret_rel").rolling_sum(window_size=30).over("symbol"),
        # realized vol
        rv_7=pl.col("ret").rolling_std(window_size=7).over("symbol"),
        rv_30=pl.col("ret").rolling_std(window_size=30).over("symbol"),
        rv_90=pl.col("ret").rolling_std(window_size=90).over("symbol"),
        # volume baselines
        log_quote_vol_mean_7=pl.col("log_quote_vol")
        .rolling_mean(window_size=7)
        .over("symbol"),
        log_quote_vol_std_7=pl.col("log_quote_vol")
        .rolling_std(window_size=7)
        .over("symbol"),
        log_quote_vol_mean_30=pl.col("log_quote_vol")
        .rolling_mean(window_size=30)
        .over("symbol"),
        log_quote_vol_std_30=pl.col("log_quote_vol")
        .rolling_std(window_size=30)
        .over("symbol"),
        # flow baselines
        flow_imb_mean_7=pl.col("flow_imb").rolling_mean(window_size=7).over("symbol"),
        flow_imb_mean_30=pl.col("flow_imb").rolling_mean(window_size=30).over("symbol"),
    )
    .with_columns(
        # vol-adjusted momentum
        mom_7_voladj=pl.col("mom_7") / (pl.col("rv_7") + pl.lit(eps)),
        mom_30_voladj=pl.col("mom_30") / (pl.col("rv_30") + pl.lit(eps)),
        rel_mom_7_voladj=pl.col("rel_mom_7") / (pl.col("rv_7") + pl.lit(eps)),
        rel_mom_30_voladj=pl.col("rel_mom_30") / (pl.col("rv_30") + pl.lit(eps)),
        # vol regime
        vol_ratio_7_30=pl.col("rv_7") / (pl.col("rv_30") + pl.lit(eps)),
        vol_ratio_30_90=pl.col("rv_30") / (pl.col("rv_90") + pl.lit(eps)),
        # volume surprise
        vol_z_7=(
            (pl.col("log_quote_vol") - pl.col("log_quote_vol_mean_7"))
            / (pl.col("log_quote_vol_std_7") + pl.lit(eps))
        ),
        vol_z_30=(
            (pl.col("log_quote_vol") - pl.col("log_quote_vol_mean_30"))
            / (pl.col("log_quote_vol_std_30") + pl.lit(eps))
        ),
        rel_volume_7=pl.col("quote_volume")
        / (
            pl.col("quote_volume").rolling_mean(window_size=7).over("symbol")
            + pl.lit(eps)
        ),
        rel_volume_30=pl.col("quote_volume")
        / (
            pl.col("quote_volume").rolling_mean(window_size=30).over("symbol")
            + pl.lit(eps)
        ),
        # flow surprise / persistence
        flow_dev_7=pl.col("flow_imb") - pl.col("flow_imb_mean_7"),
        flow_dev_30=pl.col("flow_imb") - pl.col("flow_imb_mean_30"),
        # trend quality
        trend_intensity=pl.col("ret") / (pl.col("hl_range") + pl.lit(eps)),
        rel_trend_intensity=pl.col("ret_rel") / (pl.col("hl_range") + pl.lit(eps)),
    )
    .with_columns(
        # cross-sectional ranks, centered around 0
        ret_cs=((pl.col("ret").rank().over("ts") / pl.len().over("ts")) - 0.5),
        ret_rel_cs=((pl.col("ret_rel").rank().over("ts") / pl.len().over("ts")) - 0.5),
        mom_7_cs=((pl.col("mom_7").rank().over("ts") / pl.len().over("ts")) - 0.5),
        mom_30_cs=((pl.col("mom_30").rank().over("ts") / pl.len().over("ts")) - 0.5),
        rel_mom_7_cs=(
            (pl.col("rel_mom_7").rank().over("ts") / pl.len().over("ts")) - 0.5
        ),
        rel_mom_30_cs=(
            (pl.col("rel_mom_30").rank().over("ts") / pl.len().over("ts")) - 0.5
        ),
        rv_30_cs=((pl.col("rv_30").rank().over("ts") / pl.len().over("ts")) - 0.5),
        vol_z_30_cs=(
            (pl.col("vol_z_30").rank().over("ts") / pl.len().over("ts")) - 0.5
        ),
        flow_imb_cs=(
            (pl.col("flow_imb").rank().over("ts") / pl.len().over("ts")) - 0.5
        ),
        close_loc_cs=(
            (pl.col("close_loc").rank().over("ts") / pl.len().over("ts")) - 0.5
        ),
    )
    .select(
        [
            # base columns
            "ts",
            "symbol",
            "target",
            # market state
            "ref",
            "ref_cc",
            "ref_range",
            "ref_mom_3",
            "ref_mom_7",
            "ref_mom_30",
            "ref_vol_7",
            "ref_vol_30",
            # own returns / relative returns
            "ret",
            "cc_ret",
            "gap",
            "ret_rel",
            "cc_ret_rel",
            "range_rel",
            # bar structure
            "hl_range",
            "close_loc",
            "body_frac",
            "upper_wick_frac",
            "lower_wick_frac",
            "wick_asym",
            "sigma_rs",
            "trend_intensity",
            "rel_trend_intensity",
            # volume / execution / flow
            "vwap_close",
            "vwap_open",
            "log_quote_vol",
            "log_base_vol",
            "flow_imb",
            "flow_dev_7",
            "flow_dev_30",
            "vol_z_7",
            "vol_z_30",
            "rel_volume_7",
            "rel_volume_30",
            # momentum / vol
            "mom_3",
            "mom_7",
            "mom_14",
            "mom_30",
            "mom_90",
            "rel_mom_3",
            "rel_mom_7",
            "rel_mom_30",
            "rv_7",
            "rv_30",
            "rv_90",
            "mom_7_voladj",
            "mom_30_voladj",
            "rel_mom_7_voladj",
            "rel_mom_30_voladj",
            "vol_ratio_7_30",
            "vol_ratio_30_90",
            # cross-sectional context
            "ret_cs",
            "ret_rel_cs",
            "mom_7_cs",
            "mom_30_cs",
            "rel_mom_7_cs",
            "rel_mom_30_cs",
            "rv_30_cs",
            "vol_z_30_cs",
            "flow_imb_cs",
            "close_loc_cs",
        ]
    )
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
from tqdm import tqdm

df = pl.read_parquet(output_features_file).sort(["symbol", "ts"])

featcols = sorted(
    [col for col in df.columns if col not in ["ts", "symbol", "target", "ref"]]
)
print(f"feature columns: {featcols} ({len(featcols)})")

ts = df["ts"].unique().sort()
print(f"{len(ts)} unique timestamps, from {ts[0]} to {ts[-1]}")


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
                    *[pl.col(c).fill_null(np.nan) for c in featcols],
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

    assert not np.isnan(xcube).all()
    assert not np.isinf(xcube).any()
    assert xcube.shape == (len(symbols), len(in_ts), len(featcols))

    assert ymat.shape == (len(symbols), len(in_ts))
    assert mask.shape == (len(symbols), len(in_ts))
    assert ts.shape == (len(symbols), len(in_ts))
    assert syms.shape == (len(symbols), len(in_ts))

    return (xcube, ymat, ts, syms, mask)


def extract_gbt_matrix(
    emb: np.ndarray,
    tgt: np.ndarray,
    ts: np.ndarray,
    syms: np.ndarray,
    mask: np.ndarray,
    is_inference: bool,
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

    if is_inference:
        mask = ~vec_mask
    else:
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


# fit TS2Vec to training data, restart if it doesn't converge
# returns fitted ts2vec and catboost models and the training set stats for normalizing
def fit_hybrid_model(
    train_ts, val_ts, df, symbols, featcols
) -> Tuple[TS2Vec, CatBoostRegressor, pl.DataFrame]:
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
    train_pnfc, train_tm, train_tsm, train_sm, train_fm = (
        padded_normalized_feature_cube(train_ts, df, train_stats, symbols, featcols)
    )

    loss_log = [np.nan]
    for i in range(5):  # max 5 attempts
        ts2vec_model = TS2Vec(
            input_dims=len(featcols),
            device=device,
            output_dims=ts2vec_dim,
            lr=ts2vec_lr,
        )
        loss_log = ts2vec_model.fit(train_pnfc, n_epochs=ts2vec_epochs, verbose=True)
        if not np.isnan(loss_log).any() and not np.isinf(loss_log).any():
            break

    X_train, df_train = extract_gbt_matrix(
        ts2vec_model.encode(train_pnfc, causal=True),
        train_tm,
        train_tsm,
        train_sm,
        train_fm,
        False,
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
        False,
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
        random_seed=seed,
    )
    gbt_model.fit(X_train, y_train, eval_set=(X_val, y_val))

    return ts2vec_model, gbt_model, train_stats


res_ary = []
for offset in range(0, len(ts), test_days):
    train_start = offset
    val_start = train_start + training_days
    test_start = val_start + validation_days
    test_end = min(len(ts), test_start + test_days)
    if test_start >= test_end:
        break

    train_ts = ts[train_start:val_start]
    val_ts = ts[val_start:test_start]
    test_ts = ts[test_start:test_end]

    print(f"train    {train_ts[0]}-{train_ts[-1]} ({len(train_ts)})")
    print(f"validate {val_ts[0]}-{val_ts[-1]} ({len(val_ts)})")
    print(f"test     {test_ts[0]}-{test_ts[-1]} ({len(test_ts)})")

    symbols = df["symbol"].unique().sort().to_list()
    print(f"{len(symbols)} unique symbols")

    ts2vec_model, gbt_model, train_stats = fit_hybrid_model(
        train_ts=train_ts,
        val_ts=val_ts,
        df=df.drop_nulls(["target"]),
        symbols=symbols,
        featcols=featcols,
    )

    warm_test_ts = pl.concat([val_ts, test_ts])

    for td in tqdm(test_ts):
        my_ts = warm_test_ts.filter(warm_test_ts < td).sort()[-forecast_lookback:]
        my_pnfc, my_tm, my_tsm, my_sm, my_fm = padded_normalized_feature_cube(
            my_ts, df.filter(pl.col("ts") <= td), train_stats, symbols, featcols
        )
        X_my, df_my = extract_gbt_matrix(
            ts2vec_model.encode(my_pnfc, causal=True),
            my_tm,
            my_tsm,
            my_sm,
            my_fm,
            True,
        )

        # (symbols, timestamps)
        df_my = (
            df_my.with_columns(pred=gbt_model.predict(X_my))
            .filter(pl.col("ts") == my_ts[-1])
            .with_columns(
                ts=pl.lit(td).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")
            )
            .sort(["symbol"])
        )
        res_ary.append(df_my)

pl.concat(res_ary).select(["ts", "symbol", "pred"]).write_parquet(output_forecast_file)
# %%
# portfolio evaluation with transaction costs


def normalize_and_evaluate(df: pl.DataFrame, fee_rate: float) -> pl.DataFrame:
    return (
        df
        # normalize to maintain constant leverage of 1
        .with_columns(pos=pl.col("pos") / pl.col("pos").abs().sum().over("ts"))
        .with_columns(
            gross=pl.col("pos") * (pl.col("ret").exp() - 1),
            fee=2 * pl.col("pos").abs() * fee_rate,
        )
        .with_columns(
            net=pl.col("gross") - pl.col("fee"),
        )
        .select(["ts", "symbol", "net", "fee"])
    )


fee_rate = 0.002  # 0.2% entry and exit fee (0.4% total)

strat = (
    pl.read_parquet(output_forecast_file)
    .join(
        pl.read_parquet(output_features_file).select(["ts", "symbol", "ret"]),
        on=["ts", "symbol"],
        how="inner",
    )
    .sort(["symbol", "ts"])
    # portfolio is top decile of predictions, long if pred > 0, short if pred < 0
    .filter(
        (pl.col("pred").abs().rank(descending=False).over("ts") / pl.len().over("ts"))
        > 0.9
    )
    # long/short positions equal weighted
    .with_columns([pl.when(pl.col("pred") > 0).then(1).otherwise(-1).alias("pos")])
)
strat = normalize_and_evaluate(strat, fee_rate)

naive = (
    pl.read_parquet(output_features_file)
    .select(["ts", "symbol", "ret"])
    .sort(["symbol", "ts"])
    .with_columns(naive=pl.col("ret").shift(1).over("symbol"))
    .filter(pl.col("ts").is_in(pl.lit(strat["ts"].unique()).implode()))
    .drop_nulls(["naive"])
    .with_columns(
        pos=pl.when(pl.col("naive") > 0).then(1).otherwise(-1),
    )
)

naive = normalize_and_evaluate(naive, fee_rate)

eval = (
    strat.rename({"net": "strategy_ret", "fee": "strategy_fee"})
    .join(
        naive.rename({"net": "naive_ret", "fee": "naive_fee"}),
        on=["ts", "symbol"],
        how="right",
    )
    .join(
        pl.read_parquet(output_features_file).select(["ts", "symbol", "ref"]),
        on=["ts", "symbol"],
        how="inner",
    )
    .sort(["symbol", "ts"])
)

# 5. Aggregate portfolio performance per day across all symbols
portfolio_daily = (
    eval.group_by("ts")
    .agg(
        [
            pl.col("strategy_ret").sum(),
            pl.col("naive_ret").sum(),
            pl.col("ref").first().alias("bnh_ret"),
        ]
    )
    .sort("ts")
)

# 6. Calculate Cumulative Returns
portfolio_daily = portfolio_daily.with_columns(
    cum_strategy=(1 + pl.col("strategy_ret")).cum_prod() - 1,
    cum_naive=(1 + pl.col("naive_ret")).cum_prod() - 1,
    cum_bnh=(pl.col("bnh_ret").cum_sum().exp() - 1),
)

final_strat = portfolio_daily["cum_strategy"][-1]
final_naive = portfolio_daily["cum_naive"][-1]
final_bnh = portfolio_daily["cum_bnh"][-1]

print(f"\n--- Test Set Cumulative Performance (Net of 0.2% Fees) ---")
print(f"CatBoost + TS2Vec L/S Strategy : {final_strat * 100:.2f}%")
print(f"Naive forecast L/S             : {final_naive * 100:.2f}%")
print(f"Buy & Hold {market_symbol}            : {final_bnh * 100:.2f}%")

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
    portfolio_daily["cum_naive"],
    label="Naive",
    color="gray",
    alpha=0.7,
)
plt.plot(
    portfolio_daily["ts"],
    portfolio_daily["cum_bnh"],
    label="Buy & Hold",
    color="orange",
    alpha=0.7,
)
plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
plt.title("TS2Vec Embeddings vs. Buy & Hold")
plt.ylabel("Cumulative Growth")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%
