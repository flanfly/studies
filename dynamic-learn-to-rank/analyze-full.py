# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0", "numpy>=1.26"]
# ///
"""Full-panel analysis: seed selection + tables (tau=0.015)."""
import math
import numpy as np
import polars as pl

PERIODS_PER_YEAR = 730
ORDER = ["DRL_rank", "DRL+filt:EW", "DRL+filt:IV", "DRL+filt:MinVar",
         "DRL+filt:MaxDiv", "DRL+filt:RiskParity", "DRL+filt:MinCVaR", "JT_mom", "Random"]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 42]


def metrics(rets: np.ndarray) -> dict:
    rets = rets[np.isfinite(rets)]
    ann = math.sqrt(PERIODS_PER_YEAR)
    mu, sd = rets.mean(), rets.std(ddof=1)
    neg = rets[rets < 0]
    sd_d = neg.std(ddof=1) if len(neg) > 1 else np.nan
    equity = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    return {
        "Cum. Return": equity[-1] - 1,
        "Ann. Volatility": sd * ann,
        "MDD": float(np.min(equity / peak - 1)),
        "Sharpe": mu / sd * ann if sd > 0 else np.nan,
        "Sortino": mu / sd_d * ann if sd_d and sd_d > 0 else np.nan,
        "Omega": float(rets[rets > 0].sum() / -rets[rets < 0].sum()) if (rets < 0).any() else np.nan,
    }


print("=== Full panel: seed selection on validation folds (DRL_rank EW, 2023-06..2023-11) ===")
val_rows = []
for s in SEEDS:
    df = pl.read_parquet(f"fp-t15-{s}.parquet").filter(
        (pl.col("strategy") == "DRL_rank") & (pl.col("time") < pl.datetime(2023, 12, 1)))
    m = metrics(df["ret_gross"].to_numpy())
    val_rows.append({"seed": s, **m})
val = pl.DataFrame(val_rows).sort("Sharpe", descending=True)
with pl.Config(tbl_rows=20, tbl_cols=20, float_precision=3):
    print(val)
best_seed = int(val["seed"][0])
print(f"\nBEST SEED on validation: {best_seed}")

print("\n=== Seed stability, DRL_rank full period (gross) ===")
stab = []
for s in SEEDS:
    df = pl.read_parquet(f"fp-t15-{s}.parquet").filter(pl.col("strategy") == "DRL_rank")
    m = metrics(df["ret_gross"].to_numpy())
    m["seed"] = s
    m["val_Sharpe"] = val.filter(pl.col("seed") == s)["Sharpe"][0]
    stab.append(m)
with pl.Config(tbl_rows=20, tbl_cols=20, float_precision=3):
    print(pl.DataFrame(stab).sort("Sharpe", descending=True))

df = pl.read_parquet(f"fp-t15-{best_seed}.parquet").with_columns([
    (pl.col("ret_gross") - 5e-4 * pl.col("turnover")).alias("ret_5bp"),
    (pl.col("ret_gross") - 1e-3 * pl.col("turnover")).alias("ret_10bp"),
])
for cost, col in [("gross", "ret_gross"), ("5bp", "ret_5bp"), ("10bp", "ret_10bp")]:
    rows = []
    for k in ORDER:
        d = df.filter(pl.col("strategy") == k).sort("time")
        m = metrics(d[col].to_numpy())
        m["Strategy"] = k
        m["Turnover/step"] = d["turnover"].mean()
        rows.append(m)
    print(f"\n=== Overall performance ({cost}) ===")
    with pl.Config(tbl_rows=20, tbl_cols=20, float_precision=4):
        print(pl.DataFrame(rows))

print("\n=== Yearly breakdown (gross) ===")
rows = []
for k in ORDER:
    d = df.filter(pl.col("strategy") == k).sort("time")
    ts = d["time"].to_numpy().astype("datetime64[Y]")
    r = d["ret_gross"].to_numpy()
    for y in np.unique(ts):
        m = metrics(r[ts == y])
        m["Strategy"] = k
        m["Year"] = str(y)
        rows.append(m)
with pl.Config(tbl_rows=100, tbl_cols=20, float_precision=3):
    print(pl.DataFrame(rows).sort("Strategy", "Year"))

for cost, col in [("net 5bp", "ret_5bp"), ("net 10bp", "ret_10bp")]:
    print(f"\n=== Yearly breakdown, final strategy (DRL+filt:MaxDiv), {cost} ===")
    d = df.filter(pl.col("strategy") == "DRL+filt:MaxDiv").sort("time")
    ts = d["time"].to_numpy().astype("datetime64[Y]")
    r = d[col].to_numpy()
    rows = []
    for y in np.unique(ts):
        m = metrics(r[ts == y])
        m["Year"] = str(y)
        rows.append(m)
    with pl.Config(tbl_rows=20, tbl_cols=20, float_precision=4):
        print(pl.DataFrame(rows))