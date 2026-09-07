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
# # HeroRATs: IVTS regime-switching between VBK and TLT
#
# Backtest of the strategy from Chrilly Donninger's working paper
# "How to beat the market with the Implied Volatility Term Structure: The HeroRATs
# Strategy" (Sibyl-Working-Paper, Dec. 2013).
#
# The strategy switches between a small-cap growth ETF (VBK) and a long-duration
# treasury ETF (TLT) based on the implied-volatility term structure (IVTS):
#
# * IVTS below the low threshold  -> 100% VBK
# * IVTS between the thresholds   -> 50% VBK / 50% TLT (rebalanced daily)
# * IVTS above the high threshold -> 100% TLT
#
# Only the **VIX + VIX-futures term** signal variants are backtested here (the
# VXST-based variants and the options-based VXV are not available in the provided
# data — see the missing-data section of REPORT.md):
#
# * VIX/VX30  (low 0.97, high 1.10)
# * VIX/VX45  (low 0.98, high 1.06)
# * VX30/VX45 (low 0.95, high 1.05)
#
# VX30 / VX45 are constant-maturity VIX-future prices (30 and 45 calendar days)
# interpolated from the nearest monthly VIX futures exactly as described in the
# paper. The IVTS is optionally smoothed with a trailing median filter
# (median-3 / median-5) before thresholding.

# %% tags=["parameters"]
variants = "VIX/VX30,VIX/VX45,VX30/VX45"
filters = "none,median-3,median-5"
fee_bps = "0"
start_date = "2010-01-04"
end_date = "2026-09-04"
initial_equity = "500"
paper_start = "2011-01-03"
paper_end = "2013-12-11"
fee_grid = "0,2,5,10"
fee_sensitivity_variant = "VIX/VX30|median-5"

# %%
import sys

sys.path.insert(0, "/Users/kai/studies")

import datetime as dt
import numpy as np
import polars as pl
import scrapbook as sb

import backtest_ng as bt

variants_P = [v.strip() for v in variants.split(",") if v.strip()]
filters_P = [f.strip() for f in filters.split(",") if f.strip()]
fee_bps_P = float(fee_bps)
start_date_P = dt.datetime.fromisoformat(start_date)
end_date_P = dt.datetime.fromisoformat(end_date)
initial_equity_P = float(initial_equity)
paper_start_P = dt.datetime.fromisoformat(paper_start)
paper_end_P = dt.datetime.fromisoformat(paper_end)
fee_grid_P = [float(f) for f in fee_grid.split(",")]
fee_variant_P = tuple(fee_sensitivity_variant.split("|"))

# Paper thresholds per variant (Table-1 of the paper, calibrated 2011-2013).
THRESHOLDS = {
    "VIX/VX30": (0.97, 1.10),
    "VIX/VX45": (0.98, 1.06),
    "VX30/VX45": (0.95, 1.05),
}

print(f"""
variants: {variants_P}
filters: {filters_P}
fee_bps: {fee_bps_P}
start_date: {start_date_P.date()}
end_date: {end_date_P.date()}
initial_equity: {initial_equity_P}
paper period: {paper_start_P.date()} .. {paper_end_P.date()}
thresholds: {THRESHOLDS}
""")

# %%
# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
# vix-futures.parquet: daily VIX futures term structure, columns ts, m1..m9
#   (m1 = nearest monthly VIX future, ... m9). Downloaded from vixcentral.com
#   (see ../vix.py). Covers 2010-01-04 .. 2026-09-04.
# output.parquet: daily OHLCV for ^VIX, SPY, IEF, TLT, VBK
#   (see ../yf.py). Columns ts, symbol, open, high, low, close, volume.

fut = pl.read_parquet("vix-futures.parquet").sort("ts")
px = pl.read_parquet("output.parquet")


def third_friday(y: int, m: int) -> dt.date:
    d = dt.date(y, m, 1)
    offset = (4 - d.weekday()) % 7
    return d + dt.timedelta(days=offset + 14)


def vix_future_settlements(y0: int, y1: int) -> list[dt.date]:
    """Settlement dates of monthly VIX futures.

    A VIX future with contract month M settles on the Wednesday that is
    30 days before the third Friday of the month following M.
    """
    out = []
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            nxt = m + 1
            ny, nm = (y + 1, 1) if nxt == 13 else (y, nxt)
            out.append(third_friday(ny, nm) - dt.timedelta(days=30))
    return sorted(out)


# --- constant-maturity VX30 / VX45 from the futures curve -------------------
SD = np.array(vix_future_settlements(2009, 2027), dtype="datetime64[D]")
t = fut["ts"].dt.cast_time_unit("ms").to_numpy().astype("datetime64[D]")
idx = np.searchsorted(SD, t, side="left")
D = np.stack([(SD[idx + i] - t).astype(int) for i in range(9)], axis=1)
m1 = fut["m1"].to_numpy()
m2 = fut["m2"].to_numpy()
m3 = fut["m3"].to_numpy()
d1, d2, d3 = D[:, 0], D[:, 1], D[:, 2]

# VX30: linear interpolation of the 1st and 2nd nearest futures to 30 days.
vx30 = m1 * (d2 - 30) / (d2 - d1) + m2 * (30 - d1) / (d2 - d1)
# VX45: linear interpolation of the 2nd and 3rd nearest futures to 45 days.
vx45 = m2 * (d3 - 45) / (d3 - d2) + m3 * (45 - d2) / (d3 - d2)

ivts = pl.DataFrame({"ts": fut["ts"], "vx30": vx30, "vx45": vx45}).sort("ts")

# --- VIX index --------------------------------------------------------------
vix = (
    px.filter(pl.col("symbol") == "^VIX")
    .select("ts", pl.col("close").alias("vix"))
    .sort("ts")
)
ivts = ivts.join(vix, on="ts", how="left").sort("ts").with_columns(
    vix_vx30=pl.col("vix") / pl.col("vx30"),
    vix_vx45=pl.col("vix") / pl.col("vx45"),
    vx30_vx45=pl.col("vx30") / pl.col("vx45"),
)

# --- trailing median filters (median-3 / median-5) --------------------------
for c in ["vix_vx30", "vix_vx45", "vx30_vx45"]:
    ivts = ivts.with_columns(
        pl.col(c).rolling_median(window_size=3, min_samples=1).alias(f"{c}_m3"),
        pl.col(c).rolling_median(window_size=5, min_samples=1).alias(f"{c}_m5"),
    )

# --- asset prices -----------------------------------------------------------
assets = (
    px.filter(pl.col("symbol").is_in(["VBK", "TLT", "SPY"]))
    .select("ts", "symbol", "close")
    .sort(["symbol", "ts"])
)

# %%
# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
# Universe: VBK / TLT / SPY daily closes plus the (filtered) IVTS value for
# the day. The alpha reads the IVTS, maps it to a regime and emits signals;
# EqualWeight turns 1 signal into 100% and 2 signals into 50/50, which is
# exactly the paper's allocation rule. Rebalancing is daily at the close
# (same-bar assumption, matching the paper's "calculation each day with
# close-prices / shortly before the close").


class HeroRATs(bt.AlphaModel):
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def __call__(self, history: pl.DataFrame, u: bt.Universe) -> list[bt.Signal]:
        today = u.df()[u.timestamp_col()].max()
        row = u.df().filter(pl.col(u.timestamp_col()) == today).row(0, named=True)
        iv = row["ivts"]
        if iv is None:
            return []
        if iv < self.low:
            return [bt.Signal("VBK", True, 1.0)]
        if iv > self.high:
            return [bt.Signal("TLT", True, 1.0)]
        return [bt.Signal("VBK", True, 0.5), bt.Signal("TLT", True, 0.5)]


def run_backtest(ivts_col: str, low: float, high: float, fee_bps: float, start=None):
    sig = ivts.select("ts", pl.col(ivts_col).alias("ivts"))
    uni = (
        assets.join(sig, on="ts", how="left")
        .sort(["symbol", "ts"])
        .with_columns(pl.col("ivts").forward_fill().over("symbol"))
    )
    u = bt.Manual(uni, timestamp_col="ts", symbol_col="symbol", price_col="close")
    test = bt.Backtest(
        u,
        alpha=HeroRATs(low, high),
        portfolio=bt.EqualWeight(),
        risk=bt.NoRisk(),
        execution=bt.Simple(fee_bps=fee_bps),
        period=1,  # daily rebalancing
        benchmark="SPY",
        title=f"HeroRATs {ivts_col}",
    )
    test.run(start=start or start_date_P, initial_equity=initial_equity_P)
    return test


def equity_series(test):
    hist = test.history
    eq = (hist["cash"] + hist["long_notational"] + hist["short_notational"]).to_list()
    ts = hist["ts"].to_list()
    return ts, eq


results = {}
for variant in variants_P:
    for f in filters_P:
        col = variant.replace("/", "_").lower()
        if f == "median-3":
            col += "_m3"
        elif f == "median-5":
            col += "_m5"
        low, high = THRESHOLDS[variant]
        test = run_backtest(col, low, high, fee_bps_P)
        results[(variant, f)] = test
        print(f"done {variant} / {f}")

# %%
# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def metrics_from_equity(ts, eq):
    """Overall metrics from a daily equity series (lists)."""
    ts = np.array(ts, dtype="datetime64[D]")
    eq = np.asarray(eq, dtype=float)
    rets = np.diff(eq) / eq[:-1]
    n_years = (ts[-1] - ts[0]).astype(int) / 365.25
    cagr = (eq[-1] / eq[0]) ** (1.0 / n_years) - 1.0
    sharpe = np.sqrt(365.25) * rets.mean() / rets.std() if rets.std() > 0 else 0.0
    downside = rets[rets < 0]
    sortino = (
        np.sqrt(365.25) * rets.mean() / downside.std()
        if len(downside) > 1 and downside.std() > 0
        else 0.0
    )
    peak = np.maximum.accumulate(eq)
    maxdd = float((eq / peak - 1.0).min())
    total = eq[-1] / eq[0] - 1.0
    return {
        "total": total,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "maxdd": maxdd,
        "n_days": len(rets),
    }


def yearwise(ts, eq):
    """Per-calendar-year simple return and max drawdown within the year."""
    df = pl.DataFrame({"ts": ts, "eq": eq}).sort("ts").with_columns(
        year=pl.col("ts").dt.year(),
        ret=pl.col("eq").pct_change(),
    )
    out = {}
    for y, g in df.group_by("year"):
        y = int(y[0])
        first, last = g["eq"].first(), g["eq"].last()
        e = g["eq"].to_list()
        peak = np.maximum.accumulate(e)
        out[y] = {
            "ret": last / first - 1.0,
            "maxdd": float((np.asarray(e) / peak - 1.0).min()),
        }
    return out


# Buy & hold benchmarks (aligned to the backtest window)
px_assets = (
    px.filter(pl.col("symbol").is_in(["SPY", "VBK", "TLT"]))
    .select("ts", "symbol", "close")
    .sort(["symbol", "ts"])
)
bench = {}
bench_yearly = {}
for sym in ["SPY", "VBK", "TLT"]:
    s = px_assets.filter(pl.col("symbol") == sym)
    s = s.filter((pl.col("ts") >= start_date_P) & (pl.col("ts") <= end_date_P))
    bench[sym] = metrics_from_equity(s["ts"].to_list(), s["close"].to_list())
    bench_yearly[sym] = yearwise(s["ts"].to_list(), s["close"].to_list())

# %%
# ---------------------------------------------------------------------------
# Overall summary
# ---------------------------------------------------------------------------
rows = []
for (variant, f), test in results.items():
    ts, eq = equity_series(test)
    m = metrics_from_equity(ts, eq)
    rows.append(
        {
            "variant": variant,
            "filter": f,
            "total": m["total"],
            "cagr": m["cagr"],
            "sharpe": m["sharpe"],
            "sortino": m["sortino"],
            "maxdd": m["maxdd"],
            "trades": test.trades.height,
        }
    )
summary = pl.DataFrame(rows).sort(["variant", "filter"])

print("\n=== OVERALL (gross of fees, %s .. %s) ===" % (start_date_P.date(), end_date_P.date()))
for r in rows:
    print(
        f"{r['variant']:>9} {r['filter']:>9} | total {r['total']*100:7.1f}% "
        f"cagr {r['cagr']*100:5.2f}% sharpe {r['sharpe']:5.2f} "
        f"sortino {r['sortino']:5.2f} maxdd {r['maxdd']*100:6.1f}% trades {r['trades']}"
    )
print("\n=== BUY & HOLD BENCHMARKS ===")
for sym, m in bench.items():
    print(
        f"{sym:>9} {'B&H':>9} | total {m['total']*100:7.1f}% cagr {m['cagr']*100:5.2f}% "
        f"sharpe {m['sharpe']:5.2f} sortino {m['sortino']:5.2f} maxdd {m['maxdd']*100:6.1f}%"
    )

# %%
# ---------------------------------------------------------------------------
# Paper-period validation (2011-01-03 .. 2013-12-11)
# ---------------------------------------------------------------------------
print("\n=== PAPER PERIOD VALIDATION (2011-01-03 .. 2013-12-11) ===")
PAPER = {
    ("VIX/VX30", "median-5"): (84.6, 12.9),
    ("VIX/VX45", "median-5"): (75.0, 12.9),
    ("VX30/VX45", "median-5"): (51.5, 10.3),
    ("VBK", "B&H"): (49.5, 28.9),
    ("SPY", "B&H"): (49.0, 18.6),
}
for (variant, f), test in results.items():
    if f != "median-5":
        continue
    ts, eq = equity_series(test)
    ts = np.array(ts, dtype="datetime64[D]")
    eq = np.asarray(eq)
    mask = (ts >= np.datetime64("2011-01-03")) & (ts <= np.datetime64("2013-12-11"))
    ts2, eq2 = ts[mask], eq[mask]
    m = metrics_from_equity(ts2, eq2)
    ref = PAPER.get((variant, f), (float("nan"), float("nan")))
    print(
        f"{variant:>9} median-5 | ours {m['total']*100:6.1f}% dd {m['maxdd']*100:6.1f}%"
        f" | paper {ref[0]:6.1f}% dd {ref[1]:4.1f}%"
    )
for sym in ["VBK", "SPY"]:
    s = px_assets.filter(pl.col("symbol") == sym)
    s = s.filter(
        (pl.col("ts") >= paper_start_P) & (pl.col("ts") <= paper_end_P)
    )
    m = metrics_from_equity(s["ts"].to_list(), s["close"].to_list())
    ref = PAPER[(sym, "B&H")]
    print(
        f"{sym:>9} B&H      | ours {m['total']*100:6.1f}% dd {m['maxdd']*100:6.1f}%"
        f" | paper {ref[0]:6.1f}% dd {ref[1]:4.1f}%"
    )

# %%
# ---------------------------------------------------------------------------
# Yearwise table
# ---------------------------------------------------------------------------
keys = list(results.keys())
print("\n=== YEARWISE RETURNS (%) ===")
all_ts = [t for test in results.values() for t in test.history["ts"].to_list()]
years = sorted({ts_.year for ts_ in all_ts})
strat_yearly = {}
for (variant, f) in keys:
    ts, eq = equity_series(results[(variant, f)])
    strat_yearly[(variant, f)] = yearwise(ts, eq)

hdr = "year".ljust(6) + "".join(f"{v}/{f}".ljust(22) for (v, f) in keys) + "SPY".ljust(10) + "VBK".ljust(10) + "TLT".ljust(10)
print(hdr)
for y in years:
    line = str(y).ljust(6)
    for (variant, f) in keys:
        yw = strat_yearly[(variant, f)].get(y)
        line += f"{yw['ret']*100:9.1f} " if yw else "     n/a  "
    for sym in ["SPY", "VBK", "TLT"]:
        yw = bench_yearly[sym].get(y)
        line += f"{yw['ret']*100:8.1f} " if yw else "    n/a  "
    print(line)

# %%
# ---------------------------------------------------------------------------
# Fee sensitivity (headline variant)
# ---------------------------------------------------------------------------
print("\n=== FEE SENSITIVITY (%s) ===" % (fee_sensitivity_variant,))
variant_fee, filter_fee = fee_variant_P
col = variant_fee.replace("/", "_").lower()
if filter_fee == "median-3":
    col += "_m3"
elif filter_fee == "median-5":
    col += "_m5"
low, high = THRESHOLDS[variant_fee]
for fb in fee_grid_P:
    test_fee = run_backtest(col, low, high, fb)
    ts, eq = equity_series(test_fee)
    m = metrics_from_equity(ts, eq)
    fees_paid = float(test_fee.history["fees"].sum())
    print(
        f"fee {fb:>4.0f} bps | total {m['total']*100:7.1f}% cagr {m['cagr']*100:5.2f}% "
        f"sharpe {m['sharpe']:5.2f} maxdd {m['maxdd']*100:6.1f}% fees ${fees_paid:8.0f}"
    )

# %%
# ---------------------------------------------------------------------------
# Charts for REPORT.md
# ---------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

# equity curves, normalised to 1.0
for (variant, f) in [("VIX/VX30", "median-5"), ("VIX/VX30", "none")]:
    ts, eq = equity_series(results[(variant, f)])
    ts = np.array(ts, dtype="datetime64[D]")
    eq = np.asarray(eq)
    axes[0].plot(ts, eq / eq[0], label=f"HeroRATs {variant} {f}")
for sym, color in [("SPY", "goldenrod"), ("VBK", "gray"), ("TLT", "lightsteelblue")]:
    s = px_assets.filter(pl.col("symbol") == sym).sort("ts")
    s = s.filter((pl.col("ts") >= start_date_P) & (pl.col("ts") <= end_date_P))
    ts_b = np.array(s["ts"].to_list(), dtype="datetime64[D]")
    px_b = np.asarray(s["close"].to_list())
    axes[0].plot(ts_b, px_b / px_b[0], label=f"{sym} B&H", color=color,
                 linewidth=1.2 if sym == "SPY" else 0.9,
                 alpha=1.0 if sym == "SPY" else 0.8)
axes[0].set_yscale("log")
axes[0].legend(loc="upper left")
axes[0].set_title("HeroRATs (VIX/VX30) vs buy & hold, 2010-2026, gross of fees")
axes[0].set_ylabel("Equity (normalised)")

# drawdown of headline variant + SPY
for (variant, f), color in [(("VIX/VX30", "median-5"), "tab:blue"), (("SPY", "x"), "goldenrod")]:
    if f == "x":
        s = px_assets.filter(pl.col("symbol") == "SPY").sort("ts")
        s = s.filter((pl.col("ts") >= start_date_P) & (pl.col("ts") <= end_date_P))
        ts_d = np.array(s["ts"].to_list(), dtype="datetime64[D]")
        px_d = np.asarray(s["close"].to_list())
    else:
        ts_d, px_d = equity_series(results[(variant, f)])
        ts_d = np.array(ts_d, dtype="datetime64[D]")
        px_d = np.asarray(px_d)
    dd = px_d / np.maximum.accumulate(px_d) - 1.0
    axes[1].fill_between(ts_d, dd * 100, 0, alpha=0.4, label=f"{variant} {f}" if f != "x" else "SPY B&H")
axes[1].set_ylabel("Drawdown (%)")
axes[1].legend(loc="lower left")

plt.tight_layout()
plt.savefig("herorat-equity.png", dpi=110)
print("saved herorat-equity.png")

# IVTS chart with regime thresholds (headline variant)
fig, ax = plt.subplots(figsize=(12, 4))
low, high = THRESHOLDS["VIX/VX30"]
iv = ivts.select("ts", pl.col("vix_vx30_m5").alias("ivts")).to_pandas()
ax.plot(iv["ts"], iv["ivts"], linewidth=0.6, color="tab:purple")
ax.axhline(low, color="tab:green", linestyle="--", linewidth=0.8, label=f"low {low}")
ax.axhline(high, color="tab:red", linestyle="--", linewidth=0.8, label=f"high {high}")
ax.set_title("IVTS = VIX/VX30, median-5 filtered")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("herorat-ivts.png", dpi=110)
print("saved herorat-ivts.png")

# %%
# ---------------------------------------------------------------------------
# Glue key metrics for papermill
# ---------------------------------------------------------------------------
for (variant, f), test in results.items():
    ts, eq = equity_series(test)
    m = metrics_from_equity(ts, eq)
    key = f"{variant.replace('/', '_').lower()}_{f}"
    sb.glue(f"{key}_cagr", m["cagr"])
    sb.glue(f"{key}_sharpe", m["sharpe"])
    sb.glue(f"{key}_sortino", m["sortino"])
    sb.glue(f"{key}_maxdd", m["maxdd"])
    sb.glue(f"{key}_total", m["total"])
