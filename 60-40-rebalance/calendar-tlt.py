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
buy_day = "19"
tbf_sell_day = "7"

# %%
buy_day_P = int(buy_day)
tbf_sell_day_P = int(tbf_sell_day)

assert 1 <= buy_day_P <= 23, "buy_day must be a valid trading day of the month"
assert 1 <= tbf_sell_day_P < buy_day_P, "tbf sell day must precede the tlt buy day"

print(f"""
buy_day: {buy_day_P}
tbf_sell_day: {tbf_sell_day_P}
""")

# %%
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scrapbook as sb

# %%
# Data: daily OHLCV for TLT and TBF from Yahoo Finance (dividend-adjusted).
# Regenerate with: uv run yf.py TLT TBF --output calendar-tlt/output.parquet
df = (
    pl.read_parquet("calendar-tlt/output.parquet")
    .sort("symbol", "ts")
    .with_columns(
        year=pl.col("ts").dt.year(),
        month=pl.col("ts").dt.month(),
        month_index=pl.col("ts").dt.year() * 12 + pl.col("ts").dt.month(),
        prev_month_index=pl.col("ts").dt.year().shift(1).over("symbol") * 12
        + pl.col("ts").dt.month().shift(1).over("symbol"),
        ret_cc=pl.col("close").pct_change().over("symbol"),
        ret_oc=pl.col("close") / pl.col("open") - 1,
    )
    .with_columns(
        day_rank=pl.int_range(1, pl.len() + 1).over("symbol", "year", "month"),
        first_month_index=pl.col("month_index").min().over("symbol"),
    )
)

# %%
# Backtest
# --------
# Rules:
#   * Buy TLT on the open of the 19th trading day of the month.
#   * Hold until the close of the last trading day of the month, sell TLT
#     there and buy TBF at the same close.
#   * Sell TBF on the close of the 7th trading day of the following month.
#   * Flat until the next month's 19th trading day.
# If TBF has no data yet (it starts 2009-08), the leg is skipped and the
# strategy sits in cash after selling TLT.

tlt_leg = df.filter(
    (pl.col("symbol") == "TLT") & (pl.col("day_rank") >= buy_day_P)
).select(
    "ts", "year",
    ret=(
        pl.when(pl.col("day_rank") == buy_day_P)
        .then(pl.col("ret_oc"))  # bought at the open, earn open -> close
        .otherwise(pl.col("ret_cc"))
    ),
)

tbf_leg = df.filter(
    (pl.col("symbol") == "TBF")
    & (pl.col("day_rank") <= tbf_sell_day_P)
    & (pl.col("month_index") > pl.col("first_month_index"))  # a prior month-end close exists
).select("ts", "year", ret=pl.col("ret_cc"))

# The legs never overlap (tbf_sell_day_P < buy_day_P), so summing is safe.
pos = (
    pl.concat([tlt_leg, tbf_leg])
    .group_by("ts")
    .agg(ret=pl.col("ret").sum(), year=pl.col("year").first())
    .sort("ts")
)

# Fill non-holding days with a zero return.
pos = (
    df.select("ts", "year")
    .unique()
    .sort("ts")
    .join(pos.select("ts", "ret"), on="ts", how="left")
    .with_columns(ret=pl.col("ret").fill_null(0.0))
    .with_columns(equity=pl.col("ret").fill_null(0.0).add(1).cum_prod())
)


def metrics(ret: pl.Series, ts: pl.Series) -> dict:
    """CAGR, Sharpe, Sortino and max drawdown from daily returns."""
    r = np.asarray(ret, dtype=float)
    years = (max(ts) - min(ts)).days / 365.25
    cagr = np.prod(1 + r) ** (1 / years) - 1 if years > 0 else np.nan
    mean, std = r.mean(), r.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    sharpe = mean / std * np.sqrt(252) if std > 0 else np.nan
    sortino = mean / downside * np.sqrt(252) if downside > 0 else np.nan
    eq = np.cumprod(1 + r)
    max_dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return {"CAGR": cagr, "Sharpe": sharpe, "Sortino": sortino, "MaxDD": max_dd}


def yearwise(d: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (year, ts, ret) in d.group_by("year").agg(pl.col("ts"), pl.col("ret")).sort(
        "year"
    ).iter_rows():
        m = metrics(ret, ts)
        m["year"] = int(year)
        rows.append(m)
    return pl.DataFrame(rows).select("year", *metrics_order)


metrics_order = ["CAGR", "Sharpe", "Sortino", "MaxDD"]
overall = metrics(pos["ret"], pos["ts"])
yw = yearwise(pos)

# TLT buy & hold benchmark over the same span for context.
bench = (
    df.filter(pl.col("symbol") == "TLT")
    .select("ts", "year", "ret_cc")
    .drop_nulls()
    .rename({"ret_cc": "ret"})
)
bench_yw = yearwise(bench)

print(f"\nOverall ({pos['ts'].min().date()} to {pos['ts'].max().date()}):")
print(f"  TLT/TBF calendar: " + "  ".join(f"{k}={overall[k]:.2%}" if k in ("CAGR", "MaxDD") else f"{k}={overall[k]:.2f}" for k in metrics_order))
bm = metrics(bench["ret"], bench["ts"])
print("  TLT buy & hold:   " + "  ".join(f"{k}={bm[k]:.2%}" if k in ("CAGR", "MaxDD") else f"{k}={bm[k]:.2f}" for k in metrics_order))

print("\nYearwise (strategy | TLT buy & hold):")
with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=200, float_precision=3):
    print(
        yw.join(
            bench_yw.rename({c: f"{c}_bh" for c in metrics_order}),
            on="year",
            how="full",
            coalesce=True,
        ).sort("year")
    )

sb.glue("cagr", overall["CAGR"])
sb.glue("sharpe", overall["Sharpe"])
sb.glue("sortino", overall["Sortino"])
sb.glue("max_dd", overall["MaxDD"])

plt.figure(figsize=(11, 4))
plt.plot(pos["ts"], pos["equity"], label="TLT/TBF calendar")
plt.plot(
    bench["ts"],
    (1 + bench["ret"]).cum_prod(),
    alpha=0.7,
    label="TLT buy & hold",
)
plt.legend()
plt.title("Equity curve")
plt.grid(alpha=0.3)
plt.show()

# %%
