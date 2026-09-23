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
EOM_PERIOD = 5          # trading days in the month-end leg
BOM_PERIOD = 5          # trading days in the reversal leg
SELL_THRESH = -50       # bps: strong expected bond selling -> long SPY into month-end
BUY_THRESH = 50         # bps: strong expected bond buying  -> short TLT / long SPY after month-end
SPREAD_SCALE = 1.0      # 1.0 = 100% long SPY + 100% short TLT (article); 0.5 for a cash account
COST_BPS = 1.0          # per unit of turnover (one side)

# %%
# Calendar effect, leveraged variants
# -----------------------------------
# Same signal and leg structure as calendar-tlt.py, but each exposure is filled
# with one of the 1x/2x/3x bull & bear ETFs instead of cash-market SPY/TLT:
#
#   * month-end leg  (strong bond selling): long SPY  -> SPY / SSO 2x / UPRO 3x
#   * month-end leg  (no selling):          long TLT  -> TLT / UBT 2x / TMF 3x
#   * reversal leg   (strong buying):       short TLT -> short TLT on margin,
#                                                     or long TBF 1x / TBT 2x / TMV 3x
#
# Leveraged ETFs are backed by their *actual* daily returns (daily rebalancing
# and expense ratios included), not by a naive multiple of the underlying.
# The TBF row vs the "short TLT" row is the direct comparison the user asked
# for: TBF is a 1x inverse ETF (no margin, no borrow, ~0.9% ER), shorting TLT
# is the synthetic equivalent.
#
# Run from the calendar-tlt directory:  uv run calendar-leveraged.py
# Data: output.parquet (adjusted closes). Regenerate with:
#   uv run yf.py SPY SSO UPRO IEF TLT UBT TMF TBT TBF TMV --output calendar-tlt/output.parquet

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

month_sel = [
    pl.col("ts").dt.year().alias("year"),
    pl.col("ts").dt.month().alias("month"),
]

# ---------------------------------------------------------------- daily frame
# SPY trades every NYSE session since 1993 -> its dates are the trading calendar
# (verified against exchange_calendars XNYS over 2006-2026: exact match).
nyse_open = (
    pl.read_parquet("output.parquet")
    .filter(pl.col("symbol") == "SPY")
    .select("ts")
    .sort("ts")
)

SYMS = ["spy", "sso", "upro", "ief", "tlt", "ubt", "tmf", "tbt", "tbf", "tmv"]

daily = (
    pl.read_parquet("output.parquet")
    .with_columns(pl.col("symbol").str.to_lowercase())
    .pivot(on="symbol", index="ts", values="close")     # adjusted closes
    .sort("ts")
    .join(nyse_open, on="ts", how="inner")
    .drop_nulls(["spy", "ief", "tlt"])
    .with_columns(month_sel)
    .with_columns(
        pos=pl.int_range(1, pl.len() + 1).over(month_sel),
        days=pl.len().over(month_sel),
        # month-to-date returns measured from the previous month-end close
        spy_yday=pl.col("spy").shift(1),
        ief_yday=pl.col("ief").shift(1),
    )
    .with_columns(
        spy_mtd=pl.col("spy") / pl.col("spy_yday").first().over(month_sel) - 1,
        ief_mtd=pl.col("ief") / pl.col("ief_yday").first().over(month_sel) - 1,
    )
    .with_columns(
        neg_pos=pl.col("days") - pl.col("pos"),          # 0 = last day, 5 = sixth-last
        port=0.6 * (1 + pl.col("spy_mtd")) + 0.4 * (1 + pl.col("ief_mtd")),
    )
    .with_columns(
        # bond trade a 60/40 SPY/IEF investor needs, bps of portfolio value
        pressure_bps=(0.4 - 0.4 * (1 + pl.col("ief_mtd")) / pl.col("port")) * 1e4,
    )
    .with_columns(
        **{f"{s}_yday": pl.col(s).shift(1) for s in SYMS},
    )
    .with_columns(
        **{f"{s}_ret": pl.col(s) / pl.col(f"{s}_yday") - 1 for s in SYMS},
    )
    .with_columns(short_tlt_ret=-pl.col("tlt_ret"))
)

# drop the incomplete last month (its "month-end" is not a month-end)
last_year, last_month = daily.select(pl.col("year").last(), pl.col("month").last()).row(0)
daily = daily.filter(~((pl.col("year") == last_year) & (pl.col("month") == last_month)))
daily = daily.filter(pl.col("spy_ret").is_not_null() & pl.col("tlt_ret").is_not_null())

# --------------------------------------------------------------- signal frame
sig = (
    daily.group_by(month_sel, maintain_order=True)
    .agg(
        sig=pl.col("pressure_bps").filter(pl.col("neg_pos") == EOM_PERIOD).first(),
        month_id=(pl.col("year") * 12 + pl.col("month")).first(),
    )
    .with_columns(sig_prev=pl.col("sig").shift(1))
)
assert (sig["month_id"].diff().drop_nulls() == 1).all(), "gap in months: shift(1) misaligned"

# ------------------------------------------------------------------ leg frame
test = (
    daily.join(sig.drop("month_id"), on=["year", "month"], how="left")
    .with_columns(
        in_eom=pl.col("neg_pos") < EOM_PERIOD,   # returns of the last 5 days
        in_bom=pl.col("pos") <= BOM_PERIOD,      # returns of the first 5 days
    )
    .with_columns(
        leg=pl.when(pl.col("in_eom") & (pl.col("sig") <= SELL_THRESH)).then(pl.lit("eom_spy"))
             .when(pl.col("in_eom") & (pl.col("sig") > SELL_THRESH)).then(pl.lit("eom_tlt"))
             .when(pl.col("in_bom") & (pl.col("sig_prev") >= BUY_THRESH)).then(pl.lit("bom_spread"))
             .otherwise(pl.lit("flat")),
    )
    .with_columns(
        # rows where every instrument has a return -> the common test period
        is_common=pl.all_horizontal([pl.col(f"{s}_ret").is_not_null() for s in SYMS]),
    )
)

# first day on which the full instrument set is tradeable (UBT is the youngest)
common_start = test.filter(pl.col("is_common"))["ts"].first()
print(f"common period starts: {common_start}\n")

inception = (
    pl.read_parquet("output.parquet")
    .group_by("symbol")
    .agg(pl.col("ts").min().alias("inception"), pl.len().alias("rows"))
    .sort("inception")
)
with pl.Config(tbl_rows=-1, float_precision=2):
    print(inception)

# ------------------------------------------------------------------- variants
# instrument choice per exposure; leveraged ETFs come with their real daily
# returns (volatility drag + expense ratio), not synthetic multiples
SPY_SIDE = {"SPY": "spy_ret", "SSO2": "sso_ret", "UPRO3": "upro_ret"}
TLT_SIDE = {"TLT": "tlt_ret", "UBT2": "ubt_ret", "TMF3": "tmf_ret"}
SHORT_SIDE = {"shortTLT": "short_tlt_ret", "TBF1": "tbf_ret", "TBT2": "tbt_ret", "TMV3": "tmv_ret"}


def positions(leg_col: str, scale: float) -> dict[str, pl.Expr]:
    """Fraction of the portfolio held in each instrument, per leg.

    Convention: positions p_* earn the return of the row they are on, i.e.
    they are set at the previous close; cost is charged on the day they change.
    """
    return {
        "p_spy": pl.when(pl.col(leg_col) == "eom_spy").then(1.0)
                 .when(pl.col(leg_col) == "bom_spread").then(scale)
                 .otherwise(0.0),
        "p_tlt": pl.when(pl.col(leg_col) == "eom_tlt").then(1.0).otherwise(0.0),
        "p_short": pl.when(pl.col(leg_col) == "bom_spread").then(scale).otherwise(0.0),
    }


def backtest(d: pl.DataFrame, p_expr: dict[str, pl.Expr], ret_cols: dict[str, str]) -> pl.DataFrame:
    return (
        d.with_columns(**p_expr)
        .with_columns(
            turnover=sum(
                (pl.col(k) - pl.col(k).shift(1, fill_value=0.0)).abs() for k in p_expr
            ),
        )
        .with_columns(
            ret=sum(pl.col(k) * pl.col(c) for k, c in ret_cols.items())
            - pl.col("turnover") * COST_BPS * 1e-4,
        )
    )


def run_strategy(spy_col: str, tlt_col: str, short_col: str, scale: float) -> pl.DataFrame:
    return backtest(test, positions("leg", scale), {"p_spy": spy_col, "p_tlt": tlt_col, "p_short": short_col})


def run_baseline(tlt_col: str, short_col: str) -> pl.DataFrame:
    """Unconditional turn-of-month: long TLT into month-end, short TLT after."""
    return backtest(
        test,
        {
            "p_tlt": pl.when(pl.col("in_eom")).then(1.0).otherwise(0.0),
            "p_short": pl.when(pl.col("in_eom")).then(0.0)
                     .when(pl.col("in_bom")).then(1.0).otherwise(0.0),
        },
        {"p_tlt": tlt_col, "p_short": short_col},
    )


def metrics(r) -> dict:
    r = np.asarray(r, dtype=float)
    years = len(r) / 252
    ann_mean = r.mean() * 252
    vol = r.std(ddof=1) * np.sqrt(252)      # zeros on flat days are part of the series
    eq = np.cumprod(1 + r)
    cagr = eq[-1] ** (1 / years) - 1
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(252)
    return {
        "years": years,
        "ann_ret": ann_mean,
        "cagr": cagr,
        "vol": vol,
        "sharpe": ann_mean / vol if vol > 0 else np.nan,
        "sortino": ann_mean / downside if downside > 0 else np.nan,
        "maxdd": (eq / np.maximum.accumulate(eq) - 1).min(),
    }


def row_for(frame: pl.DataFrame, **label) -> dict:
    return {**label, "start": frame["ts"].first().date(), **metrics(frame["ret"].to_numpy())}


def variant_table(scale: float) -> pl.DataFrame:
    rows = []
    for sh_name, sh_col in SHORT_SIDE.items():
        for spy_name, spy_col in SPY_SIDE.items():
            for tlt_name, tlt_col in TLT_SIDE.items():
                v = run_strategy(spy_col, tlt_col, sh_col, scale)
                rows.append(row_for(v.filter(pl.col("ret").is_not_null()),
                                    short=sh_name, spy=spy_name, tlt_leg=tlt_name, span="own"))
                rows.append(row_for(v.filter(pl.col("is_common")),
                                    short=sh_name, spy=spy_name, tlt_leg=tlt_name, span="common"))
    return pl.DataFrame(rows)


def baseline_table() -> pl.DataFrame:
    rows = []
    for tlt_name, tlt_col in TLT_SIDE.items():
        for sh_name, sh_col in SHORT_SIDE.items():
            v = run_baseline(tlt_col, sh_col)
            rows.append(row_for(v.filter(pl.col("ret").is_not_null()),
                                tlt_leg=tlt_name, short=sh_name, span="own"))
            rows.append(row_for(v.filter(pl.col("is_common")),
                                tlt_leg=tlt_name, short=sh_name, span="common"))
    return pl.DataFrame(rows)


DISP = [
    pl.col("start").cast(pl.Utf8),
    pl.col("years").round(1),
    (pl.col("ann_ret") * 100).round(2).alias("ann_ret%"),
    (pl.col("cagr") * 100).round(2).alias("cagr%"),
    (pl.col("vol") * 100).round(2).alias("vol%"),
    pl.col("sharpe").round(2),
    pl.col("sortino").round(2),
    (pl.col("maxdd") * 100).round(2).alias("maxdd%"),
]

print("\n=== strategy variants, SPREAD_SCALE = 1.0 (article: 100% long SPY + 100% short TLT) ===")
m1 = variant_table(1.0)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=2):
    print(m1.select("span", "short", "spy", "tlt_leg", *DISP).sort("span", "short", "spy", "tlt_leg"))

print("\n=== strategy variants, SPREAD_SCALE = 0.5 (cash account) ===")
m05 = variant_table(0.5)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=2):
    print(m05.select("span", "short", "spy", "tlt_leg", *DISP).sort("span", "short", "spy", "tlt_leg"))

print("\n=== unconditional baseline: long TLT into month-end, short/inverse after ===")
base = baseline_table()
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=2):
    print(base.select("span", "tlt_leg", "short", *DISP).sort("span", "tlt_leg", "short"))

# ------------------------------------------------- TBF vs shorting TLT directly
# both over the span where TBF exists (2009-08 ->)
tbf_start = pl.datetime(2009, 8, 20)
tbf_cmp_rows = []
for scale in (1.0, 0.5):
    for short_name, short_col in (("short TLT", "short_tlt_ret"), ("TBF", "tbf_ret")):
        v = run_strategy("spy_ret", "tlt_ret", short_col, scale)
        f = v.filter((pl.col("ts") >= tbf_start) & pl.col("ret").is_not_null())
        tbf_cmp_rows.append(row_for(f, variant=f"SPY + {short_name}", scale=scale, span="TBF span"))
tbf_cmp = pl.DataFrame(tbf_cmp_rows)
print("\n=== TBF (1x inverse ETF) vs shorting TLT directly, SPY+TLT article setup, 2009-08 -> ===")
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=2):
    print(tbf_cmp.select("scale", "variant", *DISP))
d1 = (tbf_cmp.filter(pl.col("scale") == 1.0))
print(
    "direct minus TBF @1.0: "
    f"ann_ret {d1['ann_ret'][0]-d1['ann_ret'][1]:+.2%}, "
    f"sharpe {d1['sharpe'][0]-d1['sharpe'][1]:+.3f}, "
    f"maxdd {d1['maxdd'][0]-d1['maxdd'][1]:+.2%}"
)

# --------------------------------------------- leverage drag of the real ETFs
# realized daily-return multiple vs the naive k * underlying return
print("\n=== realized ETF daily returns vs naive k x underlying (annualized mean) ===")
drag_rows = []
for mult, etf, under in [
    (2.0, "sso", "spy"), (3.0, "upro", "spy"),
    (2.0, "ubt", "tlt"), (3.0, "tmf", "tlt"),
    (-1.0, "tbf", "tlt"), (-2.0, "tbt", "tlt"), (-3.0, "tmv", "tlt"),
]:
    f = test.drop_nulls([f"{etf}_ret", f"{under}_ret"])
    r_etf = f[f"{etf}_ret"].to_numpy()
    r_und = f[f"{under}_ret"].to_numpy()
    drag_rows.append({
        "etf": etf.upper(),
        "mult": mult,
        "start": f["ts"].first().date(),
        "etf_ann": r_etf.mean() * 252,
        "naive_ann": (mult * r_und).mean() * 252,
        "drag_ann": r_etf.mean() * 252 - mult * r_und.mean() * 252,
        "corr": np.corrcoef(r_etf, mult * r_und)[0, 1],
    })
with pl.Config(tbl_rows=-1, tbl_width_chars=160, float_precision=3):
    print(
        pl.DataFrame(drag_rows)
        .with_columns(
            (pl.col("etf_ann") * 100).round(2).alias("etf_ann%"),
            (pl.col("naive_ann") * 100).round(2).alias("naive_ann%"),
            (pl.col("drag_ann") * 100).round(2).alias("drag_ann%"),
        )
        .select("etf", "mult", "start", "etf_ann%", "naive_ann%", "drag_ann%", "corr")
    )

# ----------------------------------------------------------- reference prints
def stats(t: pl.DataFrame, name: str) -> None:
    m = metrics(t["ret"].to_numpy())
    print(f"{name}")
    for k, v in m.items():
        print(f"  {v*100:8.2f}" if k in ("ann_ret", "cagr", "vol", "maxdd") else f"  {v:8.2f}")


own_all = test.filter(pl.col("spy_ret").is_not_null() & pl.col("tlt_ret").is_not_null())
v_base_article = run_strategy("spy_ret", "tlt_ret", "short_tlt_ret", 1.0).filter(
    pl.col("ret").is_not_null()
)
stats(v_base_article, "reference: article variant SPY / short TLT @1.0 (own span, = scratchpad)")
stats(run_baseline("tlt_ret", "short_tlt_ret").filter(pl.col("ret").is_not_null()),
      "reference: baseline TLT turn-of-month (own span)")

# ------------------------------------------------------------------ equity plot
fig, ax = plt.subplots(figsize=(11, 5))
plot_set = [
    ("base 1x: SPY + short TLT", run_strategy("spy_ret", "tlt_ret", "short_tlt_ret", 1.0)),
    ("2x: SSO + TBT (eom TLT UBT)", run_strategy("sso_ret", "ubt_ret", "tbt_ret", 1.0)),
    ("3x: UPRO + TMV (eom TLT TMF)", run_strategy("upro_ret", "tmf_ret", "tmv_ret", 1.0)),
    ("1x via TBF: SPY + TBF", run_strategy("spy_ret", "tlt_ret", "tbf_ret", 1.0)),
    ("baseline: TLT / short TLT", run_baseline("tlt_ret", "short_tlt_ret")),
]
for label, v in plot_set:
    f = v.filter(pl.col("is_common"))
    ax.plot(f["ts"], (1 + f["ret"]).cum_prod(), label=label, lw=1.1)
ax.set_yscale("log")
ax.set_title("calendar effect, leveraged ETF variants (common period, SPREAD_SCALE=1.0)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("calendar-leveraged.png", dpi=130)
plt.show()
