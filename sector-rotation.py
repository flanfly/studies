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
from typing import Callable, Literal

sector_etfs = [
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

df = pl.read_csv(
    "sharadar-sfa/SHARADAR_SP500_2_a5d269df7633595315f85e40f3491992.csv",
    null_values=["N/A", "missing", "NaN"],
).with_columns(
    date=pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"),
)

start_date = df["date"].min()
end_date = df["date"].max()

# 1. Capture all membership events based on 'added', 'removed', and 'historical'
# 'current' is excluded here to be used only for validation.
events = (
    pl.concat(
        [
            # 'added' action: ticker is added (1), contraticker is removed (0)
            df.filter(pl.col("action") == "added").select(
                date=pl.col("date"), ticker=pl.col("ticker"), state=pl.lit(1)
            ),
            df.filter(pl.col("action") == "added")
            .select(date=pl.col("date"), ticker=pl.col("contraticker"), state=pl.lit(0))
            .filter(pl.col("ticker").is_not_null()),
            # 'historical' action: ticker is a member (1)
            df.filter(pl.col("action") == "historical").select(
                date=pl.col("date"), ticker=pl.col("ticker"), state=pl.lit(1)
            ),
            # Special case for non-paired removals (share classes etc)
            df.filter(pl.col("action") == "removed")
            .filter(pl.col("contraticker").is_null())
            .select(date=pl.col("date"), ticker=pl.col("ticker"), state=pl.lit(0)),
        ]
    )
    .unique()
    .sort(["ticker", "date", "state"], descending=[False, False, True])
)

# 2. Identify 'legacy' members who were in the index at start_date
# These are tickers whose first recorded event is a removal (state 0).
first_events = events.group_by("ticker").first()
legacy_events = first_events.filter(pl.col("state") == 0).select(
    date=pl.lit(start_date), ticker=pl.col("ticker"), state=pl.lit(1)
)

# 3. Final Event Consolidation
all_events = (
    pl.concat([events, legacy_events])
    .unique()
    .sort(["ticker", "date", "state"], descending=[False, False, True])
    .filter(
        (pl.col("state") != pl.col("state").shift(1).over("ticker")).fill_null(True)
    )
)

# 4. Expand to daily grid and forward-fill state
spx = (
    all_events["ticker"]
    .unique()
    .to_frame()
    .join(
        pl.date_range(start_date, end_date, interval="1d", eager=True)
        .alias("date")
        .to_frame(),
        how="cross",
    )
    .join(all_events, on=["date", "ticker"], how="left")
    .sort(["ticker", "date"])
    .with_columns(state=pl.col("state").forward_fill().over("ticker").fill_null(0))
    .filter(pl.col("state") == 1)
    .select(["date", "ticker"])
)

# 5. Consistency check against 'current'
actual_current = (
    spx.filter(pl.col("date") == end_date).select("ticker").to_series().sort()
)

expected_current = (
    df.filter(pl.col("action") == "current")
    .select("ticker")
    .unique()
    .to_series()
    .sort()
)

assert actual_current.equals(
    expected_current
), "Mismatch between calculated table and source 'current' list!"


# %%
bottomup = (
    pl.read_csv("sharadar-sfa/SHARADAR_SEP_2_da2386a176421f8ccbec6fabe5d11c0e.csv")
    .select(
        date=pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"),
        open=pl.col("open") * pl.col("close") / pl.col("closeadj"),
        high=pl.col("high") * pl.col("close") / pl.col("closeadj"),
        low=pl.col("low") * pl.col("close") / pl.col("closeadj"),
        close=pl.col("closeadj"),
        vol=pl.col("volume"),
        ticker=pl.col("ticker"),
    )
    .join(spx, on=["ticker", "date"], how="inner")
    .join(
        pl.read_csv(
            "sharadar-sfa/SHARADAR_DAILY_3_1eb3b706c850f0fffb2209ff783014bf.csv"
        ).select(
            date=pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"),
            mcap=pl.col("marketcap"),
            ticker=pl.col("ticker"),
        ),
        on=["ticker", "date"],
        how="left",
    )
    .sort("date")
    .join_asof(
        pl.read_parquet("fred.parquet")
        .sort("ts")
        .select(
            date=pl.col("ts").dt.date(),
            # 10y tips
            tips=pl.col("DFII10"),
        ),
        on="date",
        strategy="backward",
    )
    .join_asof(
        pl.read_csv("sharadar-sfa/SHARADAR_SF1_3_a158fbb8637a13efbab2ba75fc06dc74.csv")
        .select(
            date=pl.col("datekey").str.strptime(pl.Date, format="%Y-%m-%d"),
            divyield=pl.col("divyield"),
            netinc=pl.col("netinc"),
            equity=pl.col("equity"),
            ticker=pl.col("ticker"),
        )
        .sort("date"),
        on="date",
        by="ticker",
        strategy="backward",
    )
    .with_columns(
        divyield=pl.col("divyield") - pl.col("tips"),
    )
    .join(
        pl.read_csv(
            "sharadar-sfa/SHARADAR_TICKERS_2_e2ada4bebc2110c46304bab3f8d254dd.csv",
        )
        .unique("ticker")
        .select(
            ticker=pl.col("ticker"),
            etf=pl.col("sector").replace(
                {
                    "Healthcare": "XLV",
                    "Utilities": "XLU",
                    "Industrials": "XLI",
                    "Energy": "XLE",
                    "Consumer Defensive": "XLP",
                    "Basic Materials": "XLB",
                    "Technology": "XLK",
                    "Real Estate": "XLRE",
                    "Consumer Cyclical": "XLY",
                    "Financial Services": "XLF",
                    "Communication Services": "XLC",
                }
            ),
        ),
        on="ticker",
        how="inner",
    )
    .with_columns(weight=pl.col("mcap") / pl.col("mcap").sum().over(["date", "etf"]))
    .sort(["ticker", "date"])
    .with_columns(
        sma50d=(
            pl.col("close")
            .rolling_mean_by(
                by=pl.col("date"),
                window_size="52w",
            )
            .over("ticker")
        ),
        bull=(
            pl.when(
                (
                    pl.col("close")
                    / (
                        pl.col("close")
                        .rolling_max_by(
                            by=pl.col("date"),
                            window_size="52w",
                        )
                        .over("ticker")
                    )
                )
                >= 0.98
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        ),
        bear=(
            pl.when(
                (
                    pl.col("close")
                    / (
                        pl.col("close")
                        .rolling_max_by(
                            by=pl.col("date"),
                            window_size="52w",
                        )
                        .over("ticker")
                    )
                )
                <= 0.02
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        ),
    )
    .with_columns(
        breadth=(
            pl.when(pl.col("close") > pl.col("sma50d"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
        ),
    )
    .group_by(["date", "etf"])
    .agg(
        # ratio of stocks within 20% of 52w high vs. 20% of 52w low
        highlow=(pl.col("bull").sum() - pl.col("bear").sum()) / pl.len(),
        # ratio of stocks above 50d sma
        breadth=pl.col("breadth").sum() / pl.len(),
        # mcap-weighted mean dividend yield
        divyield=(pl.col("divyield") * pl.col("weight")).mean(),
        # mcap weighted-mean return on equity of the sector
        retcap=pl.col("netinc").sum() / pl.col("equity").sum(),
    )
    .sort(["etf", "date"])
    # yielding nulls if any join key is missing or data is sparse
    .with_columns(
        # 1d change of market breadth
        breadth=pl.col("breadth") - pl.col("breadth").shift(10).over("etf"),
        # 1q change of return on capital
        retcap=pl.col("retcap") - pl.col("retcap").shift(4 * 20).over("etf"),
    )
)


def zscore(window: int) -> Callable[[pl.Expr], pl.Expr]:
    def _zscore(val: pl.Expr) -> pl.Expr:
        return (val - val.rolling_mean(window)) / val.rolling_std(window)

    return _zscore


macro = (
    # uv run fred.py DGS10 DGS2 DFII10 BAMLH0A0HYM2 T5YIE PAYEMS HOUST --output fred.parquet
    pl.read_parquet("fred.parquet")
    .sort("ts")
    .select(
        ts=pl.col("ts"),
        # yield curve
        ycrv=pl.col("DGS10") - pl.col("DGS2"),
        # high-yield credit spread
        hys=pl.col("BAMLH0A0HYM2"),
        # 10y tips
        tips=pl.col("DFII10"),
        # 5y inflation rate
        inf=pl.col("T5YIE"),
        # non-farm payrolls
        nonfarm=pl.col("PAYEMS").forward_fill(),
        # housing starts
        house=pl.col("HOUST").forward_fill(),
    )
    # uv run yf.py DX-Y.NYB --output yf.parquet
    .join(
        pl.read_parquet("yf.parquet")
        .sort("ts")
        .filter(pl.col("symbol") == "DX-Y.NYB")
        .pivot(on="symbol", index="ts", values=["close"])
        .select(
            ts=pl.col("ts"),
            # DXY
            dxy=pl.col("DX-Y.NYB"),
        ),
        on="ts",
        how="left",
    )
    # VIX & term structure
    # uv run vix.py --output vix-term.parquet
    .join(
        pl.read_parquet("vix-term.parquet")
        .sort("ts")
        .select(
            ts=pl.col("ts"),
            # VIX front spread
            vxfront=(pl.col("m2") - pl.col("m1")) / pl.col("m1"),
            # VIX macro slope
            vxmacro=(
                pl.when(pl.col("m8").is_not_nan() & pl.col("m8").is_not_null())
                .then(pl.col("m8"))
                .otherwise(
                    pl.when(pl.col("m7").is_not_nan() & pl.col("m7").is_not_null())
                    .then(pl.col("m7"))
                    .otherwise(pl.col("m6"))
                )
                - pl.col("m1")
            )
            / pl.col("m1"),
        ),
        on="ts",
        how="left",
    )
    .with_columns(
        date=pl.col("ts").dt.date(),
    )
    .sort("date")
    .drop_nulls()
    # normalize time-series wise
    .with_columns(
        # regime z-score: z-score of raw yield over 3y window
        ycrv=zscore(3 * 252)(pl.col("ycrv")),
        # tips: 1m delta and 1y z-score
        tips=zscore(252)(pl.col("tips") - pl.col("tips").shift(20)),
        # hys: 3y percentile rank and maybe 1m spread change
        hys=(pl.col("hys") - pl.col("hys").rolling_min(756))
        / (pl.col("hys").rolling_max(756) - pl.col("hys").rolling_min(756)),
        # dxy: 3m roc
        dxy=pl.col("dxy") / pl.col("dxy").shift(60) - 1,
        # vix front and macro: as is
        # non-farm payrolls: yoy change, 3m mean
        nonfarm=(pl.col("nonfarm") / pl.col("nonfarm").shift(252) - 1).rolling_mean(
            3 * 20
        ),
        # housing starts: yoy change, 3m mean
        house=(pl.col("house") / pl.col("house").shift(252) - 1).rolling_mean(3 * 20),
        # inflation: 2nd derivative over 1m of the 1st derivative over 1y
        infrate=pl.col("inf") - pl.col("inf").shift(252),
    )
    .with_columns(
        infaccel=pl.col("infrate") - pl.col("infrate").shift(20),
    )
    .select(
        [
            "date",
            "ycrv",
            "hys",
            "tips",
            "dxy",
            "vxfront",
            "vxmacro",
            "infrate",
            "infaccel",
            "nonfarm",
            "house",
        ]
    )
)


def rank(expr: pl.Expr) -> pl.Expr:
    return expr.rank(method="average") / expr.count()


# rolling OLS of y on x over the specified window
def rolling_ols(
    window: int, feature: Literal["alpha", "beta", "epsilon"]
) -> Callable[[pl.Expr, pl.Expr], pl.Expr]:
    def _rolling_ols(x: pl.Expr, y: pl.Expr) -> pl.Expr:
        x_mean = x.rolling_mean(window)
        y_mean = y.rolling_mean(window)
        cov = (x - x_mean) * (y - y_mean)
        var = (x - x_mean) ** 2
        alpha = y_mean - (cov.rolling_sum(window) / var.rolling_sum(window)) * x_mean
        beta = cov.rolling_sum(window) / var.rolling_sum(window)

        if feature == "alpha":
            return alpha
        elif feature == "beta":
            return beta
        elif feature == "epsilon":
            y_pred = beta * x + alpha
            epsilon = y - y_pred
            return epsilon
        else:
            raise ValueError(f"Invalid feature: {feature}")

    return _rolling_ols


def rs_vol(
    high: pl.Expr, low: pl.Expr, close: pl.Expr, open: pl.Expr, window: int
) -> pl.Expr:
    return (
        (high / close).log() * (high / open).log()
        + (low / close).log() * (low / open).log()
    ).rolling_mean(window)


# yang-zhang variance estimation parameters
yz_k = 0.34
yz_win = 25

# prediction horizon
fwd_win = 20

df = (
    bottomup.join(
        # uv run yf.py XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY --output yf.parquet
        pl.read_parquet("yf.parquet")
        .filter(pl.col("symbol").is_in(sector_etfs))
        .select(
            date=pl.col("ts").dt.date(),
            etf=pl.col("symbol"),
            open=pl.col("open"),
            high=pl.col("high"),
            low=pl.col("low"),
            close=pl.col("close"),
            vol=pl.col("volume"),
        )
        .sort("date"),
        on=["date", "etf"],
        how="inner",
    )
    # yz-variance estimation
    .with_columns(
        o=pl.col("open").log() - pl.col("close").log(),
        u=pl.col("high").log() - pl.col("open").log(),
        d=pl.col("low").log() - pl.col("open").log(),
        c=pl.col("close").log() - pl.col("open").log(),
    )
    .with_columns(
        rs=pl.col("u") * (pl.col("u") - pl.col("c"))
        + pl.col("d") * (pl.col("d") - pl.col("c"))
    )
    .with_columns(
        var=pl.col("o").rolling_var(yz_win)
        + yz_k * pl.col("c").rolling_var(yz_win)
        + ((1 - yz_k) * pl.col("rs").rolling_mean(yz_win))
    )
    # market return
    .join(
        pl.read_parquet("yf.parquet")
        .filter(pl.col("symbol").is_in(["SPY", "DX-Y.NYB"]))
        .sort("ts")
        .pivot(on="symbol", index="ts", values=["close"])
        .select(
            date=pl.col("ts").dt.date(),
            spy=pl.col("SPY").log() - pl.col("SPY").log().shift(1),
            dxy=pl.col("DX-Y.NYB").log() - pl.col("DX-Y.NYB").log().shift(1),
        ),
        on=["date"],
        how="left",
    )
    .sort("date")
    # tips
    .join_asof(
        pl.read_parquet("fred.parquet")
        .sort("ts")
        .select(
            date=pl.col("ts").dt.date(),
            # 10y tips
            rateroc=pl.col("DFII10") - pl.col("DFII10").shift(1),
        ),
        on="date",
        strategy="backward",
    )
    .with_columns(
        # daily return of the sector ETF
        logret=(pl.col("close").log() - pl.col("close").shift(1).log()).over("etf"),
    )
    .with_columns(
        # idosyncratic return vs market
        alpha=rolling_ols(60, "epsilon")(pl.col("spy"), pl.col("logret")).over("etf"),
        # rate sensitivity
        rate_sense=rolling_ols(60, "beta")(
            pl.col("rateroc"), pl.col("logret").exp() - 1
        ).over("etf"),
        # fx sensitivity
        fx_sense=rolling_ols(60, "beta")(pl.col("dxy"), pl.col("logret")).over("etf"),
    )
    .sort(["etf", "date"])
    .with_columns(
        # 1m,3m,6m,12m momentum
        **{
            f"mom{n}m": (pl.col("logret").rolling_sum(n * 20) - 1).over("etf")
            for n in [1, 3, 6, 12]
        },
        # 1m,3m,6m,12m alpha vs market over
        **{
            f"alpha{n}m": pl.col("alpha").rolling_sum(n * 20).over("etf")
            for n in [1, 3, 6, 12]
        },
        # distance from 20d, 50d, 200d SMA
        **{
            f"sma{n}m": (
                pl.col("close") / pl.col("close").rolling_mean(n * 20) - 1
            ).over("etf")
            for n in [1, 3, 6, 12]
        },
        # volume divergence: 5d vs 3m
        voldiv=(pl.col("vol").rolling_mean(5) - pl.col("vol").rolling_mean(60)).over(
            "etf"
        ),
        # 3m information ratio
        ir=(pl.col("logret").rolling_sum(60) / pl.col("logret").rolling_std(60)).over(
            "etf"
        ),
        # forward idiosyncratic return (target variable)
        fwdret=pl.col("alpha").shift(-21).rolling_sum(20).over("etf"),
    )
    .sort(["etf", "date"])
    # normalization: time-series z-score
    .with_columns(
        divyield=zscore(3 * 252)(pl.col("divyield")).over("etf"),
        retcap=zscore(3 * 252)(pl.col("retcap")).over("etf"),
    )
    # normalization: cross-sectional rank
    .with_columns(
        # highlow: cross-sectional rank
        highlow=rank(pl.col("highlow")).over("date"),
        # breadth (10d change): cross-sectional z-score
        breadth=rank(pl.col("breadth")).over("date"),
        # divyield: 3y ts z-score, then cross-sectional
        divyield=rank(pl.col("divyield")).over("date"),
        # retcap (1q change): 3y ts z-score, then cross-sectional
        retcap=rank(pl.col("retcap")).over("date"),
        # momentum: 1m,3m,6m,12m cross-sectional z-score
        **{f"mom{n}m": rank(pl.col(f"mom{n}m")).over("date") for n in [1, 3, 6, 12]},
        # sma distance: 1m,3m,6m,12m cross-sectional z-score
        **{f"sma{n}m": rank(pl.col(f"sma{n}m")).over("date") for n in [1, 3, 6, 12]},
        voldiv=rank(pl.col("voldiv")).over("date"),
        ir=rank(pl.col("ir")).over("date"),
        var=rank(pl.col("var")).over("date"),
        **{f"alpha{n}m": pl.col("alpha").over("date") for n in [1, 3, 6, 12]},
        rate_sense_rank=rank(pl.col("rate_sense")).over("date"),
        fx_sense_rank=rank(pl.col("fx_sense")).over("date"),
    )
    .pivot(
        index="date",
        on="etf",
        values=[
            # ratio of stocks within 20% of 52w high vs. 20% of 52w low, cross sectional rank
            "highlow",
            # 10d change of ratio of stocks above 50d sma, cross sectional z-score
            "breadth",
            "divyield",
            "retcap",
            *[f"mom{n}m" for n in [1, 3, 6, 12]],
            *[f"sma{n}m" for n in [1, 3, 6, 12]],
            *[f"alpha{n}m" for n in [1, 3, 6, 12]],
            "fwdret",
            "voldiv",
            "ir",
            "var",
            "rate_sense",
            "rate_sense_rank",
            "fx_sense",
            "fx_sense_rank",
        ],
    )
    # join macro features
    .join_asof(
        macro.sort("date"),
        on="date",
        strategy="backward",
    )
)

df.write_parquet("features.parquet")

# missing
# PMI new orders vs. inventories
# MOVE index
# analyst ratings or forward pe
# gold vs. copper
# CESI
# call and put IV
# ETF flows: https://data.nasdaq.com/databases/ETFF

# 12-1 mom
# volume and volatility
# idosyncratic volatility

# %%
import polars as pl
import itertools as it

micro_features = [
    "highlow",
    "breadth",
    "divyield",
    "retcap",
    *[f"mom{n}m" for n in [1, 3, 6, 12]],
    *[f"sma{n}m" for n in [1, 3, 6, 12]],
    "fwdret",
    # "voldiv",
    "ir",
    "var",
    "alpha",
    "rate",
    "raterank",
    "fx",
    "fxrank",
]

macro_features = [
    "ycrv",
    "hys",
    "tips",
    "dxy",
    "vxfront",
    "vxmacro",
    "infrate",
    "infaccel",
    "nonfarm",
    "house",
]

corr = (
    pl.read_parquet("features.parquet")
    .select(
        **{
            f"{p[0]}_{p[1]}": pl.corr(f"fwdret_{p[1]}", p[0], method="spearman")
            for p in it.product(macro_features, sector_etfs)
        }
    )
    .unpivot(variable_name="pair", value_name="ic")
    .with_columns(
        pl.col("pair").str.split_exact("_", 1).struct.rename_fields(["feature", "etf"])
    )
    .unnest("pair")
    .pivot(on="etf", index="feature", values="ic")
)

import seaborn as sns
import matplotlib.pyplot as plt

# 1. Convert Polars to Pandas
# (Seaborn works best with Pandas or NumPy arrays)
pandas_matrix = corr.to_pandas().set_index("feature")

# 2. Plot the Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(
    pandas_matrix,
    mask=pandas_matrix.abs() <= 0.02,
    annot=True,
    fmt=".2f",
    cmap="RdBu",
    linewidths=0.5,
    center=0,
)

plt.savefig("macro-correlations.png")
plt.show()
