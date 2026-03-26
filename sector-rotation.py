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
from typing import Callable

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
    .join(
        pl.read_csv(
            "sharadar-sfa/SHARADAR_TICKERS_2_e2ada4bebc2110c46304bab3f8d254dd.csv"
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
        )
        .drop_nulls(),
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
        roc=pl.col("netinc").sum() / pl.col("equity").sum(),
    )
    .sort(["etf", "date"])
    .with_columns(
        # 1d change of market breadth
        breadth=pl.col("breadth") - pl.col("breadth").shift(10).over("etf"),
        # 1q change of roc
        roc=pl.col("roc") - pl.col("roc").shift(4 * 20).over("etf"),
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
        nonfarm=pl.col("PAYEMS"),
        # housing starts
        house=pl.col("HOUST"),
    )
    # uv run yf.py DX-Y.NYB XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY ITA --output yf.parquet
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
                pl.when(pl.col("m8").is_nan())
                .then(
                    pl.when(pl.col("m7").is_nan())
                    .then(pl.col("m6"))
                    .otherwise(pl.col("m7"))
                )
                .otherwise("m8")
                - pl.col("m1")
            )
            / pl.col("m1"),
        ),
        on="ts",
        how="left",
    )
    .drop_nulls()
    .with_columns(
        date=pl.col("ts").dt.date(),
    )
    .with_columns(
        # regime z-score: z-score of raw yield over 3y window
        ycrv=zscore(3 * 252)(pl.col("ycrv")),
        # tips: 1m delta and 1y z-score
        tips=zscore(252)(pl.col("tips") - pl.col("tips").shift(20).over("date")),
        # hys: 3y percentile rank and maybe 1m spread change
        hys=(pl.col("hys") - pl.col("hys").rolling_min(756))
        / (pl.col("hys").rolling_max(756) - pl.col("hys").rolling_min(756)),
        # dxy: 3m roc
        dxy=pl.col("dxy") / pl.col("dxy").shift(60).over("date") - 1,
        # vix front and macro: as is
        # non-farm payrolls: yoy change, 3m mean
        nonfarm=(pl.col("nonfarm") / pl.col("nonfarm").shift(252).over("date") - 1)
        .rolling_mean(3 * 20)
        .over("date"),
        # housing starts: yoy change, 3m mean
        house=(pl.col("house") / pl.col("house").shift(252).over("date") - 1)
        .rolling_mean(3 * 20)
        .over("date"),
    )
    # inflation: 2nd derivative over 1m of the 1st derivative over 1y
    .with_columns(
        infrate=pl.col("inf") - pl.col("inf").shift(252).over("date"),
    )
    .with_columns(
        infaccel=pl.col("infrate") - pl.col("infrate").shift(20).over("date"),
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

print(macro.tail())

df = (
    macro.join(bottomup, on="date")
    # uv run yf.py DX-Y.NYB XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY ITA --output yf.parquet
    .join(
        pl.read_parquet("yf.parquet")
        .filter(
            pl.col("symbol").is_in(
                [
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
            )
        )
        .select(
            date=pl.col("ts").dt.date(),
            etf=pl.col("symbol"),
            close=pl.col("close"),
            volume=pl.col("volume"),
        )
        .sort("date"),
        on=["date", "etf"],
        how="inner",
    )
    # 1m,3m,6m,12m momentum
    .with_columns(
        **{
            f"mom{n}m": (pl.col("close") / pl.col("close").shift(n * 20) - 1).over(
                "etf"
            )
            for n in [1, 3, 6, 12]
        }
    )
    # distance from 20d, 50d, 200d SMA
    .with_columns(
        **{
            f"sma{n}m": (
                pl.col("close") / pl.col("close").rolling_mean(n * 20) - 1
            ).over("etf")
            for n in [1, 3, 6, 12]
        }
    )
    # target
    .with_columns(target=(pl.col("close") / pl.col("close").shift(-20).over("etf")) - 1)
)
# missing
# PMI new orders vs. inventories
# MOVE index
# analyst ratings or forward pe
# gold vs. copper
# CESI
# call and put IV
# ETF flows: https://data.nasdaq.com/databases/ETFF
# 12-1 mom
