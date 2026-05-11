import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

import polars as pl
import datetime as dt
import numpy as np

days_holding = 7
days_momentum = 30
days_skip = 7
max_positions = 20
fee_per_leg = 0.0005
min_market_cap_rank = 100
min_momentum_decile = 10
min_price = 0.0001
min_volume = 10_000
days_trend_filter = 2
min_frog_percentage = -100
year = 2026
benchmark = "bitcoin"

ANNUALIZATION_FACTOR = 365

ro.conversion.set_conversion(ro.default_converter + pandas2ri.converter)
crypto2 = importr("crypto2")
ro.r(
    """
flatten_for_python <- function(df) {
    df <- as.data.frame(lapply(df, function(col) {
        if (inherits(col, "POSIXlt")) return(as.character(col))
        if (is.list(col)) return(sapply(col, function(x) paste(x, collapse=",")))
        # Convert logical to integer so pyarrow doesn't choke on mixed types
        if (is.logical(col)) return(as.integer(col))
        col
    }), stringsAsFactors = FALSE)
    df
}
"""
)
flatten = ro.r["flatten_for_python"]

# Get full listings (R object used as coin_list later)
# Assign to R global env so we can filter it there
r_listings = crypto2.crypto_listings(which="latest", convert="USD")
ro.globalenv["r_listings"] = r_listings
pd_listings = ro.conversion.rpy2py(flatten(r_listings)).fillna(0).infer_objects()

df_listings = pl.from_pandas(pd_listings)

# Keep only top-N slugs in the R listings for crypto_history
selected_slugs = df_listings["slug"].to_list()
ro.globalenv["selected_slugs"] = ro.StrVector(selected_slugs)
coin_list = ro.r("subset(r_listings, slug %in% selected_slugs)")

now = dt.datetime.now() - dt.timedelta(days=(days_holding + days_momentum))
r_history = ro.conversion.rpy2py(
    flatten(
        crypto2.crypto_history(
            coin_list=coin_list,
            start_date=now.date().strftime("%Y%m%d"),
            requestLimit=200,
            limit=500,
            finalWait=True,
        )
    )
)
hpd = ro.conversion.rpy2py(flatten(r_history)).fillna(0).infer_objects()


df = (
    pl.from_pandas(hpd)
    .join(df_listings, on="slug")
    .select(
        ts=pl.col("time_open").dt.replace_time_zone("UTC"),
        ticker=pl.col("symbol").str.to_lowercase(),
        open=pl.col("open"),
        high=pl.col("high"),
        low=pl.col("low"),
        close=pl.col("close"),
        volume=pl.col("volume"),
        market_cap=pl.col("market_cap"),
        circulating_supply=pl.col("circulating_supply"),
        symbol=pl.col("slug"),
        listed=pl.col("date_added")
        .str.to_datetime(format="%Y-%m-%dT%H:%M:%S.000Z")
        .dt.replace_time_zone("UTC"),
        is_active=pl.col("is_active") > 0,
        market_pairs=pl.col("market_pair_count"),
    )
    .sort(["symbol", "ts"])
    .unique(["symbol", "ts"])  # bug?
    .filter((pl.col("market_pairs") > 0))
    .filter(pl.col("symbol").is_in(["terra-luna", "ftx-token"]).not_())
    .with_columns(
        fwd_ret_1d=(
            pl.col("close").shift(-1).over("symbol") / pl.col("close") - 1
        ).clip(-1, 5),
        fwddays=(pl.col("ts").shift(-1).over("symbol") - pl.col("ts")).dt.total_days(),
        mom=pl.col("close").shift(days_skip).over("symbol")
        / pl.col("close").shift(days_momentum).over("symbol")
        - 1,
        momdays=(
            pl.col("ts").shift(days_skip).over("symbol")
            - pl.col("ts").shift(days_momentum).over("symbol")
        ).dt.total_days(),
        mc_rank=pl.col("market_cap").rank(descending=True, method="ordinal").over("ts"),
        direction=pl.when(pl.col("open") < pl.col("close")).then(1).otherwise(-1),
        # trend=pl.col("close").ewm_mean(span=days_trend_filter, adjust=True).over('symbol'),
        trend=pl.col("close").rolling_mean(days_trend_filter).over("symbol"),
    )
    .filter(
        (pl.col("mom").is_not_null())
        & (pl.col("fwddays") == 1)
        & (pl.col("momdays") == days_momentum - days_skip)
    )
    .with_columns(
        mom_q=pl.col("mom")
        .qcut(10, labels=[str(i) for i in range(1, 11)], allow_duplicates=True)
        .over("ts"),
    )
    .with_columns(
        mask_mom=pl.when(
            (pl.col("mc_rank") >= min_market_cap_rank)
            & (pl.col("mom_q").cast(pl.Utf8).cast(pl.Int64) >= min_momentum_decile)
            # & (pl.col("close") >= min_price)
            # & (pl.col("mom") > 0)
            # & ((pl.col('direction').rolling_sum(days_momentum) / days_momentum) > (min_frog_percentage / 100.0))
            & (pl.col("close") > pl.col("trend"))
            & (pl.col("volume") > min_volume)
        )
        .then(pl.col("mom"))
        .otherwise(
            None
        )  # Coins outside top get None, making them invisible to the next ranking step
    )
    .with_columns(
        rank=pl.col("mask_mom").rank(descending=True, method="ordinal").over("ts")
    )
    # 7. SELECTION: Binary mask for the Top momentum coins (from the Top universe)
    .with_columns(
        is_selected=pl.when(pl.col("rank") <= max_positions).then(1).otherwise(0)
    )
    # 8. SUB-WEIGHTING
    .with_columns(sub_weight=pl.col("is_selected") / max_positions)
    # 9. J&T OVERLAP: Aggregate weight across 30 active sub-portfolios
    .with_columns(
        total_weight=(
            pl.col("sub_weight").rolling_sum(window_size=days_holding).over("symbol")
            / days_holding
        )
    )
)

now = df["ts"].max()
h = (
    df.filter((pl.col("ts") == now) & (pl.col("is_selected") == 1))
    .sort("rank")
    .head(max_positions)
)
print(h)
h.write_csv("cmc-portfolio.csv")
