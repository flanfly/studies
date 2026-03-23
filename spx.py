import polars as pl

df = pl.read_parquet("sharadar-sp500.parquet")

start_date = df["date"].min()
end_date = df["date"].max()

events = (
    pl.concat(
        [
            df.filter(pl.col("action") == "added").select(
                date=pl.col("date"), ticker=pl.col("ticker"), state=pl.lit(1)
            ),
            df.filter(pl.col("action") == "removed")
            .select(date=pl.col("date"), ticker=pl.col("contraticker"), state=pl.lit(1))
            .filter(pl.col("ticker").is_not_null()),
            df.filter(pl.col("action") == "removed").select(
                date=pl.col("date"), ticker=pl.col("ticker"), state=pl.lit(0)
            ),
            df.filter(pl.col("action") == "added")
            .select(date=pl.col("date"), ticker=pl.col("contraticker"), state=pl.lit(0))
            .filter(pl.col("ticker").is_not_null()),
            # Current Members:
            # If a ticker is 'current', it is IN at the end_date.
            # If it's 'current' but never 'added', it was also IN at the start_date.
            df.filter(pl.col("action") == "current").select(
                date=pl.col("date"), ticker=pl.col("ticker"), state=pl.lit(1)
            ),
        ]
    )
    .unique()
    .sort(["ticker", "date"])
)

added_tickers = (
    pl.concat(
        [
            df.filter(pl.col("action") == "added").select("ticker"),
            df.filter(pl.col("action") == "removed")
            .select(ticker=pl.col("contraticker"))
            .filter(pl.col("ticker").is_not_null()),
        ]
    )
    .unique()["ticker"]
    .to_list()
)

legacy_events = (
    df.filter(
        (pl.col("action").is_in(["current", "removed", "historical"]))
        & (~pl.col("ticker").is_in(added_tickers))
    )
    .select(date=pl.lit(start_date), ticker=pl.col("ticker"), state=pl.lit(1))
    .unique()
)

all_events = (
    pl.concat([events, legacy_events])
    .unique()
    .sort(["ticker", "date"])
    # only keep records where state actually changes (e.g. 0->1 or 1->0)
    .filter(
        (pl.col("state") != pl.col("state").shift(1).over("ticker")).fill_null(True)
    )
)

spx = (
    # date x ticker grid
    all_events["ticker"]
    .unique()
    .to_frame()
    .join(
        pl.date_range(start_date, end_date, interval="1d", eager=True)
        .alias("date")
        .to_frame(),
        how="cross",
    )
    # join with known states
    .join(all_events, on=["date", "ticker"], how="left")
    .sort(["ticker", "date"])
    .with_columns(
        # Forward fill the state within each ticker group
        state=pl.col("state")
        .forward_fill()
        .over("ticker")
        .fill_null(0)
    )
    # keep only rows for dates where the ticker is in the spx
    .filter(pl.col("state") == 1)
    .select(["date", "ticker"])
)

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
