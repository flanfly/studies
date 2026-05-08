"""Download historical crypto price and market cap data from CoinMarketCap
via the crypto2 R package and rpy2.

Fetches OHLCV + market cap for all coins listed on CMC before a cutoff date.
"""

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Setup – enable rpy2 <-> pandas conversion globally
# ---------------------------------------------------------------------------
ro.conversion.set_conversion(ro.default_converter + pandas2ri.converter)

crypto2 = importr("crypto2")

# Pre-define an R helper to flatten tibbles with POSIXlt / list-columns
# so rpy2 can handle them on the return journey.
ro.r(
    """
flatten_for_python <- function(df) {
    df <- as.data.frame(lapply(df, function(col) {
        if (inherits(col, "POSIXlt")) return(as.character(col))
        if (is.list(col)) return(sapply(col, function(x) paste(x, collapse=",")))
        col
    }), stringsAsFactors = FALSE)
    df
}
"""
)
flatten = ro.r["flatten_for_python"]

# ---------------------------------------------------------------------------
# 2. Fetch listings & filter
# ---------------------------------------------------------------------------
print("1. Fetching CMC listings...")
listings_df = ro.conversion.rpy2py(crypto2.crypto_listings())
print(f"   Got {len(listings_df)} coins total.")

# Filter: keep coins added to CMC before the cutoff date
CUTOFF = "2017-01-01"
listings_df["date_added"] = pd.to_datetime(listings_df["date_added"])
target_coins = listings_df[listings_df["date_added"] < CUTOFF].copy()
print(f"   Coins listed before {CUTOFF}: {len(target_coins)}")

# crypto_history needs at least id, name, symbol, slug
coin_cols = ["id", "name", "symbol", "slug"]

# ---------------------------------------------------------------------------
# 3. Download historical data
# ---------------------------------------------------------------------------
START_DATE = "20200101"
print(f"2. Downloading historical data from {START_DATE}...")
print("   (This will take a while – CMC API rate limits apply.)")

history_r = crypto2.crypto_history(
    target_coins[coin_cols],
    start_date=START_DATE,
    requestLimit=200,
    sleep=0.5,
    wait=30,
    finalWait=True,
)

# Convert the returned tibble (contains POSIXlt columns) via our R flattener
history_df = ro.conversion.rpy2py(flatten(history_r))
print(
    f"   Downloaded {len(history_df):,} rows across {history_df['symbol'].nunique()} coins."
)

# ---------------------------------------------------------------------------
# 4. Clean up dtypes
# ---------------------------------------------------------------------------
for col in ["timestamp", "time_open", "time_close", "time_high", "time_low"]:
    if col in history_df.columns:
        history_df[col] = pd.to_datetime(history_df[col], utc=True)

for col in ["open", "high", "low", "close", "volume", "market_cap"]:
    if col in history_df.columns:
        history_df[col] = pd.to_numeric(history_df[col], errors="coerce")

if "circulating_supply" in history_df.columns:
    history_df["circulating_supply"] = (
        pd.to_numeric(history_df["circulating_supply"], errors="coerce")
        .round()
        .astype("Int64")
    )

# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
OUTPUT = "cmc_historical_data_2020.csv"
print(f"\n3. Saving to {OUTPUT}...")
history_df.to_csv(OUTPUT, index=False)
print(f"   Done! {len(history_df):,} rows written to {OUTPUT}")

print(f"\nSample:")
print(
    history_df[["symbol", "timestamp", "close", "market_cap"]]
    .head(10)
    .to_string(index=False)
)
