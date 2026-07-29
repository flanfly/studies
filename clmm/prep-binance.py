import polars as pl
import polars_ols as pls
from os import getenv
from sgqlc.endpoint.httpx import HTTPXEndpoint

epoch_s_ms_threshold = 10_000_000_000
epoch_ms_us_threshold = 20_000_000_000_000
ou_window = 1800  # maybe try 7200, 14400
vol_window = 300
fee_pips = 500  # 0.05%
d0 = 6  # usdc
d1 = 18  # eth
# pool is usdc/weth, price is weth/usdc
# futures are eth/usdc
future_is_reciprocal_of_pool = True  # future base asset == pool quote asset?


def to_sqrt_x96(val: pl.Float64, reciprocal: bool) -> pl.String:
    from math import sqrt

    if reciprocal:
        s = sqrt(val * 10 ** (d0 - d1)) * (2**96)
    else:
        s = sqrt(1.0 / val * 10 ** (d1 - d0)) * (2**96)
    return str(int(s))


prices = (
    pl.scan_csv(
        "eth-usdc-spot.csv",
        has_header=False,
        new_columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "base_volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
        schema_overrides={
            "open_time": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "base_volume": pl.Float64,
            "close_time": pl.Int64,
            "quote_volume": pl.Float64,
            "trades": pl.Int64,
            "taker_buy_base_volume": pl.Float64,
            "taker_buy_quote_volume": pl.Float64,
            "ignore": pl.Int64,
        },
    )
    .select(
        ts=pl.when(pl.col("open_time") > epoch_ms_us_threshold)
        .then(pl.from_epoch("open_time", time_unit="us"))
        .when(pl.col("open_time") > epoch_s_ms_threshold)
        .then(pl.from_epoch("open_time", time_unit="ms"))
        .otherwise(pl.from_epoch("open_time", time_unit="s"))
        .dt.replace_time_zone("UTC")
        .dt.cast_time_unit("us"),
        # price is sqrt(y/x) in uniswap
        open=(
            1.0 / pl.col("open") if not future_is_reciprocal_of_pool else pl.col("open")
        ),
        sqrt_price_x96=pl.col("open").map_elements(
            lambda v: to_sqrt_x96(v, not future_is_reciprocal_of_pool),
            return_dtype=pl.String,
        ),
    )
    .sort("ts")
    .with_columns(diff=pl.col("open") - pl.col("open").shift(1))
    # estimate OU process variables
    .with_columns(
        coefs=pl.col("diff").least_squares.rolling_ols(
            pl.col("open").shift(1),
            window_size=ou_window,
            min_periods=ou_window,
            add_intercept=True,
            mode="coefficients",
        ),
    )
    .unnest("coefs", separator="_")
    .rename({"coefs_const": "alpha", "coefs_open": "beta"})
    .with_columns(
        theta=-pl.col("beta").fill_null(0.0),
        mu=pl.when(pl.col("beta") < 0)
        .then(-pl.col("alpha") / pl.col("beta"))
        .otherwise(pl.col("open"))
        .fill_null(pl.col("open")),
        err=(
            pl.col("diff")
            - (pl.col("alpha") + pl.col("open").shift(1) * pl.col("beta"))
        ),
        vol=(
            (pl.col("open").shift(1) / pl.col("open"))
            .log()
            .rolling_std(vol_window, min_samples=vol_window)
        ),
    )
    .with_columns(
        sigma=(
            pl.col("err").rolling_std(ou_window, min_samples=ou_window) / pl.col("open")
        ),
    )
    .select("ts", "open", "sqrt_price_x96", "mu", "theta", "sigma", "vol")
    # cut off OLS/rolling std warmup
    .drop_nulls()
)


prices.sink_parquet("prices.parquet")


from eth_utils import event_signature_to_log_topic

swap = event_signature_to_log_topic(
    "Swap(address,address,int256,int256,uint160,uint128,int24)"
)


def clip_and_sum_int(series: pl.Series) -> int:
    return sum(max(0, val) for val in series if val is not None)


def min_int(series: pl.Series) -> int:
    return min(val for val in series if val is not None)


def median_int(series: pl.Series) -> int:
    vals = sorted([val for val in series if val is not None])
    if not vals:
        return 0
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) // 2


def fee_growth(series: pl.Struct) -> pl.Float64:
    pairs = [s.split(":", 1) for s in series if s is not None]
    return sum(
        max(0, int(val)) * fee_pips / 10**6 / int(liq) if liq > 0 else 0
        for val, liq in pairs
    )


pool = (
    pl.scan_parquet("usdc-eth-005/*logs*.parquet")
    .filter(pl.col("topic0") == swap)["address"]
    .unique()
    .list()
)
assert len(pool) == 1
pool = f"0x{pool[0].hex()}"

from httpx import Client
from sgqlc.endpoint.httpx import HTTPXEndpoint

print(pool)

headers = {"Authorization": f"""bearer {getenv('GRAPH_API_KEY')}"""}
url = "https://gateway.thegraph.com/api/[api-key]/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
endpoint = HTTPXEndpoint(url, headers, timeout=30, client=Client(timeout=30))

query = """
query GetPoolTokens($poolId: ID!, $bn: Int!) {
  pool(id: $poolId) {
    id
    sqrtPrice
    tick
    liquidity
    feeTier
    token0 {
      name
      symbol
      decimals
    }
    token1 {
      name
      symbol
      decimals
    }
  }
}
"""
data = endpoint(query, {"poolId": pool})
print(data.get("data", {}).get("pool", {}))


swaps = (
    pl.scan_parquet("usdc-eth-005/*logs*.parquet")
    .filter(pl.col("topic0") == swap)
    .join(
        pl.scan_parquet("usdc-eth-005/*blocks*.parquet").select(
            pl.col("block_number"),
            ts=pl.from_epoch(pl.col("timestamp"), time_unit="s")
            .dt.replace_time_zone("UTC")
            .dt.cast_time_unit("us"),
        ),
        on=["block_number"],
    )
    .sort(["block_number", "transaction_index", "log_index"])
    .with_columns(
        **{
            f"slot{i}": pl.col("data").bin.slice(offset=32 * i, length=32)
            for i in range(5)
        }
    )
    .with_columns(
        amount0=pl.col("slot0").map_elements(
            lambda b: str(int.from_bytes(b, signed=True, byteorder="big")),
            return_dtype=pl.String,
        ),
        amount1=pl.col("slot1").map_elements(
            lambda b: str(int.from_bytes(b, signed=True, byteorder="big")),
            return_dtype=pl.String,
        ),
        liquidity=pl.col("slot3").map_elements(
            lambda b: str(int.from_bytes(b, byteorder="big")),
            return_dtype=pl.String,
        ),
        tick=pl.col("slot4").map_elements(
            lambda b: int.from_bytes(b, signed=True, byteorder="big"),
            return_dtype=pl.Int32,
        ),
        ord=pl.col("transaction_index") * 100 + pl.col("log_index"),
    )
    .sort("block_number", "ord")
    .group_by("block_number")
    .agg(
        ts=pl.col("ts").last(),
        # already includes decimals
        fee_growth0=(pl.col("amount0") + ":" + pl.col("liquidity"))
        .implode()
        .map_elements(fee_growth, return_dtype=pl.Float64),
        # already includes decimals
        fee_growth1=(pl.col("amount1") + ":" + pl.col("liquidity"))
        .implode()
        .map_elements(fee_growth, return_dtype=pl.Float64),
        min_liquidity=pl.col("liquidity")
        .implode()
        .map_elements(lambda s: str(min_int(s)), return_dtype=pl.String),
        tick=pl.col("tick").median(),
    )
    .sort("ts")
)

swaps.sink_parquet("swaps.parquet")

prices.join(
    swaps,
    on="ts",
    how="left",
).sink_parquet("data.parquet")

print("done")
