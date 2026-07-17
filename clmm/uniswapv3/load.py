import polars as pl
import polars_ols as pls

import numpy as np

import asyncio
from httpx import AsyncClient
from sgqlc.endpoint.base import BaseEndpoint

from typing import Optional, Dict, Any, Literal, Tuple
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property

from uniswapv3 import emulator
import uniswapv3.math as v3math

import logging as l

__all__ = ["Pool", "Position", "from_binance", "from_ethereum"]


@dataclass(frozen=True)
class Position:
    owner: str
    liquidity: int
    tick_lower: int
    tick_upper: int


@dataclass()
class Tick:
    liquidity_net: int
    liquidity_gross: int
    fee_growth_outside_x128: Tuple[int, int]


@dataclass(frozen=True)
class Pool:
    name0: str
    name1: str
    symbol0: str
    symbol1: str
    d0: int
    d1: int
    sqrt_price_x96: int
    tick: int
    ticks: dict[int, Tick]
    liquidity: int
    fee_pips: int

    @cached_property
    def tick_spacing(self) -> int:
        match self.fee_pips:
            case 100:
                return 1
            case 500:
                return 10
            case 3000:
                return 60
            case 10000:
                return 200
            case _:
                assert False

    @cached_property
    def protocol_fraction(self) -> float:
        match self.fee_pips:
            case 100:
                return 0.25
            case 500:
                return 0.25
            case 3000:
                return 0.1667
            case 10000:
                return 0.1667
            case _:
                assert False

    @cached_property
    def max_liquidity_per_tick(self):
        min_tick = (v3math.MIN_TICK / self.tick_spacing) * self.tick_spacing
        max_tick = (v3math.MAX_TICK / self.tick_spacing) * self.tick_spacing
        numTicks = ((max_tick - min_tick) // self.tick_spacing) + 1
        return ((1 << 128) - 1) / numTicks


def from_binance(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.select(
            ts=(pl.col("time") * 1000).cast(pl.Datetime),
            price=pl.col("price"),
            qty=pl.col("qty"),
        )
        # resample to 1s bars
        .group_by_dynamic("ts", every="1s")
        .agg(
            pl.col("price").mean(),
            pl.col("qty").sum(),
        )
        .upsample(time_column="ts", every="1s")
        .with_columns(
            pl.col("price").forward_fill(),
            pl.col("qty").fill_null(0.0),
        )
        .with_columns(
            diff=pl.col("price").shift(-1) - pl.col("price"),
        )
        .drop_nulls()
        # estimate OU process variables
        .with_columns(
            coefs=pl.col("diff").least_squares.rolling_ols(
                "price",
                window_size=1800,
                min_periods=1800,
                add_intercept=True,
                mode="coefficients",
            ),
        )
        .unnest("coefs", separator="_")
        .rename({"coefs_const": "alpha", "coefs_price": "beta"})
        .with_columns(
            ts=pl.col("ts"),
            theta=-pl.col("beta").fill_null(0.0),
            mu=pl.when(pl.col("beta") < 0)
            .then(-pl.col("alpha") / pl.col("beta"))
            .otherwise(pl.col("price"))
            .fill_null(pl.col("price")),
            sigma=(
                pl.col("diff") - (pl.col("alpha") + pl.col("price") * pl.col("beta"))
            ).fill_null(0.0),
            vol=(
                (pl.col("price").shift(1) / pl.col("price"))
                .log()
                .rolling_std(300, min_samples=300)
                .clip(0, 0.1)
            ),
        )
        .select(
            pl.col("ts"),
            pl.col("price"),
            pl.col("qty"),
            pl.col("mu").shift(1),
            pl.col("theta").shift(1).clip(0, 1),
            (
                pl.col("sigma").rolling_std(1800, min_samples=1).shift(1)
                / pl.col("price")
            )
            .clip(0, 0.1)
            .alias("sigma"),
            pl.col("vol"),
        )
        # cut off OLS/rolling std warmup
        .drop_nulls()
    )


async def gql_get_pool_tokens(
    ep: BaseEndpoint, block_number: int, contract: str
) -> dict[str, Any]:
    query = """
    query GetPoolTokens($poolId: ID!, $bn: Int!) {
      pool(id: $poolId, block: { number: $bn }) {
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
    data = await ep(query, {"poolId": contract, "bn": block_number})

    return data.get("data", {}).get("pool", {})


async def gql_get_ticks(ep: BaseEndpoint, bn: int, contract: str) -> dict[int, Tick]:
    query = """
    query Ticks($poolId: ID!, $bn: Int!, $lastTick: BigInt!) {
      ticks(
        where: { pool: $poolId, tickIdx_gte: $lastTick, liquidityGross_gt: 0 }
        block: { number: $bn }
        first: 1000, orderBy: tickIdx, orderDirection: asc
      ) {
        tickIdx
        liquidityNet
        liquidityGross
      }
    }
    """

    ret: dict[int, Tick] = {}
    last_tick = emulator.Emulator.MIN_TICK

    while last_tick <= emulator.Emulator.MAX_TICK:
        vars = {
            "poolId": contract,
            "bn": bn,
            "lastTick": last_tick,
        }
        page = await ep(query, vars)

        ticks = page.get("data", {}).get("ticks", [])
        if len(ticks) == 0:
            break

        ret |= {
            int(t["tickIdx"]): Tick(
                liquidity_net=int(t["liquidityNet"]),
                liquidity_gross=int(t["liquidityGross"]),
                fee_growth_outside_x128=(
                    0,
                    0,
                ),
            )
            for t in ticks
        }

        last_tick = int(max(ticks, key=lambda t: int(t["tickIdx"]))["tickIdx"]) + 1

    return ret


async def pool_meta(ep: BaseEndpoint, bn: int, contract: str) -> Pool:
    meta, ticks = await asyncio.gather(
        gql_get_pool_tokens(ep, bn, contract),
        gql_get_ticks(ep, bn, contract),
    )

    pool = Pool(
        name0=meta["token0"]["name"],
        name1=meta["token1"]["name"],
        symbol0=meta["token0"]["symbol"],
        symbol1=meta["token1"]["symbol"],
        d0=int(meta["token0"]["decimals"]),
        d1=int(meta["token1"]["decimals"]),
        sqrt_price_x96=int(meta["sqrtPrice"]),
        liquidity=int(meta["liquidity"]),
        tick=int(meta["tick"]),
        ticks=ticks,
        fee_pips=int(meta["feeTier"]),
    )
    return pool


async def from_ethereum(
    df: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Pool]:
    from sgqlc.endpoint.httpx import HTTPXEndpoint
    from os import getenv
    from eth_utils import event_signature_to_log_topic

    swap = event_signature_to_log_topic(
        "Swap(address,address,int256,int256,uint160,uint128,int24)"
    )
    mint = event_signature_to_log_topic(
        "Mint(address,address,int24,int24,uint128,uint256,uint256)"
    )
    burn = event_signature_to_log_topic(
        "Burn(address,int24,int24,uint128,uint256,uint256)"
    )

    topic0_to_event = {
        swap: "swap",
        mint: "mint",
        burn: "burn",
    }

    def be_int(b: bytes, signed: bool) -> int:
        if b is None:
            return 0
        return int.from_bytes(b, "big", signed=signed)

    swaps = (
        df.filter(pl.col("topic0") == swap)
        .with_columns(
            **{
                f"slot{i}": pl.col("data").bin.slice(offset=32 * i, length=32)
                for i in range(5)
            }
        )
        .with_columns(
            amount0=pl.col("slot0").map_elements(
                lambda b: int.from_bytes(b, signed=True, byteorder="big"),
                return_dtype=pl.Object,
            ),
            amount1=pl.col("slot1").map_elements(
                lambda b: int.from_bytes(b, signed=True, byteorder="big"),
                return_dtype=pl.Object,
            ),
            quote_volume=pl.col("slot0").map_elements(
                lambda b: float(abs(int.from_bytes(b, signed=True, byteorder="big"))),
                return_dtype=pl.Float64,
            ),
            sqrt_price_x96=pl.col("slot2").map_elements(
                lambda b: int.from_bytes(b, byteorder="big"),
                return_dtype=pl.Object,
            ),
            liquidity=pl.col("slot3").map_elements(
                lambda b: int.from_bytes(b, byteorder="big"),
                return_dtype=pl.Object,
            ),
            tick=pl.col("slot4").map_elements(
                lambda b: int.from_bytes(b, signed=True, byteorder="big"),
                return_dtype=pl.Int32,
            ),
            ord=pl.col("transaction_index") * 100 + pl.col("log_index"),
        )
        .with_columns(
            price=1.0
            / pow(
                pl.col("sqrt_price_x96").map_elements(
                    lambda b: float(b), return_dtype=pl.Float64
                )
                / pow(2, 96),
                2,
            ),
        )
        .select(
            [
                "ts",
                "block_number",
                "transaction_index",
                "ord",
                "log_index",
                "sqrt_price_x96",
                "price",
                "amount0",
                "amount1",
                "quote_volume",
                "liquidity",
                "tick",
            ]
        )
        .sort(["block_number", "ord"])
    )

    mint = (
        df.filter(pl.col("topic0") == mint)
        .with_columns(
            **{
                f"slot{i}": pl.col("data").bin.slice(offset=32 * i, length=32)
                for i in range(5)
            }
        )
        .with_columns(
            tick_lower=pl.col("topic2").map_elements(
                lambda b: int.from_bytes(b, signed=True, byteorder="big"),
                return_dtype=pl.Int32,
            ),
            tick_upper=pl.col("topic3").map_elements(
                lambda b: int.from_bytes(b, signed=True, byteorder="big"),
                return_dtype=pl.Int32,
            ),
            liquidity=pl.col("slot1").map_elements(
                lambda b: int.from_bytes(b, byteorder="big"),
                return_dtype=pl.Object,
            ),
            amount0=pl.col("slot2").map_elements(
                lambda b: int.from_bytes(b, byteorder="big"),
                return_dtype=pl.Object,
            ),
            amount1=pl.col("slot3").map_elements(
                lambda b: int.from_bytes(b, byteorder="big"),
                return_dtype=pl.Object,
            ),
            ord=pl.col("transaction_index") * 100 + pl.col("log_index"),
        )
    )
    burn = (
        df.filter(pl.col("topic0") == burn)
        .with_columns(
            **{
                f"slot{i}": pl.col("data").bin.slice(offset=32 * i, length=32)
                for i in range(5)
            }
        )
        .with_columns(
            tick_lower=pl.col("topic2").map_elements(
                lambda b: int.from_bytes(b, signed=True, byteorder="big"),
                return_dtype=pl.Int32,
            ),
            tick_upper=pl.col("topic3").map_elements(
                lambda b: int.from_bytes(b, signed=True, byteorder="big"),
                return_dtype=pl.Int32,
            ),
            liquidity=pl.col("slot0").map_elements(
                lambda b: -int.from_bytes(b, byteorder="big"),
                return_dtype=pl.Object,
            ),
            amount0=pl.col("slot1").map_elements(
                lambda b: int.from_bytes(b, byteorder="big"),
                return_dtype=pl.Object,
            ),
            amount1=pl.col("slot2").map_elements(
                lambda b: int.from_bytes(b, byteorder="big"),
                return_dtype=pl.Object,
            ),
            ord=pl.col("transaction_index") * 100 + pl.col("log_index"),
        )
    )

    liq = (
        pl.concat([mint, burn])
        .select(
            [
                "ts",
                "block_number",
                "transaction_index",
                "ord",
                "log_index",
                "tick_lower",
                "tick_upper",
                "liquidity",
                "amount0",
                "amount1",
            ]
        )
        .sort(["block_number", "ord"])
    )

    params = (
        swaps
        # resample to one block bars (~12s)
        .group_by("block_number")
        .agg(
            pl.col("price").mean(),
            pl.col("quote_volume").sum(),
        )
        # forward fill
        .upsample(time_column="block_number", every="1i")
        .with_columns(
            [pl.col("price").forward_fill(), pl.col("quote_volume").fill_null(0)]
        )
        .with_columns(
            diff=pl.col("price").shift(-1) - pl.col("price"),
        )
        .drop_nulls()
        # estimate OU process variables
        .with_columns(
            coefs=pl.col("diff").least_squares.rolling_ols(
                "price",
                window_size=1800 // 12,
                min_periods=1800 // 12,
                add_intercept=True,
                mode="coefficients",
            ),
        )
        .unnest("coefs", separator="_")
        .rename({"coefs_const": "alpha", "coefs_price": "beta"})
        .with_columns(
            ts=pl.col("block_number"),
            theta=-pl.col("beta").fill_null(0.0),
            mu=pl.when(pl.col("beta") < 0)
            .then(-pl.col("alpha") / pl.col("beta"))
            .otherwise(pl.col("price"))
            .fill_null(pl.col("price")),
            sigma=(
                pl.col("diff") - (pl.col("alpha") + pl.col("price") * pl.col("beta"))
            ).fill_null(0.0),
            vol=(
                (pl.col("price").shift(1) / pl.col("price"))
                .log()
                .rolling_std(300 // 12, min_samples=300 // 12)
                .clip(0, 0.1 * 12)
            ),
        )
        .select(
            pl.col("ts"),
            pl.col("block_number"),
            pl.col("price"),
            pl.col("quote_volume").alias("qty"),
            pl.col("mu").shift(1),
            pl.col("theta").shift(1).clip(0, 1),
            (
                pl.col("sigma").rolling_std(1800 // 12, min_samples=1).shift(1)
                / pl.col("price")
            )
            .clip(0, 0.1 * 12)
            .alias("sigma"),
            pl.col("vol"),
        )
        # cut off OLS/rolling std warmup
        .drop_nulls()
    )

    swaps = swaps.join(params.select("block_number"), on="block_number")
    liq = liq.join(params.select("block_number"), on="block_number")

    bn = params["block_number"].min()
    pool = df["address"].unique().to_list()

    assert len(pool) == 1
    pool = f"0x{pool[0].hex()}"

    headers = {"Authorization": f"""bearer {getenv('GRAPH_API_KEY')}"""}
    url = "https://gateway.thegraph.com/api/[api-key]/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
    endpoint = HTTPXEndpoint(url, headers, client=AsyncClient())

    meta = await pool_meta(endpoint, bn - 1, pool)

    return (
        swaps,
        liq,
        params,
        meta,
    )
