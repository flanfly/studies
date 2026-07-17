import polars as pl
import polars_ols as pls
import polars_u256_plugin as plu

import gymnasium as gym
import numpy as np
import torch

import asyncio
from httpx import AsyncClient
from sgqlc.endpoint.base import BaseEndpoint
from httpx import AsyncClient

from typing import Optional, Dict, Any, Literal, Tuple
from collections.abc import Iterable
from dataclasses import dataclass
from bisect import bisect_right
from functools import lru_cache
from copy import deepcopy

from uniswapv3.load import Tick

import sys
from dotenv import load_dotenv
import logging as l
from tqdm.contrib.logging import logging_redirect_tqdm

import uniswapv3.math as v3math

__all__ = ["Emulator", "Tick", "LiquidityPosition"]


def close(a, b, rtol=np.finfo(np.float32).eps) -> bool:
    return abs(a - b) <= max(abs(a), abs(b), 1) * rtol


@dataclass()
class LiquidityPosition:
    liquidity: int
    feeGrowthInside0Last: float
    feeGrowthInside1Last: float
    tokensOwed0: float = 0.0
    tokensOwed1: float = 0.0


@dataclass(frozen=True)
class EmulatorSwapState:
    sqrt_price_x96: int
    liquidity: int
    fee_growth_global_x128: list[int]
    protocol_fees: list[float]
    tick: int
    swap_df: pl.DataFrame
    seen_block: dict[int, int]
    fee_growth_outside_x128: dict[int, Tuple[int, int]]


@dataclass(frozen=True)
class EmulatorLiquidityPosition:
    tick_lower: int
    tick_upper: int
    liquidity: int


class Emulator:
    MIN_TICK = -887272
    MAX_TICK = 887272

    def __init__(
        self,
        sqrt_price_x96: int,
        tick: int,
        liquidity: int,
        ticks: dict[int, Tick],
        tick_spacing: int,
        fee_pips: int,
        protocol_fraction: float,
        max_liquidity_per_tick: int,
    ):
        assert tick_spacing > 0
        assert fee_pips >= 0 and fee_pips < 100_000
        assert protocol_fraction >= 0 and protocol_fraction <= 1
        assert tick_spacing >= 0

        # constant
        self.tick_spacing = tick_spacing
        self.fee_pips = fee_pips
        self.protocol_fraction = protocol_fraction
        self.max_liquidity_per_tick = max_liquidity_per_tick

        # updated by mint/burn
        self.ticks = ticks
        self.tick_keys = sorted(self.ticks.keys())

        # updated by swap
        self.sqrt_price_x96 = sqrt_price_x96
        self.liquidity = liquidity
        self.tick = tick

        self.protocol_fees = [0.0, 0.0]
        self.fee_growth_global_x128 = [0, 0]  # per L

        # set_position
        self.my_position: None | EmulatorLiquidityPosition = None

        # metrics
        self.seen_block: dict[int, int] = {}
        self.swap_df = pl.DataFrame(
            schema={
                "bn": pl.Int64,
                "ord": pl.Int32,
                "token0": pl.Float64,
                "token1": pl.Float64,
                "price": pl.Float64,
                "fee0": pl.Float64,
                "fee1": pl.Float64,
            }
        )

        # Initialize positions and tick fee growth
        self.positions: dict[tuple[int, int, bool], LiquidityPosition] = {}

        import math

        current_tick = int(math.floor(self.tick + 1e-9))
        for tick_idx, tick in self.ticks.items():
            if current_tick >= tick_idx:
                tick.fee_growth_outside_x128 = deepcopy(self.fee_growth_global_x128)
            else:
                tick.fee_growth_outside_x128 = [0, 0]

    def store_swap_state(self) -> EmulatorSwapState:
        return EmulatorSwapState(
            sqrt_price_x96=self.sqrt_price_x96,
            liquidity=self.liquidity,
            fee_growth_global_x128=deepcopy(self.fee_growth_global_x128),
            protocol_fees=deepcopy(self.protocol_fees),
            tick=self.tick,
            swap_df=self.swap_df,
            seen_block=deepcopy(self.seen_block),
            fee_growth_outside_x128={
                k: deepcopy(t.fee_growth_outside_x128) for k, t in self.ticks.items()
            },
        )

    def load_swap_state(self, state: EmulatorSwapState):
        self.sqrt_price_x96 = state.sqrt_price_x96
        self.liquidity = state.liquidity
        self.fee_growth_global_x128 = deepcopy(state.fee_growth_global_x128)
        self.protocol_fees = deepcopy(state.protocol_fees)
        self.tick = state.tick
        self.swap_df = state.swap_df
        self.seen_block = deepcopy(state.seen_block)
        for t in self.ticks.keys():
            self.ticks[t].fee_growth_outside_x128 = deepcopy(
                state.fee_growth_outside_x128[t]
            )

    def swap(
        self,
        bn: int,
        token_in: int,
        amount_specified: int,
        limit_sqrt_x96: int | None = None,
    ) -> Tuple[int, int, int]:
        # print(
        #    f"swap: amount_specified={amount_specified}, token_in={token_in}, limit_sqrt_x96={limit_sqrt_x96}"
        # )

        zero_for_one = token_in == 0
        exact_input = amount_specified > 0

        if limit_sqrt_x96 is None and zero_for_one:
            limit_sqrt_x96 = v3math.MIN_SQRT_RATIO + 1
        elif limit_sqrt_x96 is None and not zero_for_one:
            limit_sqrt_x96 = v3math.MAX_SQRT_RATIO - 1

        # XXX
        cache_fee_protocol = 0

        state_amount_specified_remaining = amount_specified
        state_amount_calculated = 0
        state_sqrt_price_x96 = self.sqrt_price_x96
        state_tick = self.tick
        state_fee_growth_global_x128 = self.fee_growth_global_x128[token_in]
        state_protocol_fee = 0
        state_liquidity = self.liquidity

        total_fee = 0
        my_fee = 0

        while (
            state_amount_specified_remaining != 0
            and state_sqrt_price_x96 != limit_sqrt_x96
        ):
            step_sqrt_price_start_x96 = state_sqrt_price_x96

            i = bisect_right(self.tick_keys, state_tick)
            if zero_for_one:
                step_tick_next = self.tick_keys[i - 1] if i > 0 else self.MIN_TICK
            else:
                step_tick_next = (
                    self.tick_keys[i] if i < len(self.tick_keys) else self.MAX_TICK
                )
            step_initialized = (
                step_tick_next != self.MIN_TICK and step_tick_next != self.MAX_TICK
            )

            step_sqrt_price_next_x96 = v3math.get_sqrt_ratio_at_tick(step_tick_next)

            if zero_for_one:
                next_tick_past_limit = step_sqrt_price_next_x96 < limit_sqrt_x96
            else:
                next_tick_past_limit = step_sqrt_price_next_x96 > limit_sqrt_x96
            if next_tick_past_limit:
                sqrt_price_target_x96 = limit_sqrt_x96
            else:
                sqrt_price_target_x96 = step_sqrt_price_next_x96

            state_sqrt_price_x96, step_amount_in, step_amount_out, step_fee_amount = (
                v3math.compute_swap_step(
                    state_sqrt_price_x96,
                    sqrt_price_target_x96,
                    state_liquidity,
                    state_amount_specified_remaining,
                    self.fee_pips,
                )
            )

            total_fee += step_fee_amount

            if exact_input:
                state_amount_specified_remaining -= step_amount_in + step_fee_amount
                state_amount_calculated = state_amount_calculated - step_amount_out
            else:
                state_amount_specified_remaining += step_amount_out
                state_amount_calculated = (
                    state_amount_calculated + step_amount_in + step_fee_amount
                )

            if cache_fee_protocol > 0:
                delta = step_fee_amount // cache_fee_protocol
                step_fee_amount -= delta
                state_protocol_fee += delta

            p = self.my_position
            if (not p is None) and p.tick_lower <= state_tick < p.tick_upper:
                assert state_liquidity >= p.liquidity
                my_fee += v3math.mul_div(step_fee_amount, p.liquidity, state_liquidity)

            if state_liquidity > 0:
                state_fee_growth_global_x128 = v3math.wrap256(
                    state_fee_growth_global_x128
                    + v3math.mul_div(step_fee_amount, v3math.Q128, state_liquidity)
                )

            if state_sqrt_price_x96 == step_sqrt_price_next_x96:
                if step_initialized:
                    # oracle omitted

                    # Ticks.cross
                    tick = self.ticks[step_tick_next]
                    if zero_for_one:
                        fo0 = v3math.wrap256(
                            state_fee_growth_global_x128
                            - tick.fee_growth_outside_x128[0]
                        )
                        fo1 = v3math.wrap256(
                            self.fee_growth_global_x128[1]
                            - tick.fee_growth_outside_x128[1]
                        )
                    else:
                        fo0 = v3math.wrap256(
                            self.fee_growth_global_x128[0]
                            - tick.fee_growth_outside_x128[0]
                        )
                        fo1 = v3math.wrap256(
                            state_fee_growth_global_x128
                            - tick.fee_growth_outside_x128[1]
                        )

                    tick.fee_growth_outside_x128 = (
                        fo0,
                        fo1,
                    )
                    liquidity_net = tick.liquidity_net

                    if zero_for_one:
                        liquidity_net = -liquidity_net
                    state_liquidity += liquidity_net
                    assert state_liquidity >= 0

                if zero_for_one:
                    state_tick = step_tick_next - 1
                else:
                    state_tick = step_tick_next

            elif state_sqrt_price_x96 != step_sqrt_price_start_x96:
                state_tick = v3math.get_tick_at_sqrt_ratio(state_sqrt_price_x96)

        self.sqrt_price_x96 = state_sqrt_price_x96
        self.liquidity = state_liquidity
        self.fee_growth_global_x128[token_in] = state_fee_growth_global_x128
        self.protocol_fees[token_in] += state_protocol_fee
        self.tick = state_tick

        ord_idx = self.seen_block.get(bn, 0)
        self.seen_block[bn] = ord_idx + 1

        if zero_for_one == exact_input:
            token0 = amount_specified - state_amount_specified_remaining
            token1 = state_amount_calculated
        else:
            token0 = state_amount_calculated
            token1 = amount_specified - state_amount_specified_remaining

        if zero_for_one:
            my_fee0 = my_fee
            my_fee1 = 0
        else:
            my_fee0 = 0
            my_fee1 = my_fee

        self.swap_df = self.swap_df.vstack(
            pl.DataFrame(
                {
                    "bn": [bn],
                    "ord": [ord_idx],
                    "token0": [token0],
                    "token1": [token1],
                    "price": [pow(float(self.sqrt_price_x96) / 2**96, 2)],
                    "fee0": [my_fee0],
                    "fee1": [my_fee1],
                },
                schema=self.swap_df.schema,
            )
        )

        return (
            token0,
            token1,
            state_amount_specified_remaining,
        )

    def modify_liquidity(
        self, tick_lower: int, tick_upper: int, delta: int
    ) -> Tuple[int, int]:
        assert Emulator.MIN_TICK <= tick_lower < tick_upper <= Emulator.MAX_TICK
        assert (
            tick_lower % self.tick_spacing == 0 and tick_upper % self.tick_spacing == 0
        )

        # print(
        #    f"modify_liquidity: tick_lower={tick_lower}, tick_upper={tick_upper}, delta={delta}"
        # )

        if delta == 0:
            return (
                0,
                0,
            )

        flip_lower = self._update_tick(
            tick_lower,
            delta,
            False,
        )
        flip_upper = self._update_tick(
            tick_upper,
            delta,
            True,
        )

        if flip_lower or flip_upper:
            for tick in self.tick_keys:
                if self.ticks[tick].liquidity_gross == 0:
                    del self.ticks[tick]
            self.tick_keys = sorted(self.ticks.keys())

        if self.tick < tick_lower:
            amount0 = int(np.sign(delta)) * v3math.get_amount0_delta(
                v3math.get_sqrt_ratio_at_tick(tick_lower),
                v3math.get_sqrt_ratio_at_tick(tick_upper),
                abs(delta),
                delta > 0,
            )
            amount1 = 0
        elif self.tick < tick_upper:
            amount0 = int(np.sign(delta)) * v3math.get_amount0_delta(
                self.sqrt_price_x96,
                v3math.get_sqrt_ratio_at_tick(tick_upper),
                abs(delta),
                delta > 0,
            )
            amount1 = int(np.sign(delta)) * v3math.get_amount1_delta(
                v3math.get_sqrt_ratio_at_tick(tick_lower),
                self.sqrt_price_x96,
                abs(delta),
                delta > 0,
            )
            self.liquidity += delta
            assert self.liquidity >= 0
        else:
            amount0 = 0
            amount1 = int(np.sign(delta)) * v3math.get_amount1_delta(
                v3math.get_sqrt_ratio_at_tick(tick_lower),
                v3math.get_sqrt_ratio_at_tick(tick_upper),
                abs(delta),
                delta > 0,
            )

        return (
            amount0,
            amount1,
        )

    def set_position(self, tick_lower: int, tick_upper: int, liquidity: int):
        assert tick_lower % self.tick_spacing == 0 == tick_upper % self.tick_spacing

        if self.my_position is not None:
            self.modify_liquidity(
                self.my_position.tick_lower,
                self.my_position.tick_upper,
                -self.my_position.liquidity,
            )
            self.my_position = None

        if liquidity == 0:
            return

        self.modify_liquidity(tick_lower, tick_upper, liquidity)
        self.my_position = EmulatorLiquidityPosition(
            tick_lower=tick_lower,
            tick_upper=tick_upper,
            liquidity=liquidity,
        )

    def position_amounts(self) -> tuple[int, int] | None:
        p = self.my_position

        if p is None:
            return None

        sa = v3math.get_sqrt_ratio_at_tick(p.tick_lower)
        sb = v3math.get_sqrt_ratio_at_tick(p.tick_upper)
        s = min(max(self.sqrt_price_x96, sa), sb)

        return (
            v3math.get_amount0_delta(s, sb, p.liquidity, False),
            v3math.get_amount1_delta(sa, s, p.liquidity, False),
        )

    def _update_tick(
        self,
        tick: int,
        delta: int,
        is_upper: bool,
    ) -> bool:
        t = self.ticks.get(tick)

        if t is None:
            t = Tick(
                liquidity_net=0,
                liquidity_gross=0,
                fee_growth_outside_x128=(0, 0),
            )
            if tick <= self.tick:
                t.fee_growth_outside_x128 = deepcopy(self.fee_growth_global_x128)

        liq_before = t.liquidity_gross
        t.liquidity_gross = liq_before + delta
        t.liquidity_net += -delta if is_upper else delta
        assert 0 <= t.liquidity_gross <= self.max_liquidity_per_tick

        self.ticks[tick] = t
        self.tick_keys = sorted(self.ticks.keys())
        return (t.liquidity_gross == 0) != (liq_before == 0)
