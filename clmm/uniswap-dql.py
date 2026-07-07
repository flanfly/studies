import polars as pl
import polars_ols as pls
import polars_u256_plugin as plu

import gymnasium as gym
import numpy as np
import torch

import asyncio
from httpx import AsyncClient


from typing import Optional, Dict, Any, Literal, Tuple
from collections.abc import Iterable
from dataclasses import dataclass

from sgqlc.endpoint.base import BaseEndpoint
from httpx import AsyncClient

import sys
from dotenv import load_dotenv
import logging as l
from tqdm.contrib.logging import logging_redirect_tqdm

from bisect import bisect_right

load_dotenv()

l.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=l.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)


@dataclass()
class Tick:
    liquidityNet: int
    liquidityGross: int
    feeGrowthOutside0: float = 0.0
    feeGrowthOutside1: float = 0.0


@dataclass()
class LiquidityPosition:
    liquidity: int
    feeGrowthInside0Last: float
    feeGrowthInside1Last: float
    tokensOwed0: float = 0.0
    tokensOwed1: float = 0.0


class V3Pool:
    MIN_TICK = -887272
    MAX_TICK = 887272

    def __init__(
        self,
        sqrtp: float,
        liquidity: int,
        ticks: dict[int, Tick],
        tick_spacing: int,
        fee_tier: float,
        protocol_fraction: float,
    ):
        assert tick_spacing > 0
        assert fee_tier >= 0 and fee_tier < 1
        assert protocol_fraction >= 0 and protocol_fraction <= 1
        assert tick_spacing >= 0

        # constant
        self.tick_spacing = tick_spacing
        self.fee_tier = fee_tier
        self.protocol_fraction = protocol_fraction

        # updated by mint/burn
        self.ticks = ticks
        self.tick_keys = sorted(self.ticks.keys())

        # updated by swap
        self.sqrtp = sqrtp
        self.liquidity = liquidity
        self.tick = self._sqrtp_to_tick(sqrtp)

        self.seen_block: dict[int, int] = {}

        self.fees = pl.DataFrame(
            schema={
                "bn": pl.Int64,
                "ord": pl.Int32,
                "fee0": pl.Float64,
                "fee1": pl.Float64,
                "dao_fee0": pl.Float64,
                "dao_fee1": pl.Float64,
                "my_fee0": pl.Float64,
                "my_fee1": pl.Float64,
                "sqrtp": pl.Float64,
            }
        )

        self.swaps = pl.DataFrame(
            schema={
                "bn": pl.Int64,
                "ord": pl.Int32,
                "token0": pl.Float64,
                "token1": pl.Float64,
                "sqrtp": pl.Float64,
            }
        )

        self.protocol_fees = [0.0, 0.0]
        self.fee_growth_global = [0.0, 0.0]  # per L

        # Initialize positions and tick fee growth
        self.positions: dict[tuple[int, int, bool], LiquidityPosition] = {}

        import math

        current_tick = int(math.floor(self.tick + 1e-9))
        for tick_idx, tick in self.ticks.items():
            if current_tick >= tick_idx:
                tick.feeGrowthOutside0 = self.fee_growth_global[0]
                tick.feeGrowthOutside1 = self.fee_growth_global[1]
            else:
                tick.feeGrowthOutside0 = 0.0
                tick.feeGrowthOutside1 = 0.0

    def swap(self, bn: int, token_in: int, amount_in: float):
        # Track my fees before swap
        my_fees_before = self.my_fees

        total_swap_fee0 = 0.0
        total_swap_fee1 = 0.0
        total_dao_fee0 = 0.0
        total_dao_fee1 = 0.0

        remaining, amount_out = float(amount_in), 0.0
        going_up = token_in == 1
        while remaining > 0:
            # 1. boundary: nearest initialized tick in trade direction,
            #    from EXPLICIT tick state (not re-derived from sqrtp)
            i = bisect_right(self.tick_keys, self.tick)
            if going_up:
                boundary = (
                    self.tick_keys[i] if i < len(self.tick_keys) else self.MAX_TICK
                )
            else:
                boundary = self.tick_keys[i - 1] if i > 0 else self.MIN_TICK
            s_target = self._tick_to_sqrtp(boundary)
            s = self.sqrtp

            if self.liquidity > 0:
                # 2. net budget (ratio-scaled, per iteration)
                net = remaining * (1 - self.fee_tier)
                # 3.+4. candidate destination, clamped at the boundary
                if going_up:
                    goal = s + net / self.liquidity
                    s_next = min(goal, s_target)
                else:
                    goal = (self.liquidity * s) / (self.liquidity + net * s)
                    s_next = max(goal, s_target)
                # 5. amounts off the interval [s, s_next]
                if going_up:
                    step_in = self.liquidity * (s_next - s)  # dy
                    step_out = self.liquidity * (s_next - s) / (s * s_next)  # dx
                else:
                    step_in = self.liquidity * (s - s_next) / (s * s_next)  # dx
                    step_out = self.liquidity * (s - s_next)  # dy
                # 6. fee, by branch
                hit_boundary = s_next == s_target
                if hit_boundary and s_next != goal:
                    fee_amt = step_in * self.fee_tier / (1 - self.fee_tier)  # gross-up
                else:
                    fee_amt = remaining - step_in  # remainder
                # 7. bookkeeping
                dao = fee_amt * self.protocol_fraction
                self.protocol_fees[token_in] += dao
                self.fee_growth_global[token_in] += (fee_amt - dao) / self.liquidity
                remaining -= step_in + fee_amt
                amount_out += step_out
                self.sqrtp = s_next

                if token_in == 0:
                    total_swap_fee0 += fee_amt
                    total_dao_fee0 += dao
                else:
                    total_swap_fee1 += fee_amt
                    total_dao_fee1 += dao
            else:
                # empty segment: price gaps across, no trade, no fee
                self.sqrtp = s_next = s_target
                hit_boundary = True

            # 8. cross or finish
            if hit_boundary:
                if boundary in self.ticks:
                    # Update tick fee growth outside when crossed
                    t_crossed = self.ticks[boundary]
                    t_crossed.feeGrowthOutside0 = (
                        self.fee_growth_global[0] - t_crossed.feeGrowthOutside0
                    )
                    t_crossed.feeGrowthOutside1 = (
                        self.fee_growth_global[1] - t_crossed.feeGrowthOutside1
                    )

                    net_L = t_crossed.liquidityNet
                    self.liquidity += net_L if going_up else -net_L
                elif boundary in (self.MIN_TICK, self.MAX_TICK):
                    break  # ran off the book
                self.tick = boundary if going_up else boundary - 1
            else:
                self.tick = self._sqrtp_to_tick(self.sqrtp)
                break

        # Get my fees after swap
        my_fees_after = self.my_fees

        my_earned0 = my_fees_after[0] - my_fees_before[0]
        my_earned1 = my_fees_after[1] - my_fees_before[1]

        lp0 = total_swap_fee0 - total_dao_fee0
        lp1 = total_swap_fee1 - total_dao_fee1

        ord_idx = self.seen_block.get(bn, 0)
        self.seen_block[bn] = ord_idx + 1

        # Append to self.fees
        new_fees_row = pl.DataFrame(
            {
                "bn": [bn],
                "ord": [ord_idx],
                "fee0": [total_swap_fee0],
                "fee1": [total_swap_fee0],
                "dao_fee0": [total_dao_fee0],
                "dao_fee1": [total_dao_fee1],
                "my_fee0": [my_earned0],
                "my_fee1": [my_earned1],
                "sqrtp": [self.sqrtp],
            },
            schema=self.fees.schema,
        )
        self.fees = self.fees.vstack(new_fees_row)

        # Append to self.swaps
        token0_amt = amount_in if token_in == 0 else -amount_out
        token1_amt = amount_in if token_in == 1 else -amount_out

        new_swaps_row = pl.DataFrame(
            {
                "bn": [bn],
                "ord": [ord_idx],
                "token0": [token0_amt],
                "token1": [token1_amt],
                "sqrtp": [self.sqrtp],
            },
            schema=self.swaps.schema,
        )
        self.swaps = self.swaps.vstack(new_swaps_row)

        return (
            amount_out,
            remaining,  # remaining > 0 only if book exhausted
            (my_earned0 > 0 or my_earned1 > 0),
        )

    def _get_fee_growth_inside(
        self, tick_lower: int, tick_upper: int
    ) -> tuple[float, float]:
        import math

        current_tick = int(math.floor(self.tick + 1e-9))

        t_lower = self.ticks.get(tick_lower)
        t_upper = self.ticks.get(tick_upper)

        f_g0, f_g1 = self.fee_growth_global[0], self.fee_growth_global[1]

        if t_lower is not None:
            fo_l0 = t_lower.feeGrowthOutside0
            fo_l1 = t_lower.feeGrowthOutside1
        else:
            if current_tick >= tick_lower:
                fo_l0, fo_l1 = f_g0, f_g1
            else:
                fo_l0, fo_l1 = 0.0, 0.0

        if t_upper is not None:
            fo_u0 = t_upper.feeGrowthOutside0
            fo_u1 = t_upper.feeGrowthOutside1
        else:
            if current_tick >= tick_upper:
                fo_u0, fo_u1 = f_g0, f_g1
            else:
                fo_u0, fo_u1 = 0.0, 0.0

        if current_tick < tick_lower:
            fi0 = fo_l0 - fo_u0
            fi1 = fo_l1 - fo_u1
        elif current_tick >= tick_upper:
            fi0 = fo_u0 - fo_l0
            fi1 = fo_u1 - fo_l1
        else:
            fi0 = f_g0 - fo_l0 - fo_u0
            fi1 = f_g1 - fo_l1 - fo_u1

        return fi0, fi1

    def mint(self, tick_lower: int, tick_upper: int, amount: int, mine: bool) -> None:
        assert tick_lower < tick_upper
        assert tick_lower % self.tick_spacing == 0
        assert tick_upper % self.tick_spacing == 0
        assert amount > 0

        import math

        current_tick = int(math.floor(self.tick + 1e-9))

        # Update tick_lower
        if tick_lower not in self.ticks:
            if current_tick >= tick_lower:
                fo0, fo1 = self.fee_growth_global[0], self.fee_growth_global[1]
            else:
                fo0, fo1 = 0.0, 0.0
            self.ticks[tick_lower] = Tick(
                liquidityNet=0,
                liquidityGross=0,
                feeGrowthOutside0=fo0,
                feeGrowthOutside1=fo1,
            )
        self.ticks[tick_lower].liquidityGross += amount
        self.ticks[tick_lower].liquidityNet += amount

        # Update tick_upper
        if tick_upper not in self.ticks:
            if current_tick >= tick_upper:
                fo0, fo1 = self.fee_growth_global[0], self.fee_growth_global[1]
            else:
                fo0, fo1 = 0.0, 0.0
            self.ticks[tick_upper] = Tick(
                liquidityNet=0,
                liquidityGross=0,
                feeGrowthOutside0=fo0,
                feeGrowthOutside1=fo1,
            )
        self.ticks[tick_upper].liquidityGross += amount
        self.ticks[tick_upper].liquidityNet -= amount

        # Sync tick keys
        self.tick_keys = sorted(self.ticks.keys())

        # Update position
        position_key = (tick_lower, tick_upper, mine)
        fi0, fi1 = self._get_fee_growth_inside(tick_lower, tick_upper)
        if position_key not in self.positions:
            self.positions[position_key] = LiquidityPosition(
                liquidity=0,
                feeGrowthInside0Last=fi0,
                feeGrowthInside1Last=fi1,
                tokensOwed0=0.0,
                tokensOwed1=0.0,
            )

        pos = self.positions[position_key]
        fees0 = pos.liquidity * (fi0 - pos.feeGrowthInside0Last)
        fees1 = pos.liquidity * (fi1 - pos.feeGrowthInside1Last)
        pos.tokensOwed0 += fees0
        pos.tokensOwed1 += fees1
        pos.feeGrowthInside0Last = fi0
        pos.feeGrowthInside1Last = fi1
        pos.liquidity += amount

        # Update active pool liquidity if in range
        if tick_lower <= current_tick < tick_upper:
            self.liquidity += amount

    def burn(self, tick_lower: int, tick_upper: int, amount: int, mine: bool) -> None:
        assert tick_lower < tick_upper
        assert tick_lower % self.tick_spacing == 0
        assert tick_upper % self.tick_spacing == 0
        assert amount > 0

        position_key = (tick_lower, tick_upper, mine)
        assert position_key in self.positions
        pos = self.positions[position_key]
        assert pos.liquidity >= amount

        assert tick_lower in self.ticks
        assert tick_upper in self.ticks
        assert self.ticks[tick_lower].liquidityGross >= amount
        assert self.ticks[tick_upper].liquidityGross >= amount

        import math

        current_tick = int(math.floor(self.tick + 1e-9))

        # Update position fees first
        fi0, fi1 = self._get_fee_growth_inside(tick_lower, tick_upper)
        fees0 = pos.liquidity * (fi0 - pos.feeGrowthInside0Last)
        fees1 = pos.liquidity * (fi1 - pos.feeGrowthInside1Last)
        pos.tokensOwed0 += fees0
        pos.tokensOwed1 += fees1
        pos.feeGrowthInside0Last = fi0
        pos.feeGrowthInside1Last = fi1
        pos.liquidity -= amount

        # Update tick_lower
        self.ticks[tick_lower].liquidityGross -= amount
        self.ticks[tick_lower].liquidityNet -= amount
        if self.ticks[tick_lower].liquidityGross == 0:
            del self.ticks[tick_lower]

        # Update tick_upper
        self.ticks[tick_upper].liquidityGross -= amount
        self.ticks[tick_upper].liquidityNet += amount
        if self.ticks[tick_upper].liquidityGross == 0:
            del self.ticks[tick_upper]

        # Sync tick keys
        self.tick_keys = sorted(self.ticks.keys())

        # Update active pool liquidity if in range
        if tick_lower <= current_tick < tick_upper:
            self.liquidity -= amount
            assert self.liquidity >= 0

    @property
    def my_fees(self) -> tuple[float, float]:
        """Returns the total fees (token0, token1) accumulated by 'mine' positions."""
        total0, total1 = 0.0, 0.0
        for pos_key, pos in self.positions.items():
            tick_lower, tick_upper, mine = pos_key
            if mine:
                fi0, fi1 = self._get_fee_growth_inside(tick_lower, tick_upper)
                fees0 = pos.liquidity * (fi0 - pos.feeGrowthInside0Last)
                fees1 = pos.liquidity * (fi1 - pos.feeGrowthInside1Last)
                total0 += pos.tokensOwed0 + fees0
                total1 += pos.tokensOwed1 + fees1
        return total0, total1

    @staticmethod
    def _tick_to_sqrtp(tick: float) -> float:
        return 1.0001 ** (tick / 2)

    @staticmethod
    def _sqrtp_to_tick(sqrtp: float) -> float:
        return np.floor(np.log(float(sqrtp)) / np.log(1.0001) * 2)


@dataclass(frozen=True)
class Position:
    owner: str
    liquidity: int
    tick_lower: int
    tick_upper: int


@dataclass(frozen=True)
class Pool:
    name0: str
    name1: str
    d0: int
    d1: int
    positions: list[Position]
    sqrt_price: int
    tick: int
    liquidity: int
    feeTier: float


@dataclass(frozen=True)
class Fold:
    positions: dict
    swaps: pl.DataFrame
    liq: pl.DataFrame


class UniswapCLMM(gym.Env):
    """Implement Uniswap liquidity ticks and fees correctly"""

    def __init__(
        self,
        position_width: int,
        rebalance_cost: float,
        my_liquidity: int,
        meta: Pool,
        swaps: pl.DataFrame,
        liq: pl.DataFrame,
        params: pl.DataFrame,
        folds: list[list[int]],
    ):
        """
        fee_tier: 0.0005, 0.003, 0.01
        tick_spacing: 10 (for 0.05%), 60 (for 0.3%), 200 (for 1%)
        position_width: number of ticks to add left and right. 2 means [current_tick-2, current_tick+2] (5 ticks wide)
        rebalance_cost: in qty
        initial_positions: liquidity positions
        swaps: bn, ord, token0, token1
        liq: bn, ord, first_tick, last_tick
        params: bn, price, quote_volume, mu, theta, sigma, vol
        """

        # fixed
        self.position_width = position_width
        self.rebalance_cost = rebalance_cost
        self.my_liquidity = my_liquidity
        self.meta = meta
        self.swaps = swaps
        self.liq = liq
        self.params = params
        self.folds = folds

        self.epsilon_floor = 0.05
        self.epsilon_decay = 0.9998
        self.reward_scale = 100.0
        self.active_bonus = 1e-4

        # changed each reset
        self.fold_index: None | int = None
        self.contract: V3Pool | None = None

        # changed each step
        self.gas = 0
        self.block_number = None
        self.active_rows = 0
        self.epsilon = 1.0

        # spaces
        self.observation_space = gym.spaces.Dict(
            {
                "price_deviation": gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "distance_to_edge": gym.spaces.Box(
                    low=-10.0, high=10.0, shape=(1,), dtype=np.float32
                ),
                "stein_signal": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "mean_deviation": gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "sigma": gym.spaces.Box(
                    low=0.0, high=0.1, shape=(1,), dtype=np.float32
                ),
                "active_fraction": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "recent_volatility": gym.spaces.Box(
                    low=0.0, high=0.1, shape=(1,), dtype=np.float32
                ),
                "in_range": gym.spaces.Discrete(n=2),  # 0: False, 1: True
            }
        )

        self.action_space = gym.spaces.Discrete(n=2)  # 0: do nothing, 1: rebalance

    def _observe(self):
        p = self.params.filter(pl.col("block_number") == self.block_number)
        print(p)

        assert p.height == 1
        assert self.fold_index != None

        row = p.row(0, named=True)

        s = row["price"]
        u = pow(self.contract._tick_to_sqrtp(self.upper_tick), 2)
        l = pow(self.contract._tick_to_sqrtp(self.lower_tick), 2)
        c = l + (u - l) / 2
        mu = row["mu"] if row["mu"] is not None and not np.isnan(row["mu"]) else s

        fold = self.folds[self.fold_index]
        numblks = len(fold)
        remblks = fold[-1] - self.block_number

        ret = {
            "price_deviation": np.array([(s - c) / c], dtype=np.float32),
            "distance_to_edge": np.array([(s - l) / (u - l) * 2 - 1], dtype=np.float32),
            "mean_deviation": np.array([(mu - s) / s], dtype=np.float32),
            "stein_signal": np.array(
                [row["theta"] if row["theta"] is not None else 0.0],
                dtype=np.float32,
            ),
            "sigma": np.array(
                [row["sigma"] if row["sigma"] is not None else 0.0],
                dtype=np.float32,
            ),
            "active_fraction": np.array(
                [self.active_rows / (numblks - remblks) if numblks != remblks else 0],
                dtype=np.float32,
            ),
            "recent_volatility": np.array(
                [row["vol"] if row["vol"] is not None else 0.0], dtype=np.float32
            ),
            "in_range": int(s >= l and s <= u),
        }

        limits = {
            "price_deviation": [-1.0, 1.0],
            "distance_to_edge": [-10.0, 10.0],
            "mean_deviation": [-1.0, 1.0],
            "stein_signal": [0.0, 1.0],
            "sigma": [0.0, 0.1],
            "active_fraction": [0.0, 1.0],
            "recent_volatility": [0.0, 0.1],
        }

        for k, l in limits.items():
            assert k in ret and isinstance(ret[k], np.ndarray)
            ret[k] = np.nan_to_num(ret[k], nan=0.0, posinf=l[1], neginf=l[0]).clip(*l)

        return ret

    def _info(self):

        return {
            "fees": self.contract.my_fees,
            "gas": self.gas,
            "epsilon": self.epsilon,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self.fold_index == None:
            self.fold_index = 0
        else:
            self.fold_index = (self.fold_index + 1) % len(self.folds)

        self.gas = 0
        self.active_rows = 0
        self.block_number = self.folds[self.fold_index][0]
        self.epsilon = 1.0

        match self.meta.feeTier:
            case 0.0001:
                protocol_fraction = 0.25
                tickSpacing = 1
            case 0.0005:
                protocol_fraction = 0.25
                tickSpacing = 10
            case 0.003:
                protocol_fraction = 0.1667
                tickSpacing = 60
            case 0.01:
                protocol_fraction = 0.1667
                tickSpacing = 200
            case _:
                assert False

        self.contract = V3Pool(
            self.meta.sqrt_price,
            self.meta.liquidity,
            {},
            tickSpacing,
            self.meta.feeTier,
            protocol_fraction,
        )

        self.lower_tick = (
            self.contract._sqrtp_to_tick(self.meta.sqrt_price) - self.position_width
        )
        self.upper_tick = (
            self.contract._sqrtp_to_tick(self.meta.sqrt_price) + self.position_width
        )

        self.lower_tick = (
            round(self.lower_tick / self.contract.tick_spacing)
            * self.contract.tick_spacing
        )
        self.upper_tick = (
            round(self.upper_tick / self.contract.tick_spacing)
            * self.contract.tick_spacing
        )

        self.contract.mint(self.lower_tick, self.upper_tick, self.my_liquidity, True)

        for pos in self.meta.positions:
            self.contract.mint(pos.tick_lower, pos.tick_upper, pos.liquidity, False)

        return self._observe(), self._info()

    def step(self, action):
        # first, the trades of the block execute, then the position is updated
        sqrtp = self.contract.sqrtp
        new_fees_t0t1 = self.contract.my_fees
        in_range = False

        swaps = self.swaps.filter(pl.col("block_number") == self.block_number)
        liq = self.liq.filter(pl.col("block_number") == self.block_number)

        for ord in sorted(set(swaps["ord"].to_list()) | set(liq["ord"].to_list())):
            for row in swaps.filter(pl.col("ord") == ord).iter_rows(named=True):
                if row["amount0"] > 0:
                    token_in = 0
                    amount_in = row["amount0"]
                    amount_out = row["amount1"]
                else:
                    token_in = 1
                    amount_in = row["amount1"]
                    amount_out = row["amount0"]

                out, remaining, hit = self.contract.swap(
                    self.block_number, token_in, amount_in
                )

                print(out, amount_out, out - amount_out)
                assert remaining == 0
                # assert out == amount_out

                in_range |= hit

            for row in liq.filter(pl.col("ord") == ord).iter_rows(named=True):
                if row["liquidity"] > 0:
                    self.contract.mint(
                        row["tick_lower"], row["tick_upper"], row["liquidity"], False
                    )
                elif row["liquidity"] < 0:
                    self.contract.burn(
                        row["tick_lower"], row["tick_upper"], -row["liquidity"], False
                    )
                else:
                    l.warning(
                        f"""mint/burn with 0 liq for ticks {row['tick_lower']} to {row['tick_upper']}"""
                    )

        new_fee_t0t1 = map(
            lambda p: p[1] - p[0], zip(new_fees_t0t1, self.contract.my_fees)
        )

        new_gas = 0
        reward = 0
        new_fees = new_fees_t0t1[0] + pow(self.contract.sqrtp, 2) * new_fees_t0t1[1]

        # rebalance
        if action == 1:
            self.contract.burn(
                self.lower_tick, self.upper_tick, self.my_liquidity, True
            )

            self.lower_tick = self.contract._sqrtp_to_tick(sqrtp) - self.position_width
            self.upper_tick = self.contract._sqrtp_to_tick(sqrtp) + self.position_width

            self.lower_tick = (
                round(self.lower_tick / self.contract.tick_spacing)
                * self.contract.tick_spacing
            )
            self.upper_tick = (
                round(self.upper_tick / self.contract.tick_spacing)
                * self.contract.tick_spacing
            )

            # XXX add IL from swapping self.my_liquidity / 2
            self.contract.mint(
                self.lower_tick, self.upper_tick, self.my_liquidity, True
            )
            new_gas += self.rebalance_cost

        # trade was within our range
        if in_range:
            self.active_rows += 1
            # Eq. 28 active bonus
            reward += self.active_bonus

        # Eq. 28 less active bonus
        reward += (new_fees - new_gas) * self.reward_scale

        # Advance to next state (t+1)
        self.block_number += 1
        truncated = self.block_number > max(
            self.swaps["block_number"].max(), self.liq["block_number"].max()
        )

        if truncated:
            next_observation = observation
        else:
            next_observation = self._observe()

        # update counters
        self.gas += new_gas
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_floor)

        return next_observation, reward, False, truncated, self._info()


class Experience:
    def __init__(self, capacity, observation_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        # ring buffer
        self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty((capacity, 1), dtype=np.int64)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.terminated = np.empty((capacity, 1), dtype=np.bool_)

    def add(self, o, a, r, o_next, t):
        self.observations[self.ptr] = o
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_observations[self.ptr] = o_next
        self.terminated[self.ptr] = t

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=min(self.size, batch_size))
        return (
            torch.as_tensor(self.observations[idx]),
            torch.as_tensor(self.actions[idx]),
            torch.as_tensor(self.rewards[idx]),
            torch.as_tensor(self.next_observations[idx]),
            torch.as_tensor(self.terminated[idx]),
        )


def make_folds(l: list, fold_size: int) -> Iterable[pl.DataFrame]:
    from random import sample
    from math import ceil
    from more_itertools import batched

    return [
        ll
        for ll in sample(list(batched(l, fold_size)), ceil(len(l) / fold_size))
        if len(ll) == fold_size
    ]


async def gql_get_pool_tokens(ep: BaseEndpoint, contract: str) -> dict[str, Any]:
    query = """
    query GetPoolTokens($poolId: ID!) {
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
    data = await ep(query, {"poolId": contract})

    return data.get("data", {}).get("pool", {})


async def gql_get_liquidity_profile(
    ep: BaseEndpoint, bn: int, contract: str
) -> list[Position]:
    query = """
    query GetPoolPositionsPaginated($poolAddress: String!, $blockNumber: Int!, $lastId: String!) {
      positions(
        where: { pool: $poolAddress, id_gt: $lastId }
        block: { number: $blockNumber }
        first: 1000
        orderBy: id
        orderDirection: asc
      ) {
        id
        owner
        liquidity
        tickLower {
          tickIdx
        }
        tickUpper {
          tickIdx
        }
      }
    }
    """

    ret: list[Position] = []
    last_id = None

    while True:
        vars = {
            "poolAddress": contract,
            "blockNumber": bn,
            "lastId": str(last_id),
        }
        page = await ep(query, vars)

        pos = page.get("data", {}).get("positions", [])
        if len(pos) == 0:
            return ret

        ret.extend(
            [
                Position(
                    owner=p["owner"],
                    liquidity=int(p["liquidity"]),
                    tick_lower=int(p["tickLower"]["tickIdx"]),
                    tick_upper=int(p["tickUpper"]["tickIdx"]),
                )
                for p in pos
            ]
        )
        last_id = pos[-1]["id"]


async def pool_meta(ep: BaseEndpoint, bn: int, contract: str) -> Pool:
    meta, positions = await asyncio.gather(
        gql_get_pool_tokens(ep, contract),
        gql_get_liquidity_profile(ep, bn, contract),
    )

    pool = Pool(
        name0=meta["token0"]["name"],
        name1=meta["token1"]["name"],
        d0=int(meta["token0"]["decimals"]),
        d1=int(meta["token1"]["decimals"]),
        sqrt_price=int(meta["sqrtPrice"]),
        liquidity=int(meta["liquidity"]),
        tick=int(meta["tick"]),
        positions=positions,
        feeTier=float(meta["feeTier"]) / 1_000_000.0,
    )
    return pool


async def from_ethereum(
    df: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Pool]:
    from sgqlc.endpoint.httpx import HTTPXEndpoint
    from os import getenv
    from eth_utils import event_signature_to_log_topic

    bn = df["block_number"].min()
    pool = df["address"].unique().to_list()

    assert len(pool) == 1
    pool = f"0x{pool[0].hex()}"

    headers = {"Authorization": f"""bearer {getenv('GRAPH_API_KEY')}"""}
    url = "https://gateway.thegraph.com/api/[api-key]/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
    endpoint = HTTPXEndpoint(url, headers, client=AsyncClient())

    meta = await pool_meta(endpoint, bn, pool)

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
                lambda b: float(int.from_bytes(b, signed=True, byteorder="big")),
                return_dtype=pl.Float64,
            ),
            amount1=pl.col("slot1").map_elements(
                lambda b: float(int.from_bytes(b, signed=True, byteorder="big")),
                return_dtype=pl.Float64,
            ),
            quote_volume=pl.col("slot0").map_elements(
                lambda b: float(abs(int.from_bytes(b, signed=True, byteorder="big")))
                / pow(10, meta.d0),
                return_dtype=pl.Float64,
            ),
            price=1.0
            / pow(
                pl.col("slot2").map_elements(
                    lambda b: float(int.from_bytes(b, byteorder="big")) / pow(2, 96),
                    return_dtype=pl.Float64,
                ),
                2,
            )
            * pow(10, meta.d1 - meta.d0),
            liquidity=pl.col("slot3").map_elements(
                lambda b: float(int.from_bytes(b, byteorder="big")),
                return_dtype=pl.Float64,
            ),
            tick=pl.col("slot4").map_elements(
                lambda b: int.from_bytes(b, signed=True, byteorder="big"),
                return_dtype=pl.Int32,
            ),
            ord=pl.col("transaction_index") * 100 + pl.col("log_index"),
        )
        .select(
            [
                "ts",
                "block_number",
                "transaction_index",
                "ord",
                "log_index",
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

    liq = (
        df.filter((pl.col("topic0") == mint) | (pl.col("topic0") == burn))
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
            liquidity=pl.col("slot2").map_elements(
                lambda b: float(int.from_bytes(b, byteorder="big")),
                return_dtype=pl.Float64,
            ),
            amount0=pl.col("slot3").map_elements(
                lambda b: float(int.from_bytes(b, byteorder="big")),
                return_dtype=pl.Float64,
            ),
            amount1=pl.col("slot4").map_elements(
                lambda b: float(int.from_bytes(b, byteorder="big")),
                return_dtype=pl.Float64,
            ),
            ord=pl.col("transaction_index") * 100 + pl.col("log_index"),
        )
        .with_columns(
            liquidity=pl.when(pl.col("topic0") == burn)
            .then(-pl.col("liquidity"))
            .otherwise(pl.col("liquidity")),
        )
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

    return (
        swaps.join(params.select("block_number"), on="block_number"),
        liq.join(params.select("block_number"), on="block_number"),
        params,
        meta,
    )


gym.register(
    id="UniswapCLMM-v1",
    entry_point=UniswapCLMM,
)


def train(swaps, liq, params, meta):
    from tqdm import tqdm
    from math import floor
    import matplotlib.pyplot as plt
    from torch import nn
    from torch import optim
    import random
    from gymnasium.wrappers import FlattenObservation
    from copy import deepcopy

    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    first_bn = min(
        swaps["block_number"].min(),
        liq["block_number"].min(),
        params["block_number"].min(),
    )
    last_bn = max(
        swaps["block_number"].max(),
        liq["block_number"].max(),
        params["block_number"].max(),
    )

    block_range = list(range(first_bn, last_bn + 1))
    blen = len(block_range)

    train_len = floor(blen * 0.7)
    val_len = floor(blen * 0.15)
    test_len = blen - train_len - val_len
    train, val, test = (
        block_range[:train_len],
        block_range[train_len : train_len + val_len],
        block_range[train_len + val_len :],
    )

    print(f"train {len(train)} blocks, val {len(val)} blocks, test {len(test)} blocks")

    train = make_folds(train, fold_size=min(train_len // 5, 36_000 // 12))
    test = make_folds(test, fold_size=min(test_len // 5, 36_000 // 12))

    print(f"train {len(train)} folds, test {len(test)} folds")

    args = {
        "position_width": 200,
        "rebalance_cost": 3,
        "my_liquidity": 1_000_000,
        "meta": meta,
        "swaps": swaps,
        "liq": liq,
        "params": params,
    }

    train_env = FlattenObservation(gym.make("UniswapCLMM-v1", folds=train, **args))
    val_env = FlattenObservation(gym.make("UniswapCLMM-v1", folds=[val], **args))

    policy = nn.Sequential(
        nn.Linear(train_env.observation_space.shape[0], 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, train_env.action_space.n),
    )

    opt = optim.Adam(policy.parameters(), lr=0.0001)

    target = deepcopy(policy)
    target.load_state_dict(policy.state_dict())
    target.eval()

    frag = []
    replay_memory = Experience(100_000, train_env.observation_space.shape[0])
    steps = 0

    gamma = 0.99
    epsilon = 1.0
    epsilon_floor = 0.05
    epsilon_decay = 0.9998

    for episode in tqdm(range(len(train)), desc="training"):
        observation, info = train_env.reset()

        episode_over = False
        total_reward = 0
        episode_loss = []

        while not episode_over:
            if torch.rand(()).item() < epsilon:
                action = train_env.action_space.sample()
            else:
                with torch.no_grad():
                    action = (
                        policy(torch.tensor(observation, dtype=torch.float32))
                        .argmax(dim=-1)
                        .item()
                    )

            next_observation, reward, terminated, truncated, info = train_env.step(
                action
            )

            replay_memory.add(
                observation, action, reward, next_observation, terminated or truncated
            )

            if replay_memory.size >= 128:
                recall = replay_memory.sample(128)

                y = recall[2].clone()  # rewards (clone to avoid buffer corruption)

                # y = r + gamma * max_a'[ Q(observation_{t+1}, a', w-) ]
                with torch.no_grad():
                    best_actions = policy(recall[3]).argmax(dim=1, keepdim=True)
                    q = target(recall[3]).gather(1, best_actions).squeeze(-1)

                y += gamma * q.unsqueeze(-1) * (1 - recall[4].float())  # terminated

                opt.zero_grad()

                # (y - Q(observation_t, a, w))^2
                loss = nn.functional.smooth_l1_loss(
                    policy(recall[0]).gather(1, recall[1]), y
                )
                loss.backward()

                episode_loss.append(loss.item())

                opt.step()

            if steps % 100 == 0:
                target.load_state_dict(policy.state_dict())

            total_reward += reward
            episode_over = terminated or truncated
            steps += 1
            observation = next_observation
            epsilon = max(epsilon * epsilon_decay, epsilon_floor)

        # validation rollout (greedy)
        val_observation, _ = val_env.reset()
        val_over = False
        val_rebals = 0
        while not val_over:
            with torch.no_grad():
                val_action = (
                    policy(torch.tensor(val_observation, dtype=torch.float32))
                    .argmax(dim=-1)
                    .item()
                )
            val_observation, _, terminated, truncated, val_info = val_env.step(
                val_action
            )
            val_over = terminated or truncated
            if val_action == 1:
                val_rebals += 1

        loss_val = np.mean(episode_loss) if episode_loss else 0.0
        frag.append(
            {
                "episode": episode,
                "fees": info["fees"],
                "gas": info["gas"],
                "loss": loss_val,
                "pnl": info["fees"] - info["gas"],
                "val_pnl": val_info["fees"] - val_info["gas"],
                "val_rebalances": val_rebals,
                "reward": total_reward,
            }
        )

        print(
            f"episode {frag[-1]['episode']}: loss={frag[-1]['loss']:.5f}, val_pnl={frag[-1]['val_pnl']:.2f}, val_rebalances={frag[-1]['val_rebalances']}"
        )

    train_env.close()

    df_frag = pl.DataFrame(frag).sort("episode").to_pandas()
    df_frag.plot(x="episode", y="val_pnl", title="validation pnl")
    plt.show()


async def main():
    df = (
        pl.read_parquet("ethereum__logs__*.parquet")
        .join(
            pl.read_parquet("ethereum__blocks__*.parquet").select(
                pl.col("block_number"),
                ts=pl.from_epoch(
                    pl.col("timestamp"), time_unit="s"
                ).dt.replace_time_zone("UTC"),
            ),
            on=["block_number"],
        )
        .sort(["block_number", "transaction_index", "log_index"])
    )
    swaps, liq, params, meta = await from_ethereum(df)

    train(swaps, liq, params, meta)


if __name__ == "__main__":
    with logging_redirect_tqdm():
        try:
            asyncio.run(main())
        except Exception as e:
            l.exception("Fatal error during sync", exc_info=e)
            sys.exit(1)
