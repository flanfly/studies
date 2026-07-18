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

from uniswapv3.emulator import Tick, Emulator
from uniswapv3.load import Pool
import uniswav3.math as v3math

import sys
from dotenv import load_dotenv
import logging as l
from tqdm.contrib.logging import logging_redirect_tqdm

load_dotenv()

l.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=l.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)


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
        self.contract: Emulator | None = None

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
        self.contract = Emulator(
            self.meta.sqrt_price_x96,
            self.meta.tick,
            self.meta.liquidity,
            self.ticks,
            self.meta.tick_spacing,
            self.meta.fee_pips,
            self.meta.protocol_fraction,
            self.meta.max_liquidity_per_tick,
        )

        # initial position
        lt = (
            v3math.get_tick_at_sqrt_ratio(self.meta.sqrt_price_x96)
            - self.position_width
        )
        ut = (
            v3math.get_tick_at_sqrt_ratio(self.meta.sqrt_price_x96)
            + self.position_width
        )
        self.set_position(lt, ut, self.my_liquidity)

        return self._observe(), self._info()

    def step(self, action):
        # first, the trades of the block execute, then the position is updated
        sqrt_price_x96 = self.contract.sqrt_price_x96
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

                assert self.contract.tick - row["tick"] <= 1, (
                    self.contract.tick,
                    row["tick"],
                )
                assert close(self.contract.sqrtp, row["sqrtp"]), (
                    self.contract.sqrtp,
                    row["sqrtp"],
                )
                assert close(self.contract.liquidity, row["liquidity"]), (
                    self.contract.liquidity,
                    row["liquidity"],
                )
                assert remaining == 0

                in_range |= hit

            for row in liq.filter(pl.col("ord") == ord).iter_rows(named=True):
                if row["liquidity"] > 0:
                    amount0, amount1 = self.contract.mint(
                        row["tick_lower"], row["tick_upper"], row["liquidity"], False
                    )
                    assert abs(amount0 - row["amount0"]) <= max(
                        1e-6 * row["amount0"], 2.0
                    ), (amount0, row["amount0"])
                    assert abs(amount1 - row["amount1"]) <= max(
                        1e-6 * row["amount1"], 2.0
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
            # self.contract.burn(
            #    self.lower_tick, self.upper_tick, self.my_liquidity, True
            # )

            # self.lower_tick = self.contract._sqrtp_to_tick(sqrtp) - self.position_width
            # self.upper_tick = self.contract._sqrtp_to_tick(sqrtp) + self.position_width

            # self.lower_tick = (
            #    round(self.lower_tick / self.contract.tick_spacing)
            #    * self.contract.tick_spacing
            # )
            # self.upper_tick = (
            #    round(self.upper_tick / self.contract.tick_spacing)
            #    * self.contract.tick_spacing
            # )

            ## XXX add IL from swapping self.my_liquidity / 2
            # self.contract.mint(
            #    self.lower_tick, self.upper_tick, self.my_liquidity, True
            # )
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
        for ll in list(batched(l, fold_size))
        # for ll in sample(list(batched(l, fold_size)), ceil(len(l) / fold_size))
        if len(ll) == fold_size
    ]


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
    last_tick = Emulator.MIN_TICK

    while last_tick <= Emulator.MAX_TICK:
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
        d0=int(meta["token0"]["decimals"]),
        d1=int(meta["token1"]["decimals"]),
        sqrt_price_x96=int(meta["sqrtPrice"]),
        liquidity=int(meta["liquidity"]),
        tick=int(meta["tick"]),
        ticks=ticks,
        fee_pips=int(meta["feeTier"]),
    )
    return pool


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
