import polars as pl
import polars_ols as pls

import gymnasium as gym
import numpy as np
import torch

import asyncio
from sgqlc.endpoint.base import BaseEndpoint
from httpx import AsyncClient

from copy import deepcopy

from typing import Optional, Dict, Any, Literal, Tuple
from dataclasses import dataclass, replace

from uniswapv3.emulator import Tick, Emulator
from uniswapv3.load import Pool, make_endpoint, from_ethereum, pool_meta
import uniswapv3.math as v3math

from cli.token import Token, liquidity_for_value, parse_amount, composition_in_range


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
    block_numbers: list[int]
    initial: Pool


class Rammstein(gym.Env):
    """Implement Uniswap liquidity ticks and fees correctly"""

    def __init__(
        self,
        position_width: int,  # in tick_spacing
        rebalance_cost: Token,  # in liquidity
        capital: Token,
        swaps: pl.DataFrame,
        liq: pl.DataFrame,
        params: pl.DataFrame,
        folds: list[Fold],
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

        assert capital.ord == rebalance_cost.ord

        # fixed
        self.position_width = position_width
        self.rebalance_cost = rebalance_cost
        self.capital = capital
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
        self.block_number: None | int = None
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
        assert self.block_number is not None and self.contract is not None

        p = self.params.filter(pl.col("block_number") == self.block_number)

        assert p.height == 1
        assert self.fold_index != None

        row = p.row(0, named=True)

        # clean this up and move to tick space, figure out what coord to use for OU params
        s = (self.contract.sqrt_price_x96 / (2**96)) ** 2
        l = (
            v3math.get_sqrt_ratio_at_tick(self.contract.my_position.tick_lower)
            / (2**96)
        ) ** 2
        u = (
            v3math.get_sqrt_ratio_at_tick(self.contract.my_position.tick_upper)
            / (2**96)
        ) ** 2
        c = l + (u - l) / 2
        p = 1.0 / s
        mu = row["mu"] if row["mu"] is not None and not np.isnan(row["mu"]) else p

        fold = self.folds[self.fold_index]
        numblks = len(fold.block_numbers)
        remblks = fold.block_numbers[-1] - self.block_number

        ret = {
            "price_deviation": np.array([(s - c) / c], dtype=np.float32),
            "distance_to_edge": np.array([(s - l) / (u - l) * 2 - 1], dtype=np.float32),
            "mean_deviation": np.array([(mu - p) / p], dtype=np.float32),
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
            "fees": self.fee,
            "il": self.impermanent_loss,
            "bn": self.block_number,
            "gas": self.gas,
            "epsilon": self.epsilon,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self.fold_index == None:
            self.fold_index = 0
        else:
            self.fold_index = (self.fold_index + 1) % len(self.folds)

        fold = self.folds[self.fold_index]

        self.gas = 0
        self.active_rows = 0
        self.block_number = fold.block_numbers[0]
        self.epsilon = 1.0
        self.impermanent_loss = 0
        self.fee = 0

        self.contract = Emulator(
            fold.initial.sqrt_price_x96,
            fold.initial.tick,
            fold.initial.liquidity,
            deepcopy(fold.initial.ticks),
            fold.initial.tick_spacing,
            fold.initial.fee_pips,
            fold.initial.protocol_fraction,
            fold.initial.max_liquidity_per_tick,
        )

        # initial position
        tsp = fold.initial.tick_spacing
        tick_lower = max(
            Emulator.MIN_TICK,
            fold.initial.tick - max(0, (self.position_width - 1) * tsp),
        )
        tick_upper = min(
            Emulator.MAX_TICK, fold.initial.tick + (self.position_width * tsp)
        )
        assert Emulator.MIN_TICK <= tick_lower < tick_upper <= Emulator.MAX_TICK

        tl = int(np.floor(tick_lower / tsp) * tsp)
        tu = int(np.ceil(tick_upper / tsp) * tsp)
        l = liquidity_for_value(fold.initial.sqrt_price_x96, tl, tu, self.capital)
        self.contract.set_position(tl, tu, l)
        self.init0, self.init1 = self.contract.position_amounts()

        return self._observe(), self._info()

    def step(self, action):
        assert self.contract is not None and self.block_number is not None

        observation = self._observe()

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

                self.contract.swap(self.block_number, token_in, int(amount_in))

            for row in liq.filter(pl.col("ord") == ord).iter_rows(named=True):
                self.contract.modify_liquidity(
                    row["tick_lower"],
                    row["tick_upper"],
                    row["liquidity"],
                )

        in_range = (
            self.contract.my_position.tick_lower
            <= self.contract.tick
            < self.contract.my_position.tick_upper
        )

        meta = self.folds[self.fold_index].initial
        df = (
            self.contract.swap_df.rechunk()
            .filter(pl.col("bn") == self.block_number)
            .sort("ord")
            .with_columns(
                fee=pl.when(self.capital.ord == 0)
                .then(pl.col("fee0") + (pl.col("fee1") / pl.col("price")))
                .otherwise(pl.col("fee1") + (pl.col("fee0") * pl.col("price"))),
            )
            .select(
                pl.col("fee").sum(),
            )
        )

        new_gas = 0
        reward = 0
        new_fees = int(df["fee"].item()) or 0

        self.fee += new_fees

        # rebalance
        if action == 1:
            # compute IL
            end0, end1 = self.contract.position_amounts()
            p = (self.contract.sqrt_price_x96 / (2**96)) ** 2

            if self.capital.ord == 0:
                pos_end = end0 + end1 / p
                hold_end = self.init0 + self.init1 / p
            else:
                pos_end = end1 + end0 * p
                hold_end = self.init1 + self.init0 * p

            self.impermanent_loss += int(pos_end - hold_end)

            # move position around latest price
            meta = self.folds[self.fold_index].initial
            tsp = meta.tick_spacing
            tick_lower = max(
                Emulator.MIN_TICK,
                self.contract.tick - max(0, (self.position_width - 1) * tsp),
            )
            tick_upper = min(
                Emulator.MAX_TICK, self.contract.tick + (self.position_width * tsp)
            )
            assert Emulator.MIN_TICK <= tick_lower < tick_upper <= Emulator.MAX_TICK

            tl = int(np.floor(tick_lower / tsp) * tsp)
            tu = int(np.ceil(tick_upper / tsp) * tsp)
            liquidity = liquidity_for_value(
                self.contract.sqrt_price_x96,
                tl,
                tu,
                replace(self.capital, amount=int(pos_end) - self.rebalance_cost.amount),
            )
            self.contract.set_position(tl, tu, liquidity)
            self.init0, self.init1 = self.contract.position_amounts()

            # XXX add swap fees
            new_gas += self.rebalance_cost.amount

        # trade was within our range
        if in_range:
            self.active_rows += 1
            # Eq. 28 active bonus
            reward += self.active_bonus

        # Eq. 28 less active bonus
        reward += (new_fees / (10**self.capital.decimals) - new_gas) * self.reward_scale

        # Advance to next state (t+1)
        self.block_number += 1
        truncated = self.block_number > self.folds[self.fold_index].block_numbers[-1]

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


async def make_folds(
    l: list, fold_size: int, ep: BaseEndpoint, contract: str
) -> list[Fold]:
    from random import sample
    from math import ceil
    from more_itertools import batched

    sem = asyncio.Semaphore(4)

    async def fold(bn: list[int], contract: str) -> Fold:
        async with sem:
            return Fold(
                block_numbers=bn, initial=await pool_meta(ep, bn[0] - 1, contract)
            )

    fut = [
        fold(ll, contract)
        for ll in sample(list(batched(l, fold_size)), ceil(len(l) / fold_size))
        if len(ll) == fold_size
    ]

    return await asyncio.gather(*fut)


gym.register(
    id="Rammstein-v1",
    entry_point=Rammstein,
)


async def prepare(
    swaps: pl.DataFrame,
    liq: pl.DataFrame,
    params: pl.DataFrame,
    ep: BaseEndpoint,
    contract: str,
) -> tuple[list[Fold], list[Fold], list[Fold], Pool]:
    from tqdm import tqdm
    from math import floor
    import matplotlib.pyplot as plt
    from torch import nn
    from torch import optim
    import random
    from gymnasium.wrappers import FlattenObservation
    from copy import deepcopy

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
    train_len = 1000
    val_len = floor(blen * 0.15)
    val_len = 100
    test_len = blen - train_len - val_len
    train, val, test = (
        block_range[:train_len],
        block_range[train_len : train_len + val_len],
        block_range[train_len + val_len :],
    )

    l.info(f"train {len(train)} blocks, val {len(val)} blocks, test {len(test)} blocks")

    trainf = await make_folds(
        train, train_len // 2, ep, contract
    )  # min(train_len // 5, 36_000 // 12), ep, contract)
    valf = await make_folds(val, val_len, ep, contract)
    testf = await make_folds(test, test_len, ep, contract)
    meta = await pool_meta(ep, first_bn, contract)

    return trainf, valf, testf, meta


def train_agent(
    train: list[Fold],
    val: list[Fold],
    swaps: pl.DataFrame,
    liq: pl.DataFrame,
    params: pl.DataFrame,
    capital: Token,
    rebalance: Token,
):
    from tqdm import tqdm
    from math import floor
    import matplotlib.pyplot as plt
    from torch import nn
    from torch import optim
    from gymnasium.wrappers import FlattenObservation
    from copy import deepcopy

    args = {
        "position_width": 200,
        "rebalance_cost": rebalance,
        "capital": capital,
        "swaps": swaps,
        "liq": liq,
        "params": params,
    }

    train_env = FlattenObservation(gym.make("Rammstein-v1", folds=train, **args))
    val_env = FlattenObservation(gym.make("Rammstein-v1", folds=val, **args))

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
        episode_over = False
        total_reward = 0
        episode_loss = []

        observation, info = train_env.reset()

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
    import argparse

    parser = argparse.ArgumentParser(
        description="Uniswap v3 backtest",
    )
    parser.add_argument(
        "--blocks", nargs="+", help="Parquet files of collected blocks", default=[]
    )
    parser.add_argument(
        "--logs", nargs="+", help="Parquet files of collected logs", default=[]
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--capital",
        help="Position's capital in one token.",
    )
    parser.add_argument(
        "--rebalance",
        help="Position's rebalance cost in one token.",
    )
    parser.add_argument(
        "--output", help="Parquet file output", default="backtest.parquet"
    )

    args, unknown = parser.parse_known_args()

    if args.verbose:
        l.getLogger().setLevel(l.DEBUG)

    blks = pl.read_parquet(args.blocks).select(
        pl.col("block_number"),
        ts=pl.from_epoch(pl.col("timestamp"), time_unit="s").dt.replace_time_zone(
            "UTC"
        ),
    )
    logs = (
        pl.read_parquet(args.logs)
        .join(blks, on=["block_number"])
        .sort(["block_number", "transaction_index", "log_index"])
    )

    contract, endpoint = make_endpoint(logs)
    swaps, liq, params, _ = await from_ethereum(logs, endpoint, contract)
    train, val, test, meta = await prepare(swaps, liq, params, endpoint, contract)

    capital = parse_amount(args.capital, meta)
    rebalance = parse_amount(args.rebalance, meta)

    l.info(
        f"backtest {meta.symbol0}/{meta.symbol1}, capital {capital.str}, rebalance cost {rebalance.str}"
    )

    train_agent(train, val, swaps, liq, params, capital, rebalance)


if __name__ == "__main__":
    import random

    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    with logging_redirect_tqdm():
        try:
            asyncio.run(main())
        except Exception as e:
            l.exception("Fatal error during sync", exc_info=e)
            sys.exit(1)
