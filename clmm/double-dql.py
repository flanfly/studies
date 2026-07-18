import polars as pl
import polars_ols as pls
import polars_u256_plugin as plu

import gymnasium as gym
import numpy as np
import torch

import asyncio

from typing import Optional, Dict, Any, Literal
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


load_dotenv()


class RammstreinEnv(gym.Env):
    """RammStein reference environment"""

    def __init__(self, folds: list[pl.DataFrame]):
        # mu, theta and vol are backwards looking and don't include the row's own price
        assert all(
            column in fold.columns
            for column in ["price", "qty", "mu", "theta", "vol", "sigma"]
            for fold in folds
        )
        assert len(folds) > 0

        self.folds = folds
        self.fold_index = -1  # reset increments
        self.action_space = gym.spaces.Discrete(n=2)  # 0: do nothing, 1: rebalance
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

        self.width = 0.01  # 1%
        self.fee = 0.0005  # 0.05%
        self.rebalance_cost = 2  # $2
        self.fee_fraction = (
            0.01 * 0.1
        )  # we own 1% of the pools active liquidity and assume pool volume is 10% of CEX
        self.capital = 10_000
        self.reward_scale = 100.0
        self.active_bonus = 1e-4

        self.epsilon_floor = 0.05
        self.epsilon_decay = 0.9998
        self.epsilon = 1.0

    def _observe(self):
        assert self.episode.height > 0

        row = self.episode.row(0, named=True)

        s = row["price"]
        c = self.center
        u = self.center * (1 + self.width)
        l = self.center * (1 - self.width)
        mu = row["mu"] if row["mu"] is not None and not np.isnan(row["mu"]) else s

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
                [
                    (
                        self.active_rows
                        / (self.folds[self.fold_index].height - self.episode.height)
                        if self.folds[self.fold_index].height != self.episode.height
                        else 0
                    )
                ],
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
            "fees": self.fees,
            "gas": self.gas,
            "epsilon": self.epsilon,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)

        # select the next fold
        self.fold_index = (self.fold_index + 1) % len(self.folds)
        self.episode = self.folds[self.fold_index]

        # skip the first trade
        self.center = self.episode["price"].first()
        self.price = self.center
        self.episode = self.episode[1:]

        # reset counter
        self.active_rows = 0
        self.fees = 0
        self.gas = 0

        return self._observe(), self._info()

    def step(self, action):
        observation = self._observe()
        row = self.episode.row(0, named=True)

        new_gas = 0
        new_fee = 0
        reward = 0

        # the order of operation is that first the trade is occuring at the previous position, then the position updates to the last observed price (one step delay).

        # move the position to the last price
        if action == 1:
            new_gas = (self.fee * self.capital / 2) + self.rebalance_cost
            self.center = self.price

        # trade was within our range
        if observation["in_range"]:
            new_fee += row["qty"] * row["price"] * self.fee_fraction * self.fee
            self.active_rows += 1
            # Eq. 28 active bonus
            reward += self.active_bonus

        # Eq. 28 less active bonus
        reward += (new_fee - new_gas) / self.capital * self.reward_scale

        # Advance to next state (t+1)
        self.episode = self.episode[1:]
        truncated = self.episode.height == 0

        if truncated:
            next_observation = observation
        else:
            next_observation = self._observe()

        # update counters
        self.fees += new_fee
        self.gas += new_gas
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_floor)
        self.price = row["price"]

        return next_observation, reward, False, truncated, self._info()


@dataclass(frozen=True)
class Fold:
    positions: dict
    swaps: pl.DataFrame
    liq: pl.DataFrame


class UniswapCLMM(gym.Env):
    """Implement Uniswap liquidity ticks and fees correctly"""

    def __init__(
        self,
        fee_tier: float,
        protocol_fraction: float,
        tick_spacing: int,
        position_width: int,
        rebalance_cost: float,
        meta: Pool,
        swaps: pl.DataFrame,
        liq: pl.DataFrame,
        params: pl.DataFrame,
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

        self.fee_tier = fee_tier
        self.protocol_fraction = protocol_fraction
        self.tick_spacing = tick_spacing
        self.position_width = position_width
        self.rebalance_cost = rebalance_cost
        self.meta = meta
        self.swaps = swaps
        self.liq = liq
        self.observation_space = gym.spaces.Dict(
            {
                "price_deviation": gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "distance_to_edge": gym.spaces.Box(
                    low=-10.0, high=10.0, shape=(1,), dtype=np.float32
                ),
                "active_liquidity_fraction": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "in_range": gym.spaces.Discrete(n=2),
                "current_tick": gym.spaces.Box(
                    low=-100000.0, high=100000.0, shape=(1,), dtype=np.float32
                ),
                "lower_tick": gym.spaces.Box(
                    low=-100000.0, high=100000.0, shape=(1,), dtype=np.float32
                ),
                "upper_tick": gym.spaces.Box(
                    low=-100000.0, high=100000.0, shape=(1,), dtype=np.float32
                ),
            }
        )

        self.action_space = gym.spaces.Discrete(n=2)  # 0: do nothing, 1: rebalance

    def _observe(self):
        p = self.params.filter(pl.col("block_number") == self.block_number)
        assert p.height == 1

        row = p.row(0, named=True)

        s = row["price"]
        u = pow(self.contract._tick_to_sqrt(self.upper_tick), 2)
        l = pow(self.contract._tick_to_sqrt(self.lower_tick), 2)
        mu = row["mu"] if row["mu"] is not None and not np.isnan(row["mu"]) else s

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
                [
                    (
                        self.active_rows
                        / (self.folds[self.fold_index].height - self.episode.height)
                        if self.folds[self.fold_index].height != self.episode.height
                        else 0
                    )
                ],
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
        from .uniswapv3_pool import V3Pool

        super().reset(seed=seed)

        if self.meta.feeTier > 0.00005:
            protocol_fraction = 0.1667
        else:
            protocol_fraction = 0.25

        self.contract = V3Pool(
            self.meta.sqrt_price,
            self.meta.liquidity,
            {},
            self.meta.tickSpacing,
            self.meta.feeTier,
            protocol_fraction,
        )

        for pos in self.meta.positions:
            self.contract.mint(pos.tick_lower, pos.tick_upper, pos.liquidity, False)

    def step(self, action):
        # first, the trades of the block execute, then the position is updated
        sqrtp = self.contract.sqrtp
        new_fees = -self.my_fees
        in_range = False

        swaps = self.swaps.filter(pl.col("block_number") == self.bn)
        liq = self.liq.filter(pl.col("block_number") == self.bn)

        for ord in (swaps["ord"] + liq["ord"]).sort():
            for row in swaps.filter(pl.col("ord") == ord).iter_rows(named=True):
                if row["token0"] > 0:
                    token_in = 0
                    amount_in = row["token0"]
                    amount_out = row["token1"]
                else:
                    token_in = 1
                    amount_in = row["token1"]
                    amount_out = row["token0"]

                out, remaining, hit = self.contract.swap(self.bn, token_in, amount_in)
                assert remaining == 0
                assert out == amount_out

                in_range |= hit

            for row in liq.filter(pl.col("ord") == ord).iter_rows(named=True):
                if row["liquidity"] >= 0:
                    self.contract.mint(
                        row["tick_lower"], row["tick_upper"], row["liquidity"], False
                    )
                else:
                    self.contract.burn(
                        row["tick_lower"], row["tick_upper"], -row["liquidity"], False
                    )

        new_fees += self.my_fees

        # rebalance
        if action == 1:
            self.contract.burn(self.lower_tick, self.upper_tick, self.liquidity, True)

            self.lower_tick = self.contract._sqrtp_to_tick(sqrtp) - self.position_width
            self.upper_tick = self.contract._sqrtp_to_tick(sqrtp) + self.position_width

            # add IL from swapping self.liquidity / 2
            self.contract.mint(self.lower_tick, self.upper_tick, self.liquidity, True)
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


def make_folds(df: pl.DataFrame, fold_size: int = 36_000) -> Iterable[pl.DataFrame]:
    from random import sample
    from math import ceil

    return [
        df
        for df in sample(list(df.iter_slices(fold_size)), ceil(df.height / fold_size))
        if df.height == fold_size
    ]


gym.register(
    id="UniswapCLMM-v0",
    entry_point=RammstreinEnv,
)


def train(bars):
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

    blen = len(bars)

    train_len = floor(blen * 0.7)
    val_len = floor(blen * 0.15)
    test_len = blen - train_len - val_len
    train, val, test = (
        bars[:train_len],
        bars[train_len : train_len + val_len],
        bars[train_len + val_len :],
    )
    train = make_folds(train, fold_size=min(train_len // 5, 36_000))
    test = make_folds(test, fold_size=min(test_len // 5, 36_000))

    print(f"train {len(train)} rows, val {len(val)} rows, test {len(test)} rows")

    train_env = FlattenObservation(gym.make("UniswapCLMM-v0", folds=train))
    val_env = FlattenObservation(gym.make("UniswapCLMM-v0", folds=[val]))

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
    bars = await from_ethereum(df)

    # print("loading data...")
    # df = pl.read_csv("../BTCUSDT-all-trades.csv")
    # bars = from_binance(df)

    train(bars)


if __name__ == "__main__":
    with logging_redirect_tqdm():
        try:
            asyncio.run(main())
        except Exception as e:
            l.exception("Fatal error during sync", exc_info=e)
            sys.exit(1)
