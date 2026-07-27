import polars as pl
import polars_ols as pls

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
import torch
from torch import nn

from tqdm import tqdm
import asyncio
from sgqlc.endpoint.base import BaseEndpoint
from httpx import AsyncClient

from copy import deepcopy

from typing import Optional, Dict, Any, Literal, Tuple
from dataclasses import dataclass, replace

from uniswapv3.emulator import Tick, Emulator
from uniswapv3.load import Pool, make_endpoint, from_ethereum, pool_meta
import uniswapv3.math as v3math

from cli.token import parse_amount, composition_in_range


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


def liquidity_for_value(
    sqrt_price_x96: int, tick_lower: int, tick_upper: int, capital: int
) -> int:
    a0, a1 = composition_in_range(sqrt_price_x96, tick_lower, tick_upper)
    a = a1 + a0 * (sqrt_price_x96**2) // (1 << 192)
    return capital * v3math.Q96 // a


class Rammstein(gym.Env):
    """Implement Uniswap liquidity ticks and fees correctly"""

    def __init__(
        self,
        position_width: int,  # in tick_spacing
        tick_spacing: int,
        rebalance_cost: int,  # in token1
        capital: int,  # in token1
        fee_pips: int,
        folds: list[pl.DataFrame],
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
        self.tick_spacing = tick_spacing
        self.initial_capital = capital
        self.folds = folds
        self.fee_pips = fee_pips

        self.epsilon_floor = 0.05
        self.epsilon_decay = 0.9998
        self.reward_scale = 100.0
        self.active_bonus = 1e-4

        # changed each reset
        self.fold_index: None | int = None
        self.fold = None
        self.capital = 0

        # changed by _reposition
        self.liquidity = 0
        self.tick_lower: int | None = None
        self.tick_upper: int | None = None
        self.init0: int | None = None
        self.init1: int | None = None

        # changed each step
        self.gas = 0
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
                "in_range": gym.spaces.Discrete(
                    n=2, dtype=np.int8
                ),  # 0: False, 1: True
            }
        )

        self.action_space = gym.spaces.Discrete(n=2)  # 0: do nothing, 1: rebalance

    def _observe(self):
        assert self.fold is not None

        row = self.fold.row(0, named=True)
        sqrt_price_x96 = int(row["sqrt_price_x96"])
        price = (sqrt_price_x96**2) // (1 << 192)

        lower_price = (v3math.get_sqrt_ratio_at_tick(self.tick_lower) / (2**96)) ** 2
        upper_price = (v3math.get_sqrt_ratio_at_tick(self.tick_upper) / (2**96)) ** 2
        center_price = lower_price + (upper_price - lower_price) / 2
        in_range = lower_price <= price < upper_price

        def col(n: str, row) -> np.ndarray:
            v = row[n]
            v = v if v is not None and not np.isnan(v) else 0.0
            return np.array([v], dtype=np.float32)

        mu = col("mu", row)
        numblks = self.folds[self.fold_index].height
        remblks = self.fold.height
        active = self.active_rows / (numblks - remblks) if numblks != remblks else 0.0

        pdevi = (price - center_price) / center_price
        dist = (price - lower_price) / (upper_price - lower_price) * 2 - 1
        mdevi = (mu - price) / price

        ret = {
            "price_deviation": np.array([pdevi], dtype=np.float32),
            "distance_to_edge": np.array([dist], dtype=np.float32),
            "mean_deviation": np.array([mdevi], dtype=np.float32),
            "active_fraction": np.array([active], dtype=np.float32),
            "in_range": np.int8(in_range),
            "stein_signal": col("theta", row),
            "sigma": col("sigma", row),
            "recent_volatility": col("vol", row),
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
            ret[k] = (
                np.nan_to_num(ret[k], nan=0.0, posinf=l[1], neginf=l[0])
                .clip(*l)
                .astype(np.float32)
            )

        return ret

    def _info(self):

        return {
            "fees": self.fee,
            "il": self.impermanent_loss,
            "gas": self.gas,
            "epsilon": self.epsilon,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.fold_index = (1 + (self.fold_index or 0)) % len(self.folds)
        self.fold = self.folds[self.fold_index]
        self.gas = 0
        self.active_rows = 0
        self.epsilon = 1.0
        self.impermanent_loss = 0
        self.fee = 0
        self.capital = self.initial_capital

        assert self.fold is not None
        row = self.fold.row(0, named=True)
        self._reposition(int(row["sqrt_price_x96"]))

        return self._observe(), self._info()

    def _reposition(self, sqrt_price_x96: int) -> int:
        tick = v3math.get_tick_at_sqrt_ratio(sqrt_price_x96)
        price = (sqrt_price_x96**2) // (1 << 192)
        l = self.liquidity

        if self.tick_lower is not None and self.tick_upper is not None:
            sqrtp_lower = v3math.get_sqrt_ratio_at_tick(self.tick_lower)
            sqrtp_upper = v3math.get_sqrt_ratio_at_tick(self.tick_upper)
            assert sqrtp_lower < sqrtp_upper

            if sqrt_price_x96 <= sqrtp_lower:
                end0 = v3math.get_amount0_delta(sqrtp_lower, sqrtp_upper, l, False)
                end1 = 0
            elif sqrt_price_x96 >= sqrtp_upper:
                end0 = 0
                end1 = v3math.get_amount1_delta(sqrtp_lower, sqrtp_upper, l, False)
            else:
                end0 = v3math.get_amount0_delta(sqrt_price_x96, sqrtp_upper, l, False)
                end1 = v3math.get_amount1_delta(sqrtp_lower, sqrt_price_x96, l, False)
        else:
            end0, end1 = 0, 0

        # compute il and update capital
        pos_end = int(end1 + end0 * price)
        pnl = self.capital - pos_end
        self.capital = pos_end

        # new position limits
        tsp = self.tick_spacing
        tick_lower = max(
            Emulator.MIN_TICK,
            tick - max(0, (self.position_width - 1) * tsp),
        )
        tick_lower = (tick_lower // tsp) * tsp
        tick_upper = min(Emulator.MAX_TICK, tick + (self.position_width * tsp))
        tick_upper = ((tick_upper + tsp - 1) // tsp) * tsp
        assert Emulator.MIN_TICK <= tick_lower < tick_upper <= Emulator.MAX_TICK

        sqrtp_lower = v3math.get_sqrt_ratio_at_tick(tick_lower)
        sqrtp_upper = v3math.get_sqrt_ratio_at_tick(tick_upper)
        assert sqrtp_lower <= sqrt_price_x96 <= sqrtp_upper

        # set new position
        self.tick_lower = tick_lower
        self.tick_upper = tick_upper

        self.liquidity = liquidity_for_value(
            sqrt_price_x96, tick_lower, tick_upper, self.capital
        )

        return pnl

    def step(self, action):
        assert self.fold_index is not None

        new_gas = 0
        new_il = 0
        reward = 0
        new_fees = 0

        observation = self._observe()
        row = self.fold.row(0, named=True)

        sqrt_price_x96 = int(row["sqrt_price_x96"])
        tick = v3math.get_tick_at_sqrt_ratio(sqrt_price_x96)
        in_range = self.tick_lower <= tick < self.tick_upper
        price = (sqrt_price_x96**2) // (1 << 192)

        fee_growth0 = float(row["fee_growth0"] or 0.0)
        fee_growth1 = float(row["fee_growth1"] or 0.0)

        if in_range:
            new_fees = fee_growth1 + fee_growth0 * price * self.liquidity
        self.fee += new_fees

        # rebalance
        if action == 1:
            new_il = self._reposition(sqrt_price_x96)

            # add swap fees
            swap_cost = 0.5 * self.capital * self.fee_pips / 1_000_000
            new_gas += self.rebalance_cost + swap_cost

        # trade was within our range
        if in_range:
            self.active_rows += 1
            # Eq. 28 active bonus
            reward += self.active_bonus

        # Eq. 28 less active bonus
        reward += (new_fees - new_gas) * self.reward_scale

        # Advance to next state (t+1)
        self.fold = self.fold.slice(1)
        truncated = self.fold.height == 0

        if truncated:
            next_observation = observation
        else:
            next_observation = self._observe()

        # update counters
        self.gas += new_gas
        self.impermanent_loss += new_il
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


def make_folds(df: pl.DataFrame, fold_size: int) -> list[pl.DataFrame]:
    from random import shuffle
    from math import ceil

    num_batches = ceil(df.height / fold_size)
    batches = list(
        [
            df.slice(i * fold_size, fold_size)
            for i in range(num_batches)
            if (i + 1) * fold_size <= df.height
        ]
    )

    shuffle(batches)
    return batches


gym.register(
    id="Rammstein-v1",
    entry_point=Rammstein,
)


def train_agent(
    train: list[pl.DataFrame],
    val: list[pl.DataFrame],
    capital: int,
    rebalance: int,
    position_width: int,
    tick_spacing: int,
    fee_pips: int,
) -> tuple[nn.Module, pl.DataFrame]:
    import matplotlib.pyplot as plt
    from torch import optim

    args = {
        "position_width": position_width,
        "tick_spacing": tick_spacing,
        "fee_pips": fee_pips,
        "rebalance_cost": rebalance,
        "capital": capital,
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
        if episode > 0 and (episode % 100 == 0 or episode + 1 == len(train)):
            valdf = evaluate_agent(
                val,
                policy,
                capital,
                rebalance,
                position_width,
                tick_spacing,
                fee_pips,
                val_env,
            )
            frag.append(
                {
                    "episode": episode,
                    "fees": info["fees"],
                    "gas": info["gas"],
                    "il": info["il"],
                    "loss": np.mean(episode_loss) if episode_loss else 0.0,
                    "pnl": (info["fees"] - info["gas"] - info["il"]),
                    "val_pnl": valdf["pnl"].sum(),
                    "val_rebalances": valdf["rebalances"].sum(),
                    "reward": total_reward,
                }
            )

            l.info(
                f"episode {frag[-1]['episode']}: loss={frag[-1]['loss']:.5f}, train_pnl={frag[-1]['pnl']:.2f}, val_pnl={frag[-1]['val_pnl']:.2f}, val_rebalances={frag[-1]['val_rebalances']}"
            )

    train_env.close()
    return policy, pl.DataFrame(frag)


def evaluate_agent(
    folds: list[pl.DataFrame],
    policy: nn.Module,
    capital: int,
    rebalance: int,
    position_width: int,
    tick_spacing: int,
    fee_pips: int,
    env: gym.Env | None = None,
) -> pl.DataFrame:
    args = {
        "position_width": position_width,
        "tick_spacing": tick_spacing,
        "fee_pips": fee_pips,
        "rebalance_cost": rebalance,
        "capital": capital,
    }

    if env is None:
        env = FlattenObservation(gym.make("Rammstein-v1", folds=folds, **args))
    frag = []

    # greedy rollout
    for episode in tqdm(range(len(folds)), desc="evaluate"):
        observation, _ = env.reset()
        num_rebalances = 0
        over = False

        while not over:
            with torch.no_grad():
                action = (
                    policy(torch.tensor(observation, dtype=torch.float32))
                    .argmax(dim=-1)
                    .item()
                )
            observation, _, terminated, truncated, info = env.step(action)
            over = terminated or truncated
            if action == 1:
                num_rebalances += 1

        frag.append(
            {
                "episode": episode,
                "fees": info["fees"],
                "gas": info["gas"],
                "il": info["il"],
                "pnl": (info["fees"] - info["gas"] - info["il"]),
                "rebalances": num_rebalances,
            }
        )

    return pl.DataFrame(
        frag,
        schema={
            "episode": pl.Int64,
            "fees": pl.Float64,
            "gas": pl.Float64,
            "il": pl.Float64,
            "pnl": pl.Float64,
            "rebalances": pl.Int64,
        },
    ).sort("episode")


async def main():
    from math import floor

    fold_len = 36_000  # 10h

    df = pl.read_parquet("data.parquet").sort("ts")

    train_len = floor(df.height * 0.7)
    val_len = floor(df.height * 0.003)
    test_len = df.height - train_len - val_len
    trainf, valf, testf = (
        make_folds(df.slice(0, train_len), fold_len),
        make_folds(df.slice(train_len, val_len), fold_len),
        make_folds(df.slice(train_len + val_len), fold_len),
    )

    l.info(f"train {len(trainf)} folds, val {len(valf)} folds, test {len(testf)} folds")

    # pool is usdc/weth -> price is weth/usdc
    capital = 10 * 10**6  # in quote currency
    rebalance = 0.02 * 10**6  # in quote currency
    tick_spacing = 1  # XXX
    fee_pips = 500
    position_width = 1

    policy, valdf = train_agent(
        trainf,
        valf,
        capital,
        rebalance,
        position_width,
        tick_spacing,
        fee_pips,
    )

    # import matplotlib.pyplot as plt
    # valdf.plot(x="episode", y="val_pnl", title=f"validation pnl {capital.symbol}")
    # plt.show()

    valdf.write_parquet("validation.parquet")
    torch.save(policy.state_dict(), "policy.pt")

    testdf = evaluate_agent(
        testf,
        policy,
        capital,
        rebalance,
        position_width,
        tick_spacing,
        fee_pips,
    )
    testdf.write_parquet("test.parquet")


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
