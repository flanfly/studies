import gymnasium as gym
import polars as pl
import polars_ols as pls
import numpy as np
import torch

from typing import Optional
from collections.abc import Iterable


class UniswapCLMM(gym.Env):

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
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
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
        self.fee_fraction = 0.01  # we own 1% of the pools active liquidity
        self.capital = 10_000
        self.reward_scale = 1.0

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
        mu = row["mu"] if row["mu"] is not None else s

        ret = {
            "price_deviation": np.array([(s - c) / c], dtype=np.float32),
            "distance_to_edge": np.array([(s - l) / (u - l) * 2 - 1], dtype=np.float32),
            "mean_deviation": np.array([(mu - s) / s], dtype=np.float32),
            "stein_signal": np.array(
                [row["theta"] if row["theta"] is not None else 0.0], dtype=np.float32
            ),
            "sigma": np.array(
                [row["sigma"] if row["sigma"] is not None else 0.0], dtype=np.float32
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
        # print(ret)
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
        reward = 0

        new_gas = 0
        new_fee = 0

        # move the position to the last price
        if action == 1:
            new_gas = (self.fee * self.capital / 2) + self.rebalance_cost
            self.center = self.price

        # trade was within our range
        if observation["in_range"]:
            new_fee += row["qty"] * row["price"] * self.fee_fraction * self.fee
            self.active_rows += 1

        reward += (new_fee - new_gas) / self.capital * self.reward_scale

        # Advance to next state (t+1)
        self.episode = self.episode[1:]
        terminated = self.episode.height == 0

        if terminated:
            next_observation = observation
        else:
            next_observation = self._observe()

        # update counters
        self.fees += new_fee
        self.gas += new_gas
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_floor)
        self.price = row["price"]

        return next_observation, reward, terminated, False, self._info()


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
        self.terminated = np.empty((capacity, 1), dtype=np.bool)

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
                min_periods=1,
                add_intercept=True,
                mode="coefficients",
            ),
        )
        .unnest("coefs", separator="_")
        .rename({"coefs_const": "alpha", "coefs_price": "beta"})
        .with_columns(
            ts=pl.col("ts"),
            theta=-pl.col("beta"),
            mu=pl.when(pl.col("beta") < 0)
            .then(-pl.col("alpha") / pl.col("beta"))
            .otherwise(pl.col("price")),
            sigma=(
                pl.col("diff") - (pl.col("alpha") + pl.col("price") * pl.col("beta"))
            ),
            vol=(
                (pl.col("price").shift(1) / pl.col("price"))
                .log()
                .rolling_std(300, min_samples=1)
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
        .drop_nulls()
    )


gym.register(
    id="UniswapCLMM-v0",
    entry_point=UniswapCLMM,
)


def train():
    from tqdm import tqdm
    from math import floor
    import matplotlib.pyplot as plt
    from torch import nn
    from torch import optim
    from random import sample
    from gymnasium.wrappers import FlattenObservation
    from copy import deepcopy

    print("loading data...")
    df = pl.read_csv("../BTCUSDT-all-trades.csv")
    folds = list(
        [from_binance(df) for df in make_folds(df)],
    )

    train_len = floor(len(folds) * 0.7)
    val_len = floor(len(folds) * 0.15)
    test_len = len(folds) - train_len - val_len
    train, val, test = (
        folds[:train_len],
        folds[train_len : train_len + val_len],
        folds[train_len + val_len :],
    )

    print(f"train {len(train)} rows, val {len(val)} rows, test {len(test)} rows")

    env = FlattenObservation(gym.make("UniswapCLMM-v0", folds=folds))

    # epsilon = 1.0

    policy = nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, env.action_space.n),
    )

    opt = optim.SGD(policy.parameters(), lr=0.0001)
    loss = nn.MSELoss()

    target = deepcopy(policy)
    target.load_state_dict(policy.state_dict())
    target.eval()

    frag = []
    replay_memory = Experience(100_000, env.observation_space.shape[0])
    steps = 0

    gamma = 0.99
    epsilon = 1.0
    epsilon_floor = 0.05
    epsilon_decay = 0.9998
    episode_loss = None

    for episode in tqdm(range(len(train)), desc="training"):
        observation, info = env.reset()

        episode_over = False
        total_reward = 0

        while not episode_over:
            if torch.rand(()).item() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = (
                        policy.forward(torch.tensor(observation, dtype=torch.float32))
                        .argmax(dim=-1)
                        .item()
                    )

            next_observation, reward, terminated, truncated, info = env.step(action)

            replay_memory.add(observation, action, reward, next_observation, terminated)
            recall = replay_memory.sample(128)

            y = recall[2]  # rewards

            # y = r + gamma * max_a'[ Q(observation_{t+1}, a', w-) ]
            with torch.no_grad():
                q = torch.max(target.forward(recall[3]), dim=1).values
            y += gamma * q.unsqueeze(-1) * (1 - recall[4].float())  # terminated

            opt.zero_grad()

            # (y - Q(observation_t, a, w))^2
            error = loss(policy.forward(recall[0]).gather(1, recall[1]), y)
            error.backward()

            episode_loss = error.mean()

            opt.step()

            if steps % 100 == 0:
                target.load_state_dict(policy.state_dict())

            total_reward += reward
            episode_over = terminated or truncated
            steps += 1

        epsilon = max(epsilon * epsilon_decay, epsilon_floor)
        frag.append(
            {
                "episode": episode,
                "fees": info["fees"],
                "gas": info["gas"],
                "loss": episode_loss.item() if episode_loss is not None else 0.0,
                "pnl": info["fees"] - info["gas"],
                "reward": total_reward,
            }
        )

        print(
            f"episode {frag[-1]['episode']}: fees={frag[-1]['fees']:.2f}, gas={frag[-1]['gas']:.2f}, pnl={frag[-1]['pnl']:.2f}"
        )

    env.close()

    (
        pl.DataFrame(frag)
        .sort("episode")
        .to_pandas()
        .plot(x="episode", y=["pnl", "loss"], secondary_y=["loss"])
    )

    plt.show()


if __name__ == "__main__":
    train()
