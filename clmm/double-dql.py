import gymnasium as gym
import polars as pl
import polars_ols as pls

from typing import Optional


class UniswapCLMM(gym.Env):

    def __init__(self, initial_price: float, trades: pl.DataFrame):
        # mu, theta and vol are backwards looking and don't include the row's own price
        assert all(
            column in trades.columns
            for column in ["price", "qty", "mu", "theta", "vol"]
        )
        assert trades.height > 0

        self.trades = trades
        self.action_space = gym.spaces.Discrete(n=2)  # 0: do nothing, 1: rebalance
        self.observation_space = gym.spaces.Dict(
            {
                "price_deviation": gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=float
                ),
                "distance_to_edge": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=float
                ),
                "stein_signal": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=float
                ),
                "mean_deviation": gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=float
                ),
                "theta": gym.spaces.Box(low=0.0, high=0.1, shape=(1,), dtype=float),
                "active_fraction": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=float
                ),
                "recent_volatility": gym.spaces.Box(
                    low=0.0, high=0.1, shape=(1,), dtype=float
                ),
                "in_range": gym.spaces.Discrete(n=2),  # 0: False, 1: True
            }
        )

        self.width = 0.01  # 1%
        self.fee = 0.0005  # 0.05%
        self.rebalance_cost = 2  # $2
        self.fee_fraction = 0.01  # we own 1% of the pools active liquidity
        self.initial_price = initial_price
        self.capital = 10_000

        self.epsilon_floor = 0.05
        self.epsilon_decay = 0.9998

    def _observe(self):

        row = self.progress.head(1).first()
        assert row is not None

        s = row["price"]
        c = self.center
        u = self.center * 1 + self.width
        mu = row["mu"]

        return {
            "price_deviation": s / (self.center - 1),
            "distance_to_edge": (s - c) / (u - c),
            "stein_signal": row["stein"],
            "mean_deviation": (mu - s) / s,
            "theta": row["theta"],
            "active_fraction": self.active_rows / self.progress.height,
            "recent_volatility": row["vol"],
            "in_range": int(s >= c - self.width and s <= c + self.width),
        }

    def _info(self):

        return {
            "fees": self.fees,
            "gas": self.gas,
            "epsilon": self.epsilon,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)

        self.current_step = self.trades["ts"].first()
        self.center = self.trades["price"].first()
        self.active_rows = 0
        self.last_price = self.initial_price
        self.fees = 0
        self.gas = 0
        self.progress = self.trades

        return self._observe(), self.info()

    def step(self, action):

        # rebalance?
        new_gas = 0
        if action == 1:
            new_gas = self.rebalance_cost
            self.center = self.last_price

        observation, row = self._observe()
        info = self._info()

        # advance one row (trade)
        self.progress = self.progress[1:]
        terminated = self.progress.height == 0

        new_fee = (
            row["qty"] * row["price"] * self.fee_fraction * self.fee * self.capital
        )

        reward = (
            new_fees - new_gas
        ) / self.initial_capital * self.reward_scale + self.epsilon * observation[
            "in_range"
        ]

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_floor)
        self.active_rows += observation["in_range"]
        self.fees += new_fee
        self.gas += new_gas

        return observation, reward, terminated, False, info


gym.register(
    id="UniswapCLMM-v0",
    entry_point=UniswapCLMM,
)


def train():
    df = (
        pl.scan_csv("../BTCUSDT-all-trades.csv")
        .head(10_000)
        .select(
            ts=(pl.col("time") * 1000).cast(pl.Datetime),
            price=pl.col("price"),
            qty=pl.col("qty"),
        )
        .group_by("ts", maintain_order=True)
        .agg(
            price=pl.col("price").last(),
            qty=pl.col("qty").sum(),
        )
        .with_columns(
            diff=pl.col("price").shift(-1) - pl.col("price"),
        )
        .drop_nulls()
        .rolling(
            index_column="ts",
            period="1800s",
            closed="right",
        )
        .agg(
            coefs=pl.col("diff").least_squares.ols(
                "price",
                add_intercept=True,
                mode="coefficients",
            ),
            price=pl.col("price").last(),
            diff=pl.col("diff").last(),
            qty=pl.col("qty").sum(),
        )
        .unnest("coefs", separator="_")
        .rename({"coefs_const": "alpha", "coefs_price": "beta"})
        .with_columns(
            ts=pl.col("ts"),
            theta=-pl.col("price") / 1800,
            mu=-pl.col("alpha") / pl.col("price"),
        )
        .select(
            pl.col("ts"),
            pl.col("price"),
            pl.col("qty"),
            pl.col("mu").shift(1),
            pl.col("theta").shift(1),
        )
        .rolling(
            index_column="ts",
            period="300s",
            closed="right",
        )
        .agg(
            price=pl.col("price").last(),
            qty=pl.col("qty").last(),
            mu=pl.col("mu").last(),
            theta=pl.col("theta").last(),
            vol=(pl.col("price").shift(1) / pl.col("price"))
            .log()
            .std()
            .sqrt()
            .clip(0, 0.1),
        )
    )

    print(df.head().collect())

    env = gym.make("UniswapCLMM-v0", initial_price=60_000, trades=df.collect())

    observation, info = env.reset()
    print(f"Starting observation: {observation}")

    episode_over = False
    total_reward = 0

    while not episode_over:
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    train()
