"""
RAmmStein: Regime Adaptation in Mean-reverting Markets with Stein thresholds.

Double DQN agent that decides when to rebalance a concentrated AMM liquidity
position, using OU process parameters as a regime indicator.

Reference:
    Anchuri, P. "RAmmStein: Regime Adaptation in Mean-reverting Markets with
    Stein thresholds." arXiv:2602.19419v2, 2026.

Required input schema (polars-loaded parquet, irregular trades):

    ts         : datetime / int64    trade timestamp (monotonic)
    price      : float64             trade price (or mid)
    cex_volume : float64             USD notional of the trade

The script first resamples irregular trades into 1-second bars via
polars `group_by_dynamic` (forward-filling price, summing USD volume),
then precomputes OU parameters (theta, mu, sigma) per bar with polars-ols
rolling OLS, then trains a DDQN agent on rolling-window episodes sliced
from the bar series, and finally evaluates the greedy policy on the
test split.

    ts        : datetime / int64         bar timestamp (monotonic, 1Hz)
    price     : float64                  mid price (S_t)
    volume_cex: float64                  CEX trade volume in the bar
                                          (DEX volume = volume_cex * DEX_CEX_RATIO)

The script precomputes (theta, mu, sigma) for every bar once via polars-ols
rolling OLS, then trains a DDQN agent on rolling-window episodes sliced from
the input series, and finally evaluates the greedy policy on the test split.

Install (not yet in pyproject.toml):

    uv pip install tf-agents tensorflow tf-keras tensorboard polars polars-ols

Run:

    uv run python rammstein.py \
        --trades data/eth_usd_1hz.parquet \
        --output results/rammstein_run

Note on numbers: the paper's evaluation fee/gas figures (Table III:
~$389 fees, ~$228 gas over the test split) depend on the CEX trade
volume per bar of the input dataset. The synthetic loader here uses
whatever `volume_cex` the parquet contains; if your data has small
volumes, fees may look tiny relative to gas. The agent's *decisions*
(when to rebalance) will be qualitatively similar regardless, since
the rewards scale uniformly.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import polars as pl
import polars_ols as polars_ols
import tensorflow as tf

# Keep Keras-2 (tf-keras); TF-Agents doesn't yet support Keras 3.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common


# =============================================================================
# Environment / pool parameters (Section V-B of the paper, Table II)
# =============================================================================

# AMM pool.
RANGE_WIDTH = 0.01              # 2w: position width as fraction of center (1%)
POOL_FEE_TIER = 0.0005          # phi: pool fee tier (5 bps)
POOL_TVL = 10_000_000.0         # L_total: pool TVL in USD
DEX_CEX_RATIO = 0.10            # alpha: DEX volume / CEX volume per bar
INITIAL_CAPITAL = 10_000.0      # K: LP capital deployed in USD

# Rebalancing cost (Section V-B, eq. 30): C = phi * 0.5 * K + G.
# We expose them separately and combine in env.step.
SWAP_FEE_FRACTION = 0.5         # share of position swapped on rebalance
GAS_COST = 2.0                  # G: gas cost per rebalance in USD

# Reward shaping (Section IV-D, eq. 28).
REWARD_SCALE = 100.0            # lambda: stabilises training
ACTIVE_BONUS = 0.01             # epsilon: per-step bonus for being in-range


# =============================================================================
# DDQN / training hyperparameters (Section IV-F, Table I)
# =============================================================================

LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.99          # gamma
REPLAY_BUFFER_SIZE = 100_000
BATCH_SIZE = 128
TARGET_UPDATE_PERIOD = 100      # hard update every N train steps
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9998          # multiplicative per training step

NUM_EPISODES = 150
EPISODE_LENGTH = 36_000         # 10 hours at 1Hz

EVAL_EPISODES = 10

# Train / val / test split (Section V-A).
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
# remaining is test.


# =============================================================================
# State / observation (Section IV-B)
# =============================================================================

STATE_DIM = 8
# Index of each component in the state vector s_t.
IDX_DELTA_P = 0     # normalised price deviation: S_t / c - 1
IDX_D_EDGE = 1      # distance to edge: (S_t - c) / (u - c)
IDX_THETA = 2       # Stein signal: mean-reversion speed, clipped to [0, 1]
IDX_DELTA_MU = 3    # mean deviation: (mu - S_t) / S_t
IDX_SIGMA = 4       # normalised sigma: sigma / S_t, clipped to 0.1
IDX_PHI_ACTIVE = 5  # active fraction over current episode so far
IDX_VOL = 6         # rolling realised volatility, clipped to 0.1
IDX_IN_RANGE = 7    # 1{S_t in [c(1-w), c(1+w)]}

# Normalisation constants from the paper.
THETA_CLIP = 1.0
SIGMA_CLIP = 0.1
VOL_CLIP = 0.1


# =============================================================================
# OU estimation (Section V-C)
# =============================================================================
#
# theta, mu, sigma are precomputed for every bar in the input series using
# polars-ols' rolling-least-squares, then looked up by bar index at env
# step time. This is O(1) per step instead of O(window) and keeps the inner
# loop free of any polars/numpy linalg work.
#
# The rolling OLS regresses dS_t = alpha + beta * S_t + eps on a sliding
# window of OU_WINDOW_SECONDS bars; eq. (31) then recovers the OU parameters
# (theta = -beta, mu = -alpha / beta, sigma = residual std). dt = 1 second
# since the bars are 1Hz.

OU_WINDOW_SECONDS = 1800        # rolling OLS window in seconds (1 bar/sec)


def compute_ou_params(
    df: pl.DataFrame, window: int = OU_WINDOW_SECONDS
) -> pl.DataFrame:
    """Augment a polars frame with rolling OLS-based (theta, mu, sigma).

    Input frame must contain a `price` column. Adds columns:
        d_price    : S_{t+1} - S_t
        alpha_hat  : rolling OLS intercept
        beta_hat   : rolling OLS slope on S_t
        resid      : d_price - alpha_hat - beta_hat * price
        theta      : max(0, -beta_hat)            (eq. 31, dt=1)
        mu         : -alpha_hat / beta_hat        (eq. 31)
        sigma      : rolling std of residuals     (eq. 31)

    The first ~`window` rows will be null; downstream code treats those as
    theta=0, mu=price, sigma=0 (no mean-reversion signal).
    """
    if "price" not in df.columns:
        raise ValueError("compute_ou_params needs a 'price' column")
    out = df.with_columns(
        (pl.col("price").shift(-1) - pl.col("price")).alias("d_price")
    )
    rolling_kwargs = polars_ols.RollingKwargs(
        window_size=window, min_periods=window
    )
    # `mode='coefficients'` returns a struct {const, price} with alpha and beta.
    # `add_intercept=True` is the right way to ask for the intercept column
    # (it's added implicitly as `const`).
    out = out.with_columns(
        polars_ols.compute_rolling_least_squares(
            out["d_price"],
            out["price"],
            add_intercept=True,
            mode="coefficients",
            rolling_kwargs=rolling_kwargs,
        ).alias("coefs")
    )
    out = out.with_columns(
        pl.col("coefs").struct.field("const").alias("alpha_hat"),
        pl.col("coefs").struct.field("price").alias("beta_hat"),
    ).drop("coefs")
    # Residual = y - (alpha + beta * x). With null_policy='drop' inside
    # polars-ols, the first `window` rows of alpha/beta are null; residuals
    # are therefore null there too.
    out = out.with_columns(
        (pl.col("d_price") - pl.col("alpha_hat") - pl.col("beta_hat") * pl.col("price")).alias("resid")
    )
    # Rolling std of residuals (ddof=0 to match the paper's eq. 13 OLS).
    # `min_samples=100` lets sigma fill in faster than the OLS itself, so
    # episodes starting just after t=window still have a usable sigma. The
    # paper starts episodes at random offsets within the data, so this only
    # affects a handful of episodes at the very beginning of the series.
    out = out.with_columns(
        pl.col("resid").pow(2).rolling_mean(window_size=window, min_samples=100).sqrt().alias("sigma")
    )
    out = out.with_columns(
        pl.max_horizontal(pl.lit(0.0), -pl.col("beta_hat")).alias("theta"),
        (-pl.col("alpha_hat") / pl.col("beta_hat")).alias("mu"),
    )
    # Fill nulls with neutral values so the env never sees NaN.
    out = out.with_columns(
        pl.col("theta").fill_null(0.0),
        pl.col("mu").fill_null(pl.col("price")),
        pl.col("sigma").fill_null(0.0),
    )
    return out


# =============================================================================
# Python environment (tf_agents.environments.py_environment.PyEnvironment)
# =============================================================================

@dataclass
class PoolConfig:
    """Constants derived from the pool + capital parameters."""
    range_width: float = RANGE_WIDTH              # 2w
    pool_fee_tier: float = POOL_FEE_TIER
    pool_tvl: float = POOL_TVL
    dex_cex_ratio: float = DEX_CEX_RATIO
    initial_capital: float = INITIAL_CAPITAL
    swap_fee_fraction: float = SWAP_FEE_FRACTION
    gas_cost: float = GAS_COST

    @property
    def swap_fee(self) -> float:
        """USD cost of the swap portion of a rebalance (eq. 30, second term)."""
        return self.pool_fee_tier * self.swap_fee_fraction * self.initial_capital

    @property
    def rebalance_cost(self) -> float:
        """Total cost C of one rebalance event."""
        return self.swap_fee + self.gas_cost

    @property
    def liquidity_share(self) -> float:
        """L_LP / L_total proxy for an equal-value deposit in [c(1-w), c(1+w)].

        For narrow ranges around c this is roughly proportional to
        K / (L_total * 2 * sqrt(w)). The absolute scale only matters for
        fee magnitude relative to gas; tune INITIAL_CAPITAL / POOL_TVL /
        RANGE_WIDTH if your fee numbers come out too small or too large.
        """
        return self.initial_capital / (
            self.pool_tvl * 2.0 * np.sqrt(self.range_width)
        )


class RammsteinEnv:
    """Concentrated-liquidity LP rebalancing environment.

    Observations: STATE_DIM-dimensional float vector as defined above.
    Actions:      0 = hold, 1 = rebalance.
    Rewards:      (delta_fees - delta_gas) / K * REWARD_SCALE
                  + ACTIVE_BONUS * 1{in_range}.
    """

    def __init__(
        self,
        price: np.ndarray,
        volume_cex: np.ndarray,
        theta: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        pool: PoolConfig | None = None,
        episode_length: int = EPISODE_LENGTH,
    ):
        assert len(price) == len(volume_cex) == len(theta) == len(mu) == len(sigma)
        assert len(price) > episode_length + 2
        self.price = price
        self.volume_cex = volume_cex
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.pool = pool or PoolConfig()
        self.episode_length = episode_length

        # Episode state.
        self._t = 0
        self._c = 0.0                    # position center
        self._in_range = 0
        self._t_in_range = 0             # seconds in-range so far in episode
        self._vol_window: list[float] = []   # rolling returns for realised vol
        self._prev_price = 0.0

    # ------------------------------------------------------------------ spec
    @property
    def observation_spec(self):
        # BoundedArraySpec (so QNetwork can infer num_actions from the action
        # spec and so bounds are preserved through TFPyEnvironment wrapping).
        return tensor_spec.array_spec.BoundedArraySpec(
            shape=(STATE_DIM,),
            dtype=np.float32,
            minimum=-10.0,
            maximum=10.0,
            name="observation",
        )

    @property
    def action_spec(self):
        return tensor_spec.array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name="action"
        )

    # ----------------------------------------------------------- helpers
    def _is_in_range(self, s: float) -> int:
        lower = self._c * (1.0 - self.pool.range_width)
        upper = self._c * (1.0 + self.pool.range_width)
        return int(lower <= s <= upper)

    def _build_state(self, s: float, theta: float, mu: float, sigma: float) -> np.ndarray:
        d_edge = (s - self._c) / (self._c * self.pool.range_width) if self._c > 0 else 0.0
        delta_p = (s / self._c - 1.0) if self._c > 0 else 0.0
        delta_mu = (mu - s) / s if s > 0 else 0.0
        realised_vol = float(np.std(self._vol_window)) if len(self._vol_window) >= 2 else 0.0
        phi_active = self._t_in_range / max(self._t_in_episode(), 1)
        state = np.array(
            [
                delta_p,
                d_edge,
                min(theta, THETA_CLIP),
                delta_mu,
                min(abs(sigma / s) if s > 0 else 0.0, SIGMA_CLIP),
                phi_active,
                min(realised_vol, VOL_CLIP),
                float(self._in_range),
            ],
            dtype=np.float32,
        )
        # Clip to spec bounds (TF-Agents enforces them).
        return np.clip(state, -10.0, 10.0)

    def _t_in_episode(self) -> int:
        """Steps elapsed in the current episode (used for active fraction)."""
        # We track this indirectly: it's the number of times we've called
        # _step() since reset(). Track via a counter.
        return self._step_count

    # ------------------------------------------------------------- API
    def reset(self) -> tuple[np.ndarray, dict]:
        # Random start: leave enough headroom for OU warmup + episode.
        max_start = len(self.price) - self.episode_length - 2
        min_start = OU_WINDOW_SECONDS + 1
        self._t = np.random.randint(min_start, max_start)

        s = float(self.price[self._t])
        self._c = s
        self._in_range = 1
        self._t_in_range = 0
        self._step_count = 0
        self._vol_window = []
        self._prev_price = s

        theta = float(self.theta[self._t])
        mu = float(self.mu[self._t])
        sigma = float(self.sigma[self._t])
        obs = self._build_state(s, theta, mu, sigma)
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        s = float(self.price[self._t])
        s_next = float(self.price[self._t + 1])
        v_dex = self.pool.dex_cex_ratio * float(self.volume_cex[self._t])

        # Fee accrual: eq. (2) multiplied by the concentration proxy.
        if self._in_range:
            self._t_in_range += 1
            fee_t = v_dex * self.pool.pool_fee_tier * self.pool.liquidity_share
        else:
            fee_t = 0.0

        # Action.
        gas_t = 0.0
        if action == 1:
            self._c = s_next
            gas_t = self.pool.rebalance_cost

        # Reward: eq. (28).
        net_pnl = (fee_t - gas_t) / self.pool.initial_capital
        reward = net_pnl * REWARD_SCALE + ACTIVE_BONUS * self._in_range

        # Move forward.
        if self._prev_price > 0:
            ret = (s_next - self._prev_price) / self._prev_price
            self._vol_window.append(ret)
            if len(self._vol_window) > 300:
                self._vol_window.pop(0)
        self._prev_price = s_next
        self._step_count += 1
        self._t += 1
        self._in_range = self._is_in_range(s_next)

        theta = float(self.theta[self._t])
        mu = float(self.mu[self._t])
        sigma = float(self.sigma[self._t])
        obs = self._build_state(s_next, theta, mu, sigma)

        terminated = self._step_count >= self.episode_length
        return obs, float(reward), terminated, False, {}

    def close(self) -> None:
        pass


class PyRammsteinEnv(py_environment.PyEnvironment):
    """TF-Agents PyEnvironment wrapper around RammsteinEnv."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._env = RammsteinEnv(*args, **kwargs)

    def action_spec(self):
        return self._env.action_spec

    def observation_spec(self):
        return self._env.observation_spec

    def _reset(self):
        obs, _ = self._env.reset()
        return ts.restart(obs.astype(np.float32))

    def _step(self, action):
        obs, reward, terminated, _, _ = self._env.step(action)
        if terminated:
            return ts.termination(obs.astype(np.float32), reward)
        return ts.transition(obs.astype(np.float32), reward, discount=1.0)

    def close(self):
        self._env.close()


# =============================================================================
# Q-network + agent
# =============================================================================

def build_agent(train_env: tf_py_environment.TFPyEnvironment):
    # After TFPyEnvironment wrapping, observation_spec() returns the full
    # array spec, not a nested spec. Action spec is the same shape.
    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=(128, 64),
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # int64 step counter; tf-agents complains if it's int32.
    train_step_counter = tf.Variable(0, dtype=tf.int64)
    agent = dqn_agent.DdqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
        target_update_period=TARGET_UPDATE_PERIOD,
        gamma=DISCOUNT_FACTOR,
        epsilon_greedy=lambda: EPSILON_END
        + (EPSILON_START - EPSILON_END) * EPSILON_DECAY ** int(train_step_counter),
        # Skip TensorBoard summaries (we're not running TB here).
        debug_summaries=False,
        summarize_grads_and_vars=False,
    )
    agent.initialize()
    return agent, q_net


# =============================================================================
# Training loop
# =============================================================================


def train(
    price: np.ndarray,
    volume_cex: np.ndarray,
    theta: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    log_dir: str,
    num_episodes: int = NUM_EPISODES,
) -> tuple[dqn_agent.DdqnAgent, q_network.QNetwork]:
    train_py_env = PyRammsteinEnv(price, volume_cex, theta, mu, sigma)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)

    agent, q_net = build_agent(train_env)

    random_policy = random_tf_policy.RandomTFPolicy(
        train_env.time_step_spec(), train_env.action_spec()
    )
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=REPLAY_BUFFER_SIZE,
    )

    # Warm up the buffer with random actions.
    print("[rammstein] Filling replay buffer with random actions...")
    init_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        random_policy,
        observers=[replay_buffer.add_batch],
        num_steps=min(BATCH_SIZE * 4, 5_000),
    )
    init_driver.run()

    dataset = replay_buffer.as_dataset(
        sample_batch_size=BATCH_SIZE, num_steps=2, num_parallel_calls=3
    ).prefetch(3)
    iterator = iter(dataset)

    print(
        f"[rammstein] Training DDQN for {num_episodes} episodes, "
        f"{EPISODE_LENGTH} steps each..."
    )
    t0 = time.time()
    for ep in range(num_episodes):
        driver = dynamic_step_driver.DynamicStepDriver(
            train_env,
            agent.collect_policy,
            observers=[replay_buffer.add_batch],
            num_steps=EPISODE_LENGTH,
        )
        driver.run(train_env.reset())
        # Train once per collected step (matches Table I).
        for _ in range(EPISODE_LENGTH):
            experience, _ = next(iterator)
            agent.train(experience)
        if (ep + 1) % 5 == 0:
            print(
                f"  episode {ep + 1}/{num_episodes} "
                f"elapsed={time.time() - t0:.1f}s"
            )

    os.makedirs(log_dir, exist_ok=True)
    # QNetwork doesn't expose save_weights; persist weights as numpy arrays.
    weights_path = os.path.join(log_dir, "q_network_weights.npz")
    weights = q_net.get_weights()
    np.savez(weights_path, *weights)
    print(f"[rammstein] Saved Q-network weights to {weights_path}")
    return agent, q_net


# =============================================================================
# Evaluation (greedy rollout, mirroring Section VIII)
# =============================================================================


def evaluate(
    price: np.ndarray,
    volume_cex: np.ndarray,
    theta: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    q_net: q_network.QNetwork,
    pool: PoolConfig,
    n_episodes: int = EVAL_EPISODES,
    seed: int = 0,
) -> dict:
    """Greedy evaluation: run n_episodes end-to-end and report paper metrics."""
    np.random.seed(seed)
    active_fracs, rebal_counts, fees_tot, gas_tot = [], [], [], []

    for _ in range(n_episodes):
        env = RammsteinEnv(price, volume_cex, theta, mu, sigma, pool=pool)
        env.reset()
        rebalances = 0
        fees = 0.0
        gas = 0.0
        in_range_steps = 0

        for _ in range(EPISODE_LENGTH):
            s = float(env.price[env._t])
            obs = env._build_state(
                s,
                float(env.theta[env._t]),
                float(env.mu[env._t]),
                float(env.sigma[env._t]),
            )
            q_values, _ = q_net(obs[None, ...], step_type=ts.StepType.MID)
            action = int(tf.argmax(q_values[0]).numpy())

            # Compute fee for the bar we're about to act on (env.step uses
            # volume_cex at the current bar to compute fee, then advances t).
            v_dex = pool.dex_cex_ratio * float(env.volume_cex[env._t])
            if env._in_range:
                fees += v_dex * pool.pool_fee_tier * pool.liquidity_share
                in_range_steps += 1
            if action == 1:
                rebalances += 1
                gas += pool.rebalance_cost

            _, _, terminated, _, _ = env.step(action)
            if terminated:
                break

        active_fracs.append(in_range_steps / EPISODE_LENGTH)
        rebal_counts.append(rebalances)
        fees_tot.append(fees)
        gas_tot.append(gas)

    fees_tot = np.array(fees_tot)
    gas_tot = np.array(gas_tot)
    return {
        "active_pct": float(np.mean(active_fracs) * 100),
        "rebalances": int(np.sum(rebal_counts)),
        "fees_usd": float(fees_tot.sum()),
        "gas_usd": float(gas_tot.sum()),
        "net_pnl_usd": float((fees_tot - gas_tot).sum()),
        "net_roi_pct": float(
            (fees_tot - gas_tot).sum() / (pool.initial_capital * n_episodes) * 100
        ),
        "fee_to_gas": float(fees_tot.sum() / max(gas_tot.sum(), 1e-9)),
    }


# =============================================================================
# Polars input
# =============================================================================
#
# Expected CSV format: Binance historical trades dump, one row per fill.
# No header; seven columns in this order:
#     trade_id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch
# `time` is microseconds since epoch (UTC). `quoteQty` is USDT notional,
# which we use directly as `cex_volume`.
#
# Schema reference (parquet file format):
#     ts         : datetime    trade timestamp (UTC)
#     price      : float64     trade price
#     cex_volume : float64     USDT notional of the trade
#
# Either source ends up as (ts, price, cex_volume) and is then resampled
# into 1-second bars by `resample_to_1hz`.

BINANCE_CSV_COLUMNS = [
    "trade_id",
    "price",
    "qty",
    "quoteQty",
    "time",
    "isBuyerMaker",
    "isBestMatch",
]


def _load_binance_csv(path: str) -> pl.DataFrame:
    """Read a Binance historical-trades CSV into (ts, price, cex_volume)."""
    return (
        pl.read_csv(
            path,
            has_header=False,
            new_columns=BINANCE_CSV_COLUMNS,
        )
        .select(
            ts=pl.col("time").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
            price=pl.col("price"),
            cex_volume=pl.col("quoteQty"),
        )
    )


def load_trades(path: str) -> pl.DataFrame:
    """Load irregular trades from a Binance historical-trades CSV.

    Output schema (pre-resample):
        ts         : datetime[UTC]
        price      : float64
        cex_volume : float64

    Resampling to 1-second bars is a separate step; see `resample_to_1hz`.
    """
    df = _load_binance_csv(path)
    df = df.sort("ts")
    # Drop rows where the price is null/NaN/<=0 (defensive; Binance feeds
    # shouldn't produce these, but a malformed CSV can).
    df = df.filter(pl.col("price").is_finite() & (pl.col("price") > 0))
    return df


def resample_to_1hz(df: pl.DataFrame) -> pl.DataFrame:
    """Resample irregular trades into 1-second OHLCV+-ish bars.

    For each 1-second bucket:
        price      = mean of trade prices in the bucket (forward-filled
                     across empty buckets; leading nulls are dropped)
        volume_cex = sum of cex_volume (USD) in the bucket; empty buckets
                     are 0.0 by construction

    Output schema:
        ts         : datetime[UTC]   one row per second
        price      : float64         forward-filled
        volume_cex : float64         per-second USD volume
    """
    out = (
        df.group_by_dynamic("ts", every="1s")
        .agg(
            pl.col("price").mean().alias("price"),
            pl.col("cex_volume").sum().alias("volume_cex"),
        )
        .with_columns(
            pl.col("price").forward_fill(),
            pl.col("volume_cex").fill_null(0.0),
        )
    )
    # Drop leading seconds that have no trade yet (nothing to ffill from).
    out = out.filter(pl.col("price").is_not_null())
    return out


def prepare_arrays(
    df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample to 1Hz, compute OU params, and return numpy arrays.

    Returns (price, volume_cex, theta, mu, sigma), all float64, length n.
    """
    df = resample_to_1hz(df)
    df = compute_ou_params(df)
    return (
        df["price"].to_numpy().astype(np.float64),
        df["volume_cex"].to_numpy().astype(np.float64),
        df["theta"].to_numpy().astype(np.float64),
        df["mu"].to_numpy().astype(np.float64),
        df["sigma"].to_numpy().astype(np.float64),
    )


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="RAmmStein DDQN trainer.")
    parser.add_argument(
        "--trades", required=True,
        help="Path to a Binance historical-trades CSV (no header, 7 cols: "
             "trade_id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch)."
    )
    parser.add_argument("--output", default="results/rammstein", help="Output dir.")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    args = parser.parse_args()

    print(f"[rammstein] Loading {args.trades}")
    df = load_trades(args.trades)
    print(f"[rammstein] Loaded {len(df):,} 1Hz bars; precomputing OU params...")
    price, volume_cex, theta, mu, sigma = prepare_arrays(df)
    print(f"[rammstein] OU params ready (median theta={float(np.median(theta)):.4f})")

    # Train / val / test split (Section V-A).
    n = len(price)
    n_train = int(n * TRAIN_FRACTION)
    n_test_start = int(n * (TRAIN_FRACTION + VAL_FRACTION))
    train_price, train_vol = price[:n_train], volume_cex[:n_train]
    train_theta, train_mu, train_sigma = theta[:n_train], mu[:n_train], sigma[:n_train]
    test_price, test_vol = price[n_test_start:], volume_cex[n_test_start:]
    test_theta, test_mu, test_sigma = (
        theta[n_test_start:],
        mu[n_test_start:],
        sigma[n_test_start:],
    )

    if args.mode == "train":
        agent, q_net = train(
            train_price,
            train_vol,
            train_theta,
            train_mu,
            train_sigma,
            log_dir=args.output,
            num_episodes=args.episodes,
        )
        print("[rammstein] Evaluating on held-out test split...")
        metrics = evaluate(
            test_price,
            test_vol,
            test_theta,
            test_mu,
            test_sigma,
            q_net,
            PoolConfig(),
            args.eval_episodes,
        )
    else:
        # Eval-only path: requires weights already saved to args.output.
        env = tf_py_environment.TFPyEnvironment(
            PyRammsteinEnv(train_price, train_vol, train_theta, train_mu, train_sigma)
        )
        q_net = q_network.QNetwork(
            env.observation_spec(),
            env.action_spec(),
            fc_layer_params=(128, 64),
        )
        # Dummy forward pass to instantiate variables before loading weights.
        q_net(
            tf.zeros((1, STATE_DIM), dtype=tf.float32),
            step_type=ts.StepType.MID,
        )
        weights_path = os.path.join(args.output, "q_network_weights.npz")
        with np.load(weights_path) as data:
            q_net.set_weights([data[k] for k in data.files])
        metrics = evaluate(
            test_price,
            test_vol,
            test_theta,
            test_mu,
            test_sigma,
            q_net,
            PoolConfig(),
            args.eval_episodes,
        )

    print("\n[rammstein] Headline metrics on test data:")
    for k, v in metrics.items():
        print(f"  {k:>14s} = {v}")


if __name__ == "__main__":
    main()
