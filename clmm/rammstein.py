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

    uv pip install stable-baselines3 torch gymnasium polars polars-ols

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

Implementation note: We use a thin :class:`DoubleDQN` subclass of SB3's
``DQN`` policy (PyTorch back-end). SB3's stock DQN is vanilla Deep
Q-Learning (the Bellman target uses ``max_a Q_target(s', a)``), which
re-introduces the max-overestimation bias that Double DQN was designed
to fix. To preserve the paper's DDQN formulation, we override
``train()`` so the next action is *selected* by the online Q-net and
*evaluated* by the target Q-net (Hasselt et al. 2016).
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from dataclasses import dataclass

import gymnasium
import numpy as np
import polars as pl
import polars_ols as polars_ols
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm.auto import tqdm

import torch as th
import torch.nn.functional as F

# =============================================================================
# Environment / pool parameters (Section V-B of the paper, Table II)
# =============================================================================

# AMM pool.
RANGE_WIDTH = 0.01  # 2w: position width as fraction of center (1%)
POOL_FEE_TIER = 0.0005  # phi: pool fee tier (5 bps)
POOL_TVL = 10_000_000.0  # L_total: pool TVL in USD
DEX_CEX_RATIO = 0.10  # alpha: DEX volume / CEX volume per bar
INITIAL_CAPITAL = 10_000.0  # K: LP capital deployed in USD

# Rebalancing cost (Section V-B, eq. 30): C = phi * 0.5 * K + G.
# We expose them separately and combine in env.step.
SWAP_FEE_FRACTION = 0.5  # share of position swapped on rebalance
GAS_COST = 2.0  # G: gas cost per rebalance in USD

# Reward shaping (Section IV-D, eq. 28).
REWARD_SCALE = 100.0  # lambda: stabilises training
ACTIVE_BONUS = 0.01  # epsilon: per-step bonus for being in-range


# =============================================================================
# DDQN / training hyperparameters (Section IV-F, Table I)
# =============================================================================

LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.99  # gamma
REPLAY_BUFFER_SIZE = 100_000
BATCH_SIZE = 128
TARGET_UPDATE_PERIOD = 100  # hard update every N train steps

NUM_EPISODES = 150
EPISODE_LENGTH = 36_000  # 10 hours at 1Hz

# Epsilon schedule: SB3's DQN uses a *linear* decay from EPSILON_START to
# EPSILON_END over ``exploration_fraction`` of the total timesteps. The
# paper uses an exponential decay (0.9998) which hits 0.05 around step 15,000.
# We match this timeframe by setting the linear fraction accordingly.
EPSILON_START = 1.0
EPSILON_END = 0.05
EXPLORATION_FRACTION = 15_000 / (NUM_EPISODES * EPISODE_LENGTH)  # Approx 0.0027

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
IDX_DELTA_P = 0  # normalised price deviation: S_t / c - 1
IDX_D_EDGE = 1  # distance to edge: (S_t - c) / (u - c)
IDX_THETA = 2  # Stein signal: mean-reversion speed, clipped to [0, 1]
IDX_DELTA_MU = 3  # mean deviation: (mu - S_t) / S_t
IDX_SIGMA = 4  # normalised sigma: sigma / S_t, clipped to 0.1
IDX_PHI_ACTIVE = 5  # active fraction over current episode so far
IDX_VOL = 6  # rolling realised volatility, clipped to 0.1
IDX_IN_RANGE = 7  # 1{S_t in [c(1-w), c(1+w)]}

# Normalisation constants from the paper.
THETA_CLIP = 1.0
SIGMA_CLIP = 0.1
VOL_CLIP = 0.1
STATE_BOUND = 10.0  # Box low/high for the observation space


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

OU_WINDOW_SECONDS = 1800  # rolling OLS window in seconds (1 bar/sec)


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
        (pl.col("price") - pl.col("price").shift(1)).alias("d_price"),
        pl.col("price").shift(1).alias("price_lag"),
    )
    rolling_kwargs = polars_ols.RollingKwargs(window_size=window, min_periods=window)
    # `mode='coefficients'` returns a struct {const, price} with alpha and beta.
    # `add_intercept=True` is the right way to ask for the intercept column
    # (it's added implicitly as `const`).
    out = out.with_columns(
        polars_ols.compute_rolling_least_squares(
            out["d_price"],
            out["price_lag"],
            add_intercept=True,
            mode="coefficients",
            rolling_kwargs=rolling_kwargs,
        ).alias("coefs")
    )
    out = out.with_columns(
        pl.col("coefs").struct.field("const").alias("alpha_hat"),
        pl.col("coefs").struct.field("price_lag").alias("beta_hat"),
    ).drop("coefs")
    # Residual = y - (alpha + beta * x). With null_policy='drop' inside
    # polars-ols, the first `window` rows of alpha/beta are null; residuals
    # are therefore null there too.
    out = out.with_columns(
        (
            pl.col("d_price")
            - pl.col("alpha_hat")
            - pl.col("beta_hat") * pl.col("price_lag")
        ).alias("resid")
    )
    # Rolling std of residuals (ddof=0 to match the paper's eq. 13 OLS).
    # `min_samples=100` lets sigma fill in faster than the OLS itself, so
    # episodes starting just after t=window still have a usable sigma. The
    # paper starts episodes at random offsets within the data, so this only
    # affects a handful of episodes at the very beginning of the series.
    out = out.with_columns(
        pl.col("resid")
        .pow(2)
        .rolling_mean(window_size=window, min_samples=100)
        .sqrt()
        .alias("sigma")
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
# Pool + env
# =============================================================================


@dataclass
class PoolConfig:
    """Constants derived from the pool + capital parameters."""

    range_width: float = RANGE_WIDTH  # 2w
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
        return self.initial_capital / (self.pool_tvl * 2.0 * np.sqrt(self.range_width))


class RammsteinEnv(gymnasium.Env):
    """Concentrated-liquidity LP rebalancing environment (gymnasium.Env).

    Observations: ``STATE_DIM``-dimensional float32 vector, bounded to
        ``[-STATE_BOUND, STATE_BOUND]`` for all components (see IDX_*).
    Actions:      ``Discrete(2)``  -- 0 = hold, 1 = rebalance.
    Rewards:      (delta_fees - delta_gas) / K * REWARD_SCALE
                  + ACTIVE_BONUS * 1{in_range}.

    The environment is stateless across ``reset()`` calls: each episode
    samples a random start index in the bar series (so long as there's
    enough headroom for the OU warmup window plus the episode length).
    """

    metadata = {"render_modes": []}

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
        super().__init__()
        assert len(price) == len(volume_cex) == len(theta) == len(mu) == len(sigma)
        assert len(price) > episode_length + OU_WINDOW_SECONDS + 2
        self.price = price
        self.volume_cex = volume_cex
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.pool = pool or PoolConfig()
        self.episode_length = episode_length

        # Gymnasium spaces.
        self.observation_space = spaces.Box(
            low=-STATE_BOUND,
            high=STATE_BOUND,
            shape=(STATE_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)

        # Episode state (re-initialised on every reset()).
        self._t = 0
        self._c = 0.0
        self._in_range = 0
        self._t_in_range = 0
        self._step_count = 0
        self._vol_window: deque[float] = deque(maxlen=300)
        self._prev_price = 0.0
        # Per-episode metrics, exposed via info dict.
        self._ep_fees = 0.0
        self._ep_gas = 0.0
        self._ep_rebalances = 0

    # ----------------------------------------------------------- helpers
    def _is_in_range(self, s: float) -> int:
        lower = self._c * (1.0 - self.pool.range_width)
        upper = self._c * (1.0 + self.pool.range_width)
        return int(lower <= s <= upper)

    def _build_state(
        self, s: float, theta: float, mu: float, sigma: float
    ) -> np.ndarray:
        d_edge = (
            (s - self._c) / (self._c * self.pool.range_width) if self._c > 0 else 0.0
        )
        delta_p = (s / self._c - 1.0) if self._c > 0 else 0.0
        delta_mu = (mu - s) / s if s > 0 else 0.0
        realised_vol = (
            float(np.std(self._vol_window)) if len(self._vol_window) >= 2 else 0.0
        )
        phi_active = self._t_in_range / max(self._step_count, 1)
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
        return np.clip(state, -STATE_BOUND, STATE_BOUND)

    # ----------------------------------------------------------- gym API
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Random start: leave enough headroom for OU warmup + episode.
        max_start = len(self.price) - self.episode_length - 2
        min_start = OU_WINDOW_SECONDS + 1
        self._t = int(self.np_random.integers(min_start, max_start))

        s = float(self.price[self._t])
        self._c = s
        self._in_range = 1
        self._t_in_range = 0
        self._step_count = 0
        self._vol_window.clear()
        self._prev_price = s
        self._ep_fees = 0.0
        self._ep_gas = 0.0
        self._ep_rebalances = 0

        theta = float(self.theta[self._t])
        mu = float(self.mu[self._t])
        sigma = float(self.sigma[self._t])
        obs = self._build_state(s, theta, mu, sigma)
        return obs, {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        s = float(self.price[self._t])
        s_next = float(self.price[self._t + 1])
        v_dex = self.pool.dex_cex_ratio * float(self.volume_cex[self._t])

        # Fee accrual: eq. (2) multiplied by the concentration proxy.
        fee_t = 0.0
        if self._in_range:
            self._t_in_range += 1
            fee_t = v_dex * self.pool.pool_fee_tier * self.pool.liquidity_share
        self._ep_fees += fee_t

        # Action.
        gas_t = 0.0
        if action == 1:
            self._c = s_next
            gas_t = self.pool.rebalance_cost
            self._ep_rebalances += 1
        self._ep_gas += gas_t

        # Reward: eq. (28).
        net_pnl = (fee_t - gas_t) / self.pool.initial_capital
        reward = net_pnl * REWARD_SCALE + ACTIVE_BONUS * self._in_range

        # Move forward.
        if self._prev_price > 0:
            ret = (s_next - self._prev_price) / self._prev_price
            self._vol_window.append(ret)
        self._prev_price = s_next
        self._step_count += 1
        self._t += 1
        self._in_range = self._is_in_range(s_next)

        theta = float(self.theta[self._t])
        mu = float(self.mu[self._t])
        sigma = float(self.sigma[self._t])
        obs = self._build_state(s_next, theta, mu, sigma)

        terminated = self._step_count >= self.episode_length
        info = {
            "fees": fee_t,
            "gas": gas_t,
            "in_range": self._in_range,
            "rebalances": int(action == 1),
            # Cumulative metrics, snapshotted on terminal step.
            "ep_fees": self._ep_fees,
            "ep_gas": self._ep_gas,
            "ep_rebalances": self._ep_rebalances,
            "ep_active_frac": self._t_in_range / max(self._step_count, 1),
        }
        return obs, float(reward), terminated, False, info

    def close(self) -> None:
        pass


# =============================================================================
# SB3 model + training
# =============================================================================


class DoubleDQN(DQN):
    """DDQN variant of SB3's DQN.

    SB3's stock ``DQN`` is vanilla Deep Q-Learning: the Bellman target uses
    ``max_a Q_target(s', a)``, which over-estimates Q-values whenever the
    target network's estimates are noisy. The original RAmmStein paper relies
    on Double DQN (Decouple action selection from action evaluation) to
    avoid this bias, so we override ``train()`` to compute the target as

        a*   = argmax_a Q_online(s', a)
        Q*   = Q_target(s', a*)
        tgt  = r + gamma * (1 - done) * Q*

    Everything else (replay sampling, Huber loss, gradient clipping,
    target soft / hard update) is inherited unchanged.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            with th.no_grad():
                # --- Double DQN target -----------------------------------
                # Use the *online* net to pick the next action.
                next_q_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                # Use the *target* net to evaluate the value of that action.
                next_q_target = self.q_net_target(replay_data.next_observations)
                next_q_values = th.gather(next_q_target, dim=1, index=next_actions)
                # 1-step TD target.
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )
                # ---------------------------------------------------------

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps


def make_vec_env(
    price: np.ndarray,
    volume_cex: np.ndarray,
    theta: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    seed: int = 0,
) -> DummyVecEnv:
    """Wrap RammsteinEnv in a single-process VecEnv for SB3.

    The env is wrapped with :class:`Monitor` so that the per-episode
    metrics we stash in ``info`` (``ep_fees``, ``ep_gas``,
    ``ep_rebalances``, ``ep_active_frac``) get aggregated into SB3's
    ``ep_info_buffer`` on every terminal step. The custom callback reads
    from that buffer to render progress-bar postfix values.
    """

    def _thunk():
        env = RammsteinEnv(price, volume_cex, theta, mu, sigma)
        # CRITICAL: Wrap with Monitor to populate ep_info_buffer
        env = Monitor(
            env,
            info_keywords=(
                "ep_fees",
                "ep_gas",
                "ep_rebalances",
                "ep_active_frac",
            ),
        )
        return env

    return DummyVecEnv([_thunk])


def build_model(
    vec_env: DummyVecEnv,
    log_dir: str,
) -> DoubleDQN:
    """Build the DDQN model with the paper's hyperparameters.

    The MLP architecture (128, 64) is paper-faithful (Table I, "Net size");
    the rest are the standard SB3 DQN hyperparameters and match Table I.

    We use :class:`DoubleDQN` (a thin subclass of ``DQN`` that overrides
    ``train()``) so that the Bellman target is computed with decoupled
    action selection (online net) and action evaluation (target net) --
    matching the paper's DDQN formulation.
    """
    policy_kwargs = dict(
        net_arch=[128, 64],
    )
    return DoubleDQN(
        policy="MlpPolicy",
        env=vec_env,
        # Force CPU: PCIe overhead for tiny tensors (8 floats) dominates on GPU.
        device="cpu",
        learning_rate=LEARNING_RATE,
        buffer_size=REPLAY_BUFFER_SIZE,
        learning_starts=BATCH_SIZE * 4,
        batch_size=BATCH_SIZE,
        gamma=DISCOUNT_FACTOR,
        target_update_interval=TARGET_UPDATE_PERIOD,
        train_freq=1,
        gradient_steps=1,
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_initial_eps=EPSILON_START,
        exploration_final_eps=EPSILON_END,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tb"),
        seed=0,
    )


class TqdmCallback(BaseCallback):
    """Update a tqdm bar with metrics from SB3's ep_info_buffer."""

    def __init__(self, total_timesteps: int):
        super().__init__()
        self.pbar = tqdm(
            total=total_timesteps,
            desc="training",
            unit="step",
            dynamic_ncols=True,
        )
        self._loss_sum = 0.0
        self._loss_n = 0

    def _on_step(self) -> bool:
        # Read from SB3's buffer without draining it.
        postfix = {}
        if self.model.ep_info_buffer:
            ep = self.model.ep_info_buffer[-1]
            profit = float(ep["ep_fees"] - ep["ep_gas"])
            postfix.update(
                {
                    "profit": f"${profit:.2f}",
                    "active": f"{ep['ep_active_frac'] * 100:.1f}%",
                    "rebal": f"{int(ep['ep_rebalances'])}",
                    "fees": f"${ep['ep_fees']:.0f}",
                    "gas": f"${ep['ep_gas']:.0f}",
                }
            )
        self.pbar.set_postfix(postfix)
        self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


def train(
    price: np.ndarray,
    volume_cex: np.ndarray,
    theta: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    log_dir: str,
    num_episodes: int = NUM_EPISODES,
    seed: int = 0,
) -> DoubleDQN:
    vec_env = make_vec_env(price, volume_cex, theta, mu, sigma, seed=seed)
    model = build_model(vec_env, log_dir)

    total_timesteps = num_episodes * EPISODE_LENGTH
    print(
        f"[rammstein] Training DQN for {num_episodes} episodes, "
        f"{EPISODE_LENGTH} steps each ({total_timesteps:,} total timesteps)..."
    )
    t0 = time.time()
    callback = TqdmCallback(total_timesteps=total_timesteps)
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False,  # we use our own
    )
    elapsed = time.time() - t0
    print(f"[rammstein] Training finished in {elapsed / 60:.1f} min")

    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, "dqn_model.zip")
    model.save(model_path)
    print(f"[rammstein] Saved DQN model to {model_path}")
    return model


# =============================================================================
# Evaluation (greedy rollout, mirroring Section VIII)
# =============================================================================


def evaluate(
    price: np.ndarray,
    volume_cex: np.ndarray,
    theta: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    model: DQN,
    pool: PoolConfig,
    n_episodes: int = EVAL_EPISODES,
    seed: int = 0,
) -> dict:
    """Greedy evaluation: run n_episodes end-to-end and report paper metrics."""
    np.random.seed(seed)
    active_fracs, rebal_counts, fees_tot, gas_tot = [], [], [], []

    for _ in range(n_episodes):
        env = RammsteinEnv(price, volume_cex, theta, mu, sigma, pool=pool)
        env.reset(seed=seed)
        rebalances = 0
        fees = 0.0
        gas = 0.0
        in_range_steps = 0

        for _ in range(EPISODE_LENGTH):
            obs = env._build_state(
                float(env.price[env._t]),
                float(env.theta[env._t]),
                float(env.mu[env._t]),
                float(env.sigma[env._t]),
            )
            # deterministic=True -> greedy argmax
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

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
]


def _load_binance_csv(path: str) -> pl.DataFrame:
    """Read a Binance historical-trades CSV into (ts, price, cex_volume)."""
    return pl.read_csv(
        path,
        has_header=True,
        new_columns=BINANCE_CSV_COLUMNS,
    ).select(
        ts=pl.col("time").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        price=pl.col("price"),
        cex_volume=pl.col("quoteQty"),
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
        .upsample(time_column="ts", every="1s")
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
        "--trades",
        required=True,
        help="Path to a Binance historical-trades CSV (no header, 7 cols: "
        "trade_id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch).",
    )
    parser.add_argument("--output", default="results/rammstein", help="Output dir.")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    args = parser.parse_args()

    print(f"[rammstein] Loading {args.trades}")
    df = load_trades(args.trades)
    print(f"[rammstein] Loaded {len(df):,} raw trades; resampling to 1Hz...")
    price, volume_cex, theta, mu, sigma = prepare_arrays(df)
    print(
        f"[rammstein] {len(price):,} 1Hz bars ready "
        f"(median theta={float(np.median(theta)):.4f}, "
        f"mean volume USD={float(np.mean(volume_cex)):.0f})"
    )

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
        model = train(
            train_price,
            train_vol,
            train_theta,
            train_mu,
            train_sigma,
            log_dir=args.output,
            num_episodes=args.episodes,
        )
        print("[rammstein] Evaluating on held-out test split...")
        if args.eval_episodes > 0:
            metrics = evaluate(
                test_price,
                test_vol,
                test_theta,
                test_mu,
                test_sigma,
                model,
                PoolConfig(),
                args.eval_episodes,
            )
        else:
            print("[rammstein] Skipping eval (--eval-episodes=0)")
            metrics = {}
    else:
        # Eval-only path: load saved model.
        model_path = os.path.join(args.output, "dqn_model.zip")
        model = DQN.load(model_path)
        if args.eval_episodes > 0:
            metrics = evaluate(
                test_price,
                test_vol,
                test_theta,
                test_mu,
                test_sigma,
                model,
                PoolConfig(),
                args.eval_episodes,
            )
        else:
            print("[rammstein] Skipping eval (--eval-episodes=0)")
            metrics = {}

    print("\n[rammstein] Headline metrics on test data:")
    for k, v in metrics.items():
        print(f"  {k:>14s} = {v}")


if __name__ == "__main__":
    main()
