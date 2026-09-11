# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars>=1.0",
#     "numpy>=1.26",
#     "scipy>=1.12",
#     "torch>=2.3",
#     "scikit-learn>=1.4",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""
Backtest of the dynamic learn-to-rank trading system described in paper.md
(Barak, Mousavi & Hosseini: "Deep Reinforcement Learning for Dynamic Learn to
Rank: A Risk-Aware Framework for Cryptocurrency").

Three-stage framework:
  1. A2C deep RL agent producing cross-sectional ranking scores (price-only state)
  2. Supervised meta-learning filter (logistic regression on the strategy's own
     returns) gating trade execution by volatility regime
  3. Risk-based portfolio construction (EW, IV, MinVar, MaxDiv, RiskParity,
     MinCVaR) with Ledoit-Wolf shrinkage, cash-neutral long/short top-K/bottom-K.

Walk-forward validation: 4-month train / 1-month test, rolling monthly; agent and
filter retrained from scratch each fold (seed 42).

Data: KuCoin futures 12h OHLCV (kc-futures-12h-ohlcv.parquet, from kucoin.py).
Universe: top-60 USDT-margined perps by trailing 3-month average dollar volume,
re-selected at each fold start using data available at that time (no look-ahead).
"""

import argparse
import math
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from scipy.optimize import linprog, minimize
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression

# ----------------------------------------------------------------------------- config
SEED = 42
N_ASSETS = 60           # universe size (paper: top-60 by 3m volume)
K = 5                   # long/short leg size (paper: 5/5)
H = 30                  # state lookback: last H normalized closes (unspecified in paper)
TRAIN_MONTHS = 4        # walk-forward training window
GAMMA = 0.99
LR = 1e-5
ENTROPY_BETA = 0.01
N_STEPS = 5             # A2C n-step update
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TRAIN_PASSES = 1        # passes over training data (paper: timesteps = length of data)

W_VOL = 10              # rolling window for strategy volatility (meta filter)
TAU = 0.009             # volatility threshold (paper primary)
W_FEAT = 10             # filter features: recent strategy returns
T_HIST = 60             # historical scenarios for covariance / CVaR estimation
MOM_LOOKBACK = 30       # JT momentum lookback in 12h bars (unspecified in paper)

COST_BPS = {"gross": 0.0, "5bp": 5e-4, "10bp": 1e-3}  # per side

PERIODS_PER_YEAR = 730  # 12h bars, 365 days


def month_window(grid: np.ndarray, month: np.datetime64, delta: int) -> int:
    """First grid index at/after `month` + delta months."""
    return int(np.searchsorted(
        grid,
        (month.astype("datetime64[M]") + np.timedelta64(delta, "M")).astype("datetime64[us]"),
        side="left"))


# ----------------------------------------------------------------------------- data
def load_data(path: str):
    df = (
        pl.scan_parquet(path)
        .filter(pl.col("symbol").str.ends_with("USDTM"))
        .with_columns(pl.col("volume").cast(pl.Float64, strict=False))
        .select("time", "symbol", "close", "volume")
        .collect()
    )
    close_w = df.pivot(on="symbol", index="time", values="close").sort("time")
    vol_w = df.pivot(on="symbol", index="time", values="volume").sort("time")
    grid = close_w["time"].to_numpy()
    C = close_w.drop("time").to_numpy().astype(np.float64)
    V = vol_w.drop("time").to_numpy().astype(np.float64)
    return grid, C, V


def select_universe(C: np.ndarray, V: np.ndarray, grid: np.ndarray, t_test: int,
                    n_assets: int = N_ASSETS, look_months: int = 3,
                    min_cov: float = 0.85, full_panel: bool = False) -> np.ndarray:
    """Trailing 3-month average dollar volume ranking, requiring coverage.
    Uses only data strictly before t_test.  With full_panel=True, all eligible
    symbols are returned (no top-N cut)."""
    lo = month_window(grid, grid[t_test].astype("datetime64[us]"), -look_months)
    if t_test - lo < 60:
        return np.array([], dtype=int)
    cv = np.where(np.isnan(C[lo:t_test]) | np.isnan(V[lo:t_test]), np.nan,
                  V[lo:t_test] * C[lo:t_test])
    cov_frac = np.sum(~np.isnan(cv), axis=0) / (t_test - lo)
    dvol = np.nanmean(cv, axis=0)
    ok = (cov_frac >= min_cov) & ~np.isnan(dvol) & ~np.isnan(C[t_test])
    idx = np.where(ok)[0]
    idx = idx[np.argsort(-dvol[idx])]
    return idx if full_panel else idx[:n_assets]


# ----------------------------------------------------------------------------- A2C agent
class A2CNet(nn.Module):
    """Shared extractor (128-64-32, sigmoid) -> Gaussian policy head + value head."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.Sigmoid(),
        )
        self.pi = nn.Linear(32, n_out)
        self.v = nn.Linear(32, 1)
        self.log_std = nn.Parameter(torch.zeros(n_out))

    def forward(self, x):
        h = self.shared(x)
        return self.pi(h), self.v(h).squeeze(-1)


LOG2PI = math.log(2 * math.pi)


def train_a2c(Snorm: np.ndarray, R: np.ndarray, t_range: np.ndarray, seed: int,
              passes: int = 1) -> A2CNet:
    """One chronological pass over the training segment with n-step A2C updates.

    For each decision bar t in t_range: state = last H normalized closes,
    action = sampled scores, reward = +/-1 by sign of the EW top-K/bottom-K
    portfolio return over (t, t+1].
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    N = R.shape[1]
    net = A2CNet(N * H, N)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    obs, acts, rews, = [], [], []

    def update(v_next_state: torch.Tensor):
        o = torch.tensor(np.array(obs), dtype=torch.float32)
        a = torch.tensor(np.array(acts), dtype=torch.float32)
        r = torch.tensor(np.array(rews), dtype=torch.float32)
        with torch.no_grad():
            _, v_b = net(v_next_state)
        G = torch.zeros(len(rews))
        g = v_b[0]
        for i in reversed(range(len(rews))):
            g = r[i] + GAMMA * g
            G[i] = g
        h = net.shared(o)
        mu = net.pi(h)
        log_std = net.log_std.clamp(-5.0, 2.0)
        std = torch.exp(log_std)
        v = net.v(h).squeeze(-1)
        adv = (G - v).detach()
        logp = (-0.5 * ((a - mu) / std) ** 2 - log_std - 0.5 * LOG2PI).sum(dim=1)
        entropy = (log_std + 0.5 * LOG2PI + 0.5).sum() / N
        loss = -(logp * adv).mean() - ENTROPY_BETA * entropy + VF_COEF * ((G - v) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
        opt.step()
        obs.clear(); acts.clear(); rews.clear()

    for _ in range(passes):
      for j, t in enumerate(t_range):  # t is LOCAL (fold-relative) here
        with torch.no_grad():
            win = torch.tensor(Snorm[t - H + 1: t + 1].T.reshape(-1), dtype=torch.float32)
            h = net.shared(win)
            mu = net.pi(h)
            std = torch.exp(net.log_std.clamp(-5.0, 2.0))
            a = (mu + std * torch.randn_like(mu)).numpy()
        idx = np.argsort(-a)
        r_t = 0.5 * R[t + 1, idx[:K]].mean() - 0.5 * R[t + 1, idx[-K:]].mean()
        obs.append(win.numpy()); acts.append(a); rews.append(1.0 if r_t > 0 else -1.0)
        if len(obs) == N_STEPS:
            nxt = torch.tensor(Snorm[t + 1 - H + 1: t + 2].T.reshape(1, -1), dtype=torch.float32)
            update(nxt)
    if obs:
        t_end = min(t_range[-1] + 1, Snorm.shape[0] - 1)
        update(torch.tensor(Snorm[t_end - H + 1: t_end + 1].T.reshape(1, -1), dtype=torch.float32))
    return net


def agent_scores(net: A2CNet, Snorm: np.ndarray, t_range: np.ndarray) -> np.ndarray:
    """Deterministic (mean) scores, shape (len(t_range), N) with N = Snorm.shape[1]."""
    N = Snorm.shape[1]
    states = np.empty((len(t_range), N * H), dtype=np.float32)
    for j, t in enumerate(t_range):
        states[j] = Snorm[t - H + 1: t + 1].T.reshape(-1)
    with torch.no_grad():
        mu = net.pi(net.shared(torch.tensor(states)))
    return mu.numpy().astype(np.float64)


# ----------------------------------------------------------------------------- portfolio construction
def alloc_weights(method: str, R_hist: np.ndarray) -> np.ndarray:
    """Long-leg weights, K assets, w >= 0, sum(w) = 1."""
    Kk = R_hist.shape[1]
    if method == "EW":
        return np.full(Kk, 1.0 / Kk)
    sigma = np.std(R_hist, axis=0)
    sigma = np.maximum(sigma, 1e-12)
    if method == "IV":
        w = 1.0 / sigma
        return w / w.sum()
    Sg = LedoitWolf().fit(R_hist).covariance_
    if method == "MinVar":
        res = minimize(lambda w: w @ Sg @ w, np.full(Kk, 1.0 / Kk),
                       jac=lambda w: 2 * Sg @ w, method="SLSQP",
                       bounds=[(0, 1)] * Kk,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                       options={"maxiter": 200, "ftol": 1e-12})
        return res.x if res.success else np.full(Kk, 1.0 / Kk)
    if method == "MaxDiv":
        def neg_ratio(w):
            var = w @ Sg @ w
            return 1e12 if var <= 1e-18 else -(w @ sigma) / math.sqrt(var)
        res = minimize(neg_ratio, np.full(Kk, 1.0 / Kk), method="SLSQP",
                       bounds=[(0, 1)] * Kk,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                       options={"maxiter": 200, "ftol": 1e-12})
        return res.x if res.success else np.full(Kk, 1.0 / Kk)
    if method == "RiskParity":
        def obj(w):
            mrc = Sg @ w
            pr = w @ mrc
            return 1e12 if pr <= 1e-18 else np.sum((w * mrc - pr / Kk) ** 2)
        res = minimize(obj, np.full(Kk, 1.0 / Kk), method="SLSQP",
                       bounds=[(1e-4, 1)] * Kk,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                       options={"maxiter": 300, "ftol": 1e-14})
        w = np.clip(res.x if res.success else np.full(Kk, 1.0 / Kk), 1e-4, None)
        return w / w.sum()
    if method == "MinCVaR":
        alpha, Th = 0.95, R_hist.shape[0]
        c = np.zeros(Kk + 1 + Th)
        c[Kk] = 1.0
        c[Kk + 1:] = 1.0 / ((1 - alpha) * Th)
        A_ub = np.hstack([-R_hist, -np.ones((Th, 1)), -np.eye(Th)])
        A_eq = np.zeros((1, Kk + 1 + Th)); A_eq[0, :Kk] = 1.0
        bounds = [(0, 1)] * Kk + [(None, None)] + [(0, None)] * Th
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(Th), A_eq=A_eq, b_eq=[1.0],
                      bounds=bounds, method="highs")
        return res.x[:Kk] if res.success else np.full(Kk, 1.0 / Kk)
    raise ValueError(method)


# ----------------------------------------------------------------------------- metrics
def metrics(rets: np.ndarray) -> dict:
    rets = rets[np.isfinite(rets)]
    ann = math.sqrt(PERIODS_PER_YEAR)
    mu, sd = rets.mean(), rets.std(ddof=1)
    neg = rets[rets < 0]
    sd_d = neg.std(ddof=1) if len(neg) > 1 else np.nan
    equity = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    return {
        "Cum. Return": equity[-1] - 1,
        "Ann. Volatility": sd * ann,
        "MDD": float(np.min(equity / peak - 1)),
        "Sharpe": mu / sd * ann if sd > 0 else np.nan,
        "Sortino": mu / sd_d * ann if sd_d and sd_d > 0 else np.nan,
        "Omega": float(rets[rets > 0].sum() / -rets[rets < 0].sum()) if (rets < 0).any() else np.nan,
        "n": len(rets),
    }


# ----------------------------------------------------------------------------- fold runner
def run_fold(C: np.ndarray, V: np.ndarray, R_all: np.ndarray, grid: np.ndarray, t_test: int,
             seed: int, passes: int = 1, w_vol: int = W_VOL, tau: float = TAU,
             full_panel: bool = False) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Train on the 4 months before t_test, trade the month starting t_test.

    Evaluated returns for a fold: returns over (t, t+1] with t+1 in the test
    month, i.e. decisions at t in [t_test-1, t_test_hi-2].  Consecutive folds
    stitch into a continuous, non-overlapping return stream.

    Returns {strategy: (return timestamps, gross returns, per-step turnover)}.
    """
    t_train_lo = month_window(grid, grid[t_test].astype("datetime64[us]"), -TRAIN_MONTHS)
    t_test_hi = month_window(grid, grid[t_test].astype("datetime64[us]"), 1)
    t_test_hi = min(t_test_hi, C.shape[0] - 1)
    t_lo = t_train_lo - H  # warmup for the first state windows
    if t_lo < 0 or t_test_hi - t_test < 30:
        return {}

    uni = select_universe(C, V, grid, t_test, full_panel=full_panel)
    if len(uni) < (20 if full_panel else N_ASSETS):
        return {}
    n = len(uni)

    # --- prices (forward-fill gaps) and train-only min-max normalization ------
    Cw = C[t_lo: t_test_hi][:, uni].copy()
    n_rows = Cw.shape[0]
    for i in range(Cw.shape[1]):
        col = Cw[:, i]
        mask = np.isnan(col)
        if mask.any():
            idx = np.where(~mask, np.arange(n_rows), 0)
            np.maximum.accumulate(idx, out=idx)
            col = col[idx]
            still = np.isnan(col)
            if still.any():
                first_valid = np.argmax(~np.isnan(col))
                col[still] = col[first_valid] if (~np.isnan(col)).any() else 1.0
            Cw[:, i] = col
    Cw = np.where(np.isnan(Cw), 1.0, Cw)

    tr_min = np.nanmin(Cw[H: t_test - t_lo], axis=0)
    tr_max = np.nanmax(Cw[H: t_test - t_lo], axis=0)
    rng = np.where(tr_max - tr_min < 1e-12, 1e-12, tr_max - tr_min)
    Snorm = (Cw - tr_min) / rng  # ~[0,1] in train; test may exceed

    R = np.where(np.isfinite(R_all[t_lo: t_test_hi][:, uni]), R_all[t_lo: t_test_hi][:, uni], 0.0)

    # --- A2C training (chronological pass) ------------------------------------
    # decision bars in LOCAL coordinates; state window [t-H+1, t] fully inside
    # the training segment; last reward uses return over (t_test-2, t_test-1]
    train_t = np.arange(H - 1, t_test - 1 - t_lo)  # local indices
    if len(train_t) < 50:
        return {}
    net = train_a2c(Snorm, R, train_t, seed, passes)

    # in-sample deterministic (mean-action) returns -> meta-filter training set
    det_scores = agent_scores(net, Snorm, train_t)
    det_t = train_t + 1 + t_lo             # return timestamps (grid indices)
    det_r = np.array([
        0.5 * R[t + 1, np.argsort(-det_scores[j])[:K]].mean()
        - 0.5 * R[t + 1, np.argsort(-det_scores[j])[-K:]].mean()
        for j, t in enumerate(train_t)
    ])

    # --- meta-learning filter: logistic regression -----------------------------
    sig = np.array([det_r[max(0, i - w_vol + 1): i + 1].std(ddof=1) for i in range(len(det_r))])
    y = (sig < tau).astype(int)
    X = np.array([det_r[max(0, i - W_FEAT + 1): i + 1] if i + 1 >= W_FEAT
                  else np.pad(det_r[:i + 1], (W_FEAT - i - 1, 0))
                  for i in range(len(det_r))])
    valid = W_VOL
    filt = None
    if len(y[valid:]) > 20 and len(np.unique(y[valid:])) > 1:
        filt = LogisticRegression().fit(X[valid:], y[valid:])

    # --- out-of-sample walk ----------------------------------------------------
    test_t_local = np.arange(t_test - 1 - t_lo, t_test_hi - 1 - t_lo)  # local decision bars
    test_scores = agent_scores(net, Snorm, test_t_local)
    n_test = len(test_t_local)
    ret_ts = grid[t_test: t_test_hi]                # t+1 timestamps

    # strategy return stream (index g = return over (g-1, g]) for filter features:
    # in-sample train returns + realized test returns
    strat = np.full(C.shape[0], np.nan)
    strat[det_ts := det_t] = det_r

    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rng_np = np.random.default_rng(seed)

    strategies = ["DRL_rank", "DRL+filt:EW", "DRL+filt:IV", "DRL+filt:MinVar",
                  "DRL+filt:MaxDiv", "DRL+filt:RiskParity", "DRL+filt:MinCVaR",
                  "JT_mom", "Random"]

    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    gates_open = 0
    gates_total = 0
    for name in strategies:
        w_prev = np.zeros(n)
        rets = np.zeros(n_test)
        tos = np.zeros(n_test)
        for j, t in enumerate(test_t_local):  # t is LOCAL
            if name == "DRL_rank":
                gate = True
            elif name in ("DRL+filt:EW", "DRL+filt:IV", "DRL+filt:MinVar",
                          "DRL+filt:MaxDiv", "DRL+filt:RiskParity", "DRL+filt:MinCVaR"):
                if filt is not None:
                    hist = strat[: t + 1 + t_lo]  # t is local; strat is grid-indexed
                    x = hist[~np.isnan(hist)][-W_FEAT:]
                    if len(x) < W_FEAT:
                        x = np.pad(x, (W_FEAT - len(x), 0))
                    gate = filt.predict(x.reshape(1, -1))[0] == 1
                else:
                    gate = True
            else:
                gate = True

            if gate:
                gates_total += 1
            else:
                gates_open += 1
                gates_total += 1
            w = np.zeros(n)
            if gate:
                if name.startswith("DRL"):
                    idx = np.argsort(-test_scores[j])
                elif name == "JT_mom":
                    cum = np.prod(1 + R[max(0, t - MOM_LOOKBACK + 1): t + 1], axis=0) - 1
                    idx = np.argsort(-cum)
                else:  # Random
                    idx = np.argsort(-rng_np.random(n))
                long_idx, short_idx = idx[:K], idx[-K:]
                if name in ("DRL_rank", "DRL+filt:EW", "JT_mom", "Random"):
                    wl = ws = np.full(K, 1.0 / K)
                else:
                    Rh = R[max(0, t + 1 - T_HIST): t + 1]  # scenarios ending at t
                    method = name.split(":")[1]
                    wl = alloc_weights(method, Rh[:, long_idx])
                    ws = alloc_weights(method, Rh[:, short_idx])
                w[long_idx] = 0.5 * wl
                w[short_idx] = -0.5 * ws
            tos[j] = np.abs(w - w_prev).sum()
            rets[j] = w @ R[t + 1]
            w_prev = w
            # update the shared strategy stream with the EW unfiltered DRL return
            if name == "DRL_rank":
                strat[t + 1 + t_lo] = rets[j]  # grid index
        results[name] = (ret_ts, rets, tos)
    results["_gate_rate"] = (ret_ts, np.full(n_test, gates_open / max(gates_total, 1)), None)
    return results


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="kc-futures-12h-ohlcv.parquet")
    ap.add_argument("--out", default="backtest-results.parquet")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--folds", type=int, default=0, help="only run the first N folds (0=all)")
    ap.add_argument("--passes", type=int, default=TRAIN_PASSES)
    ap.add_argument("--w-vol", type=int, default=W_VOL)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--panel", choices=["top60", "full"], default="top60")
    args = ap.parse_args()

    grid, C, V = load_data(args.data)
    print(f"grid: {len(grid)} bars, {C.shape[1]} perp symbols, {grid[0]} .. {grid[-1]}")

    R_all = np.full_like(C, np.nan)
    R_all[1:] = C[1:] / C[:-1] - 1.0

    # fold start months: test month must have 4 prior months + H-bar warmup
    starts = [ms for ms in np.unique(grid.astype("datetime64[M]"))]
    folds = []
    for ms in starts:
        t_test = month_window(grid, ms.astype("datetime64[us]"), 0)
        t_train_lo = month_window(grid, ms.astype("datetime64[us]"), -TRAIN_MONTHS)
        if t_test - t_train_lo >= 60 + H and t_train_lo >= H and month_window(grid, ms.astype("datetime64[us]"), 1) - t_test >= 30 \
                and month_window(grid, ms.astype("datetime64[us]"), -3) >= 60:  # full 3-month universe lookback + H warmup
            folds.append(t_test)
    print(f"folds: {len(folds)} (first test month {grid[folds[0]].astype('datetime64[M]')})")

    streams: dict[str, list] = {}
    t0 = time.time()
    for fi, t_test in enumerate(folds):
        if args.folds and fi >= args.folds:
            break
        try:
            out = run_fold(C, V, R_all, grid, t_test, args.seed, args.passes,
                           args.w_vol, args.tau, args.panel == "full")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"fold {fi} ({grid[t_test]}) FAILED: {e}")
            continue
        for k, v in out.items():
            if k.startswith("_"):
                continue
            streams.setdefault(k, []).append(v)
        if (fi + 1) % 5 == 0 or fi == len(folds) - 1:
            print(f"[{fi+1}/{len(folds)}] {grid[t_test].astype('datetime64[M]')} "
                  f"gate_open={out.get('_gate_rate', (None, np.array([np.nan]), None))[1][0]:.2f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    rows = []
    if not streams:
        raise SystemExit("no folds produced results")
    for k, chunks in streams.items():
        for ts, rets, tos in chunks:
            for ti, r, to in zip(ts, rets, tos):
                rows.append({"strategy": k, "time": np.datetime64(ti, "us").tolist(),
                             "ret_gross": r, "turnover": to})
    res = pl.DataFrame(rows).unique(subset=["time", "strategy"], keep="first").sort("time", "strategy")
    res.write_parquet(args.out)

    for cost_key in ["gross", "5bp", "10bp"]:
        s, y = summarize(cost_key, streams)
        print(f"\n=== Overall performance ({cost_key}) ===")
        with pl.Config(tbl_rows=20, tbl_cols=20):
            print(s)
        if cost_key == "gross":
            print("\n=== Yearly breakdown (gross) ===")
            with pl.Config(tbl_rows=200, tbl_cols=20):
                print(y.sort("strategy", "year"))
    y.sort("strategy", "year").write_parquet("yearly.parquet")
    print(f"\nDone in {time.time()-t0:.0f}s. Results -> {args.out}")


def summarize(cost_key, streams):
    summary, yearly = [], []
    for k in streams:
        ts = np.concatenate([c[0] for c in streams[k]])
        rr = np.concatenate([c[1] for c in streams[k]])
        tt = np.concatenate([c[2] for c in streams[k]])
        rnet = rr - COST_BPS[cost_key] * tt
        m = metrics(rnet)
        m["strategy"] = k
        m["turnover/step"] = float(tt.mean())
        summary.append(m)
        years = ts.astype("datetime64[Y]")
        for yy in np.unique(years):
            my = metrics(rnet[years == yy])
            my["strategy"] = k
            my["year"] = str(yy)
            yearly.append(my)
    return pl.DataFrame(summary).sort("Sharpe", descending=True), pl.DataFrame(yearly)


if __name__ == "__main__":
    main()