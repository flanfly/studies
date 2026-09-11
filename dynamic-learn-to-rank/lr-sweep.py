# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars>=1.0",
#     "numpy>=1.26",
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
"""Does harder training (more passes / higher lr) improve the DRL ranking signal?"""
import math
import numpy as np
import torch
import torch.nn as nn
import backtest as bt

grid, C, V = bt.load_data("kc-futures-12h-ohlcv.parquet")
R_all = np.full_like(C, np.nan); R_all[1:] = C[1:] / C[:-1] - 1.0

starts = np.unique(grid.astype("datetime64[M]"))
folds = []
for ms in starts:
    t_test = bt.month_window(grid, ms.astype("datetime64[us]"), 0)
    t_train_lo = bt.month_window(grid, ms.astype("datetime64[us]"), -4)
    t_hi = bt.month_window(grid, ms.astype("datetime64[us]"), 1)
    if t_test - t_train_lo >= 60 + bt.H and t_train_lo >= bt.H and t_hi - t_test >= 30 \
            and bt.month_window(grid, ms.astype("datetime64[us]"), -3) >= 60:
        folds.append(t_test)
folds = folds[::4]


def train_passes(net, opt, Snorm, R, train_t, lr):
    obs, acts, rews = [], [], []

    def update(v_next_state):
        o = torch.tensor(np.array(obs), dtype=torch.float32)
        a = torch.tensor(np.array(acts), dtype=torch.float32)
        r = torch.tensor(np.array(rews), dtype=torch.float32)
        with torch.no_grad():
            _, v_b = net(v_next_state)
        G = torch.zeros(len(rews))
        g = v_b[0]
        for i in reversed(range(len(rews))):
            g = r[i] + bt.GAMMA * g
            G[i] = g
        h = net.shared(o)
        mu = net.pi(h)
        log_std = net.log_std.clamp(-5.0, 2.0)
        std = torch.exp(log_std)
        v = net.v(h).squeeze(-1)
        adv = (G - v).detach()
        logp = (-0.5 * ((a - mu) / std) ** 2 - log_std - 0.5 * bt.LOG2PI).sum(dim=1)
        entropy = (log_std + 0.5 * bt.LOG2PI + 0.5).sum() / bt.N_ASSETS
        loss = -(logp * adv).mean() - bt.ENTROPY_BETA * entropy + bt.VF_COEF * ((G - v) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), bt.MAX_GRAD_NORM)
        opt.step()
        obs.clear(); acts.clear(); rews.clear()

    for j, t in enumerate(train_t):
        with torch.no_grad():
            win = torch.tensor(Snorm[t - bt.H + 1: t + 1].T.reshape(-1), dtype=torch.float32)
            h = net.shared(win)
            mu = net.pi(h)
            std = torch.exp(net.log_std.clamp(-5.0, 2.0))
            a = (mu + std * torch.randn_like(mu)).numpy()
        idx = np.argsort(-a)
        r_t = 0.5 * R[t + 1, idx[:bt.K]].mean() - 0.5 * R[t + 1, idx[-bt.K:]].mean()
        obs.append(win.numpy()); acts.append(a); rews.append(1.0 if r_t > 0 else -1.0)
        if len(obs) == bt.N_STEPS:
            update(torch.tensor(Snorm[t + 1 - bt.H + 1: t + 2].T.reshape(1, -1), dtype=torch.float32))
    if obs:
        t_end = min(train_t[-1] + 1, Snorm.shape[0] - 1)
        update(torch.tensor(Snorm[t_end - bt.H + 1: t_end + 1].T.reshape(1, -1), dtype=torch.float32))


for lr, passes in [(1e-5, 1), (1e-4, 10), (1e-4, 50), (1e-3, 10)]:
    for seed in [8, 42]:
        rets, val_rets = [], []
        for t_test in folds:
            t_train_lo = bt.month_window(grid, grid[t_test].astype("datetime64[us]"), -4)
            t_test_hi = bt.month_window(grid, grid[t_test].astype("datetime64[us]"), 1)
            t_lo = t_train_lo - bt.H
            uni = bt.select_universe(C, V, grid, t_test)
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
                        fv = np.argmax(~np.isnan(col))
                        col[still] = col[fv] if (~np.isnan(col)).any() else 1.0
                Cw[:, i] = np.where(np.isnan(col), 1.0, col) if mask.any() else col
            Cw = np.where(np.isnan(Cw), 1.0, Cw)
            tr_min = np.nanmin(Cw[bt.H: t_test - t_lo], axis=0)
            tr_max = np.nanmax(Cw[bt.H: t_test - t_lo], axis=0)
            rng_ = np.where(tr_max - tr_min < 1e-12, 1e-12, tr_max - tr_min)
            Snorm = (Cw - tr_min) / rng_
            R = np.where(np.isfinite(R_all[t_lo: t_test_hi][:, uni]), R_all[t_lo: t_test_hi][:, uni], 0.0)
            train_t = np.arange(bt.H - 1, t_test - 1 - t_lo)
            test_t = np.arange(t_test - 1 - t_lo, t_test_hi - 1 - t_lo)

            torch.manual_seed(seed)
            net = bt.A2CNet(bt.N_ASSETS * bt.H, bt.N_ASSETS)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            for _ in range(passes):
                train_passes(net, opt, Snorm, R, train_t, lr)
            scores = bt.agent_scores(net, Snorm, test_t)
            rr = []
            for j, t in enumerate(test_t):
                idx = np.argsort(-scores[j])
                rr.append(0.5 * R[t + 1, idx[:bt.K]].mean() - 0.5 * R[t + 1, idx[-bt.K:]].mean())
            rr = np.array(rr)
            rets.append(rr)
            if t_test <= folds[1]:
                val_rets.append(rr)
        r = np.concatenate(rets)
        m = bt.metrics(r)
        v = bt.metrics(np.concatenate(val_rets))
        print(f"lr={lr:g} passes={passes:3d} seed={seed}: full cum={m['Cum. Return']:+.1%} "
              f"sharpe={m['Sharpe']:+.2f} | val sharpe={v['Sharpe']:+.2f}", flush=True)