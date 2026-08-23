"""portfolio.py — per-factor books -> combined target -> turnover cap -> weights."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_book(s_final, fwd_ret, quintile_frac=0.20, min_cross_section=10):
    """One dollar-neutral book from a single factor's final scores (vectorised).

    s_final, fwd_ret: week x symbol. Returns weights DataFrame (week x symbol).
    Long = top quintile_frac, short = bottom quintile_frac of valid; weights
    +1/len(long), -1/len(short), gross = 2, net = 0. Flat week if cross-section
    < min_cross_section.
    """
    valid = (s_final.notna() & fwd_ret.notna()).to_numpy()
    vals = s_final.to_numpy(dtype=float)
    n = valid.sum(axis=1)                                  # per-week valid count
    nsel = np.round(n * quintile_frac).clip(min=1)         # per-week leg size
    nsel = np.minimum(nsel, n // 2)
    nsel = np.maximum(nsel, 1)

    sv = np.where(np.isfinite(vals) & valid, vals, -np.inf)
    order = np.argsort(-sv, axis=1, kind="stable")         # descending; NaN at end
    rows = np.arange(len(order))[:, None]
    desc_rank = np.empty_like(order, dtype=np.int64)
    desc_rank[rows, order] = np.arange(order.shape[1])[None, :]

    long_mask = (desc_rank < nsel[:, None]) & valid
    short_mask = (desc_rank >= n[:, None] - nsel[:, None]) & valid

    denom = np.repeat(nsel[:, None], vals.shape[1], axis=1)
    w = np.zeros(vals.shape, dtype=float)
    w[long_mask] = 1.0 / denom[long_mask]
    w[short_mask] = -1.0 / denom[short_mask]
    w[n < min_cross_section, :] = 0.0                     # flat weeks
    return pd.DataFrame(w, index=s_final.index, columns=s_final.columns)


def build_blend_book(scores, fwd_ret, quintile_frac=0.20, min_cross_section=10):
    """Composite book: average the factor scores into one, form one book."""
    valid = {}
    for k, s in scores.items():
        valid[k] = s
    S = sum(scores.values()) / len(scores)
    return build_book(S, fwd_ret, quintile_frac, min_cross_section)


def book_returns(weights, fwd_ret):
    """Weekly return of each book = sum(w_k[t] * fwd_ret[t])."""
    return (weights * fwd_ret).sum(axis=1)


def combine_weights(book_weights, cfg, fwd_ret):
    """Combine per-factor book weights into a target weight vector.

    book_weights: dict factor -> weights DataFrame. Returns w* (week x symbol).
    """
    if cfg.book_weighting == "equal":
        W = sum(book_weights.values()) / len(book_weights)
        return W

    # risk_parity: weight_k[t] = 1/sigma_k[t], renormalised; flat books get 0
    V = cfg.vol_window_weeks
    # vectorised: sigma_k[t] = std of book k returns over [t-V, t-1]
    inv = {}      # book -> Series (weeks) of 1/sigma (NaN where not estimable)
    for k, bw in book_weights.items():
        br = book_returns(bw, fwd_ret)
        sigma = br.shift(1).rolling(V, min_periods=cfg.vol_min_periods).std(ddof=1)
        invk = 1.0 / sigma
        invk = invk.where(np.isfinite(invk) & (sigma > 0))
        inv[k] = invk
    inv_df = pd.DataFrame(inv)                          # weeks x books
    norm = inv_df.div(inv_df.sum(axis=1), axis=0)       # renormalise over estimable books
    # w*[t] = sum_k norm[k,t] * book_k[t]
    nw = pd.DataFrame(0.0, index=fwd_ret.index, columns=fwd_ret.columns)
    for k, bw in book_weights.items():
        nk = norm[k].reindex(fwd_ret.index)
        nw = nw + nk.fillna(0.0).values[:, None] * bw.values
    return nw


def apply_turnover_cap(target, turnover_cap=0.50):
    """Uniform partial adjustment toward the target weight vector."""
    w = target.copy()
    prev = pd.Series(0.0, index=w.columns)
    for t in w.index:
        raw_turnover = (w.loc[t] - prev).abs().sum()
        alpha = min(1.0, turnover_cap / raw_turnover) if raw_turnover > 0 else 1.0
        w.loc[t] = prev + alpha * (w.loc[t] - prev)
        prev = w.loc[t]
    return w
