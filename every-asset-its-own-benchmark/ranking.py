"""ranking.py — to_score, rank_xs, rank_ts, rank_xs_standardised.

Panels are (week x symbol). All ranking helpers map to [-1,+1] and preserve NaN.
They take an explicit axis/by argument and assert output shape (leak-test safe).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_score(rank, n):
    """rank is 1-based ascending; n = cross-sectional count."""
    return ((rank - 0.5) / n - 0.5) * 2.0


def rank_xs(f):
    """Cross-sectional score: rank f[i,t] among valid assets at week t.

    f: week x symbol. Returns (score, n_per_week).
    """
    f = f.copy()
    n = f.notna().sum(axis=1).replace(0, np.nan)      # valid count per week
    r = f.rank(axis=1, method="average")               # 1-based ascending, ties avg
    # row-wise: score = ((rank - 0.5)/n - 0.5)*2 ; n aligns to rows (axis=0)
    score = ((r - 0.5).div(n, axis=0) - 0.5) * 2.0
    score = score.where(f.notna())
    return score, n


def _ts_score(s_vals, window, min_periods):
    """Vectorised self-referential score for one symbol's series.

    score[t] = to_score(rank of s[t] among strictly-prior valid window, len+1).
    Built as a (n x window) matrix of strictly-prior values.
    """
    n = len(s_vals)
    if n == 0:
        return np.full(0, np.nan)
    # prior[t, k] = s_vals[t-1-k], NaN outside the panel
    k_idx = np.arange(window)[None, :]
    t_idx = np.arange(n)[:, None]
    src = t_idx - 1 - k_idx                      # source index for prior
    valid_src = src >= 0
    srcc = np.where(valid_src, src, 0)
    prior = s_vals[srcc]
    prior = np.where(valid_src, prior, np.nan)
    x = s_vals[:, None]
    n_valid = np.isfinite(prior).sum(axis=1)
    count_less = np.nansum((prior < x) & np.isfinite(prior), axis=1)
    rank = 1.0 + count_less
    nn = n_valid + 1.0
    score = ((rank - 0.5) / nn - 0.5) * 2.0
    score = np.where(n_valid >= min_periods, score, np.nan)
    score = np.where(np.isfinite(s_vals), score, np.nan)
    return score


def rank_ts(f, window=52, min_periods=26):
    """Self-referential score, per symbol (column)."""
    f = f.copy()
    out = pd.DataFrame(np.nan, index=f.index, columns=f.columns)
    for col in f.columns:
        out[col] = _ts_score(f[col].to_numpy(), window, min_periods)
    assert out.shape == f.shape
    return out


def rank_xs_standardised(f, window=52, min_periods=26):
    """z-score each asset vs its own trailing window, then rank cross-sectionally."""
    f = f.copy()
    zm = pd.DataFrame(np.nan, index=f.index, columns=f.columns)
    for col in f.columns:
        s = f[col]
        mean = s.shift(1).rolling(window, min_periods=min_periods).mean()
        std = s.shift(1).rolling(window, min_periods=min_periods).std(ddof=1)
        zm[col] = (s - mean) / std
    score, _ = rank_xs(zm)
    assert score.shape == f.shape
    return score


def score_panel(f_raw, cfg):
    """Apply smoothing then the configured ranking frame to a raw factor panel.

    Returns the raw score panel s[i,t] in [-1,1].
    """
    # smoothing: rolling mean over trailing smooth_window_weeks (incl. current)
    smooth = f_raw.rolling(cfg.smooth_window_weeks, min_periods=cfg.smooth_window_weeks).mean()
    if cfg.ranking_frame == "XS":
        s, _ = rank_xs(smooth)
    elif cfg.ranking_frame == "TS":
        s = rank_ts(smooth, cfg.rank_window_weeks, cfg.rank_min_periods)
    elif cfg.ranking_frame == "XS_standardised":
        s = rank_xs_standardised(smooth, cfg.rank_window_weeks, cfg.rank_min_periods)
    else:
        raise ValueError(f"unknown ranking_frame {cfg.ranking_frame}")
    return s
