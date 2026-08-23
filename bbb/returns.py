"""returns.py — weights + fwd + funding + costs -> weekly return series."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cost_per_symbol(adv_w, cfg):
    """Liquidity-scaled maker fee, cross-sectional per week.

    p = percentile rank of adv in [0,1]; c = fee_bp_liquid*1e-4*(5-4*p) scaled.
    Returns week x symbol DataFrame.
    """
    p = adv_w.rank(axis=1, pct=True, method="average")
    base = cfg.fee_bp_illiquid - (cfg.fee_bp_illiquid - cfg.fee_bp_liquid) * p
    return 1e-4 * base * cfg.cost_multiple


def build_returns(w, fwd_ret, funding_shift, adv_w, cfg):
    """Weekly return series from final weights.

    w: week x symbol (post turnover cap).
    fwd_ret: week x symbol, clipped.
    funding_shift: week x symbol of funding for the *forward* week (week t+1),
      aligned to week t rows (i.e. funding_fwd).
    Returns (return_series, turnover_series, costs_series).
    """
    fwd = fwd_ret.copy()
    if cfg.clip_forward_return is not None:
        fwd = fwd.clip(upper=cfg.clip_forward_return)

    c = cost_per_symbol(adv_w, cfg)

    dw = w.diff().fillna(0.0)
    r_price = (w * fwd).sum(axis=1)
    r_fund = (w * funding_shift).sum(axis=1)
    turnover = dw.abs().sum(axis=1)
    costs = (c * dw.abs()).sum(axis=1)
    r = r_price - r_fund - costs
    return r, turnover, costs


def build_funding_shift(funding_w):
    """Return funding series aligned so that row t holds funding paid during week t+1.

    funding_w: week x symbol, sum of funding in each week.
    Returns array funding_shift where funding_shift[t] = funding_w[t+1].
    """
    return funding_w.shift(-1)


def weighted_series(x, w):
    """A series over weeks from per-symbol panel x and weight w (pointwise)."""
    return (w * x).sum(axis=1)
