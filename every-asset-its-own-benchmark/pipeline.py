"""pipeline.py — ties ranking, funding penalty, portfolio, turnover, returns together.

Reads cached weekly raw panels (or takes them in-memory), runs the configured
pipeline, and returns a rich result.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from ranking import score_panel, rank_xs
from portfolio import build_book, combine_weights, apply_turnover_cap
from returns import build_returns, build_funding_shift
from config import Config
from paths import CACHE_ROOT


def load_panels(anchor, cfg=None):
    """Load prebuilt weekly panels for an anchor from the cache dir.

    Returns dict: factors (factor->week x symbol), close_w, fwd_ret_w, adv_w,
    ret_w, volume_w, funding_w, symbols.
    If cfg is given, the anchor's manifest is checked against it and an
    exception raised on mismatch, so a stale cache can never silently feed a
    shuffled config.
    """
    d = os.path.join(CACHE_ROOT, anchor)
    manifest_p = os.path.join(d, "manifest.json")
    if cfg is not None and os.path.exists(manifest_p):
        import json
        with open(manifest_p) as fh:
            man = json.load(fh)
        mcfg = man.get("config", {})
        dcfg = cfg.as_dict()
        for k in ("require_continuous_trading", "require_finite_positive_prices",
                  "book_terminal_return", "q_top_frac", "min_days_per_week",
                  "tskd_min_bars_per_side", "tskd_diff_order", "cpvv_window",
                  "avol_lookback_weeks"):
            if mcfg.get(k) != dcfg.get(k):
                raise ValueError(
                    f"cache manifest for {anchor} says {k}={mcfg.get(k)!r} "
                    f"but requested cfg says {dcfg.get(k)!r}; rebuild the cache")
    weekly = {
        k: pd.read_parquet(os.path.join(d, f"{k}.parquet"))
        for k in ("close_w", "fwd_ret_w", "adv_w", "ret_w", "volume_w", "funding_w")
    }
    factor_raw = {}
    for k in ["AVOL", "Q", "RSJ", "OFI", "CPVm", "CPVv",
              "WRspread", "TopChg", "Quad", "TKU", "TSKD"]:
        p = os.path.join(d, f"factor_{k}.parquet")
        if os.path.exists(p):
            factor_raw[k] = pd.read_parquet(p)
    symbols = list(weekly["close_w"].columns)
    return weekly, factor_raw, symbols


def run_pipeline(cfg, weekly, factor_raw):
    """Full pipeline from prebuilt weekly panels.

    weekly: dict with close_w, fwd_ret_w, adv_w, ret_w, volume_w, funding_w.
    factor: dict factor -> weekly raw panel (week x symbol).
    """
    factors = cfg.factor_list()
    fwd = weekly["fwd_ret_w"]
    funding_w = weekly["funding_w"]

    # --- scores ---
    s_final = {}
    funding_score, _ = rank_xs(funding_w)
    for k in factors:
        s = score_panel(factor_raw[k], cfg)
        s_final[k] = s - cfg.funding_weight * funding_score

    # --- portfolio ---
    if cfg.construction == "books":
        books = {}
        for k in factors:
            books[k] = build_book(s_final[k], fwd, cfg.quintile_frac, cfg.min_cross_section)
        w_star = combine_weights(books, cfg, fwd)
    else:  # blend
        # average the per-factor scores over the factors that have data
        stack = pd.concat(list(s_final.values()), axis=0)
        S = stack.groupby(level=0).mean()
        S = S.reindex(index=fwd.index, columns=fwd.columns)
        w_star = build_book(S, fwd, cfg.quintile_frac, cfg.min_cross_section)
        books = {"blend": w_star}

    w = apply_turnover_cap(w_star, cfg.turnover_cap)

    funding_shift = build_funding_shift(funding_w)
    r, turnover, costs = build_returns(w, fwd, funding_shift, weekly["adv_w"], cfg)

    return {
        "cfg": cfg,
        "weights": w,
        "target_weights": w_star,
        "books": books,
        "s_final": s_final,
        "funding_score": funding_score,
        "returns": r,
        "turnover": turnover,
        "costs": costs,
        "factors": factors,
    }
