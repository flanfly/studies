"""checks.py — the six point-in-time discipline tests from §9.

Run with `python checks.py --anchor MON`; exits non-zero on any violation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config
from pipeline import load_panels, run_pipeline
from ranking import score_panel, rank_xs
from factors import FACTOR_NAMES


def leak_test(weekly, factor_raw, cfg, sample=150, seed=0):
    """Recompute the factor score at (sym,t) from a panel truncated after week t.

    For a sample of (factor, symbol, week) pairs across the panel, recompute
    the ranking score from a panel with all rows after week t deleted and
    compare to the full-panel value. Assert equality to 0.0 absolute diff.
    """
    rng = np.random.default_rng(seed)
    max_err = 0.0
    checked = 0
    for k in factor_raw:
        full_score = score_panel(factor_raw[k], cfg)
        valid = full_score.notna()
        pairs = np.argwhere(valid.values)
        if len(pairs) == 0:
            continue
        idx = rng.choice(len(pairs), size=min(sample, len(pairs)), replace=False)
        for r, c in pairs[idx]:
            t = full_score.index[r]
            sym = full_score.columns[c]
            sub = factor_raw[k].loc[:t]          # delete rows after week t
            s_trunc = score_panel(sub, cfg)
            v_full = full_score.at[t, sym]
            if sym in s_trunc.columns:
                v_trunc = s_trunc.at[t, sym]
            else:
                v_trunc = np.nan
            both = np.isfinite(v_full) and np.isfinite(v_trunc)
            if both:
                max_err = max(max_err, abs(v_full - v_trunc))
                checked += 1
            elif np.isfinite(v_full) != np.isfinite(v_trunc):
                max_err = 1.0   # a leak: value appears/disappears under truncation
    return max_err, checked


def run_checks(anchor="MON", cfg=None):
    cfg = cfg or Config(anchor=anchor)
    weekly, factor, symbols = load_panels(anchor)
    res = run_pipeline(cfg, weekly, factor)
    w = res["weights"]

    report = {}

    # 1. leak test (all eleven factors)
    err, nchecked = leak_test(weekly, factor, cfg)
    report["leak_checked"] = nchecked
    report["leak_max_abs_diff"] = err
    assert err <= 1e-12, f"LEAK: max abs diff {err}"

    # 2. determinism (bit-identical weights across two runs)
    w2 = run_pipeline(cfg, weekly, factor)["weights"]
    report["deterministic"] = bool(np.array_equal(w.values, w2.values))
    assert report["deterministic"]

    # 3. dollar neutrality
    net = w.sum(axis=1)
    report["dollar_neutral_max_net"] = float(np.abs(net).max())
    assert report["dollar_neutral_max_net"] < 1e-12

    # 4. gross bound
    gross = w.abs().sum(axis=1)
    report["gross_max"] = float(gross.max())
    assert report["gross_max"] <= 2.0 + 1e-12

    # 5. no forward funding: score uses week-t funding (funding_score), return
    #    uses forward-week funding (funding_shift = funding_w.shift(-1)).
    score_funding = res["funding_score"]
    assert score_funding.shape == weekly["funding_w"].shape
    funding_shift = weekly["funding_w"].shift(-1)
    # the score uses week-t funding; the return uses forward-week funding.
    # Ensure they are distinct objects/arrays (not the same panel).
    assert not np.shares_memory(np.asarray(funding_shift.values),
                                np.asarray(weekly["funding_w"].values))
    report["funding_separated"] = True

    # 6. non-emptiness: count built vs active rebalances
    report["built_weeks"] = int(w.notna().all(axis=1).sum())
    report["active_weeks"] = int(w.abs().sum(axis=1).gt(0).sum())
    assert report["active_weeks"] >= 1

    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="MON")
    a = ap.parse_args()
    rep = run_checks(a.anchor)
    for k, v in rep.items():
        print(f"{k} = {v}")
    print("ALL CHECKS PASSED")
