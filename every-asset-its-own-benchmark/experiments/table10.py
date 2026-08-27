"""Table 10 — walk-forward × rebalance anchor.

Select config on a trailing 104-week window, evaluate on the next 26, roll
forward. 36 candidates per fold (quintile_frac x3, funding_weight x3,
turnover_cap x2, rank_window_weeks x2). The pipeline is point-in-time, so each
candidate is run once on the full panel and train/test windows are sliced.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from itertools import product
from config import Config
from pipeline import load_panels, run_pipeline
from experiments.util import save_csv
import metrics as M
from paths import CACHE_ROOT

ANCHORS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
TRAIN = 104
TEST = 26


def grid():
    return list(product([0.10, 0.20, 0.30],   # quintile_frac
                        [0.25, 0.5, 1.0],     # funding_weight
                        [0.35, 0.5],          # turnover_cap
                        [26, 52]))            # rank_window_weeks


def run_anchor(anchor):
    weekly, factor, syms = load_panels(anchor)
    fwd = weekly["fwd_ret_w"]
    cand = {}
    for qf, fw, tc, rw in grid():
        cfg = Config(anchor=anchor, ranking_frame="TS", funding_weight=fw,
                     quintile_frac=qf, turnover_cap=tc, rank_window_weeks=rw,
                     factors=None)
        res = run_pipeline(cfg, weekly, factor)
        cand[(qf, fw, tc, rw)] = res["returns"]

    all_weeks = fwd.index
    folds = []
    test_series = []
    t = 55
    while t + TRAIN + TEST <= len(all_weeks):
        tr = all_weeks[t: t + TRAIN]
        te = all_weeks[t + TRAIN: t + TRAIN + TEST]
        best = None
        best_s = -np.inf
        for key, r in cand.items():
            sr = M.annualized_sharpe(r.reindex(tr))
            if np.isfinite(sr) and sr > best_s:
                best_s = sr
                best = key
        test_series.append(cand[best].reindex(te).dropna())
        folds.append({"train_start": str(tr[0]), "train_end": str(tr[-1]),
                      "test_start": str(te[0]), "test_end": str(te[-1]),
                      "best_q": best[0], "best_fw": best[1], "best_tc": best[2],
                      "best_rw": best[3], "train_sharpe": round(best_s, 3)})
        t += TEST
    out_r = test_series[0]
    for x in test_series[1:]:
        out_r = pd.concat([out_r, x])
    return folds, out_r


def run():
    rows = []
    for anchor in ANCHORS:
        p = os.path.join(CACHE_ROOT, anchor, "factor_AVOL.parquet")
        if not os.path.exists(p):
            print(f"{anchor}: panels not built, skipping", flush=True)
            continue
        folds, out_r = run_anchor(anchor)
        sr = M.annualized_sharpe(out_r)
        rows.append({"anchor": anchor, "walkforward_sharpe": round(sr, 3),
                     "n_folds": len(folds),
                     "n_test_weeks": int(out_r.dropna().shape[0])})
    save_csv("table10", rows)
    for r in rows:
        print(f"{r['anchor']}: WF Sharpe={r['walkforward_sharpe']}  "
              f"folds={r['n_folds']} weeks={r['n_test_weeks']}")


if __name__ == "__main__":
    run()
