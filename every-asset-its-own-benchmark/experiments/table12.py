"""Table 12 — held-out universe: select on 70% of symbols, evaluate on the other 30%.

Eight random splits. The cross-section is rebuilt within each side (ranks,
quintile boundaries and the cost model use only that side's symbols). Includes
the matched-breadth control: the selected config evaluated on a random subset of
the training symbols of the same size as the test set.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from config import Config
from pipeline import load_panels, run_pipeline
from experiments.util import save_csv
import metrics as M

NSPLITS = 8
TRAIN_FRAC = 0.70


def subset(weekly, factor, syms):
    w = {k: v.reindex(columns=syms) for k, v in weekly.items()}
    f = {k: v.reindex(columns=syms) for k, v in factor.items()}
    return w, f


def grid():
    return [Config(anchor="MON", ranking_frame="TS", funding_weight=0.5,
                   quintile_frac=q, turnover_cap=t, factors=None)
            for q in [0.10, 0.20, 0.30] for t in [0.5, 1.0]]


def run():
    weekly, factor, syms = load_panels("MON")
    rng = np.random.default_rng(0)
    all_syms = np.array(syms)
    n_test = int(len(all_syms) * (1 - TRAIN_FRAC))
    rows = []
    for split in range(NSPLITS):
        perm = rng.permutation(len(all_syms))
        train_syms = list(all_syms[perm[n_test:]])
        test_syms = list(all_syms[perm[:n_test]])
        # select best config on train symbols
        wtr, ftr = subset(weekly, factor, train_syms)
        best_cfg = None; best_sr = -np.inf
        for cfg in grid():
            res = run_pipeline(cfg, wtr, ftr)
            sr = M.annualized_sharpe(res["returns"])
            if np.isfinite(sr) and sr > best_sr:
                best_sr = sr; best_cfg = cfg
        # evaluate on test symbols
        wt, ft = subset(weekly, factor, test_syms)
        res_test = run_pipeline(best_cfg, wt, ft)
        sr_test = M.annualized_sharpe(res_test["returns"])
        # matched-breadth control: random subset of training symbols of size n_test
        control_syms = list(rng.choice(train_syms, size=n_test, replace=False))
        wc, fc = subset(weekly, factor, control_syms)
        res_ctl = run_pipeline(best_cfg, wc, fc)
        sr_ctl = M.annualized_sharpe(res_ctl["returns"])
        rows.append({"split": split, "train_n": len(train_syms), "test_n": n_test,
                     "train_sharpe": round(best_sr, 3), "test_sharpe": round(sr_test, 3),
                     "control_sharpe": round(sr_ctl, 3)})
    save_csv("table12", rows)
    for r in rows:
        print(f"split {r['split']}: train={r['train_n']} test={r['test_n']} "
              f"train_sr={r['train_sharpe']} test_sr={r['test_sharpe']} ctl_sr={r['control_sharpe']}")


if __name__ == "__main__":
    run()
