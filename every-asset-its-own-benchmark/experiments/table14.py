"""Table 14 — cost robustness: sweep cost_multiple for turnover_cap 1.0 and 0.5."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from config import Config
from experiments.util import ensure, evaluate, save_csv

MULT = [1, 2, 4, 8, 16, 32]


def run():
    weekly, factor_raw, symbols = ensure("MON")
    rows = []
    for cap in [1.0, 0.5]:
        for cm in MULT:
            cfg = Config(anchor="MON", ranking_frame="TS", funding_weight=0.5,
                         turnover_cap=cap, cost_multiple=cm, factors=None)
            sharpe, res, sm = evaluate(cfg, weekly, factor_raw)
            rows.append({"turnover_cap": cap, "cost_multiple": cm,
                         "sharpe": round(sharpe, 3),
                         "ann_return": round(sm["ann_return"], 4)})
    save_csv("table14", rows)
    for r in rows:
        print(f"cap={r['turnover_cap']:.2f} cost={r['cost_multiple']:3d}  "
              f"sharpe={r['sharpe']}  ann={r['ann_return']}")


if __name__ == "__main__":
    run()
