"""Table 3 — ranking frame: single-factor books under XS vs TS, funding 0.5/0.0.

Reproduces the paper's Table 3: annualised Sharpe per factor and the mean.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from config import Config
from experiments.util import ensure, evaluate, save_csv

FACTORS = ["AVOL", "Q", "RSJ", "OFI", "CPVm", "CPVv",
           "WRspread", "TopChg", "Quad", "TKU", "TSKD"]


def run():
    weekly, factor_raw, symbols = ensure("MON")
    rows = []
    for frame in ["XS", "TS"]:
        for fw in [0.5, 0.0]:
            mean_s = []
            for f in FACTORS:
                cfg = Config(anchor="MON", ranking_frame=frame,
                             funding_weight=fw, factors=(f,))
                sharpe, res, sm = evaluate(cfg, weekly, factor_raw)
                mean_s.append(sharpe)
                rows.append({"frame": frame, "funding_weight": fw,
                             "factor": f, "sharpe": round(sharpe, 3)})
            rows.append({"frame": frame, "funding_weight": fw, "factor": "MEAN",
                         "sharpe": round(np.nanmean(mean_s), 3)})
    save_csv("table3", rows)
    print_table(rows)


def print_table(rows):
    for r in rows:
        if r["factor"] == "MEAN":
            print(f"{r['frame']:3s} fw={r['funding_weight']:.1f}  MEAN  {r['sharpe']:.3f}")
        else:
            print(f"{r['frame']:3s} fw={r['funding_weight']:.1f}  {r['factor']:10s} {r['sharpe']:.3f}")


if __name__ == "__main__":
    run()
