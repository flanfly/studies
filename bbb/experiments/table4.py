"""Table 4 — mechanism: add XS_standardised frame; report means."""
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
    means = {}
    for frame in ["XS", "XS_standardised", "TS"]:
        vals = []
        for f in FACTORS:
            cfg = Config(anchor="MON", ranking_frame=frame, funding_weight=0.5, factors=(f,))
            sharpe, res, sm = evaluate(cfg, weekly, factor_raw)
            vals.append(sharpe)
            rows.append({"frame": frame, "factor": f, "sharpe": round(sharpe, 3)})
        means[frame] = np.nanmean(vals)
        rows.append({"frame": frame, "factor": "MEAN", "sharpe": round(means[frame], 3)})
    save_csv("table4", rows)
    for frame, m in means.items():
        print(f"{frame:16s} mean Sharpe = {m:.3f}")
    # recovery of 41% claimed: (TS-XS_std)/(TS-XS)
    num = means["TS"] - means["XS_standardised"]
    den = means["TS"] - means["XS"]
    print(f"standardisation recovery: {(num/den):.3f}")


if __name__ == "__main__":
    run()
