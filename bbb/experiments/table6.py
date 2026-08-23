"""Table 6 — construction: sweep {XS,TS} x {books,blend} x {equal,risk_parity}."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from config import Config
from experiments.util import ensure, evaluate, save_csv

FACTORS = None  # full eleven-factor book


def run():
    weekly, factor_raw, symbols = ensure("MON")
    rows = []
    for frame in ["XS", "TS"]:
        for construction in ["books", "blend"]:
            for weighting in ["equal", "risk_parity"]:
                cfg = Config(anchor="MON", ranking_frame=frame,
                             construction=construction, book_weighting=weighting,
                             funding_weight=0.5, factors=None)
                sharpe, res, sm = evaluate(cfg, weekly, factor_raw)
                rows.append({"frame": frame, "construction": construction,
                             "weighting": weighting, "sharpe": round(sharpe, 3)})
    save_csv("table6", rows)
    # matched contrast: books vs blend at fixed frame+weighting (risk_parity)
    print("--- books vs blend (fixed TS, risk_parity) ---")
    for frame in ["XS", "TS"]:
        for w in ["equal", "risk_parity"]:
            b = next(r["sharpe"] for r in rows
                     if r["frame"] == frame and r["construction"] == "books" and r["weighting"] == w)
            bl = next(r["sharpe"] for r in rows
                      if r["frame"] == frame and r["construction"] == "blend" and r["weighting"] == w)
            print(f"{frame} {w:12s} books={b:.3f} blend={bl:.3f}")


if __name__ == "__main__":
    run()
