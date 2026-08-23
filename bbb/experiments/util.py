"""experiments/util.py — shared helpers for the table experiments."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from config import Config
from pipeline import load_panels, run_pipeline
import metrics as M
import registry

OUT = "/home/kai/studies/bbb/experiments/out"
os.makedirs(OUT, exist_ok=True)


def panels(anchor="MON"):
    weekly, factor_raw, symbols = load_panels(anchor)
    return weekly, factor_raw, symbols


def evaluate(cfg, weekly, factor_raw, log=True):
    """Run a config, log to registry, return (sharpe, result, summary)."""
    res = run_pipeline(cfg, weekly, factor_raw)
    r = res["returns"]
    sharpe = M.annualized_sharpe(r)
    if log:
        registry.log(cfg, sharpe, extra={"anchor": cfg.anchor,
                                        "n_weeks": int(r.dropna().shape[0])})
    sm = M.summary(r, res["turnover"])
    return sharpe, res, sm


def save_csv(name, rows):
    out = os.path.join(OUT, f"{name}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")
    return out


def ensure(anchor="MON"):
    weekly, factor_raw, symbols = load_panels(anchor)
    return weekly, factor_raw, symbols
