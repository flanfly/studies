"""registry.py — append-only log of every evaluated config + its Sharpe (Table 13).

Written by the evaluation code itself, not by hand. DSR trial count N and
dispersion sd(SR) are read from this log.
"""
from __future__ import annotations

import os
import json
import pandas as pd

REGISTRY = "/home/kai/studies/bbb/registry.csv"


def _load():
    if os.path.exists(REGISTRY):
        return pd.read_csv(REGISTRY)
    return pd.DataFrame(columns=["hash", "config_json", "sharpe", "anchor"])


def log(config, sharpe, extra=None):
    df = _load()
    d = config.as_dict()
    row = {"hash": config.hash(), "config_json": json.dumps(d, sort_keys=True),
           "sharpe": sharpe}
    if extra:
        for k, v in extra.items():
            row[k] = v
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(REGISTRY, index=False)


def load():
    return _load()


def stats():
    """DSR trial count N and dispersion sd(SR) from the log."""
    df = _load()
    return {"n_configs": int(len(df)), "sd_sharpe": float(df["sharpe"].std())}
