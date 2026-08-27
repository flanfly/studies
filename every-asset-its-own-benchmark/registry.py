"""registry.py — append-only, machine-written log of evaluated configs (Table 13).

Every evaluated configuration is a trial. Rows are appended under an exclusive
flock with fsync so a crash mid-write truncates nothing and the read-modify-
write race under mp.Pool is gone. Sharpe is stored WEEKLY (per-period), the
form that the DSR formula consumes, not annualised.
"""
from __future__ import annotations

import os
import csv
import json
import fcntl
import pandas as pd

from paths import REGISTRY

FIELDS = ["ts", "hash", "config_json", "sharpe_weekly", "n_weeks", "anchor", "context"]


def log(config, sharpe_weekly, n_weeks=None, anchor=None, context="", extra=None):
    """Append one trial row under an exclusive lock + fsync."""
    d = config.as_dict()
    row = {
        "ts": pd.Timestamp.utcnow().isoformat(),
        "hash": config.hash(),
        "config_json": json.dumps(d, sort_keys=True),
        "sharpe_weekly": sharpe_weekly,
        "n_weeks": int(n_weeks) if n_weeks is not None else None,
        "anchor": anchor if anchor is not None else d.get("anchor", ""),
        "context": context,
    }
    if extra:
        for k, v in extra.items():
            row[k] = v
    new = not os.path.exists(REGISTRY)
    os.makedirs(os.path.dirname(REGISTRY), exist_ok=True)
    with open(REGISTRY, "a", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        wr = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            wr.writeheader()
        wr.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())
        fcntl.flock(fh, fcntl.LOCK_UN)


def load():
    if os.path.exists(REGISTRY):
        return pd.read_csv(REGISTRY)
    return pd.DataFrame(columns=FIELDS)


def stats():
    """DSR trial count N and dispersion sd(SR_weekly) from the log."""
    df = load()
    return {"n_configs": int(len(df)),
            "sd_sharpe": float(df["sharpe_weekly"].dropna().std())}