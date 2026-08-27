"""build_daily.py — split 5m into per-symbol parquet, compute daily factor panels.

Memory-safe strategy:
  1. Load the full 5m panel once, write one parquet per symbol under SCRATCH.
     Then free the big frame.
  2. Spawn workers (parent holds no big frame); each worker reads only its own
     symbols' parquet files and writes per-symbol daily factor parquet.
  3. Combine per-symbol daily results into daily factor panels (date x symbol),
     cached under cache/daily/<key>/*.parquet.

Daily factor values are anchor-independent, so this runs once and every weekly
(re)aggregation for any anchor reuses them. The cache directory is keyed by the
factor-relevant config subset, so sweeping q_top_frac / tskd_min_bars_per_side /
tskd_diff_order rebuilds the cache instead of silently reusing stale panels.
"""
from __future__ import annotations

import os
import gc
import numpy as np
import pandas as pd

from factors import _compute_symbol_daily, DAILY_FACTORS
from config import Config
from paths import SCRATCH, CACHE_ROOT

SYM_DIR = os.path.join(SCRATCH, "symbols")
_CFG = None


def daily_cache_key(cfg):
    return cfg.factor_cache_key()


def daily_dirs(cfg):
    key = daily_cache_key(cfg)
    return (os.path.join(SCRATCH, "daily", key),
            os.path.join(CACHE_ROOT, "daily", key))


def split_symbols(symbols):
    """Write per-symbol parquet files from the full 5m panel."""
    from data_load import get_5m
    os.makedirs(SYM_DIR, exist_ok=True)
    p5 = get_5m()
    cols = ["open_time", "close", "volume", "trades", "taker_buy_base"]
    for i, sym in enumerate(symbols):
        out = os.path.join(SYM_DIR, f"{sym}.parquet")
        if os.path.exists(out):
            continue
        sub = p5[p5["symbol"] == sym]
        sub[cols].to_parquet(out)
        if (i + 1) % 100 == 0:
            print(f"  split {i+1}/{len(symbols)}", flush=True)
    return symbols


def _worker_daily(chunk):
    """Compute + write daily factors for a chunk of (symbols, daily_dir, cfg)."""
    symbols, daily_dir, cfg = chunk
    nrows = 0
    for sym in symbols:
        out = os.path.join(daily_dir, f"{sym}.parquet")
        if os.path.exists(out):
            continue
        path = os.path.join(SYM_DIR, f"{sym}.parquet")
        df = pd.read_parquet(path)
        if len(df) == 0:
            continue
        df = df.sort_values("open_time")
        nrows += len(df)
        res = _compute_symbol_daily(df, cfg)
        rec = {"date": res["Q"][0]}
        for f in DAILY_FACTORS:
            if f in res:
                rec[f] = res[f][1]
        pd.DataFrame(rec).to_parquet(out)
    return nrows


def build_daily_panels(symbols, cfg, nprocs=12):
    """Split (if needed) and compute daily factor panels.

    Returns dict factor -> daily DataFrame (date x symbol).
    """
    daily_dir, cache_daily = daily_dirs(cfg)
    os.makedirs(SYM_DIR, exist_ok=True)
    os.makedirs(daily_dir, exist_ok=True)
    os.makedirs(cache_daily, exist_ok=True)

    if any(not os.path.exists(os.path.join(SYM_DIR, f"{s}.parquet")) for s in symbols):
        print("splitting 5m into per-symbol parquet ...", flush=True)
        split_symbols(symbols)
        import data_load
        data_load._5M = None
        gc.collect()

    missing = [s for s in symbols
               if not os.path.exists(os.path.join(daily_dir, f"{s}.parquet"))]
    print(f"computing daily factors for {len(missing)} symbols (nprocs={nprocs})", flush=True)
    if missing:
        chunks = list(np.array_split(np.array(missing, dtype=object),
                                     min(nprocs, len(missing))))
        chunks = [list(c) for c in chunks]
        # cfg travels inside the chunk tuple so this works under fork and spawn
        tasks = [(c, daily_dir, cfg) for c in chunks]
        import multiprocessing as mp
        if len(tasks) > 1:
            with mp.get_context("fork").Pool(nprocs) as pool:
                res = pool.map(_worker_daily, tasks)
        else:
            res = [_worker_daily(tasks[0])]
        print("daily rows processed:", sum(res), flush=True)

    print("combining daily panels ...", flush=True)
    per_factor = {f: {} for f in DAILY_FACTORS}
    for sym in symbols:
        path = os.path.join(daily_dir, f"{sym}.parquet")
        if not os.path.exists(path):
            continue
        d = pd.read_parquet(path).set_index("date")
        for f in DAILY_FACTORS:
            if f in d.columns:
                per_factor[f][sym] = d[f]

    daily = {}
    for f in DAILY_FACTORS:
        df = pd.DataFrame(per_factor[f])
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        daily[f] = df
        df.to_parquet(os.path.join(cache_daily, f"{f}.parquet"))
    return daily


def load_daily_panels(cfg=None):
    """Load cached daily panels for a config (or the default key), else None."""
    if cfg is None:
        cfg = Config()
    _, cache_daily = daily_dirs(cfg)
    if not os.path.isdir(cache_daily) or not os.listdir(cache_daily):
        return None
    daily = {}
    for f in DAILY_FACTORS:
        p = os.path.join(cache_daily, f"{f}.parquet")
        if os.path.exists(p):
            daily[f] = pd.read_parquet(p)
    return daily