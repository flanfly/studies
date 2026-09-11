# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0", "numpy>=1.26"]
# ///
import math
import numpy as np
import polars as pl

PPY = 730

def metrics(r):
    r = r[np.isfinite(r)]
    ann = math.sqrt(PPY)
    mu, sd = r.mean(), r.std(ddof=1)
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    return dict(cum=eq[-1]-1, vol=sd*ann, mdd=float(np.min(eq/peak-1)), sharpe=mu/sd*ann)

def get(file, strat, col='ret_gross'):
    try:
        d = pl.read_parquet(file).filter(pl.col('strategy') == strat).sort('time')
        return d[col].to_numpy(), d['turnover'].to_numpy()
    except Exception:
        return None, None

CONFIGS = [("1e-5 (paper)", "seed-t15-{s}.parquet"),
           ("1e-4, 1 pass", "lr1e-4-p1-s{s}.parquet"),
           ("1e-4, 10 passes", "lr1e-4-p10-s{s}.parquet"),
           ("1e-3, 1 pass", "lr1e-3-p1-s{s}.parquet"),
           ("1e-3, 10 passes", "lr1e-3-p10-s{s}.parquet")]
SEEDS = [8, 4, 42]
STRATS = ["DRL_rank", "DRL+filt:EW", "DRL+filt:IV", "DRL+filt:MaxDiv"]

for label, pat in CONFIGS:
    rows = []
    for k in STRATS:
        shs, cums = [], []
        for s in SEEDS:
            r, to = get(pat.format(s=s), k)
            if r is None:
                continue
            m = metrics(r)
            shs.append(m['sharpe']); cums.append(m['cum'])
        if not shs:
            continue
        rows.append(f"{k:18s} sharpe={np.mean(shs):+5.2f} (per-seed {' '.join(f'{x:+.2f}' for x in shs)})  cum mean={np.mean(cums):+7.1%}")
    print(f"--- lr {label} ---")
    print("\n".join(rows))
