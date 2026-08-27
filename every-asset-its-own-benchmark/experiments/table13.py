"""Table 13 — deflated Sharpe from the registry log."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import registry
from experiments.util import save_csv

# Deflated Sharpe Ratio (Bailey & López de Prado):
#   DSR = 1 - Phi( (sqrt(T)-1)*SR / sqrt(1-gamma4*SR^2)
#                  ... full form below using the max-SR corrected expectation )
def deflated_sharpe(sr_weekly, T, N):
    from scipy.stats import norm
    import numpy as np
    if N < 2:
        return np.nan
    # per-period (weekly) SR under H0: N(0,1)/sqrt(T); max over N trials ~
    # sqrt(2*log N)/... using the standard Bailey-Lopez de Prado approximate
    # upward adjustment for the best of N.
    assert abs(sr_weekly) < 1.0, "Sharpe must be per-period (weekly), not annualised"
    e_max = np.sqrt(2.0 * np.log(float(N)) / float(T)) if T > 0 else np.nan
    x = (sr_weekly - e_max) * np.sqrt(T)
    return float(norm.cdf(x))


def run():
    stats = registry.stats()
    df = registry.load()
    cols = [c for c in ("sharpe_weekly", "sharpe") if c in df.columns]
    sr_col = "sharpe_weekly" if "sharpe_weekly" in cols else cols[0]
    best = df[sr_col].max()
    best_row = df.loc[df[sr_col].idxmax()]
    T = int(best_row["n_weeks"]) or int(df["n_weeks"].max())
    N_all = stats["n_configs"]

    def dsr_for(sub, label):
        if len(sub) < 2:
            return [{"trial_set": label, "n_configs": len(sub),
                     "sd_sharpe_weekly": float("nan"), "dsr": float("nan")}]
        b = sub[sr_col].max()
        b_row = sub.loc[sub[sr_col].idxmax()]
        tt = int(b_row["n_weeks"]) or T
        v = deflated_sharpe(b, tt, len(sub))
        return [{"trial_set": label, "n_configs": int(len(sub)),
                 "sd_sharpe_weekly": round(float(sub[sr_col].std()), 4),
                 "best_sharpe_weekly": round(b, 3),
                 "best_sharpe_annual": round(float(b * np.sqrt(52.0)), 3),
                 "dsr": round(v, 3)}]

    # strategy-search trial set = configs logged outside the walk-forward/ablation
    # exploration loops (empty context), i.e. the table experiments + baselines.
    search = df[df["context"].fillna("").isin(["", "phase1_olduni_clip",
                                                "phase1_olduni_noclip",
                                                "phase2_pit_clip", "phase2_pit_noclip"])]
    rows = dsr_for(df, "all_cells") + dsr_for(search, "strategy_search_only")
    save_csv("table13", rows)
    for r in rows:
        print(f"{r['trial_set']}: N={r['n_configs']} sd_w={r.get('sd_sharpe_weekly')} "
              f"best_ann={r.get('best_sharpe_annual')} DSR={r['dsr']}")


if __name__ == "__main__":
    run()
