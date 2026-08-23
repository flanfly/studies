"""Table 13 — deflated Sharpe from the registry log."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import registry
from experiments.util import save_csv

# Deflated Sharpe Ratio (Bailey & López de Prado):
#   DSR = 1 - Phi( (sqrt(T)-1)*SR / sqrt(1-gamma4*SR^2)
#                  ... full form below using the max-SR corrected expectation )
def deflated_sharpe(sr, T, N):
    from scipy.stats import norm
    import numpy as np
    if N < 2:
        return np.nan
    e_max = np.sqrt(np.log(N)) / np.sqrt(T) if T > 0 else np.nan
    # approximate: SR under H0 ~ N(0,1)/sqrt(T), max over N ~ sqrt(log N)/sqrt(T)
    x = (sr - e_max) * np.sqrt(T)
    return float(norm.cdf(x))


def run():
    stats = registry.stats()
    df = registry.load()
    # best observed Sharpe among all logged configs (the full eleven-factor TS book)
    best = df["sharpe"].max()
    T = 347  # panel weeks
    N = stats["n_configs"]
    dsr = deflated_sharpe(best, T, N)
    rows = [{"n_configs": N, "sd_sharpe": stats["sd_sharpe"],
             "best_sharpe": round(best, 3), "dsr": round(dsr, 3)}]
    save_csv("table13", rows)
    print(f"N={N} sd(SR)={stats['sd_sharpe']:.3f} best={best:.3f} DSR={dsr:.3f}")


if __name__ == "__main__":
    run()
