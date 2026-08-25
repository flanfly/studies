"""Hedged carry portfolio — short leg of the deployed factor book, spot-hedged.

Take only the short positions of the combined factor book (TS / books /
risk_parity, the deployed baseline from ../README.md), hedge each 1-for-1 with
spot, and count only the funding carry collected. Because every unit of short
notional needs an equal unit of spot held as collateral, the return per unit of
deployed capital is half the raw carry.

    carry_gross = -(w_short * funding_shift).sum(axis=1)   # collected on shorts
    r_hedged    = 0.5 * carry_gross                        # 1:1 spot hedge

Price PnL is assumed fully hedged away and is not counted (per spec).

Outputs (in out/):
  summary.csv                  overall stats
  yearly.csv                   per-year breakdown
  weekly_returns.csv           weekly hedged carry returns
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config                    # noqa: E402
from pipeline import load_panels, run_pipeline  # noqa: E402
import metrics as M                          # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

HEDGE_SCALE = 0.5  # 1 unit short notional needs 1 unit spot collateral


def main() -> None:
    weekly, factor, syms = load_panels("MON")
    cfg = Config(anchor="MON", ranking_frame="TS", construction="books",
                 book_weighting="risk_parity")
    res = run_pipeline(cfg, weekly, factor)
    w = res["weights"]

    funding_shift = weekly["funding_w"].shift(-1)      # funding over holding week
    w_short = w.clip(upper=0.0)                        # short legs only (negative)

    short_gross = w_short.abs().sum(axis=1)            # short notional per week
    carry_gross = -(w_short * funding_shift).sum(axis=1)  # carry collected on shorts
    r = (HEDGE_SCALE * carry_gross).dropna()           # hedged carry return

    # ---- overall stats ----
    active = r.index[short_gross.reindex(r.index) > 0]
    n_active = len(active)
    total = (1 + r).prod() - 1
    ann_ret = M.annualized_return(r)
    ann_vol = M.annualized_vol(r)
    sharpe = M.annualized_sharpe(r)
    t = M.t_stat(r)
    maxdd = M.max_drawdown(r)
    recap = M.weeks_to_recover(r)

    summary = pd.DataFrame({
        "metric": ["total_return", "annualized_return", "annualized_vol",
                   "sharpe", "t_stat", "max_drawdown", "weeks_to_recover",
                   "n_weeks", "n_active_weeks", "mean_weekly_return_pct",
                   "positive_week_pct", "mean_short_gross", "hedge_scale",
                   "mean_weekly_carry_gross_pct"],
        "value": [total, ann_ret, ann_vol, sharpe, t, maxdd, recap,
                  len(r), n_active, float(r.mean() * 100),
                  100 * float((r > 0).mean()), float(short_gross.mean()),
                  HEDGE_SCALE, float(carry_gross.dropna().mean() * 100)],
    })
    summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    # ---- yearly breakdown ----
    rows = []
    for yr, g in r.groupby(r.index.year):
        g = g.dropna()
        mkt_f = weekly["funding_w"].reindex(g.index).mean().mean()
        rows.append({
            "year": yr,
            "weeks": len(g),
            "return": float((1 + g).prod() - 1),
            "ann_return": float(M.annualized_return(g)),
            "ann_vol": float(M.annualized_vol(g)),
            "sharpe": float(M.annualized_sharpe(g)),
            "max_drawdown": float(M.max_drawdown(g)),
            "mean_week_pct": float(g.mean() * 100),
            "positive_week_pct": float((g > 0).mean()) * 100,
            "market_funding_avg": float(mkt_f),
        })
    yearly = pd.DataFrame(rows)
    yearly.to_csv(os.path.join(OUT, "yearly.csv"), index=False)

    weekly_out = pd.DataFrame({
        "week": r.index.strftime("%Y-%m-%d"),
        "short_gross": short_gross.reindex(r.index),
        "carry_gross": carry_gross.reindex(r.index),
        "hedged_return": r.values,
    })
    weekly_out.to_csv(os.path.join(OUT, "weekly_returns.csv"), index=False)

    print("=== HEDGED CARRY PORTFOLIO (MON anchor, deployed book short leg) ===")
    print(f"total return (compound): {total:+.2%} over {len(r)} weeks")
    print(f"annualized return: {ann_ret:+.2%} | vol: {ann_vol:.2%} | Sharpe: {sharpe:.2f} (t={t:.2f})")
    print(f"max drawdown: {maxdd:.2%} | weeks to recover: {recap}")
    print(f"active weeks: {n_active}/{len(r)} | mean weekly: {r.mean():+.3%} | pos weeks: {(r>0).mean():.1%}")
    print()
    print(yearly.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()