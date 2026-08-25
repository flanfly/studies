"""Full factor-book strategy re-run WITHOUT the survivorship screen and WITHOUT
the 100% forward-return clip; the 100% cap is kept enabled (+100%).

Differences vs the deployed baseline (root README / experiments/table9):
  * require_continuous_trading = False  -> drop the "must survive to last week"
    filter. We do NOT know ex-ante whether a symbol will trade in the future, so
    requiring it is look-ahead bias. Symbols are kept point-in-time as long as
    they have realisable prices (finite / positive).
  * clip_forward_return          = 1.0   -> weekly forward returns capped to
    +100% (baseline behaviour, kept).

Everything else matches the deployed full strategy in PLAN.md: TS ranking over
all 11 factors, per-factor dollar-neutral books, funding penalty (weight 0.5),
risk-parity combination, turnover cap 0.5, liquidity-scaled costs, MON anchor.

This is the FULL factor book (not the short-only pure-carry hedged leg).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from data_load import get_5m, get_funding, apply_universe_screen  # noqa: E402
from resample import build_weekly, build_weekly_funding           # noqa: E402
from build_daily import build_daily_panels                        # noqa: E402
from factors import build_weekly_raw_factors_from_daily           # noqa: E402
from config import Config                                         # noqa: E402
from pipeline import run_pipeline                                 # noqa: E402
import metrics as M                                               # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)


def main():
    anchor = "MON"
    cfg = Config(anchor=anchor,
                 require_continuous_trading=False,   # no survivorship look-ahead
                 clip_forward_return=1.0,            # re-enable the 100% clip
                 nprocs=12)

    p5 = get_5m()
    weekly_all = build_weekly(p5, anchor)
    symbols = apply_universe_screen(weekly_all["close_w"],
                                    require_continuous_trading=False,   # <-- the fix
                                    require_finite_positive_prices=True)
    weekly = {k: df.reindex(columns=symbols) for k, df in weekly_all.items()}

    funding = get_funding()
    funding_w = build_weekly_funding(funding, anchor)
    funding_w = funding_w.reindex(index=weekly["close_w"].index, columns=symbols)
    weekly["funding_w"] = funding_w

    print(f"universe symbols (no continuous filter): {len(symbols)}", flush=True)

    # daily factor panels for EVERY retained symbol (reuses cached per-symbol
    # daily files; computes factors for the delisted/mid-market symbols)
    daily = build_daily_panels(symbols, cfg, cfg.nprocs)

    # weekly raw factor panels
    factor = build_weekly_raw_factors_from_daily(daily, symbols, weekly, cfg)

    # run the full pipeline (no clip)
    res = run_pipeline(cfg, weekly, factor)
    r = res["returns"]
    w = res["weights"]
    sm = M.summary(r, res["turnover"])

    # ---- overall summary ----
    n_active = int((w.abs().sum(axis=1) > 0).sum())
    gross = float(w.abs().sum(axis=1).max())  # max weekly gross exposure
    net = float(w.sum(axis=1).abs().max())
    funding_term = -(w * weekly["funding_w"].shift(-1)).sum(axis=1).dropna()
    summary_rows = {
        "cfg": cfg.hash(),
        "anchor": anchor,
        "n_symbols": len(symbols),
        "n_weeks": sm["n_weeks"],
        "n_active_weeks": n_active,
        "cum_return": float((1 + r.dropna()).prod() - 1),
        "ann_return": sm["ann_return"],
        "ann_vol": sm["ann_vol"],
        "sharpe": sm["sharpe"],
        "t_stat": sm["t_stat"],
        "max_drawdown": sm["max_dd"],
        "recover_weeks": sm["recover_weeks"],
        "worst_week": sm["worst_week"],
        "pos_week_pct": sm["pos_week_pct"],
        "mean_turnover": sm["mean_turnover"],
        "gross_max": gross,
        "net_max": net,
        "mean_funding_term_week_pct": float(funding_term.mean() * 100),
        "funding_positive_week_pct": float((funding_term > 0).mean() * 100),
    }
    pd.DataFrame({"metric": list(summary_rows), "value": list(summary_rows.values())}) \
        .to_csv(os.path.join(OUT, "summary.csv"), index=False)

    # ---- yearly breakdown ----
    yearly_rows = []
    for yr, g in r.groupby(r.index.year):
        g = g.dropna()
        if len(g) == 0:
            continue
        yearly_rows.append({
            "year": yr,
            "weeks": len(g),
            "return": float((1 + g).prod() - 1),
            "ann_return": float(M.annualized_return(g)),
            "ann_vol": float(M.annualized_vol(g)),
            "sharpe": float(M.annualized_sharpe(g)),
            "max_drawdown": float(M.max_drawdown(g)),
            "mean_wk_pct": float(g.mean() * 100),
            "pos_wk_pct": float((g > 0).mean()) * 100,
        })
    yearly = pd.DataFrame(yearly_rows)
    yearly.to_csv(os.path.join(OUT, "yearly.csv"), index=False)

    # ---- weekly equity / returns / gross ----
    weekly_out = pd.DataFrame({
        "week": r.index.strftime("%Y-%m-%d"),
        "return": r.values,
        "equity": M.equity_curve(r).reindex(r.index).values,
        "gross_exposure": w.abs().sum(axis=1).reindex(r.index).values,
    })
    weekly_out.to_csv(os.path.join(OUT, "weekly_returns.csv"), index=False)

    cfg_hash = cfg.hash()
    print(f"config hash: {cfg_hash}")
    print(f"total return: {summary_rows['cum_return']:+.2%} over {sm['n_weeks']} weeks")
    print(f"ann return: {sm['ann_return']:+.2%} | vol: {sm['ann_vol']:.2%} | Sharpe: {sm['sharpe']:.2f} (t={sm['t_stat']:.2f})")
    print(f"max drawdown: {sm['max_dd']:.2%} | recover: {sm['recover_weeks']}w | turnover: {sm['mean_turnover']:.3f}")
    print(f"gross: {gross:.2f} | net: {net:.2e} | funding: {summary_rows['mean_funding_term_week_pct']:+.3f}%/wk")
    print(f"active weeks: {n_active}/{sm['n_weeks']}")
    print()
    print(yearly.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()