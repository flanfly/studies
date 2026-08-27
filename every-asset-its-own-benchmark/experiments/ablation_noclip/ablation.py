"""Ablation study — leave-one-factor-out on the FULL factor-book strategy.

Setup mirrors the `full_no_cont_screen` experiment (no survivorship screen, so
the universe is all 853 symbols), and the forward-return clip is DISABLED
(clip_forward_return = None), per the request to "remove the clip again".

For each factor:
  * baseline: all 11 factors (clip off, no screen)
  * ablation: same config with exactly one factor removed from the factor list.

We record the full metric set for each run and report which factors actually
move the result. The four positioning factors (WRspread, TopChg, Quad) and one
of the CPV variants are data-unavailable (all-NaN), so they show up as "no
effect"; the return-savvy factors move the numbers.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from data_load import get_5m, get_funding, apply_universe_screen  # noqa: E402
from resample import build_weekly, build_weekly_funding           # noqa: E402
from build_daily import build_daily_panels                        # noqa: E402
from factors import build_weekly_raw_factors_from_daily           # noqa: E402
from config import Config                                         # noqa: E402
from pipeline import run_pipeline                                 # noqa: E402
import metrics as M                                               # noqa: E402
import registry                                                    # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ALL_FACTORS = ["AVOL", "Q", "RSJ", "OFI", "CPVm", "CPVv",
               "WRspread", "TopChg", "Quad", "TKU", "TSKD"]


def build_panels(anchor, nprocs):
    cfg = Config(anchor=anchor,
                 require_continuous_trading=False,
                 clip_forward_return=None,          # no clip
                 nprocs=nprocs)
    p5 = get_5m()
    weekly_all = build_weekly(p5, anchor, cfg.book_terminal_return)
    symbols = apply_universe_screen(weekly_all["close_w"],
                                    require_continuous_trading=False,
                                    require_finite_positive_prices=True)
    weekly = {k: df.reindex(columns=symbols) for k, df in weekly_all.items()}
    funding = get_funding()
    funding_w = build_weekly_funding(funding, anchor)
    funding_w = funding_w.reindex(index=weekly["close_w"].index, columns=symbols)
    weekly["funding_w"] = funding_w
    daily = build_daily_panels(symbols, cfg, nprocs)
    factor = build_weekly_raw_factors_from_daily(daily, symbols, weekly, cfg)
    return weekly, factor, len(symbols)


def run(weekly, factor, cfg, label):
    res = run_pipeline(cfg, weekly, factor)
    r = res["returns"]
    w = res["weights"]
    sm = M.summary(r, res["turnover"])
    eq = M.equity_curve(r)
    gross = float(w.abs().sum(axis=1).max())
    net = float(w.sum(axis=1).abs().max())
    return {
        "experiment": label,
        "cfg": cfg.hash(),
        "n_factors": len(res["factors"]),
        "n_weeks": sm["n_weeks"],
        "cum_return": float((1 + r.dropna()).prod() - 1),
        "ann_return": sm["ann_return"],
        "ann_vol": sm["ann_vol"],
        "sharpe": sm["sharpe"],
        "t_stat": sm["t_stat"],
        "max_dd": sm["max_dd"],
        "recover_weeks": sm["recover_weeks"],
        "worst_week": sm["worst_week"],
        "pos_week_pct": sm["pos_week_pct"],
        "mean_turnover": sm["mean_turnover"],
        "gross_exposure_max": gross,
        "net_exposure_max": net,
        "active_weeks": int((w.abs().sum(axis=1) > 0).sum()),
    }


def main():
    anchor = "MON"
    weekly, factor, n_symbols = build_panels(anchor, 12)
    print(f"universe (no screen, no clip): {n_symbols} symbols\n", flush=True)

    def base_cfg(omit):
        return Config(anchor=anchor,
                      require_continuous_trading=False,
                      clip_forward_return=None,
                      funding_weight=0.5,
                      construction="books",
                      book_weighting="risk_parity",
                      factors=tuple(f for f in ALL_FACTORS if f != omit),
                      nprocs=12)

    rows = []

    # baseline: all factors (factors=None triggers the all-eleven path)
    cfg = Config(anchor=anchor,
                 require_continuous_trading=False,
                 clip_forward_return=None,
                 funding_weight=0.5,
                 construction="books",
                 book_weighting="risk_parity",
                 factors=None,
                 nprocs=12)
    rows.append(run(weekly, factor, cfg, "baseline_all"))

    # leave-one-out
    for f in ALL_FACTORS:
        c = base_cfg(f)
        r = run(weekly, factor, c, f"remove_{f}")
        rows.append(r)

    # log every evaluated config to the registry (Phase 4.3): each ablation is
    # a trial; context tags the trial set. weekly SR = annualised / sqrt(52).
    for row in rows:
        c = Config(anchor=anchor, require_continuous_trading=False,
                   clip_forward_return=None, funding_weight=0.5,
                   construction="books", book_weighting="risk_parity",
                   factors=None if row["experiment"] == "baseline_all"
                   else tuple(x for x in ALL_FACTORS if x != row["experiment"][7:]),
                   nprocs=12)
        sw = float(row["sharpe"] / np.sqrt(52.0))
        registry.log(c, sw, n_weeks=int(row["n_weeks"]), anchor=anchor,
                     context="ablation")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "ablation.csv"), index=False)

    cols = ["experiment", "n_factors", "sharpe", "ann_return", "ann_vol",
            "max_dd", "t_stat", "cum_return", "active_weeks"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)))


if __name__ == "__main__":
    main()