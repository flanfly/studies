"""e2e: driver used to confirm the survivorship screen is a lookahead leak
and to record Phase 1 / Phase 2 baselines. Run from repo root."""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import Config
from data_load import get_5m, get_funding, apply_universe_screen
from resample import build_weekly, build_weekly_funding
from build_daily import build_daily_panels
from factors import build_weekly_raw_factors_from_daily
from pipeline import run_pipeline
import metrics as M
import registry


def run_cfg_conf(anchor, rct, clip, book_terminal_return=True, nprocs=12, log_context=""):
    cfg = Config(anchor=anchor, require_continuous_trading=rct,
                 clip_forward_return=clip, book_terminal_return=book_terminal_return,
                 nprocs=nprocs)
    p5 = get_5m()
    weekly_all = build_weekly(p5, anchor, book_terminal_return)
    syms = apply_universe_screen(weekly_all["close_w"],
                                 cfg.require_continuous_trading,
                                 cfg.require_finite_positive_prices)
    weekly = {k: df.reindex(columns=syms) for k, df in weekly_all.items()}
    fw = build_weekly_funding(get_funding(), anchor)
    weekly["funding_w"] = fw.reindex(index=weekly["close_w"].index, columns=syms)
    daily = build_daily_panels(syms, cfg, nprocs)
    factor = build_weekly_raw_factors_from_daily(daily, syms, weekly, cfg)
    res = run_pipeline(cfg, weekly, factor)
    r = res["returns"]
    sm = M.summary(r, res["turnover"])
    rw = r.dropna()
    sw = float(rw.mean() / rw.std(ddof=1)) if len(rw) > 1 else float("nan")
    if log_context:
        registry.log(cfg, sw, n_weeks=len(rw), anchor=anchor, context=log_context)
    return {
        "n_symbols": len(syms),
        "n_weeks": sm["n_weeks"],
        "ann_return": sm["ann_return"],
        "ann_vol": sm["ann_vol"],
        "sharpe": sm["sharpe"],
        "t": M.t_stat(r),
        "t_nw": M.newey_west_t(r),
        "max_dd": sm["max_dd"],
        "recover_weeks": sm["recover_weeks"],
        "worst_week": sm["worst_week"],
        "skew": sm["skew"],
        "excess_kurt": sm["excess_kurt"],
        "mean_turnover": sm["mean_turnover"],
        "active_weeks": int(res["weights"].abs().sum(axis=1).gt(0).sum()),
    }


if __name__ == "__main__":
    pass