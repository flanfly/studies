"""Table 9 — equity-curve properties + Nov 2021 - Dec 2022 sub-window."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from config import Config
from experiments.util import ensure, evaluate, save_csv
import metrics as M

SUBWIN = ("2021-11-01", "2023-01-01")


def run():
    weekly, factor_raw, symbols = ensure("MON")
    cfg = Config(anchor="MON", ranking_frame="TS", funding_weight=0.5)
    sharpe, res, sm = evaluate(cfg, weekly, factor_raw)
    r = res["returns"]

    # equal-weight market return of the tradeable universe
    fwd = weekly["fwd_ret_w"].clip(upper=cfg.clip_forward_return)
    mkt = fwd.mean(axis=1)

    sub = r[(r.index >= SUBWIN[0]) & (r.index <= SUBWIN[1])]
    mkt_sub = mkt[(mkt.index >= SUBWIN[0]) & (mkt.index <= SUBWIN[1])]
    corr = np.corrcoef(sub.dropna(), mkt_sub.reindex(sub.index).dropna())[0, 1]

    props = {
        "cum_return": float((1 + r.dropna()).prod() - 1),
        "ann_return": sm["ann_return"],
        "ann_vol": sm["ann_vol"],
        "sharpe": sm["sharpe"],
        "max_dd": sm["max_dd"],
        "recover_weeks": sm["recover_weeks"],
        "worst_week": sm["worst_week"],
        "worst_month": float(r.dropna().resample("ME").sum().min()),
        "pos_week_pct": sm["pos_week_pct"],
        "skew": sm["skew"],
        "excess_kurt": sm["excess_kurt"],
        "n_weeks": sm["n_weeks"],
        "sub_start": str(sub.dropna().index[0]),
        "sub_end": str(sub.dropna().index[-1]),
        "sub_n_weeks": int(sub.dropna().shape[0]),
        "sub_ann_return": round(M.annualized_return(sub), 4),
        "sub_ann_vol": round(M.annualized_vol(sub), 4),
        "sub_sharpe": round(M.annualized_sharpe(sub), 4),
        "sub_max_dd": round(M.max_drawdown(sub), 4),
        "mkt_sub_ann_return": round(M.annualized_return(mkt_sub), 4),
        "corr_market": round(corr, 3) if np.isfinite(corr) else np.nan,
    }
    save_csv("table9", [props])
    for k, v in props.items():
        print(f"{k:20s} = {v}")


if __name__ == "__main__":
    run()
