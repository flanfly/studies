"""Table 5 — carry contribution: per-factor Sharpe fw=0.5 vs 0.0 + funding-only book."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from config import Config
from experiments.util import ensure, evaluate, save_csv
from ranking import rank_xs
from portfolio import build_book
from returns import build_returns
import metrics as M

FACTORS = ["AVOL", "Q", "RSJ", "OFI", "CPVm", "CPVv",
           "WRspread", "TopChg", "Quad", "TKU", "TSKD"]


def funding_only_book(weekly, cfg):
    """A book formed purely on the cross-sectional funding score (carry standalone).

    Net funding receiver: long LOW-funding, short HIGH-funding, i.e. formed on
    -s_XS(funding). Positive funding = longs pay, so shorting high-funding
    makes the book a net funding receiver (matching the sign verification).
    """
    fund_score, _ = rank_xs(weekly["funding_w"])
    w = build_book(-fund_score, weekly["fwd_ret_w"],
                   cfg.quintile_frac, cfg.min_cross_section)
    funding_shift = weekly["funding_w"].shift(-1)
    r, _, _ = build_returns(w, weekly["fwd_ret_w"], funding_shift, weekly["adv_w"], cfg)
    return M.annualized_sharpe(r), M.t_stat(r)


def run():
    weekly, factor_raw, symbols = ensure("MON")
    rows = []
    for f in FACTORS:
        s0, r0, _ = evaluate(Config(anchor="MON", ranking_frame="TS",
                                    funding_weight=0.5, factors=(f,)), weekly, factor_raw)
        s1, r1, _ = evaluate(Config(anchor="MON", ranking_frame="TS",
                                    funding_weight=0.0, factors=(f,)), weekly, factor_raw)
        rows.append({"factor": f, "sharpe_fw05": round(s0, 3),
                     "sharpe_fw00": round(s1, 3), "carry_delta": round(s0 - s1, 3)})
    cfg = Config(anchor="MON", ranking_frame="TS")
    sh_fund, t_fund = funding_only_book(weekly, cfg)
    rows.append({"factor": "FUNDING-ONLY", "sharpe_fw05": round(sh_fund, 3),
                 "sharpe_fw00": np.nan, "carry_delta": np.nan})
    save_csv("table5", rows)
    for r in rows:
        print(f"{r['factor']:12s} fw.5={r['sharpe_fw05']}  fw0={r['sharpe_fw00']}  delta={r['carry_delta']}")
    print(f"funding-only Sharpe {sh_fund:.3f}  t={t_fund:.2f}")


if __name__ == "__main__":
    run()
