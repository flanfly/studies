"""Walk-forward ablation: select the factor subset on each 104-week training
block, evaluate on the next 26 weeks, roll.

Method (leave-one-out selection, matching the ablation study):
  * Setup: no survivorship screen (853 symbols), clip_forward_return = None.
  * Fold k: train = weeks [t_k, t_k+104), test = weeks [t_k+104, t_k+130);
    roll by 26 weeks.
  * On each training block we run the full pipeline for 12 candidates
    (all-11 factors + each of the 11 leave-one-out removals) and select the
    candidate with the best full-block training Sharpe (ties broken toward
    more factors / the all-11 baseline).
  * We then evaluate that selected subset on the held-out next-26-week block,
    and the same block under the fixed all-11 model.

The question this answers: is the factor-subset choice stable out-of-sample?
If the same subset wins fold after fold (and beats all-11 out of sample), the
selection is real. If it churns every fold / wins on noise margins, the honest
conclusion is to keep all eleven factors.
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ALL_FACTORS = ["AVOL", "Q", "RSJ", "OFI", "CPVm", "CPVv",
               "WRspread", "TopChg", "Quad", "TKU", "TSKD"]

TRAIN_WEEKS = 104
TEST_WEEKS = 26
STEP = 26


def build_panels(anchor, nprocs):
    cfg = Config(anchor=anchor,
                 require_continuous_trading=False,
                 clip_forward_return=None,          # no clip
                 nprocs=nprocs)
    p5 = get_5m()
    weekly_all = build_weekly(p5, anchor)
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
    return weekly, factor


def fold_bounds(n):
    """(train_start, train_end, test_end) per fold; test length <= TEST_WEEKS."""
    bounds = []
    i = 0
    while i + TRAIN_WEEKS < n:
        te = min(i + TRAIN_WEEKS + TEST_WEEKS, n)
        bounds.append((i, i + TRAIN_WEEKS, te))
        i += STEP
    return bounds


def safe_sharpe(r):
    s = M.annualized_sharpe(r)
    return s if np.isfinite(s) else 0.0


def candidate_list():
    cands = [("all_11", None, 11)]
    for f in ALL_FACTORS:
        fs = tuple(x for x in ALL_FACTORS if x != f)
        cands.append((f"no_{f}", fs, len(fs)))
    return cands


def pick_best(cand_scores):
    """cand_scores: list of (label, n_factors, train_sharpe)."""
    best = max(c for _, _, c in cand_scores)
    tied = [(l, n, c) for l, n, c in cand_scores if abs(c - best) < 1e-9]
    # prefer more factors, then the all-11 baseline
    tied.sort(key=lambda x: (-x[1], 0 if x[0] == "all_11" else 1))
    return tied[0][0]


def main():
    anchor = "MON"
    weekly, factor = build_panels(anchor, 12)
    weeks = list(weekly["close_w"].index)
    n = len(weeks)
    bounds = fold_bounds(n)
    print(f"universe: {len(weekly['close_w'].columns)} symbols | {n} weeks | "
          f"{len(bounds)} folds (train {TRAIN_WEEKS}w, test <= {TEST_WEEKS}w)", flush=True)

    base = dict(anchor=anchor,
                require_continuous_trading=False,
                clip_forward_return=None,
                funding_weight=0.5,
                construction="books",
                book_weighting="risk_parity",
                nprocs=12)
    cands = candidate_list()

    # cache pipeline results per (fold, candidate) — cheap but keeps memory small
    fold_rows = []
    cand_rows = []
    selected_test_series = {}   # label -> concatenated test returns (chronological)
    all11_test = []

    for fold, (t0, t1, t2) in enumerate(bounds):
        tr = slice(t0, t1)
        te = slice(t1, t2)
        win = f"{weeks[t0].date()} → {weeks[t2-1].date()}"
        train_win = f"{weeks[t0].date()} → {weeks[t1-1].date()}"
        test_win = f"{weeks[t1].date()} → {weeks[t2-1].date()}"

        scores = []
        for label, factors_tuple, nf in cands:
            cfg = Config(**base, factors=factors_tuple)
            res = run_pipeline(cfg, weekly, factor)
            r = res["returns"]
            w = res["weights"]
            rt = r.iloc[tr]
            re_ = r.iloc[te]
            ts = safe_sharpe(rt)
            es = safe_sharpe(re_)
            scores.append((label, nf, ts))
            active_tr = int((w.iloc[tr].abs().sum(axis=1) > 0).sum())
            active_te = int((w.iloc[te].abs().sum(axis=1) > 0).sum())
            cand_rows.append({
                "fold": fold, "window": win,
                "train_weeks": train_win, "test_weeks": test_win,
                "candidate": label, "n_factors": nf,
                "removed": None if label == "all_11" else label[3:],
                "train_sharpe": ts, "test_sharpe": es,
                "train_ann_ret": float(M.annualized_return(rt)),
                "test_ann_ret": float(M.annualized_return(re_)),
                "test_ann_vol": float(M.annualized_vol(re_)),
                "test_max_dd": float(M.max_drawdown(re_)),
                "active_train": active_tr, "active_test": active_te,
                "n_train": int(rt.dropna().shape[0]), "n_test": int(re_.dropna().shape[0]),
            })
            if label == "all_11":
                all11_test.append(re_.rename(fold))

        chosen = pick_best(scores)
        sorted_sc = sorted(scores, key=lambda x: -x[2])
        second = sorted_sc[1][2]
        margin = sorted_sc[0][2] - second

        # chosen candidate's test returns, appended chronologically
        chosen_cfg = Config(**base, factors=dict((l, f) for l, f, _ in cands)[chosen])
        res_c = run_pipeline(chosen_cfg, weekly, factor)
        rc_test = res_c["returns"].iloc[te]
        selected_test_series.setdefault(chosen, []).append((t1, rc_test))
        wc = res_c["weights"]
        rc_train = res_c["returns"].iloc[tr]

        fold_rows.append({
            "fold": fold,
            "train_weeks": train_win,
            "test_weeks": test_win,
            "n_train_weeks": t1 - t0,
            "n_test_weeks": t2 - t1,
            "chosen": chosen,
            "removed": None if chosen == "all_11" else chosen[3:],
            "chosen_train_sharpe": safe_sharpe(rc_train),
            "chosen_train_active": int((wc.iloc[tr].abs().sum(axis=1) > 0).sum()),
            "chosen_test_sharpe": safe_sharpe(rc_test),
            "chosen_test_ann_ret": float(M.annualized_return(rc_test)),
            "chosen_test_ann_vol": float(M.annualized_vol(rc_test)),
            "chosen_test_max_dd": float(M.max_drawdown(rc_test)),
            "chosen_test_active": int((wc.iloc[te].abs().sum(axis=1) > 0).sum()),
            "all11_test_sharpe": safe_sharpe(pd.concat(all11_test[-1:]) if all11_test else rc_test),
            "oracle_test_sharpe": max(c["test_sharpe"] for c in cand_rows if c["fold"] == fold),
            "train_margin_best2nd": margin,
            "n_near_ties_0.05": sum(1 for l, nf, ts in scores if ts >= sorted_sc[0][2] - 0.05),
        })

    folds = pd.DataFrame(fold_rows)
    cands_df = pd.DataFrame(cand_rows)
    folds.to_csv(os.path.join(OUT, "folds.csv"), index=False)
    cands_df.to_csv(os.path.join(OUT, "candidates.csv"), index=False)

    # ---- stability stats ----
    chosen_seq = folds["chosen"].tolist()
    distinct = sorted(set(chosen_seq))
    churn = sum(1 for a, b in zip(chosen_seq, chosen_seq[1:]) if a != b)
    removal_freq = {}
    for f in ALL_FACTORS:
        removal_freq[f] = int((folds["removed"] == f).sum())
    most_freq = max(removal_freq, key=removal_freq.get)

    # ---- walk-forward out-of-sample series ----
    sel_series = pd.concat([s for _, s in sorted(
        [(t1, rc) for label, lst in selected_test_series.items() for t1, rc in lst],
        key=lambda x: x[0])])   # already datetime-indexed
    all11_series = pd.concat(all11_test)   # already datetime-indexed

    stats = {
        "n_folds": len(folds),
        "distinct_chosen": len(distinct),
        "chosen_list": "|".join(chosen_seq),
        "churn_count": churn,
        "churn_rate": churn / (len(folds) - 1) if len(folds) > 1 else np.nan,
        "most_frequent_removal": most_freq,
        "removal_freq": ";".join(f"{f}={removal_freq[f]}" for f in ALL_FACTORS),
        "wf_selected_sharpe": safe_sharpe(sel_series),
        "wf_all11_sharpe": safe_sharpe(all11_series),
        "wf_selected_ann_ret": float(M.annualized_return(sel_series)),
        "wf_all11_ann_ret": float(M.annualized_return(all11_series)),
        "wf_selected_max_dd": float(M.max_drawdown(sel_series)),
        "wf_all11_max_dd": float(M.max_drawdown(all11_series)),
        "mean_chosen_test_sharpe": float(folds["chosen_test_sharpe"].mean()),
        "mean_all11_test_sharpe": float(folds["all11_test_sharpe"].mean()),
        "mean_oracle_test_sharpe": float(folds["oracle_test_sharpe"].mean()),
        "mean_train_margin": float(folds["train_margin_best2nd"].mean()),
    }
    stats_df = pd.DataFrame({"metric": list(stats), "value": list(stats.values())})
    stats_df.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    # ---- print ----
    print("\n=== per-fold selection ===")
    print(folds[["fold", "train_weeks", "test_weeks", "chosen", "removed",
                 "chosen_train_sharpe", "chosen_test_sharpe", "all11_test_sharpe",
                 "oracle_test_sharpe", "train_margin_best2nd"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== stability ===")
    print(f"distinct chosen subsets : {distinct}")
    print(f"churn (fold-to-fold)    : {churn}/{len(folds)-1}")
    print(f"removal frequency       : {removal_freq}")
    print(f"most frequent removal   : {most_freq} ({removal_freq[most_freq]}/{len(folds)} folds)")
    print("\n=== walk-forward out-of-sample (concatenated test blocks) ===")
    print(f"selected subset : Sharpe {stats['wf_selected_sharpe']:.3f} | ann {stats['wf_selected_ann_ret']:+.2%} | maxDD {stats['wf_selected_max_dd']:.2%}")
    print(f"all 11 fixed    : Sharpe {stats['wf_all11_sharpe']:.3f} | ann {stats['wf_all11_ann_ret']:+.2%} | maxDD {stats['wf_all11_max_dd']:.2%}")
    print(f"oracle (peek)   : mean fold test Sharpe {stats['mean_oracle_test_sharpe']:.3f}")
    print(f"mean train margin best-vs-2nd : {stats['mean_train_margin']:.3f}")
    pd.DataFrame({"week": sel_series.index.strftime("%Y-%m-%d"),
                  "selected_return": sel_series.values,
                  "all11_return": all11_series.values}) \
        .to_csv(os.path.join(OUT, "wf_test_returns.csv"), index=False)


if __name__ == "__main__":
    main()