"""checks.py — point-in-time discipline tests from §9 + remediation Phase 3.

Run with `python checks.py --anchor MON`; exits non-zero on any violation.

The six checks are the ones that can actually fail:
  1. leak test (panel truncation)
  2. determinism (single- vs multi-process daily factor build)
  3. dollar neutrality
  4. gross bound
  5. funding separation (behavioural perturbation, catches off-by-one)
  6. non-emptiness (evaluable vs active weeks)

Plus the end-to-end leak test (5m-level truncation) run on a symbol subset,
which with require_continuous_trading=True is EXPECTED to fail (the screen is
the lookahead) — that is the point of the test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config
from pipeline import load_panels, run_pipeline
from ranking import score_panel
from returns import build_returns, build_funding_shift
from factors import FACTOR_NAMES
from data_load import get_5m, apply_universe_screen
from resample import build_weekly
from factors import build_weekly_raw_factors_from_daily
from build_daily import build_daily_panels
from portfolio import combine_weights


def leak_test(weekly, factor_raw, cfg, sample=150, seed=0):
    """Recompute the factor score at (sym,t) from a panel truncated after week t."""
    rng = np.random.default_rng(seed)
    max_err = 0.0
    checked = 0
    for k in factor_raw:
        full_score = score_panel(factor_raw[k], cfg)
        valid = full_score.notna()
        pairs = np.argwhere(valid.values)
        if len(pairs) == 0:
            continue
        idx = rng.choice(len(pairs), size=min(sample, len(pairs)), replace=False)
        for r, c in pairs[idx]:
            t = full_score.index[r]
            sym = full_score.columns[c]
            sub = factor_raw[k].loc[:t]          # delete rows after week t
            s_trunc = score_panel(sub, cfg)
            v_full = full_score.at[t, sym]
            v_trunc = s_trunc.at[t, sym] if sym in s_trunc.columns else np.nan
            both = np.isfinite(v_full) and np.isfinite(v_trunc)
            if both:
                max_err = max(max_err, abs(v_full - v_trunc))
                checked += 1
            elif np.isfinite(v_full) != np.isfinite(v_trunc):
                max_err = 1.0
    return max_err, checked


def leak_test_end_to_end(cfg, anchor="MON", cut_weeks=(80, 160, 240), sample=40):
    """Rebuild the pipeline from 5m bars truncated at week t and assert weights
    at every week <= t are bit-identical to the full-panel run.

    Runs on a random symbol subset (sample) so it completes quickly. With
    require_continuous_trading=True this test should FAIL — the screen uses the
    panel's final week, which is future information relative to t.
    """
    p5 = get_5m()

    def run_from_5m(panel5m):
        weekly_all = build_weekly(panel5m, anchor, cfg.book_terminal_return)
        syms = apply_universe_screen(weekly_all["close_w"],
                                     cfg.require_continuous_trading,
                                     cfg.require_finite_positive_prices)
        weekly = {k: df.reindex(columns=syms) for k, df in weekly_all.items()}
        from data_load import get_funding
        from resample import build_weekly_funding
        fw = build_weekly_funding(get_funding(), anchor)
        weekly["funding_w"] = fw.reindex(index=weekly["close_w"].index, columns=syms)
        daily = build_daily_panels(syms, cfg, cfg.nprocs)
        factor = build_weekly_raw_factors_from_daily(daily, syms, weekly, cfg)
        res = run_pipeline(cfg, weekly, factor)
        return res["weights"]

    full_w = run_from_5m(p5)
    # Sample the subset from the truncated run's OWN universe so the comparison
    # only covers symbols observable at the cut point. Relisted symbols carry
    # stale funding under their old listing (e.g. BNT: funding in 2021, price
    # only from 2023) and are absent from the point-in-time universe; sampling
    # from the trunc universe keeps those out of the assertion.
    buf = cfg.vol_window_weeks + 2
    fails = []
    for t in cut_weeks:
        if t + buf > len(full_w.index):
            continue
        cut_ts = full_w.index[t]
        cut_ts_utc = pd.Timestamp(cut_ts).tz_localize("UTC")
        p5_trunc = p5[p5["open_time"] < cut_ts_utc + pd.Timedelta("7D")]
        trunc_w = run_from_5m(p5_trunc)
        trunc_cols = list(trunc_w.columns)
        rng2 = np.random.default_rng(0)
        subset = list(rng2.choice(trunc_cols, size=min(sample, len(trunc_cols)),
                                  replace=False))
        common = trunc_w.index[trunc_w.index <= cut_ts]
        common = common[:(t - buf)]   # drop boundary + vol-window weeks
        if len(common) == 0:
            continue
        f_ = full_w.loc[common, subset]
        t_ = trunc_w.loc[common, subset]
        # combine_weights is a dense zero frame within each run; symbols may be
        # 0.0 vs NaN across runs, treat as equal after fillna(0).
        if not np.allclose(f_.fillna(0).values, t_.fillna(0).values,
                           atol=0, rtol=0, equal_nan=True):
            fails.append(t)
    return fails


def run_checks(anchor="MON", cfg=None, run_e2e=False, e2e_expected_fail=False):
    cfg = cfg or Config(anchor=anchor)
    weekly, factor, symbols = load_panels(anchor, cfg)
    res = run_pipeline(cfg, weekly, factor)
    w = res["weights"]

    report = {}

    # 1. leak test (all eleven factors, panel truncation)
    err, nchecked = leak_test(weekly, factor, cfg)
    report["leak_checked"] = nchecked
    report["leak_max_abs_diff"] = err
    assert err <= 1e-12, f"LEAK: max abs diff {err}"

    # 2. determinism: single- vs multi-process daily factor build must agree.
    #    run_pipeline alone is a pure function and cannot fail; the chunking
    #    that can actually break is the parallel daily build.
    rng = np.random.default_rng(1)
    syms = list(weekly["close_w"].columns)
    det_syms = list(rng.choice(syms, size=min(24, len(syms)), replace=False))
    cfg1 = Config(**{**cfg.as_dict(), "nprocs": 1})
    cfg2 = Config(**{**cfg.as_dict(), "nprocs": 4})
    daily1 = build_daily_panels(det_syms, cfg1, cfg1.nprocs)
    daily2 = build_daily_panels(det_syms, cfg2, cfg2.nprocs)
    det_err = max(float((daily1[k] - daily2[k]).abs().max().max())
                  for k in daily1 if k in daily2)
    report["determinism_max_abs_diff"] = det_err
    assert det_err <= 1e-12, f"NONDETERMINISM: {det_err}"

    # 3. dollar neutrality
    net = w.sum(axis=1)
    report["dollar_neutral_max_net"] = float(np.abs(net).max())
    assert report["dollar_neutral_max_net"] < 1e-12

    # 4. gross bound
    gross = w.abs().sum(axis=1)
    report["gross_max"] = float(gross.max())
    assert report["gross_max"] <= 2.0 + 1e-12

    # 5. funding separation: perturb forward funding only; the return series
    #    must change, and it must change in week t-1 (not week t). This catches
    #    an off-by-one in the shift direction. Use a deep week where the book
    #    has non-zero gross (early PIT weeks are too thin).
    w_f = weekly["funding_w"].copy()
    perturbed = w_f.copy()
    # pick a week where the book is active, its previous week also active, and
    # at least one weighted symbol has observable forward funding; then perturb
    # exactly the weighted symbols so the change must move the return series.
    active_mask = w.abs().sum(axis=1) > 0
    idx_active = list(w.index[active_mask])
    pt = None
    for candidate in idx_active[len(idx_active) // 2:]:
        rp = w.index.get_loc(candidate)
        if rp < 1 or not active_mask.iloc[rp - 1]:
            continue
        wgt = w.iloc[rp - 1]
        wsym = wgt[wgt != 0].index
        fwd_funding_here = w_f.loc[candidate, wsym]
        if fwd_funding_here.notna().any():
            pt = candidate
            rowpos = rp
            break
    assert pt is not None, "no perturbable funding week found"
    wgt = w.iloc[rowpos - 1]
    wsyms = wgt[wgt != 0].index.tolist()
    # perturb ONE weighted symbol: a uniform shift across the whole dollar-
    # neutral book would net to zero and test nothing.
    perturbed.loc[pt, wsyms[0]] += 0.01
    fwd = weekly["fwd_ret_w"]
    adv = weekly["adv_w"]
    r_base = build_returns(w, fwd, build_funding_shift(w_f), adv, cfg)[0]
    r_pert = build_returns(w, fwd, build_funding_shift(perturbed), adv, cfg)[0]
    assert not np.allclose(r_base.fillna(0), r_pert.fillna(0)), \
        "funding perturbation had no effect: build_returns ignores funding"
    diff_weeks = r_base.index[(r_base - r_pert).abs().fillna(0) > 1e-15]
    assert list(diff_weeks) == [r_base.index[rowpos - 1]], \
        f"funding shifted to wrong week: {diff_weeks}"
    report["funding_separated"] = True

    # 6. non-emptiness: evaluable vs active weeks. w is a dense zero frame, so
    #    w.notna().all(axis=1) counts panel length. Report weeks where at least
    #    one factor produced a non-NaN score instead.
    s_final = res["s_final"]
    score_stack = pd.concat([s[~s.index.duplicated()] for s in s_final.values()], axis=1)
    evaluable = score_stack.notna().any(axis=1)
    report["evaluable_weeks"] = int(evaluable.sum())
    report["active_weeks"] = int(w.abs().sum(axis=1).gt(0).sum())
    assert report["active_weeks"] >= 1

    # 7. (optional) end-to-end leak test on 5m truncation
    if run_e2e:
        fails = leak_test_end_to_end(cfg, anchor)
        report["e2e_fail_weeks"] = fails
        if e2e_expected_fail:
            print(f"E2E leak test found {len(fails)} failing cut weeks: {fails} "
                  f"(expected with require_continuous_trading=True)")
        else:
            assert not fails, f"E2E LEAK at cut weeks {fails}"

    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="MON")
    ap.add_argument("--e2e", action="store_true",
                    help="run the end-to-end 5m-truncation leak test")
    a = ap.parse_args()
    cfg = Config(anchor=a.anchor)
    rep = run_checks(a.anchor, cfg, run_e2e=a.e2e)
    for k, v in rep.items():
        print(f"{k} = {v}")
    print("ALL CHECKS PASSED")