"""Phase 1 tests — factor definition fixes.

Each test fails on the pre-fix code and passes after:
  1.1 AVOL: constant weekly volume => AVOL == 0 every week past warm-up.
  1.2 TSKD: weekly panel has NaN in the first valid week (nothing to diff
      against), and a constant daily asymmetry produces exactly 0.
  1.3 CPVv: cfg.cpvv_window is wired ("week" and "trailing_20d" both run).
  1.4 Log guard: aggregate_daily_to_weekly raises on inf input.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config  # noqa: E402
from factors import compute_avol, build_weekly_raw_factors_from_daily, aggregate_daily_to_weekly  # noqa: E402


def _cfg(**kw):
    d = dict(anchor="MON", require_continuous_trading=False,
             clip_forward_return=None, tskd_diff_order="avg_then_diff",
             cpvv_window="week")
    d.update(kw)
    return Config(**d)


def _full_daily(dates, tskd_vals=None, cprho=0.3):
    d = {}
    for col in ("Q", "RSJ", "OFI", "CPVm", "CPVrho", "TKU", "TSKD_level"):
        if col == "CPVrho":
            d[col] = pd.DataFrame(cprho, index=dates, columns=["S1"])
        elif col == "TSKD_level":
            v = tskd_vals if tskd_vals is not None else np.nan
            d[col] = pd.DataFrame(v, index=dates, columns=["S1"])
        else:
            d[col] = pd.DataFrame(np.nan, index=dates, columns=["S1"])
    return d


def test_avol_constant_volume_is_zero():
    """AVOL on a constant weekly-volume symbol must be exactly 0 after warm-up
    (current week equals its own trailing mean)."""
    vol = pd.DataFrame({"S1": [100.0] * 40},
                       index=pd.date_range("2020-01-06", periods=40, freq="7D"))
    a = compute_avol(vol, lookback_weeks=12)
    vals = a.iloc[12:, 0]
    assert np.isfinite(vals).all()
    assert np.allclose(vals.to_numpy(), 0.0, atol=1e-12)


def test_tskd_week1_is_nan_and_constant_is_zero():
    cfg = _cfg()
    # daily TSKD_level: constant asymmetry -> weekly level constant -> diff zero
    dates = pd.date_range("2021-01-04", periods=14, freq="D")
    daily = _full_daily(dates, tskd_vals=0.5)
    weekly = {
        "volume_w": pd.DataFrame(100.0, index=pd.date_range("2021-01-04", periods=2, freq="7D"), columns=["S1"]),
        "ret_w": pd.DataFrame(0.0, index=pd.date_range("2021-01-04", periods=2, freq="7D"), columns=["S1"]),
    }
    panels = build_weekly_raw_factors_from_daily(daily, ["S1"], weekly, cfg)
    tskd = panels["TSKD"]
    assert np.isnan(tskd.iloc[0, 0]), "first valid week must be NaN (no prior to diff against)"
    # second week: level constant => diff = 0
    assert np.allclose(tskd.iloc[1, 0], 0.0, atol=1e-12)


def test_cpvv_window_wired():
    cfg_week = _cfg(cpvv_window="week")
    cfg_20d = _cfg(cpvv_window="trailing_20d")
    dates = pd.date_range("2021-01-04", periods=30, freq="D")
    daily = _full_daily(dates, cprho=0.3)
    weekly = {
        "volume_w": pd.DataFrame(100.0, index=pd.date_range("2021-01-04", periods=5, freq="7D"), columns=["S1"]),
        "ret_w": pd.DataFrame(0.0, index=pd.date_range("2021-01-04", periods=5, freq="7D"), columns=["S1"]),
    }
    p1 = build_weekly_raw_factors_from_daily(daily, ["S1"], weekly, cfg_week)["CPVv"]
    p2 = build_weekly_raw_factors_from_daily(daily, ["S1"], weekly, cfg_20d)["CPVv"]
    assert not p1.empty and not p2.empty
    # constant rho => zero dispersion either way
    assert np.allclose(p1.dropna(), 0.0, atol=1e-9)
    assert np.allclose(p2.dropna(), 0.0, atol=1e-9)


def test_inf_guard_raises():
    dates = pd.date_range("2021-01-04", periods=5, freq="D")
    bad = pd.DataFrame({"S1": [1.0, 2.0, np.inf, 4.0, 5.0]}, index=dates)
    try:
        aggregate_daily_to_weekly(bad, "MON", min_days_per_week=3)
    except AssertionError:
        return
    raise AssertionError("aggregate_daily_to_weekly accepted inf input")


if __name__ == "__main__":
    for fn in (test_avol_constant_volume_is_zero, test_tskd_week1_is_nan_and_constant_is_zero,
               test_cpvv_window_wired, test_inf_guard_raises):
        fn()
        print(f"{fn.__name__}: PASSED")