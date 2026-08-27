"""Phase 2/3 tests — delisting returns, universe screen, funding, registry.

  2.1 terminal return: a symbol that stops trading gets a realised forward
      return on its last week (settlement mark), not NaN.
  2.2 universe default: require_continuous_trading defaults to False.
  2.2 universe lookahead: the screen with True keeps only survivors; the
      panel's own index is used (no generated 7D range).
  3.2 funding separation: perturbing forward funding changes the return series
      in the correct week (off-by-one catches).
  4.1 registry append-only: rows are appended under flock + fsync; a read after
      a log sees the new row.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config  # noqa: E402
from resample import build_weekly  # noqa: E402
from data_load import apply_universe_screen  # noqa: E402
from returns import build_returns, build_funding_shift  # noqa: E402
import registry  # noqa: E402


def test_terminal_return_booked():
    """A delists at week 1 (its last week). With booking ON its fwd_ret at the
    final week is FINITE (settled at the final mark, i.e. 0 if the week-close is
    the last bar); with booking OFF it is NaN and the book exits a week early.
    """
    base = pd.to_datetime("2021-01-04", utc=True)
    rows = []
    # A: full week 0, then partial week 1, then stops
    for w, nbars in ((0, 3), (1, 2)):
        for b in range(nbars):
            rows.append({"symbol": "A",
                         "open_time": base + pd.Timedelta(weeks=w) + pd.Timedelta(hours=b),
                         "close": 100.0 + w * 10.0 + b, "quote_volume": 1000.0})
    # B: trades through both weeks
    for w in range(2):
        for b in range(3):
            rows.append({"symbol": "B",
                         "open_time": base + pd.Timedelta(weeks=w) + pd.Timedelta(hours=b),
                         "close": 100.0 + w * 10.0 + b, "quote_volume": 1000.0})
    p5 = pd.DataFrame(rows).sort_values(["symbol", "open_time"])

    w_on = build_weekly(p5, "MON", book_terminal_return=True)
    w_off = build_weekly(p5, "MON", book_terminal_return=False)
    idx = w_on["close_w"].index
    # A's last valid week is week 1; B's is week 1 too but both book; check
    # that A's terminal fwd (week 1) is finite when booked, NaN when not,
    # and B's week-0 fwd is the real 110/102-1 return in both.
    assert np.isfinite(w_on["fwd_ret_w"].loc[idx[1], "A"])
    assert np.isnan(w_off["fwd_ret_w"].loc[idx[1], "A"])
    assert abs(w_on["fwd_ret_w"].loc[idx[0], "B"] - (112.0 / 102.0 - 1.0)) < 1e-9
    # A's week-0 fwd uses the real week-1 close: 111/102-1
    assert abs(w_on["fwd_ret_w"].loc[idx[0], "A"] - (111.0 / 102.0 - 1.0)) < 1e-9
    assert abs(w_on["fwd_ret_w"].loc[idx[1], "A"]) < 1e-9  # settled at final mark


def test_terminal_return_opt_out():
    base = pd.to_datetime("2021-01-04", utc=True)
    rows = [{"symbol": "A", "open_time": base + pd.Timedelta(hours=h),
             "close": 100 + h, "quote_volume": 1000.0} for h in range(3)]
    p5 = pd.DataFrame(rows)
    w1 = build_weekly(p5, "MON", book_terminal_return=True)
    w0 = build_weekly(p5, "MON", book_terminal_return=False)
    assert np.isfinite(w1["fwd_ret_w"].iloc[0, 0])
    assert np.isnan(w0["fwd_ret_w"].iloc[0, 0])


def test_universe_screen_default_is_false():
    cfg = Config()
    assert cfg.require_continuous_trading is False


def test_universe_screen_uses_panel_index():
    """A symbol that stops mid-panel is kept when False and dropped when True."""
    idx = pd.date_range("2021-01-04", periods=5, freq="7D")
    close = pd.DataFrame({
        "SURVIVOR": [1.0, 1.1, 1.2, 1.3, 1.4],
        "DELISTED": [1.0, 1.1, np.nan, np.nan, np.nan],
    }, index=idx)
    ok_false = apply_universe_screen(close, require_continuous_trading=False)
    assert "DELISTED" in ok_false
    ok_true = apply_universe_screen(close, require_continuous_trading=True)
    assert "SURVIVOR" in ok_true and "DELISTED" not in ok_true


def test_funding_separation_behavioural():
    cfg = Config(anchor="MON")
    idx = pd.date_range("2021-01-04", periods=12, freq="7D")
    syms = ["S1", "S2"]
    w = pd.DataFrame(np.ones((12, 2)) * 0.5, index=idx, columns=syms)
    fwd = pd.DataFrame(0.01, index=idx, columns=syms)
    adv = pd.DataFrame(1e7, index=idx, columns=syms)
    w_f = pd.DataFrame(0.001, index=idx, columns=syms)
    perturbed = w_f.copy()
    perturbed.iloc[10, :] += 0.01
    r_base = build_returns(w, fwd, build_funding_shift(w_f), adv, cfg)[0]
    r_pert = build_returns(w, fwd, build_funding_shift(perturbed), adv, cfg)[0]
    assert not np.allclose(r_base.fillna(0), r_pert.fillna(0))
    diff_weeks = r_base.index[(r_base - r_pert).abs().fillna(0) > 1e-15]
    assert list(diff_weeks) == [r_base.index[9]], \
        f"funding shifted to wrong week: {list(diff_weeks)}"


def test_registry_append(tmp_path=None):
    """registry.log appends a row readable immediately after."""
    import importlib
    import registry as reg
    # point registry at a temp file
    old = reg.REGISTRY
    import tempfile
    tmp = tempfile.mktemp(suffix=".csv")
    reg.REGISTRY = tmp
    cfg = Config(anchor="MON")
    try:
        reg.log(cfg, 0.123, n_weeks=10, anchor="MON", context="test")
        df = reg.load()
        assert len(df) == 1
        assert abs(df.iloc[0]["sharpe_weekly"] - 0.123) < 1e-12
        assert df.iloc[0]["n_weeks"] == 10
    finally:
        reg.REGISTRY = old
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    for fn in (test_terminal_return_booked, test_terminal_return_opt_out,
               test_universe_screen_default_is_false,
               test_universe_screen_uses_panel_index,
               test_funding_separation_behavioural, test_registry_append):
        fn()
        print(f"{fn.__name__}: PASSED")