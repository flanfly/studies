"""resample.py — 5m -> 1h -> daily -> weekly, anchor-aware.

The whole panel is rebuilt from 5-min bars for each anchor. Weeks are 7
calendar days ending at the rebalance anchor. The week label is the anchor
weekday date (e.g. Monday 00:00 UTC for anchor="MON"), i.e. week t covers
[monday(t), monday(t)+7d). The rebalance decision is made at the close of the
last day of week t using only data up to that instant.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

_ANCHOR = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}


def week_label(ts, anchor="MON"):
    """Map a UTC timestamp to its week's anchor date (the week start).

    week t = [anchor_date(t), anchor_date(t)+7d).
    """
    a = _ANCHOR[anchor]
    idx = pd.DatetimeIndex(ts)
    day = idx.normalize().tz_localize(None)  # tz-naive UTC
    wd = day.weekday.values  # Monday=0
    back = (wd - a) % 7
    return day - pd.to_timedelta(back, unit="D")


def build_weekly(panel5m, anchor="MON", book_terminal_return=True):
    """Aggregate the 5m panel to a weekly panel.

    Returns weekly DataFrame (week x symbol) of:
        close_w, volume_w (quote), adv_w (mean daily quote volume),
        ret_w, fwd_ret_w.

    book_terminal_return: give each symbol's LAST observed week a realised
    forward return from the week's close to the final observed 5m close
    (settlement mark) instead of NaN. Without this the book always exits the
    week before a delisting and never books the terminal move — for LUNA/SRM/
    BZRX/COCOS that move is the entire economic content of including them.
    If the final observed bar IS the last bar of the last week the return is
    0, not NaN.
    """
    t = panel5m["open_time"]
    wk = pd.Series(week_label(t, anchor).values, index=panel5m.index, name="week")
    day = pd.Series(pd.DatetimeIndex(t).normalize().tz_localize(None).values,
                    index=panel5m.index, name="day")

    key = [panel5m["symbol"], wk]
    g_close = panel5m["close"].groupby(key).last()
    g_vol = panel5m["quote_volume"].groupby(key).sum()
    g_day = day.groupby(key).nunique()

    close_w = g_close.unstack("week").T.sort_index()
    vol_w = g_vol.unstack("week").T.sort_index()
    n_days = g_day.unstack("week").T.sort_index()

    # convert all column indexes together, before any reindex
    close_w.columns = close_w.columns.astype(str)
    vol_w.columns = vol_w.columns.astype(str)
    n_days.columns = n_days.columns.astype(str)
    symbols = close_w.columns
    vol_w = vol_w.reindex(columns=symbols)
    n_days = n_days.reindex(columns=symbols)

    adv_w = vol_w / n_days.replace(0, np.nan)

    ret_w = close_w / close_w.shift(1) - 1.0
    fwd_ret_w = close_w.shift(-1) / close_w - 1.0

    if book_terminal_return:
        # per symbol: last week with a valid close -> return from that week's
        # close to the final observed 5m close (settlement mark).
        last_valid = close_w.apply(lambda s: s.last_valid_index())
        final_mark = panel5m.groupby("symbol")["close"].last()
        for sym, wk_last in last_valid.items():
            if wk_last is None or sym not in final_mark.index:
                continue
            c0 = close_w.at[wk_last, sym]
            if np.isfinite(c0) and c0 > 0:
                fwd_ret_w.at[wk_last, sym] = final_mark[sym] / c0 - 1.0

    out = {
        "close_w": close_w,
        "volume_w": vol_w,
        "adv_w": adv_w,
        "ret_w": ret_w,
        "fwd_ret_w": fwd_ret_w,
    }
    return out


def build_weekly_funding(funding, anchor="MON"):
    """Sum funding_rate per (symbol, week). Returns weeks x symbols."""
    wk = week_label(funding["funding_time"], anchor)
    df = funding.assign(week=wk)
    g = df.groupby(["symbol", "week"])["funding_rate"].sum()
    panel = g.unstack("week").T
    return panel
