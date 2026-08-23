"""factors.py — eleven factor functions: intraday data -> daily -> weekly raw panel.

Each intraday factor is computed per (symbol, day), then aggregated to a weekly
raw panel (week x symbol) by mean of daily values (min_days_per_week). AVOL and
Quad are natively weekly. Positioning factors (WRspread, TopChg, Quad's OI term)
emit all-NaN because positioning data is unavailable in this dataset.

Panels are (week x symbol) float64 DataFrames with a shared symbol index and
week index (the anchor date).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from resample import week_label

FACTOR_NAMES = [
    "AVOL", "Q", "RSJ", "OFI", "CPVm", "CPVv",
    "WRspread", "TopChg", "Quad", "TKU", "TSKD",
]


# --------------------------------------------------------------------------
# daily intraday computation (heavy) - parallel over symbols
# --------------------------------------------------------------------------

def _compute_symbol_daily(sym_df, cfg):
    """Compute daily factor values for one symbol.

    Returns dict factor -> (dates ndarray, values ndarray).
    """
    c = sym_df["close"].to_numpy()
    v = sym_df["volume"].to_numpy()
    nn = sym_df["trades"].to_numpy(dtype=float)
    b = sym_df["taker_buy_base"].to_numpy()
    day = sym_df["open_time"].dt.normalize().dt.tz_localize(None).to_numpy()

    # 5-minute log returns
    r5 = np.full(len(c), np.nan)
    r5[1:] = np.log(c[1:] / np.maximum(c[:-1], 1e-12))
    r5 = np.where(np.isfinite(r5), r5, np.nan)
    del r5  # not used directly for factors defined here; 5m returns unused
    # (TKU/TSKD use volume/trades; 5m return not required)

    # --- hourly aggregation (within symbol) ---
    hdf = sym_df.copy()
    hdf["h"] = hdf["open_time"].dt.floor("h")
    hdf["day"] = day
    h = hdf.groupby("h", observed=True).agg(
        c=("close", "last"), v=("volume", "sum"),
        nn=("trades", "sum"), b=("taker_buy_base", "sum"),
        day=("day", "first")).reset_index()
    hc = h["c"].to_numpy()
    hv = h["v"].to_numpy()
    hb = h["b"].to_numpy()
    hr = np.full(len(hc), np.nan)
    hr[1:] = np.log(hc[1:] / np.maximum(hc[:-1], 1e-12))
    hr = np.where(np.isfinite(hr), hr, np.nan)
    hS = np.abs(hr) / np.sqrt(hv)
    hday = h["day"].to_numpy()

    # day axis from 5m
    ud, dstart = np.unique(day, return_index=True)
    day_axis = ud
    nidx = len(day_axis)

    # hourly day-start index map
    ud_h, hstart = np.unique(hday, return_index=True)
    hidx = {d: i for i, d in enumerate(ud_h)}

    res = {k: [] for k in ("Q", "RSJ", "OFI", "CPVm", "CPVrho", "TKU", "TSKD")}

    for i in range(nidx):
        d = day_axis[i]
        s = dstart[i]
        e = dstart[i + 1] if i + 1 < nidx else len(c)

        # hourly slice for this day
        hi = hidx.get(d, -1)
        if hi >= 0:
            hs = hstart[hi]
            he = hstart[hi + 1] if hi + 1 < len(ud_h) else len(hc)
        else:
            hs = he = -1

        if hs >= 0 and he > hs:
            nb = he - hs
            cday = hc[hs:he]; vday = hv[hs:he]; rday = hr[hs:he]; bday = hb[hs:he]
            vi = np.isfinite(rday)
            # Q
            if nb >= 10:
                Sv = np.abs(rday) / np.sqrt(np.maximum(vday, 1e-12))
                m = np.isfinite(Sv)
                if m.sum() >= 1:
                    ksel = max(1, int(np.ceil(cfg.q_top_frac * m.sum())))
                    sc = cday[m]; sv_ = vday[m]; SS = Sv[m]
                    top = np.argsort(SS)[-ksel:]
                    vwap_top = (sc[top] * sv_[top]).sum() / sv_[top].sum()
                    vwap_all = (sc * sv_).sum() / sv_.sum()
                    res["Q"].append(vwap_top / vwap_all if vwap_all else np.nan)
                else:
                    res["Q"].append(np.nan)
            else:
                res["Q"].append(np.nan)
            # RSJ
            r2 = rday * rday
            tot = r2.sum()
            if tot > 0:
                pos = r2[rday > 0].sum()
                neg = r2[rday < 0].sum()
                res["RSJ"].append(-(pos - neg) / tot)
            else:
                res["RSJ"].append(np.nan)
            # OFI
            sv = vday.sum()
            res["OFI"].append((2 * bday.sum() - sv) / sv if sv > 0 else np.nan)
            # CPV
            if nb >= 10:
                cc = np.corrcoef(cday, vday)
                rho = cc[0, 1] if np.isfinite(cc[0, 1]) else np.nan
            else:
                rho = np.nan
            res["CPVm"].append(-rho if np.isfinite(rho) else np.nan)
            res["CPVrho"].append(rho)
        else:
            for k in ("Q", "RSJ", "OFI", "CPVm", "CPVrho"):
                res[k].append(np.nan)

        # 5m factors (TKU, TSKD)
        if e > s:
            v5 = v[s:e]; nn5 = nn[s:e]; b5 = b[s:e]
            m5 = (nn5 > 0) & (v5 > 0)
            x5 = np.log(v5[m5] / nn5[m5])
            if len(x5) >= 20:
                res["TKU"].append(float(stats.kurtosis(x5, fisher=True)))
            else:
                res["TKU"].append(np.nan)
            # TSKD
            vsafe = np.maximum(v5, 1e-12)
            Bm = (b5 > vsafe / 2.0) & (nn5 > 0) & (v5 > 0)
            Sm = (b5 <= vsafe / 2.0) & (nn5 > 0) & (v5 > 0)
            xB = np.log(v5[Bm] / nn5[Bm])
            xS = np.log(v5[Sm] / nn5[Sm])
            if len(xB) >= cfg.tskd_min_bars_per_side and len(xS) >= cfg.tskd_min_bars_per_side:
                res["TSKD"].append(float(stats.skew(xB, bias=False) - stats.skew(xS, bias=False)))
            else:
                res["TSKD"].append(np.nan)
        else:
            res["TKU"].append(np.nan)
            res["TSKD"].append(np.nan)

    out = {name: (day_axis, np.array(res[name])) for name in res}
    return out


def _process_chunk(args):
    panel5m, symbols, cfg = args
    out = {}
    for sym in symbols:
        sub = panel5m[panel5m["symbol"] == sym]
        if len(sub) == 0:
            continue
        sub = sub.sort_values("open_time")
        out[sym] = _compute_symbol_daily(sub, cfg)
    return out


def compute_daily_panels(panel5m, symbols, cfg):
    """Return dict factor -> daily DataFrame (date x symbol)."""
    import multiprocessing as mp

    nprocs = max(1, getattr(cfg, "nprocs", 1))
    syms = list(symbols)
    if nprocs <= 1 or len(syms) <= 1:
        chunks = [syms]
    else:
        k = min(nprocs, len(syms))
        splits = np.array_split(np.array(syms, dtype=object), k)
        chunks = [list(s) for s in splits]

    tasks = [(panel5m, s, cfg) for s in chunks]
    if len(chunks) > 1:
        with mp.get_context("fork").Pool(nprocs) as pool:
            results = pool.map(_compute_chunk_daily, tasks)
    else:
        results = [_compute_chunk_daily(tasks[0])]

    all_sym = {}
    for r in results:
        all_sym.update(r)

    per_factor = {f: {} for f in ("Q", "RSJ", "OFI", "CPVm", "CPVrho", "TKU", "TSKD")}
    for sym, dd in all_sym.items():
        for f in per_factor:
            if f in dd:
                dates, vals = dd[f]
                per_factor[f][sym] = pd.Series(vals, index=dates)

    daily = {}
    for f in per_factor:
        daily[f] = pd.DataFrame(per_factor[f])
    return daily


def _compute_chunk_daily(args):
    panel5m, symbols, cfg = args
    out = {}
    for sym in symbols:
        sub = panel5m[panel5m["symbol"] == sym]
        if len(sub) == 0:
            continue
        sub = sub.sort_values("open_time")
        out[sym] = _compute_symbol_daily(sub, cfg)
    return out


# --------------------------------------------------------------------------
# weekly aggregation
# --------------------------------------------------------------------------

def aggregate_daily_to_weekly(daily_panel, anchor, min_days_per_week=3):
    """Mean daily values over each week, requiring >= min_days valid days.

    daily_panel: date x symbol. Returns week x symbol.
    """
    idx = pd.DatetimeIndex(daily_panel.index)
    wk = week_label(idx, anchor)
    df = daily_panel.copy()
    df["__week"] = wk
    cnt = df.groupby("__week").apply(lambda g: g.notna().sum(axis=0))
    val = df.groupby("__week").mean()
    val = val.where(cnt >= min_days_per_week, np.nan)
    val = val.reindex(columns=daily_panel.columns).sort_index()
    val = val.astype(float)
    return val


def compute_avol(volume_w, lookback_weeks=12):
    """AVOL_t = -log( volume_w_t / mean(volume_w_{t-lookback..t-1}) )."""
    mean = volume_w.shift(1).rolling(lookback_weeks, min_periods=lookback_weeks).mean()
    return -np.log(volume_w / mean)


def compute_quad(ret_w, oi_w):
    """Quad_t = -sign(ret_w) * (log(OI_w) - log(OI_{w-1}))."""
    if oi_w is None or oi_w.empty:
        return pd.DataFrame(np.nan, index=ret_w.index, columns=ret_w.columns)
    dlog = np.log(oi_w) - np.log(oi_w.shift(1))
    return -np.sign(ret_w) * dlog


def build_weekly_raw_factors(panel5m, symbols, weekly, cfg):
    """Build all 11 weekly raw factor panels (week x symbol) from the 5m panel."""
    daily = compute_daily_panels(panel5m, symbols, cfg)
    return build_weekly_raw_factors_from_daily(daily, symbols, weekly, cfg)


def build_weekly_raw_factors_from_daily(daily, symbols, weekly, cfg):
    """Build all 11 weekly raw factor panels from cached daily panels.

    daily: dict of daily factor DataFrames (date x symbol), keys among
      Q, RSJ, OFI, CPVm, CPVrho, TKU, TSKD.
    weekly: dict from build_weekly.
    """
    panels = {}
    vol = weekly["volume_w"].reindex(columns=symbols)
    panels["AVOL"] = compute_avol(vol, cfg.avol_lookback_weeks)

    for name, col in [("Q", "Q"), ("RSJ", "RSJ"), ("OFI", "OFI"),
                      ("CPVm", "CPVm"), ("TKU", "TKU"), ("TSKD", "TSKD")]:
        d = daily[col].reindex(columns=symbols)
        panels[name] = aggregate_daily_to_weekly(d, cfg.anchor, cfg.min_days_per_week)

    # CPVv from CPVrho daily (dispersion): weekly = -std(rho over days in week), >=4 days
    cp = daily["CPVrho"].reindex(columns=symbols)
    idx = pd.DatetimeIndex(cp.index)
    wk = week_label(idx, cfg.anchor)
    df = cp.copy(); df["__week"] = wk
    cnt = df.groupby("__week").apply(lambda g: g.notna().sum(axis=0))
    std = df.groupby("__week").std(ddof=1)
    std = std.where(cnt >= 4, np.nan).sort_index()
    panels["CPVv"] = -std.astype(float)

    # positioning factors -> all NaN (data unavailable)
    ref_idx = panels["AVOL"].index
    cols = panels["AVOL"].columns
    panels["WRspread"] = pd.DataFrame(np.nan, index=ref_idx, columns=cols)
    panels["TopChg"] = pd.DataFrame(np.nan, index=ref_idx, columns=cols)

    # Quad (OI unavailable -> all NaN)
    ret = weekly["ret_w"].reindex(columns=symbols)
    oi = None
    panels["Quad"] = compute_quad(ret, oi).reindex(index=ref_idx, columns=cols)

    # standardise to the weekly index shared by AVOL
    for k in panels:
        panels[k] = panels[k].reindex(index=ref_idx, columns=cols)
    return panels
