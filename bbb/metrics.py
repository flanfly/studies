"""metrics.py — Sharpe, ann. return/vol, max DD, recovery, skew/kurtosis, turnover."""
from __future__ import annotations

import numpy as np
from scipy import stats

WEEKS_PER_YEAR = 52.0


def annualized_sharpe(r):
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(WEEKS_PER_YEAR))


def annualized_return(r):
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    cum = (1 + r).prod()
    return float((cum ** (WEEKS_PER_YEAR / len(r))) - 1)


def annualized_vol(r):
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    return float(np.std(r, ddof=1) * np.sqrt(WEEKS_PER_YEAR))


def t_stat(r):
    r = r.dropna()
    n = len(r)
    if n < 2:
        return np.nan
    return float(np.mean(r) / (np.std(r, ddof=1) / np.sqrt(n)))


def equity_curve(r):
    return (1 + r.dropna()).cumprod()


def max_drawdown(r):
    eq = equity_curve(r)
    if len(eq) == 0:
        return np.nan
    peak = eq.cummax()
    dd = eq / peak - 1
    return float(dd.min())


def weeks_to_recover(r):
    eq = equity_curve(r)
    if len(eq) == 0:
        return np.nan
    peak = eq.cummax()
    dd = eq / peak - 1
    trough = dd.idxmin()
    idx = eq.index.get_loc(trough)
    for j in range(idx, len(eq)):
        if eq.iloc[j] >= peak.iloc[idx]:
            return j - idx
    return len(eq) - 1 - idx


def skew_kurt(r):
    r = r.dropna()
    if len(r) < 3:
        return np.nan, np.nan
    return float(stats.skew(r, bias=False)), float(stats.kurtosis(r, bias=False))


def positive_week_pct(r):
    r = r.dropna()
    return float((r > 0).mean()) if len(r) else np.nan


def worst_week(r):
    r = r.dropna()
    return float(r.min()) if len(r) else np.nan


def summary(r, turnover=None):
    s = {}
    s["sharpe"] = annualized_sharpe(r)
    s["ann_return"] = annualized_return(r)
    s["ann_vol"] = annualized_vol(r)
    s["t_stat"] = t_stat(r)
    s["max_dd"] = max_drawdown(r)
    s["recover_weeks"] = weeks_to_recover(r)
    s["worst_week"] = worst_week(r)
    sk, ku = skew_kurt(r)
    s["skew"] = sk
    s["excess_kurt"] = ku
    s["pos_week_pct"] = positive_week_pct(r)
    s["n_weeks"] = int(r.dropna().shape[0])
    if turnover is not None:
        s["mean_turnover"] = float(np.mean(turnover.dropna()))
    return s


def newey_west_t(r, lags=4):
    r = r.dropna().to_numpy()
    n = len(r)
    x = r - r.mean()
    var = np.mean(x * x)
    for l in range(1, lags + 1):
        cov = np.mean(x[l:] * x[:-l])
        var += 2 * (1 - l / (lags + 1)) * cov
    se = np.sqrt(var / n)
    return float(r.mean() / se)
