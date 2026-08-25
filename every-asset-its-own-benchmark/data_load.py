"""data_load.py — parquet -> 5m panel; universe screen; funding.

Reads the Binance USDT-M futures 5-minute klines and the funding-rate
observations, maps Binance column names onto the spec schema, applies the
universe screen, and exposes cached in-memory panels.

5m kline Binance columns -> spec schema:
    open_time            -> open_time
    open,high,low,close  -> same
    volume               -> volume (base)
    quote_volume         -> quote_volume (USDT)
    count                -> trades (number of trades in bar)
    taker_buy_volume     -> taker_buy_base (base volume, taker as buyer)
    taker_buy_quote_volume -> taker_buy_quote
"""
from __future__ import annotations

import os
import pandas as pd
import numpy as np

FUTURES_PARQUET = "/home/kai/node/data/studies/bn-futures-5m.parquet"
FUNDING_PARQUET = "/home/kai/node/data/studies/bn-funding-rates.parquet"

_RENAME = {
    "open_time": "open_time",
    "close": "close",
    "volume": "volume",
    "quote_volume": "quote_volume",
    "count": "trades",
    "taker_buy_volume": "taker_buy_base",
}

# Only the columns actually used by the factors / resampling.
_NEEDED = [
    "open_time", "close", "volume", "quote_volume",
    "count", "taker_buy_volume", "symbol",
]


def _load_5m() -> pd.DataFrame:
    df = pd.read_parquet(FUTURES_PARQUET, columns=_NEEDED)
    df = df.rename(columns=_RENAME)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    return df


_5M = None


def get_5m() -> pd.DataFrame:
    """Load (once) and return the full 5m panel."""
    global _5M
    if _5M is None:
        _5M = _load_5m()
    return _5M


def get_funding() -> pd.DataFrame:
    """Load funding-rate observations.

    Columns: symbol, funding_time (UTC), funding_rate.
    """
    df = pd.read_parquet(FUNDING_PARQUET)
    df = df.rename(columns={"calc_time": "funding_time",
                            "last_funding_rate": "funding_rate"})
    df["funding_time"] = pd.to_datetime(df["funding_time"], utc=True)
    df = df[["symbol", "funding_time", "funding_rate"]]
    return df.sort_values(["symbol", "funding_time"]).reset_index(drop=True)


def apply_universe_screen(weekly_close, require_continuous_trading=True,
                          require_finite_positive_prices=True):
    """Return the list of retained symbols given the weekly close panel.

    weekly_close: weeks x symbols DataFrame of last-5m close per week.
    """
    if require_finite_positive_prices:
        ok = []
        for c in weekly_close.columns:
            s = weekly_close[c].dropna()
            if len(s) == 0:
                continue
            if not np.isfinite(s.values).all():
                continue
            if (s <= 0).any():
                continue
            ok.append(c)
    else:
        ok = list(weekly_close.columns)

    if require_continuous_trading:
        sub = weekly_close[ok]
        last_week = sub.index[-1]
        keep = []
        for sym in sub.columns:
            ser = sub[sym]
            valid = ser.dropna()
            if len(valid) == 0:
                continue
            if valid.index[-1] != last_week:
                continue  # stopped trading mid-panel
            # unbroken run from first valid week to panel last week
            weeks = valid.index
            expected = pd.date_range(weeks[0], last_week, freq="7D")
            if len(weeks) == len(expected) and (weeks == expected).all():
                keep.append(sym)
        return keep
    return list(ok)
