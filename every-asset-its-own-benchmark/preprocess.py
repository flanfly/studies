"""preprocess.py — build and cache weekly raw factor panels + weekly returns/funding.

For each anchor it writes:
    cache/<anchor>/{close_w,fwd_ret_w,adv_w,ret_w,volume_w,funding_w}.parquet
    cache/<anchor>/factor_<FACTOR>.parquet
    cache/<anchor>/n_symbols_by_week.parquet
    cache/<anchor>/manifest.json

The heavy intraday->daily factor computation is done once (anchor-independent)
by build_daily.build_daily_panels and cached under cache/daily/. Weekly
(re)aggregation per anchor reuses those cached daily panels.
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd

from data_load import get_5m, get_funding, apply_universe_screen
from resample import build_weekly, build_weekly_funding
from factors import build_weekly_raw_factors_from_daily, FACTOR_NAMES
from build_daily import build_daily_panels, load_daily_panels
from config import Config

CACHE_ROOT = "/home/kai/studies/bbb/cache"


def _screen_symbols(anchor, cfg):
    p5 = get_5m()
    weekly = build_weekly(p5, anchor)
    symbols = apply_universe_screen(
        weekly["close_w"], cfg.require_continuous_trading,
        cfg.require_finite_positive_prices)
    return symbols, weekly


def ensure_daily_panels(cfg, symbols):
    daily = load_daily_panels()
    if daily is None or len(daily) < 7:
        daily = build_daily_panels(symbols, cfg, cfg.nprocs)
    return daily


def build_anchor(anchor, cfg):
    out_dir = os.path.join(CACHE_ROOT, anchor)
    os.makedirs(out_dir, exist_ok=True)

    p5 = get_5m()
    weekly = build_weekly(p5, anchor)
    symbols = apply_universe_screen(
        weekly["close_w"], cfg.require_continuous_trading,
        cfg.require_finite_positive_prices)

    for k, df in weekly.items():
        weekly[k] = df.reindex(columns=symbols)

    funding = get_funding()
    funding_w = build_weekly_funding(funding, anchor)
    funding_w = funding_w.reindex(index=weekly["close_w"].index, columns=symbols)
    weekly["funding_w"] = funding_w

    for k in ("close_w", "fwd_ret_w", "adv_w", "ret_w", "volume_w", "funding_w"):
        weekly[k].to_parquet(os.path.join(out_dir, f"{k}.parquet"))

    n_sym = weekly["close_w"].notna().sum(axis=1)
    n_sym.to_frame("n").to_parquet(os.path.join(out_dir, "n_symbols_by_week.parquet"))

    manifest = {
        "anchor": anchor,
        "n_symbols": len(symbols),
        "n_weeks": len(weekly["close_w"]),
        "weeks_min": str(weekly["close_w"].index.min()),
        "weeks_max": str(weekly["close_w"].index.max()),
        "config": cfg.as_dict(),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"preprocess {anchor}: {len(symbols)} symbols, {len(weekly['close_w'])} weeks")
    return weekly, symbols


def build_factors_anchor(anchor, symbols, weekly, daily, cfg):
    out_dir = os.path.join(CACHE_ROOT, anchor)
    factor = build_weekly_raw_factors_from_daily(daily, symbols, weekly, cfg)
    for k in FACTOR_NAMES:
        factor[k].to_parquet(os.path.join(out_dir, f"factor_{k}.parquet"))
    return factor


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="MON")
    ap.add_argument("--nprocs", type=int, default=12)
    ap.add_argument("--factors-only", action="store_true",
                    help="skip weekly returns/funding rebuild, reuse cache")
    a = ap.parse_args()
    cfg = Config(anchor=a.anchor, nprocs=a.nprocs)
    if not a.factors_only:
        weekly, symbols = build_anchor(a.anchor, cfg)
    else:
        from pipeline import load_panels
        weekly, _, symbols = load_panels(a.anchor)
    daily = load_daily_panels()
    # Rebuild from per-symbol files (idempotent; reuses existing symbol dailies),
    # so the combined panels always cover the current symbol set.
    daily = build_daily_panels(symbols, cfg, cfg.nprocs)
    build_factors_anchor(a.anchor, symbols, weekly, daily, cfg)
    print(f"done factors for {a.anchor}")


def get_panel():
    return get_5m()


if __name__ == "__main__":
    main()
