"""experiments/build_all_anchors.py — build weekly panels + factors for all 7 anchors.

Reuses the cached anchor-independent daily factor panels; only the cheap weekly
(re)aggregation runs per anchor. Loads the 5m panel once.
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from data_load import get_5m, get_funding, apply_universe_screen
from resample import build_weekly, build_weekly_funding
from factors import build_weekly_raw_factors_from_daily, FACTOR_NAMES
from build_daily import load_daily_panels
from config import Config
from paths import CACHE_ROOT

ANCHORS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]


def build_all():
    cfg = Config(nprocs=12)
    p5 = get_5m()
    funding = get_funding()
    daily = load_daily_panels(cfg)
    for anchor in ANCHORS:
        out = os.path.join(CACHE_ROOT, anchor)
        os.makedirs(out, exist_ok=True)
        if os.path.exists(os.path.join(out, "factor_AVOL.parquet")):
            print(f"{anchor}: cached, skip", flush=True)
            continue
        weekly = build_weekly(p5, anchor, cfg.book_terminal_return)
        symbols = apply_universe_screen(weekly["close_w"],
                                        cfg.require_continuous_trading,
                                        cfg.require_finite_positive_prices)
        for k in weekly:
            weekly[k] = weekly[k].reindex(columns=symbols)
        funding_w = build_weekly_funding(funding, anchor)
        funding_w = funding_w.reindex(index=weekly["close_w"].index, columns=symbols)
        weekly["funding_w"] = funding_w
        for k in ("close_w", "fwd_ret_w", "adv_w", "ret_w", "volume_w", "funding_w"):
            weekly[k].to_parquet(os.path.join(out, f"{k}.parquet"))
        factor = build_weekly_raw_factors_from_daily(daily, symbols, weekly, cfg)
        for k in FACTOR_NAMES:
            factor[k].to_parquet(os.path.join(out, f"factor_{k}.parquet"))
        weekly["close_w"].notna().sum(axis=1).to_frame("n").to_parquet(
            os.path.join(out, "n_symbols_by_week.parquet"))
        json.dump({"anchor": anchor, "n_symbols": len(symbols),
                   "n_weeks": len(weekly["close_w"]), "config": cfg.as_dict()},
                  open(os.path.join(out, "manifest.json"), "w"), indent=2, default=str)
        print(f"{anchor}: {len(symbols)} symbols, {len(weekly['close_w'])} weeks", flush=True)


if __name__ == "__main__":
    build_all()
