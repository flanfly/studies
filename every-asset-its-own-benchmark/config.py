"""Config dataclass — every experiment is a sweep over one or more of these fields."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # data / resampling
    anchor: str = "MON"
    book_terminal_return: bool = True  # book each symbol's final observed return
    # universe
    require_continuous_trading: bool = False  # survivorship screen is opt-in
    require_finite_positive_prices: bool = True
    # factors
    smooth_window_weeks: int = 20
    avol_lookback_weeks: int = 12   # AVOL = -log(vol[t] / mean(vol[t-12..t-1]))
    q_top_frac: float = 0.20
    min_days_per_week: int = 3
    tskd_min_bars_per_side: int = 20
    tskd_diff_order: str = "avg_then_diff"  # "avg_then_diff" | "diff_then_avg"
    cpvv_window: str = "week"               # "week" (within-week std) | "trailing_20d"
    cpvv_min_days_week: int = 4
    # ranking
    ranking_frame: str = "TS"            # "XS" | "TS" | "XS_standardised"
    rank_window_weeks: int = 52
    rank_min_periods: int = 26
    # funding
    funding_weight: float = 0.50
    # portfolio
    construction: str = "books"          # "books" | "blend"
    book_weighting: str = "risk_parity"  # "risk_parity" | "equal"
    quintile_frac: float = 0.20
    vol_window_weeks: int = 26
    vol_min_periods: int = 13
    min_cross_section: int = 10
    turnover_cap: float = 0.50
    # returns
    clip_forward_return: float | None = 1.0
    fee_bp_liquid: float = 1.0
    fee_bp_illiquid: float = 5.0
    cost_multiple: float = 1.0
    # factor subset (None = all eleven); for single-factor experiments
    factors: tuple | None = None
    # compute
    nprocs: int = 1

    # ---- factor-cache subset (changes require a daily-cache rebuild) ----
    def factor_cache_key(self):
        """Hash of the config fields that change daily-factor values."""
        import hashlib, json
        d = {k: getattr(self, k) for k in
             ("q_top_frac", "tskd_min_bars_per_side", "tskd_diff_order")}
        return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def factor_list(self):
        if self.factors is not None:
            return list(self.factors)
        return ["AVOL", "Q", "RSJ", "OFI", "CPVm", "CPVv",
                "WRspread", "TopChg", "Quad", "TKU", "TSKD"]

    def as_dict(self):
        d = asdict(self)
        return d

    def hash(self):
        import hashlib, json
        return hashlib.md5(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()
