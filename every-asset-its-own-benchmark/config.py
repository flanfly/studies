"""Config dataclass — every experiment is a sweep over one or more of these fields."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # data / resampling
    anchor: str = "MON"
    # universe
    require_continuous_trading: bool = True
    require_finite_positive_prices: bool = True
    # factors
    smooth_window_weeks: int = 20
    avol_lookback_weeks: int = 12   # AVOL = -log(Sum of trailing N-week volume)
    q_top_frac: float = 0.20
    min_days_per_week: int = 3
    tskd_min_bars_per_side: int = 20
    tskd_diff_order: str = "avg_then_diff"
    cpvv_window: str = "week"
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
