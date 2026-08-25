# Implementation Spec — Cross-Sectional vs Self-Referential Factor Book on USDT Perpetual Futures

Target: a Python backtest reproducing the strategy in *"Every Asset Its Own Benchmark: Market-Neutral Alpha in Perpetual Futures"* (Dhanya MD, Aug 2026), from 5-minute kline parquet files.

**Scope note for the implementer:** reproduce the paper's design as written. Where the paper is ambiguous, this spec picks a default and marks it `[AMBIGUOUS]`. Do not silently substitute your own judgement — every choice must be a named config parameter so alternatives can be swept later. Do not add features, filters or "improvements" not listed here.

---

## 0. Inputs

### 0.1 Required: 5-minute klines (parquet, one file per symbol or one partitioned dataset)

Expected columns (Binance USDT-M futures kline schema):

| column | type | meaning |
|---|---|---|
| `open_time` | int64 ms or datetime64[ns, UTC] | bar open |
| `open`, `high`, `low`, `close` | float64 | prices |
| `volume` | float64 | base-asset volume |
| `quote_volume` | float64 | quote-asset (USDT) volume |
| `trades` | int64 | number of trades in bar |
| `taker_buy_base` | float64 | base volume where taker was the buyer |
| `taker_buy_quote` | float64 | quote volume where taker was the buyer |

Add a `symbol` column. Index/sort by `(symbol, open_time)`, all timestamps UTC.

### 0.2 Required: funding rates

Per symbol, funding rate observations (Binance pays every 8h). Columns: `symbol`, `funding_time` (UTC), `funding_rate` (decimal, e.g. `0.0001` = 1bp).

### 0.3 Required for 4 of 11 factors: positioning / open interest

Daily frequency, per symbol:

| column | used by |
|---|---|
| `open_interest` (or `open_interest_value`) | Quad |
| `top_trader_ls_ratio` (top-trader long/short account or position ratio) | WRspread, TopChg |
| `all_account_ls_ratio` (global long/short account ratio) | WRspread |

If these are unavailable, build the pipeline anyway with those factors emitting all-NaN. The combination step must handle a permanently-flat book (see §6.2). **Do not** drop them from the factor list — the paper fixes the eleven-factor specification a priori and Table 8 depends on all eleven being present.

---

## 1. Time grid

Three resolutions, built by aggregation, in this order:

1. **5-minute** — as loaded. Used only by `TKU` and `TSKD`.
2. **1-hour** — aggregate 12 consecutive 5-min bars. Used by `Q`, `RSJ`, `OFI`, `CPVm`, `CPVv`.
   - `open` = first, `high` = max, `low` = min, `close` = last
   - `volume`, `quote_volume`, `trades`, `taker_buy_base`, `taker_buy_quote` = sum
   - Drop hours with fewer than 12 constituent 5-min bars? **No** — keep, but record `n_bars`. Only exclude if `n_bars == 0`.
3. **Daily (UTC calendar day)** — every intraday factor reduces to exactly one record per `(symbol, date)`.
4. **Weekly** — daily records aggregated to one record per `(symbol, week)`.

### 1.1 Week definition

A week is 7 calendar days ending at the **rebalance anchor**. Default anchor = **Monday** (`anchor = "MON"`), meaning week `t` spans Monday 00:00 UTC through Sunday 23:59:59 UTC, and the rebalance decision is made at the close of that Sunday using only data up to that instant.

`anchor` must be a config parameter accepting `{MON, TUE, WED, THU, FRI, SAT, SUN}`. The whole panel is rebuilt from 5-min bars for each anchor — this is required later for the rebalance-timing-luck test (Table 10). Do not implement anchor variation by shifting an already-built Monday panel.

### 1.2 Weekly price and volume

Per `(symbol, week)`:
- `close_w` = last 5-min close in the week
- `volume_w` = sum of `quote_volume` over the week (USDT)
- `ret_w` = `close_w / close_w.shift(1) - 1` (simple return, used for `Quad`'s `sign(r_w)`)
- `fwd_ret_w` = `close_w.shift(-1) / close_w - 1` — **the forward return, the thing being predicted**
- `adv_w` = mean daily quote volume over the week (used by the cost model)

---

## 2. Universe

Screen, applied once over the whole panel (this reproduces the paper; keep it, and expose the flags):

```python
UNIVERSE = dict(
    quote_asset            = "USDT",
    require_finite_positive_prices = True,   # drop symbols with any non-finite or <=0 close
    require_continuous_trading     = True,   # see below
    min_weeks               = None,          # alternative to continuous screen
)
```

`require_continuous_trading = True` means: a symbol is retained only if it has an unbroken run of weekly observations from its first observation to the last week of the panel. Symbols that stop trading (delisted) mid-panel are excluded entirely.

Target shape: **112 symbols, 363 weeks**.

> This screen is a survivorship filter and is the paper's largest unexamined design choice. Implement it as written for the baseline, but make it a flag — a later run with `require_continuous_trading = False` plus point-in-time membership is the first thing we'll want.

Also emit a diagnostic the paper never reports: `n_symbols_by_week`, the count of symbols with a valid weekly observation in each week. Save it.

---

## 3. Factor definitions

Notation, all within one UTC day `d` for one symbol, over that day's intraday bars indexed `n`:

- `v_n` = bar volume, `q_n` = bar quote volume, `nn_n` = bar trade count
- `b_n` = taker-buy volume, **in the same units as `v_n`**
- `c_n` = bar close, `r_n = log(c_n / c_{n-1})` (first bar of day uses previous day's last close)

Each factor produces one value per `(symbol, day)` unless stated otherwise. Days with insufficient bars emit `NaN` — never a fallback value.

| # | name | resolution | daily value |
|---|---|---|---|
| 1 | AVOL | weekly | see below |
| 2 | Q | 1h | `VWAP(top 20% of bars by \|r_n\|/sqrt(v_n)) / VWAP(all bars)` |
| 3 | RSJ | 1h | `-(Σ_{r_n>0} r_n² − Σ_{r_n<0} r_n²) / Σ r_n²` |
| 4 | OFI | 1h | `(2·Σb_n − Σv_n) / Σv_n` |
| 5 | CPVm | 1h | `-corr(c_n, v_n)` (Pearson, within the day) |
| 6 | CPVv | 1h | `corr(c_n, v_n)` — daily value; the *factor* is the negated trailing s.d. (see below) |
| 7 | WRspread | daily | `-(top_trader_ls − all_account_ls)` |
| 8 | TopChg | daily | `-(top_trader_ls_d − top_trader_ls_{d-1})` |
| 9 | Quad | weekly | `-sign(ret_w) · (log(OI_w) − log(OI_{w-1}))` |
| 10 | TKU | 5m | `kurtosis(log(v_n / nn_n))` over the day's bars |
| 11 | TSKD | 5m | see §3.2 |

### 3.1 Per-factor detail

**AVOL** — weekly, not daily:
```
AVOL_t = -log( Sum_{w=t-11..t} volume_w )
```
Requires 12 weeks. The paper's table renders `− log Σ12w volume` (the negative log
of the trailing 12-week volume total); that is what is implemented. The earlier
ratio-to-trailing-mean construction was dropped as a mis-reading. Config:
`avol_lookback_weeks = 12`.

**Q (smart money)** — within each day, score every hourly bar `S_n = |r_n| / sqrt(v_n)`. Take the top 20% of bars by `S_n`. Compute VWAP over that subset (`Σ c_n·v_n / Σ v_n`) and VWAP over all bars. Factor = ratio. Requires ≥ 10 bars in the day. Config: `q_top_frac = 0.20`.
> Sign: the paper inverts this relative to its A-share source (reversal → momentum-consistent). Table 2 carries **no** leading minus, so use the ratio as-is: higher = long. Record this as a known researcher degree of freedom.

**RSJ** — `Σr²` is total realised variance over the day. Note the leading minus in the definition: upside-jump-dominated names get a *negative* score.

**OFI** — `b_n` and `v_n` must be the same units. Use `taker_buy_base` with `volume`, or `taker_buy_quote` with `quote_volume`. Pick one and be consistent. Result is bounded in `[-1, +1]`.

**CPVm / CPVv** — both derive from the same daily series `ρ_d = corr(c_n, v_n)` computed within day `d` across that day's hourly bars (requires ≥ 10 bars).
- `CPVm` daily value = `-ρ_d`; the weekly value is the mean of daily `-ρ_d` over the week.
- `CPVv` is a *dispersion* measure. `[AMBIGUOUS]` Implement as: weekly value = `-std(ρ_d over the days in the week)`, requiring ≥ 4 valid days. Config: `cpvv_window = "week"` with an alternative `"trailing_20d"`.

**WRspread / TopChg** — take the daily positioning values, then the weekly value is the mean over the week's days. `TopChg` differences at daily frequency *before* the weekly mean.

**Quad** — weekly by construction. `OI_w` = last open-interest observation in the week. `sign(ret_w)` uses the contemporaneous weekly return (week `t`, not forward). Both terms are point-in-time as of the week close.

**TKU** — `log(v_n / nn_n)` is log ticket size per 5-min bar. Drop bars where `nn_n == 0` or `v_n == 0`. Fisher (excess) kurtosis. Requires ≥ 20 valid bars in the day.

### 3.2 TSKD — directional trade-size asymmetry

For each day, partition the day's 5-minute bars by initiating side:

```
B = { n : b_n >  v_n / 2 }      # buy-initiated bars
S = { n : b_n <= v_n / 2 }      # sell-initiated bars
```

Compute log ticket size `x_n = log(v_n / nn_n)` for each bar, then:

```
A_d = skew({x_n : n in B}) - skew({x_n : n in S})
```

Require **≥ 20 bars on each side** (this is the reason the factor needs 5-minute bars; at hourly resolution a day yields ~12 bars per side and computability is 0%). Emit `NaN` otherwise.

Weekly value:
```
A_w    = mean(A_d over the days in week w)     # requires >= 3 valid days
TSKD_w = A_w - A_{w-1}
```
`[AMBIGUOUS]` The paper writes `Δ[...]` without specifying whether the difference is taken daily-then-averaged or averaged-then-differenced. The above (average, then difference) is implemented. Config: `tskd_diff_order = "avg_then_diff" | "diff_then_avg"`.

Fisher-Pearson skewness (`scipy.stats.skew(x, bias=False)`).

### 3.3 Daily → weekly aggregation

For every factor whose daily value is defined (2–8, 10, 11): the weekly raw value is the **mean of the daily values over the days in that week**, requiring at least `min_days_per_week = 3` valid days. AVOL and Quad are natively weekly.

### 3.4 Smoothing

After weekly aggregation and **before** any ranking:

```
f_smoothed[i, t] = rolling_mean(f_raw[i, :t+1], window=20, min_periods=20)
```

20 **weekly** periods. Applied to all eleven factors. Config: `smooth_window_weeks = 20`.

---

## 4. Ranking primitives

Both map to `[-1, +1]` via:

```python
def to_score(rank, n):        # rank is 1-based ascending
    return ((rank - 0.5) / n - 0.5) * 2.0
```

This is centred, antisymmetric, and must preserve `NaN`.

### 4.1 Cross-sectional score (`XS`)

```
s_XS[i, t] = to_score( rank of f[i,t] among {f[j,t] : j valid at t}, n = count valid at t )
```
Ranks ascending, ties averaged. Assets with `NaN` factor get `NaN` score and are excluded from `n`.

### 4.2 Self-referential score (`TS`)

```
window   = {f[i, t-w], ..., f[i, t-1]}          # STRICTLY prior, excludes t
valid    = non-NaN entries of window
rank     = 1 + count(valid < f[i,t])            # position of today's value in its own history
n        = len(valid) + 1
s_TS[i,t]= to_score(rank, n)   if len(valid) >= min_periods else NaN
```

Config: `rank_window_weeks = 52`, `rank_min_periods = 26`.

**This is the paper's central variable.** `ranking_frame ∈ {"XS", "TS", "XS_standardised"}` must be a top-level config switch, because Table 3 and Table 4 are produced by running the identical pipeline under each.

### 4.3 Intermediate frame (`XS_standardised`, for Table 4 only)

Per-asset z-score against its own trailing window, then rank cross-sectionally:
```
z[i,t]   = (f[i,t] - mean(window)) / std(window)      # same 52w/26w window as 4.2
s[i,t]   = to_score( rank of z[i,t] among {z[j,t]}, n )
```

### 4.4 Implementation warning

The paper documents a bug in which a ranking helper was applied along the wrong array axis, ranking each observation against the asset's entire history including the future, altering scores for 99 of 112 contracts. Write the ranking functions to take an explicit `axis`/`by` argument, assert the output shape, and cover them with the leak test in §9.

---

## 5. Funding penalty

Funding is the one quantity that *is* commensurate across contracts, so it is ranked cross-sectionally under every frame.

Weekly funding rate per symbol:
```
funding_w[i, t] = sum of funding_rate observations with funding_time in week t
```
(~21 observations per week at 8h intervals.)

Penalise every factor score:
```
s_final[i, t, k] = s[i, t, k] - funding_weight * s_XS(funding_w)[i, t]
```

Positive funding = longs pay, so a high funding rank pushes a contract toward the short leg. Verify the sign empirically: the resulting book must be a **net funding receiver in ~98% of weeks**, and the realised funding term should average about **−0.173% per week (−8.99% annualised)** in the return equation. If your sign is flipped you will see the mirror image and must fix it.

`[AMBIGUOUS]` The paper never states `funding_weight`. Config: `funding_weight = 0.5`, sweep `{0.0, 0.25, 0.5, 1.0}`.

---

## 6. Portfolio formation

### 6.1 One book per factor

For each of the eleven factors independently, at each week `t`:

```
valid   = symbols with non-NaN s_final and non-NaN fwd_ret
long    = top    q of valid by s_final
short   = bottom q of valid by s_final
w_long  = +1 / len(long)
w_short = -1 / len(short)
```

Dollar-neutral: long weights sum to `+1`, short to `-1`, gross `= 2` per book. Config: `quintile_frac = 0.20`, sweep `{0.10, 0.20, 0.30}`.

Require `len(valid) >= min_cross_section = 10`; otherwise the book is flat that week.

**Scores are never blended across factors.** The alternative (`construction = "blend"`, for Table 6) averages the eleven `s_final` scores into a composite and forms **one** book from the induced ordering. Both must be implementable via a single config switch: `construction ∈ {"books", "blend"}`.

### 6.2 Combining book returns

```
sigma_k[t]   = std(book_return_k over weeks t-V .. t-1)     # LAGGED, no week t data
raw_w_k[t]   = 1 / sigma_k[t]
weight_k[t]  = raw_w_k[t] / sum_j raw_w_j[t]                # over books with estimable sigma
```

Books with non-estimable `sigma` (insufficient history, or all-NaN factor) are **held flat** — weight 0, and the remaining weights renormalise to sum to 1. They are *not* given an equal weight.

`[AMBIGUOUS]` Vol window unspecified. Config: `vol_window_weeks = 26`, `vol_min_periods = 13`.

Alternative for the Table 6 sweep: `book_weighting ∈ {"risk_parity", "equal"}`.

The combined target weight vector is `w*[t] = Σ_k weight_k[t] · w_k[t]`. Its gross exposure will be **below** 2.0 because books partially offset (paper measures 1.462).

### 6.3 Turnover cap (uniform partial adjustment)

```
raw_turnover = sum(|w*[t] - w[t-1]|)
alpha        = min(1.0, turnover_cap / raw_turnover)     if raw_turnover > 0 else 1.0
w[t]         = w[t-1] + alpha * (w*[t] - w[t-1])
```

The adjustment fraction `alpha` is **uniform across all positions** — do not prioritise large signal changes or scale by estimated cost. (The paper tested both and rejected them: they destroy the property that makes the cap work.)

`[AMBIGUOUS]` Cap value unspecified. Realised weekly turnover in the deployed book is `0.369` of gross, so the cap binds. Config: `turnover_cap = 0.50`, sweep `{0.35, 0.50, 1.0}` where `1.0` ≈ uncapped.

Note: the cap is *not* Sharpe-maximising at the modelled fee — uncapped scores 2.364 vs deployed 2.161 at 1× costs. It is deliberately suboptimal insurance. Keep it in the deployed config.

---

## 7. Return construction

```
r[t] =   w[t] · fwd_ret[t]              # price
       - w[t] · funding[t+1]            # carry paid (negative in aggregate = receipt)
       - |w[t] - w[t-1]| · c            # transaction cost
```

Where:

- `fwd_ret[t]` is the simple return from close of week `t` to close of week `t+1`, **clipped at +100%**:
  ```python
  fwd = fwd.clip(upper=clip_forward_return)   # clip_forward_return = 1.0
  ```
  Config: `clip_forward_return = 1.0`, with `None` to disable. Keep `1.0` for the baseline — it is the paper's spec.
- `funding[t+1]` = sum of funding rates over the *forward* week (the week the position is held).
- `c[i]` = liquidity-scaled maker fee, computed cross-sectionally each week from average daily dollar volume:
  ```python
  p    = pct_rank(adv[:, t])          # in [0,1], 1 = most liquid
  c[i] = 1e-4 * (5.0 - 4.0 * p[i])    # 5bp least liquid -> 1bp most liquid
  ```
  Config: `fee_bp_liquid = 1.0`, `fee_bp_illiquid = 5.0`, `cost_multiple = 1.0` (the multiplier swept in Table 14: `{1,2,4,8,16,32}`).

Cost is charged on `|Δw|` per asset, so entering and exiting each pay once.

---

## 8. Warm-up and evaluability

Longest chain: 12w (AVOL) or 20w (smoothing) → 52w (ranking window) → the first evaluable week is around week 55.

Expected shape, to check against:

| quantity | expected |
|---|---|
| panel weeks | 363 |
| evaluable rebalances | 308 |
| active rebalances (non-flat book) | 282 |
| weeks in the return series | 281 |

A rebalance is **evaluable** if the pipeline can produce scores; **active** if at least one book has a non-empty long and short leg.

---

## 9. Point-in-time discipline — mandatory tests

These are not optional; the paper's own audit found a leak here that both research and production code shared.

1. **Leak test.** For a sample of `(symbol, week)` pairs spread across the panel, recompute `s_final[i,t]` from a panel with all rows after week `t` deleted. Assert the result equals the full-panel value to `0.0` absolute difference. Run for all eleven factors.
2. **Determinism.** Two runs of the same config produce bit-identical weights.
3. **Dollar neutrality.** `max(|sum(w[t])|) < 1e-12` at every rebalance.
4. **Gross bound.** `max(sum(|w[t]|)) <= 2.0`.
5. **No forward funding.** Assert the funding series used in the score (§5) is week `t` and the funding series in the return (§7) is week `t+1`, and that they are never the same array.
6. **Non-emptiness.** Count of built vs active rebalances matches §8.

Fail the run (non-zero exit) on any violation.

---

## 10. Config object

```python
@dataclass
class Config:
    # data
    anchor: str = "MON"
    # universe
    require_continuous_trading: bool = True
    # factors
    smooth_window_weeks: int = 20
    avol_lookback_weeks: int = 12
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
```

The deployed configuration is the above with defaults. Every experiment in §12 is a sweep over one or more of these fields.

---

## 11. Module layout

```
data_load.py      parquet -> 5m panel; universe screen; funding; positioning
resample.py       5m -> 1h -> daily -> weekly, anchor-aware
factors.py        eleven factor functions, each (daily_or_weekly_panel) -> weekly raw panel
ranking.py        to_score, rank_xs, rank_ts, rank_xs_standardised
portfolio.py      score -> per-factor books -> combined target -> turnover cap -> weights
returns.py        weights + fwd + funding + costs -> weekly return series
metrics.py        Sharpe, ann. return/vol, max DD, recovery weeks, skew, kurtosis, turnover
checks.py         the six tests in §9
experiments/      one script per table in §12
registry.py       append-only log: every evaluated config + its Sharpe (see §12.5)
```

Panels are `(week × symbol)` float64 DataFrames throughout, with a shared symbol index and week index. Never reindex silently — assert alignment at every join.

---

## 12. Experiments to reproduce

Run in this order; each is a separate script writing a CSV plus the source table number.

### 12.1 Table 3 — ranking frame
Eleven single-factor books (each factor alone, `construction="books"` with one book), under `ranking_frame ∈ {XS, TS}`, `funding_weight ∈ {0.5, 0.0}` (as-traded vs carry-removed). Report annualised Sharpe per factor and the mean.
Expected: XS mean 1.011 → TS mean 1.451, 9/11 improve. Carry removed: 0.383 → 0.648, 8/11.

### 12.2 Table 4 — mechanism
Same, adding `ranking_frame="XS_standardised"`. Expected means: 1.011 / 1.192 / 1.451, i.e. 41% recovery.

### 12.3 Table 5 — carry contribution
Per-factor Sharpe with `funding_weight = 0.5` vs `0.0`, matched samples. Also the funding component traded standalone (a book formed on `s_XS(funding)` alone) — expected Sharpe 1.250, t = 2.91.

### 12.4 Table 6 — construction
Sweep the six variants `{XS, TS} × {books, blend} × {equal, risk_parity}` over an identical grid of the remaining hyperparameters. Report best and mean per variant.
> Panel A of the paper compares `TS/books` against `XS/blend/equal`, which varies three things at once. Report the properly matched contrast as well: `books` vs `blend` at fixed frame and weighting.

### 12.5 Table 13 — deflated Sharpe
`registry.py` must append one row per evaluated configuration — written by the evaluation code itself, not by hand — with the config hash, the config fields, and the resulting Sharpe. The DSR trial count `N` and dispersion `sd(SR)` are then read from that log, never supplied by a human.

### 12.6 Table 9 — equity curve properties
Cumulative and annualised return, annualised vol, Sharpe, max drawdown, weeks to recover, worst week, worst month, % positive weeks, skewness, excess kurtosis. Plus the Nov 2021 – Dec 2022 sub-window (61 weeks) against an equal-weight market return of the tradeable universe, and the correlation over that window.

### 12.7 Table 10 — walk-forward × rebalance anchor
Select config on a trailing 104 weeks, evaluate on the next 26, roll forward. Every hyperparameter is re-selected each training block, including `quintile_frac` — pin nothing to a full-sample winner. 36 candidates per fold (suggested axes: `quintile_frac` × 3, `funding_weight` × 3, `turnover_cap` × 2, `rank_window_weeks` × 2). Repeat across all seven anchors, rebuilding the panel from 5-min bars each time.

### 12.8 Table 12 — held-out universe
Select on 78 symbols, evaluate on the other 34, eight random splits. Rebuild the cross-section within each side so ranks, quintile boundaries and the cost model use only symbols present on that side. Include the matched-breadth control: the same selected config evaluated on a random 34 of the *training* symbols.

### 12.9 Table 14 — cost robustness
Sweep `cost_multiple ∈ {1,2,4,8,16,32}` for `turnover_cap = 1.0` (uncapped) and `0.50` (deployed).

---

## 13. Metric definitions

- Annualised Sharpe = `mean(r_w) / std(r_w) * sqrt(52)`, zero risk-free rate, `std` with `ddof=1`.
- Annualised return = `(1 + cumulative)^(52/T) - 1` on compounded weekly returns.
- t-statistic = `mean(r_w) / (std(r_w) / sqrt(T))`.
- Newey–West t with 4 lags for the alpha regression (§12 optional extension).
- Max drawdown on the compounded equity curve.
- Skewness and excess kurtosis on weekly returns, both bias-corrected.
- Turnover = `sum(|w[t] - w[t-1]|)` per week, reported as a fraction of gross exposure.

---

## 14. Known ambiguities, collected

Every `[AMBIGUOUS]` item above, for the record. Each is a config field with a stated default; none should be resolved by guessing at runtime.

1. `AVOL`: paper table renders `−log Σ12w volume`; implemented as the negative log
   of the trailing 12-week volume total (weeks t-11..t). No ratio-to-trailing-mean.
2. `CPVv` dispersion window: within-week s.d. of daily correlations.
3. `TSKD` difference ordering: average daily asymmetry over the week, then difference.
4. `funding_weight` magnitude: never stated in the paper.
5. `turnover_cap` value: never stated; inferred from realised turnover of 0.369.
6. `vol_window_weeks` for inverse-vol book weighting: never stated.
7. Daily → weekly aggregation of intraday factors: mean of daily values.
8. Whether the 20-period smoothing is applied to weekly or daily values: weekly.
9. Exact composition of the 36-candidate walk-forward grid: never enumerated.

Log the resolved value of every one of these in the run manifest alongside the results.
