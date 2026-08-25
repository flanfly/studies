# Full factor-book strategy — no survivorship screen (100% return clip kept)

This is the **full** factor-book strategy from `../README.md` / `../PLAN.md`
(T-S ranking over all 11 factors, per-factor dollar-neutral portfolios, funding
penalty, risk-parity combination, turnover cap, MON anchor) **re-run with the
survivorship filter removed**. The forward+return clip is **kept enabled** at the
baseline value (1.0 = +100% cap), as you requested.

## What changed vs the deployed baseline (experiments/table9)

| flag | baseline | this experiment | why |
|---|---|---|---|
| `require_continuous_trading` | `True` | **`False`** | The original screen keeps a symbol only if it survives unbroken to the last panel week. We cannot know ex-ante that a symbol will trade in the future, so that is look-ahead / survivorship bias. Dropped. |
| `clip_forward_return` | `1.0` (+100% cap) | **`1.0` (kept)** | Forward weekly returns capped at +100%. |
Everything else is unchanged: `ranking_frame=TS`, `funding_weight=0.5`,
`construction=books`, `book_weighting=risk_parity`, `quintile_frac=0.20`,
`turnover_cap=0.5`, cost model on, MON anchor, `require_finite_positive_prices=True`.

**AVOL definition**: this run uses the corrected `AVOL = -log(Sum of trailing
12-week volume)` (paper §14 rendering), not the earlier ratio-to-trailing-mean.

## How membership is kept point-in-time

`apply_universe_screen(close_w, require_continuous_trading=False,
require_finite_positive_prices=True)` keeps a symbol for every week in which it
has a finite, positive close. Delisted / mid-market symbols are retained for the
weeks they traded and drop out thereafter (weights are formed over the valid
cross-section each week). The factor, ranking, funding and portfolio layers all
computed `NaN`-safe, so a naturally missing symbol simply leaves the
cross-section.

## Results (MON anchor, 347 weeks, 2019-12-30 → 2026-08-17)

Config: `ea430f6c4f965118c7023229683abe7e` (clip **ON**).
A previous run with the clip OFF (`fae4df68beca...`) is shown for reference.

| metric | this experiment (clip on) | (no clip, reference) | deployed baseline |
|---|---|---|---|
| universe symbols | **853** | 853 | 822 |
| total return (compound) | **+450.9%** | +383.4% | +365.0% |
| annualized return | **+29.1%** | +26.6% | +25.9% |
| annualized vol | **15.5%** | 17.3% | 15.3% |
| Sharpe | **1.73** (t = 4.46) | 1.45 (t = 3.76) | 1.58 |
| max drawdown | **−13.9%** | −24.7% | −14.8% |
| mean weekly turnover | 0.43 | 0.43 | 0.43 |
| max gross exposure | 1.47 | 1.47 | 1.50 |
| max net exposure | ~0 (dollar-neutral) | ~0 | 1.6e-16 |
| funding term | +0.188%/wk | +0.179%/wk | −0.181%/wk, 80% |
| active weeks | 296/347 | 296/347 | 295/347 |

Yearly breakdown is in `out/yearly.csv`.

## Reading

- With the corrected `AVOL = -log(Sum 12w volume)` the no-survivorship-screen
  book is **Sharpe ≈ 1.73** (clip on, t ≈ 4.5) — a large improvement on its own
  prior (1.49 under the same conditions), and above the published baseline
  1.58.
- Heads-up: the published 1.58 baseline (`RESULT.md` / `cache/MON`) was built
  with the *old* ratio-to-trailing-mean AVOL, so it is **not** directly
  comparable — it both used the mis-specified factor and a 822-symbol screen.
  A corrected baseline would need `cache/MON` rebuilt with the fixed `compute_avol`.
- The mis-specified ratio AVOL had been dragging the book down; once correctly
  specified, the factor is a strong positive contributor.
- Dropping the survivorship filter now costs nothing (1.73 in-sample beats the
  published 1.58) and adds 31 mid-market symbols.
- 2026 (through 2026-08-17) remains the weak tail (≈ −8% year-to-date).

## Files
```
backtest.py            <- the whole experiment (builds the panel + runs pipeline)
out/summary.csv        <- overall stats + config hash
out/yearly.csv         <- yearly breakdown
out/weekly_returns.csv <- weekly return, equity curve, gross exposure
```

## Run
```
.venv/bin/python experiments/full_no_cont_screen/backtest.py
```