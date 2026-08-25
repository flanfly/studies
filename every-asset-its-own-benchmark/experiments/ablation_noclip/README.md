# Ablation study — leave-one-factor-out (no clip, no survivorship screen)

Runs the **full** factor-book strategy (`PLAN.md`) with one factor removed at a
time, to see which factors influence the result. Does **not** run the pure-carry
short leg.

## Setup
- `clip_forward_return = None` — 100% forward-return clip **disabled** (per
  request to "remove the clip again").
- `require_continuous_trading = False` — no survivorship screen; universe =
  **853** symbols.
- Everything else = deployed baseline: TS ranking, construction=books,
  risk-parity weighting, funding_weight=0.5, turnover cap 0.5, MON anchor,
  347 weeks.
- Baseline = all 11 factors; each ablation removes exactly one factor.

## Results

| experiment | n_factors | Sharpe | ann_return | ann_vol | max_dd | t_stat | cum_return |
|---|---|---|---|---|---|---|---|
| **baseline_all** | 11 | **1.455** | 26.6% | 17.3% | −24.7% | 3.76 | +383.3% |
| remove_AVOL | 10 | 1.360 | 24.2% | 17.0% | −28.2% | 3.51 | +323.6% |
| remove_Q | 10 | 1.419 | 26.9% | 17.9% | −23.8% | 3.67 | +389.9% |
| remove_RSJ | 10 | 1.512 | 28.2% | 17.4% | −22.5% | 3.91 | +424.5% |
| remove_OFI | 10 | 1.437 | 26.6% | 17.5% | −26.6% | 3.71 | +383.1% |
| remove_CPVm | 10 | 1.449 | 27.0% | 17.5% | −23.7% | 3.74 | +392.5% |
| remove_CPVv | 10 | 1.326 | 25.3% | 18.3% | −24.3% | 3.43 | +350.2% |
| remove_WRspread | 10 | 1.455 | 26.6% | 17.3% | −24.7% | 3.76 | +383.3% |
| remove_TopChg | 10 | 1.455 | 26.6% | 17.3% | −24.7% | 3.76 | +383.3% |
| remove_Quad | 10 | 1.455 | 26.6% | 17.3% | −24.7% | 3.76 | +383.3% |
| remove_TKU | 10 | 1.516 | 27.2% | 16.8% | −23.5% | 3.92 | +398.7% |
| remove_TSKD | 10 | 1.377 | 25.7% | 17.8% | −25.0% | 3.56 | +360.2% |

Full metrics (incl. turnover, worst week, recovery, weekly series) in
`out/ablation.csv`.

> **AVOL note**: this table uses the **corrected** `AVOL = -log(Sum 12w
> volume)` (paper §14 rendering). With the earlier ratio-to-trailing-mean
> implementation, AVOL looked like a *detrimental* factor (removing it raised
> Sharpe 1.22 → 1.36). Correctly specified, AVOL is a strong *positive*
> contributor (removing it drops Sharpe 1.455 → 1.360).

## Interpretation

**Factors that matter (removing them lowers Sharpe)** — i.e. they contribute
predictive signal:
- remove_AVOL → 1.360 (biggest drop from 1.455)
- remove_CPVv → 1.326
- remove_TSKD → 1.377
- remove_OFI → 1.437
- remove_CPVm → 1.449
- remove_Q → 1.419

**Factors that *hurt* the combined book (removing them *raises* Sharpe)**:
- remove_TKU → 1.516
- remove_RSJ → 1.512

**Factors with no effect** (identical metrics to baseline — these emit all-NaN
because positioning/open-interest data does not exist in this dataset):
- remove_WRspread, remove_TopChg, remove_Quad → all exactly replicate baseline
  Sharpe 1.4549.

## Reading
- The signal load is carried by the trade-based factors: **AVOL** (now correctly
  specified as −log Σ12w volume), **CPVv (return dispersion), TSKD, OFI, Q**
  and CPVm. Dropping any of them erodes the Sharpe.
- RSJ and TKU are slightly *detrimental* to the balanced risk-parity book —
  leaving either out improves Sharpe (1.455 → ~1.51). Their signal is weak
  enough that the combination over-weights their noise under risk parity.
- WRspread, TopChg, Quad contribute exactly nothing here (data unavailable),
  so they neither add nor remove value.

## Files
```
ablation.py     <- the experiment (builds panels once, runs 1 baseline + 11 ablations)
out/ablation.csv <- full metric table for all 13 runs
```