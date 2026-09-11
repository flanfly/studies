# Backtest Report — Full-Panel Run ("all listed coins", not top-60)

Same framework as `REPORT.md` (A2C ranker + logistic-regression volatility filter +
risk-based allocation), but the cross-section is **the full panel of eligible KuCoin
USDT-margined perpetuals** instead of the top-60 by trailing 3-month dollar volume.
Implementation: `backtest.py --panel full` (network input dimension inferred per fold);
analysis in `analyze-full.py`, raw per-fold results in `fp-t15-<seed>.parquet`.

## 1. Setup differences vs the top-60 run

- **Universe**: every perp with ≥85% trailing 3-month bar coverage and a price at fold
  start (still strictly causal). Eligible panel per fold: 117–304 symbols (mean ≈ 187,
  median 181); 870 distinct perps trade at some point in the dataset (129–378 active per
  calendar month).
- The A2C network dimension now varies per fold (state = n_panel × H = up to ~9,000
  inputs); the agent is retrained from scratch per fold, so this is seamless. The
  long/short leg size stays K=5.
- Everything else identical to `REPORT.md`: 4-month train / 1-month test walk-forward, 39
  folds (2023-06 → 2026-08), 2,376 12-hour periods, one chronological training pass,
  filter τ = 0.015, Ledoit-Wolf covariance, T_hist = 60, seed as validation-selected
  hyperparameter (13 seeds).
- **Validation seed selection picked seed 7** (validation Sharpe 2.38, 2023-06…2023-11).
  Notably, on the full panel the selected seed does *not* have the best full-period
  Sharpe (see stability table) — the validation window was too short to predict
  out-of-sample ranking quality.

## 2. Headline results (seed 7, filter τ = 0.015)

### Overall performance (gross)

| Strategy | Cum. Return | Ann. Vol | MDD | Sharpe | Sortino | Turnover/step |
|---|---|---|---|---|---|---|
| DRL rank (EW, unfiltered) | +65.6% | 40.7% | −56.5% | 0.57 | 1.01 | 0.025 |
| + filter: EW | +80.0% | 39.6% | −52.7% | 0.64 | 1.11 | 0.022 |
| + filter: Inverse Vol | **+100.3%** | 32.0% | −46.5% | **0.82** | 1.43 | 0.036 |
| + filter: MinVar | +14.3% | 31.4% | −53.4% | 0.29 | 0.40 | 0.063 |
| + filter: MaxDiver | −23.6% | 47.8% | −76.8% | 0.06 | 0.08 | 0.051 |
| + filter: Risk Parity | +62.0% | 35.3% | −54.1% | 0.59 | 0.95 | 0.032 |
| + filter: MinCVaR | −43.9% | 43.8% | −72.0% | −0.19 | −0.22 | 0.083 |
| JT momentum (benchmark) | +185.7% | 83.1% | −69.5% | 0.80 | 1.23 | 0.410 |
| Random (benchmark) | −19.3% | 34.1% | −53.3% | −0.02 | −0.03 | 1.923 |

### Transaction-cost impact

| | Gross | 5 bp/side | 10 bp/side |
|---|---|---|---|
| Best allocation (IV here) Sharpe | 0.82 | 0.78 | 0.74 |
| IV Cum. Return | +100.3% | +91.9% | +83.9% |
| MaxDiver Sharpe | 0.06 | 0.02 | −0.02 |

### Yearly breakdown

Final strategy per the paper (DRL + filter + **MaxDiver**), net 5 bp:

| Year | Cum. Return | Ann. Vol | MDD | Sharpe |
|---|---|---|---|---|
| 2023 (Jun–Dec) | +32.4% | 36.0% | −17.4% | 1.50 |
| 2024 | +36.3% | 66.1% | −57.9% | 0.79 |
| 2025 | −39.0% | 38.3% | −48.8% | −1.10 |
| 2026 (Jan–Aug) | −34.7% | 35.0% | −37.4% | −1.65 |

Best allocation on the full panel (**Inverse Vol**, net 5 bp):

| Year | Cum. Return | Ann. Vol | MDD | Sharpe |
|---|---|---|---|---|
| 2023 (Jun–Dec) | +17.0% | 24.6% | −15.3% | 1.21 |
| 2024 | +126.4% | 43.1% | −19.7% | 2.09 |
| 2025 | −21.3% | 28.0% | −33.2% | −0.72 |
| 2026 (Jan–Aug) | −7.9% | 22.3% | −22.9% | −0.44 |

Unfiltered DRL rank by year (gross): 2023 +29.8% (Sharpe 1.60), 2024 +111.8% (1.61),
2025 −24.1% (−0.70), 2026 −20.6% (−0.81).

### Seed stability (DRL rank, full period, gross)

Full-panel DRL Sharpe across the 13 seeds: mean **+0.21**, range −1.12 to +0.73. The wider
cross-section lifts the average seed performance versus the top-60 run (mean +0.06) —
more assets means more cross-sectional dispersion, which helps a 5/5 long-short book —
but the spread across seeds remains enormous, and validation-fold Sharpe is essentially
uninformative about full-period Sharpe (rank correlation ≈ 0 between the two columns).

## 4. Comparison: full panel vs top-60 (both seed-validation protocol, τ = 0.015)

| | Top-60 (REPORT.md) | Full panel |
|---|---|---|
| DRL rank Sharpe (selected seed, gross) | 0.69 | 0.57 |
| DRL rank Sharpe (mean over 13 seeds) | +0.06 | +0.21 |
| Best integrated Sharpe (gross) | 1.18 (EW) | 0.82 (IV) |
| Paper's headline allocation MaxDiver | 0.89 (3rd) | 0.06 (5th of 6) |
| Worst allocation | MinCVaR 0.53 | MinCVaR −0.19 |
| Ann. vol of DRL rank | 31.6% | 40.7% |
| Filter activity | 90%+ open | ~92.5% open (178/2376 steps closed) |
| JT momentum | +259% / 0.98 | +186% / 0.80 |

**Takeaways:**

1. **The full panel raises average seed-level DRL performance but lowers the selected
   seed's realized performance.** Wider dispersion helps the long-short structure in
   expectation, but it also inflates volatility (40.7% vs 31.6%) and tail events.
2. **The paper's headline combination (MaxDiver) breaks on the full panel.** MaxDiver
   concentrates weights on the lowest-volatility assets within each leg; in a panel that
   includes thin, illiquid small caps, those are exactly the assets with stale prices and
   adverse microstructure, and it turns a +66% gross DRL stream into −24%. The paper's
   finding that MaxDiver adds value on a curated top-60 universe does not survive
   universe expansion; Inverse Vol (a robust heuristic that also caps concentration in
   degenerate covariance estimates) takes its place.
3. **The filter adds much less on the full panel** (0.57 → 0.64 EW) than on top-60
   (0.69 → 1.18): the full-panel strategy's own return features are noisier, so the
   logistic meta-learner rarely finds a stable "high-vol" signature — it closed only 7.5%
   of steps.
4. **2024 is where the full panel earns** (DRL rank +112% gross); 2025–2026 are negative
   across essentially all variants, so the full-panel strategy has no robust edge in the
   later, lower-dispersion regime.
5. All conclusions from `REPORT.md` about the DRL agent's learning behavior carry over:
   at the paper's hyperparameters the policy is essentially the validation-selected
   initialization; the filter and allocation modules do the heavy lifting.

## 5. Deviations (in addition to those listed in REPORT.md §4)

- **Universe**: "full panel" = every symbol passing the causal eligibility filter
  (≥85% trailing 3-month coverage, price at fold start), re-selected monthly. No
  liquidity cut beyond coverage, so micro-cap perps are included — deliberately, since
  that is what "full panel of coins" means. No market-cap or listing-age filter is
  applied (none is available in the KuCoin dataset).
- Assets whose data ends mid-fold (delisted or truncated by the ~1,000-bar per-symbol
  history cap) keep their last price and earn zero returns until they exit the universe
  at the next fold re-selection; this affects only symbols listed before ~2025 (the
  KuCoin historical-data S3 bucket truncates older symbols).
- Everything else (hyperparameters, costs, metric conventions, seed protocol) is identical
  to `REPORT.md` §4; τ = 0.015 is used for the same reason as there (the paper's absolute
  τ = 0.009 would leave the gate closed for whole years at full-panel volatility levels).

## 6. Reproduction

```bash
for s in 0 1 2 3 4 5 6 7 8 9 10 11 42; do
  uv run backtest.py --seed $s --tau 0.015 --panel full --out fp-t15-$s.parquet
done
uv run analyze-full.py > fp-analysis.txt
```

Runtime ≈ 70 s per full walk-forward run (39 folds) on CPU; the seed sweep takes ~15 min.
Raw streams: `fp-t15-<seed>.parquet`; tables: `analyze-full.py` → `fp-analysis.txt`.

## 7. Bottom line

On the full panel of KuCoin perps (117–304 symbols per fold), the paper's framework
produces a real but modest long-short DRL signal (mean seed Sharpe +0.21, best 0.73),
the best integrated configuration (filter + Inverse Vol) reaches a Sharpe of 0.82 gross /
0.78 net of 5 bp, and the paper's headline MaxDiver allocation actively destroys value
off the curated top-60 universe (−0.02 net Sharpe, −77% MDD). The paper's Sharpe 2.85 is
not approachable on either universe; as in the top-60 run, the meta-learning filter and
allocation choice dominate the outcome, and seed selection remains the largest unpriced
risk in the paper's methodology.