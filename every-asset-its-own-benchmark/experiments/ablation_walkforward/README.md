# Walk-forward ablation — factor-subset selection rolls out-of-sample

Tests the factor-subset choice *inside* a walk-forward, the way a real
deployment would have to do it: on each **104-week** training block we select a
factor subset (leave-one-out ablated against the block's Sharpe), evaluate it on
the **next 26 weeks**, then roll 26 weeks. The question: is the chosen subset
**stable across folds** (real signal) or does it **churn** (we're fitting
noise → keep all eleven)?

## Setup (same data conditions as the previous ablation study)
- No survivorship screen (`require_continuous_trading=False`) → 853 symbols.
- No 100% return clip (`clip_forward_return=None`).
- Everything else = deployed baseline: TS ranking, books + risk-parity, funding
  weight 0.5, turnover cap 0.5, MON anchor, 347 weeks.
- Selection rule: run all **12 candidates** (all-11 + one-factor-out for each of
  the 11 factors) on the 104-week training block, pick the max **training-block
  Sharpe** (ties broken toward more factors / all-11).
- 10 folds: train [t, t+104), test [t+104, t+130), step 26 (last test truncated
  to the panel end).

## Per-fold selection

| fold | train block | test block | chosen | train Sharpe (chosen) | test Sharpe (chosen) | test Sharpe all-11 | test Sharpe (oracle) | margin best−2nd |
|---|---|---|---|---|---|---|---|---|
| 0 | 2019-12-30 → 2021-12-20 | 2021-12-27 → 2022-06-20 | no_TKU | 2.04 | 4.17 | 3.92 | 4.40 | 0.04 |
| 1 | 2020-06-29 → 2022-06-20 | 2022-06-27 → 2022-12-19 | no_TKU | 2.54 | 0.15 | −0.04 | 0.55 | 0.06 |
| 2 | 2020-12-28 → 2022-12-19 | 2022-12-26 → 2023-06-19 | no_TKU | 2.99 | 1.75 | 1.44 | 1.75 | 0.03 |
| 3 | 2021-06-28 → 2023-06-19 | 2023-06-26 → 2023-12-18 | no_TKU | 2.38 | 3.20 | 3.36 | 4.51 | 0.11 |
| 4 | 2021-12-27 → 2023-12-18 | 2023-12-25 → 2024-06-17 | no_RSJ | 2.43 | 0.56 | 0.38 | 0.85 | 0.04 |
| 5 | 2022-06-27 → 2024-06-17 | 2024-06-24 → 2024-12-16 | no_TKU | 1.55 | 3.29 | 3.41 | 3.82 | 0.03 |
| 6 | 2022-12-26 → 2024-12-16 | 2024-12-23 → 2025-06-16 | no_TKU | 2.32 | 1.62 | 1.77 | 2.37 | 0.01 |
| 7 | 2023-06-26 → 2025-06-16 | 2025-06-23 → 2025-12-15 | no_RSJ | 2.41 | 0.38 | 0.66 | 1.01 | 0.02 |
| 8 | 2023-12-25 → 2025-12-15 | 2025-12-22 → 2026-06-15 | no_OFI | 1.70 | −1.09 | −0.74 | 0.15 | 0.02 |
| 9 | 2024-06-24 → 2026-06-15 | 2026-06-22 → 2026-08-17 | no_RSJ | 1.40 | −2.91 | −2.80 | −2.41 | 0.06 |

## Stability of the selected subset
- **distinct winners: 3** (no_TKU, no_RSJ, no_OFI) out of 10 folds.
- **TKU removal wins 6/10 folds** (folds 0-3 and 5-6) — the first four folds
  are unanimous.
- The last four folds churn to RSJ/OFI with the *smallest* in-sample margins of
  the whole run (0.01–0.06 Sharpe vs 0.02–0.11 overall).
- Removal frequency: TKU 6, RSJ 3, OFI 1; everything else 0 — **including
  AVOL, which is *never* removed now that it is correctly specified**.
- Churn rate (consecutive fold changes): 5/9.

> **AVOL note**: before the §14 fix, the (mis-specified) ratio AVOL was the
> most-removed factor (7/10 folds). With `AVOL = -log(Sum 12w volume)` it is
> **never** removed — the selection consistently wants it in the book. The
> instability has shifted to TKU/RSJ, which the full-sample ablation already
> flagged as the weakest genuine factors.

## Does the selection pay off out-of-sample?

Concatenated held-out test blocks (243 weeks, same weeks for both counterfactuals):

| strategy | OOS Sharpe | OOS ann. return | OOS max DD |
|---|---|---|---|
| **walk-forward selected subset** | **1.170** | +15.9% | −25.8% |
| **all 11 fixed** | **1.163** | +15.4% | −24.7% |
| oracle (would need future) | 1.700 (mean fold) | — | — |

Fixed-commitment view (mean test Sharpe per candidate had you stuck with it all
10 folds): no_RSJ 1.26, no_TSKD 1.16, no_Q 1.14, no_CPVm 1.14, all_11 1.14,
no_OFI 1.13, no_TKU 1.12, no_CPVv 1.05, **no_AVOL 1.00 (worst)**. Notably no_TKU
— the subset the walk-forward would have picked most often (6/10) — is only
mid-pack out-of-sample, while dropping AVOL (never selected, correctly) is the
single worst fixed choice. The differences are ~0.01-0.1 Sharpe, i.e. small and
inconsistent with the training ranking: the in-sample selection is not
reproducing out of sample.

## Verdict

**Keep all eleven.** The selection is not stable *enough*:
1. AVOL is now rock-solid in the book (never removed across all 10 folds) — the
   §14 correction turned a fake "weakest factor" into a clearly valuable one.
2. The remaining churn is TKU (6/10) and RSJ (3/10), exactly the factors the
   full-sample ablation flags as noise. But their win margins are tiny (mean
   0.042 Sharpe best-vs-2nd), so the selection is riding noise.
3. The walk-forward that "would have been deployed" (1.170 OOS Sharpe) ties
   all-11 (1.163) — the ablation buys nothing out-of-sample, and it *churns*
   (5/9 folds). The oracle (1.70) still shows the over-fit tax is far bigger
   than any selection benefit.

The honest conclusion: in-sample margins are noise-sized, the folds churn, and
out-of-sample the ablated book matches — not beats — the full 11-factor book. There's a weak hint TKU (and RSJ) are dispensable, but not enough to justify
dropping them (or any other factor) on the strength of this walk-forward.

## Files
```
ablation.py     <- builds panels once, runs 12 candidates × 10 folds
out/folds.csv       <- per-fold selection + train/test Sharpe
out/candidates.csv  <- per fold × candidate train/test metrics
out/summary.csv     <- stability stats + OOS comparison
out/wf_test_returns.csv <- concatenated held-out returns (selected vs all-11)
```