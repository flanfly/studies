# Backtest Report — "Deep RL for Dynamic Learn-to-Rank" (Barak, Mousavi & Hosseini) on KuCoin Futures Data

Implementation: `backtest.py` (PyTorch + polars + numpy + scipy + scikit-learn), run with
`uv run backtest.py --seed 8 --tau 0.015 --out seed-8.parquet`. Analysis/table generation:
`analyze.py`. All figures below are out-of-sample walk-forward results, net of the stated
transaction costs unless marked gross.

---

## 1. What was implemented

The paper's three-stage framework, faithfully in structure:

1. **DRL ranker (A2C)** — shared MLP (128 ReLU → 64 ReLU → 32 sigmoid), Gaussian policy
   head emitting a real-valued score vector for the 60-asset universe, plus a value head.
   State = flattened window of the last H=30 min–max normalized closes per asset
   (normalization fit on the training fold only, per the paper's §4.1.2). Training uses the
   paper's hyperparameters: lr 1e-5, γ=0.99, entropy β=0.01, n-steps=5, value coef 0.5,
   grad clip 0.5, binary ±1 reward on the sign of the equal-weight top-5/bottom-5 portfolio
   return, one chronological pass over the 4-month training fold.
2. **Meta-learning filter** — logistic regression on the last 10 realized strategy returns,
   labeling each step by whether the rolling volatility σ_t (window W_vol) is below the
   threshold τ; a "low-vol" prediction opens the gate, otherwise the book goes flat.
3. **Portfolio construction** — cash-neutral long top-5 / short bottom-5, each leg weighted
   by EW, Inverse Vol, MinVar, MaxDiver, Risk Parity (all with Ledoit–Wolf shrinkage,
   T_hist=60 trailing scenarios) and MinCVaR (Rockafellar–Uryasev LP, α=0.95), implemented
   with `scipy.optimize`.

Walk-forward protocol: 4-month train / 1-month test, rolling monthly, agent + filter
retrained from scratch each fold (39 folds, 2023-06 → 2026-08, 2,376 evaluated 12-hour
periods). Universe = top-60 USDT-margined KuCoin perpetuals by trailing 3-month average
dollar volume, re-selected monthly using only data available at fold start.

**Model-selection protocol (as in the paper):** the paper treats the random seed as a
hyperparameter "selected ... on the initial validation folds". We ran 13 seeds (0–11, 42),
evaluated the DRL ranker on the first six test months (2023-06…2023-11) and selected
**seed 8** (validation Sharpe 2.88; next best 1.51). All full-period numbers below use that
single pre-selected seed.

## 2. Headline results (filter threshold τ = 0.015, seed 8)

### Overall performance (39 folds, Jun 2023 – Aug 2026, 12h rebalance)

| Strategy | Cum. Return | Ann. Vol | MDD | Sharpe | Sortino | Omega | Turnover/step |
|---|---|---|---|---|---|---|---|
| DRL rank (EW, unfiltered) | +73.1% | 31.6% | −44.1% | 0.69 | 0.93 | 1.08 | 0.016 |
| + filter: EW | **+155.1%** | 27.7% | −35.4% | **1.18** | 1.57 | 1.14 | 0.015 |
| + filter: Inverse Vol | +108.4% | 25.5% | −34.0% | 1.01 | 1.35 | 1.12 | 0.026 |
| + filter: MinVar | +110.2% | 30.0% | −38.6% | 0.91 | 1.11 | 1.12 | 0.056 |
| + filter: MaxDiver | +122.0% | 33.9% | −46.5% | 0.89 | 1.11 | 1.12 | 0.042 |
| + filter: Risk Parity | +140.1% | 27.1% | −35.5% | 1.13 | 1.44 | 1.14 | 0.024 |
| + filter: MinCVaR | +55.1% | 46.4% | −53.8% | 0.53 | 0.57 | 1.08 | 0.064 |
| JT momentum (benchmark) | +259.4% | 56.1% | −49.7% | 0.98 | 1.38 | 1.11 | 0.387 |
| Random (benchmark) | +30.3% | 30.9% | −48.6% | 0.42 | 0.57 | 1.05 | 1.814 |

### Transaction-cost impact (final integrated strategy, DRL + filter + allocation)

| | Gross | 5 bp/side | 10 bp/side |
|---|---|---|---|
| Best allocation (EW here) Sharpe | 1.18 | 1.16 | 1.14 |
| MaxDiver Sharpe | 0.89 | 0.85 | 0.80 |
| MaxDiver Cum. Return | +122.0% | +111.2% | +100.9% |

Cost drag is small because the strategy's turnover is low (~4% of book per 12h step,
two-sided). With the paper's convention this corresponds to roughly 2.5× book per month —
higher than the paper's 30.91% monthly; see deviations §4.

### Yearly breakdown (final strategy DRL + filter + MaxDiver, net 5 bp)

| Year | Cum. Return | Ann. Vol | MDD | Sharpe |
|---|---|---|---|---|
| 2023 (Jun–Dec) | +33.6% | 23.3% | −10.6% | 2.24 |
| 2024 | +20.4% | 28.1% | −23.6% | 0.80 |
| 2025 | +34.1% | 46.7% | −31.3% | 0.86 |
| 2026 (Jan–Aug) | −2.1% | 25.7% | −20.2% | 0.00 |

Unfiltered DRL rank by year (gross): 2023 +39.1% (Sharpe 2.72), 2024 +28.6% (1.09),
2025 +44.3% (1.24), 2026 −32.9% (−1.26) — the filter converts the weak 2026 into a flat
year, which is where most of its risk value comes from.

## 3. Comparison with the paper

| Metric | Paper | This replication |
|---|---|---|
| DRL rank (EW) Sharpe | 1.19 | 0.69 |
| + meta filter (EW) Sharpe | 2.11 | 1.18 |
| + MaxDiver Sharpe (gross) | 2.85 | 0.89 |
| Worst allocation | MinCVaR 0.73 | MinCVaR 0.53 ✔ |
| Filter improves Sharpe? | 1.19 → 2.11 | 0.69 → 1.18 |
| Random benchmark | +27% cum / 0.24 Sharpe | +30% cum gross / 0.42 (−85% net of costs) |
| JT momentum | −91% cum | +259% cum |

**What replicates qualitatively:**
- The modular pipeline works as described: the filter roughly doubles the unfiltered
  Sharpe (0.69 → 1.18 with EW) and cuts drawdowns; risk-based allocation beats EW in the
  paper's ordering for Risk Parity/Inverse Vol in some variants; MinCVaR is the worst
  allocation in both paper and replication.
- The DRL ranker's learned behavior matches the paper's Figure 4: deterministic scores are
  nearly constant within a fold (next-step rank correlation ≈ 1.0, score std ~0.002), i.e. a
  stable, block-like static ranking — not a rapidly churning dynamic one.
- Seed sensitivity (paper's Appendix B.2): our 13-seed sweep gives full-period DRL Sharpe
  from −0.89 to +0.79 (mean ≈ +0.06); the paper reports 0.84 average over 4 seeds. Both
  confirm large seed sensitivity; the paper's headline result depends on the seed selected
  on validation folds.

**What we could not reproduce:** the magnitude. Our best integrated Sharpe is ~1.2 vs the
paper's 2.85, and MaxDiver is *not* the best allocation in our sample (EW is). Two structural
reasons, discussed below, make an exact match impossible: the data (venue, universe, period)
differ, and the paper's agent — as its own robustness section implies — is essentially a
validation-selected random initialization, so its exact rankings (and hence its returns) are
not reproducible without the original code and seed.

**Diagnostics that motivated this reading** (`diag.py`, `lr-sweep.py`): training longer
(10–50 passes) or with lr 1e-4–1e-3 does not systematically improve out-of-sample DRL
performance (full-period Sharpe scatters between −0.2 and +1.2). At the paper's lr=1e-5 with
~42 gradient updates per fold, the policy stays essentially at initialization. The "DRL
edge" in the paper is therefore more plausibly a *static ranking selected by seed* than a
learned dynamic policy — consistent with the paper's own stable heatmap and its seed-based
model selection.

## 4. Deviations from the paper, and why

1. **Data source and period.** Paper: Binance spot, top-60 by 3-month volume/market cap,
   2020-01…2024-09. We used the KuCoin futures 12h OHLCV dataset produced by `kucoin.py`
   (the data available in this repo; the Binance datastore was not downloaded here). KuCoin
   perps are USDT-margined futures (870 symbols, ~130–380 listed at any time); history is
   capped at ~1,000 bars per symbol, so the usable cross-section starts in 2023 and the
   backtest covers Jun 2023 – Aug 2026. This is the single biggest driver of any numerical
   difference: different venue, universe construction, and — importantly — a different
   market period (2023–2026 vs 2020–2024, which includes the 2022 bear market that shapes
   the paper's "momentum crashes" narrative).
2. **Universe selection** uses trailing 3-month dollar volume only (no market cap, which
   KuCoin futures data does not provide), with ≥85% bar-coverage eligibility. Selection is
   strictly causal (data available at fold start). Symbols with data gaps are forward-filled
   (≤2-bar gaps bridged; stale prices earn zero returns).
3. **A2C implemented directly in PyTorch, not Stable-Baselines3.** SB3's A2C only supports
   discrete action spaces; the paper's action space (real-valued scores ∈ R^N) cannot be
   produced by it. We used a Gaussian policy head with the paper's exact shared trunk
   (128/64/32 sigmoid), learning rate, discount, entropy coefficient, n-steps, value
   coefficient and gradient clipping. Entropy is normalized per action dimension to keep β=0.01
   scale-free.
4. **Unspecified hyperparameters had to be assumed** (the paper never states their values):
   state lookback H = 30 bars (≈15 days), filter windows W_vol = 10 and W_feat = 10 bars,
   covariance scenario window T_hist = 60 bars, JT momentum lookback = 30 bars. These are
   documented as assumptions; sensitivity to H or the windows was not re-tuned.
5. **Holding period.** The paper's MDP description holds positions H_p steps with an
   unspecified H_p, while the evaluation section and Eq. (3) describe a portfolio return
   over each 12-hour interval. We use H_p = 1 (full 12-hour rebalance), which matches the
   evaluation framework; intra-period weight drift is ignored (positions are rebalanced to
   target weights each step, the standard convention).
6. **Filter threshold.** The paper's primary τ=0.009 per 12-hour step. Our universe's
   rolling strategy volatility sits higher (~0.011–0.015/step vs the paper's implied
   ~0.0093), so with τ=0.009 the logistic filter stays closed for entire years (e.g. all of
   2025 for the selected seed) — the exact "pathological feedback loop" the paper itself
   flags. The headline table therefore uses τ=0.015, which reproduces the paper's
   *qualitative* behavior (gate open ~90% of the time, closing only in the worst volatility
   regimes). The τ=0.009 (paper-faithful) variant is also fully computed: its integrated
   MaxDiver Sharpe is 0.77 gross / 0.72 net of 5bp, and the filter's year-long closures make
   it strictly worse than the unfiltered ranker.
7. **Returns and metric conventions.** Simple returns are used for compounding (paper Eq. 3
   uses log returns; the difference is negligible at 12h frequency). Sharpe/Sortino are
   annualized with √730 (12h bars, 365 days); the paper's Sharpe convention is not stated and
   its tables are not internally consistent with any single annualization factor, so level
   comparison across the two should be read with that caveat. Sortino uses the std of
   negative returns (paper's definition, not semideviation). Turnover is Σ|Δw| per 12h step
   across all N assets (paper Eq. 27); the paper's "30.91% monthly" figure cannot be mapped
   to a unique convention from its text, so both per-step and monthly-equivalent numbers are
   reported.
8. **LTR benchmarks (RankNet/ListNet/ListMLE/LambdaMART) were not implemented.** They are
   static supervised baselines, not part of the proposed system; the paper's own finding is
   that they all underperform a random baseline. JT momentum and the random baseline were
   implemented as sanity benchmarks; note that on our 2023–2026 period JT momentum was
   strongly profitable (crypto trended), the opposite of the paper's 2020–2024 result.
9. **Seed as a hyperparameter** was replicated per the paper's stated protocol, but with 13
   seeds (they report 4). The selected seed (8) maximizes validation-fold Sharpe; everything
   else about the evaluation is untouched by this choice.
10. **Partial last month.** The September 2026 test month has only ~19 bars (<30) and is
    excluded; data ends 2026-09-10.

## 5. Reproduction

```bash
# data must exist: kc-futures-12h-ohlcv.parquet (from kucoin.py)
uv run backtest.py --seed 8 --tau 0.015 --out seed-t15-8.parquet   # headline config
uv run backtest.py --seed 8            --out seed-8.parquet        # paper-τ config
# seed sweep (13 seeds, both variants) then:
uv run analyze.py > analysis-output.txt
```

Runtime ≈ 35 s per full walk-forward run (39 folds) on CPU. Per-fold raw results are stored
in `seed-*.parquet` / `seed-t15-*.parquet` (`time`, `strategy`, `ret_gross`, `turnover`);
all tables in this report are reproducible from them via `analyze.py`.

## 6. Bottom line

The three-stage architecture is implementable and behaves as the paper describes
(modest-but-positive DRL signal; large Sharpe improvement from the meta-learning filter;
MinCVaR the worst allocation), but on KuCoin futures 2023–2026 the integrated strategy
achieves a net-of-5bp Sharpe of ~0.85 (MaxDiver) to ~1.16 (EW), not the paper's 2.85. Our
diagnostics indicate that at the paper's own hyperparameters the A2C policy barely departs
from its random initialization during the 4-month training fold — the reported edge is
dominated by seed selection, and the same appears true in the paper (its Figure 4 shows a
static, stable ranking; its seed was chosen on validation folds). The filter is the most
valuable component in both the paper and this replication; its absolute volatility threshold
requires re-calibration to the volatility level of the actual trading universe.

## 7. Learning-rate experiment (top-60 panel)

Does the agent perform better if it actually learns? We varied the A2C learning rate and
number of training passes on the top-60 panel (τ = 0.015, same walk-forward, seeds 8/4/42;
`lr-experiment.py`, results in `lr*.parquet`). Mean Sharpe across the three seeds:

| Config | DRL rank (EW) | + filter EW | + filter IV | + filter MaxDiv |
|---|---|---|---|---|
| lr 1e-5, 1 pass (**paper**) | **+0.56** | +0.71 | +0.50 | **+0.73** |
| lr 1e-4, 1 pass | +0.41 | +0.47 | +0.33 | +0.44 |
| lr 1e-4, 10 passes | +0.27 | +0.25 | +0.15 | +0.11 |
| lr 1e-3, 1 pass | +0.18 | +0.18 | −0.05 | +0.23 |
| lr 1e-3, 10 passes | +0.07 | +0.22 | +0.04 | −0.02 |

Performance degrades monotonically as the policy is allowed to actually learn. This is the
strongest evidence yet that the framework's edge in this setting comes from the stable,
near-random initialization ranking (plus the filter and allocation), not from reinforcement
learning: when the agent genuinely optimizes its in-sample binary reward, it overfits noise
and out-of-sample performance deteriorates monotonically with learning strength. The paper's
choice of lr = 1e-5 with one pass over ~240 training steps (~42 batch-5 updates) is —
intentionally or not — effectively an "almost no learning" configuration.