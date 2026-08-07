---
title: Regime-Aware Universal Portfolio
authors: Dmitrii Vlasiuk, Mikhail Smirnov
identifier: none given (working paper, Dept. of Mathematics, Columbia University, December 2025)
source_file: papers/.extracted/vlasiuk-and-smirnov-regime-aware-universal-portfolio/vlasiuk-and-smirnov-regime-aware-universal-portfolio.md
---

**Category:** modification — a bull/cash regime gate applied to the universal portfolios of Cover (1991) and Cover and Ordentlich (1996). Baseline and modified variants are reported side by side for both the universal portfolio and its hindsight best-CRP comparator.

### Claim

The paper applies a two-state (bull/bear) daily regime process, inferred by the weighted sparse jump model of Shu and Mulvey (2025) from 24 technical, volatility, and macro-financial features on a fixed CRSP-100 index, as side information for a universal portfolio. The regime-aware strategy invests fully in the risky universe in bull regimes and holds cash in bear regimes (§2.5), i.e. a "bull-cash" policy rather than long/short. On the evaluation window 2018-09-30 to 2025-09-30 the regime-aware universal portfolio earns an annualized return of 13.63% with 11.16% volatility, Sharpe 1.20, max drawdown −13.65%, versus 12.43% / 18.48% / 0.73 / −34.69% regime-blind (Table 3). The paper's claim is that regimes add value primarily through risk reduction — roughly 40% lower volatility and 60% smaller drawdown (calculated by the authors as 0.1848→0.1116 and 0.3469→0.1365) — not through higher mean return, and that the regime-aware universal portfolio tracks its regime-aware hindsight CRP qualitatively as finite-sample regret predicts (§4.2, §5). No t-statistics are reported anywhere; Sharpe ratios are computed with a zero risk-free rate (§4.2).

### Fit with our constraints

Instrument classes: one value-weighted equity index (CRSP-100) as the only risky asset class; cash is the bear leg. The strategy is long-only with a cash leg, so it needs no short stock, options, or short futures — the bear action is `e_{m+1}` (cash) with one-period gross return 1 (eq. 2.5.1). Long and short legs are not applicable; exposure gating is reported through volatility and drawdown rather than leg returns. No leverage is used; weights sum to one on a simplex (eq. 2.1.3). Replication needs CRSP daily total returns and shares outstanding for the top-100 names by market cap at 2015-09-30 (via WRDS), plus SPY (^SPY) and VIX (^VIX) series from Yahoo Finance via yfinance (§3.4). The regime label used for trading on day t is computed from information available by t−1 (causally, §3.5). Returns are gross; the paper states in §5 that sensitivity to transaction costs is future work, and no transaction cost or financing model is described anywhere.

### Strategies

**Baseline: regime-blind universal portfolio (Cover 1991; Cover and Ordentlich 1996).** The universal wealth is the mixture of CRP wealths `U_n := ∫_{Δ^{m+1}} S_n(b) π(db)` (eq. 2.3.1), with CRP wealth `S_n(b) := ∏_{t=1}^n b^T x_t` (eq. 2.2.2) where `x_t` collects daily price relatives and cash has `x_{m+1,t} ≡ 1` (eq. 2.1.2). Implementation: K = 4096 particles drawn from a mixture of Dirichlet distributions on Δ^m with concentration parameters α ∈ {0.1, 0.3, 1.0, 3.0} and mixture weights (0.20, 0.30, 0.30, 0.20), plus simplex corners including the all-cash corner (§4.2). Particle wealth updates multiplicatively and the traded portfolio is the wealth-weighted average `b̂_t = Σ_{k=1}^K π_{t-1}^{(k)} b^{(k)}` with `π_{t-1}^{(k)} = S_{t-1}^{(k)} / Σ_ℓ S_{t-1}^{(ℓ)}`. Applied daily, rebalanced to target weights each day; the hindsight comparator is `b* ∈ arg max_{b∈Δ^m} Σ_t log ⟨b, x_t⟩`, solved by mirror descent (exponentiated gradient, step size η_t = η_0/√t) but used only as comparator, not a trading rule (§4.1).

**Modification: regime-aware universal portfolio.** The delta: the admissible set is regime-conditioned — `B^+ := {(b,0) ∈ Δ^{m+1} : b ∈ Δ^m}` in bull, `B^- := {e_{m+1}}` in bear (eq. 2.5.1). The regime-conditioned CRP wealth is `S_n(b^+ | R) := ∏_{t=1}^n ((b̃^+)^T x_t)^{1{R_t=+}} ((b̃^-)^T x_t)^{1{R_t=-}}` (eq. 2.5.2), and the regime-aware universal portfolio is the mixture `U_n(R) := ∫_{Δ^m} S_n(b^+ | R) π^+(db^+)` (eq. 2.5.3). Operationally: if `Z_t = 0` (bear) then `b̂_t = e_cash` and particle weights are not updated; if `Z_t = 1` (bull) the discrete universal update runs and particles are updated on realized bull-day gross returns (§4.2). Theory: Theorem A.5.2 gives `log Ŝ_n^reg ≥ log S_n^{*,reg} − C_reg log n` with C_reg = 2C^A (Appendix A.5), i.e. per-period log-wealth regret vanishes.

**The regime signal itself** (the modification's conditioning variable). Estimated on daily log returns of the CRSP-100 index (eq. 3.2.5, value-weighted, top-100 by market cap at anchor t0 = 2015-09-30, frozen to avoid look-ahead; delisted names' proceeds parked in cash, §3.2). The weighted sparse jump model of Shu and Mulvey (2025) specializes to K = 2 regimes, minimizing the penalized objective (3.3.2): `min_{z,θ0,θ1,w} Σ_{t=1}^T ‖D_w^{1/2}(X_t − θ_{z_t})‖_2^2 + λ Σ_{t=2}^T 1{z_t≠z_{t-1}}` s.t. `z_t ∈ {0,1}`, `w_j ≥ 0`, `Σ w_j ≤ κ`. Features: p = 24, standardized by median/robust scale within each training window (§3.4, Table 1); supervised pre-screening keeps the 24 features most correlated with one-step-ahead CRSP-100 log return, done strictly inside the training window (§3.5). Estimation: annual expanding windows from 2018-09-30 to 2025-09-30; inner 80/20 train/validation split; grid κ ∈ {0.5, 1, 2, 4, 6}, λ ∈ {10, 20, 40, 80, 160, 320}; the pair maximizing the validation Sharpe of the bull-cash series `R̃_t^bc(λ,κ) = 1{z_t=1} R_t^crsp` minus a penalty proportional to regime switches per year is selected (§3.5). Tuning parameter usage: λ ∈ {80, 160} chosen in six of seven re-estimation dates; κ = 0.5 in five of seven (Table 4, Appendix B.1). Feature weight mass concentrates on drawdowns: dd 63, dd 126, dd 252 jointly ≈54% of normalized weight (Table 5, Appendix B.3).

**Universe and execution.** CRSP common equities listed NYSE/NASDAQ/AMEX as of 2015-09-30; top 100 by market cap `M_{i,t} = P_{i,t} q_{i,t}` (eq. 3.2.1) frozen at t0. Returns are CRSP total returns (dividends, splits included, eq. 3.2.2). Rebalance frequency: daily for CRPs and universal portfolios; the index itself is buy-and-hold. Execution timing: the regime used on day t is computed from information up to t−1 only (§3.5). Turnover is not reported for any strategy.

### Test setup

Sample period 2015-09-30 to 2025-09-30 for data and index construction; evaluation window 2018-09-30 to 2025-09-30 (regime design begins 2018-09-30, §3.5). No out-of-sample or post-publication period beyond this single window is reported, and no subperiod split is reported. Number of instruments: 100 equities in the universe; the traded risky asset is the single CRSP-100 index. Data sources: CRSP daily database via WRDS for returns/shares; Yahoo Finance via yfinance for ^VIX and ^SPY (§3.4). Survivorship/delisting: the universe is frozen at t0 and delisting proceeds are held in cash, described as avoiding survivorship and look-ahead bias (§3.2). Transaction costs: none — returns are gross; the paper lists transaction-cost sensitivity as future work (§5). Benchmark: the CRSP-100 buy-and-hold index (14.32% ann., 19.41% vol., 0.79 Sharpe, −30.89% max DD, Table 3) and the in-sample hindsight best CRPs, which the paper notes are not implementable (they use the full evaluation sample, §4.1).

### Results

Table 3 (all numbers below, evaluation window 2018-09-30 to 2025-09-30; annualized on 252 trading days; Sharpe at zero risk-free rate; max drawdown from cumulative wealth):

| Strategy | Ann. return (%) | Ann. vol (%) | Sharpe | Max drawdown (%) |
|---|---|---|---|---|
| CRSP-100 index (buy-and-hold) | 14.32 | 19.41 | 0.7870 | −30.89 |
| CRSP-100 bull-cash overlay | 14.73 | 11.65 | 1.2381 | −14.76 |
| UP regime-blind | 12.43 | 18.48 | 0.7268 | −34.69 |
| Best CRP regime-blind | 12.91 | 18.57 | 0.7472 | −34.77 |
| UP regime-aware | 13.63 | 11.16 | 1.2014 | −13.65 |
| Best CRP regime-aware | 14.17 | 11.19 | 1.2402 | −13.74 |

Earlier regime-signal-only results (Table 2): hold 14.32% / 19.41% / 0.75 vs. bull-cash 14.73% / 11.65% / 1.24.

The authors' conclusions (§4.2, §5): (i) the universal portfolios track their hindsight best CRPs with modest finite-sample gaps — regime-blind UP trails best CRP by ≈0.48 pp annualized return and ≈0.02 Sharpe; regime-aware by ≈0.53 pp and ≈0.04 — consistent with finite-sample regret and Monte Carlo approximation error; (ii) the regime gate is the dominant driver: the bull-cash overlay's Sharpe (1.238) is nearly identical to the regime-aware best CRP's (1.240), so cross-sectional optimization across constituents adds little; (iii) the buy-and-hold index out-earning both regime-blind strategies and the regime-aware UP does not contradict CRP theory because buy-and-hold is a drifting-weight strategy outside the CRP class (§4.1, §4.2). The best CRP has the higher Sharpe of the two regime-aware strategies (1.240 vs. 1.201), while the regime-aware UP has the smallest drawdown of all six series (−13.65%). Volatility and drawdown reductions are ≈40% and ≈61% (author-computed: 0.1848→0.1116, 0.3469→0.1365).

Commonly reported metrics absent: no t-statistics or standard errors anywhere; no information ratio; no turnover; no gross-vs-net split (all figures gross); no transaction-cost-adjusted results.

### Robustness

The paper reports no robustness section as such. Sensitivity of the regime classifier to alternative feature sets and penalty choices, to different universes, and to cash proxies is explicitly deferred to future work (§5). The only parameter-stability evidence is the reported usage distribution of λ and κ across the seven annual re-estimations (Table 4) and the normalized feature weights (Table 5).

### Specification search

The inner validation grid is κ ∈ {0.5, 1, 2, 4, 6} × λ ∈ {10, 20, 40, 80, 160, 320} — 30 (λ, κ) pairs evaluated at each of seven annual re-estimation dates — with the selected pair and its usage proportion reported (Table 4). Feature pre-screening ranks all candidate features and retains the top 24 by absolute correlation with one-step-ahead returns (§3.5); the full candidate feature count is not stated. Performance results are reported only for the selected specification and the six headline strategies.

### Limitations and future work

The authors' own: transaction-cost sensitivity is unquantified; stability of the regime classifier under alternative feature sets and penalty choices is untested; robustness under different universes and cash proxies is untested (§5). They also caution, citing White (2000), Hansen (2005), and Bailey et al. (2014), that small differences in average return across closely related rules should be read conservatively given multiple testing, and that their ordering claim rests on volatility/drawdown rather than mean return (§4.2).

### Separable variants for our pipeline

This is a modification paper, so the strategy list is mostly the variants already enumerated above:
- Bull-cash overlay on the CRSP-100 index, gated by the SJM regime signal (Table 2; §3.5) — the simplest tradable form.
- Regime-aware universal portfolio over the 100-asset universe with K = 4096 Dirichlet-mix particles (Table 3; §4.2).
- Regime-aware best CRP in hindsight (comparator only; Table 3; §4.1).
- Regime-blind universal portfolio and regime-blind best CRP (baselines; Table 3).
- The regime signal itself, defined by the objective (3.3.2), feature set (Table 1, Appendix B.2), and (λ, κ) grids, is separable and could be re-estimated on other factors or universes (Appendix B).

### Extraction gaps

The four figures (Figures 1–4) are embedded as images and were not readable from the text extraction; their captions are reproduced. Equation 3.3.2 is partially garbled in the extraction (the weighted-norm term is rendered as `D_w^{1/2}(X_t - θ_{z_t})` without the enclosing norm) but is reconstructible from the surrounding text. Equation 2.5.3, Table 4, and Table 5 are consistent between body and appendices. The paper gives no DOI, arXiv ID, or SSRN identifier. No instructions addressed to an automated system appear in the document. Table 2's header is rendered "Sharpt" (truncated "Sharpe"); the Sharpe figures (0.75, 1.24) are given in prose as well. The exact candidate feature count before pre-screening is not stated.
