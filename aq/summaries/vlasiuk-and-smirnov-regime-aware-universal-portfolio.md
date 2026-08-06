---
title: Regime-Aware Universal Portfolio
authors: Dmitrii Vlasiuk, Mikhail Smirnov (Department of Mathematics, Columbia University)
identifier: none stated in the extracted text (no DOI/arXiv/SSRN given; dated December 2025)
source_file: papers/.extracted/vlasiuk-and-smirnov-regime-aware-universal-portfolio/vlasiuk-and-smirnov-regime-aware-universal-portfolio/vlasiuk-and-smirnov-regime-aware-universal-portfolio.md
---

Category: **comparison** with a modification core. Six strategies are evaluated side by side on one sample; the unit of interest is the *delta* of adding a regime gate to a universal portfolio. Baseline: the regime-blind universal portfolio of Cover (1991) and Cover and Ordentlich (1996). Change: condition the action set on a two-state bull/bear regime, fully invested in the risky universe on bull days and holding cash on bear days. The paper also contributes theory (Appendix A): a cone-based proof that the regime-aware universal portfolio's per-period log-wealth converges to that of the best regime-conditioned CRP, with regret constant C_reg = 2C_A log n (Theorem A.5.2).

### Claim

The paper asks whether market-regime side information improves universal portfolios on a large-cap US equity universe. It builds a value-weighted index from the 100 largest CRSP stocks at an anchor date (t0 = 30 September 2015), infers daily bull/bear labels from the weighted sparse jump model (SJM) of Shu and Mulvey (2025) estimated on p = 24 technical, volatility, and macro-financial features, and gates a Cover-style universal portfolio: long the risky universe in bull regimes, cash in bear. Over the evaluation window 2018-09-30 to 2025-09-30, regime awareness delivers similar annualized return to the regime-blind construction (13.63% vs 12.43% for the universal portfolios; 14.17% vs 12.91% for the best-CRP comparators) but substantially lower volatility (11.16% vs 18.48%) and maximum drawdown (−13.65% vs −34.69%), raising Sharpe ratios from about 0.73 to above 1.20 (Table 3). No t-statistic or standard error is reported for any figure. The paper's stated central finding is that regimes matter primarily through risk reduction, not mean return, and that the regime-aware universal portfolio tracks its hindsight CRP comparator in the qualitative sense predicted by universal portfolio theory.

### Fit with our constraints

Instrument classes: US large-cap common equities (100 names), plus a cash asset; SPY and VIX enter only as regime-model inputs, not as tradeable legs. The strategy is strictly long-only — the bear regime is cash, and the paper states explicitly that "systematic short exposure is not robust in our sample," which is why it adopts a bull-cash rather than long-short policy (§2.5). No short leg exists, so nothing needs expressing in futures or perpetuals; the long leg is a value-weighted basket of 100 equities that could be approximated by futures on a cap-weighted index only at the cost of the cross-sectional CRP allocation. Long and short legs are not reported separately because there is no short leg. Leverage: none used. Replication needs: CRSP daily total returns and shares outstanding via WRDS for the 100-name universe, SPY and VIX daily closes from Yahoo Finance (yfinance); the regime model requires the full 24-feature construction (Table 1, Appendix B.2). Survivorship is handled by freezing the universe at the anchor date and freezing delisted names' returns at their exit (delisting terminal cash flow included, proceeds then held as cash). Turnover is not quantified anywhere in the paper.

### Strategies

All six strategies share the CRSP-100 universe, the value-weighted index, the daily regime process, and the evaluation window 2018-09-30 to 2025-09-30. Performance metrics are defined once, identically for all: annualized geometric return from daily gross returns, annualized volatility of daily simple returns scaled by √252, Sharpe ratio with a zero risk-free rate, and maximum drawdown from cumulative wealth (Table 3).

**CRSP-100 index (buy-and-hold).** The baseline risky asset: a value-weighted portfolio of the 100 largest names at t0, weights w_i = M_{i,t0}/Σ_j M_{j,t0} with M_{i,t} = P_{i,t}q_{i,t}, total returns reinvested, delisting proceeds parked in cash (§3.2). Weights drift with relative prices; it is not a CRP, and the paper stresses that buy-and-hold is outside the CRP benchmark class, so the best CRP is not required to beat it.

**Bull-cash overlay.** The same index gated by the regime: return series R̃_t^bc = 1{z_t=1} R_t^crsp — fully long on bull days, cash on bear days (§3.5). This is the paper's simplest demonstration that the regime signal carries information; its Sharpe (1.238) nearly equals that of the regime-aware best CRP (1.240), which the authors cite as evidence that regime timing rather than cross-sectional optimization drives the improvement.

**Regime-blind universal portfolio.** The classical Cover construction: a wealth-weighted mixture over constant rebalanced portfolios b ∈ Δ^m, updated daily (§2.3, §4.2). The continuous mixture is approximated by K = 4096 particles drawn from a mixture of Dirichlet distributions with concentration parameters α ∈ {0.1, 0.3, 1.0, 3.0} and mixture weights (0.20, 0.30, 0.30, 0.20), plus simplex corners including the all-cash corner. Day-t weights are the wealth-weighted posterior mean b̂_t = Σ_k π_{t-1}^{(k)} b^{(k)} with π_{t-1}^{(k)} = S_{t-1}^{(k)}/Σ_ℓ S_{t-1}^{(ℓ)}. Rebalanced daily.

**Regime-blind best CRP in hindsight.** The oracle comparator b* ∈ arg max_{b∈Δ^m} Σ_t log⟨b, x_t⟩, solved numerically by mirror descent / exponentiated gradient with step size η_t = η_0/√t (§4.1). Used only as a benchmark, not a trading rule; not implementable ex ante.

**Regime-aware universal portfolio.** The modification. Same particle construction, but the action set is regime-conditioned: on bear days (Z_t = 0) b̂_t = e_cash and particle wealths are not updated; on bull days (Z_t = 1) b̂_t is the wealth-weighted particle mean and particles update on realized bull-day gross returns (§4.2). The regime label used for trading on day t is computed causally from information available up to t−1 (§3.5, §5). On bull days it is fully invested with zero cash weight; the admissible sets are B^+ = {(b,0) ∈ Δ^{m+1} : b ∈ Δ^m} and B^- = {e_{m+1}} (§2.5).

**Regime-aware best CRP in hindsight.** The oracle over the regime-conditioned class: b_+* ∈ arg max_{b∈Δ^m} Σ_{t∈T_+} log⟨b, x_t⟩ where T_+ = {t : Z_t = 1}, applied on bull days and returning 1 on bear days (§4.1).

The comparison the paper draws: the regime gate (strategies 2, 5, 6) compresses the left tail and cuts volatility roughly 40% and drawdown roughly 60% versus the regime-blind analogues, while the regime-aware universal portfolio stays within about 0.5 percentage points of annualized return and 0.04 of Sharpe of its hindsight oracle — the finite-sample analog of the theoretical log-wealth convergence (Theorem A.5.2). The regime-blind universal portfolio's larger drawdowns are attributed to continuous rebalancing into underperformers with no timing mechanism.

### Test setup

Sample period 2015-09-30 to 2025-09-30, with the first three years (to 2018-09-30) used for regime-model training and the evaluation window 2018-09-30 to 2025-09-30 for all reported strategies; how this choice was made beyond "anchor date" and "regime design window begins on 30 September 2018" is not explained (§3.5). No out-of-sample or post-publication period beyond 2025-09-30 is reported; the paper is dated December 2025, so the sample ends roughly two months before publication. Instruments: 100 equities, fixed. Data sources: CRSP daily stock database via WRDS (total returns, prices, shares outstanding); CBOE VIX (ticker ^VIX) and SPY from Yahoo Finance via yfinance. Survivorship/delisting: universe frozen at t0, delisted names' return histories frozen at exit, terminal delisting cash flow included, proceeds held in cash thereafter (§3.2). Transaction costs: not modeled; no net-of-cost figures appear anywhere, and the paper's concluding section lists transaction-cost sensitivity as future work. Financing costs: none; cash gross return ≡ 1 and Sharpe uses a zero risk-free rate. Regime labels are lagged one day relative to the data they condition on ("the label used for trading at time t is determined from information available by time t−1"); feature pre-screening is performed strictly inside each training window and does not access the evaluation segment (§3.5).

### Results

Table 3 is the paper's results table (six strategies); Table 2 reports the two regime-process diagnostics. All figures below are annualized except max drawdown.

| Strategy | Ann. return (%) | Ann. vol (%) | Sharpe | Max drawdown (%) |
|---|---|---|---|---|
| CRSP-100 index (buy-and-hold) | 14.32 | 19.41 | 0.7870 | −30.89 |
| Bull-cash overlay on CRSP-100 | 14.73 | 11.65 | 1.2381 | −14.76 |
| Regime-blind universal portfolio | 12.43 | 18.48 | 0.7268 | −34.69 |
| Regime-blind best CRP (hindsight) | 12.91 | 18.57 | 0.7472 | −34.77 |
| Regime-aware universal portfolio | 13.63 | 11.16 | 1.2014 | −13.65 |
| Regime-aware best CRP (hindsight) | 14.17 | 11.19 | 1.2402 | −13.74 |

Source: Table 3. Table 2 reports the regime-process diagnostics on the same window: hold (the index) 14.32% / 19.41% / Sharpe 0.75, bull-cash 14.73% / 11.65% / Sharpe 1.24 (the 0.75 vs 0.7870 discrepancy between Tables 2 and 3 for the same series is a rounding difference in Table 2).

The authors' reading: regime awareness changes the benchmark class so that the comparator optimizes only over bull dates; the regime-aware best CRP dominates the regime-blind best CRP (14.17 vs 12.91, Sharpe 1.24 vs 0.75); the universal portfolios track their respective oracles within 0.48–0.53 pp of annualized return and 0.02–0.04 of Sharpe; and the buy-and-hold index beating the regime-blind best CRP (14.32 vs 12.91) does not contradict CRP theory because buy-and-hold is outside the CRP class. They emphasize that commonly reported quantities absent here include any t-statistic, standard error, net-of-cost figure, or turnover number; the paper reports Sharpe ratios without significance tests and comments that small mean-return differences across closely related rules should be read conservatively in light of data-snooping concerns (citing White 2000; Hansen 2005; Bailey et al. 2014), with the case resting on volatility and drawdown reductions from a structural design choice rather than point estimates of mean return.

### Robustness

There is no robustness section. The only parameter-level evidence is in Appendix B: the annual grid selection of (λ, κ) is stable — λ ∈ {80, 160} chosen in six of seven re-estimation years, κ = 0.5 in five of seven — and the normalized feature weights concentrate on drawdown measures (dd 63, dd 126, dd 252 jointly ≈ 54%, Table 5), which the authors read as the segmentation keying off persistent stress. No subperiod splits, alternative universes, delayed-execution tests, alternative cash proxies, or factor controls are reported; the paper states the stability of the classifier under alternative feature sets and penalty choices as future work.

### Specification search

The reported search is the annual inner grid over λ ∈ {10, 20, 40, 80, 160, 320} × κ ∈ {0.5, 1, 2, 4, 6} (30 pairs), re-run at each of seven annual re-estimation dates, with selection by validation-block Sharpe of the bull-cash series minus a linear penalty proportional to regime switches per year (§3.5, Table 4). Feature pre-screening ranks the 24 features by in-training-window correlation with one-step-ahead CRSP-100 returns and retains 24 (i.e., the full set, since p = 24). Results are reported only for the selected (λ, κ) each year and the single final regime path; the paper does not report performance of rejected parameter pairs or alternative feature subsets.

### Limitations and future work

The authors' own list (§5): quantify sensitivity of conclusions to transaction costs; test stability of the regime classifier under alternative feature sets and penalty choices; examine robustness under different universes and cash proxies. They also note the finite-sample and Monte-Carlo-approximation gaps between the universal portfolio and its hindsight oracle (gradual learning, long stretches of similar returns where learning signals are weak), and flag data-snooping / backtest-overfitting concerns motivating conservative interpretation of small average-return differences.

### Separable variants for our pipeline

- The **regime gate itself**: daily bull/bear labels from the SJM on 24 features, selection grids λ × κ, causality constraint (label for day t uses data to t−1) (§3.5). This is the paper's separable conditioning signal.
- **Bull-cash overlay** on a cap-weighted index (Table 2/Table 3) — the regime signal used with no cross-sectional optimization.
- **Regime-blind universal portfolio** with the Dirichlet particle mixture, α ∈ {0.1, 0.3, 1.0, 3.0}, weights (0.20, 0.30, 0.30, 0.20), K = 4096 (§4.2).
- **Regime-aware universal portfolio** — same particles, bear days hold cash with no particle updates, bull days update (§4.2).
- The **feature set** of p = 24 predictors with exact windowed definitions (Appendix B.2) is itself a separable input for any regime classifier; Table 5 gives its learned importance weights.

### Extraction gaps

- Figures 1–4 are embedded images (four JPEGs in the extraction directory) and could not be read; their captions and the prose describe their content (cumulative wealth paths in Figure 2; λ and κ chronological paths in Figures 3–4, both piecewise constant by construction). Figure 1's caption is attached in the extraction to the file named `_page_13_Figure_2.jpeg`, so the figure-file mapping is unreliable.
- Table 3's row labels were jumbled by the table extraction (e.g., "best crp regime aware | 14 | 17 | 11.19 | 1.2402 | -13.74"); all values were cross-checked against the prose, which states each figure explicitly and matches.
- Equation (3.3.2) renders as `D_w^{1/2}(X_t − θ_{z_t})\ _2^2` — the squared-norm notation is garbled but the content is unambiguous (weighted squared deviations from regime centroids).
- No DOI/arXiv/SSRN identifier appears in the extracted text.
- No text addressed to an automated system was found in the document.
