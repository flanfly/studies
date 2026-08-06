# Regime-Aware Universal Portfolio

Dmitrii Vlasiuk and Mikhail Smirnov (Department of Mathematics, Columbia University), December 2025, working paper.

## Overview

The paper asks whether exogenous side information in the form of market regimes improves the performance of universal portfolios in a large equity universe. The authors build a fixed large-capitalization universe — the 100 largest US stocks by market capitalization at an anchor date of 30 September 2015, drawn from CRSP common equities primarily listed on NYSE, NASDAQ, and AMEX — and form a value-weighted buy-and-hold "CRSP-100" index from it, using total returns with distributions reinvested and delisting proceeds parked in cash. The universe is frozen at the anchor date to avoid survivorship and look-ahead bias.

On this index, daily bull and bear regimes are inferred using the weighted sparse jump model (SJM) of Shu and Mulvey (2025), estimated on 24 technical, volatility, and macro-financial features. The regime label used for trading on day *t* is computed causally, using only information available by *t*−1. Regime awareness is implemented as a bull-cash exposure gate: in bull regimes the portfolio allocates across the risky universe with long-only weights; in bear regimes it holds cash. The economic rationale is tail-risk reduction rather than return forecasting.

The paper tests three hypotheses: (1) regime awareness improves risk-adjusted performance through lower volatility; (2) the regime-aware hindsight best constant rebalanced portfolio (CRP), which optimizes only over bull dates, can dominate the regime-blind best CRP because it lives in a different admissible class; (3) the regime-aware universal portfolio tracks its regime-aware hindsight CRP in the qualitative sense predicted by universal portfolio theory.

## Method

**Framework.** The market has *m* risky assets plus cash with price relatives `x_t = (x_{1t},...,x_{mt},x_{m+1,t})^T`, cash having `x_{m+1,t} ≡ 1`. A CRP holds fixed weights b and rebalances daily, with wealth `S_n(b) = ∏_{t=1}^n b^T x_t`; the hindsight benchmark is `S_n^* = sup_{b∈∆^{m+1}} S_n(b)`. The universal portfolio is the mixture `U_n = ∫ S_n(b) π(db)` over CRPs under a prior π (Cover 1991). With finite side information (Cover and Ordentlich 1996), a state-dependent CRP is a mapping β from states to portfolios; the regime-aware variant restricts the bear state to the all-cash corner `e_{m+1}` and the bull state to fully invested risky allocations `b^+ ∈ ∆^m`. Appendix A proves (Theorem A.5.2) that the regime-aware universal portfolio asymptotically competes in average log-wealth with the best bull allocation executed under the same regime path, with a regret constant `C_reg = 2C_A`.

**Regime identification.** The SJM segments the standardized feature series into K = 2 regimes (0 = low-mean/bear, 1 = high-mean/bull) by minimizing a weighted sum of squared deviations from regime centroids plus a jump penalty λ on regime switches, subject to nonnegative feature weights summing to at most κ (eq. 3.3.2). Hyperparameters are selected annually on a grid κ ∈ {0.5, 1, 2, 4, 6} and λ ∈ {10, 20, 40, 80, 160, 320} using an inner 80/20 train/validation split, choosing the pair with the best out-of-sample Sharpe of the bull-cash series `R̃_t^bc = 1{z_t=1} R_t^crsp` minus a penalty proportional to regime switches per year. The 24 features span trend/oscillator indicators (momentum 21/63, trend slope 63, MACD, stochastic %K, RSI), risk and drawdown measures (drawdowns at 63/126/252 days, realized volatility 21/63/126, downside semi-deviation, volatility-of-volatility), VIX level and transforms, and market/active-risk features versus SPY (Table 1; definitions in Appendix B.2).

**Universal portfolio implementation.** The continuous mixture is approximated with K = 4096 CRPs sampled from a mixture of Dirichlet distributions with concentration parameters α ∈ {0.1, 0.3, 1.0, 3.0} and mixture weights (0.20, 0.30, 0.30, 0.20), plus the simplex corners. The traded portfolio on day *t* is the wealth-weighted average of particle weights, updated multiplicatively on bull days; on bear days the portfolio is cash and particle weights are not updated. Rebalancing is daily.

## Results

Evaluation window: 30 September 2018 – 30 September 2025 (seven annual re-estimations). All figures are annualized; Sharpe uses a zero risk-free rate; no t-statistics are reported.

| Strategy | Ann. return (%) | Ann. vol (%) | Sharpe | Max drawdown (%) |
|---|---|---|---|---|
| CRSP-100 index (buy-and-hold) | 14.32 | 19.41 | 0.75–0.79 | −30.89 |
| Bull-cash overlay on index | 14.73 | 11.65 | 1.24 | −14.76 |
| Regime-blind universal portfolio | 12.43 | 18.48 | 0.73 | −34.69 |
| Regime-blind best CRP (hindsight) | 12.91 | 18.57 | 0.75 | −34.77 |
| Regime-aware universal portfolio | 13.63 | 11.16 | 1.20 | −13.65 |
| Regime-aware best CRP (hindsight) | 14.17 | 11.19 | 1.24 | −13.74 |

(Tables 2–3.)

Key findings:

- Regime awareness reduces maximum drawdown from 0.3469 to 0.1365 (about 61%) and annualized volatility from 0.1848 to 0.1116 (about 40%) relative to regime-blind trading, while annualized returns move only modestly (13.63% vs. 12.43% for the universal portfolios). Sharpe ratios rise above 1.2.
- The universal portfolios track their hindsight best-CRP comparators closely: the regime-blind universal portfolio underperforms its best CRP by ~0.48 pp annualized return and ~0.02 Sharpe; the regime-aware by ~0.53 pp and ~0.04 Sharpe — consistent with finite-sample regret and the particle approximation.
- The bull-cash overlay (Sharpe 1.238) nearly matches the regime-aware best CRP (Sharpe 1.240), indicating that most of the improvement comes from regime timing itself rather than cross-sectional optimization across constituents.
- The buy-and-hold index (14.32%) beating the regime-blind best CRP is not a contradiction: buy-and-hold is not in the CRP class, since CRPs rebalance to fixed weights daily while buy-and-hold weights drift.
- Model selection favored strong regularization: λ ∈ {80, 160} in six of seven re-estimation years and κ = 0.5 in five of seven (Table 4). Normalized feature weights concentrate on drawdown measures — dd 63/126/252 jointly account for about 54% of weight — while return lags receive negligible weight (Table 5).

## Conclusions and limitations

The authors conclude that regimes inferred from the Shu-Mulvey model provide economically meaningful side information for universal portfolios, working primarily through risk reduction (volatility, drawdown, and tail protection) rather than higher mean returns, and that the regime-aware universal portfolio exhibits behavior consistent with convergence toward its hindsight CRP benchmark within its admissible class.

Future work, per the authors: sensitivity to transaction costs, stability of the regime classifier under alternative feature sets and penalty choices, and robustness under different universes and cash proxies. The paper reports gross returns with no transaction-cost model, uses no leverage, and is strictly long-only (the authors note systematic short exposure was not robust in their sample). No text addressed to an automated system appears in the document.
