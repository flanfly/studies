---
title: The VIX Futures Basis — Evidence and Trading Strategies
authors: David P. Simon (Bentley University); Jim Campasano (University of Massachusetts Amherst)
identifier: none stated in extraction (working paper dated April 16, 2013; no DOI/SSRN/arXiv given)
source_file: papers/.extracted/vix-paper-efma/vix-paper-efma.md
category: single-strategy
---

### Claim

The VIX futures basis — the front VIX futures price minus the VIX — has no significant forecast power for subsequent changes in the spot VIX over January 2006–December 2011, but has substantial forecast power for subsequent VIX futures price changes: a one-percentage-point basis is associated with a statistically significant 0.79-point VIX futures price decline over the following month (coefficient −.791, s.e. .276, Exhibit 4). The authors read this as the basis reflecting a harvestable volatility risk premium rather than the risk-neutral expected path of a mean-reverting VIX: with the spot VIX not forecast by the basis, convergence to an on-average-unchanged VIX means futures roll down a contango curve and up a backwardated curve. The strategy shorts the front VIX futures contract when the curve is sufficiently in contango and buys it when sufficiently in backwardation, hedging market exposure with mini-S&P 500 futures; mean per-contract profits after transaction costs are $792 (62 short trades, p = .003) and $1,018 (40 long trades), cumulative gains are $89,835 over January 2007–December 2011, and results are robust to costs, out-of-sample hedge ratios, and sample-split (Exhibit 5, Exhibit 7, Exhibit 8).

### Fit with our constraints

Instrument classes are VIX futures (the nearest contract with at least 10 business days to settlement) and mini-S&P 500 (E-mini) futures; no stocks or options are involved. The short leg is short VIX futures, expressible in futures (or a perpetual equivalent) rather than short stock or short options. Long and short legs are reported separately, as are hedged and unhedged P&L (Exhibit 5, Exhibit 8). Leverage is margin-based: minimum initial margins at the time of writing were $6,900 for one VIX futures contract and $4,375 for one mini-S&P contract, giving 55% annual compound returns on fully margined equity and 21% annual compounded at 5× margin — the authors call the former an extreme upper bound (§III). Replication needs tick-level CQG VIX futures quotes, spot VIX index values, and one-minute mini-S&P 500 futures data; the basis is constructed from 3:00–3:15 pm CST quotes, so execution timing is precisely defined. Microcaps, penny stocks, and delisting handling are not relevant (index futures).

### Strategies

**Short-in-contango / long-in-backwardation VIX futures roll capture (hedged with mini-S&P futures).** The signal is the *daily roll*, defined as the difference between the price of the front VIX futures contract that has at least ten business days to settlement and the VIX, scaled by the number of business days until settlement. Entry: short VIX futures when the basis is in contango and the daily roll exceeds +0.10 VIX futures points ($100 per day); long VIX futures when the basis is in backwardation and the daily roll is less than −0.10 VIX futures points (§III). Exit: when the daily roll falls below +0.05 (short trades) or rises above −0.05 (long trades), or after 9 business days if not triggered — the time cap avoids settlement issues (the contract settles 10+ days out) and generates an adequate number of trades. Trades are entered and exited in the last 15 minutes of the trading day (3:00–3:15 pm CST) at the first quote with a bid-ask spread ≤ .10 VIX futures point, or at the final close quotes; one VIX futures point is worth $1,000. The market hedge is sized by an out-of-sample hedge ratio, re-estimated daily on data from the start of 2006 through the previous day, from the regression

$$\Delta \text{VIX}^F_t = \beta_0 + \beta_1 \cdot \text{SPRET}_t + \beta_2 \cdot [\text{SPRET}_t \cdot \text{TTS}_t] + \mu_t \quad (3)$$

(full-sample estimates: −.018 (.022), −.717 (.047), +.011 (.002); RBAR² = .45, DW = 2.26, NOBS = 1511), where SPRET is the percentage change of the front mini-S&P futures and TTS is business days to settlement; a 1% S&P move maps to a .61-point VIX futures move at 10 business days to settlement. The hedge ratio is

$$\text{HR}_t = [\beta_1 \cdot 1000 + \beta_2 \cdot \text{TTS}_{t-1} \cdot 1000] / [0.1 \cdot \text{ES}_{t-1} \cdot 50] \quad (4)$$

averaging close to one mini-S&P contract per VIX futures contract (range roughly ½ to 2), fixed at trade outset. The economic rationale is risk-premium/flow-based: the authors attribute the persistently upward-sloped curve to tail-risk insurance demand bidding up long VIX futures, and cite the commodity carry and FX carry literatures (Erb and Harvey 2006; Gorton and Rouwenhorst 2006; Darvas 2009; Burnside et al. 2011); the other side of the short trade is long tail-risk insurance buyers. Forecast regressions are estimated monthly (last trading day of the month, NOBS = 71):

$$VIX^S_{t+1} - VIX^S_t = \alpha_0 + \alpha_1[VIX^F_t - VIX^S_t] + u_t \quad (1)$$

$$VIX^F_{t+1} - VIX^F_t = B_0 + B_1[VIX^F_t - VIX^S_t] + u_t \quad (2)$$

with α₁ = .231 (.279) insignificant, and −.178 (.549) in contango, −.483 (.779) in backwardation; β₁ = −.791 (s.e. .276, significant at 1%), −1.12 (.542, 5%) in contango, −1.54 (.76, 10%) in backwardation; RBAR² for the futures-change regressions .095/.058/.162 — the basis explains only about 10% of the variation (Exhibit 4). No parameter is winsorised or standardised; the threshold rules are as stated. Turnover: 62 short and 40 long trades over five years, average durations 6.4 days (short) and 5.0/3.4 days (long, sub-periods). Transaction costs: full bid-ask spreads on VIX futures both sides, half the minimum ¼-point tick ($6.25) on mini-S&P each side, $3 round-trip brokerage per contract; average total round-trip cost ≈ $140 per trade (§III). Statistical significance comes from 10,000 randomization trials (random entry days, actual average durations, interpolated p-values), not conventional t-tests.

### Test setup

Sample: January 2006–December 2011 for data and forecast regressions; trading simulations January 2007–December 2011, split into 1/07–6/09 and 7/09–12/11 halves (§I, Exhibit 8). The sample starts in 2006 because of gaps in earlier VIX futures trading. Data sources by name: CQG Market Data for all VIX futures bid/ask quotes, trades, and sizes; Pi Trading for spot VIX and rollover-adjusted front mini-S&P 500 futures (one-minute OHLC). Closing VIX futures quotes are the first 3:00–3:15 pm CST quote with bid-ask ≤ .10 point, else the final quotes of the day; spot VIX and mini-S&P values are the average of open and close of the matching minute, so the basis is synchronous (§I). No survivorship or delisting treatment is needed (futures). Headline numbers are net of the cost model; gross figures are not reported separately. The benchmark is the unhedged VIX futures position, reported alongside.

### Results

Exhibit 5 (full sample, from prose; the exhibit's table body is an image) — short trades: 62 trades averaging 6.4 days; mean hedged P&L $792 (p = .003), unhedged $861, hedge −$69, roll $831; winners-to-losers ≈ 2:1; bottom-decile cutoffs −$1,045 (hedged) vs −$1,973 (unhedged); Sortino 1.26 (hedged) vs 0.88 (unhedged), unhedged downside volatility about 50% higher. Long trades: 40 trades; mean hedged P&L $1,018 (described as highly statistically significant; p-value not stated in prose), roughly equal winners and losers; hedging cuts downside volatility by about a third (bottom decile −$1,539 vs −$2,683 unhedged) at a cost of $387 in mean P&L; Sortino 1.03 (hedged) vs 0.97 (unhedged); long-trade downside volatility about 50% higher than short-trade. Exhibit 7: cumulative hedged gains $89,835 over 2007–2011 with only minor drawdowns. Exhibit 8 (sub-periods, all net of costs):

| Trades | Period | Hedged P&L | Unhedged P&L | S&P Hedge P&L | Roll P&L | Semi Std (hedged/unhedged) | Sortino (hedged/unhedged) | Winners/Losers | P (hedged/unhedged) |
|---|---|---|---|---|---|---|---|---|---|
| Short | 1/07–6/09 | $991 | $444 | $547 | $533 | 503 / 1,170 | 1.97 / .38 | 12/7 | (.009) / (.135) |
| Short | 7/09–12/11 | $704 | $1,045 | −$341 | $963 | 680 / 875 | 1.04 / 1.19 | 28/15 | (.074) / (.109) |
| Long | 1/07–6/09 | $1,211 | $1,864 | −$653 | $1,711 | 1,205 / 1,678 | 1.00 / 1.11 | 10/13 | (.014) / (.016) |
| Long | 7/09–12/11 | $757 | $785 | −$28 | $822 | 585 / 1,055 | 1.29 / .74 | 9/8 | (.006) / (.029) |

(Exhibit 8; mean durations: short 4.8 and 7.1 days, long 5.0 and 3.4 days.) The authors conclude both legs are profitable in both halves; the first-half short-trade edge is largely the hedge tailwind (equity declines), while unhedged short P&L more than doubles from first to second half though neither unhedged short mean is statistically significant. Commonly-reported metrics absent from the paper: Sharpe ratios (Sortino is used because P&L distributions are frequently non-normal), gross returns, and annualized figures — only the $89,835 cumulative total and the margin-based compound estimates (55% fully margined, 21% at 5× margin) are given.

### Robustness

Robustness reported: sample halves (Exhibit 8, above); entry thresholds ±0.15 daily roll instead of ±0.10 — unreported results show "more profitable but fewer trades" (fn. 15); shorter un-triggered holding periods "fairly robust" (fn. 17); trimming the largest winning long and short trades leaves both legs significantly profitable over the full sample (§III); out-of-sample hedge ratios (constructed from 2006-through-previous-day estimates, updated daily); correlation between VIX futures P&L and mini-S&P P&L of −.7 with stable hedge-ratio estimates (fn. 24). The authors note the second-half short-trade p-value of .074 reflects the high bar set by randomly-entered short trades earning $243 in that half (fn. 27). The paper has no formal robustness section; these results appear inline.

### Specification search

The paper reports two entry thresholds (±0.10 tested in the exhibits; ±0.15 mentioned with unreported results), two exit forms (roll-based and 9-day cap, with shorter caps described as fairly robust), and hedged vs unhedged variants for each of two directional legs. Counts stated: 71 monthly regression observations; 62 short and 40 long trades. No other parameter grid is described; the ±0.15 results are not tabulated.

### Limitations and future work

Authors' own: whether the findings extend to other volatility products; the relationship between the VIX futures curve and the term structure of S&P 500 index option implied volatilities; and whether the roll in other futures markets can be exploited with market-risk hedging as here (§IV). They also note the sample is a single episode (2007–2011, including the financial crisis) and that tail-risk insurance demand during the period plausibly contributed to the short-side profitability (§IV).

### Separable variants for our pipeline

- Short leg: short front VIX futures when daily roll > +0.10, exit < +0.05 or 9 days (§III, Exhibit 5).
- Long leg: long front VIX futures when daily roll < −0.10, exit > −0.05 or 9 days (§III, Exhibit 5).
- Each leg hedged vs unhedged (hedge per eq. 4, out-of-sample) — reported separately throughout.
- Entry-threshold variant ±0.15 daily roll — defined but unreported (fn. 15).
- Basis-as-forecast regressions (eqs. 1–2) as a standalone conditioning relationship, monthly (Exhibit 4).

### Extraction gaps

Exhibit 5's table body is an image (`_page_31_Figure_2.jpeg`) — the full-sample winners/losers counts and the long-trade p-value appear only there and are not in extractable text; I could not verify the full-sample long-trade p-value or exact win/loss counts. Exhibits 6 and 7 are figures only (trade-level and cumulative P&L), read through the prose. The document is a working-paper PDF extraction with no DOI, journal reference, or appendix; it contains no instructions addressed to an automated system. The 2006–2011 forecast-test table (Exhibit 4) is rendered with ambiguous column headers in the extraction, but values were recovered from the prose and numeric cells.
