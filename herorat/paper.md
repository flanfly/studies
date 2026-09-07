

# How to beat the market with the Implied Volatility Term Structure: The HeroRATs Strategy. Chrilly Donninger Chief Scientist, Sibyl-Project Sibyl-Working-Paper, Dec. 2013 <http://www.godotfinance.com/>

*The **giant pouched rats** of sub-Saharan Africa are large muroid rodents. Their head and body lengths range from 25–45 cm with scaly tails ranging from 36–46 cm. They weigh between 1.0 and 1.5 kg. Giant pouched rats are only distantly related to the true rats. These rats are becoming useful in some areas for detecting land mines, as their acute sense of smell is very effective in detecting explosives, and they are light enough to not detonate any of the mines. The rats are being trained by APOPO, a nonprofit social venture based in Tanzania. APOPO is also training the rats to detect tuberculosis by sniffing sputum samples; the rats can test many more samples than a scientist using more traditional methods. Land mine and tuberculosis sniffing rats are called **HeroRATs**.*  
([en.wikipedia.org/wiki/Giant\\_pouched\\_rat](http://en.wikipedia.org/wiki/Giant_pouched_rat))

Note: The HeroRAT on the wikipedia-pic is missing his two best rather impressive parts.

## Abstract:

In [1] the authors propose a simple paired-switching strategy. One selects a market-long ETF (SPY) and a negative correlated ETF (TLT). Depending on the 13-weeks momentum one holds either the SPY or the TLT.

This working-paper is a considerable refinement of this simple idea. Instead of the momentum the implied-volatility-term-structure (IVTS) is used as a selection criterion. There are no fixed periods, but the decision is taken on a daily basis. The IVTS was used successfully in a series of previous working-papers (see [2],[3],[4],[5]). This working paper extends the IVTS by the recently introduced VIX-Short-Term Index VXST ([6]). The performance can further be significantly improved by filtering the IVTS with a median filter. The resulting strategy is very easy to implement and has minimal trading-costs. It beats the market (SPY) by a wide margin.

## Introduction:

In [7] the authors extract the implied S&P-500 distribution from market options prices. This distribution is subsequently transformed to the corresponding risk-adjusted one. In the next step optimal portfolios consisting of a risky (S&P-500) and a risk-free asset (LIBOR-rate) are formed and their out-of-sample performance is evaluated. Extracting the implied S&P-500 distribution from option prices is in an ideal world mathematically relatively straightforward. For real data it is complex and unreliable. One has to differentiate the options prices twice by their strike. But these prices are only available at a few discrete points. Especially for (far) OTM options the prices are unreliable (see [5]). If one uses the last sell price of a trading day, one compares prices at different times. The usual remedy is to use the mean of bid and ask. But bid and asks are sometimes completely unrealistic. It is next to impossible to adjust to these points a double-differentiable spline which meets basic consistency criterion's. In a second attempt the authors of [7] use an estimation method from Bayesian statistics to calculate the implied distribution. This method works slightly better.

The subsequent transformation to a risk-adjusted distribution is completely arbitrary. One has to assume a utility function and risk-averse parameters. The results in [7] are also not very convincing.

The method gives – under some utility-measures – acceptable results in the 1990, but does not significantly outperform the market in the last decade. The approach in [1] is in contrast a very simple-minded momentum strategy. One switches every quarter (13-weeks) between the SPY or TLT (20+ Year Treasury Bond). The selection criterion is the performance in the last quarter.

This work is in spirit alongside the ideas of [7]. But the hard work is delegated to the CBOE. The strategy is as easy to implement as the momentum-approach.

Instead of the SPY one uses the VBK (Vanguard Small Cap Growth ETF). The VBK is one of the largest (14.87 Billion \$) and most liquid ETFs. The VBK outperforms the SPY in quiet market-regimes. It has a greater downside risk in times of troubles. But the IVTS smells like real life HeroRATs the danger and one switches to the TLT. The TLT is not at all risk-free, but it has in turbulent times an attractive return. One could use the more conservative SHY (1-3y), IEI (3-7y) or IEF (7-10y) instead.

The HeroRATs have not only to smell the danger, but also when the danger is over and one should switch back to the VBK.

The allocation in [7] is continuous. In this study the allocation is considerable simplified. One has only 3 market-regimes. In quiet times one holds 100% the VBK, in the intermediate state one splits equally 50:50 and in the turbulent market regime one invest 100% in TLT.

## The Implied Volatility Term Structure:

The Implied Volatility Term Structure is generally defined as:

$$IVTS(t) = \text{Short-Term-IV}(t) / \text{Long-Term-IV}(t) \quad (1)$$

One selects on this curve points with readily available implied volatility measures.

This study considers for the *Short-Term-IV* the VXST (9-days IV calculated from weekly options), the VIX (1-month from monthly options) and the VX30.

The VX30 is the price of a VIX future with a maturity of 30 calendar days. Usually such a future does not exist. In this case one calculates the weighted mean of the 1<sup>st</sup> and the 2<sup>nd</sup> nearest future. If the first future has a maturity of 20 days, and the second a maturity of 50 days, one calculates the VIX futures 30 value as  $2/3 * \text{future}_1 + 1/3 * \text{future}_2$ . VIX futures have an own inherent logic, but there is obviously a close relation with the implied volatility-surface of SPX options.

For the *Long-Term-IV* the VIX, the VXV (3-months from monthly options), the VX30 and the VX45 are used. The VX45 is the price of a VIX future with 45 days maturity. It is the weighted mean of the 2<sup>nd</sup> and 3<sup>rd</sup> future. One can select any maturity, but 30 and 45 days have been the most useful ones in previous studies.

The considered combinations are:

$$\begin{aligned} \text{VXST}(t) / \text{VIX}(t) & \quad (1a) \\ \text{VXST}(t) / \text{VXV}(t) & \quad (1b) \\ \text{VXST}(t) / \text{VX30}(t) & \quad (1c) \\ \text{VIX}(t) / \text{VXV}(t) & \quad (1d) \\ \text{VIX}(t) / \text{VX30}(t) & \quad (1e) \\ \text{VIX}(t) / \text{VX45}(t) & \quad (1f) \\ \text{VX30}(t) / \text{VX45}(t) & \quad (1g) \end{aligned}$$

The VXST was introduced in Oct. 2013. CBOE provides a daily updated time series starting at 2011-01-03. In a first step the efficiency of (1a) to (1g) is compared since this date. In a second step the performance of the measures (1d) to (1g) are evaluated since 2008-01-03.

The calculation ends at 2013-12-11 (the latest available daily data at this writing). Within the shorter period there is one larger crash in August 2011. The longer period contains additionally the 2008 market meltdown.

As in previous studies one starts with an initial index of 500.00\$. One defines the market-regimes by a low- and high IVTS threshold. The thresholds depend on the IVTS combination. If the IVTS is below the low-threshold, one invests the full index in VBK. If the IVTS is between the low- and high-threshold, one splits 50:50 between VBK and TLT. If the IVTS is above the high-threshold, the index is fully invested into TLT.

**Trading-Note:** There are no trading-activities in the low- and high-state. But one has to re-balance in the intermediate state to keep the 50:50 ratio. This was done in the historic simulation.

The calculation is done each day with close-prices. In real trading live one does this shortly before the close. According to the trading experience in the Sibyl Fund with other IVTS based strategies, there is in the long run no systematic difference between the real- and the theoretic behavior. But it was noticed that the strategy suffers from IVTS spikes (this is especially for trading the VXX aka the Mojito Strategy [2] a problem). It is well known from image-processing that median-filters are very efficient to filter out salt and pepper distortions (salt and pepper are the image processing terms for spikes). The median-3 filter uses for the trading decision at time  $t$  the median of  $IVTS(t)$ ,  $IVTS(t-1)$  and  $IVTS(t-2)$ . The median-5 filter extends the median calculation from  $IVTS(t)$  back to  $IVTS(t-4)$ . The median-3 is able to filter a single-day spike away. The median-5 filter is robust to a 2-days spike or 2 single-days spikes. The general idea is to ignore short-term market overreactions. The downside is that any filter delays the reaction to long-term market turnarounds.

**Statistical Note:** The median is the second-best combination of robustness and minimum delay. The best is the recursive-median filter. For the recursive-filter, one uses for the old values not the original but the already filtered data. One can apply the recursive median only in symmetric-filter applications. In case of the median-3 filter one takes for time- $t$  the values of  $t-1$ ,  $t$ ,  $t+1$ . Medical- or image-processing applications are symmetric problems. But in our case one can't wait for tomorrow's IVTS for trading today. One can smooth the median-filtered values again with the median-filter and gets - after several passes - a stable result. The recursive-filter is already after the first pass stable (see [9])

Graphic-1 shows the performance of the (1d) IVTS: VIX/VXV. The low-threshold is 0.96, the high-threshold is 1.02. The IVTS is filtered with the median-5. The yellow line is in all graphics the performance of the SPY.

One sees the effect of the median-5 in the blocky appearance of the bottom IVTS chart. The HeroRATs activate in the 2011 crash on August 1<sup>st</sup> the alarm. The position is switched to 100% TLT (red-line in Graphic-1). The danger is already smelled before. At 2011-07-27 one switches from 100% VBK to the intermediate 50:50 state. The maximum relative drawdown is hence not during this marked crash, but on 2012-07-04. This was not a crash-period. The markets dragged slowly down and the IVTS moved only slightly up. The HeroRATs are trained to detect market-mines and not usual head-wind or sideways conditions. The HeroRATs beat the SPY in this time period with 81.3% to 49.0%. The maximum relative drawdown is 12.9% to 18.6%. There is less risk and more fun.

Graphic-2 shows the same setting without any filtering. The overall advantage of the HeroRATs has gone. The main difference is the behavior in Autumn 2011. The market is still nervous and the positions switch several times. This whipsaw effect costs performance. The median-5 filters ignores most of the whipsaws. The behavior of the median-3 filter is between. This results holds for all IVTS measures. The effect is most pronounced when the very reactive VXST is involved.

![Screenshot of a financial charting interface showing VIX-VXV with a median-5 filter. The interface includes input fields for SymbolLow (VBK), SymbolHigh (TLT), IVTS (VIX-VXV), Filter (Med-5), LowThres (0.96), HighThres (1.02), LowWeights (100-0), MidWeights (50-50), HighWeights (0-100), and From (2011-03). Four charts are displayed: 'value' (VIX-VXV in orange, SPY in yellow), 'Drawdown' (VIX-VXV in green), 'IVTS' (VIX-VXV in purple), and a comparison chart. The 'value' chart shows a significant dip in August 2011, with a red vertical line and a label 'Aug 01, 2011'. The 'Drawdown' chart shows a peak of 7.58 at that time. The 'IVTS' chart shows a peak of 1.03. The comparison chart shows VIX-VXV and SPY from 2011-03 to 2013-12-11.](5d92d5c9cc01a262b0389d138caa9aea_img.jpg)

Firefox

Market-Timing Strategy

SymbolLow: VBK SymbolHigh: TLT IVTS: VIX-VXV Filter: Med-5 LowThres: 0.96 HighThres: 1.02 LowWeights: 100-0 MidWeights: 50-50 HighWeights: 0-100 From: 2011-03 IVTS: On Draw

value VIX-VXV 2.32% SPY 2.31%

chart by amcharts.com

80% 60% 40% 20% 0% -20%

172.75 511,607.07

Aug 01, 2011 2012 Jul 2013 Jul

Drawdown VIX-VXV 7.58

chart by amcharts.com

10 5 0

7.58

IVTS VIX-VXV 1.03

chart by amcharts.com

1.0 0.8 0.6

1.03

chart by amcharts.com

From: 03-01-2011 to: 12-12-2013 Zoom: 1-M 3-M 6-M 1-Y 2-Y MAX

Screenshot of a financial charting interface showing VIX-VXV with a median-5 filter. The interface includes input fields for SymbolLow (VBK), SymbolHigh (TLT), IVTS (VIX-VXV), Filter (Med-5), LowThres (0.96), HighThres (1.02), LowWeights (100-0), MidWeights (50-50), HighWeights (0-100), and From (2011-03). Four charts are displayed: 'value' (VIX-VXV in orange, SPY in yellow), 'Drawdown' (VIX-VXV in green), 'IVTS' (VIX-VXV in purple), and a comparison chart. The 'value' chart shows a significant dip in August 2011, with a red vertical line and a label 'Aug 01, 2011'. The 'Drawdown' chart shows a peak of 7.58 at that time. The 'IVTS' chart shows a peak of 1.03. The comparison chart shows VIX-VXV and SPY from 2011-03 to 2013-12-11.

Graphic-1: VIX-VXV with median-5 filter. 2011-01-03 to 2013-12-11

![Screenshot of a financial charting interface showing VIX-VXV without a filter. The interface is similar to Graphic-1, but the Filter is set to 'None'. The 'value' chart shows VIX-VXV (orange) and SPY (yellow) from 2011-03 to 2013-12-11. The 'Drawdown' chart shows a peak of 1.4. The 'IVTS' chart shows a peak of 1.4. The comparison chart shows VIX-VXV and SPY from 2011-03 to 2013-12-11.](7fb5215fd72210a2e4cce6df55550c89_img.jpg)

Firefox

Market-Timing Strategy

SymbolLow: VBK SymbolHigh: TLT IVTS: VIX-VXV Filter: None LowThres: 0.96 HighThres: 1.02 LowWeights: 100-0 MidWeights: 50-50 HighWeights: 0-100 From: 2011-03 IVTS: On Draw

value VIX-VXV SPY

chart by amcharts.com

40% 20% 0% -20%

Jul 2012 Jul 2013 Jul

Drawdown VIX-VXV

chart by amcharts.com

10 5 0

1.4

IVTS VIX-VXV

chart by amcharts.com

1.4 1.0 0.8 0.6

chart by amcharts.com

From: 03-01-2011 to: 12-12-2013 Zoom: 1-M 3-M 6-M 1-Y 2-Y MAX

Screenshot of a financial charting interface showing VIX-VXV without a filter. The interface is similar to Graphic-1, but the Filter is set to 'None'. The 'value' chart shows VIX-VXV (orange) and SPY (yellow) from 2011-03 to 2013-12-11. The 'Drawdown' chart shows a peak of 1.4. The 'IVTS' chart shows a peak of 1.4. The comparison chart shows VIX-VXV and SPY from 2011-03 to 2013-12-11.

Graphic-2: VIX-VXV without filter. 2011-01-03 to 2013-12-11

![Screenshot of the 'Market-Timing Strategy' software interface showing performance charts for VXTS-VIX and SPY from 2011-01-03 to 2013-12-11. The interface includes input fields for SymbolLow (VBK), SymbolHigh (TLT), IVTS (VXTS-VIX), Filter (Med 5), Low-Thres (1.01), High-Thres (1.13), Low-Weights (100-0), Mid-Weights (50-50), High-Weights (0-100), and From (2011-03). The main chart displays the value of VXTS-VIX (orange line) and SPY (yellow line) over time, with a final value of 925,638.33 for VXTS-VIX and 178.72 for SPY. Below the main chart are two smaller charts: 'Drawdown' for VXTS-VIX (2.91) and 'IVTS' for VXTS-VIX (1.02). The interface also includes a 'Select' dropdown for VXTS-VIX and a 'Compare to' dropdown for SPY.](de6e8b740c69dac308cce9edfec3eff4_img.jpg)

Screenshot of the 'Market-Timing Strategy' software interface showing performance charts for VXTS-VIX and SPY from 2011-01-03 to 2013-12-11. The interface includes input fields for SymbolLow (VBK), SymbolHigh (TLT), IVTS (VXTS-VIX), Filter (Med 5), Low-Thres (1.01), High-Thres (1.13), Low-Weights (100-0), Mid-Weights (50-50), High-Weights (0-100), and From (2011-03). The main chart displays the value of VXTS-VIX (orange line) and SPY (yellow line) over time, with a final value of 925,638.33 for VXTS-VIX and 178.72 for SPY. Below the main chart are two smaller charts: 'Drawdown' for VXTS-VIX (2.91) and 'IVTS' for VXTS-VIX (1.02). The interface also includes a 'Select' dropdown for VXTS-VIX and a 'Compare to' dropdown for SPY.

Graphic-3: VXTS-VIX with median-5 filter. 2011-01-03 to 2013-12-11

Graphic-3 shows the performance of IVTS (1a), VXST/VIX. The result is improved from 81.3 to 85.1%. The max. relative drawdown reduces from 12.9% to 10.7%. One has to increase the thresholds to 1.01 and 1.13. The upper bound is generally not very sensitive. The results are more sensitive to a change of the lower threshold. The VXST reacts very fast. Using the median-5 filter is essential. The unfiltered signal is outperformed by the SPY. The performance of the median-3 is in between.

| IVTS      | Low-Threshold | High-Threshold | P&L    | Drawdown |
|-----------|---------------|----------------|--------|----------|
| VXST/VIX  | 1.01          | 1.13           | +85.1% | 10.7%    |
| VXST/VXV  | 1.10          | 1.15           | +67.7% | 21.0%    |
| VXST/VX30 | 1.00          | 1.13           | +85.9% | 12.9%    |
| VIX/VXV   | 0.96          | 1.02           | +77.7% | 12.9%    |
| VIX/VX30  | 0.97          | 1.10           | +84.6% | 12.9%    |
| VIX/VX45  | 0.98          | 1.06           | +75.0% | 12.9%    |
| VX30/VX45 | 0.95          | 1.05           | +51.5% | 10.3%    |
| VBK       | —             | —              | +49.5% | 28.9%    |
| SPY       | —             | —              | +49.0% | 18.6%    |

Table-1: Thresholds and Performance from 2011-01-03 to 2013-12-11

Table-1 shows the thresholds and the performance of the different IVTS measures. The last 2 rows are Buy&Hold for VBK and SPY. The overall performance is almost identical, but the drawdown of the VBK is during the crash in summer 2011 considerable higher. The highest profit is for the combination VXST/VX30. But the faster reacting VXST/VIX ratio is close and shows a lower drawdown. If one excludes the VXST, the VIX/VX30 is clearly the best. This is the fastest reacting combination. The result is in agreement with previous working-papers (see [2] to [5]). It is nevertheless not completely obvious. A fast responding measure has an edge but the very beneficial median-5 filter dampens and delays the reaction too.

## The HeroRATs and the meltdown:

After finding reasonable thresholds for the last 3 years I tested the performance of (1d) to (1g) in the time period from 2008-01-03 till now. One can't perform the test for the VXST measures, because there are no data available.

Usually one does such a test the other way round. One optimizes the parameters in a first step for an in-sample time-period and tests the performance out-of sample for newer data. But I was primarily interested in the comparison with the new VXST and the performance of the median filter idea. So I have chosen the reverse back-testing approach.

The IVTS (1e) VIX/VX30 with the parameters given in table-1 dwarfs the SPY (yellow line). The final P&L is 234.1% to 43.1%. The max. relative drawdown is on 2008-10-27 with 17.4%. The SPY is on 2009-03-02 down with 51.3% (see Graphic-4).

| IVTS      | Low-Threshold | High-Threshold | P&L     | Drawdown |
|-----------|---------------|----------------|---------|----------|
| VIX/VXV   | 0.96          | 1.02           | +192.9% | 19.8%    |
| VIX/VX30  | 0.97          | 1.10           | +234.1% | 17.4%    |
| VIX/VX45  | 0.98          | 1.06           | +228.0% | 24.5%    |
| VX30/VX45 | 0.95          | 1.05           | +148.4% | 17.3%    |
| VBK       | –             | –              | +77.4%  | 55.6%    |
| SPY       | –             | –              | +43.1%  | 51.3%    |

Table-2: Thresholds and Performance from 2008-01-03 to 2013-12-11

The results in table-2 are in very good agreement with table-1. The ranks in the P&L and the drawdown columns are the same. The most slowly reacting VX30/VX45 has in both periods the lowest P&L but also the smallest drawdown. The fastest responding VIX/VX30 is also in the long run the best combination between P&L and drawdown. The HeroRATs have a very fine nose in the 2008-crash. Actually the real problem of an IVTS based strategy are the short swings in the recovery phase after the crash. But the median-5 filter does a good job in filtering the trembles of the invisible hand away.

![Screenshot of the 'Market-Timing Strategy' interface in a Firefox browser. The interface includes a control panel at the top with settings for SymbolLow (VBK), SymbolHigh (TLT), IVTS (VIX-Fut30), Filter (Med 5), LowThres (0.97), HighThres (1.10), LowWeights (100-0), MidWeights (50-50), HighWeights (0-100), and From (2008-01-03). The IVTS checkbox is checked and set to 'On'. Below the controls are four charts: 1) 'value' chart comparing VIX-Fut30 (orange area) and SPY (yellow line) from 2009 to 2013, showing VIX-Fut30 rising from 0% to ~200% and SPY rising from 0% to ~50%. 2) 'Drawdown' chart for VIX-Fut30 (teal area) showing peaks around 10%. 3) 'IVTS' chart for VIX-Fut30 (purple area) showing values between 0.5 and 1.0. 4) A partially visible chart at the bottom. The bottom right shows 'Select: VIX-Fut30' and 'Compare to: SPY'.](4086a572c080354982c11f1de4d6921d_img.jpg)

Screenshot of the 'Market-Timing Strategy' interface in a Firefox browser. The interface includes a control panel at the top with settings for SymbolLow (VBK), SymbolHigh (TLT), IVTS (VIX-Fut30), Filter (Med 5), LowThres (0.97), HighThres (1.10), LowWeights (100-0), MidWeights (50-50), HighWeights (0-100), and From (2008-01-03). The IVTS checkbox is checked and set to 'On'. Below the controls are four charts: 1) 'value' chart comparing VIX-Fut30 (orange area) and SPY (yellow line) from 2009 to 2013, showing VIX-Fut30 rising from 0% to ~200% and SPY rising from 0% to ~50%. 2) 'Drawdown' chart for VIX-Fut30 (teal area) showing peaks around 10%. 3) 'IVTS' chart for VIX-Fut30 (purple area) showing values between 0.5 and 1.0. 4) A partially visible chart at the bottom. The bottom right shows 'Select: VIX-Fut30' and 'Compare to: SPY'.

Graphic-4: VIX-VX30 with median-5 filter. 2008-01-03 to 2013-12-11

## Conclusion:

The VXST is an interesting new implied volatility measure. It seems to improve the results slightly. But the real boost is the median-5 filter. With the HeroRATs one gets the higher performance of the small-cap growth index for free. The strategy makes even nice wins in the 2008 and 2011 crashes. The strategy is easy to compute and generates minimal trading costs. It is under normal market conditions a Buy&Hold strategy. This is also the only drawback. If the market moves sideways or drags slowly down, the strategy does the same. The methodology is not restricted to the VBK/TLT pair. It can be applied to any trading strategy which depends on the S&P-500 regimes.

## Coming soon:

Apply the HeroRATs to the Mojito.

## References:

- [1] Maewal A., Scalaton J.: Paired-switching for tactical portfolio allocation, 2011-09-11.
- [2] Ch. Donninger: Improving the S&P Dynamic VIX-Futures Strategy: The Mojito 2.0 Strategy. Sibyl-Working-Paper, Rev. 1: 2013-06-04
- [3] Ch. Donninger: Boosting the SPY with the VXX: The Daiquiri Strategy. Sibyl-Working-Paper, April 2013
- [4] Ch. Donninger: A broad hint from the VIX: Timing the market with implied volatility, Sibyl-Working-Paper, April 2013.

- [5] Ch. Donninger: SPX-Options Trading: The Kir Strategy. Sibyl-Working-Paper, Rev. 2, 2013-03-08.
- [6] CBOE: CBOE Short-Term Volatility Index (VXST), October 2013
- [7] Kostakis, A., Panigirtzoglou, N., Skiadopoulos, G.: Asset Allocation with Option-Implied Distributions: A Forward-Looking Approach.
- [8] DeMiguel, V., Plyakha, Y., Uppal, R.: Improving Portfolio Selection Using Option-Implied Volatility and Skewness.
- [9] Stork M.: Median Filters Theory and Applications.