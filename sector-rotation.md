# Skill: ETF Sector Rotation Strategy Development

## Goal

To develop a quantitative trading strategy that predicts the 1-month
forward relative returns of the 11 SPDR Select Sector ETFs (XLC, XLY, XLP, XLE,
XLF, XLV, XLI, XLB, XLRE, XLK, XLU). The strategy aims to identify
idiosyncratic sector strength by stripping out market beta and macro noise,
while prioritizing persistent, high-quality trends over volatile price shocks.

## 1. Feature Engineering & Data Requirements

### A. Price, Momentum & Autocorrelation (Micro)
*   **Idiosyncratic Momentum:** 12-month minus 1-month (12m-1m) cumulative
    residuals from a rolling 252-day regression of Sector Returns against SPY.
    Use **SPY** (Total Return) instead of SPX to account for dividends and
    market frictions.
*   **Momentum Smoothness (Frog-in-the-Pan):** The ratio of positive return
    days vs. negative return days over a 126-day window. High smoothness
    indicates higher trend persistence.
*   **First-Order Autocorrelation (AR1):** Rolling 60-day autocorrelation of
    daily returns to identify shifting regimes between mean-reversion and
    trending.
*   **Hurst Exponent:** A 126-day rolling Hurst calculation to filter for
    trending ($H > 0.5$) vs. mean-reverting ($H < 0.5$) sectors.
*   **Relative Strength:** The sector's trailing performance (RoC) divided by
    the SPY trailing performance.
*   **Distance from Moving Averages:** Percentage distance from the 20-day,
    50-day, and 200-day moving averages to capture overbought/oversold
    conditions.

### B. Orthogonalized Macro & Regime Features Features must be "pure" by using
the residuals of rolling regressions (126-day to 252-day windows) to isolate
sector-specific signals:
*   **Rate-Neutral Momentum:** Residuals of Sector Returns regressed against
    10-Year Treasury Yield changes. Extract **Rolling Betas ($\beta$)** as
    dynamic sensitivity scores.
*   **Commodity-Neutral Momentum:** (For XLE/XLB) Residuals regressed against
    WTI Crude Oil and Copper prices.
*   **Dollar-Neutral Performance:** Residuals regressed against the DXY Index.
*   **Macro-Neutral Valuation:** Sector Forward P/E residuals regressed against
    10Y Real Yields (TIPS) and High-Yield Credit Spreads (OAS). Also evaluate
    Relative Valuation Z-scores (Sector Fwd P/E vs its own 5-year rolling
    average).
*   **PMI Leading Indicator:** ISM Manufacturing PMI spread: **New Orders minus
    Inventories**.
*   **Copper-to-Gold Ratio:** Market-priced indicator of global economic
    health.
*   **Citi Economic Surprise Index (CESI):** Measures macro momentum vs
    expectations.
*   **VIX Term Structure:** Ratio of 1-month to 3-month VIX futures (Contango
    vs Backwardation).
*   **Cross-Asset Volatility:** The **MOVE Index** (bond volatility) as a
    leading indicator for equity regime shifts.

### C. Volatility & Risk (HAR Model)
*   **HAR Volatility Forecast:** Use a Heterogeneous Autoregressive (HAR) model
    to forecast 21-day forward realized volatility.
    *   *Inputs:* 1-day, 5-day (weekly), and 22-day (monthly) lagged realized
        volatility.
*   **Vol-Adjusted Momentum:** Divide raw or idiosyncratic momentum by the HAR
    volatility forecast to normalize signals across high-vol (XLK) and low-vol
    (XLP) sectors.

### D. Alternative Sentiment
*   **Idiosyncratic (Market-Neutral) Fund Flows:** Calculate net flows as
    $(Shares\_t - Shares_{t-1}) \times NAV_t$. Use rolling residuals of ETF
    flows regressed against Sector Returns to isolate "unexpected"
    institutional accumulation from price-driven passive flows.
*   **Earnings Revision Breadth:** Ratio of analyst upward EPS revisions to
    downward revisions for sector constituents.
*   **Options Skew:** The 30-day 25-delta Put/Call IV skew for each ETF to
    measure downside hedging demand.
*   **Market Breadth:** Percentage of S&P 500 stocks trading above their 50-day
    and 200-day moving averages.

## 2. Evaluation Framework (Single-Factor Alpha Research)

Before model training, every feature must be evaluated using the following
metrics:
*   **Information Coefficient (IC):** Spearman Rank Correlation between the
    feature rank and the 1-month forward return rank, calculated daily and
    averaged.
*   **HAC (Newey-West) Adjustment:** Use HAC-consistent standard errors when
    evaluating feature significance to handle heteroskedasticity and
    autocorrelation.
*   **Quantile Spread:** The 1-month forward return of the Top 3 sectors minus
    the Bottom 3 sectors when sorted by the feature. Look for monotonic decay
    across quantiles.
*   **Hit Rate:** The percentage of time the feature correctly predicts a
    sector outperforming the median sector return.
*   **Regime-Conditioned IC:** Evaluation of feature performance during
    different macro environments (e.g., ISM PMI > 50 vs. < 50).
*   **Feature Autocorrelation:** Check for stability to avoid excessive
    portfolio turnover and transaction costs.

## 3. Implementation Status & TODO List

### Core Price & Momentum Features
- [x] Raw Time-Series Momentum (1m, 3m, 6m, 12m)
- [x] SMA Distance (20d, 50d, 200d)
- [ ] **Skip-Month Momentum (12m-1m)**: To avoid short-term reversal effects.
- [ ] **Momentum Smoothness (Frog-in-the-Pan)**: Ratio of pos/neg days (126d).
- [ ] **Rolling AR(1) of Daily Returns**: Identify trend persistence regimes.
- [ ] **Hurst Exponent**: Rolling 126d window to filter for trending vs.
  mean-reverting.

### Orthogonalization (Residuals)
- [x] Idiosyncratic Momentum vs. SPY (Single-regressor OLS)
- [x] Rate Sensitivity (Single-regressor Beta)
- [x] FX Sensitivity (DXY Beta)
- [ ] **Multi-Factor Orthogonalization**: Update `rolling_ols` to handle
  multiple regressors (e.g., SPY + Rates + Oil simultaneously).
- [ ] **Commodity-Neutral Momentum**: Specifically for XLE and XLB against
  WTI/Copper.
- [ ] **Pure Value**: Residuals of Forward P/E vs. TIPS and Credit Spreads.

### Volatility & Risk
- [x] 20-day Realized Variance
- [ ] **HAR Volatility Forecast**: 1d/5d/22d OLS for 1-month forward vol
  prediction.
- [ ] **Vol-Adjusted Features**: Scale momentum/alpha by HAR-forecasted
  volatility.

### Macro & Bottom-Up Features
- [x] Yield Curve (10Y-2Y)
- [x] Credit Spreads (HY OAS)
- [x] Real Rates (10Y TIPS)
- [x] Economic Indicators (Non-farm Payrolls, Housing Starts)
- [x] Sector Breadth (Aggregated from constituents)
- [x] Sector ROC (Aggregated from constituents)
- [ ] **PMI New Orders vs. Inventories**: Leading indicator for cyclical
  sectors.
- [ ] **Copper-to-Gold Ratio**: Proxy for global growth/risk-on.
- [ ] **VIX Term Structure**: 1m/3m futures ratio.
- [ ] **MOVE Index**: Bond volatility regime filter.
- [ ] **CESI**: Citi Economic Surprise Index.

### Sentiment & Flows
- [ ] **Idiosyncratic Fund Flows**: Residuals of ETF flows vs. Sector returns.
- [ ] **Earnings Revision Breadth**: Up/Down revision ratio.
- [ ] **Options Skew**: 25-delta Put/Call IV skew.
- [x] **Market Breadth**: % of stocks above 50d/200d SMA (implemented via
  constituent aggregation).

## 4. Implementation Guidelines
*   **Source Code & Notebook:** The implementation is located in
    `sector-rotation.py`. This is a paired file with a Jupyter notebook
    (`sector-rotation.ipynb`) via **Jupytext**.
*   **Workflow:** Use `uv run sector-rotation.py` to execute the script and `uv
    run jupytext --sync sector-rotation.py` to keep the paired `.py` and
    `.ipynb` files in sync. Any updates to the `.py` script must be synced to
    the notebook and vice-versa.
*   **Point-in-Time Integrity:** All macro data (PMI, CPI, Yields) must be
    lagged to their actual release dates to avoid look-ahead bias.
*   **Computation:** Use `statsmodels.regression.rolling.RollingOLS` or
    matrix-based NumPy operations for efficient rolling regressions across the
    11-sector universe.
*   **Position Sizing:** Use Inverse-Volatility weighting (based on HAR
    forecasts) to ensure equal risk contribution across selected sectors.
