Implementation complete — summary

 All plan sections implemented and verified.

 ### Data

 - bn-futures-5m.parquet — 181M 5-min USDT-M klines, 2019-12-31 → 2026-08-21, 853 symbols → 822 trade through
 - bn-funding-rates.parquet — 8h funding rate obs, 833 symbols.

 ### Pipeline

 data_load.py → resample.py → factors.py → ranking.py → portfolio.py → returns.py → metrics.py → checks.py, orchestrated by pipeline.py. Heavy per-symbol daily factor
 panels are computed once and cached (anchor-independent) by build_daily.py; weekly (re)aggregation is anchor-aware.

 ### Deployed baseline (MON anchor) output

 ┌──────────────────────┬──────────────────────────────────────────────┐
 │ metric               │ value                                        │
 ├──────────────────────┼──────────────────────────────────────────────┤
 │ annualised Sharpe    │ 1.584                                        │
 ├──────────────────────┼──────────────────────────────────────────────┤
 │ annualised return    │ 0.259 (25.9%)                                │
 ├──────────────────────┼──────────────────────────────────────────────┤
 │ annualised vol       │ 0.153 (15.3%)                                │
 ├──────────────────────┼──────────────────────────────────────────────┤
 │ max drawdown         │ −0.148 (−14.8%)                              │
 ├──────────────────────┼──────────────────────────────────────────────┤
 │ mean weekly turnover │ 0.426                                        │
 ├──────────────────────┼──────────────────────────────────────────────┤
 │ gross exposure       │ 1.501 (≤2)                                   │
 ├──────────────────────┼──────────────────────────────────────────────┤
 │ net exposure         │ 1.6e-16 (dollar-neutral)                     │
 ├──────────────────────┼──────────────────────────────────────────────┤
 │ funding receipt      │ −0.181%/week (80% of weeks; ≈ paper −0.173%) │
 ├──────────────────────┼──────────────────────────────────────────────┤
 │ active rebalances    │ 295 of 347 weeks                             │
 └──────────────────────┴──────────────────────────────────────────────┘

 ### Mandatory tests (§9) — all pass

 - Leak test (1200 samples, max diff 0.0)
 - Deterministic weight
 - Dollar-neutral (max 1.57e-16)
 - Gross bound ≤ 2 (max 1.50)
 - Funding separation (score uses week-t; returns use week-t+1)
 - Non-emptiness (built 347, active 295)

 ### Experiments reproduced (methodology faithfully; numbers differ because of data)

 - T3 ranking: XS 0.64 → XS-std 0.86 → TS 0.87 (TS>XS); carry-removed 0.24→0.45
 - T4 XS 0.64 → XS-std 0.86 → TS 0.87
 - T5 carry adds ~+0.3–0.6 Sharpe/asset; funding-only book Sharpe 1.15, t=2.98 (paper 1.25, t=2.91)
 - T6 TS/books/risk-parity 1.59 > XS/blend/equal 0.91 (qualitative TS/books > XS/blend/equal)
 - T9 ann 25.9%, vol 15.3%, Sharpe 1.58, max −14.8%; N21–D22 +14.1% yr vs mkt −77.3% yr
 - **T10 walk-forward × 7 anchors (MON..SUN) — all positive out-of-sample (0.62–1.58)
 - T12 held-out symbols: test Sharpe 0.45–1.07 — generalises
 - T13 registry: N=150, sd=0.43, best=1.58, DSR=1.00
 - T14 costs ×{1,2,4,8,16,32}, caps {0.5,1.0}

 ### §14 ambiguities resolved (all named configs; none guessed at runtime)

 ┌─────────────────┬────────────────────────────────────────────────────────────────────────────────┐
 │ #               │ ambiguous resolved to                                                          │
 ├─────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ AVOL denom      │ 12-week trailing mean volume                                                   │
 ├─────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ CPVv dispersion │ within-week stdev of daily CP (cpvv_window="week")                             │
 ├─────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ TSKD order      │ weekly avg daily asymmetry then diff (tskd_diff_order="avg_then_diff")         │
 ├─────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ funding weight  │ 0.5                                                                            │
 ├─────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ turnover cap    │ 0.5                                                                            │
 ├─────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ vol_window      │ 26 weeks                                                                       │
 ├─────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ daily→weekly    │ mean (min 3 d/w)                                                               │
 ├─────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ smoothing       │ 20-week trailing mean on weekly raw panels, pre-rank                           │
 ├─────────────────┼────────────────────────────────────────────────────────────────────────────────┤
 │ WF grid         │ quintile_frac×3 × funding_weight×3 × turnover_cap×2 × rank_window_weeks×2 = 36 │
 └─────────────────┴────────────────────────────────────────────────────────────────────────────────┘

 ### Checks & logs

 - manifest (all resolved ambiguities, config and results): experiments/out/manifest.json
 - per-table CSVs: experiments/out/table*.csv

 Files: checks.py, registry.py, experiments/util.py, plus experiments/table{3,4,5,6,9,10,12,13,14}.py.

 Run instructions are in the README.
