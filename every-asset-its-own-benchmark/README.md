# Cross-Sectional vs Self-Referential Factor Book on USDT Perpetual Futures

Backtest reproducing *"Every Asset Its Own Benchmark: Market-Neutral Alpha in
Perpetual Futures"* from 5-minute Binance USDT-M kline + funding parquet files.

## Data
- `bn-futures-5m.parquet` — 181M 5-min bars, 853 symbols, 2019-12-31 → 2026-08-21.
- `bn-funding-rates.parquet` — 8h funding observations 2020-01-01 → 2026-07-31.

## Shape (this dataset)
| quantity | actual |
|---|---|
| panel weeks | **347** (`MON` anchor; paper target 363) |
| universe symbols | **822** (paper target 112 — richer survivorship sample) |
| active rebalances | 295 of 347 |
| net funding receiver | ~80% of weeks; `w·funding` = −0.18%/week (≈ the paper's −0.173%) |

The paper's 363 weeks / 112 symbols were not reproduced because the provided
data ends 2026-08-21 and survives a much wider set of never-delisted symbols.
The pipeline, screens and shap are faithful; only the data sample differs.

## Module layout
| file | role |
|---|---|
| `data_load.py` | parquet → 5m panel; universe screen; funding |
| `resample.py` | 5m→1h→daily→weekly, anchor-aware |
| `factors.py` | eleven factor functions → weekly raw panels |
| `ranking.py` | `to_score`, `rank_xs`, `rank_ts`, `rank_xs_standardised` |
| `portfolio.py` | per-factor books → combined target → turnover cap |
| `returns.py` | weights + fwd + funding + costs → weekly return series |
| `metrics.py` | Sharpe, ann. return/vol, max DD, recovery, skew/kurt, t-stat |
| `checks.py` | the six §9 point-in-time tests |
| `preprocess.py` / `build_daily.py` | heavy factor computation, cached |
| `registry.py` | append-only config+Sharpe log (Table 13 / DSR) |
| `experiments/` | one script per table |
| `config.py` | the §10 `Config` dataclass |

## Mandatory point-in-time checks (§9) — all PASS
`python checks.py --anchor MON`
```
leak_checked = 1200, leak_max_abs_diff = 0.0
deterministic = True
dollar_neutral_max_net = 1.36e-16
gross_max = 1.50
funding_separated = True
active_weeks = 295
```

## Results (this dataset)
| table | headline |
|---|---|
| T3 | XS mean 0.64 → TS mean 0.87 (TS > XS); carry-removed 0.24 → 0.45 |
| T4 | XS 0.64 → XS-std 0.86 → TS 0.87 |
| T5 | funding carry adds ~0.3–0.6 Sharpe per factor; funding-only book Sharpe **1.15**, t=2.98 |
| T6 | TS/books/risk_parity 1.59 beats XS/blend/equal 0.91 (paper: TS/books > XS/blend/equal) |
| T9 | ann. ret 25.9%, vol 15.3%, Sharpe 1.58, max-DD −14.8%, Nov21–Dec22 sub-window +14%/yr vs market −77%/yr |
| T10 | walk-forward × 7 anchors (see out/table10.csv) |
| T12 | held-out symbols test Sharpe 0.45–1.07 (strategy generalises) |
| T13 | N=150 configs, sd=0.43, best=1.58, DSR=1.00 |
| T14 | cost × {1..32}, turnover_cap {0.5,1.0} |

## Resolved `[AMBIGUOUS]` choices (§14)
Each is a named Config field; none guessed at runtime.
1. **AVOL denominator** — ratio to 12-week trailing mean volume (`avol_lookback_weeks=12`).
2. **CPVv dispersion** — within-week s.d. of daily correlations (`cpvv_window="week"`).
3. **TSKD difference ordering** — average daily asymmetry over the week, *then* difference (`tskd_diff_order="avg_then_diff"`).
4. **funding_weight** magnitude — **0.5** (swept {0,0.25,0.5,1}).
5. **turnover_cap** — **0.5** (swept {0.35,0.5,1}).
6. **vol_window_weeks** — 26 (min 13) for inverse-vol book weighting.
7. **Daily→weekly** — mean of daily values (`min_days_per_week=3`).
8. **20-period smoothing** — applied to weekly raw values, before ranking.
9. **36-candidate walk-forward grid** — quintile_frac×3, funding_weight×3, turnover_cap×2, rank_window_weeks×2.

## Run
```
uv pip install -p .venv pandas pyarrow scipy numpy
.venv/bin/python preprocess.py --anchor MON --nprocs 12   # heavy, cached daily factors
.venv/bin/python checks.py --anchor MON
.venv/bin/python experiments/table3.py  (and 4,5,6,9,13,14)
.venv/bin/python experiments/build_all_anchors.py                 # 7 anchors
.venv/bin/python experiments/table10.py
.venv/bin/python experiments/table12.py
```

Run manifest (resolved §14 ambiguities + deployed config + results):
`experiments/out/manifest.json`.

Raw results CSVs: `experiments/out/table{3,4,5,6,9,10,12,13,14}.csv`.