# Remediation pass — correctness fixes against PLAN.md

This pass fixed defects, only. No strategy design change, no new factors, no
tuning. Every change is against the `PLAN.md` spec. All modules fixed in place;
no parallel scripts/experiment folders created except the regression tests and
a `phase6.py` driver.

> **Path note:** the repo was `git mv`'d from `bbb/` to this directory, which
> broke every hardcoded `/home/kai/studies/bbb/...` path. Added `paths.py`
> (repo-root-relative cache/registry/out) and rewired all modules. Also the raw
> data parquets moved to `/home/kai/node/data/studies/prev/` — updated
> `data_load.py`.

## Defects fixed (by phase)

**Phase 0 — preconditions**
- 0.1 Added `tests/test_config_fields_used.py`; `tskd_diff_order` and
  `cpvv_window` were declared-but-never-read. Both are now wired (see 1.2, 1.3).
- 0.2 Daily cache now keyed by `Config.factor_cache_key()` (`q_top_frac`,
  `tskd_min_bars_per_side`, `tskd_diff_order`) in `build_daily.py`;
  per-anchor `factor_*.parquet` guarded by `load_panels(anchor, cfg)`
  manifest verification. Deleted stale caches before rebuilding.

**Phase 1 — factors (required cache rebuild)**
- 1.1 AVOL reverted to **ratio-to-trailing-mean** (the `-log Σ12w` was negative
  log of a size/level measure, not abnormal turnover). Test: constant volume ⇒
  AVOL=0 past warm-up.
- 1.2 TSKD daily column renamed `TSKD_level`; the **Δ is now applied at the
  weekly stage** per `tskd_diff_order` (`avg_then_diff` default). Test: first
  valid week per symbol is NaN; constant asymmetry ⇒ 0.
- 1.3 `cpvv_window` wired: `"week"` (within-week std) and `"trailing_20d"`
  (rolling-20d std sampled at week end); day threshold `cpvv_min_days_week`.
- 1.4 `aggregate_daily_to_weekly` raises on `±inf` input.

**Phase 2 — delisting returns**
- 2.1 `build_weekly(..., book_terminal_return=True)` books each symbol's final
  observed return to the settlement mark (config `book_terminal_return`).
  Binance settles USDT-M perps at final mark; no forced −100%.
- 2.2 `require_continuous_trading` **default flipped to `False`** (PIT universe);
  the `True` branch now compares against the panel's own index slice (no
  generated 7D range that fails on a missing panel week).
- 2.3 manifest surfaces `n_symbols_by_week` {min, median, max}; warns on weeks
  below `min_cross_section`.

**Phase 3 — checks that can fail**
- 3.1 determinism now compares the **parallel daily-factor build** single-vs-multi
  process (run_pipeline alone is pure/vacuous).
- 3.2 funding separation is behavioural: perturbs one weighted symbol's forward
  funding, asserts the return changes **in the week-t-1** (off-by-one).
- 3.3 non-emptiness reports `evaluable_weeks` (≥1 factor scored) and
  `active_weeks`, not a vacuous panel-length count.
- 3.4 added end-to-end leak test truncating at the **5m** level.
- 3.5 DSR inputs assert per-period Sharpe.

**Phase 4 — infra**
- 4.1 registry append-only via `fcntl` flock + `os.fsync`, logs **weekly**
  Sharpe, `context` column. Old annualised registry archived to
  `registry_old_annualised.csv.bak`.
- 4.2 DSR units fixed; Table 13 now asserts `|sr_weekly| < 1`.
- 4.3 walk-forward (10 folds × 12 candidates) and ablation (12) trials logged
  with `context`; DSR reported under two trial sets.

**Phase 5 — minor**
- trimmed unused `rank_xs(min_cross_section)`, removed the index-uniqueness
  drop line, unified symbol-column dtype conversion in `resample`, deleted
  dead `r5`/`hS`/`vi`/_process_chunk code, cfg passed through chunk tuple in
  `build_daily` (works under spawn), `__week` dropped from the groupby count.

## Baselines (tracking how each defect moved the result)

| config | Sharpe | ann. ret | max DD | note |
|---|---|---|---|---|
| deployed baseline (before fixes, old code) | 1.584 | 25.9% | −14.8% | erroneous AVOL/TSKD, survivorship screen |
| **Phase 1** old universe (822), clip on | **1.358** | +22.2% | −15.3% | AVOL+TSKD fixed, survivorship on |
| **Phase 2** PIT universe (853), clip on | **1.305** | +21.8% | −15.7% | + terminal returns |
| Phase 1 old universe, clip off | 1.100 | +19.3% | −27.5% | |
| Phase 2 PIT universe, clip off | 1.052 | +18.5% | −27.8% | |

The old 1.584 was inflated by the mis-specified factors (AVOL as a size/level
sort under XS, TSKD level instead of change) and the survivorship screen.

## Checks

```
checks.py --anchor MON           => ALL CHECKS PASSED
  leak_checked=1200  leak_max_abs_diff=0.0
  determinism_max_abs_diff=0.0
  dollar_neutral_max_net=1.3e-16  gross_max=1.62
  funding_separated=True
  evaluable_weeks=299  active_weeks=296
```

End-to-end leak on the 5m truncation: with `require_continuous_trading=True`
it **fails** (the screen is the lookahead) — that is the point of the test. With
`False` the weights are bit-identical apart from an out-of-scope relisted-funding
artifact (35 symbols whose funding predates their price coverage; BNT is the
canonical case where funding ranks differ by one).

## Table re-runs (corrected PIT panels)

- T3–T6, T9 MON: written to `experiments/out/`
- T10 walk-forward × anchor: MON 1.391, TUE 1.241, WED 0.871, THU 0.762,
  FRI 1.001, SAT 0.603, SUN 0.878
- T12 held-out universe: test Sharpe 0.32–1.58
- T13 DSR (units fix + both trial sets): strategy-search DSR **0.719**,
  all-cells **0.634** (was 1.0000)
- T14 cost robustness: cap 0.5 cost 1 Sharpe 1.305 → 0.109 at cost 32
- full_no_cont_screen: Sharpe **1.305** (no screen, clip on)
- ablation (leave-one-out, no clip): AVOL now **beneficial** to drop
  (remove_AVOL → 1.19 vs baseline 1.05); CPVv the strongest signal carrier;
  WRspread/TopChg/Quad inert (data unavailable)
- walk-forward ablation: selection churns (3/9), AVOL picked 7/10, but selected
  OOS 0.704 does not beat all-11 0.796 → **keep all eleven**

## Regression tests (all pass)
- `tests/test_config_fields_used.py`
- `tests/test_factors_phase1.py`
- `tests/test_phase2_3.py`

## Out of scope (deferred, per plan)
Symbol renames (MATIC→POL etc.), index-product drops, positioning factors
(WRspread/TopChg/Quad — data not present), any factor selection/tuning,
alternative book combination. Relisted-funding coverage (35 symbols, e.g. BNT)
is a data-coverage artifact, not corrected here.