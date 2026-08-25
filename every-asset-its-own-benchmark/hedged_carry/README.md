# Hedged Carry Portfolio

Short-only, spot-hedged carry book derived from the deployed factor strategy
(sibling experiment to the factor book in the repo root — see
[`../README.md`](../README.md)).

## Concept

The combined factor book (TS ranking / per-factor books / risk-parity, MON
anchor) is dollar-neutral: a long leg and a short leg. Perpetual funding is
paid long→short on every position, so *shorts* collect funding in positive
funding regimes. This experiment keeps **only the short leg**, hedges it
1-for-1 with spot (killing the price PnL), and counts **only the funding
carry** on the short futures positions.

Because every unit of short notional requires an equal unit of spot
collateral, the return per unit of deployed capital is **half** the raw carry:

```
carry_gross = -(w_short * funding_shift).sum(axis=1)   # what the shorts collect
r_hedged    = 0.5 * carry_gross                        # 1:1 spot hedge
```

Price PnL is fully hedged and excluded; no trading costs are modelled (per
spec: "only the carry we collect is counted towards returns").

## Overall results (2019-12-30 → 2026-08-17, 347 weeks, MON anchor)

| metric | value |
|---|---|
| total return (compounded) | **+27.4%** |
| annualized return | **+3.69%** |
| annualized vol | 1.12% |
| Sharpe (ann.) | **3.25** (t = 8.39) |
| max drawdown | **−0.37%** (9 weeks to recover) |
| active weeks | 295 / 347 |
| positive weeks | 76.1% |
| mean weekly return | +0.070% |
| mean short notional | 0.49 × book capital |
| mean weekly carry (gross) | +0.14% |

## Yearly breakdown (hedged carry return after 1:1 hedge)

| year | weeks | return | ann. return | ann. vol | Sharpe | max DD | mean wk | pos wks | mkt funding avg |
|---|---|---|---|---|---|---|---|---|---|
| 2019 | 1 | 0.00% | 0.00% | – | – | 0.00% | 0.00% | 0.0% | +0.06% |
| 2020 | 52 | +0.65% | +0.65% | 0.54% | 1.21 | 0.00% | +0.01% | 3.8% | +0.15% |
| 2021 | 52 | **+13.62%** | +13.62% | 2.30% | 5.57 | −0.37% | +0.25% | 96.2% | +0.60% |
| 2022 | 52 | +0.79% | +0.79% | 0.28% | 2.78 | −0.20% | +0.02% | 73.1% | −0.12% |
| 2023 | 52 | +2.78% | +2.78% | 0.24% | 11.44 | 0.00% | +0.05% | 100.0% | +0.10% |
| 2024 | 53 | +4.71% | +4.62% | 0.71% | 6.35 | −0.03% | +0.09% | 98.1% | +0.19% |
| 2025 | 52 | +1.74% | +1.74% | 0.24% | 7.25 | −0.08% | +0.03% | 90.4% | −0.26% |
| 2026* | 33 | +0.93% | +1.47% | 0.26% | 5.63 | −0.03% | +0.03% | 69.7% | −0.19% |

\* 2026 through 2026-08-17.

## Reading

- **2021 dominates** (+13.6%): the alts melt-up regime with strongly positive
  funding paid long→short; shorts collected in ~96% of weeks.
- **2022** (bear, negative funding): carry collapses to +0.8% — when funding
  flips negative the shorts pay instead of collect, and the leg just idles.
- **2023–2025**: steady small positive carry (+1.7% to +4.7%/yr) even though
  the *market-average* funding was negative in 2025 (−0.26%/wk avg): the
  factor book's short leg selects high-funding (crowded-long) names, so the
  selected shorts still get paid while the broad market pays.
- Very low vol (0.24–2.3%) because price risk is hedged; the roll is smooth
  carry. Max drawdown of the whole backtest is just −0.37% (2021-05 correction).
- Cost note: spot/futures transaction costs and margin funding costs on the
  hedge are **not** modelled — 0.5%–3%/yr of the gross carry is a plausible
  cost band, so the 2022 and 2025 years are realistically near-breakeven after
  costs while 2021/2023/2024 tolerate them.

## Files

```
backtest.py                    <- the whole experiment (one file)
out/summary.csv                <- overall stats
out/yearly.csv                 <- yearly breakdown
out/weekly_returns.csv         <- weekly short gross, carry gross, hedged return
```

## Run

```
.venv/bin/python hedged_carry/backtest.py
```