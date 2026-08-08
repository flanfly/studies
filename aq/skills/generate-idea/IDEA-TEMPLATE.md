# IDEA.md structure

Four sections plus frontmatter. Prose, not a form — but the Strategy section must be exact
enough to execute literally, because the implementer has this file and `CONTEXT.md` and nothing
else.

```markdown
# <short descriptive title>

## What we are doing
Three to six sentences. Instruments and venue, the mechanism being exploited, and what the
strategy is built to capture. Then who is on the other side and why they accept the loss, or
"no counterparty identified" and what that implies.

Describe the construction, not its prospects. "Ranks perps by 30-day return and holds the top
decile" belongs here; "should earn 8-12% annualised" does not.

## How this fits the campaign
What earlier ideas established — measured results only — and what this one changes relative to
them. If building on a parent, name it and state what is inherited unchanged versus what
differs. On the first idea of a campaign, what CAMPAIGN.md and refs/ point to as a starting
point.

State the change, not the reason you think it will help. "Holding period moves from 5 days to
30" is the delta; "which should reduce cost drag" is a prediction and does not belong anywhere
in this file.

## Strategy
The bulk of the document. Write it as an algorithm, in order. If a step could reasonably be
implemented two ways, it is underspecified.

- Universe: which instruments, from which dataset, what screens, reconstructed point-in-time at
  each rebalance
- Data fields used, and behaviour when one is missing or stale
- Signal: every transformation in sequence, with formulas. Lookbacks, warmup length,
  standardisation, winsorisation, ranking rule, tie-breaking, sign convention
- Positions: sizing, weighting, leverage rule and how it respects CONTEXT's caps, entry and exit
  conditions, maximum position count
- Timing: what is observable when, and the lag between signal and fill
- Rebalance frequency, and behaviour when the universe changes between rebalances
- Costs and frictions specific to this variant. For perpetuals, require funding to be reported
  as a separate line item
- Edge cases: insufficient history, delisting mid-position, zero-volume days, ties at a
  threshold

Then, before anything can run:

- **Screen configuration** — the single set of parameter values to run first, each with one line
  on where the value comes from: a reference, a convention, or stated reasoning. Not a claim
  about how it will perform
- **Screen gate** — the numeric criterion for proceeding to optimization, declared now. Unless
  argued otherwise: net Sharpe > 0 on train, and gross Sharpe > 0.5 for the strategy as
  specified rather than its inverse. Also state what result would mean abandoning the
  construction rather than tuning it
- **Optimization grid** — parameters, ranges, ≤ 24 configurations total, executed only if the
  gate passes. Justify each parameter; unjustified ones stay fixed with no range

## Data required
Every input the strategy needs, stated without reference to what is available. Do not trim the
hypothesis to fit the datasets in CONTEXT, and do not silently substitute a proxy for something
assumed missing — that filtering is invisible once done. If the right test needs an implied-vol
surface we do not have, say so and write the idea anyway.

Per input: the series, instruments covered, frequency, history depth including warmup, and
whether it must be point-in-time or as-of vintage. Typical inputs are OHLCV, funding rates, open
interest, implied volatility surfaces or ATM IV, realised-vol estimates, term structures and
curves, market cap, on-chain measures, fundamentals, sentiment, macro series.

Split into two groups, because it decides whether a screen can run at all:

- **Required** — without it there is no strategy
- **Optional** — sharpens the signal, but the screen is meaningful without it

Name acceptable substitutes where the hypothesis tolerates one, and say where it does not.

## Diagnostics to report
Measurements this variant needs beyond CONTEXT's defaults, listed neutrally — what to compute,
not what it will show. Anything whose value would distinguish one explanation of the result from
another: per-leg contribution, breakdown by liquidity decile, PnL by volatility quartile,
funding as a share of gross return, decay by holding period.

Ask for what you would need to interpret any outcome, not what you need to confirm one.

## Directions set aside
Other directions considered and deprioritised on expectation rather than evidence, one line
each. Omit the heading if there were none.
```
