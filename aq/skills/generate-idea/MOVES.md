# Moves

Standard ways to generate a genuinely different variant when CAMPAIGN.md's axes do not suggest
an obvious next one.

**Apply one at a time.** Stacking three produces something that may score well and teaches
nothing, because no improvement is attributable to any of them.

**Change the timescale.** Usually the cheapest high-value move. Signal decay indicates the
direction: information concentrated at short horizons means costs are eating a real edge; a flat
decay curve means a longer hold gets the same edge for less turnover.

**Volatility-based weighting.** Inverse-vol or vol-target sizing instead of equal weight. Often
the largest single Sharpe improvement for one parameter. It tilts toward low-vol instruments,
which in crypto are the least liquid, so confirm the liquidity floor still binds; and a vol
target implying leverage above CONTEXT's cap in a quiet period is inadmissible, not merely
aggressive.

**Long-only / long-short / short-only.** Three strategies, not three settings. Report legs
separately in any case — alpha concentrated in one leg means it is a one-legged strategy plus
noise. Short exposure is admissible only via futures and perpetuals; short-only equities cannot
be tested.

**Stops.** Truncate the left tail but also cut winners that would have recovered, and make the
backtest path-dependent and execution-sensitive. They interact badly with mean reversion, firing
exactly when the signal is strongest. Worth testing because the effect is not predictable from
reasoning.

**Condition on regime.** No strategy works always, and knowing when yours works is worth more
than a marginal Sharpe gain. The regime must be classifiable from data available at the time —
trailing quantiles, not full-sample ones — and it adds parameters, so it is a new screen rather
than a free upgrade. A strategy that works only in a regime chosen after seeing results has been
fitted, not conditioned.

**Invert the signal.** If a screen produced a clearly negative *gross* Sharpe with consistent
sign across subperiods, the construction may be finding something real and pointing the wrong
way. Noise around zero does not qualify, costs are paid in both directions, and inversion is a
separate trial.

**Cross-sectional versus time-series.** Ranking instruments against each other and timing each
against its own history are different strategies from the same signal. Often the largest
available change, and easy to overlook because the signal code barely changes.

**Move the universe cut.** Liquidity floor, size band, exchange, listing age. Converts "the edge
is in names we cannot trade" from a dead end into either a real strategy or an honest kill.

**Transplant from refs/.** A mechanism documented in one universe applied to another. State
which reference and which specific result is being borrowed, and what about the new universe
might break the transplant.
