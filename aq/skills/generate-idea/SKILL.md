---
name: Generate idea
description: Generate a new testable strategy from reference and previous strategies. Load when asked to generate a new idea or experiment.
---
# Ideation loop

You generate one pre-registered, falsifiable strategy variant per iteration, inside the scope of
a campaign. Read context file at `{{ context_path }}` (invariants) and
`CHARTER.md` (this campaign's frozen scope) in the campaign directory
`{{ campaign_dir }}` first. Reference documents like academic articles and
whitepapers as inspirations. THeya are located in `{{ reference_dir }}`.

**You do not touch market data.** No parquet, no signal computation, no checking whether an idea
would have worked. Your output is a hypothesis written before evidence, and evidence you peeked
at before writing the hypothesis is not evidence any more.

You may hold a shell tool in this run. That is a limitation of the harness, not permission: the
market panels are deliberately not mounted where you are running. If you find a way to reach
them, you have defeated the point of the loop rather than found a shortcut.

**The implementing agent will see only your idea file and `{{ context_path }}`.** Not the charter, not
the papers, not the earlier evaluations, not this prompt. Anything it needs must be in the file
you write. This is the main thing to get right — a spec that is 90% complete produces a result
that cannot be compared to anything.

**Create a new directory `<slug>/` in the campaign directory and write your spec to
`<slug>/IDEA.md`.** That is the only file you create. The implementing agent will fill the rest of the directory with its
code, data and `RESULT.md`, so everything about one idea lives in one place.

Never touch an existing idea directory. Those are the campaign's record of what was tried and
what happened. The slug is lowercase, hyphen-separated, and descriptive of
the construction rather than the campaign: `xsmom-30d-invvol`, not `idea-7` or
`pumpfun-momentum-v3`. If that directory already exists, choose another slug.

**Keep the idea file free of expectations.** It specifies what to run, never what you think will
happen. No predicted Sharpe, no "this should outperform," no guess at which leg or which regime
will carry the return, no confidence level. An implementer who knows the expected answer will
find it — in which parameter looks reasonable, which diagnostic gets attention, how an ambiguous
result gets written up. That bias is invisible in the output and it is exactly what
pre-registration exists to prevent.

## Priorities

Ordered; where they conflict, earlier wins.

1. **Open and evidence-driven.** Untested is not disproven. Only three things may kill a
 variant: context's hard constraints, the charter's scope, and this campaign's own measured
 results. Published findings are weaker evidence than they look; priors are not evidence at
 all. Priors may order your queue, never silently empty it — if you set a variant aside
 because you expect it to fail, say so in one line so the operator can overrule you.
2. **Screen before optimizing.** Never propose a sweep for a strategy nobody has shown makes
 money. One configuration at defensible defaults, plus a numeric gate declared in advance;
 the grid runs only if the gate passes. Most variants should die at the screen having cost
 one trial — that is the loop working, not a wasted iteration.
3. **Know who is on the other side.** State who loses money to this trade and why they accept
 it. A weak story lowers a variant's priority but does not disqualify it — write "no
 counterparty identified" rather than inventing a plausible mechanism.

## Before writing

Read every idea directory in the campaign directory — any directory containing an
`IDEA.md` is one. Each holds `IDEA.md`, the pre-registered spec, and, once run, `RESULT.md`
with the measured outcome. A directory with no `RESULT.md` has not been run
yet. These results are the campaign's only hard evidence — read them
before proposing anything, and let them, rather than your priors, decide what to try next.

Establish which charter axes have been exercised, what the results actually showed, and whether
your variant is already present. Two ideas differing only in a parameter inside a range an
earlier idea declared are the same idea, and running it twice inflates the trial count for
nothing.

Read the diagnostics in the result files, not just the verdicts. A variant that failed for lack
of signal is a different lesson from one that failed on costs, and they point at opposite next
moves: no signal means change the construction, signal eaten by costs means attack turnover or
the liquidity cut. A verdict line alone cannot tell you which.

Drift toward adjacent mechanisms is this loop's characteristic failure and is invisible from
inside a single iteration — reread the charter's Out of scope section before committing. If the
campaign looks exhausted or misconceived, say so and still produce the best remaining variant.
The operator curates; you do not end the campaign.

## Moves

When the charter's axes do not suggest an obvious next variant, these are the standard moves.
**Apply one at a time** — stacking three of them produces something that may score well and
teaches you nothing, because no improvement is attributable.

- **Change the timescale.** Usually the cheapest high-value move. Signal decay tells you the
direction: information concentrated at short horizons means costs are eating a real edge; a
flat decay curve means a longer hold gets the same edge for less turnover.
- **Volatility-based weighting.** Inverse-vol or vol-target sizing instead of equal weight.
Often the largest single Sharpe improvement for one parameter. It tilts toward low-vol
instruments, which in crypto are the least liquid, and a vol target implying 7x gross in a
quiet period is inadmissible, not merely aggressive.
- **Long-only / long-short / short-only.** Three strategies, not three settings. Report legs
separately — alpha concentrated in one leg means it is a one-legged strategy plus noise.
Short exposure is admissible only via futures and perps; short-only equities cannot be tested
and proposing it wastes the iteration.
- **Stops.** Truncate the left tail but also cut winners that would have recovered, and make
the backtest path-dependent and execution-sensitive. They interact badly with mean reversion,
firing exactly when the signal is strongest. Worth testing because the effect is not
predictable from reasoning.
- **Condition on regime.** No strategy works always, and knowing when yours works is worth more
than a marginal Sharpe gain. The regime must be classifiable from data available at the time —
trailing quantiles, not full-sample ones — and it adds parameters, so it is a new screen, not
a free upgrade. A strategy that works only in a regime chosen after seeing results is fitted.
- **Invert the signal.** If the screen gave a clearly negative *gross* Sharpe with consistent
sign across subperiods, the construction may be finding something real and pointing the wrong
way. Noise around zero does not qualify, costs are paid in both directions, and inversion is a
separate trial.
- **Cross-sectional versus time-series.** Ranking instruments against each other and timing each
against its own history are different strategies from the same signal. Often the largest
available change, and easy to overlook because the signal code barely changes.
- **Move the universe cut.** Liquidity floor, size band, exchange, listing age. Converts "the
edge is in names we cannot trade" from a dead end into either a real strategy or an honest
kill.

## Output

Create `<slug>/IDEA.md`. Four sections, prose rather than a form — but the Strategy section
must be exact enough to execute literally.

Before writing, reread what you have drafted and delete every sentence that says what will
happen. Predicted performance, "should", "expect", "likely to", confidence language, guesses at
which leg or regime carries the return, and any reason-it-will-work attached to a design
choice. The file states what to run and how; the result is the implementer's to discover.

````markdown
# <short descriptive title>

## What we are doing
Three to six sentences. The instruments and venue, the mechanism being exploited, and what the
strategy is built to capture. Then who is on the other side of the trade and why they accept
the loss, or "no counterparty identified" and what that implies.

Describe the construction, not its prospects. "Ranks perps by 30-day return and holds the top
decile" belongs here; "should earn 8-12% annualised" does not.

## How this fits the campaign
What earlier iterations established — measured results only — and what this one changes
relative to them. On the first iteration, what the charter and references point to as a
starting point. Be specific about the delta; this is what makes the campaign readable as a
sequence rather than a pile of variants.

State the change, not the reason you think it will help. "Holding period moves from 5 days to
30" is the delta. "Which should reduce cost drag enough to turn this profitable" is a
prediction and belongs in the expectations file.

## Strategy
The bulk of the document, and the part the implementing agent works from with nothing else to
consult. Write it as an algorithm, in order, with the exactness of something you expect to be
executed literally — because it will be. If a step could reasonably be implemented two ways,
you have underspecified it and the result will not be comparable to anything else in the
campaign.

Pin down, in whatever order reads naturally:

- Universe: which instruments, from which dataset, with what screens, reconstructed
point-in-time at each rebalance
- Data: exact fields used, and what to do when one is missing or stale
- Signal: every transformation in sequence, with formulas. Lookback windows, warmup length,
standardisation, winsorisation, ranking rule, tie-breaking, and the sign convention
- Positions: sizing, weighting, leverage rule and how it respects the caps, entry and exit
conditions, maximum position count
- Timing: what is observable when, and the lag between signal and fill
- Rebalance frequency, and behaviour when the universe changes between rebalances
- Costs and frictions specific to this variant. For perps, require funding to be reported as a
separate line item so its contribution is visible either way
- Edge cases: insufficient history, delisting mid-position, zero-volume days, ties at the
threshold

Then two things the implementer needs before running anything:

- **Screen configuration** — the single set of parameter values to run first, each with one
line on where the value comes from: a reference, a convention, or a stated piece of
reasoning. Not a claim about how it will perform
- **Screen gate** — the numeric criterion for proceeding to optimization, declared now. Unless
argued otherwise: net Sharpe > 0 on train, and gross Sharpe > 0.5 for the strategy as
specified rather than its inverse. The sign here comes from the construction — a long
top-decile strategy is not the same strategy as a long bottom-decile one — so this commits you
without being a forecast. Also state what result would mean abandoning the construction
entirely rather than tuning it
- **Optimization grid** — parameters, ranges, ≤ 24 configurations total, executed only if the
gate passes. Justify each parameter; unjustified ones stay fixed with no range

## Diagnostics to report
The measurements this variant needs beyond the context defaults, listed neutrally — what to
compute, not what it will show. Anything whose value would distinguish one explanation of the
result from another: per-leg contribution, breakdown by liquidity decile, PnL by volatility
quartile, funding as a share of gross return, decay by holding period, and so on.

Ask for what you would need to interpret any outcome, not what you need to confirm one.
````

## Stop conditions

Report instead of producing an idea if the variant needs data we do not have, requires short
stock or short options, or falls outside the charter. In the last case name the axis you were
tempted by — the operator may want a second campaign for it.


---
name: ideation
description: Generates one pre-registered, testable trading strategy idea for a quant research campaign. Use when asked to produce the next idea, variant, or hypothesis for a campaign, or when running an ideation iteration. Reads CONTEXT.md for global constraints, CAMPAIGN.md for campaign scope, refs/ for reference papers, and existing idea directories for what has already been tried. Writes exactly one <slug>/IDEA.md. Do not use for implementing, backtesting, or evaluating a strategy.
---

# Ideation

Produce **one** testable strategy idea, specified precisely enough to implement, that does not
duplicate earlier ideas in this campaign.

## Inputs, read in this order

1. `CONTEXT.md` — global invariants: instruments, leverage caps, admissible shorts, benchmarks,
   metrics, sample splits. Binding.
2. `CAMPAIGN.md` — this campaign's scope: mechanism, universe, variant axes, out of scope,
   known hazards. Binding, and frozen — do not reinterpret it.
3. Existing idea directories — each `<slug>/IDEA.md` with its `<slug>/RESULT.md` if present.
   A directory without `RESULT.md` has not been run.
4. `refs/*` — reference paper summaries.

## Hard rules

**One idea per run.** Not two, not a list.

**Do not look at market data.** No parquet, no signal computation, no checking whether an idea
would have worked. The output is a hypothesis written before evidence; evidence consulted first
is not evidence any more. If a shell tool is available, that is a limitation of the harness, not
permission.

**State no expectations.** The file says what to run, never what you think will happen. No
predicted Sharpe, no "should outperform", no confidence, no guess at which leg or regime carries
the return, and no reason-it-will-work attached to a design choice. Before writing, reread the
draft and delete every such sentence — they hide in justifications after the obvious ones are
gone. An implementer who knows the expected answer finds it, invisibly.

**The implementer sees only `IDEA.md` and `CONTEXT.md`.** Not `CAMPAIGN.md`, not `refs/`, not
earlier results, not this skill. Anything needed must be in the file. A step that admits two
readings is underspecified, and two variants implemented differently are not comparable — which
destroys the only thing the campaign produces.

**Never modify an existing idea directory.** They are the record of what was tried.

## Priorities

Ordered; where they conflict, earlier wins.

1. **Open and evidence-driven.** Untested is not disproven. Only three things may kill a
   variant: CONTEXT's hard constraints, CAMPAIGN.md's scope, and this campaign's measured
   results. Published findings are weaker evidence than they look; priors are not evidence.
   Priors may order the queue, never empty it — if you set a direction aside because you expect
   it to fail, say so in one line under `## Directions set aside` so it can be overruled.
2. **Screen before optimizing.** Never propose a sweep for a strategy nobody has shown makes
   money. One configuration at defensible defaults plus a numeric gate declared in advance; the
   grid runs only if the gate passes. Most variants should die at the screen having cost one
   trial. That is the loop working.
3. **Know who is on the other side.** State who loses money to this trade and why they accept
   it. A weak story lowers priority but does not disqualify — write "no counterparty identified"
   rather than inventing a mechanism.

## Procedure

**1. Survey.** From the existing idea directories, establish which of CAMPAIGN.md's variant axes
have been exercised and what the results showed. Read the diagnostics, not just the verdicts: a
variant that failed for lack of signal is a different lesson from one that failed on costs, and
they point at opposite next steps — no signal means change the construction, signal eaten by
costs means attack turnover, holding period, or the liquidity cut.

**2. Choose a direction.** Three legitimate shapes:

- **New** — an axis nobody has moved, or a mechanism from `refs/` not yet tried here. Usually
  worth more than refining an axis already swept.
- **Built on** — extends a specific earlier idea. Legitimate and often the best move, but name
  the parent explicitly, state exactly what is inherited unchanged and what single thing
  differs. Inheriting a construction is fine; inheriting it *because it scored well* and then
  adding three changes at once produces a result you cannot attribute to any of them.
- **Composed** — combines two earlier strategies. Also legitimate, but it is one change: the
  composition. Do not simultaneously retune either component.

Not legitimate: an idea differing only in a parameter value inside a range an earlier idea
already declared. That is the same idea, and running it again inflates the trial count for
nothing. See `MOVES.md` for the standard ways to generate a genuinely different variant.

**3. Specify.** Write the algorithm. Follow `IDEA-TEMPLATE.md` exactly — four sections plus
frontmatter. The Strategy section carries the whole spec and is the part that matters.

**4. Write.** Create a new directory `<slug>/` beside the other idea directories and write
`<slug>/IDEA.md` inside it. That is the only file you create. The slug is lowercase,
hyphen-separated, and describes the construction rather than the campaign — `xsmom-30d-invvol`,
not `idea-7`. If the directory exists, choose another slug.

End by reporting the path you created.

## Stop conditions

Report instead of writing an idea only if the variant requires short stock or short options, or
falls outside CAMPAIGN.md's scope. In the latter case name the axis you were tempted by — it may
justify a second campaign.

**Data we do not have is not a stop condition.** Specify what the hypothesis needs, let the next
stage establish whether it can run. An idea parked for missing data is a standing argument for
acquiring that data; an idea never written is nothing.
