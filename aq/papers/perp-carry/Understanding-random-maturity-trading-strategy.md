> From: https://claude.ai/chat/0e28580d-c05e-4f4c-abc7-1eb89942cc08

# you asked

message time: 2026-08-06 14:06:17

explain their trading stragegy for me. this random maturity thing is going over my head.

---

# claude response

## The problem they're solving

With a normal futures contract, arbitrage is easy to reason about. If the futures price is out of line with spot, you put on the trade (long spot, short futures), and you *know* that on expiration day the two prices must converge, so you know exactly when you collect. Fixed maturity = guaranteed payday.

Perpetuals never expire. So there's no date on which the gap is forced to close. You can put on a trade that looks profitable and then just... sit there, with the gap staying open, or widening. Classic no-arbitrage pricing gives you nothing here, because it needs that terminal convergence.

## What "random maturity" means

Their fix is to weaken the definition of arbitrage. Normal arbitrage: zero cost today, guaranteed non-negative payoff **at a known time τ**. Random-maturity arbitrage: zero cost today, guaranteed non-negative payoff **at *some* time τ̃, which is random but bounded**. You will get paid. You just don't know when.

Why is that guaranteed? Because of the funding rate. While you sit in the trade, the wide side pays you every 8 hours, and — crucially — the funding rate *increases with the size of the gap*. So the wider the gap gets, the faster you're being paid to wait. Their key insight is on p. 17: even if the gap is still open when you eventually unwind, the accumulated funding payments will have exceeded any finite loss on the spread. The only way you lose is if the gap runs off to infinity, which they rule out by assumption (Assumption 2, the "no-bubble" condition).

## Where the fundamental price comes from

Set the payoff of the hedged position to zero and you get one condition: **funding rate = interest rate**, Φ = r. That's it. In equilibrium the futures should trade at whatever premium makes the funding you collect exactly compensate the cost of financing the spot leg.

The wrinkle is Binance's *clamp*. Funding isn't a smooth function of the gap — when the futures-spot premium is small, funding is pinned flat at ι = 0.01% per 8 hours regardless of the gap. So if r ≠ ι, the condition Φ = r can't be satisfied anywhere inside the flat region; the equilibrium has to sit right at the *edge* of the clamp, where funding starts moving again. That's what the γ term in Equation 13 is doing. Ignoring the clamp and just assuming F = S is what the "No Clamp" column in Table 7 does, and it roughly halves the Bitcoin Sharpe ratio.

## The actual trade

Every hour they compute ρ (Equation 21): how far the observed futures-spot gap is from the theoretical one, expressed as an annualized interest rate. Then:

- ρ above the threshold → futures too expensive → **long spot, short perp**, financed by borrowing cash
- ρ below the negative threshold → **short spot, long perp**, lend out the proceeds
- ρ returns to 0 → **close**

The threshold is just your round-trip trading cost annualized: ρ = κc, with κ = 1095 (three funding periods a day × 365). A retail trader's round-trip cost of ~4.9 bps becomes a 53% threshold; the fattest fee tier needs 179%.

Those percentages look absurd until you realize they're annualized rates on trades that close in hours. BTC's mean |ρ| of 52%/year is a futures-spot gap of about **5 basis points**. It's a tiny price discrepancy earning a large annualized return because it closes fast — average holding time is under 5 hours at zero fees.

## What they find

Sharpe of 11.65 for BTC at zero fees, still 3.35 at retail fees. Higher fees → wider threshold → fewer, longer trades (96 hours average instead of 5).

Two things worth noting. First, Table 9: most of the profit comes from **price convergence, not funding payments**, which contradicts how the industry usually describes this trade. Second, the honest caveat — this isn't riskless. You're financing a margined position with no guaranteed exit date. Their defense is the drawdown data: BTC's worst drawdown is about 4%, so 4% of capital in reserve would have avoided liquidation over the whole sample.

---
Powered by [AI Exporter](https://saveai.net)