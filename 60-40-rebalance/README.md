a simple, crazy-effective, calendar effect trade in macro etfs
robot james
Sep 11, 2026
∙ Paid
i like calendar effects.

trades exploiting calendar effects tend to be easy to understand, easy to test, and (usually) easy to execute.

they don’t need complicated models or hundreds of inputs.

to trade them, you often just need to know the time and date.

. . .

simple things ain’t necessarily easy though.

finding tradeable calendar effects isn’t a data mining problem.

you need to dig deeper to cause and effect.

. . .

in this article, i’m going to show you exactly how to do that.

we’re going to start with a really simple dumb calendar effect in bonds (green line), then we’re going to dig into it, understand it, and then ultimately build a much better trading strategy (blue line) based on that understanding.




unlevered this strategy would have returned almost 4000% over the last 20 years, betting full-stack on each trade, and trading less than twice a month on average.

. . .

i’m going to explain all the nerd thinking behind it, and then give you the trade, and a colab notebook which you can use to trade it, even if you have no brain.

if you’re impatient and just wanna get to the trade, scroll to the bottom.

if you wanna read why it works, i love you, i appreciate you, thank you, stay with me.

. . .

you have to start with an understanding that calendar effects are not caused by the calendar.

the end of the month does not have any special power.

a friday is not naturally better than a tuesday.

what matters is that people and institutions often do certain things at certain times.

. . .

some examples:

pension funds tend to rebalance around month-end

investors sell losing stocks before the end of the tax year

traders reduce risky positions and buy protection before the weekend

people tend to buy risky assets when they get paid

index funds trade when an index changes

liquidity changes around tradfi open and close (and lunch)

futures and options expire on fixed dates

crypto traders pay funding at scheduled times

tokens are unlocked on fixed dates

team and investor tokens vest every month or quarter

staking rewards and protocol emissions get paid on a regular schedule (and many recipients sell the rewards soon after they receive them)

exchanges and protocols may do large twap orders at predictable times (and market makers rebalance inventory around those twaps)

asian traders become active when their working day starts (and europeans and americans do later)

token buybacks can be daily or weekly

protocol revenue can be distributed on fixed days

lots of crypto accounting, reporting, re-balancing is concentrated around the 0UTC boundary, leading to reversal effects.

. . .

these repeated trades create repeated flows. if the flow is large enough (enough similar people doing similar things at a similar time) then this may move prices in a predictable way.

patterns you see in asset returns may look like calendar effects, but the calendar is only telling you when the trade flow happens.

. . .

discovering that returns have been unusually high or low at a certain time isn’t very useful unless you can understand why.

you should ask:

what scheduled flow could be causing it?

who is likely to be trading?

are they forced to trade, or do they just want to?

does the effect become stronger when the flow should be larger?

does price reverse after the flow finishes?

can you see the flow directly in the tape, or in volume, open interest, funding, exchange deposits, or order book imbalance?

what else (other than your pet theory) might be causing it instead?

. . .

the best calendar trades ain’t “buy because it’s midnight”

they are more like “buy because a predictable group of sellers has probably finished around midnight.”

. . .

here’s a really simple example i’ve shown you before.

long-duration treasury bonds have historically:

risen in price towards the end of the month

sold off near the start of the next month.

i’ve proxied the effect here with the tlt etf.




from this observation alone, we might propose a very simple trading strategy, which i’ve mentioned many times before:

buy tlt over the final five trading days of each month.

reverse short at the month-end close.

cover after the first five trading-day returns of the new month.

with no leverage, and assuming conservative round trip costs, that looks like this:




you might think this is proof enough that this trade is good.

but, if you search through enough charts like the above, you’ll find tons of stuff that looks like this.

to be confident, you really need to understand why the pattern exists.

and, in doing this, you’ll often discover the conditions in which the trade is likely to work and the conditions in which it probably won’t.

then you can make a better trading strategy than just blindly swinging the bat every month.

. . .

we need to think of plausible reasons why there might be excess buying pressure at the end of the month.

here are the ones i can come up with:

. . .

duration extension and benchmark flows

bond indices rebalance near month-end

as time passes and duration tends to reduce, index-tracking investors all tend to buy bonds around the same time

this would suggest regular month-end buying flow in treasury bonds, even when nothing else exciting is happening.

. . .

asset-allocation rebalancing

many “balanced” funds and pension managers target fixed weights in stocks and bonds (the 60/40 portfolio is an example)

if stocks outperform bonds, which they do more often than not, investors end up with too much stock exposure and not enough bond exposure.

so on average, near month end they tend to:

sell stocks

buy bonds

this would suggest excess buying flow for bonds at the end of month on average, cos stocks tend to outperform bonds on average

and we’d effect these flows to be bigger the more stocks have outperformed recently.

. . .

window-dressing

if you’re a manager with a bunch of trash in your portfolio, you might want to make it look more conservative when you have to disclose your holdings.

this would suggest stronger end of the month buying flow for bonds:

at quarter-end

and especially at year-end

and maybe, after risky assets have sold off and managers are most embarrassed.

. . .

so, are any of these true?

well, yeah. all of these are things that happen.

but markets are highly efficient systems that tend to absorb the price impact of predictable stuff like this.

the important question for traders is, not whether these things happen, but:

are these flows so massive that they move price enough that we can trade them?

. . .

you want to start with a model that explains the effect you see.

for example, excess tlt returns at month-end might be explained as:

month end excess return = baseline duration extension flow + equity rebalance flow + window dressing flow + noise

and the early month reversal might be explained as:

early month reversal = reversal of flow impact + noise

. . .

then you want try to prove that your model is wrong.

you can get good quality data and build complicated statistical models.

or you can propose some simple experiments, get your hands dirty in some free data, go slightly crazy for a while, and then make some educated guesses.

we’re gonna do the latter ofc.

. . .

take each possible reason you can think of why this anomaly would exist and:

ask yourself “this were true, what would i expect to see in the data?”

then look for it in the data

then ask yourself “what else might be causing any effect i just saw?”

then loop around again

and keep looping until you gain insight or go insane.

. . .

we’ll start with window dressing.

if managers were hiding out in treasuries when they’re embarrassed about disclosing their positions we might expect to see the baseline strategy behave in the following ways:

it would perform better at year-end and quarter-end than an ordinary month-end

it might perform better when spy has sold off recently

. . .

we actually don’t see either of those things:

the strategy actually does worse on-average at year-end...




and it does worse when SPY has sold off month-to-date...




this doesn’t necessarily mean that window dressing doesn’t have any impact.

it just means that it’s probably not that important and swamped by other effects.

it could also likely that this crude experiment is just too crappy to quantify a real window dressing effect.

but, overall, this doesn’t look very promising. so you should move on and come back to this if you find something that actually explains the bulk of the effect...

. . .

so, next, we’ll look at asset-allocation rebalancing

let’s assume the average “balanced fund” or pension manager wants to be in roughly:

60% stocks (which we’ll proxy with spy)

40% bonds (which we’ll proxy with ief)

this ain’t true, obviously. but it’s true enough and easy to model and will get us close enough to general rebalancing tendencies.

now we can estimate how far the portfolio has moved away from it’s 60/40 target and what the estimated rebalance trades are:

. . .

start with target weights in each asset:

target stock weight wS = 0.6

target bond weight wB = 0.4

if stocks move by RS and bonds move by RB then:

the new stock weight w′S​ = wS.(1+RB​) / (wS.(1+RS​) + wB.(1+RB​))

the new bond weight w′B​ = wB.(1+RB​) / (wS.(1+RS​) + wB.(1+RB​))​

and to get back to the 60/40 target:

the required stock trade is wB - ​w′S

the required bond trade is wB - w′B

. . .

so, if returns cause the portfolio to drift to:

62% stocks

38% bonds

then:

stock trade = 0.6 - 0.62 = -0.02

manager should sell stocks equal to 2% of portfolio value

bond trade = 0.4 - 0.38 = +0.02

manager should buy bonds equal to 2% of portfolio value

. . .

in python slop:

def implied_rebalance_trades(
    stock_return: float,
    bond_return: float,
    target_stock_weight: float = 0.60,
    target_bond_weight: float = 0.40,
) -> tuple[float, float]:
    stock_value = target_stock_weight * (1.0 + stock_return)
    bond_value = target_bond_weight * (1.0 + bond_return)
    portfolio_value = stock_value + bond_value

    current_stock_weight = stock_value / portfolio_value
    current_bond_weight = bond_value / portfolio_value

    stock_trade = target_stock_weight - current_stock_weight
    bond_trade = target_bond_weight - current_bond_weight

    return stock_trade, bond_trade

the outputs are fractions of the current portfolio value:

positive stock_trade = buy stocks

negative stock_trade = sell stocks

positive bond_trade = buy bonds

negative bond_trade = sell bonds

. . .

the indexing and wealth management world is a bunch of dinosaurs that tend to rebalance on a fixed calendar schedule.

so we can calculate rebalance pressure on a monthly horizon:

. . .

monthly pressure analysis

in every month: previous month-end → sixth-last trading-day close

. . .

we can sort each month into five buckets by the size of the month-end rebalancing pressure we expect in bonds.

bucket 1 contains the strongest expected bond selling

bucket 2 contains slight expected bond selling

bucket 3 contains mild expected bond buying

bucket 4 contains strong expected bond buying

bucket 5 contains the strongest expected bond buying

then we look at the returns of tlt (orange) and spy (blue) and long spy / short tlt (green) for each of those rebalancing strength buckets.





this is interesting.

the green bars show clear evidence of an end of month rebalancing effect between stocks and bonds. (we already knew that: see the post below)




. . .

we already knew that: see the post below:

three dead simple edges in macro etfs
three dead simple edges in macro etfs
robot james
·
13 Apr
Read full story
. . .

but, if rebalancing were the whole effect, we’d expect to see the orange bars (tlt returns) show a clear positive relationship with rebalance pressure - but they do not.




month-end tlt returns are positive even when when we see strong month-end selling pressure from rebalancing.

so, clearly, rebalancing pressure is not the full story.

. . .

we can also look at the first five days of the next month in the same day, where we expect the impact that the rebalance had to revert.




see an almost monotonic positive relationship in the green spy-tlt bars (see big green arrow.)




so, there’s extremely clear evidence of a rebalancing effect reversion between spy and tlt.

. . .

but we also see start of month tlt returns being negative in nearly all quintiles, even when the the EOM bond selling pressure is high and we’d expect reversion at the start of the month (see orange doodles.)




this is consistent with there being a constant bond bid at the end of the month, which reverts at the start of the month - completely separate from rebalancing effects.

. . .

now we should really check whether these effects are uniquely a month-end phenomenon.

we can do that with a placebo test.

. . .

we build three non-overlapping five-day windows in each month:

trading days 6-10

trading days 11-15

the final five trading days

at the start of each window, we calculate the hypothetical 60/40 drift since the previous month-end.

then we ask the question:

does rebalance pressure matter more at month-end than during the arbitrary mid-month windows?

if we see the same effects in the middle of the month as we do at the end, then we may have just discovered basic stock/bond mean-reversion.

if it becomes stronger at month-end, then that points to an actual scheduled month-end rebalancing flow.

we also expect to see a month-end bid in tlt, even when we control oft the rebalancing effect.

. . .

so, for each window, we recalculate the 60/540 rebalance pressure using spy and ief returns from the previous month-end up to the start of that window. then we measure the following five-day return of spy, tlt and spy minus tlt.

so conceptually:




then we compare the relationships by doing a simple linear regression.

the regression is basically

return = normal return + effect of rebalance pressure + constant month-end effect + extra effect of pressure at month-end


the most important two columns are:

pressure_effect_midmonth_bps_per_pct: how much pressure predicts returns during ordinary mid-month windows.

additional_pressure_effect_at_month_end: how much stronger that relationship becomes specifically at month-end.

the spy-minus-tlt result is really clear:

mid-month pressure effect: −5.9 bps

additional month-end effect: −80.3 bps


So the total month-end relationship is roughly:

-5,5 - 80.3 = -86.2bps per 1 percentage point of estimated bond-buying pressure.

remember: positive pressure = investors should buy bonds and sell stocks at the end of the month.

so:

so, during the middle of the month, estimated rebalance pressure has basically no relationship with subsequent stock-vs-bond returns. but at month-end, the relationship becomes enormous.

. . .

we also see clear evidence of a persistent bid in bonds at month-end.

for tlt, we see +22.3bp additional baseline month-end return even when rebalance pressure is zero-ish.


. . .

so, now we can more confidently say that our model of tlt returns around the turn of the month looks like:

tlt month end excess return = baseline duration extension flow + equity rebalance flow + noise


tlt early month reversal = reversal of flow impact + noise

and our model of spy excess returns looks like:

spy month end excess return = bond rebalance flow + noise


spy early month reversal = reversal of flow impact + noise

. . .

now we think we understand what’s going on, we can put together a simple trading strategy to take advantage of these effects.

basically we wanna do the following:

before month-end:

if strong expected bond selling, then buy spy into month-end.

if not, then buy tlt into month-end

at month-end:

if strong expected bond buying into month-end: short tlt at month-end and buy spy

if not, no position

. . .

to create a trading strategy, we need to come up with some cutoffs to estimate “strong expected bond selling / buying”

to do that, i plotted a histogram of the 60/40 spy/ief rebalance pressure and shaded the quintiles.




and, because round numbers are good, i came up with the following:

-50bps bond rebalance pressure = strong expected bond selling

+50bps bond rebalance pressure = strong expected bond buying

. . .

now we can define a dead simple trade. . .

the trade
at the close before the final 5 trading days of the month, calculate the 60/40 rebalance pressure (the amount of portfolio value a 60/40 spy/ief investor would need to move into bonds to restore 60/40)

then trade:


so operationally:

sixth-last trading-day close: calculate rebalance pressure and enter the appropriate final-five-day position.

month-end close: close that trade and immediately enter the appropriate first-five-day position.

fifth trading-day close of the new month: close everything.

stay in cash until the next signal date.

. . .

the trade (blue) looks like this if you bet full stack on each trade:




this strategy isn’t in the market that much, so you could lever it up with futures or leveraged etfs for greater returns.

. . .

i made a colab notebook to make it easy to trade it: https://colab.research.google.com/drive/1e4GAmTWCGR4zXjyC0S45xZuvm09E_00z?usp=sharing

run this at any time and it’ll show you the positions you should be in over the next 12 days:




(right now it’s showing everything as flat cos it’s the middle of the month.)

. . .

happy trading.

i love you

. . .

beep . . . boop.
