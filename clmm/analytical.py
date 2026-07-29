import polars as pl

from tqdm import tqdm

import uniswapv3.math as v3math
from uniswapv3.emulator import Emulator
from cli.token import composition_in_range

from math import floor
import datetime as dt

import sys
from dotenv import load_dotenv
import logging as l
from tqdm.contrib.logging import logging_redirect_tqdm

load_dotenv()

l.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=l.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)


def as_base(token0: int, token1: int, sqrt_price_x96: int) -> int:
    return token0 + (token1 << 192) // (sqrt_price_x96**2)


def liquidity_for_value(
    sqrt_price_x96: int, tick_lower: int, tick_upper: int, capital: int
) -> int:
    a0, a1 = composition_in_range(sqrt_price_x96, tick_lower, tick_upper)
    return capital * v3math.Q96 // as_base(a0, a1, sqrt_price_x96)


def main() -> None:
    df = pl.read_parquet("data.parquet").sort("ts")
    start = df.filter(
        (pl.col("fee_growth0").is_not_null()) & (pl.col("fee_growth1").is_not_null())
    ).min()
    df = df.filter((pl.col("ts") >= start["ts"]) & (pl.col("ts").dt.year() == 2026))

    train_len = floor(df.height * 0.99)
    train, test = (
        df.slice(0, train_len),
        df.slice(train_len),
    )

    # const parameters
    d0 = 6
    d1 = 18
    f0 = 10**d0
    f01 = 10 ** (d1 - d0)

    # approx. gas cost of burn+swap+mint in token, not incl. swap fee
    rebalance_cost = 2 * f0

    # approx. gas cost of collect+swap in token0, not incl. swap fee
    harvest_cost = 2 * f0

    # position shape
    tick_spacing = 10  # for 0.005%
    position_width = 5  # half num ticks left and right of active tick, 100 bps

    # min. time between harvest and reposition
    harvest_interval = dt.timedelta(weeks=1)
    reposition_interval = dt.timedelta(hours=2)

    # economics
    fee_tier = 500
    initial_capital = 100 * f0  # in token0

    # running state
    tick_lower: int | None = None
    tick_upper: int | None = None
    liquidity: int | None = None
    last_reposition: dt.datetime | None = None
    last_harvest: dt.datetime | None = None
    pending_fees = 0
    capital = initial_capital

    # eval metrics
    frag = []

    l.info(f"start at {start['ts'].item()} with ${initial_capital/f0:.2f}")

    for row in tqdm(train.iter_rows(named=True), total=train.height):
        # const state
        now = row["ts"]
        sqrt_price_x96 = int(row["sqrt_price_x96"])
        tick = v3math.get_tick_at_sqrt_ratio(sqrt_price_x96)
        pos_value = capital

        # timeouts
        harvest_blocked = now - (last_harvest or now) < harvest_interval
        reposition_blocked = now - (last_reposition or now) < reposition_interval

        # predicates
        active_position = tick_lower is not None and tick_upper is not None
        in_range = active_position and tick_lower <= tick < tick_upper
        harvest = not harvest_blocked and pending_fees > 0
        reposition = False

        # trace
        new_gas = 0
        new_fees = 0
        lower = None
        upper = None

        # add swap fees
        if in_range:
            assert liquidity is not None and liquidity > 0

            fee0 = int(float(row["fee_growth0"] or 0.0) * liquidity)
            fee1 = int(float(row["fee_growth1"] or 0.0) * liquidity)

            new_fees = as_base(fee0, fee1, sqrt_price_x96)
            pending_fees += new_fees

        # see if we should reposition
        elif active_position:
            assert tick_lower is not None and tick_upper is not None
            assert liquidity is not None

            theta, mu = row["theta"], row["mu"]
            lower = (v3math.get_sqrt_ratio_at_tick(tick_lower) / (2**96)) ** 2
            upper = (v3math.get_sqrt_ratio_at_tick(tick_upper) / (2**96)) ** 2
            assert lower < upper

            if lower <= mu < upper:
                # XXX do something with theta
                pass
            else:
                reposition = not reposition_blocked

            pos0, pos1 = composition_in_range(
                sqrt_price_x96, tick_lower, tick_upper, liquidity
            )
            pos_value = as_base(pos0, pos1, sqrt_price_x96)

        # swap and reposition
        if reposition or not active_position:
            # mark position to market
            if active_position:
                # add swap fees
                new_gas += int(
                    (0.5 * pos_value * fee_tier / 1_000_000) + rebalance_cost
                )

                capital -= new_gas

            tick = v3math.get_tick_at_sqrt_ratio(sqrt_price_x96)

            # new position limits
            tsp = tick_spacing
            tick_lower = max(
                Emulator.MIN_TICK,
                tick - max(0, (position_width - 1) * tsp),
            )
            tick_lower = (tick_lower // tsp) * tsp
            tick_upper = min(Emulator.MAX_TICK, tick + (position_width * tsp))
            tick_upper = ((tick_upper + tsp - 1) // tsp) * tsp
            assert Emulator.MIN_TICK <= tick_lower < tick_upper <= Emulator.MAX_TICK

            liquidity = liquidity_for_value(
                sqrt_price_x96, tick_lower, tick_upper, capital
            )

            last_reposition = now

        # harvest pending fees
        if harvest:
            new_gas += int((0.5 * pending_fees * fee_tier / 1_000_000) + harvest_cost)

            capital += pending_fees - new_gas
            pending_fees = 0

            last_harvest = now

        # still gas in the tank?
        if capital < f0:
            l.info(f"broke at {row['ts']}")
            break

        # update metrics
        human_price = f01 / ((sqrt_price_x96 / (2**96)) ** 2)
        human_lower = f01 / lower if lower is not None else None
        human_upper = f01 / upper if upper is not None else None

        frag.append(
            {
                "ts": row["ts"],
                # state delta
                "new_fees": new_fees,
                "new_gas": new_gas,
                # running equity
                "pending_fees": pending_fees,
                "position": pos_value,
                # state
                "did_reposition": reposition,
                "did_harvest": harvest,
                "is_active": active_position and in_range,
                # position
                "price": human_price,
                "lower": human_lower,
                "upper": human_upper,
            }
        )

    if len(frag) == 0:
        l.error("no metrics")
        return

    res = (
        pl.DataFrame(frag)
        .with_columns(
            equity=pl.col("pending_fees") + pl.col("position"),
            fees=pl.col("new_fees").cum_sum(),
            gas=pl.col("new_gas").cum_sum(),
            utilization=pl.col("is_active").mean(),
            num_repositions=pl.col("reposition").cum_sum(),
        )
        .sort("ts")
    )

    assert tick_lower is not None and tick_upper is not None
    assert liquidity is not None

    # XXX mark position to market
    # XXX add swap and gas fees
    # XXX retrieve pending_fees, add gas and swap fees

    start = res.head(1)
    end = res.tail(1)
    days = (end["ts"] - start["ts"]).days()
    pnl = (end["equity"] - start["equity"]).item() / f0
    active = end["utilization"].item() * 100
    roi = pnl / (initial_capital / f0) * 100
    ann = ((1 + pnl) ** (365 / days) - 1) * 100
    fees = end["fees"].item() / f0
    gas = end["gas"].item() / f0
    num_rp = end["num_repositions"].item()

    l.info(
        f"${pnl:.2f} ({roi:.2f}% ROI, {ann:.2f}% pa) pnl, ${fees:.2f} fees earned, spend ${gas:.2f} gas for {num_rp} rebalances over {days}, {active:.2f}% utilized"
    )

    res.write_parquet("res.parquet")
    print("done")


if __name__ == "__main__":
    with logging_redirect_tqdm():
        try:
            main()
        except Exception as e:
            l.exception("Fatal error during sync", exc_info=e)
            sys.exit(1)
