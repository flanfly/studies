import polars as pl

from tqdm import tqdm
from typing import Optional, Dict, Any, Literal, Tuple

import uniswapv3.math as v3math
from uniswapv3.emulator import Emulator
from cli.token import parse_amount, composition_in_range

from math import floor

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


def main():
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
    rebalance_cost = 2 * 10**6  # in token0
    harvest_cost = 2 * 10**6  # in token0
    tick_spacing = 10  # for 0.005%
    fee_pips = 500
    position_width = 5  # 100 bps
    harvest_interval = dt.timedelta(week=1)
    reposition_interval = dt.hours(hours=2)
    initial_capital = 100 * 10**6  # in token0

    # running state
    tick_lower, tick_upper = None, None
    liquidity = None
    last_rebalance = None
    last_harvest = None
    pending_fees = 0
    capital = initial_capital

    # eval metrics
    frag = []

    l.info(f"start at {start['ts'].item()} with ${capital/10**6:.2f}")

    for i, row in tqdm(enumerate(train.iter_rows(named=True)), total=train.height):
        sqrt_price_x96 = int(row["sqrt_price_x96"])
        tick = v3math.get_tick_at_sqrt_ratio(sqrt_price_x96)
        pos_value = capital

        # predicates
        active_position = tick_lower is not None and tick_upper is not None
        in_range = active_position and tick_lower <= tick < tick_upper
        reposition = not active_position
        harvest = dt.timedelta(seconds=last_harvest - i) >= harvest_interval

        # trace
        new_gas = 0
        new_fees = 0
        lower = None
        upper = None

        # collect fees
        if in_range:
            assert liquidity > 0
            fee0 = int(float(row["fee_growth0"] or 0.0) * liquidity)
            fee1 = int(float(row["fee_growth1"] or 0.0) * liquidity)

            new_fees = as_base(fee0, fee1, sqrt_price_x96)
            pending_fees += new_fees

        # see if we should reposition
        elif active_position:
            theta, mu = row["theta"], row["mu"]
            lower = (v3math.get_sqrt_ratio_at_tick(tick_lower) / (2**96)) ** 2
            upper = (v3math.get_sqrt_ratio_at_tick(tick_upper) / (2**96)) ** 2
            assert lower < upper

            if lower <= mu < upper:
                # XXX do something with theta
                pass
            elif i - last_rebalance > 12 * 3600:
                reposition = True

            pos0, pos1 = composition_in_range(
                sqrt_price_x96, tick_lower, tick_upper, liquidity
            )
            pos_value = as_base(pos0, pos1, sqrt_price_x96)

        # swap and reposition
        if reposition:
            # mark position to market
            if active_position:
                # harvest fees
                if harvest:
                    new_gas += int(
                        (0.5 * pending_fees * fee_pips / 1_000_000) + harvest_cost
                    )
                    capital += fees
                    pending_fees = 0
                    last_harvest = i

                # add swap fees
                new_gas += int(
                    (0.5 * pos_value * fee_pips / 1_000_000) + rebalance_cost
                )

                capital -= new_gas
                if capital < 10**6:
                    l.info(f"broke at {row['ts']}")
                    break

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

            num_rebalances += 1
            last_rebalance = i

        # update running state

        human_price = (10 ** (d1 - d0)) / ((sqrt_price_x96 / (2**96)) ** 2)
        human_lower = (10 ** (d1 - d0)) / lower if lower is not None else None
        human_upper = (10 ** (d1 - d0)) / upper if upper is not None else None

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
                "reposition": reposition,
                "is_active": active_position and in_range,
                # position
                "price": human_price,
                "lower": human_lower,
                "upper": human_upper,
            }
        )

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

    # mark position to market
    # XXX add swap and gas fees
    # XXX retrieve pending_fees, add gas and swap fees
    pos0, pos1 = composition_in_range(sqrt_price_x96, tick_lower, tick_upper, liquidity)
    pos_value = as_base(pos0, pos1, sqrt_price_x96)

    d = 10**6
    start = res.head(1)
    end = res.tail(1)
    days = (end["ts"] - start["ts"]).days()
    pnl = (end["equity"] - start["equity"]).item() / d
    active = end["utilization"].item() * 100
    roi = pnl / (initial_capital / d) * 100
    ann = ((1 + pnl) ** (365 / days) - 1) * 100
    fees = end["fees"].item() / d
    gas = end["gas"].item() / d
    num_rp = end["num_repositions"].item() / d

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
