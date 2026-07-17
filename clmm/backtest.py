import polars as pl
import asyncio
import numpy as np

from uniswapv3.emulator import Emulator
from uniswapv3.load import Pool, from_ethereum, Tick
import uniswapv3.math as v3math

from dataclasses import dataclass
from functools import cached_property

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

TOKEN_RE = r"([0-9.]+)\s*([a-zA-Z]+)"


@dataclass(frozen=True)
class Token:
    symbol: str
    ord: int
    amount: int
    decimals: int

    @cached_property
    def str(self) -> str:
        return f"{self.amount / pow(10, self.decimals)}{self.symbol}"


def parse_amount(s: str, meta: Pool) -> Token:
    import re

    m = re.match(TOKEN_RE, s)
    if not m:
        raise ValueError(f"Could not parse amount: {s}")

    sym = m.group(2)
    amount = float(m.group(1))

    print(amount, sym)

    if sym.lower() == meta.symbol0.lower():
        return Token(
            symbol=sym, ord=0, amount=int(amount * pow(10, meta.d0)), decimals=meta.d0
        )
    elif sym.lower() == meta.symbol1.lower():
        return Token(
            symbol=sym, ord=1, amount=int(amount * pow(10, meta.d1)), decimals=meta.d1
        )
    else:
        raise ValueError(f"Unknown symbol: {sym}")


def token_to_sqrtp(tok: Token, meta: Pool) -> int:
    match tok.ord:
        case 0:
            ratio = 10**meta.d1 / tok.amount
        case 1:
            ratio = tok.amount / 10**meta.d0
        case _:
            assert False

    return int(np.sqrt(ratio) * (2**96))


def liquidity_for_value(
    sqrt_price_x96: int, tick_lower: int, tick_upper: int, tok: Token
) -> int:
    """L such that the position's total value at the current price equals tok.amount."""
    sa = v3math.get_sqrt_ratio_at_tick(tick_lower)
    sb = v3math.get_sqrt_ratio_at_tick(tick_upper)
    sc = min(max(sqrt_price_x96, sa), sb)  # clamp: one-sided ranges
    a0 = v3math.get_amount0_delta(sc, sb, v3math.Q96, False)  # token0 leg per Q of L
    a1 = v3math.get_amount1_delta(sa, sc, v3math.Q96, False)  # token1 leg per Q of L
    a1_in_0 = a1 * (1 << 192) // (sqrt_price_x96**2)  # token1 leg valued in token0
    if tok.ord == 0:
        return tok.amount * v3math.Q96 // (a0 + a1_in_0)
    a0_in_1 = a0 * (sqrt_price_x96**2) // (1 << 192)
    return tok.amount * v3math.Q96 // (a0_in_1 + a1)


async def load(
    blk_glob: str, log_glob: str
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Pool, pl.DataFrame]:
    blks = pl.read_parquet(blk_glob).select(
        pl.col("block_number"),
        ts=pl.from_epoch(pl.col("timestamp"), time_unit="s").dt.replace_time_zone(
            "UTC"
        ),
    )
    df = (
        pl.read_parquet(log_glob)
        .join(blks, on=["block_number"])
        .sort(["block_number", "transaction_index", "log_index"])
    )
    swaps, liq, params, meta = await from_ethereum(df)
    return swaps, liq, params, meta, blks


def simulate(
    swaps: pl.DataFrame,
    liq: pl.DataFrame,
    meta: Pool,
    lower_sqrtp: int,
    upper_sqrtp: int,
    my_liquidity: Token,
) -> tuple[pl.DataFrame, Token, Token]:
    from tqdm import tqdm

    pool = Emulator(
        meta.sqrt_price_x96,
        meta.tick,
        meta.liquidity,
        meta.ticks,
        meta.tick_spacing,
        meta.fee_pips,
        meta.protocol_fraction,
        meta.max_liquidity_per_tick,
    )

    tick_lower = v3math.get_tick_at_sqrt_ratio(lower_sqrtp)
    tick_upper = v3math.get_tick_at_sqrt_ratio(upper_sqrtp)
    tick_lower, tick_upper = max(min(tick_lower, tick_upper), Emulator.MIN_TICK), min(
        max(tick_lower, tick_upper), Emulator.MAX_TICK
    )
    assert Emulator.MIN_TICK <= tick_lower < tick_upper <= Emulator.MAX_TICK

    tl = int(np.floor(tick_lower / meta.tick_spacing) * meta.tick_spacing)
    tu = int(np.ceil(tick_upper / meta.tick_spacing) * meta.tick_spacing)
    l = liquidity_for_value(meta.sqrt_price_x96, tl, tu, my_liquidity)
    pool.set_position(tl, tu, l)
    init0, init1 = pool.position_amounts()

    blocks = set(swaps["block_number"].to_list()) | set(liq["block_number"].to_list())

    for bn in tqdm(sorted(blocks)):
        this_swaps = swaps.filter(pl.col("block_number") == bn)
        this_liq = liq.filter(pl.col("block_number") == bn)

        for ord in sorted(
            set(this_swaps["ord"].to_list()) | set(this_liq["ord"].to_list())
        ):
            for row in this_swaps.filter(pl.col("ord") == ord).iter_rows(named=True):
                if row["amount0"] > 0:
                    token_in = 0
                    amount_in = row["amount0"]
                    amount_out = row["amount1"]
                else:
                    token_in = 1
                    amount_in = row["amount1"]
                    amount_out = row["amount0"]

                pool.swap(bn, token_in, int(amount_in))

            for row in this_liq.filter(pl.col("ord") == ord).iter_rows(named=True):
                pool.modify_liquidity(
                    row["tick_lower"],
                    row["tick_upper"],
                    row["liquidity"],
                )

    end0, end1 = pool.position_amounts()

    if my_liquidity.ord == 0:
        pos_end = end0 + end1 / (pool.sqrt_price_x96 / (2**96)) ** 2
        hold_end = init0 + init1 / (pool.sqrt_price_x96 / (2**96)) ** 2
    else:
        pos_end = end1 + end0 * (pool.sqrt_price_x96 / (2**96)) ** 2
        hold_end = init1 + init0 * (pool.sqrt_price_x96 / (2**96)) ** 2

    il = Token(
        symbol=my_liquidity.symbol,
        ord=my_liquidity.ord,
        amount=int(pos_end - hold_end),
        decimals=my_liquidity.decimals,
    )
    hold = Token(
        symbol=my_liquidity.symbol,
        ord=my_liquidity.ord,
        amount=int(hold_end - my_liquidity.amount),
        decimals=my_liquidity.decimals,
    )

    return pool.swap_df.rechunk(), il, hold


def report(
    res: pl.DataFrame,
    blks: pl.DataFrame,
    meta: Pool,
    lower_sqrtp: int,
    upper_sqrtp: int,
    liquidity: Token,
    il: Token,
    hold: Token,
):
    import matplotlib.pyplot as plt

    d = 10 ** (meta.d1 - meta.d0)
    df = (
        res.join(
            blks.with_columns(ts=pl.col("ts"), bn=pl.col("block_number")),
            on="bn",
        )
        .sort(["ts", "ord"])
        .with_columns(
            price=pl.col("price") / d,
            fee0=pl.col("fee0") / (10**meta.d0),
            fee1=pl.col("fee1") / (10**meta.d1),
        )
        .with_columns(
            fee=pl.when(liquidity.ord == 0)
            .then(pl.col("fee0") + (pl.col("fee1") / pl.col("price")))
            .otherwise(pl.col("fee1") + (pl.col("fee0") * pl.col("price")))
            .cum_sum()
        )
    )

    last = df.with_columns(
        pl.col("fee0").cum_sum(),
        pl.col("fee1").cum_sum(),
    ).tail(1)
    if liquidity.ord == 0:
        fee_end = (last["fee0"] + (last["fee1"] / last["price"])).item()
    else:
        fee_end = (last["fee1"] + (last["fee0"] * last["price"])).item()
    fee_m2m = df.tail(1)["fee"].item()

    fee_pct = fee_end / (liquidity.amount / (10**liquidity.decimals))
    fee_drift = fee_m2m - fee_end

    print(f"""
        return    {fee_end} {liquidity.symbol}
        drift     {fee_drift} {liquidity.symbol}
        roic      {fee_pct * 100}%
        il        {il.str}
        price pnl {hold.str}
        """)

    fig, (ax_price, ax_fee) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    lower = ((lower_sqrtp / (2**96)) ** 2) / d
    upper = ((upper_sqrtp / (2**96)) ** 2) / d

    ax_price.plot(
        df["ts"],
        df["price"],
        label=f"{meta.symbol0}/{meta.symbol1}",
        color="#1f77b4",
        linewidth=2,
    )

    ax_price.axhline(
        upper, color="crimson", linestyle="--", alpha=0.8, label=f"upper tick"
    )
    ax_price.axhline(
        lower, color="forestgreen", linestyle="--", alpha=0.8, label=f"lower tick"
    )

    ax_price.set_ylabel(f"price ({meta.symbol0}/{meta.symbol1})", fontsize=12)
    ax_price.grid(True, linestyle=":", alpha=0.6)
    ax_price.legend(loc="upper left", frameon=True)

    price_diff = df["price"].diff().fill_null(0)
    bar_colors = ["#2ca02c" if change >= 0 else "#d62728" for change in price_diff]

    ax_fee.plot(df["ts"], df["fee"])

    ax_fee.set_ylabel(f"fee ({liquidity.symbol})", fontsize=12)
    ax_fee.set_xlabel("ts", fontsize=12)
    ax_fee.grid(True, linestyle=":", alpha=0.6)

    plt.subplots_adjust(hspace=0.05)

    plt.show()


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Uniswap v3 backtest",
    )
    parser.add_argument(
        "--blocks", nargs="+", help="Parquet files of collected blocks", default=[]
    )
    parser.add_argument(
        "--logs", nargs="+", help="Parquet files of collected logs", default=[]
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--lower", help="Lower price to cover with the position in quote currency."
    )
    parser.add_argument(
        "--upper", help="Upper price to cover with the position in quote currency."
    )
    parser.add_argument(
        "--liquidity",
        help="Position's liquidity in one token.",
    )
    parser.add_argument(
        "--output", help="Parquet file output", default="backtest.parquet"
    )

    args, unknown = parser.parse_known_args()

    if args.verbose:
        l.getLogger().setLevel(l.DEBUG)

    swaps, liq, params, meta, blocks = await load(args.blocks, args.logs)

    my_liquidity = parse_amount(args.liquidity, meta)
    my_lower = parse_amount(args.lower, meta)
    lower_sqrt = token_to_sqrtp(my_lower, meta)
    my_upper = parse_amount(args.upper, meta)
    upper_sqrt = token_to_sqrtp(my_upper, meta)

    print(
        f"backtest {meta.symbol0}/{meta.symbol1} at range {my_lower.str}-{my_upper.str} with {my_liquidity.str}"
    )

    res, il, hold = simulate(swaps, liq, meta, lower_sqrt, upper_sqrt, my_liquidity)
    res.write_parquet(args.output)
    print(f"result written to {args.output}")
    report(res, blocks, meta, lower_sqrt, upper_sqrt, my_liquidity, il, hold)


if __name__ == "__main__":
    with logging_redirect_tqdm():
        try:
            asyncio.run(main())
        except Exception as e:
            l.exception("Fatal error during sync", exc_info=e)
            sys.exit(1)
