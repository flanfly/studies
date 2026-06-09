import argparse
import asyncio
import os
import sys

from httpx import AsyncClient
import tqdm
from tqdm import ProgressBar

import polars as pl

from dotenv import load_dotenv
import logging as l
from tqdm.contrib.logging import logging_redirect_tqdm

from typing import Tuple

from . import Exchange, Binance, KuCoin, HTX, Kraken

l.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=l.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)


async def retrieve(
    client: AsyncClient,
    bar: ProgressBar,
    ex: Exchange,
    num_bars: int,
    assets: list[str],
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    # fetch pairs
    # update bar.total += pairs
    # fetch pair klines
    # return dfs
    pass


async def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-n",
        "--num-bars",
        help="Number of days to fetch.",
        type=str,
        default=1000,
    )
    p.add_argument(
        "-k",
        "--output-klines",
        help="Destination for raw klines.",
        type=str,
        default="klines.parquet",
    )
    p.add_argument(
        "-s",
        "--output-symbols",
        type=str,
        help="Destination for symbols and their borrow rates.",
        default="symbols.parquet",
    )
    p.add_argument(
        "-q",
        "--quote-assets",
        type=str,
        default="USDT,USD,USD1",
        help="Quote assets to fetch klines for. Comma separated list.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )

    args = p.parse_args()

    if args.debug:
        l.getLogger().setLevel(l.DEBUG)

    assets = [a.strip() for a in args.quote_assets.split(",") if len(a.strip()) > 0]
    exchanges = [
        Binance(),
        KuCoin(),
        HTX(),
        Kraken(),
    ]
    frag_klines = []
    frag_symbols = []

    async with AsyncClient as client:
        with tqdm(desc="fetching") as bar:
            fut = [retrieve(client, bar, ex, args.num_bars, assets) for ex in exchanges]
            res = asyncio.gather(*fut)
            frag_symbols, frag_klines = unzip(res)

    pl.concat(frag_klines).write_parquet(args.output_klines)
    pl.concat(frag_symbols).write_parquet(args.output_symbols)


if __name__ == "__main__":
    with logging_redirect_tqdm():
        try:
            asyncio.run(main())
        except Exception as e:
            l.exception("Fatal error during sync", exc_info=e)
            sys.exit(1)
