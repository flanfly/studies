import sys
import asyncio
import os

import polars as pl
from coingecko_sdk import AsyncCoingecko, APIError, RateLimitError
from tqdm.asyncio import tqdm

from typing import List, Tuple

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

PD_TO_GC = {
    "atom": "cosmos",
    "spx": "spx6900",
    "adi": "adi-token",
    "pyth": "pyth-network",
    "zec": "zcash",
    "eigen": "eigenlayer",
    "qnt": "quant",
    "aero": "aerodrome-finance",
    "vvs": "vvs-finance",
    "theta": "theta-network",
    "sand": "the-sandbox",
    "flr": "flare-networks",
    "pin": "pi-network",
    "virtual": "virtual-protocol",
    "tel": "telcoin",
    "snx": "synthetix",
    "ip": "story-2",
    "ondo": "ondo-finance",
    "mew": "cat-in-a-dogs-world",
    "vsn": "vision-3",
    "btse": "btse-token",
    "thbill": "theo-short-duration-us-treasury-fund",
    "akt": "akash-network",
    "ff": "falcon-finance-ff",
    "ohm": "olympus",
    "xcn": "chain-2",
    "op": "optimism",
    "hnt": "helium",
    "hot": "holotoken",
    "hbar": "hedera-hashgraph",
    "lunc": "terra-luna",
    "zil": "zilliqa",
    "acred": "apollo-diversified-credit-securitize-fund",
    "bnb": "binancecoin",
    "xmr": "monero",
    "stx": "blockstack",
    "dcr": "decred",
    "cake": "pancakeswap-token",
    "sei": "sei-network",
    "mana": "decentraland",
    "wlfi": "world-liberty-financial",
    "enj": "enjincoin",
    "arb": "arbitrum",
    "wif": "dogwifcoin",
    "m": "memecore",
    "chz": "chiliz",
    "fil": "filecoin",
    "wfi": "wefi",
    "ub": "unibase",
    "usdt": "tether",
    "vet": "vechain",
    "real": "reallink",
    "axs": "axie-infinity",
    "stable": "stable-2",
    "pol": "polygon-ecosystem-token",
    "algo": "algorand",
    "xlm": "stellar",
    "9bit": "the9bit",
    "ar": "arweave",
    "xaut": "tether-gold",
    "usda": "usda-2",
    "aster": "aster-2",
    "s": "sonic-3",
    "leo": "leo-token",
    "hash": "hash-2",
    "sol": "solana",
    "币安人生": "bianrensheng",
    "bsb": "block-street",
    "cc": "canton-network",
    "ausd": "agora-dollar",
    "soso": "sosovalue",
    "usd0": "usual-usd",
    "apt": "aptos",
    "sfp": "safepal",
    "reusd": "re-protocol-reusd",
    "gno": "gnosis",
    "grt": "the-graph",
    "pc0000031": "tradable-na-rent-financing-platform-sstn",
    "rose": "oasis-network",
    "kag": "kinesis-silver",
    "jito": "jito-governance-token",
    "usdf": "falcon-finance",
    "gwei": "ethgas-2",
}


async def main():
    if len(sys.argv) > 1:
        syms = [PD_TO_GC.get(s, s) for s in sys.argv[1:]]
    else:
        df = pl.read_parquet(
            "polarity/latest-data/*.parquet", missing_columns="insert"
        ).with_columns(asset=pl.col("asset").replace(PD_TO_GC, default=pl.col("asset")))
        syms = df["asset"].unique().to_list()
    l.info(f"Fetching exchange data for {len(syms)} unique assets...")

    async with AsyncCoingecko(
        demo_api_key=os.environ.get("COINGECKO_API_KEY"),
        environment="demo",
    ) as client:
        fut = [exchanges(client, r) for r in syms]
        results = dict(zip(syms, await tqdm.gather(*fut)))

    for sym, exch in results.items():
        print(f"{sym}:")
        for market, pair, vol in exch:
            print(f"  {market} - {pair} - {vol}")

    with open("exchange_data.json", "w") as f:
        import json

        json.dump(results, f, indent=2)


coingecko_sem = asyncio.Semaphore(2)


async def exchanges(client, coin_id: str) -> List[Tuple[str, str, float]]:
    async def _exchanges() -> List[Tuple[str, str, float]] | None:
        try:
            resp = await client.coins.tickers.get(id=coin_id)
            return [
                (t.market.name, f"{t.base}/{t.target}", t.volume)
                for t in resp.tickers[:5]
            ]

        except RateLimitError as e:
            l.error(f"Rate limit hit for {coin_id}: {e.message}")
            return None

        except APIError as e:
            l.error(f"API error for {coin_id}: {e.message}")
            return []

    async with coingecko_sem:
        while True:
            result = await _exchanges()
            if result is not None:
                return result
            l.warning(f"Retrying {coin_id} after API error...")
            await asyncio.sleep(1)  # Backoff before retrying


if __name__ == "__main__":
    with logging_redirect_tqdm():
        try:
            asyncio.run(main())
        except Exception as e:
            l.exception("Fatal error during sync", exc_info=e)
            sys.exit(1)
