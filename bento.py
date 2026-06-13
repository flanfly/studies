import databento as db
import polars as pl

import argparse

from dotenv import load_dotenv
import os

load_dotenv()

import logging as l
import sys

l.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=l.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


def download(
    client, dataset: str, symbols: list[str], schema: str, start: str, end: str
):
    l.info(
        f"Downloading {len(symbols)} symbols from {dataset} with schema {schema} from {start} to {end}"
    )
    df = client.timeseries.get_range(
        dataset=dataset,
        symbols=symbols,
        start=start,
        stype_in="parent",
        end=end,
        schema=schema,
    ).to_df()
    l.info(f"Downloaded {len(df)} rows")
    return pl.from_pandas(df)


def run():
    p = argparse.ArgumentParser()
    p.add_argument("symbols", type=str, nargs="+", help="Symbols to download")
    p.add_argument("--dataset", type=str, default="GLBX.MDP3")
    p.add_argument("--schema", type=str, default="ohlcv-1d")
    p.add_argument("--start", type=str, default="2010-06-06T00:00:00")
    p.add_argument("--end", type=str, default="2020-12-31T00:00:00")
    p.add_argument("--output", type=str, default="databento.parquet")
    args = p.parse_args()

    client = db.Historical(os.getenv("DATABENTO_APIKEY"))

    df = download(client, args.dataset, args.symbols, args.schema, args.start, args.end)
    df.write_parquet(args.output)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        import traceback

        traceback.print_exc()
