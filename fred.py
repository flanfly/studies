import pandas_datareader.data as web
import datetime as dt

import polars as pl
from tqdm import tqdm
import pyarrow.parquet as pq

import logging as l
import sys

l.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=l.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

START_DATE = dt.datetime(2010, 1, 1)


def fred_download(series: list[str], output: str):
    with open(output, "wb") as fd:
        now = dt.datetime.now()
        pddf = web.DataReader(series, "fred", START_DATE, now)
        if pddf is None or pddf.empty:
            raise ValueError(f"No data for {series}")

        df = (
            pl.from_pandas(pddf.reset_index())
            .select(
                ts=pl.col("DATE").cast(pl.Datetime),
                **{s: pl.col(s).cast(pl.Float32) for s in series},
            )
            .sort("ts")
        )

        table = df.to_arrow()
        writer = pq.ParquetWriter(
            fd,
            table.schema,
            compression="zstd",
        )
        writer.write_table(table)
        writer.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("series", nargs="*", help="List of series to download")
    p.add_argument(
        "--output", help="Output parquet file path", default="output.parquet"
    )
    args = p.parse_intermixed_args()

    l.info(
        f"""Downloading {len(args.series)} series{"s" if len(args.series) != 1 else ""} to {args.output}"""
    )
    try:
        fred_download(args.series, args.output)
    except Exception as e:
        l.error(f"Error downloading data: {e}")
        sys.exit(1)
