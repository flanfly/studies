import yfinance as yf
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


def yf_download(tickers: list[str], output: str):
    with open(output, "wb") as fd:
        writer = None
        for t in tqdm(tickers, desc="downloading data", unit="ticker"):
            l.debug(f"Downloading {t} to {output}")

            pddf = yf.download(
                t,
                period="max",
                interval="1d",
                multi_level_index=False,
                auto_adjust=True,
                progress=False,
            )
            if pddf is None or pddf.empty:
                l.warning(f"No data for {t}, skipping")
                continue

            df = (
                pl.from_pandas(pddf.reset_index())
                .select(
                    open=pl.col("Open").cast(pl.Float32),
                    high=pl.col("High").cast(pl.Float32),
                    low=pl.col("Low").cast(pl.Float32),
                    close=pl.col("Close").cast(pl.Float32),
                    volume=pl.col("Volume").cast(pl.Int64),
                    ts=pl.col("Date").cast(pl.Datetime),
                    symbol=pl.lit(t.upper()).cast(pl.Utf8),
                )
                .sort("ts")
            )

            table = df.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(
                    fd,
                    table.schema,
                    compression="zstd",
                )
            writer.write_table(table)
            df = None
            pddf = None

        if writer is not None:
            writer.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("tickers", nargs="*", help="List of tickers to download")
    p.add_argument(
        "--output", help="Output parquet file path", default="output.parquet"
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_intermixed_args()

    l.info(
        f"""Downloading {len(args.tickers)} ticker{"s" if len(args.tickers) != 1 else ""} to {args.output}"""
    )
    try:
        yf_download(args.tickers, args.output)
    except Exception as e:
        l.error(f"Error downloading data: {e}")
        sys.exit(1)
