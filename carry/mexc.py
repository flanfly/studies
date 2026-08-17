#!/usr/bin/env python3
"""Downloader for MEXC historical 5-minute spot klines.

The download page https://www.mexc.co/market-data-download?type=kline serves
files from a public S3-like bucket via two endpoints:

  * directory listing:
        GET https://www.mexc.co/file-svc/history/download?filePath=<path>
    `data` is a list of either "dir/" strings or
    {"fileName", "maskedUrl", "lastModified", "fileSize"} dicts.
  * file download: plain GET on the file's `maskedUrl` (CloudFront), no auth.

Each symbol has two buckets of packs:
  * `daily/`   -> one CSV per day per interval (intraday intervals only)
  * `monthly/` -> one CSV per month per interval (all intervals, incl. Week1)

This script uses the daily packs for 5-minute klines:

    SPOT2/kline/<symbol_id>/daily/Min5/<BASE>_<QUOTE>-Min5-<YYYY-MM-DD>.csv

Resolving the "hash" directories
--------------------------------
The directory names are NOT hashes of the symbol. Each trading pair has an
opaque 32-char `id` returned by

    GET https://www.mexc.co/api/platform/spot/market-v2/web/symbolsV2

which lists all spot symbols grouped by quote currency. Each entry carries
`id` (== the bucket directory name) and `vn` (base currency); the quote is
the grouping key. The website keeps this mapping client-side in a store named
`staticSpotSymbolsIdMap` (id -> symbol), built from symbolsV2.

The CSV file names inside each directory also embed the pair
(`BTC_USDT-Min5-2023-01-01.csv` -> BTCUSDT), which is the only way to
identify delisted symbols that are no longer in symbolsV2 (~7.2k of ~9.3k
dirs). This script therefore derives the symbol column from the file name
and downloads every directory in the bucket, delisted ones included.

CSV format: open_time,open,high,low,close,volume,amount,close_time
(epoch ms; volume = base volume, amount = quote volume).
"""

import asyncio
import io
import logging as l
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

import polars as pl
import pyarrow.parquet as pq
from aiostream import pipe, stream
from httpx import AsyncClient, Limits
from tqdm.asyncio import tqdm

EPOCH_S_MS_THRESHOLD = 10_000_000_000
EPOCH_MS_US_THRESHOLD = 20_000_000_000_000
CONCURRENCY = 20

SCHEMA = {
    "open_time": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "amount": pl.Float64,
    "close_time": pl.Int64,
}

BASE = "https://www.mexc.co"
FILE_SVC = BASE + "/file-svc/history/download"
HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "referer": "https://www.mexc.co/market-data-download",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
    ),
}

INTERVAL = "Min5"
BUCKET = "daily"  # one CSV per day

l.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=l.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)


class Listing(BaseModel):
    class File(BaseModel):
        fileName: str
        maskedUrl: str
        lastModified: str
        fileSize: int | None

    data: list[str] | list["Listing.File"]


async def mc_list(c: AsyncClient, filePath: str) -> dict[str, str | None]:
    while True:
        try:
            resp = await c.get(
                "https://www.mexc.co/file-svc/history/download",
                params={"filePath": filePath},
            )
            resp.raise_for_status()
        except Exception as e:
            l.error(f"mc_list {filePath}: {e}")
            await asyncio.sleep(1)

        break

    model = Listing(**resp.json())

    if len(model.data) > 0 and isinstance(model.data[0], str):
        return {fn: None for fn in model.data}
    else:
        return {fn.maskedUrl: fn.fileName for fn in model.data}


async def mc_fetch_csv(c: AsyncClient, url: str, fn: str) -> pl.DataFrame:
    symbol = fn.split("-")[0].replace("_", "")

    for attempt in range(8):
        try:
            resp = await c.get(url)
            resp.raise_for_status()
            break
        except Exception as e:
            l.error(f"{fn}: {e}")
            await asyncio.sleep(1 + attempt)
    else:
        raise RuntimeError(f"failed to download {fn} after 8 attempts")

    with io.BytesIO(resp.content) as bio:
        df = pl.read_csv(bio, schema=SCHEMA, null_values=["null"]).with_columns(
            [
                pl.when(pl.col("open_time") > EPOCH_MS_US_THRESHOLD)
                .then(pl.from_epoch("open_time", time_unit="us"))
                .when(pl.col("open_time") > EPOCH_S_MS_THRESHOLD)
                .then(pl.from_epoch("open_time", time_unit="ms"))
                .otherwise(pl.from_epoch("open_time", time_unit="s"))
                .dt.replace_time_zone("UTC")
                .alias("open_time"),
                pl.when(pl.col("close_time") > EPOCH_MS_US_THRESHOLD)
                .then(pl.from_epoch("close_time", time_unit="us"))
                .when(pl.col("close_time") > EPOCH_S_MS_THRESHOLD)
                .then(pl.from_epoch("close_time", time_unit="ms"))
                .otherwise(pl.from_epoch("close_time", time_unit="s"))
                .dt.replace_time_zone("UTC")
                .alias("close_time"),
                pl.lit(symbol).alias("symbol"),
            ]
        )
    return df


async def download(
    output: str,
    concurrency: int,
):
    async with AsyncClient(
        follow_redirects=True,
        timeout=60.0,
        limits=Limits(max_connections=concurrency * 2),
        headers=HEADERS,
    ) as c:
        dirs = [d.removesuffix("/") for d in await mc_list(c, "SPOT2/kline/")]
        l.info("kline bucket: %d symbol dirs (incl. delisted)", len(dirs))

        async def list_files(sid: str):
            files = await mc_list(c, f"SPOT2/kline/{sid}/daily/Min5/")
            return stream.iterate([(url, fn) for url, fn in files.items()])

        async def fetch(pair: Tuple[str, str]) -> pl.DataFrame:
            url, fn = pair
            return await mc_fetch_csv(c, url, fn)

        gen = (
            stream.iterate(dirs)
            | pipe.map(list_files, ordered=False, task_limit=concurrency)
            | pipe.flatten()
            | pipe.map(fetch, ordered=False, task_limit=concurrency)
        )

        with open(output, "wb") as fd:
            writer: pq.ParquetWriter | None = None
            try:
                async with gen.stream() as streamer:
                    async for df in tqdm(streamer, desc="download"):
                        table = df.to_arrow()
                        if writer is None:
                            writer = pq.ParquetWriter(
                                fd,
                                table.schema,
                                compression="zstd",
                            )
                        writer.write_table(table)
            finally:
                if writer is not None:
                    writer.close()


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-j",
        "--concurrency",
        type=int,
        help="Number of concurrent downloads.",
        default=CONCURRENCY,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--output",
        default="mexc-5m.parquet",
        help="Output parquet path.",
    )

    args = parser.parse_args()

    if args.debug:
        l.getLogger().setLevel(l.DEBUG)

    if args.concurrency < 1:
        l.error("Concurrency must be at least 1.")
        return

    await download(
        args.output,
        args.concurrency,
    )


if __name__ == "__main__":
    from tqdm.contrib.logging import logging_redirect_tqdm

    with logging_redirect_tqdm():
        try:
            asyncio.run(main())
        except Exception as e:
            l.exception("Fatal error during sync", exc_info=e)
