import asyncio
from tqdm.asyncio import tqdm

from httpx import AsyncClient
from pydantic_xml import BaseXmlModel, element

from tempfile import TemporaryDirectory
import os
from os import path
import sys
from posixpath import join, basename
import more_itertools as it
from urllib.parse import urlparse
import re
from datetime import date, datetime, timedelta, timezone
import io
import zipfile
from hashlib import sha256
import argparse
from functools import wraps

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import fs

import logging as l
from tqdm.contrib.logging import logging_redirect_tqdm

from typing import (
    Dict,
    Iterable,
    Tuple,
    Any,
    Set,
    NamedTuple,
    List,
    AsyncGenerator,
    Callable,
)

EPOCH_S_MS_THRESHOLD = 10_000_000_000
EPOCH_MS_US_THRESHOLD = 20_000_000_000_000
CONCURRENCY = 20
FUNDING_SCHEMA = {
    "symbol": pl.Utf8,
    "time": pl.Int64,
    "fundingRate": pl.Float64,
}
FUTURES_SCHEMA = {
    "time": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Utf8,
}


nsmap = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


class Model(BaseXmlModel, ns="s3", nsmap=nsmap, search_mode="ordered"):
    pass


class CommonPrefix(Model, tag="CommonPrefixes"):
    prefix: str = element(tag="Prefix")


class Content(Model, tag="Contents"):
    key: str = element(tag="Key")
    last_modified: datetime = element(tag="LastModified")
    etag: str = element(tag="ETag")
    size: int = element(tag="Size")


class ListBucketResult(
    BaseXmlModel, tag="ListBucketResult", ns="s3", nsmap=nsmap, search_mode="unordered"
):
    name: str = element(tag="Name")
    prefix: str = element(tag="Prefix")
    max_keys: int = element(tag="MaxKeys")
    is_truncated: bool = element(tag="IsTruncated")
    next_marker: str | None = element(tag="NextMarker", default=None)

    common_prefixes: List[CommonPrefix] = element(
        tag="CommonPrefixes", default_factory=list
    )
    contents: List[Content] = element(tag="Contents", default_factory=list)


l.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=l.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)


def with_backoff(fn, start=10, step=1, limit=15):
    @wraps(fn)
    async def run(*args, **kwargs):
        i = start
        while True:
            wait = 2.0**i / 1000.0

            try:
                return await fn(*args, **kwargs)
            except e:
                l.error(f"{e}: wait {wait}s")
                await asyncio.sleep(wait)

            i = min(i + 1, limit)

    return run


async def kc_list(c: AsyncClient, prefix: str) -> list[str]:
    out: list[str] = []
    marker: str | None = None
    while True:
        q = {
            "delimiter": "/",
            "prefix": prefix,
        }
        if marker:
            q["marker"] = marker
        while True:
            try:
                resp = await c.get("https://historical-data.kucoin.com/", params=q)
                break
            except Exception as e:
                l.error(f"{prefix}: {e}")
                await asyncio.sleep(1)

        resp.raise_for_status()
        model = ListBucketResult.from_xml(resp.content)

        out.extend(p.prefix for p in model.common_prefixes)
        out.extend(c.key for c in model.contents)

        if not model.is_truncated:
            break
        marker = model.next_marker
        if not marker:
            l.warning(f"{prefix}: listing truncated but no NextMarker; stopping")
            break

    return out


async def kc_fetch_zip(
    c: AsyncClient,
    path: str,
    schema: dict[str, pl.DataType],
    tscols: list[str] = [],
    symbol: str | None = None,
) -> pl.DataFrame:
    while True:
        try:
            resp = await c.get(f"https://historical-data.kucoin.com/{path}")
            break
        except Exception as e:
            l.error(f"{path}: {e}")
            await asyncio.sleep(1)

    resp.raise_for_status()

    with io.BytesIO(resp.content) as bio:
        with zipfile.ZipFile(bio) as zf:
            if len(zf.namelist()) != 1:
                raise ValueError("Unexpected number of files in zip archive.")

            csv_filename = zf.namelist()[0]
            if not csv_filename.endswith(".csv"):
                raise ValueError("Expected a CSV file in the zip archive.")

            with zf.open(csv_filename) as csv_file:
                df = pl.read_csv(csv_file, schema=schema).with_columns(
                    [
                        pl.when(pl.col(col) > EPOCH_MS_US_THRESHOLD)
                        .then(pl.from_epoch(col, time_unit="us"))
                        .when(pl.col(col) > EPOCH_S_MS_THRESHOLD)
                        .then(pl.from_epoch(col, time_unit="ms"))
                        .otherwise(pl.from_epoch(col, time_unit="s"))
                        .dt.replace_time_zone("UTC")
                        .alias(col)
                        for col in tscols
                    ]
                    + ([pl.lit(symbol).alias("symbol")] if symbol else [])
                )
            return df


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-j",
        "--concurrency",
        type=int,
        help="Number of concurrent downloads/uploads.",
        default=20,
    )
    parser.add_argument(
        "-d", "--dataset", type=str, help="Data set to download.", default="ohlcv-12h"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.parquet",
    )

    args = parser.parse_args()

    if args.debug:
        l.getLogger().setLevel(l.DEBUG)

    if args.concurrency < 1:
        l.error("Concurrency must be at least 1.")
        return

    match args.dataset:
        case "ohlcv-5m":
            await download(
                FUTURES_SCHEMA,
                "data/futures/daily/klines/",
                "5m/",
                args.output,
                args.concurrency,
            )

        case "ohlcv-12h":
            await download(
                FUTURES_SCHEMA,
                "data/futures/daily/klines/",
                "12h/",
                args.output,
                args.concurrency,
            )

        case "funding-rates":
            await download(
                FUNDING_SCHEMA,
                "data/futures/daily/fundingRates/",
                "",
                args.output,
                args.concurrency,
            )

        case _:
            l.error(f"unknown data set {args.dataset}")


async def download(
    schema: dict[str, pl.DataType],
    level1: str,
    level2: str,
    output: str,
    concurrency: int,
):
    async with AsyncClient() as c:
        from aiostream import stream, pipe

        async def fetch(pfx: str) -> pl.DataFrame:
            symbol = None
            if "symbol" not in schema:
                symbol = pfx.removeprefix(level1).split("/")[0]

            return await with_backoff(kc_fetch_zip)(
                c, pfx, schema, tscols=["time"], symbol=symbol
            )

        async def list_dir(pfx: str):
            return stream.iterate(await with_backoff(kc_list)(c, f"{pfx}{level2}"))

        gen = (
            stream.iterate(await kc_list(c, level1))
            | pipe.map(list_dir, ordered=False, task_limit=concurrency)
            | pipe.flatten()
            | pipe.filter(lambda pfx: pfx.endswith(".zip"))
            | pipe.map(
                fetch,
                ordered=False,
                task_limit=concurrency,
            )
        )

        with open(output, "wb") as fd:
            writer: pq.ParquetWriter | None = None
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

            if writer is not None:
                writer.close()


if __name__ == "__main__":
    with logging_redirect_tqdm():
        try:
            asyncio.run(main())
        except Exception as e:
            l.exception("Fatal error during sync", exc_info=e)
