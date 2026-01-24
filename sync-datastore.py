import botocore
import botocore.config as botoconfig
from aiobotocore.session import get_session as boto_session

import asyncio
import aiohttp
from tqdm.asyncio import tqdm

import os
from posixpath import join, basename
import more_itertools as it
from urllib.parse import urlparse
import re
from datetime import date, datetime
import io
import zipfile
from hashlib import sha256
import argparse
import re

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import fs

from dotenv import load_dotenv
import logging as l
from tqdm.contrib.logging import logging_redirect_tqdm

from typing import Dict, Iterable, Tuple, Set, NamedTuple, List
from collections import namedtuple
from dataclasses import dataclass

Pair = namedtuple("Pair", ["base", "quote"])


@dataclass(frozen=True, slots=True)
class Obj:
    key: str
    bucket: str
    last_modified: datetime
    date: date
    pair: str
    public_url: str

    def __init__(self, bucket: str, key: str, last_modified: datetime):
        pair, day = parse_binance_filename(key)

        object.__setattr__(self, "key", key)
        object.__setattr__(self, "bucket", bucket)
        object.__setattr__(self, "last_modified", last_modified)
        object.__setattr__(self, "date", day)
        object.__setattr__(self, "pair", pair)
        object.__setattr__(
            self,
            "public_url",
            f"https://s3.ap-northeast-1.amazonaws.com/{bucket}/{key}",
        )


class TransientError(Exception):
    pass


load_dotenv()

BINANCE_API_EXCHANGE_INFO = "https://api.binance.com/api/v3/exchangeInfo"
BINANCE_VISION_DAILY_SPOT_ARCHIVE = (
    "s3://data.binance.vision/data/spot/daily/klines/%s/1m"
)

MIRROR_BUCKET = f"r2://studies-binance-store/spot-1m-mirror/"
ONE_MINUTE_BUCKET = f"r2://studies-binance-store/spot-1m-store/"
ONE_DAY_FILE = f"r2://studies-binance-store/spot-1d.parquet"

EPOCH_S_MS_THRESHOLD = 10_000_000_000

l.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=l.INFO,
    datefmt="%H:%M:%S",
)


def new_r2(sess):
    return sess.create_client(
        "s3",
        aws_access_key_id=os.getenv("R2_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("R2_SECRET_KEY"),
        endpoint_url=f"""https://{os.getenv("R2_ACCOUNT_ID")}.r2.cloudflarestorage.com""",
        region_name="auto",
    )


def new_r2_fs():
    return fs.S3FileSystem(
        access_key=os.getenv("R2_ACCESS_KEY"),
        secret_key=os.getenv("R2_SECRET_KEY"),
        endpoint_override=f"""https://{os.getenv("R2_ACCOUNT_ID")}.r2.cloudflarestorage.com""",
        region="auto",
    )


def new_s3(sess):
    return sess.create_client(
        "s3", config=botoconfig.Config(signature_version=botocore.UNSIGNED)
    )


def parse_binance_filename(filename: str) -> Tuple[str, date]:
    # ETHBTC-1m-2021-01-01.zip
    m = re.match(
        r"^(\w+)-\w+-(\d{4}-\d{2}-\d{2}).(csv|zip(.CHECKSUM)?)$", basename(filename)
    )
    if m is None:
        raise ValueError(f"Unexpected archive filename format: {filename}")

    pair = m[1]
    day = date.fromisoformat(m[2])
    return pair, day


def parse_object_store_url(url: str) -> Tuple[str, str]:
    parsed = urlparse(url)
    if parsed.scheme not in ("s3", "r2"):
        raise ValueError("URL must start with s3:// or r2://")

    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    return bucket, prefix


def parse_pattern(arg_value: str) -> re.Pattern:
    try:
        return re.compile(arg_value)
    except re.error as e:
        raise argparse.ArgumentTypeError(f"Invalid regular expression: {e}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--symbol-pattern",
        type=parse_pattern,
        help="PCRE pattern to filter trading pairs.",
        default=".*",
    )
    parser.add_argument(
        "-p",
        "--archive-pattern",
        type=parse_pattern,
        help="PCRE pattern to filter archives.",
        default=".*",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List available trading pairs and exit, -s is applied.",
    )
    args = parser.parse_args()

    async with aiohttp.ClientSession() as httpsess:
        pairs = await retrieve_spot_pairs(httpsess)

        num_all = len(pairs)
        pairs = {k: v for k, v in pairs.items() if args.symbol_pattern.match(k)}
        num_sel = len(pairs)
        if num_sel == 0:
            l.error(f"no trading pairs match '{args.symbol_pattern.pattern}'.")
            return
        elif num_sel < num_all:
            l.info(f"filtered {num_all - num_sel} pairs.")

        if args.list:
            for symbol, pair in pairs.items():
                print(f"{symbol}: {pair.base}/{pair.quote}")
            return

        botosess = boto_session()
        async with new_s3(botosess) as s3, new_r2(botosess) as r2:
            l.info("Catalog existing mirrored files")
            have = set(
                [
                    basename(obj.key).replace(".csv", ".zip")
                    for obj in await catalog_bucket(r2, MIRROR_BUCKET)
                    if obj.key.endswith(".csv")
                ]
            )

            for pair in tqdm(list(pairs.keys()), desc="processing pairs", position=0):
                l.info(f"Fetching changes for {pair}")
                avail = await catalog_bucket(
                    s3, BINANCE_VISION_DAILY_SPOT_ARCHIVE % pair
                )

                todo = [obj for obj in avail if basename(obj.key) not in have]

                l.info(f"Verifying and extracting pair {pair}")
                changed_days = await verify_and_extract_partitioned(
                    httpsess,
                    todo,
                    r2,
                    MIRROR_BUCKET,
                )

            changed_months = set(
                [
                    (day.year, day.month)
                    for days in changed_days.values()
                    for day in days
                ]
            )

            for year, month in tqdm(
                changed_months, desc="compacting months", position=0
            ):
                l.info(f"Compacting year={year}, month={month} to parquet")
                partition = f"year={year}/month={month:02d}"
                objs = await catalog_bucket(r2, join(MIRROR_BUCKET, partition))

                await compact_prefix_to_parquet(
                    r2,
                    objs,
                    join(ONE_MINUTE_BUCKET, partition, f"data.parquet"),
                )

            l.info("Derive daily klines")
            files = await catalog_hive(r2, ONE_MINUTE_BUCKET)
            await resample_daily_klines(r2, files, ONE_DAY_FILE)


async def retrieve_spot_pairs(session: aiohttp.ClientSession) -> Dict[str, Pair]:
    async with session.get(BINANCE_API_EXCHANGE_INFO) as resp:
        return {
            symbol["symbol"]: Pair(base=symbol["baseAsset"], quote=symbol["quoteAsset"])
            for symbol in (await resp.json())["symbols"]
            if symbol["isSpotTradingAllowed"]
        }


async def catalog_hive(c, url: str) -> List[str]:
    ret: List[str] = []

    bucket, prefix = parse_object_store_url(url)
    pg = c.get_paginator("list_objects_v2")
    async for page in pg.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            bn = basename(key)
            if bn.startswith("."):
                continue

            m = re.match(r"^[\w,-]+/year=(\d{4})/month=(\d{2})/\w+.parquet$", key)
            if m is None:
                l.warning(f"Unexpected file in hive partitioned store: {key}")
                continue
            ret.append(f"s3://{bucket}/{key}")

    return ret


async def catalog_bucket(c, url: str) -> List[Obj]:
    ret: List[Obj] = []

    bucket, prefix = parse_object_store_url(url)
    pg = c.get_paginator("list_objects_v2")
    bar = tqdm(desc=f"cataloging {bucket}/{prefix}", position=0, unit="page")
    async for page in pg.paginate(Bucket=bucket, Prefix=prefix):
        bar.update(1)
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            bn = basename(key)
            if bn.startswith("."):
                continue
            ret.append(Obj(bucket=bucket, key=key, last_modified=obj["LastModified"]))
    bar.close()

    return ret


def decompress_csv(zip_bytes):
    with io.BytesIO(zip_bytes) as bio:
        with zipfile.ZipFile(bio) as zf:
            if len(zf.namelist()) != 1:
                raise ValueError("Unexpected number of files in zip archive.")

            csv_filename = zf.namelist()[0]
            if not csv_filename.endswith(".csv"):
                raise ValueError("Expected a CSV file in the zip archive.")

            with zf.open(csv_filename) as csv_file:
                return csv_file.read()


async def process_single_archive(
    r2, session: aiohttp.ClientSession, dsturl: str, obj: Obj
) -> pl.DataFrame:
    async with session.get(obj.public_url) as resp:
        body = await asyncio.to_thread(decompress_csv, await resp.read())

    dstbkt, dstkey = parse_object_store_url(
        join(
            dsturl,
            f"year={obj.date.year}",
            f"month={obj.date.month:02d}",
            f"day={obj.date.day:02d}",
            basename(obj.key),
        )
    )

    await r2.put_object(Bucket=dstbkt, Key=dstkey, Body=body)


async def read_csv_object(r2, obj: Obj) -> pl.DataFrame:
    resp = await r2.get_object(Bucket=obj.bucket, Key=obj.key)
    async with resp["Body"] as body:
        return await asyncio.to_thread(transform_csv_data, await body.read(), obj.pair)


async def read_parquet_object(r2, url: str) -> pl.DataFrame:
    bucket, key = parse_object_store_url(url)
    resp = await r2.get_object(Bucket=bucket, Key=key)
    async with resp["Body"] as body:
        return pl.read_parquet(io.BytesIO(await body.read()))


def transform_csv_data(data: bytes, pair: str) -> pl.DataFrame:
    return (
        pl.read_csv(
            io.BytesIO(data),
            has_header=False,
            new_columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "base_volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base_volume",
                "taker_buy_quote_volume",
                "ignore",
            ],
            schema_overrides={
                "open_time": pl.Int64,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "base_volume": pl.Float64,
                "close_time": pl.Int64,
                "quote_volume": pl.Float64,
                "trades": pl.Int64,
                "taker_buy_base_volume": pl.Float64,
                "taker_buy_quote_volume": pl.Float64,
                "ignore": pl.Int64,
            },
        )
        .with_columns(
            [
                pl.when(pl.col("open_time") > EPOCH_S_MS_THRESHOLD)
                .then(pl.from_epoch("open_time", time_unit="ms"))
                .otherwise(pl.from_epoch("open_time", time_unit="s"))
                .dt.replace_time_zone("UTC")
                .alias("open_time"),
                pl.when(pl.col("close_time") > EPOCH_S_MS_THRESHOLD)
                .then(pl.from_epoch("close_time", time_unit="ms"))
                .otherwise(pl.from_epoch("close_time", time_unit="s"))
                .dt.replace_time_zone("UTC")
                .alias("ts"),
                pl.lit(pair).alias("symbol"),
            ]
        )
        .drop("ignore")
        .sort("ts")
    )


async def compact_prefix_to_parquet(r2, objs: List[Obj], pqurl: str):
    wrfd = new_r2_fs()
    objs = sorted(objs, key=lambda o: (o.date, o.pair))
    writer = None
    num = (len(objs) + 19) // 20

    for batch in tqdm(
        it.batched(objs, 20),
        total=num,
        desc="compacting to parquet",
        position=1,
        leave=False,
    ):
        fut = [read_csv_object(r2, obj) for obj in batch]
        df = pl.concat(await asyncio.gather(*fut))
        table = df.to_arrow()

        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(
                pqurl.replace("r2://", ""),
                schema,
                filesystem=wrfd,
                compression="zstd",
            )

        writer.write_table(table)


async def resample_daily_klines(r2, files: List[str], pqurl: str):
    wrfd = new_r2_fs()
    writer = None

    files = sorted(files)
    num = (len(files) + 19) // 20

    for batch in tqdm(
        it.batched(files, 20), total=num, desc="resampling daily klines", position=0
    ):
        fut = [read_parquet_object(r2, file) for file in batch]
        for df in await asyncio.gather(*fut):
            daily = (
                df.with_columns([(pl.col("ts").dt.truncate("1d")).alias("ts")])
                .group_by(["ts", "symbol"])
                .agg(
                    [
                        pl.col("open").first().alias("open"),
                        pl.col("high").max().alias("high"),
                        pl.col("low").min().alias("low"),
                        pl.col("close").last().alias("close"),
                        pl.col("base_volume").sum().alias("base_volume"),
                        pl.col("quote_volume").sum().alias("quote_volume"),
                        pl.col("close_time").max().alias("close_time"),
                        pl.col("trades").sum().alias("trades"),
                        pl.col("taker_buy_base_volume")
                        .sum()
                        .alias("taker_buy_base_volume"),
                        pl.col("taker_buy_quote_volume")
                        .sum()
                        .alias("taker_buy_quote_volume"),
                    ]
                )
                .sort(["ts", "symbol"])
            )

            table = daily.to_arrow()
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(
                    pqurl.replace("r2://", ""),
                    schema,
                    filesystem=wrfd,
                    compression="zstd",
                )
            writer.write_table(table)


extract_object_partitioned_sem = asyncio.Semaphore(20)


async def extract_object_partitioned(
    session: aiohttp.ClientSession, r2, prefix, zip, chksm, csv
) -> Tuple[str, date] | None:
    async with extract_object_partitioned_sem:
        # see if csv exists and is up to date
        if csv is not None:
            if csv.last_modified >= zip.last_modified:
                return None
            else:
                l.info(f"Archive {zip} outdated, re-extracting.")

        # fetch zip and verify checksum
        async with session.get(zip.public_url) as resp:
            compressed = await resp.read()

        if chksm is not None:
            if chksm.last_modified >= zip.last_modified:
                got = sha256(compressed).hexdigest()
                async with session.get(chksm.public_url) as resp:
                    want = (await resp.read()).decode("utf-8").split()[0]

                if got != want:
                    l.info(f"Archive {zip} checksum mismatch: got {got}, want {want}")
                    raise TransientError("Checksum mismatch, re-download required.")
        else:
            l.warn(f"Archive {zip} missing checksum file, skipping verification.")

        # decompress csv and upload
        try:
            csv_data = decompress_csv(compressed)
        except Exception as e:
            l.error(f"Failed to decompress archive {zip}: {e}")
            raise TransientError("Decompression failed, re-download required.") from e

        pair, day = parse_binance_filename(zip.key)
        dsturl = join(
            prefix,
            f"year={day.year}",
            f"month={day.month:02d}",
            f"day={day.day:02d}",
            basename(zip.key).replace(".zip", ".csv"),
        )
        dstbkt, dstkey = parse_object_store_url(dsturl)
        await r2.put_object(
            Bucket=dstbkt,
            Key=dstkey,
            Body=csv_data,
        )

        return pair, day


async def verify_and_extract_partitioned(
    session, objs: List[Obj], r2, prefix: str
) -> Dict[str, Set[date]]:
    by_basename = {basename(obj.key): obj for obj in objs}

    # zip -> checksum
    files: Dict[Obj, Tuple[Obj | None, Obj | None]] = {
        zip: (
            by_basename.get(basename(zip.key).replace(".zip", ".zip.CHECKSUM")),
            by_basename.get(basename(zip.key).replace(".zip", ".csv")),
        )
        for zip in [obj for obj in by_basename.values() if obj.key.endswith(".zip")]
    }

    fut = [
        extract_object_partitioned(session, r2, prefix, zip, chksm, csv)
        for zip, (chksm, csv) in files.items()
    ]
    if len(fut) == 0:
        return {}

    return dict(
        it.map_reduce(
            [
                (x[0], x[1])
                for x in await tqdm.gather(
                    *fut,
                    total=len(fut),
                    desc="verifying and extracting daily archives",
                    position=1,
                    leave=False,
                )
                if x is not None
            ],
            keyfunc=lambda a: a[0],
            valuefunc=lambda a: a[1],
            reducefunc=set,
        )
    )


if __name__ == "__main__":
    with logging_redirect_tqdm():
        try:
            asyncio.run(main())
            l.info("Sync'd")
        except Exception as e:
            l.exception("Fatal error during sync:")
