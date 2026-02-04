import botocore
import botocore.config as botoconfig
from botocore.exceptions import ClientError
from aiobotocore.session import get_session as boto_session

import asyncio
import aiohttp
from tqdm.asyncio import tqdm
from contextlib import asynccontextmanager, AsyncExitStack

import os
import sys
from posixpath import join, basename
import more_itertools as it
from itertools import product
from urllib.parse import urlparse
import re
from datetime import date, datetime, timedelta
import io
import zipfile
from hashlib import sha256
import argparse
import re
from calendar import Calendar

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import fs

from dotenv import load_dotenv
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
from collections import namedtuple
from dataclasses import dataclass

Pair = namedtuple("Pair", ["base", "quote"])


@dataclass(frozen=True, slots=True)
class MonthYear:
    month: int
    year: int

    @classmethod
    def all_since(cls, start: date) -> Iterable["MonthYear"]:
        today = date.today()
        for y in range(start.year, date.today().year + 1):
            s = start.month if y == start.year else 1
            e = today.month if y == today.year else 12
            for m in range(s, e + 1):
                yield cls(month=m, year=y)


@dataclass(frozen=True, slots=True)
class Obj:
    key: str
    bucket: str
    filename: str
    last_modified: datetime
    date: date
    pair: str
    public_url: str

    def __init__(self, bucket: str, key: str, last_modified: datetime):
        pair, day = parse_binance_filename(key)

        object.__setattr__(self, "key", key)
        object.__setattr__(self, "bucket", bucket)
        object.__setattr__(self, "filename", basename(key))
        object.__setattr__(self, "last_modified", last_modified)
        object.__setattr__(self, "date", day)
        object.__setattr__(self, "pair", pair)
        object.__setattr__(
            self,
            "public_url",
            f"https://s3.ap-northeast-1.amazonaws.com/{bucket}/{key}",
        )

    @classmethod
    def from_url(cls, url: str) -> "Obj":
        bkt, key = parse_object_store_url(url)
        return cls(bucket=bkt, key=key, last_modified=datetime.now())


@dataclass(frozen=True, slots=True)
class Context:
    session: aiohttp.ClientSession
    s3: Any
    r2: Any
    r2fs: fs.S3FileSystem


@asynccontextmanager
async def make_context():
    boto = boto_session()
    r2key = os.getenv("R2_ACCESS_KEY")
    r2sec = os.getenv("R2_SECRET_KEY")
    r2id = os.getenv("R2_ACCOUNT_ID")

    async with AsyncExitStack() as stack:
        session = await stack.enter_async_context(aiohttp.ClientSession())
        r2 = await stack.enter_async_context(
            boto.create_client(
                "s3",
                aws_access_key_id=r2key,
                aws_secret_access_key=r2sec,
                endpoint_url=f"https://{r2id}.r2.cloudflarestorage.com",
                region_name="auto",
            )
        )
        s3 = await stack.enter_async_context(
            boto.create_client(
                "s3", config=botoconfig.Config(signature_version=botocore.UNSIGNED)
            )
        )
        r2fs = fs.S3FileSystem(
            access_key=r2key,
            secret_key=r2sec,
            endpoint_override=f"https://{r2id}.r2.cloudflarestorage.com",
            region="auto",
        )

        yield Context(session=session, s3=s3, r2=r2, r2fs=r2fs)


class TransientError(Exception):
    pass


load_dotenv()

BINANCE_API_EXCHANGE_INFO = "https://api.binance.com/api/v3/exchangeInfo"
# ETHBTC-1m-2021-01-01.zip
BINANCE_VISION_DAILY_SPOT_ARCHIVE_PAIR_MONTH_YEAR = (
    "s3://data.binance.vision/data/spot/daily/klines/{0}/1m/{0}-1m-{1:04d}-{2:02d}"
)

# hive partitioned by year/month/day, one parquet file per day
MINUTE_BUCKET = (
    "r2://studies-binance-store/spot-1m/"  # year=YYYY/month=MM/day=DD/dataNN.parquet
)
DAILY_BUCKET = "r2://studies-binance-store/spot-1m-store/"
ONE_DAY_FILE = "r2://studies-binance-store/spot-1d.parquet"

EPOCH_S_MS_THRESHOLD = 10_000_000_000
CONCURRENCY = 20

l.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=l.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stdout,
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


def make_binance_prefix(
    pair: str,
    year: int | None,
    month: int | None,
    day: int | None,
    public_url: bool = False,
) -> str:
    if public_url:
        pfx = f"https://s3.ap-northeast-1.amazonaws.com/data.binance.vision/data/spot/daily/klines/{pair.upper()}/1m/{pair.upper()}-1m-"
    else:
        pfx = f"s3://data.binance.vision/data/spot/daily/klines/{pair.upper()}/1m/{pair.upper()}-1m-"
    if year is not None:
        pfx += f"{year:04d}-"
        if month is not None:
            pfx += f"{month:02d}-"
            if day is not None:
                pfx += f"{day:02d}"

    return pfx


def make_mirror_prefix(
    year: int | None, month: int | None, day: int | None, pair: str | None
) -> str:
    pfx = f"r2://studies-binance-store/spot-1m-mirror/year="
    if year is not None:
        pfx += f"{year:04d}/month="
        if month is not None:
            pfx += f"{month:02d}/day="
            if day is not None:
                pfx += f"{day:02d}"
                if pair is not None:
                    pfx += f"/{pair.upper()}-1m-{year:04d}-{month:02d}-{day:02d}"

    return pfx


def make_packed_prefix(year: int | None, month: int | None, day: int | None) -> str:
    pfx = f"r2://studies-binance-store/spot-1m/"
    if year is not None:
        pfx += f"year={year:04d}/"
        if month is not None:
            pfx += f"month={month:02d}/"
            if day is not None:
                pfx += f"day={day:02d}/"

    return pfx


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
        "-F",
        "--from-date",
        type=date.fromisoformat,
        help="Date in YYYY-MM-DD format to start syncing from (inclusive).",
        default=None,
    )
    parser.add_argument(
        "-S",
        "--sense-date",
        type=date.fromisoformat,
        help="Pair to sense earliest available archive date from, used only if --from-date is not given. Defaults to ETHBTC.",
        default=None,
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List available trading pairs and exit, -s is applied.",
    )
    parser.add_argument(
        "-a",
        "--available",
        action="store_true",
        help="List available archives for selected pairs and exit, -s is applied.",
    )
    parser.add_argument(
        "-j",
        "--concurrency",
        type=int,
        help="Number of concurrent downloads/uploads.",
        default=20,
    )

    args = parser.parse_args()

    if args.concurrency < 1:
        l.error("Concurrency must be at least 1.")
        return
    global CONCURRENCY
    CONCURRENCY = args.concurrency

    async with make_context() as ctx:
        pairs = await retrieve_spot_pairs(ctx)

        num_all = len(pairs)
        pairs = {k: v for k, v in pairs.items() if args.symbol_pattern.match(k)}
        num_sel = len(pairs)
        if num_sel == 0:
            l.error(f"no trading pairs match '{args.symbol_pattern.pattern}'.")
            return
        elif num_sel < num_all:
            l.info(f"filtered {num_all - num_sel} pairs.")

        if args.list:
            return await action_list_symbols(ctx)
        if args.available:
            return await action_catalog_available_archives(ctx, pairs)

        if args.from_date is None:
            if args.sense_date is not None:
                from_date = await determine_start_date(ctx, args.sense_date)
            else:
                from_date = await determine_start_date(ctx, "ETHBTC")
        else:
            from_date = args.from_date
        if from_date is None:
            l.error("No start date could be determined, and --from-date not given.")
            return

        await action_synchronize(ctx, pairs, args.from_date, date.today())


async def action_synchronize(ctx: Context, pairs: List[str], start: date, end: date):
    dates = [
        start + timedelta(days=i)
        for i in range(((end - timedelta(days=1)) - start).days)
    ]
    fut = [synchonize_day(ctx, pairs, d) for d in dates]
    await tqdm.gather(*fut, desc="synchronizing days", position=0)


async def action_list_symbols(ctx, pairs: Dict[str, Pair]):
    for symbol, pair in pairs.items():
        print(f"{symbol}: {pair.base}/{pair.quote}")


async def action_catalog_available_archives(ctx: Context, pairs: Dict[str, Pair]):
    for symbol in pairs.keys():
        cat = catalog_bucket(ctx.s3, make_binance_prefix(symbol, None, None, None))
        async for p in cat:
            if p.filename.endswith(".zip"):
                print(p.filename)


synchronize_day_sem = asyncio.Semaphore(2)


async def synchonize_day(ctx: Context, pairs: List[str], day: date):
    dst = join(
        make_packed_prefix(day.year, day.month, day.day).replace("r2://", ""),
        "data00.parquet",
    )

    async with synchronize_day_sem:
        fut = [process_single_archive(ctx, pair, day) for pair in pairs]
        res = [
            df
            for df in await tqdm.gather(
                *fut, desc=f"compacting {day} to {dst}", position=1
            )
            if df is not None
        ]
        if len(res) == 0:
            l.info(f"No data found for partition {day}, skipping.")
            return

        df = pl.concat(res)
        table = df.to_arrow()
        pq.ParquetWriter(
            dst,
            table.schema,
            filesystem=ctx.r2fs,
            compression="zstd",
        ).write_table(table)


async def process_single_archive(
    ctx: Context, pair: str, day: date
) -> pl.DataFrame | None:
    zip_url = make_binance_prefix(pair, day.year, day.month, day.day, True) + ".zip"
    chksm_url = zip_url + ".CHECKSUM"

    csv = await exponential_backoff(download_and_verify_csv, [ctx, zip_url, chksm_url])
    if csv is None:
        return None

    return await asyncio.to_thread(transform_csv_data, csv, pair)


async def bisect_bucket(
    c, pair: str, horizon: date, make_prefix: Callable[[str, date], str], find_end=True
) -> date | None:
    low = horizon
    high = date.today() - timedelta(days=1)
    ret = None

    while low <= high:
        mid = low + timedelta(days=(high - low).days // 2)
        u = urlparse(make_prefix(pair, mid))
        try:
            await c.head_object(Bucket=u.netloc, Key=u.path.lstrip("/"))
            ret = mid
            if find_end:
                low = mid + timedelta(days=1)
            else:
                high = mid - timedelta(days=1)
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                if find_end:
                    high = mid - timedelta(days=1)
                else:
                    low = mid + timedelta(days=1)
            else:
                raise

    return ret


async def find_latest_archive(r2, pair: str, horizon: date) -> date | None:
    """Finds the latest daily archive for ``pair`` between yesterday and ``horizon``."""

    return await bisect_bucket(
        r2,
        pair,
        horizon,
        make_prefix=lambda p, d: f"{make_mirror_prefix( d.year, d.month, d.day,p)}.csv",
        find_end=True,
    )


async def find_earliest_archive(s3, pair: str, horizon: date) -> date | None:
    """Finds the earliest daily archive for ``pair`` after ``horizon``."""

    return await bisect_bucket(
        s3,
        pair,
        horizon,
        make_prefix=lambda p, d: f"{make_binance_prefix(p, d.year, d.month, d.day)}.zip",
        find_end=False,
    )


async def determine_start_date(ctx: Context, pair: str) -> date | None:
    start = await find_latest_archive(ctx.r2, pair, date(2017, 1, 1))
    if start is None:
        return await find_earliest_archive(ctx.s3, pair, date(2016, 1, 1))
    else:
        return start + timedelta(days=1)


async def retrieve_spot_pairs(ctx: Context) -> Dict[str, Pair]:
    async with ctx.session.get(BINANCE_API_EXCHANGE_INFO) as resp:
        return {
            symbol["symbol"]: Pair(base=symbol["baseAsset"], quote=symbol["quoteAsset"])
            for symbol in (await resp.json())["symbols"]
            if symbol["isSpotTradingAllowed"]
        }


async def catalog_hive(ctx: Context, url: str) -> List[str]:
    ret: List[str] = []

    bucket, prefix = parse_object_store_url(url)
    pg = ctx.r2.get_paginator("list_objects_v2")
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


async def catalog_bucket(c, url: str) -> AsyncGenerator[Obj, None]:
    """List all objects under given prefix in bucket.

    Args:
        c: aiobotocore S3 client.
        url: s3://bucket/prefix

    Yields:
        Obj instances for each object found.
    """

    bucket, prefix = parse_object_store_url(url)
    pg = c.get_paginator("list_objects_v2")

    with tqdm(desc=f"cataloging {bucket}/{prefix}", position=0, unit="page") as bar:
        async for page in pg.paginate(Bucket=bucket, Prefix=prefix):
            bar.update(1)

            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/") or basename(key).startswith("."):
                    continue

                yield Obj(bucket=bucket, key=key, last_modified=obj["LastModified"])


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


read_csv_object_sem = asyncio.Semaphore(CONCURRENCY)


async def read_csv_object(
    ctx: Context, obj: Obj, required=False
) -> pl.DataFrame | None:
    async with read_csv_object_sem:
        try:
            resp = await ctx.r2.get_object(Bucket=obj.bucket, Key=obj.key)
            async with resp["Body"] as body:
                data = await body.read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey" and not required:
                return None
            else:
                raise
    return await asyncio.to_thread(transform_csv_data, data, obj.pair)


async def read_parquet_object(ctx: Context, url: str) -> pl.DataFrame:
    bucket, key = parse_object_store_url(url)
    resp = await ctx.r2.get_object(Bucket=bucket, Key=key)
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


async def compact_prefix_to_parquet(ctx: Context, objs: List[Obj], pqurl: str):
    objs = sorted(objs, key=lambda o: (o.date, o.pair))
    writer = None

    fut = [read_csv_object(ctx, obj) for obj in objs]
    res = [
        d
        for d in await tqdm.gather(*fut, desc=f"compacting {pqurl}", position=1)
        if d is not None
    ]
    if len(res) == 0:
        l.info(f"No data found for partition {pqurl}, skipping.")
        return
    df = pl.concat(res)
    table = df.to_arrow()

    pq.ParquetWriter(
        pqurl.replace("r2://", ""),
        table.schema,
        filesystem=ctx.r2fs,
        compression="zstd",
    ).write_table(table)


async def resample_daily_klines(ctx: Context, files: List[str], pqurl: str):
    writer = None

    files = sorted(files)
    num = (len(files) + CONCURRENCY - 1) // CONCURRENCY

    for batch in tqdm(
        it.batched(files, CONCURRENCY),
        total=num,
        desc="resampling daily klines",
        position=0,
    ):
        fut = [read_parquet_object(ctx, file) for file in batch]
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
                    filesystem=ctx.r2fs,
                    compression="zstd",
                )
            writer.write_table(table)


async def exponential_backoff(
    fn: Callable,
    args: List,
    retries: int | None = None,
    base_delay: float = 100.0,
    growth: float = 2.0,
    max_delay: float = 10.0,
):
    i = 0
    while retries is None or retries > 0:
        try:
            return await fn(*args)
        except TransientError:
            delay = base_delay * (growth**i)
            delay = min(delay, max_delay)
            l.info(f"Transient error, retrying in {delay:.1f} seconds...")
            await asyncio.sleep(delay)
            if retries is not None and i >= retries:
                raise
            i += 1


extract_object_partitioned_sem = asyncio.Semaphore(CONCURRENCY)


async def download_and_verify_csv(
    ctx: Context, zip_url: str, chksm_url: str | None
) -> bytes | None:
    async with extract_object_partitioned_sem:
        # fetch zip and verify checksum
        async with ctx.session.get(zip_url) as resp:
            if resp.status == 404:
                return None
            elif resp.status != 200:
                l.error(f"Failed to fetch archive {zip_url}, status code {resp.status}")
                raise TransientError("Failed to fetch archive, retrying.")
            compressed = await resp.read()

        if chksm_url is not None:
            got = sha256(compressed).hexdigest()
            async with ctx.session.get(chksm_url) as resp:
                if resp.status != 200:
                    l.warn(
                        f"Archive {zip_url} missing checksum file, skipping verification."
                    )
                else:
                    want = (await resp.read()).decode("utf-8").split()[0]
                    if got != want:
                        l.info(
                            f"Archive {zip_url} checksum mismatch: got {got}, want {want}"
                        )
                        raise TransientError("Checksum mismatch, re-download required.")
        else:
            l.warn(f"Archive {zip_url} missing checksum file, skipping verification.")

    # decompress csv and upload
    try:
        return decompress_csv(compressed)
    except Exception as e:
        l.error(f"Failed to decompress archive {zip_url}: {e}")
        raise TransientError("Decompression failed, re-download required.") from e


#        u = urlparse(zip_url)
#        pair, day = parse_binance_filename(u.path.lstrip("/"))
#        dsturl = f"{make_mirror_prefix( day.year, day.month, day.day, pair)}.csv"
#        dstbkt, dstkey = parse_object_store_url(dsturl)
#        await ctx.r2.put_object(
#            Bucket=dstbkt,
#            Key=dstkey,
#            Body=csv_data,
#        )
#
#        return pair, day


if __name__ == "__main__":
    with logging_redirect_tqdm():
        try:
            asyncio.run(main())
            l.info("Sync'd")
        except Exception as e:
            l.exception("Fatal error during sync", exc_info=e)
