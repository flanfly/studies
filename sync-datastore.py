import botocore
import botocore.config as botoconfig
from botocore.exceptions import ClientError
import aioboto3

from binance_common.configuration import ConfigurationRestAPI
from binance_common.constants import SPOT_REST_API_PROD_URL
from binance_sdk_spot.spot import Spot

import obstore
from obstore.store import S3Store, GCSStore, LocalStore

import asyncio
import aiohttp
from tqdm.asyncio import tqdm
from contextlib import asynccontextmanager, AsyncExitStack

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
class Context:
    session: aiohttp.ClientSession
    s3: Any
    r2: Any
    r2fs: fs.S3FileSystem
    storage_options: Dict[str, str]
    binance: Any


@asynccontextmanager
async def make_context():
    boto = aioboto3.Session()
    r2key = os.getenv("R2_ACCESS_KEY")
    r2sec = os.getenv("R2_SECRET_KEY")
    r2id = os.getenv("R2_ACCOUNT_ID")
    r2ep = f"https://{r2id}.r2.cloudflarestorage.com"
    so = {
        "aws_access_key_id": r2key,
        "aws_secret_access_key": r2sec,
        "aws_endpoint_url": r2ep,
        "aws_region": "auto",
    }
    bid = os.getenv("BINANCE_API_KEY")
    bkey = os.getenv("BINANCE_API_SECRET")

    async with AsyncExitStack() as stack:
        session = await stack.enter_async_context(aiohttp.ClientSession())
        r2 = await stack.enter_async_context(
            boto.client(
                "s3",
                aws_access_key_id=r2key,
                aws_secret_access_key=r2sec,
                endpoint_url=r2ep,
                region_name="auto",
            )
        )
        s3 = await stack.enter_async_context(
            boto.client(
                "s3", config=botoconfig.Config(signature_version=botocore.UNSIGNED)
            )
        )
        r2fs = fs.S3FileSystem(
            access_key=r2key,
            secret_key=r2sec,
            endpoint_override=r2ep,
            region="auto",
        )

        binance = Spot(
            config_rest_api=ConfigurationRestAPI(
                api_key=bid, api_secret=bkey, base_path=SPOT_REST_API_PROD_URL
            )
        )

        yield Context(
            session=session,
            s3=s3,
            r2=r2,
            r2fs=r2fs,
            storage_options=so,
            binance=binance.rest_api,
        )


class TransientError(Exception):
    pass


load_dotenv()

BINANCE_API_EXCHANGE_INFO = "https://api.binance.com/api/v3/exchangeInfo"
# ETHBTC-1m-2021-01-01.zip
BINANCE_VISION_DAILY_SPOT_ARCHIVE_PAIR_MONTH_YEAR = (
    "s3://data.binance.vision/data/spot/daily/klines/{0}/1m/{0}-1m-{1:04d}-{2:02d}"
)
BINANCE_KLINE_SCHEMA = {
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
}
KLINE_COLUMNS = [
    "ts",
    "symbol",
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
]
EPOCH_S_MS_THRESHOLD = 10_000_000_000
EPOCH_MS_US_THRESHOLD = 20_000_000_000_000
CONCURRENCY = 20
PACKED_DEFAULT_FILENAME = "data00.parquet"
RESAMPLED_ALL_FILENAME = "all.parquet"
RESAMPLED_STABLE_FILENAME = "stables.parquet"

l.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=l.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)


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


def mkurl_src(
    pair: str,
    year: int,
    month: int,
    day: int,
    public_url: bool = False,
) -> str:
    if public_url:
        pfx = f"https://s3.ap-northeast-1.amazonaws.com/data.binance.vision/data/spot/daily/klines/{pair.upper()}/1m/{pair.upper()}-1m-"
    else:
        pfx = f"s3://data.binance.vision/data/spot/daily/klines/{pair.upper()}/1m/{pair.upper()}-1m-"
    return pfx + f"{year:04d}-{month:02d}-{day:02d}"


def mkurl_1m(year: int, month: int, day: int, scheme: str = "r2://") -> str:
    return f"{scheme}studies-binance-store/spot-1m/year={year:04d}/month={month:02d}/day={day:02d}/{PACKED_DEFAULT_FILENAME}"


def date_range(value):
    """Custom argparse type to handle YYYY-MM-DD or YYYY-MM-DD,YYYY-MM-DD."""
    try:
        if "," in value:
            # Case: YYYY-MM-DD,YYYY-MM-DD
            start_str, end_str = value.split(",")
            start = datetime.strptime(start_str.strip(), "%Y-%m-%d").date()
            end = datetime.strptime(end_str.strip(), "%Y-%m-%d").date()
        else:
            # Case: YYYY-MM-DD (to today)
            start = datetime.strptime(value.strip(), "%Y-%m-%d").date()
            end = date.today() - timedelta(days=1)

        if start > end:
            raise argparse.ArgumentTypeError(
                f"Start date ({start}) must be before end date ({end})"
            )

        return start, end
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: '{value}'. Use YYYY-MM-DD or YYYY-MM-DD,YYYY-MM-DD."
        )


def _make_obstore_from_url(url: str):
    u = urlparse(url)
    if u.scheme == "gs":
        return GCSStore.from_url(f"gs://{u.netloc}")

    elif u.scheme == "r2":
        r2id = os.getenv("R2_ACCOUNT_ID")
        r2ep = f"https://{r2id}.r2.cloudflarestorage.com"
        return S3Store(
            u.netloc,
            access_key_id=os.getenv("R2_ACCESS_KEY"),
            secret_access_key=os.getenv("R2_SECRET_KEY"),
            endpoint=r2ep,
            region="auto",
        )

    elif u.scheme == "s3":
        return S3Store.from_url(f"s3://{u.netloc}")

    elif u.scheme == "file" or u.scheme == "":
        if u.netloc:
            return LocalStore(u.netloc)
        return LocalStore("/") if path.isabs(u.path) else LocalStore(".")

    else:
        raise ValueError(f"Unsupported URL scheme: {u.scheme}")


async def put_object(dst: str, src: Any):
    u = urlparse(dst)
    store = _make_obstore_from_url(dst)
    await obstore.put_async(store, path.normpath(u.path.lstrip("/")), src)


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
        "-l",
        "--list",
        action="store_true",
        help="List available trading pairs and exit, -s is applied.",
    )
    parser.add_argument(
        "-j",
        "--concurrency",
        type=int,
        help="Number of concurrent downloads/uploads.",
        default=20,
    )
    parser.add_argument(
        "-d",
        "--daily",
        action="store_true",
        help="Resample data into daily klines after synchronization.",
    )
    parser.add_argument(
        "--window",
        type=int,
        help="Number of days to synchronize, counting backwards from yesterday.",
    )
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        help="Fill missing daily klines until yesterday with live data from Binance API, used with --daily.",
    )
    parser.add_argument(
        "--output-daily-file",
        type=str,
        default="r2://studies-binance-store/spot-1d/all.parquet",
        help="Destination for resampled daily klines, used with --daily.",
    )
    parser.add_argument(
        "--output-stables-file",
        type=str,
        default="r2://studies-binance-store/spot-1d/stables.parquet",
        help="Destination for resampled daily klines with stablecoin quote assets, used with --daily.",
    )
    parser.add_argument(
        "--stable-coin",
        type=str,
        default="USDT",
        help="Quote asset symbol to identify stablecoin pairs for separate output, used with --daily. Default is USDT.",
    )
    parser.add_argument(
        "--kline-offset",
        type=int,
        default=0,
        help="Hour offset for daily kline boundaries, used with --daily. Default is 0 (UTC midnight).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    if args.debug:
        l.getLogger().setLevel(l.DEBUG)

    if args.concurrency < 1:
        l.error("Concurrency must be at least 1.")
        return
    global CONCURRENCY
    CONCURRENCY = args.concurrency

    global synchronize_day_sem, read_parquet_object_sem, extract_object_partitioned_sem, retrieve_klines_sem
    synchronize_day_sem = asyncio.Semaphore(2)
    read_parquet_object_sem = asyncio.Semaphore(CONCURRENCY)
    extract_object_partitioned_sem = asyncio.Semaphore(CONCURRENCY)
    retrieve_klines_sem = asyncio.Semaphore(CONCURRENCY)

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
            return await action_list_symbols(ctx, pairs)

        if not args.daily:
            start_date = await determine_start_date(ctx, "ETHBTC")
            dates = [start_date, date.today() - timedelta(days=1)]

            if dates is None:
                l.error("No start date could be determined, and --from-date not given.")
                return

            if dates[0] > dates[1]:
                l.info("No new data to synchronize.")
                print(dates[0])
                return

            horizon = await action_synchronize_minute_klines(
                ctx, pairs, dates[0], dates[1]
            )
            if horizon is not None:
                print(horizon)
            else:
                print(dates[0])

        else:
            yd = date.today() - timedelta(days=1)
            if args.window is None:
                dates = [date(2017, 7, 14), yd]
            else:
                dates = [yd - timedelta(days=args.window - 1), yd]

            await action_synchronize_daily_klines(
                ctx,
                args.output_daily_file,
                args.output_stables_file,
                args.stable_coin,
                args.kline_offset,
                args.fill_missing,
                dates[0],
                dates[1],
            )


async def action_synchronize_minute_klines(
    ctx: Context, pairs: List[str], start: date, end: date
) -> date | None:
    dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    fut = [synchronize_day(ctx, pairs, d) for d in dates]
    res = await tqdm.gather(
        *fut, desc=f"synchronizing {start} to {end} ({len(dates)} days)"
    )
    procd = [p[1] for p in zip(res, dates) if p[0] is not None and p[0] > 0]
    if len(procd) > 0:
        return max(procd)
    return None


async def action_synchronize_daily_klines(
    ctx: Context,
    alldst: str,
    stabledst: str,
    stable: str,
    kline_offset: int,
    fill_missing: bool,
    start: date,
    end: date,
):
    l.info(f"resampling daily klines from {start} to {end}")

    dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    files = [mkurl_1m(d.year, d.month, d.day) for d in dates]
    t = len(files) // CONCURRENCY + (1 if len(files) % CONCURRENCY != 0 else 0)
    b = it.batched(sorted(files), CONCURRENCY)

    with TemporaryDirectory() as tmp:
        pass1 = join(tmp, "pass1.parquet")

        # resample, possibly unaligned and buffer locally
        with open(pass1, "wb") as fd:
            writer = None
            for batch in tqdm(b, unit="batch", position=0, total=t):
                writer = await resample_daily_klines(
                    ctx, writer, batch, kline_offset, fd
                )
            if writer is not None:
                writer.close()

        pass2 = join(tmp, "all.parquet")
        resample_sorted_dataframe(
            pl.scan_parquet(pass1).set_sorted("ts"), kline_offset
        ).sink_parquet(pass2, compression="zstd")

        if fill_missing:
            pass2a = join(tmp, "pass2a.parquet")
            yd = datetime.now(timezone.utc).replace(
                hour=kline_offset, minute=0, second=0, microsecond=0
            ) - timedelta(days=1)
            active = (await retrieve_spot_pairs(ctx)).keys()
            tsmax = (
                pl.scan_parquet(pass2)
                .filter(pl.col("symbol").is_in(active))
                .group_by("symbol")
                .agg(pl.col("ts").max())
                .filter(pl.col("ts") < yd)
                .collect()
            )
            fut = [
                retrieve_klines(ctx, r["symbol"], r["ts"], yd + timedelta(hours=23))
                for r in tsmax.iter_rows(named=True)
                if r["ts"] < yd
            ]
            live = [
                df.lazy()
                for df in await tqdm.gather(
                    *fut, desc="retrieving live klines", position=1
                )
                if df is not None
            ]
            frames = [pl.scan_parquet(pass2).set_sorted("ts"), *live]
            resample_sorted_dataframe(
                pl.concat([df.select(KLINE_COLUMNS) for df in frames]).sort("ts"),
                kline_offset,
            ).sink_parquet(pass2a, compression="zstd")
            pass2 = pass2a

        l.info(f"uploading resampled daily klines to {alldst}...")
        await put_object(alldst, pass2)

        pass3 = join(tmp, "stable.parquet")
        l.info(f"uploading stable coin quoted daily klines to {stabledst}...")

        pl.scan_parquet(pass2).filter(
            pl.col("symbol").str.to_lowercase().str.ends_with(stable.lower())
        ).sink_parquet(pass3, compression="zstd")

        await put_object(stabledst, pass3)


async def action_list_symbols(ctx, pairs: Dict[str, Pair]):
    for symbol, pair in pairs.items():
        print(f"{symbol}: {pair.base}/{pair.quote}")


synchronize_day_sem: asyncio.Semaphore | None = None


async def synchronize_day(ctx: Context, pairs: List[str], day: date):
    dst = mkurl_1m(day.year, day.month, day.day, "")

    if synchronize_day_sem is None:
        raise RuntimeError("synchronize_day_sem is not initialized")

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
            return None

        df = pl.concat(res)
        table = df.to_arrow()
        pq.ParquetWriter(
            dst,
            table.schema,
            filesystem=ctx.r2fs,
            compression="zstd",
        ).write_table(table)

        return len(df)


async def process_single_archive(
    ctx: Context, pair: str, day: date
) -> pl.DataFrame | None:
    zip_url = mkurl_src(pair, day.year, day.month, day.day, True) + ".zip"
    chksm_url = zip_url + ".CHECKSUM"

    csv = await exponential_backoff(download_and_verify_csv, [ctx, zip_url, chksm_url])
    if csv is None:
        return None

    return await asyncio.to_thread(transform_csv_data, csv, pair)


async def bisect_bucket(
    c, horizon: date, make_prefix: Callable[[date], str], find_end=True
) -> date | None:
    low = horizon
    high = date.today() - timedelta(days=1)
    ret = None

    while low <= high:
        mid = low + timedelta(days=(high - low).days // 2)
        u = urlparse(make_prefix(mid))
        try:
            l.debug(f"Checking for object {u.netloc}/{u.path.lstrip('/')}...")
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


async def find_latest_archive(r2, horizon: date) -> date | None:
    """Finds the latest daily archive for ``pair`` between yesterday and ``horizon``."""

    return await bisect_bucket(
        r2,
        horizon,
        make_prefix=lambda d: mkurl_1m(d.year, d.month, d.day),
        find_end=True,
    )


async def find_earliest_archive(s3, pair: str, horizon: date) -> date | None:
    """Finds the earliest daily archive for ``pair`` after ``horizon``."""

    return await bisect_bucket(
        s3,
        horizon,
        make_prefix=lambda d: f"{mkurl_src(pair, d.year, d.month, d.day)}.zip",
        find_end=False,
    )


async def determine_start_date(ctx: Context, pair: str) -> date | None:
    start = await find_latest_archive(ctx.r2, date(2017, 1, 1))
    if start is None:
        return await find_earliest_archive(ctx.s3, pair, date(2016, 1, 1))
    else:
        return start + timedelta(days=1)


retrieve_klines_sem: asyncio.Semaphore | None = None


async def retrieve_klines(
    ctx: Context, pair: str, start: datetime, end: datetime
) -> pl.DataFrame | None:
    if (end - start).total_seconds() / 3600 > 1000:
        l.warn(f"more than 1000 klines: {start}-{end} ({pair})")
    if start >= end:
        raise ValueError("Start time must be before end time.")
    if retrieve_klines_sem is None:
        raise RuntimeError("retrieve_klines_sem is not initialized")
    async with retrieve_klines_sem:
        resp = await asyncio.to_thread(
            ctx.binance.klines,
            symbol=pair,
            interval="1h",
            start_time=int(start.timestamp()) * 1000,
            end_time=int(end.timestamp()) * 1000,
            limit=1000,
        )
        if resp.status != 200:
            raise TransientError(
                f"Failed to retrieve klines for {pair} from {start} to {end}: {resp.text}"
            )

        rows = resp.data()
        if len(rows) == 0:
            return None

        ret = (
            pl.DataFrame(rows, orient="row", schema=BINANCE_KLINE_SCHEMA)
            .with_columns(
                [
                    pl.when(pl.col("open_time") > EPOCH_MS_US_THRESHOLD)
                    .then(pl.from_epoch("open_time", time_unit="us"))
                    .when(pl.col("open_time") > EPOCH_S_MS_THRESHOLD)
                    .then(pl.from_epoch("open_time", time_unit="ms"))
                    .otherwise(pl.from_epoch("open_time", time_unit="s"))
                    .dt.replace_time_zone("UTC")
                    .alias("ts"),
                    pl.lit(pair).alias("symbol"),
                ]
            )
            .select(KLINE_COLUMNS)
            .sort("ts")
        )

        return ret


async def retrieve_spot_pairs(ctx: Context) -> Dict[str, Pair]:
    async with ctx.session.get(BINANCE_API_EXCHANGE_INFO) as resp:
        return {
            symbol["symbol"]: Pair(base=symbol["baseAsset"], quote=symbol["quoteAsset"])
            for symbol in (await resp.json())["symbols"]
            if symbol["isSpotTradingAllowed"]
        }


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


read_parquet_object_sem = None


async def read_parquet_object(ctx: Context, url: str, required=False) -> pl.DataFrame:
    if read_parquet_object_sem is None:
        raise RuntimeError("read_parquet_object_sem is not initialized")

    async with read_parquet_object_sem:
        try:
            bucket, key = parse_object_store_url(url)
            resp = await ctx.r2.get_object(Bucket=bucket, Key=key)
            async with resp["Body"] as body:
                data = await body.read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey" and not required:
                return None
            raise TransientError(f"Failed to read object {url}: {e}")
        except Exception as e:
            raise TransientError(f"Failed to read object {url}: {e}")

    return await asyncio.to_thread(pl.read_parquet, io.BytesIO(data))


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
            schema_overrides=BINANCE_KLINE_SCHEMA,
        )
        .with_columns(
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
                .alias("ts"),
                pl.lit(pair).alias("symbol"),
            ]
        )
        .drop("ignore")
        .sort("ts")
    )


async def resample_daily_klines(
    ctx: Context, writer, files: List[str], start_hour: int, fd
):
    files = sorted(files)

    async def w(idx, file):
        return idx, await download_and_resample_daily_klines(ctx, file, start_hour)

    fut = [w(idx, file) for idx, file in enumerate(files)]

    next_idx = 0
    results = {}
    with tqdm(
        desc=f"resampling daily klines",
        unit="file",
        position=1,
        total=len(files),
        leave=False,
    ) as bar:
        for f in asyncio.as_completed(fut):
            idx, df = await f
            results[idx] = df
            bar.n = len(results)
            bar.refresh()

            while next_idx in results:
                df = results[next_idx]
                results[next_idx] = None
                next_idx += 1

                if df is None:
                    l.warning(f"file {files[next_idx-1]} is missing, skipping.")
                    continue

                table = df.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(
                        fd,
                        table.schema,
                        compression="zstd",
                    )
                writer.write_table(table)
    return writer


def resample_sorted_dataframe(df: pl.DataFrame, start_hour: int) -> pl.DataFrame:
    return df.group_by_dynamic(
        "ts", every="1d", offset=f"{start_hour}h", group_by="symbol"
    ).agg(
        [
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("base_volume").sum().alias("base_volume"),
            pl.col("quote_volume").sum().alias("quote_volume"),
            pl.col("close_time").max().alias("close_time"),
            pl.col("trades").sum().alias("trades"),
            pl.col("taker_buy_base_volume").sum().alias("taker_buy_base_volume"),
            pl.col("taker_buy_quote_volume").sum().alias("taker_buy_quote_volume"),
        ]
    )


async def download_and_resample_daily_klines(
    ctx, src: str, start_hour: int
) -> pl.DataFrame:
    df = await exponential_backoff(read_parquet_object, [ctx, src], retries=5)
    if df is None:
        return None

    return await asyncio.to_thread(
        lambda df: resample_sorted_dataframe(df.sort("ts"), start_hour),
        df,
    )


async def exponential_backoff(
    fn: Callable,
    args: List,
    retries: int | None = None,
    base_delay: float = 1.0,
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


extract_object_partitioned_sem: asyncio.Semaphore | None = None


async def download_and_verify_csv(
    ctx: Context, zip_url: str, chksm_url: str | None
) -> bytes | None:
    if extract_object_partitioned_sem is None:
        raise RuntimeError("extract_object_partitioned_sem is not initialized")

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


if __name__ == "__main__":
    with logging_redirect_tqdm():
        try:
            asyncio.run(main())
        except Exception as e:
            l.exception("Fatal error during sync", exc_info=e)
            sys.exit(1)
