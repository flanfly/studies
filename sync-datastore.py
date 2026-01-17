import os

# import itertools as it
# import posixpath
# import asyncio
# from asyncio import gather, Semaphore
# import aiohttp
# import aiobotocore.session
import botocore
import botocore.config as botoconfig

# from collections import namedtuple
# from typing import Dict, Tuple, Iterable
# from datetime import datetime
# from urllib.parse import urlparse
# from tqdm.asyncio import tqdm
# import requests

from dotenv import load_dotenv

load_dotenv()

# from typing import Tuple, Dict, Iterable

BINANCE_API_EXCHANGE_INFO = "https://api.binance.com/api/v3/exchangeInfo"
BINANCE_VISION_DAILY_SPOT_ARCHIVE = (
    "s3://data.binance.vision/data/spot/daily/klines/%s/1m"
)

R2_ENDPOINT_URL = os.environ.get(
    "R2_ENDPOINT_URL", "https://<ACCOUNT_ID>.r2.cloudflarestorage.com"
)
DATASTORE_BUCKET = f"r2://studies-binance-archive/spot-1m/%s"

# S3_REQ_SEMAPHORE = Semaphore(25)
#
#
# async def main():
#    await tqdm.gather(
#        *map(
#            lambda pair: sync_s3_and_r2(
#                BINANCE_VISION_DAILY_SPOT_ARCHIVE % pair,
#                posixpath.join(DATASTORE_BUCKET, pair),
#            ),
#            spot_pairs().keys(),
#        )
#    )
#
#
# def spot_pairs() -> Dict[str, Tuple[str, str]]:
#    resp = requests.get(BINANCE_API_EXCHANGE_INFO)
#    resp.raise_for_status()
#
#    data = resp.json()
#
#    pairs = dict()
#    for s in data["symbols"]:
#        if s["isSpotTradingAllowed"] is not True:
#            continue
#
#        pairs[s["symbol"]] = (s["baseAsset"], s["quoteAsset"])
#
#    return pairs
#
#
# async def sync_s3_and_r2(s3str: str, r2str: str):
#    s3url = urlparse(s3str)
#    r2url = urlparse(r2str)
#
#    if s3url.scheme != "s3":
#        raise ValueError("URL must start with s3://")
#    if r2url.scheme != "r2":
#        raise ValueError("URL must start with r2:// for R2")
#
#    srcbkt = s3url.netloc
#    srcprefix = s3url.path.lstrip("/")
#
#    dstbkt = r2url.netloc
#    dstprefix = r2url.path.lstrip("/")
#
#    sess = aiobotocore.session.get_session()
#    async with sess.create_client(
#        "s3", config=botoconfig.Config(signature_version=botocore.UNSIGNED)
#    ) as s3, sess.create_client(
#        "s3",
#        aws_access_key_id=os.getenv("R2_ACCESS_KEY"),
#        aws_secret_access_key=os.getenv("R2_SECRET_KEY"),
#        endpoint_url=f"""https://{os.getenv("R2_ACCOUNT_ID")}.r2.cloudflarestorage.com""",
#        region_name="auto",
#    ) as r2:
#        s3keys, r2keys = await gather(
#            catalog_s3_objects(s3, srcbkt, srcprefix),
#            catalog_s3_objects(r2, dstbkt, dstprefix),
#        )
#
#        seen = set()
#        for k in s3keys.keys():
#            bn = os.path.basename(k)
#            if bn in seen:
#                raise ValueError(f"Duplicate S3 object basename detected: {bn}")
#            seen.add(bn)
#
#        Obj = namedtuple("Obj", ["key", "updated"])
#        s3objs = {os.path.basename(k): Obj(k, v) for k, v in s3keys.items()}
#        r2objs = {os.path.basename(k): Obj(k, v) for k, v in r2keys.items()}
#
#        missing = set(s3objs.keys()) - set(r2objs.keys())
#        outdated = filter(
#            lambda k: s3objs[k].updated > r2objs[k].updated,
#            set(s3objs.keys()).intersection(set(r2objs.keys())),
#        )
#        superfluous = set(r2objs.keys()) - set(s3objs.keys())
#
#        await gather(
#            remove_r2_objects(r2, dstbkt, map(lambda f: r2objs[f].key, superfluous)),
#            copy_s3_object(
#                s3,
#                r2,
#                srcbkt,
#                dstbkt,
#                dstprefix,
#                map(lambda f: s3objs[f].key, missing | set(outdated)),
#            ),
#        )
#
#
# async def remove_r2_objects(r2, dstbkt: str, keys: Iterable[str]):
#    for batch in it.batched(tqdm(list(keys), "delete superfluous file"), 1000):
#        with S3_REQ_SEMAPHORE:
#            print(f"Deleting r2://{dstbkt}/{{{', '.join(batch)}}}")
#            await r2.delete_objects(
#                Bucket=dstbkt, Delete={"Objects": [{"Key": k} for k in batch]}
#            )
#
#
# async def copy_s3_object(
#    s3, r2, srcbkt: str, dstbkt: str, dstprefix: str, keys: Iterable[str]
# ):
#    srcurls = dict(
#        map(
#            lambda key: (
#                key,
#                [
#                    f"https://s3.amazonaws.com/{srcbkt}/{key}",
#                    posixpath.join(dstprefix, os.path.basename(key)),
#                ],
#            ),
#            keys,
#        )
#    )
#
#    for key, (s3url, r2key) in tqdm(srcurls.items(), "copy missing/outdated file"):
#        with S3_REQ_SEMAPHORE:
#            print(f"Copying {s3url} to r2://{dstbkt}/{r2key}")
#            try:
#                async with aiohttp.ClientSession() as sess:
#                    async with sess.get(s3url) as resp:
#                        resp.raise_for_status()
#                        await r2.put_object(
#                            Bucket=dstbkt,
#                            Key=r2key,
#                            Body=await resp.read(),  # For small files (klines), reading into RAM is faster/safer
#                            ContentType=resp.headers.get("Content-Type"),
#                        )
#            except Exception as e:
#                print(f"Failed to copy {s3url}: {e}")
#
#
# async def catalog_s3_objects(client, bucket: str, prefix: str) -> Dict[str, datetime]:
#    paginator = client.get_paginator("list_objects_v2")
#    obj_map = {}
#
#    async with S3_REQ_SEMAPHORE:
#        async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
#            for obj in page.get("Contents", []):
#                if not obj["Key"].endswith("/"):
#                    obj_map[obj["Key"]] = obj["LastModified"]
#    return obj_map

import asyncio
import aiohttp
from aiobotocore.session import get_session as boto_session
from os.path import basename
import logging as log
import more_itertools as it
from urllib.parse import urlparse
from tqdm.asyncio import tqdm

from typing import Dict, Iterable
from collections import namedtuple

Pair = namedtuple("Pair", ["base", "quote"])
Obj = namedtuple("Obj", ["key", "last_modified"])

BINANCE_EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"

log.basicConfig(level=log.INFO)


def new_r2(sess):
    return sess.create_client(
        "s3",
        aws_access_key_id=os.getenv("R2_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("R2_SECRET_KEY"),
        endpoint_url=f"""https://{os.getenv("R2_ACCOUNT_ID")}.r2.cloudflarestorage.com""",
        region_name="auto",
    )


def new_s3(sess):
    return sess.create_client(
        "s3", config=botoconfig.Config(signature_version=botocore.UNSIGNED)
    )


async def main():
    pairs = await retrieve_spot_pairs()

    sess = boto_session()
    async with new_s3(sess) as s3, new_r2(sess) as r2:
        for pair in pairs.keys():
            log.info(f"Syncing pair {pair}")
            await sync_archive_for_prefix(
                s3,
                r2,
                BINANCE_VISION_DAILY_SPOT_ARCHIVE % pair,
                DATASTORE_BUCKET % pair,
            )


async def retrieve_spot_pairs() -> Dict[str, Pair]:
    async with aiohttp.ClientSession() as sess:
        async with sess.get(BINANCE_EXCHANGE_INFO_URL) as resp:
            return {
                symbol["symbol"]: Pair(
                    base=symbol["baseAsset"], quote=symbol["quoteAsset"]
                )
                for symbol in (await resp.json())["symbols"]
                if symbol["isSpotTradingAllowed"]
            }


async def catalog_bucket(c, bucket: str, prefix: str) -> Dict[str, Obj]:
    ret: Dict[str, Obj] = {}

    pg = c.get_paginator("list_objects_v2")
    async for page in pg.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            bn = basename(key)
            if bn.startswith("."):
                continue
            if bn in ret:
                log.warn(f"Dup archive: {ret[bn].key}, {key})")
                continue
            ret[bn] = Obj(key=key, last_modified=obj["LastModified"])

    return ret


async def sync_archive_for_prefix(s3, r2, src: str, dst: str):
    srcurl = urlparse(src)
    dsturl = urlparse(dst)

    if srcurl.scheme != "s3":
        raise ValueError("source URL must start with s3://")
    if dsturl.scheme != "r2":
        raise ValueError("destination URL must start with r2://")

    srcbkt = srcurl.netloc
    srcprefix = srcurl.path.lstrip("/")

    dstbkt = dsturl.netloc
    dstprefix = dsturl.path.lstrip("/")

    srcobjs, dstobjs = await asyncio.gather(
        catalog_bucket(s3, srcbkt, srcprefix), catalog_bucket(r2, dstbkt, dstprefix)
    )

    missing = set(srcobjs.keys()) - set(dstobjs.keys())
    superfluous = set(dstobjs.keys()) - set(srcobjs.keys())
    outdated = {
        k
        for k in (set(srcobjs.keys()) & set(dstobjs.keys()))
        if srcobjs[k].last_modified > dstobjs[k].last_modified
    }

    await asyncio.gather(
        copy_objects(
            s3,
            r2,
            srcbkt,
            dstbkt,
            dstprefix,
            map(lambda k: srcobjs[k], missing | outdated),
        ),
        delete_objects(r2, dstbkt, map(lambda k: dstobjs[k], superfluous)),
    )


async def delete_objects(r2, dstbkt, objs: Iterable[Obj]):
    fut = [
        r2.delete_objects(
            Bucket=dstbkt,
            Delete={"Objects": [{"Key": o.key} for o in b], "Quiet": True},
        )
        for b in it.batched(objs, 1000)
    ]
    await asyncio.gather(*fut)


copy_objects_sem = asyncio.Semaphore(20)


async def copy_objects(s3, r2, srcbkt, dstbkt, dstprefix, objs: Iterable[Obj]):
    async with aiohttp.ClientSession() as sess:

        async def _do(s3, r2, sess, srcbkt, dstbkt, obj: Obj):
            async with copy_objects_sem:
                url = f"https://s3.amazonaws.com/{srcbkt}/{obj.key}"
                async with sess.get(url) as resp:
                    await r2.put_object(
                        Bucket=dstbkt,
                        Key=os.path.join(dstprefix, obj.key),
                        Body=await resp.read(),
                    )

        objs = list(objs)
        fut = [_do(s3, r2, sess, srcbkt, dstbkt, obj) for obj in objs]
        await tqdm.gather(
            *fut,
            total=len(objs),
            desc=f"copying objects from {basename(srcbkt)} to {os.path.join(dstbkt, dstprefix)}",
        )


def download_daily_archive():
    retry = False

    sync_s3()
    seen = verify_and_extract_daily_archives()

    con = duckdb.connect("binance-1m-spot", hive_partitioning=True)
    todo = con.sql(
        """
        all ts, symbol missing from ?
    """,
        seen,
    )

    for year in todo:
        pl.concat(todo[year]).sort(["ts", "symbol"]).write_parquet(
            f"binance-1m-spot/year={year}/data-{ord}.parquet"
        )

    compact_1m_partitions()


def derive_1d_klines():
    for year in s3.ls_dirs("binance-1m-spot/"):
        s3.rm(f"binance-1d-spot/year={year}/")

        check_its_sorted("binance-1m-spot/year={year}/*.parquet", ["ts", "symbol"])
        pl.read_parquet(f"binance-1m-spot/year={year}/*.parquet").with_columns(
            [(pl.col("ts").dt.truncate("1d")).alias("ts_1d")]
        ).groupby(["ts_1d", "symbol"]).agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ]
        ).sort(
            ["ts_1d", "symbol"]
        ).write_parquet(
            f"binance-1d-spot/year={year}/data.parquet"
        )


def verify_and_extract_daily_archives():
    seen = dict()

    for zip_file in s3.ls("*.zip"):
        try:
            symbol, date = re.match(
                r"^(.*)_(\d{4}-\d{2}-\d{2})\.zip$", zip_file
            ).groups()
            if symbol not in seen:
                seen[symbol] = [date]
            else:
                seen[symbol].append(date)
        except Exception as e:
            l.warn(f"Failed to parse filename {zip_file}: {e}")
            continue

        if not s3.exists(zip_file.replace(".zip", ".csv")):
            if s3.exists(zip_file.replace(".zip", ".CHECKSUM")):
                if s3.read(zip_file.replace(".zip", ".CHECKSUM")) != s3.sha256(
                    zip_file
                ):
                    l.info(f"Archive {zip_file} checksum mismatch, re-extracting.")
                    s3.remove(zip_file)
                    raise TransientError("Checksum mismatch, re-download required.")
            else:
                l.warn(f"Archive {zip_file} missing checksum file, re-extracting.")
            s3.write(zip.unpack(zip_file), zip_file.replace(".zip", ".csv"))
            l.info(f"Extracted archive {zip_file}.")

    return seen


if __name__ == "__main__":
    asyncio.run(main())
