import polars as pl
from tqdm import tqdm
import datetime as dt
import numpy as np

import asyncio
import aiohttp
from tqdm.asyncio import tqdm
from typing import Tuple

import logging as l
import sys

l.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=l.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


async def single(sess, date: dt.datetime, sem) -> Tuple[dt.datetime, np.ndarray] | None:
    url = f"""http://vixcentral.com/ajax_historical?n1={date.strftime("%Y-%m-%d")}"""
    h = {
        "Referer": "http://vixcentral.com/historical?days=30",
        "X-Requested-With": "XMLHttpRequest",
    }

    async with sem:
        async with sess.get(url, headers=h) as resp:
            if resp.status != 200:
                l.warning(
                    f"Failed to fetch data for {date}, status code: {resp.status}"
                )
                return None
            data = await resp.json()

    if type(data) == str:
        l.warning(f"Unexpected string response for {date}: {data}, skipping")
        return None

    if len(data) == 0:
        l.warning(f"No data for {date}, skipping")
        return None

    # front_month = data[0]
    # pad up to 9 months with 0
    data = [*data[1:], *([0] * (9 - len(data[1:])))]
    return (
        date,
        np.array([float(m) if m > 0.0 else np.nan for m in data]),
    )


async def vix_download(output: str, concurrency: int = 5):
    import holidays

    sem = asyncio.Semaphore(concurrency)
    start_date = dt.datetime(2010, 1, 2)
    end_date = dt.datetime.now()
    us_holidays = holidays.US()
    all_dates = [
        start_date + dt.timedelta(days=i)
        for i in range((end_date - start_date).days + 1)
    ]
    trading_dates = [d for d in all_dates if d.weekday() < 5 and d not in us_holidays]

    async with aiohttp.ClientSession() as sess:
        fut = [single(sess, d, sem) for d in trading_dates]
        res = await tqdm.gather(*fut, desc="downloading data", unit="day")

    rows = [[r[0], *r[1]] for r in res if r is not None]
    df = pl.DataFrame(
        rows,
        orient="row",
        schema={"ts": pl.Datetime, **{f"m{i}": pl.Float32 for i in range(1, 10)}},
    )

    df.write_parquet(output, compression="zstd")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--output", help="Output parquet file path", default="output.parquet"
    )
    p.add_argument(
        "--concurrency",
        help="Number of concurrent requests to make",
        type=int,
        default=5,
    )
    args = p.parse_intermixed_args()

    try:
        asyncio.run(vix_download(args.output, args.concurrency))
    except Exception as e:
        l.error(f"Error downloading data: {e}")
        sys.exit(1)
