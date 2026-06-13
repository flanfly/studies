from typing import Tuple, Dict, List, Any
import time
import os
from dataclasses import dataclass

import polars as pl

import argparse

import asyncio
from httpx import AsyncClient
from pydantic import BaseModel
from playwright.async_api import Playwright

import dotenv  # type: ignore

dotenv.load_dotenv()

import logging as l
import sys

l.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=l.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)


class PolarityResponse(BaseModel):
    status: int
    message: str
    data: dict[str, Any] | List[dict[str, Any]]


@dataclass(frozen=True)
class CoinMetadata:
    asset: str
    """ticker"""
    coingecko_slug: str
    metrics: set[str]


class Unrecoverable(Exception):
    """
    An error indicating that the operation cannot be retried.
    """

    def __init__(self, message="Unrecoverable error"):
        self.message = message
        super().__init__(self.message)


async def pd_metrics(c: AsyncClient) -> dict[str, Any]:
    resp = await c.get("https://api.polaritydigital.io/api/metrics")
    resp.raise_for_status()
    model = PolarityResponse(**resp.json())

    if model.status != 1:
        raise Unrecoverable(f"pd_metrics: {model.message}")
    if not isinstance(model.data, dict):
        raise Unrecoverable(f"pd_metrics: unrecognizable response {model.data}")

    return model.data


async def pd_dashboard_data(c: AsyncClient, idtoken: str) -> dict[str, Any]:
    headers = {"authorization": f"Bearer {idtoken}"}
    resp = await c.get(
        "https://api.polaritydigital.io/api/dashboardData", headers=headers
    )
    resp.raise_for_status()
    model = PolarityResponse(**resp.json())

    if model.status != 1:
        raise Unrecoverable(f"pd_dashboard_data: {model.message}")
    if not isinstance(model.data, dict):
        raise Unrecoverable(f"pd_dashboard_data: unrecognizable response {model.data}")

    return model.data


async def pd_historical_data(
    c: AsyncClient, asset: str, metric: str, idtoken: str
) -> pl.DataFrame | None:
    headers = {"authorization": f"Bearer {idtoken}"}
    params = {"coin": asset, "metric": metric}

    resp = await c.get(
        "https://api.polaritydigital.io/api/historicalData",
        headers=headers,
        params=params,
    )
    if resp.status_code == 401 and "rate limit exceeded" in resp.text.lower():
        raise Exception(f"Rate limit exceeded: {resp.text}")
    resp.raise_for_status()

    model = PolarityResponse(**resp.json())
    if model.status != 1:
        raise Exception(f"pd_historical_data: {model.message}")
    if not isinstance(model.data, list):
        raise Unrecoverable(f"pd_historical_data: unrecognizable response {model.data}")

    if len(model.data) == 0:
        return None

    return (
        pl.DataFrame(
            model.data,
            schema={
                "closetime": pl.String,
                "closeprice": pl.Float64,
            },
        )
        .select(
            [
                pl.col("closetime")
                .str.to_datetime()
                .dt.replace_time_zone("UTC")
                .alias("ts"),
                pl.lit(asset).alias("asset"),
                pl.lit(metric).alias("metric"),
                pl.col("closeprice").cast(pl.Float64).alias("value"),
            ]
        )
        .unique(subset=["ts"], keep="last", maintain_order=False)
    )


async def pd_get_subscription(c: AsyncClient, idtoken: str) -> bool:
    if not idtoken:
        l.error("ID token is required")
        return False

    headers = {"authorization": "Bearer " + idtoken}
    resp = await c.get(
        "https://api.polaritydigital.io/api/getSubscription",
        headers=headers,
        timeout=10,
    )

    if resp.status_code == 401:
        l.error("pd_get_subscription: invalid token")
        return False
    resp.raise_for_status()
    model = PolarityResponse(**resp.json())

    if model.status != 1 or model.data.get("id") is None:
        return False

    return True


async def coin_metadata(c: AsyncClient, idtoken: str) -> dict[str, CoinMetadata]:
    metrics, dd = await asyncio.gather(pd_metrics(c), pd_dashboard_data(c, idtoken))

    metrics_by_asset: dict[str, set[str]] = {}
    for m in metrics["allDashboardMetrics"]:
        if not m.get("show_on_workbench", False):
            continue
        for c in m.get("coins", []):
            metrics_by_asset.setdefault(c, set()).add(m["key"])

    ret: dict[str, CoinMetadata] = {}
    for r in dd.get("data", []):
        sym = r.get("symbol", "(unknown symbol)")
        if not sym in metrics_by_asset:
            l.warning("coin_metadata: {sym} not in metric set")
            continue
        ret[sym] = CoinMetadata(
            asset=sym, coingecko_slug=r.get("coin_id"), metrics=metrics_by_asset[sym]
        )

    return ret


coin_metric_sem: asyncio.Semaphore | None = None


async def coin_metric(
    c: AsyncClient, asset: str, metric: str, idtoken: str
) -> pl.DataFrame | None:
    from math import exp

    if coin_metric_sem is None:
        raise Unrecoverable("coin_metric: semaphore not initialized")

    delay = 0
    while True:
        try:
            async with coin_metric_sem:
                df = await pd_historical_data(c, asset, metric, idtoken)
            return df
        except Unrecoverable as e:
            l.error(f"coin_metric: {e}")
            return None
        except Exception as e:
            s = min(exp(delay), 60)
            l.warning(f"coin_metric got {e}, wait {s}s")
            await asyncio.sleep(s)
            delay += 1


async def retrieve_coins(
    c: AsyncClient, idtoken: str, output: str, max_coins: int, parallelism: int
) -> None:
    from tempfile import NamedTemporaryFile

    global coin_metric_sem
    coin_metric_sem = asyncio.Semaphore(parallelism)

    l.info("fetching available metrics...")
    coin_meta = await coin_metadata(c, idtoken)

    schema = {
        "ts": pl.Datetime("ns"),
        "asset": pl.String,
        "coingecko_slug": pl.String,
        "price": pl.Float64,
        "market_cap": pl.Float64,
        "udpil": pl.Float64,
        "udpim": pl.Float64,
        "udpis": pl.Float64,
        "mdccv": pl.Float64,
        "mbi": pl.Float64,
        "tci": pl.Float64,
        "tcicv": pl.Float64,
        "upprob": pl.Float64,
        "total_volume": pl.Float64,
        "mean_realized_price_usd_7d": pl.Float64,
        "mean_realized_price_usd_14d": pl.Float64,
        "mean_realized_price_usd_30d": pl.Float64,
        "mean_realized_price_usd_180d": pl.Float64,
    }

    whitelist = [k for k in schema.keys() if k not in ["ts", "asset", "coingecko_slug"]]

    l.info(f"writing to {output}")

    if max_coins and max_coins > 0:
        coin_meta = {k: v for k, v in list(coin_meta.items())[:max_coins]}

    l.info(f"processing {len(coin_meta)} coins")

    with NamedTemporaryFile(delete=False) as fd:
        from tqdm import tqdm  # type: ignore
        from itertools import chain

        pairs = list(
            chain.from_iterable(
                [
                    [
                        [m.asset, m.coingecko_slug, mm]
                        for mm in m.metrics
                        if mm in whitelist
                    ]
                    for m in coin_meta.values()
                ]
            )
        )
        writer = None
        with tqdm(desc="Polarity Digital metrics", total=len(pairs)) as bar:
            import more_itertools as it
            import pyarrow.parquet as pq

            for batch in it.batched(pairs, parallelism * 2):
                fut = [coin_metric(c, i[0], i[2], idtoken) for i in batch]
                res = [df for df in await asyncio.gather(*fut)]
                slugs = [i[1] for i in batch]

                bar.update(len(fut))
                if len(res) == 0:
                    l.warning(f"main: no results in batch")
                    continue

                df = pl.concat(
                    [
                        df.with_columns(coingecko_slug=pl.lit(slug))
                        for df, slug in zip(res, slugs)
                        if df is not None
                    ]
                )
                table = df.to_arrow()

                if writer is None:
                    writer = pq.ParquetWriter(
                        fd,
                        table.schema,
                        compression="zstd",
                    )

                writer.write_table(table)
                del df
                df = None

        if writer is not None:
            writer.close()
        fd.flush()

        (
            pl.scan_parquet(fd.name)
            .pivot(
                on="metric",
                index=["ts", "asset", "coingecko_slug"],
                values="value",
                on_columns=whitelist,
            )
            .sink_parquet(output)
        )


async def browser_login(p: Playwright, datadir: str) -> str | None:
    from playwright.async_api import TimeoutError

    idtoken = None

    def intercept_request(request):
        nonlocal idtoken

        if "api.polaritydigital.io/api/getSubscription" in request.url:
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                idtoken = auth_header[7:]

    ctx = await p.chromium.launch_persistent_context(
        user_data_dir=datadir,
        channel="chrome",
        headless=False,
        args=["--disable-blink-features=AutomationControlled"],
    )

    page = await ctx.new_page()
    page.on("request", intercept_request)

    await page.goto("https://www.polaritydigital.io/")

    for _ in range(2):
        await page.wait_for_load_state("networkidle")

        try:
            await page.locator(
                "div.ant-space-item:has-text('kai.h.michaelis')"
            ).first.wait_for(timeout=5000)

        except TimeoutError:
            import re

            await page.get_by_role("button", name="Login").click()
            await page.locator("text=Sign in with Google").click()

    return idtoken


async def main() -> None:
    p = argparse.ArgumentParser(
        description="Update Polarity Digital metrics and store in HDF5 file"
    )
    p.add_argument(
        "--idtoken",
        type=str,
        default=os.environ.get("POLARITY_IDTOKEN", ""),
        help="Polarity Digital ID token (or set POLARITY_IDTOKEN env variable)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="polarity.parquet",
        help="Output file name",
    )
    p.add_argument(
        "--parallelism",
        type=int,
        default=4,
        help="Number of parallel requests to make",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="File to write logs to (if not set, logs to stdout)",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    p.add_argument(
        "--max-coins",
        type=int,
        default=0,
        help="Limit to first N coins (0 = all)",
    )
    args = p.parse_args()

    l.basicConfig(
        level=getattr(l, args.log_level.upper(), l.INFO),
        filename=args.log_file,
    )

    from platformdirs import user_cache_dir, user_cache_path

    cache_path = user_cache_path("polarity.py", ensure_exists=True)
    idtoken_path = cache_path / "idtoken"
    browser_data_path = cache_path / "browser"

    async with AsyncClient() as c:
        idtoken = args.idtoken

        if not await pd_get_subscription(c, idtoken):
            try:
                idtoken = idtoken_path.read_text()
            except FileNotFoundError:
                pass

            if not await pd_get_subscription(c, idtoken):
                from playwright.async_api import async_playwright

                async with async_playwright() as p:
                    idtoken = await browser_login(p, browser_data_path.as_posix())

                if idtoken is not None:
                    idtoken_path.write_text(idtoken)

            if not await pd_get_subscription(c, idtoken):
                raise Unrecoverable("no idtoken")

        await retrieve_coins(c, idtoken, args.output, args.max_coins, args.parallelism)


if __name__ == "__main__":
    from tqdm.contrib.logging import logging_redirect_tqdm  # type: ignore

    with logging_redirect_tqdm():
        try:
            asyncio.run(main())
        except Exception as e:
            l.exception("Fatal error during sync", exc_info=e)
            sys.exit(1)
