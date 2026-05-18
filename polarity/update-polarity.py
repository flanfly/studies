from typing import Tuple, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging as l
import os
import unittest

import polars as pl

import requests  # type: ignore
import dotenv  # type: ignore
import argparse
from tqdm import tqdm  # type: ignore
from tqdm.contrib.logging import logging_redirect_tqdm  # type: ignore


class Unrecoverable(Exception):
    """
    An error indicating that the operation cannot be retried.
    """

    def __init__(self, message="Unrecoverable error"):
        self.message = message
        super().__init__(self.message)


def available_metrics() -> dict[str, set[str]]:
    resp = requests.get("https://api.polaritydigital.io/api/metrics")
    body = resp.json()

    if resp.status_code != requests.codes.ok:
        raise Exception(
            "Error fetching metrics: %s" % body.get("message", "Unknown error")
        )
    if "status" in body and body["status"] != 1:
        raise Exception(
            "Error fetching metrics: %s" % body.get("message", "Unknown error")
        )

    data = body["data"]

    ret: dict[str, set[str]] = {}
    for metric in data["allDashboardMetrics"]:
        if not metric.get("show_on_workbench", False):
            continue
        for coin in metric.get("coins", []):
            ret.setdefault(coin, set()).add(metric["key"])

    return ret


def get_data(asset: str, metric: str, idtoken: str):
    headers = {"authorization": "Bearer " + idtoken}
    params = {"coin": asset, "metric": metric}

    resp = requests.get(
        "https://api.polaritydigital.io/api/historicalData",
        headers=headers,
        params=params,
        timeout=30,
    )

    if resp.status_code == 401 and "rate limit exceeded" in resp.text.lower():
        raise Exception(f"Rate limit exceeded: {resp.text}")
    if resp.status_code == 401 or resp.status_code == 404:
        raise Unrecoverable(f"Client error {resp.status_code}: {resp.text}.")
    if resp.status_code != requests.codes.ok:
        raise Exception(f"HTTP {resp.status_code}: {resp.text}")
    body = resp.json()
    if "status" in body and body["status"] != 1:
        raise Exception(
            "Error fetching data: %s" % body.get("message", "Unknown error")
        )

    l.debug(f"response body: {body}")

    timestamps = []
    values = []
    for row in body["data"]:
        timestamps.append(row["closetime"])
        values.append(
            float(row["closeprice"]) if row["closeprice"] is not None else None
        )

    if not timestamps:
        return pl.DataFrame(schema={"timestamp": pl.Utf8, metric: pl.Float64})

    return (
        pl.DataFrame(
            {"timestamp": timestamps, metric: pl.Series(values, dtype=pl.Float64)}
        )
        .with_columns(pl.col("timestamp").str.to_datetime())
        .unique(subset=["timestamp"], keep="last", maintain_order=False)
    )


def save(df: pl.DataFrame, schema: Dict[str, pl.DataType], coin: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    p = os.path.join(output_dir, f"{coin.lower()}.parquet")
    l.info(f"saving data to {p}...")

    df = df.filter(pl.col("asset") == coin.lower())

    # Add missing metric columns as null, cast existing to schema types
    metric_schema = {c: t for c, t in schema.items() if c not in ("timestamp", "asset")}
    for col, ty in metric_schema.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(ty).alias(col))
        else:
            df = df.with_columns(pl.col(col).cast(ty))

    select_exprs = {
        "ts": pl.col("timestamp").dt.cast_time_unit("us"),
        "asset": pl.col("asset"),
        **{c: pl.col(c) for c in metric_schema},
    }
    df.select(**select_exprs).write_parquet(p)


def do_work(item: Tuple[str, str], idtoken: str) -> Tuple[str, str, pl.DataFrame]:
    coin, metric = item
    asset = coin.lower()

    l.info(f"fetching {asset} {metric}...")

    wait = 0
    while True:
        try:
            df = get_data(asset, metric, idtoken)
            l.debug(f"fetched {len(df)} rows for {asset} {metric}")
            df = df.with_columns(pl.lit(asset).alias("asset"))

            return (coin, metric, df)

        except Unrecoverable as e:
            l.error(f"unrecoverable error fetching {asset} {metric}: {e}")
            return (coin, metric, pl.DataFrame())

        except Exception as e:
            l.error(f"error fetching {asset} {metric}: {e}")
            w = min(2**wait, 30)
            l.info(f"waiting {w} seconds before retrying...")
            time.sleep(w)
            wait += 1


class TestMergeMetrics(unittest.TestCase):
    def test_merge_same_timestamps(self):
        """Two metrics for the same coin with identical timestamp ranges."""
        df1 = pl.DataFrame(
            {
                "timestamp": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "asset": ["btc", "btc", "btc"],
                "m1": [1.0, 2.0, 3.0],
            }
        ).with_columns(pl.col("timestamp").str.to_datetime())

        df2 = pl.DataFrame(
            {
                "timestamp": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "asset": ["btc", "btc", "btc"],
                "m2": [4.0, 5.0, 6.0],
            }
        ).with_columns(pl.col("timestamp").str.to_datetime())

        merged = merge_metrics([df1, df2])
        expected = (
            pl.DataFrame(
                {
                    "timestamp": ["2023-01-01", "2023-01-02", "2023-01-03"],
                    "asset": ["btc", "btc", "btc"],
                    "m1": [1.0, 2.0, 3.0],
                    "m2": [4.0, 5.0, 6.0],
                }
            )
            .with_columns(pl.col("timestamp").str.to_datetime())
            .sort("timestamp")
        )
        self.assertTrue(merged.equals(expected))

    def test_merge_overlapping_timestamps(self):
        """Metrics with partially overlapping timestamp ranges."""
        df1 = pl.DataFrame(
            {
                "timestamp": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "asset": ["btc", "btc", "btc"],
                "m1": [1.0, 2.0, 3.0],
            }
        ).with_columns(pl.col("timestamp").str.to_datetime())

        df2 = pl.DataFrame(
            {
                "timestamp": ["2023-01-02", "2023-01-03", "2023-01-04"],
                "asset": ["btc", "btc", "btc"],
                "m2": [10.0, 20.0, 30.0],
            }
        ).with_columns(pl.col("timestamp").str.to_datetime())

        merged = merge_metrics([df1, df2])
        expected = (
            pl.DataFrame(
                {
                    "timestamp": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
                    "asset": ["btc", "btc", "btc", "btc"],
                    "m1": [1.0, 2.0, 3.0, None],
                    "m2": [None, 10.0, 20.0, 30.0],
                }
            )
            .with_columns(pl.col("timestamp").str.to_datetime())
            .sort("timestamp")
        )
        self.assertTrue(merged.equals(expected))

    def test_merge_three_metrics(self):
        """Three metrics with staggered timestamp ranges."""
        df1 = pl.DataFrame(
            {
                "timestamp": ["2023-01-01", "2023-01-02"],
                "asset": ["btc", "btc"],
                "m1": [1.0, 2.0],
            }
        ).with_columns(pl.col("timestamp").str.to_datetime())

        df2 = pl.DataFrame(
            {
                "timestamp": ["2023-01-02", "2023-01-03"],
                "asset": ["btc", "btc"],
                "m2": [10.0, 20.0],
            }
        ).with_columns(pl.col("timestamp").str.to_datetime())

        df3 = pl.DataFrame(
            {
                "timestamp": ["2023-01-03", "2023-01-04"],
                "asset": ["btc", "btc"],
                "m3": [100.0, 200.0],
            }
        ).with_columns(pl.col("timestamp").str.to_datetime())

        merged = merge_metrics([df1, df2, df3])
        self.assertEqual(merged.height, 4)
        self.assertEqual(set(merged.columns), {"timestamp", "asset", "m1", "m2", "m3"})
        # No duplicate timestamps
        dupes = merged.group_by("timestamp").len().filter(pl.col("len") > 1)
        self.assertEqual(dupes.height, 0)


def merge_metrics(dfs: List[pl.DataFrame]) -> pl.DataFrame:
    """Merge a list of metric DataFrames for the SAME coin, aligned on timestamp."""
    if not dfs:
        raise ValueError("empty dfs list")

    # All dfs for the same coin — drop asset from subsequent dfs (avoid suffix),
    # join on timestamp only, then backfill null asset values.
    base = dfs[0]
    for df in dfs[1:]:
        df = df.drop("asset")
        base = base.join(df, on="timestamp", how="full", coalesce=True)

    # Rows that only exist in subsequent dfs have null asset — fill from the known asset.
    non_null = base["asset"].drop_nulls()
    if len(non_null) > 0:
        base = base.with_columns(pl.col("asset").fill_null(non_null[0]))

    return base.sort("timestamp")


def verify_token(idtoken: str) -> bool:
    if not idtoken:
        l.error("ID token is required")
        return False

    headers = {"authorization": "Bearer " + idtoken}
    resp = requests.get(
        "https://api.polaritydigital.io/api/getSubscription",
        headers=headers,
        timeout=10,
    )

    if resp.status_code == 401:
        l.error("Unauthorized: Invalid ID token")
        return False
    if resp.status_code != requests.codes.ok:
        l.error(f"HTTP {resp.status_code}: {resp.text}")
        return False

    body = resp.json()
    print(body)
    if "status" in body and body["status"] != 1:
        l.error("Error verifying token: %s" % body.get("message", "Unknown error"))
        return False
    if "data" not in body or "id" not in body["data"]:
        l.error("No subscription data found in response")
        return False

    return True


def main():
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(
        description="Update Polarity Digital metrics and store in HDF5 file"
    )
    parser.add_argument(
        "--idtoken",
        type=str,
        default=os.environ.get("POLARITY_IDTOKEN", ""),
        help="Polarity Digital ID token (or set POLARITY_IDTOKEN env variable)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory to store data files",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=4,
        help="Number of parallel requests to make",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="File to write logs to (if not set, logs to stdout)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--max-coins",
        type=int,
        default=0,
        help="Limit to first N coins (0 = all)",
    )
    args = parser.parse_args()

    l.basicConfig(
        level=getattr(l, args.log_level.upper(), l.INFO),
        filename=args.log_file,
    )

    if not verify_token(args.idtoken):
        l.error("ID token verification failed. Exiting.")
        return

    with logging_redirect_tqdm():
        l.info("fetching available metrics...")
        metrics_by_coin = available_metrics()

        schema = {
            "timestamp": pl.Datetime("ns"),
            "asset": pl.String,
            "price": pl.Float64,
            "market_cap": pl.Float64,
            "udpil": pl.Float64,
            "udpim": pl.Float64,
            "udpis": pl.Float64,
            "mdccv": pl.Float64,
            "mbi": pl.Float64,
            "tci": pl.Float64,
            "mtm": pl.Float64,
            "mcm": pl.Float64,
            "tcicv": pl.Float64,
            "upprob": pl.Float64,
            "total_volume": pl.Float64,
            "mean_realized_price_usd_7d": pl.Float64,
            "mean_realized_price_usd_14d": pl.Float64,
            "mean_realized_price_usd_30d": pl.Float64,
            "mean_realized_price_usd_180d": pl.Float64,
        }

        whitelist = [k for k in schema.keys() if k not in ["timestamp", "asset"]]
        filtered_metrics_by_coin = {
            coin: {m for m in metrics if m in whitelist}
            for coin, metrics in metrics_by_coin.items()
            if len({m for m in metrics if m in whitelist}) > 1
        }

        l.info(f"writing to {args.output_dir}")
        l.info(f"{len(filtered_metrics_by_coin)} coins found")

        # Limit coins for testing
        if args.max_coins and args.max_coins > 0:
            coins_to_process = sorted(filtered_metrics_by_coin.keys())[: args.max_coins]
            filtered_metrics_by_coin = {
                c: filtered_metrics_by_coin[c] for c in coins_to_process
            }
            l.info(
                f"limited to {len(filtered_metrics_by_coin)} coins: {coins_to_process}"
            )

        # Buffer results per coin to avoid cross-coin contamination
        dfs_by_coin: Dict[str, List[pl.DataFrame]] = {
            c: [] for c in filtered_metrics_by_coin
        }
        pending_counts: Dict[str, int] = {
            c: len(ms) for c, ms in filtered_metrics_by_coin.items()
        }

        with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
            items = [(c, m) for c, ms in filtered_metrics_by_coin.items() for m in ms]

            futures = {
                executor.submit(do_work, item, args.idtoken): item for item in items
            }

            for future in tqdm(
                as_completed(futures),
                total=len(items),
                desc="fetching data",
            ):
                coin = None
                try:
                    coin, metric, df2 = future.result()
                    if not df2.is_empty():
                        dfs_by_coin[coin].append(df2)
                except Exception as e:
                    item = futures[future]
                    coin = item[0]
                    l.error(f"error fetching {coin}/{item[1]}: {e}")

                if coin is not None and coin in pending_counts:
                    pending_counts[coin] -= 1
                    if pending_counts[coin] <= 0:
                        if dfs_by_coin[coin]:
                            l.info(f"all metrics collected for {coin}, merging...")
                            merged = merge_metrics(dfs_by_coin[coin])
                            save(merged, schema, coin, args.output_dir)
                        else:
                            l.warning(f"no data collected for {coin}, skipping")
                        del dfs_by_coin[coin]
                        del pending_counts[coin]


if __name__ == "__main__":
    main()
