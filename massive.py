"""Download all cryptocurrency tickers from the Massive (Polygon.io) API
and write them to a Parquet file using Polars.

Credentials are read from the .env file via MASSIVE_APIKEY.

Usage:
    uv run massive.py [--out PATH] [--page-size N]
"""

import os
import sys
import time
import random
from urllib.parse import urlencode, urlparse, parse_qs

import polars as pl
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
API_KEY = os.environ["MASSIVE_APIKEY"]
BASE_URL = "https://api.massive.com/v3/reference/tickers"
DEFAULT_PAGE_SIZE = 1000  # max allowed by Polygon.io
OUTPUT = "massive_tickers.parquet"
MAX_RETRIES = 7


# ---------------------------------------------------------------------------
# exponential backoff
# ---------------------------------------------------------------------------
class TransientError(Exception):
    pass


def exponential_backoff(fn, *args, retries: int = MAX_RETRIES, **kwargs):
    """Retry *fn* with exponential backoff + jitter on transient failures."""
    base_delay = 1.0
    growth = 2.0
    max_delay = 120.0

    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except TransientError as e:
            delay = min(base_delay * (growth ** i), max_delay)
            jitter = random.uniform(0, delay * 0.5)
            total = delay + jitter
            print(f"  transient error ({e}), retrying in {total:.1f}s …", flush=True)
            time.sleep(total)
    raise TransientError(f"Max retries ({retries}) exceeded")


# ---------------------------------------------------------------------------
# low-level HTTP
# ---------------------------------------------------------------------------
def _fetch_page(session: requests.Session, params: dict[str, str]) -> dict:
    """Fetch a single page. Raises TransientError on 429 / 5xx."""
    url = f"{BASE_URL}?{urlencode(params)}"
    resp = session.get(url, timeout=30)
    if resp.status_code == 429:
        raise TransientError("rate-limited (429)")
    if resp.status_code >= 500:
        raise TransientError(f"server error ({resp.status_code})")
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") == "ERROR":
        raise RuntimeError(f"API error: {data.get('error', 'unknown')}")
    return data


def fetch_page(session: requests.Session, params: dict[str, str]) -> dict:
    """Fetch a single page, wrapped in exponential backoff."""
    return exponential_backoff(_fetch_page, session, params)


# ---------------------------------------------------------------------------
# paginate
# ---------------------------------------------------------------------------
def fetch_all_tickers(
    api_key: str, page_size: int = DEFAULT_PAGE_SIZE,
) -> list[dict]:
    """Paginate through all crypto tickers and return a flat list of dicts."""
    params: dict[str, str] = {
        "market": "crypto",
        "active": "true",
        "limit": str(page_size),
        "order": "asc",
        "sort": "ticker",
        "apiKey": api_key,
    }

    all_tickers: list[dict] = []
    session = requests.Session()
    page = 0

    while True:
        page += 1
        resp = fetch_page(session, params)
        results = resp.get("results", [])
        all_tickers.extend(results)
        n = len(all_tickers)
        print(f"  page {page:>3}: {len(results)} tickers → total {n:,}", flush=True)

        next_url = resp.get("next_url")
        if not next_url or not results:
            break

        # Extract cursor from next_url and feed it back
        parsed = urlparse(next_url)
        next_params = parse_qs(parsed.query)
        cursor = next_params.get("cursor", [None])[0]
        if not cursor:
            break
        params["cursor"] = cursor

        # Respect free-tier rate limit: 5 req / min → 12 s between pages
        time.sleep(12.0)

    return all_tickers


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download all crypto tickers from Massive API and save as Parquet."
    )
    parser.add_argument(
        "--out", default=OUTPUT, help=f"Output path (default: {OUTPUT})"
    )
    parser.add_argument(
        "--page-size", type=int, default=DEFAULT_PAGE_SIZE,
        help=f"Tickers per page (default: {DEFAULT_PAGE_SIZE})",
    )
    args = parser.parse_args()

    print(f"Downloading crypto tickers from {BASE_URL} ...")
    tickers = fetch_all_tickers(API_KEY, page_size=args.page_size)
    print(f"\nDone. Got {len(tickers):,} tickers total.")

    df = pl.DataFrame(tickers)
    print(f"\nSchema ({len(df.columns)} columns):")
    print(df.schema)

    df.write_parquet(args.out)
    print(f"\nWrote {df.shape[0]:,} rows × {df.shape[1]} columns to {args.out}")


if __name__ == "__main__":
    main()
