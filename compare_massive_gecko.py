"""Compare CMC top-600 coins (via gecko.py's data API) against
Massive (Polygon.io) tickers filtered to USDC / USDT quote pairs.

Lists coins in CMC top 600 missing from Massive in those quote pairs.
"""

import os
import sys
import time
import asyncio
import random
from uuid import uuid4
from urllib.parse import urljoin, urlencode, urlparse, parse_qs

import aiohttp
import polars as pl
import requests
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

load_dotenv()

API_KEY = os.environ["MASSIVE_APIKEY"]
OUTPUT = "missing_from_massive.parquet"

# ---------------------------------------------------------------------------
# shared backoff
# ---------------------------------------------------------------------------
class TransientError(Exception):
    pass


# ---------------------------------------------------------------------------
# CMC (CoinMarketCap) – async, from gecko.py
# ---------------------------------------------------------------------------
CMC_BASE = "https://api.coinmarketcap.com/data-api/"

BROWSER_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "origin": "https://coinmarketcap.com",
    "referer": "https://coinmarketcap.com/",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
    ),
}

CONCURRENCY = 5

LISTING_AUX = (
    "ath,atl,high24h,low24h,num_market_pairs,cmc_rank,date_added,"
    "max_supply,circulating_supply,total_supply,volume_7d,volume_30d,"
    "self_reported_circulating_supply,self_reported_market_cap"
)


async def cmc_exponential_backoff(fn, *args, retries: int = 5, **kwargs):
    base_delay = 1.0
    growth = 2.0
    max_delay = 60.0
    for i in range(retries):
        try:
            return await fn(*args, **kwargs)
        except TransientError:
            delay = min(base_delay * (growth ** i), max_delay) + random.uniform(0, 2)
            print(f"  retrying in {delay:.1f}s …", flush=True)
            await asyncio.sleep(delay)
    raise TransientError("Max retries exceeded")


def _make_headers() -> dict:
    return {**BROWSER_HEADERS, "x-request-id": uuid4().hex}


async def _cmc_get_json(
    sem: asyncio.Semaphore, session: aiohttp.ClientSession,
    path: str, params: dict
) -> dict:
    url = urljoin(CMC_BASE, path)
    async with sem:
        async with session.get(url, params=params, headers=_make_headers()) as resp:
            if resp.status == 429:
                raise TransientError("rate-limited (429)")
            if resp.status >= 500:
                raise TransientError(f"server error {resp.status}")
            if resp.status != 200:
                print(f"  HTTP {resp.status} on {path}", flush=True)
                return {}
            return await resp.json()


async def fetch_cmc_top_n(n: int = 600, page_size: int = 100) -> list[dict]:
    """Fetch top *n* coins from CMC by market cap."""
    sem = asyncio.Semaphore(CONCURRENCY)
    pages = range(1, n + 1, page_size)

    async with aiohttp.ClientSession() as session:
        async def _fetch_one(start: int) -> dict:
            return await cmc_exponential_backoff(
                _cmc_get_json, sem, session,
                "v3/cryptocurrency/listing",
                {
                    "start": start,
                    "limit": page_size,
                    "sortBy": "rank",
                    "sortType": "desc",
                    "convert": "USD",
                    "cryptoType": "all",
                    "tagType": "all",
                    "audited": "false",
                    "aux": LISTING_AUX,
                },
            )

        fut = [_fetch_one(s) for s in pages]
        results = await tqdm.gather(*fut, desc="fetching CMC listings")

    coins: list[dict] = []
    for page in results:
        data = page.get("data", {})
        currency_list = data.get("cryptoCurrencyList", [])
        coins.extend(currency_list)
    return coins[:n]


# ---------------------------------------------------------------------------
# Massive (Polygon.io) – sync, from massive.py
# ---------------------------------------------------------------------------
MASSIVE_BASE = "https://api.massive.com/v3/reference/tickers"


def massive_exponential_backoff(fn, *args, retries: int = 7, **kwargs):
    base_delay = 1.0
    growth = 2.0
    max_delay = 120.0
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except TransientError as e:
            delay = min(base_delay * (growth ** i), max_delay) + random.uniform(0, delay * 0.5)
            print(f"  transient ({e}), retrying in {delay:.1f}s …", flush=True)
            time.sleep(delay)
    raise TransientError("Max retries exceeded")


def _massive_fetch_page(sess: requests.Session, params: dict) -> dict:
    url = f"{MASSIVE_BASE}?{urlencode(params)}"
    resp = sess.get(url, timeout=30)
    if resp.status_code == 429:
        raise TransientError("rate-limited (429)")
    if resp.status_code >= 500:
        raise TransientError(f"server error ({resp.status_code})")
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") == "ERROR":
        raise RuntimeError(f"API error: {data.get('error', 'unknown')}")
    return data


def fetch_massive_tickers(quote_filter: list[str] | None = None) -> list[dict]:
    """Paginate all active crypto tickers, optionally filtering by quote currency."""
    params: dict[str, str] = {
        "market": "crypto",
        "active": "true",
        "limit": "1000",
        "order": "asc",
        "sort": "ticker",
        "apiKey": API_KEY,
    }
    all_tickers: list[dict] = []
    sess = requests.Session()
    page = 0

    while True:
        page += 1
        resp = massive_exponential_backoff(_massive_fetch_page, sess, params)
        results = resp.get("results", [])
        if quote_filter:
            results = [r for r in results if r.get("currency_symbol") in quote_filter]
        all_tickers.extend(results)
        n = len(all_tickers)
        label = ",".join(quote_filter) if quote_filter else "all"
        print(f"  page {page:>3} ({label}): +{len(results)} → total {n:,}", flush=True)

        next_url = resp.get("next_url")
        if not next_url or not resp.get("results"):
            break
        cursor = parse_qs(urlparse(next_url).query).get("cursor", [None])[0]
        if not cursor:
            break
        params["cursor"] = cursor
        time.sleep(12.0)  # free-tier: 5 req/min

    return all_tickers


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    # --- 1. Fetch CMC top 600 ---
    print("=== 1. Fetching CMC top 600 coins ===", flush=True)
    cmc_coins = asyncio.run(fetch_cmc_top_n(600))
    print(f"  Got {len(cmc_coins)} coins\n", flush=True)

    cmc_by_symbol: dict[str, dict] = {}
    for i, c in enumerate(cmc_coins):
        sym = c["symbol"].upper()
        if sym not in cmc_by_symbol:
            cmc_by_symbol[sym] = c
    cmc_syms = set(cmc_by_symbol.keys())
    print(f"  Unique CMC symbols: {len(cmc_syms)}", flush=True)

    # --- 2. Fetch Massive tickers ---
    # Free-tier Polygon.io only returns fiat-quoted pairs (X:BTCUSD, etc.).
    # USDC/USDT quote pairs are not available, so we use all massive tickers
    # as a best-effort proxy. Quote currencies found are listed below.
    print("\n=== 2. Fetching Massive crypto tickers ===", flush=True)
    massive_tickers = fetch_massive_tickers(quote_filter=None)
    quotes_seen = sorted({r["currency_symbol"] for r in massive_tickers})
    print(f"\n  Total Massive tickers: {len(massive_tickers)}")
    print(f"  Quote currencies found: {quotes_seen}")

    massive_bases: set[str] = set()
    for r in massive_tickers:
        massive_bases.add(r["base_currency_symbol"].upper())
    print(f"  Unique Massive base symbols: {len(massive_bases)}", flush=True)

    # --- 3. Compare ---
    present = cmc_syms & massive_bases
    missing = cmc_syms - massive_bases

    print(f"\n=== 3. Results ===", flush=True)
    print(f"  CMC top 600 unique symbols: {len(cmc_syms)}")
    print(f"  Present in Massive:          {len(present)}")
    print(f"  Missing from Massive:        {len(missing)}")
    print(f"  (NB: Massive free tier has no USDC/USDT quote pairs — using fiat pairs as proxy)")

    # --- 4. Show missing coins ---
    if missing:
        ranked = [
            (cmc_by_symbol[s].get("cmcRank", 99999), s, cmc_by_symbol[s].get("name", ""))
            for s in missing
        ]
        ranked.sort(key=lambda x: x[0] if x[0] is not None else 99999)
        print(f"\n--- Missing from Massive ({len(ranked)} coins) ---")
        print(f"  {'rank':>5}  {'symbol':<12}  name")
        print(f"  {'-----':>5}  {'------':<12}  ----")
        for rank, sym, name in ranked:
            rank_str = str(rank) if rank is not None else "?"
            print(f"  {rank_str:>5}  {sym:<12}  {name}")

    # --- 5. Save missing list as parquet ---
    rows = [
        {"cmc_rank": cmc_by_symbol[s].get("cmcRank"),
         "symbol": s,
         "name": cmc_by_symbol[s].get("name", ""),
         "slug": cmc_by_symbol[s].get("slug", "")}
        for s in missing
    ]
    if rows:
        missing_df = pl.DataFrame(rows)
    if rows:
        missing_df = pl.DataFrame(rows)
        missing_df.write_parquet(OUTPUT)
        print(f"\n  Missing coins written to {OUTPUT}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
