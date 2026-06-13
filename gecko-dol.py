import sys
import asyncio
import os
from uuid import uuid4

import aiohttp
from tqdm.asyncio import tqdm

from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin

from dotenv import load_dotenv
import logging as l
from tqdm.contrib.logging import logging_redirect_tqdm

load_dotenv()

l.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=l.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
CONCURRENCY = 5
CMC_BASE = "https://api.coinmarketcap.com/data-api/"
BROWSER_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "de-DE,de;q=0.8",
    "cache-control": "no-cache",
    "origin": "https://coinmarketcap.com",
    "platform": "web",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "referer": "https://coinmarketcap.com/",
    "sec-ch-ua": '"Chromium";v="148", "Brave";v="148", "Not/A)Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "sec-gpc": "1",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
    ),
}
LISTING_AUX = (
    "ath,atl,high24h,low24h,num_market_pairs,cmc_rank,date_added,"
    "max_supply,circulating_supply,total_supply,volume_7d,volume_30d,"
    "self_reported_circulating_supply,self_reported_market_cap"
)

# concurrency control — set in main()
_cmc_sem: Optional[asyncio.Semaphore] = None


def _make_headers() -> Dict[str, str]:
    return {**BROWSER_HEADERS, "x-request-id": uuid4().hex}


# ---------------------------------------------------------------------------
# exponential backoff  (mirrors sync-datastore.py)
# ---------------------------------------------------------------------------
class TransientError(Exception):
    pass


async def exponential_backoff(fn, *args, retries: int = 5, **kwargs):
    base_delay = 1.0
    growth = 2.0
    max_delay = 30.0

    for i in range(retries):
        try:
            return await fn(*args, **kwargs)
        except TransientError:
            delay = min(base_delay * (growth**i), max_delay)
            l.info(f"Transient error, retrying in {delay:.1f}s …")
            await asyncio.sleep(delay)
    raise TransientError("Max retries exceeded")


# ---------------------------------------------------------------------------
# low-level HTTP helpers
# ---------------------------------------------------------------------------
async def _get_json(
    session: aiohttp.ClientSession, path: str, params: Dict[str, Any]
) -> dict:
    """GET *path* (relative to CMC_BASE) and return parsed JSON."""
    if _cmc_sem is None:
        raise RuntimeError("_cmc_sem not initialised")

    url = urljoin(CMC_BASE, path)

    async with _cmc_sem:
        async with session.get(url, params=params, headers=_make_headers()) as resp:
            if resp.status == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else 10
                l.warning(f"429 on {path} – waiting {wait:.0f}s")
                await asyncio.sleep(wait)
                raise TransientError("rate-limited")
            if resp.status >= 500:
                raise TransientError(f"server error {resp.status} on {path}")
            if resp.status != 200:
                l.error(f"HTTP {resp.status} on {path}: {await resp.text()}")
                return {}
            return await resp.json()


# ---------------------------------------------------------------------------
# listing  (paginate, up to *limit* coins sorted by market-cap rank)
# ---------------------------------------------------------------------------
async def fetch_listing_page(
    session: aiohttp.ClientSession, start: int, limit: int = 100
) -> dict:
    return await _get_json(
        session,
        "v3/cryptocurrency/listing",
        params={
            "start": start,
            "limit": limit,
            "sortBy": "rank",
            "sortType": "desc",
            "convert": "USD,BTC,ETH",
            "cryptoType": "all",
            "tagType": "all",
            "audited": "false",
            "aux": LISTING_AUX,
        },
    )


async def fetch_all_listings(
    session: aiohttp.ClientSession, total: int = 500, page_size: int = 100
) -> List[dict]:
    """Return flat list of coin dicts for the first *total* coins by rank."""
    pages = range(1, total + 1, page_size)
    fut = [
        exponential_backoff(fetch_listing_page, session, s, page_size) for s in pages
    ]
    results = await tqdm.gather(*fut, desc="fetching listings")

    coins: List[dict] = []
    for page in results:
        data = page.get("data", {})
        currency_list = data.get("cryptoCurrencyList", [])
        coins.extend(currency_list)
    return coins[:total]


# ---------------------------------------------------------------------------
# chart  (daily price & market-cap history)
# ---------------------------------------------------------------------------
async def fetch_chart(
    session: aiohttp.ClientSession, coin_id: int, convert_id: int = 2781
) -> dict:
    return await _get_json(
        session,
        "v3.3/cryptocurrency/detail/chart",
        params={
            "id": coin_id,
            "interval": "1d",
            "convertId": convert_id,
            "range": "All",
        },
    )


async def fetch_charts(
    session: aiohttp.ClientSession,
    coins: List[dict],
) -> Dict[int, dict]:
    """Fetch daily chart for every coin. Returns {coin_id: chart_json}."""

    async def _one(c: dict):
        cid = c["id"]
        try:
            return cid, await exponential_backoff(fetch_chart, session, cid)
        except TransientError:
            l.error(f"failed to fetch chart for {c.get('slug', cid)}")
            return cid, {}

    fut = [_one(c) for c in coins]
    pairs = await tqdm.gather(*fut, desc="fetching charts")
    return dict(pairs)


# ---------------------------------------------------------------------------
# market pairs  (spot exchanges per coin)
# ---------------------------------------------------------------------------
async def fetch_market_pairs(
    session: aiohttp.ClientSession,
    slug: str,
    start: int = 1,
    limit: int = 10,
) -> dict:
    return await _get_json(
        session,
        "v3/cryptocurrency/market-pairs/latest",
        params={
            "slug": slug,
            "start": start,
            "limit": limit,
            "category": "spot",
            "centerType": "all",
            "sort": "cmc_rank_advanced",
            "direction": "desc",
            "spotUntracked": "true",
        },
    )


async def fetch_all_market_pairs(
    session: aiohttp.ClientSession, coins: List[dict]
) -> Dict[str, list]:
    """Fetch market pairs for every coin. Returns {slug: [market_pair_dict]}."""

    async def _one(c: dict):
        slug = c.get("slug", "")
        if not slug:
            return slug, []
        try:
            resp = await exponential_backoff(fetch_market_pairs, session, slug)
            return slug, resp.get("data", {}).get("marketPairs", [])
        except TransientError:
            l.error(f"failed to fetch market pairs for {slug}")
            return slug, []

    fut = [_one(c) for c in coins]
    pairs = await tqdm.gather(*fut, desc="fetching market pairs")
    return dict(pairs)


# ---------------------------------------------------------------------------
# summary / pretty-print helpers
# ---------------------------------------------------------------------------
def _find_quote(quotes: list, name: str) -> dict:
    """Pick the quote dict for *name* (e.g. 'USD', 'BTC') from the list."""
    for q in quotes:
        if q.get("name") == name:
            return q
    return {}


def _safe_field(quotes: list, field: str, quote_name: str = "USD") -> Optional[float]:
    q = _find_quote(quotes, quote_name)
    v = q.get(field)
    return float(v) if v is not None else None


def print_listing_summary(coins: List[dict]):
    print(
        f"\n{'rank':>5} {'symbol':<10} {'name':<30} "
        f"{'price (USD)':>14} {'mcap (USD)':>18}"
    )
    print("-" * 82)
    for c in coins:
        quotes = c.get("quotes", [])
        price = _safe_field(quotes, "price")
        mcap = _safe_field(quotes, "marketCap")
        print(
            (
                f"{c.get('cmcRank', '?'):>5} "
                f"{c.get('symbol', '?'):<10} "
                f"{c.get('name', '?')[:29]:<30} "
                f"{price:>14.6f} "
                if price is not None
                else " " * 15
            ),
            f"{mcap:>18,.0f}" if mcap is not None else "",
        )


def print_chart_sample(charts: Dict[int, dict], coins: List[dict]):
    print()
    for c in coins[:10]:
        cid = c["id"]
        ch = charts.get(cid, {})
        points = ch.get("data", {}).get("points", [])
        if points:
            first_ts = list(points.keys())[0] if isinstance(points, dict) else points[0]
            last_ts = (
                list(points.keys())[-1] if isinstance(points, dict) else points[-1]
            )
            print(
                f"  {c['symbol']:<8} {len(points):>5} daily candles"
                f"  ({first_ts} .. {last_ts})"
            )
        else:
            print(f"  {c['symbol']:<8} no chart data")


def print_market_pairs_sample(market_pairs: Dict[str, list]):
    print()
    for slug, pairs in list(market_pairs.items())[:10]:
        if not pairs:
            continue
        print(f"  {slug}:")
        for p in pairs[:3]:
            name = p.get("exchangeName", "?")
            base = p.get("baseSymbol", "?")
            quote = p.get("quoteSymbol", "?")
            vol = p.get("volumeUsd", 0)
            print(f"    {name:<25} {base}/{quote:<10} vol={vol:>14,.0f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(
        description="Scrape CoinMarketCap listings, charts, and market pairs."
    )
    parser.add_argument(
        "-n",
        "--num-coins",
        type=int,
        default=10,
        help="Number of top coins to scrape (default: 10).",
    )
    parser.add_argument(
        "-j",
        "--concurrency",
        type=int,
        default=CONCURRENCY,
        help="Max concurrent HTTP requests (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-charts",
        action="store_true",
        help="Skip downloading daily price/mcap history.",
    )
    parser.add_argument(
        "--skip-pairs",
        action="store_true",
        help="Skip downloading spot market pairs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    if args.debug:
        l.getLogger().setLevel(l.DEBUG)

    global _cmc_sem
    _cmc_sem = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession() as session:
        # ---- 1. listings ----
        l.info(f"fetching top {args.num_coins} coins by market cap …")
        coins = await fetch_all_listings(session, total=args.num_coins)
        l.info(f"got {len(coins)} coins")
        print_listing_summary(coins)

        # ---- 2. charts ----
        if not args.skip_charts:
            l.info("fetching daily charts …")
            charts = await fetch_charts(session, coins)
            print_chart_sample(charts, coins)
        else:
            charts = {}

        # ---- 3. market pairs ----
        if not args.skip_pairs:
            l.info("fetching spot market pairs …")
            pairs = await fetch_all_market_pairs(session, coins)
            print_market_pairs_sample(pairs)
        else:
            pairs = {}


if __name__ == "__main__":
    import argparse

    with logging_redirect_tqdm():
        try:
            asyncio.run(main())
        except Exception as e:
            l.exception("Fatal error", exc_info=e)
            sys.exit(1)
