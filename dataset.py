"""
dataset.py — Pull OHLCV klines for the top N CMC coins via TradingView WebSocket.

1. Fetches top N coins by market cap from CoinMarketCap.
2. For each coin, finds a USDC/USDT/USD1-quoted spot pair across multiple exchanges,
   preferring Binance → KuCoin → MEXC → HTX → Kraken → Uniswap v3 → PancakeSwap v3 → Raydium CLMM → PancakeSwap Infinity CLAMM.
3. Downloads historical OHLCV from TradingView's WebSocket API.
4. Saves results as partitioned Parquet files (one per coin).

Usage:
    uv run dataset.py -n 10                          # test with 10 coins
    uv run dataset.py -n 600                         # full dataset
    uv run dataset.py -n 10 --interval 1h --n-bars 5000
"""

import sys
import asyncio
import os
import argparse
import json
import re
from uuid import uuid4
from posixpath import join as pjoin
from typing import List, Dict, Any, Optional, Tuple

import aiohttp
import polars as pl
from tqdm.asyncio import tqdm
from urllib.parse import urljoin

from dotenv import load_dotenv
import logging as l
from tqdm.contrib.logging import logging_redirect_tqdm

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
load_dotenv()

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
    "ath,atl,high24h,low24h,num_market_pairs,date_added,"
    "max_supply,circulating_supply,total_supply,volume_7d,volume_30d,"
    "self_reported_circulating_supply,self_reported_market_cap,tags"
)
CMC_CONCURRENCY = 5
TV_CONCURRENCY = 3
DEFAULT_OUTPUT_DIR = "data/dataset"

# Quote assets we accept, in descending priority order.
QUOTE_PRIORITY = ["USDC", "USDT", "USD1"]

# TradingView exchange codes for the CMC exchange names we care about.
EXCHANGE_PRIORITY = [
    "Binance", "KuCoin", "MEXC", "HTX", "Kraken",
    "Uniswap v3 (Ethereum)", "Uniswap v3",
    "PancakeSwap v3 (BSC)", "PancakeSwap v3",
    "Raydium (CLMM)",
    "PancakeSwap Infinity CLAMM",
]

CMC_TO_TV_EXCHANGE = {
    "Binance": "BINANCE",
    "KuCoin": "KUCOIN",
    "MEXC": "MEXC",
    "HTX": "HTX",
    "Huobi": "HTX",
    "Kraken": "KRAKEN",
    "Uniswap v3 (Ethereum)": "UNISWAP3ETH",
    "Uniswap v3": "UNISWAP3ETH",
    "PancakeSwap v3 (BSC)": "PANCAKESWAP",
    "PancakeSwap v3": "PANCAKESWAP",
    "Raydium (CLMM)": "RAYDIUM",
    "PancakeSwap Infinity CLAMM": "PANCAKESWAP",
}

# Column names in the output parquet for each whitelisted exchange.
EXCHANGE_COLUMNS = [
    ("Binance", "binance"),
    ("KuCoin", "kucoin"),
    ("MEXC", "mexc"),
    ("HTX", "htx"),
    ("Kraken", "kraken"),
    ("Uniswap v3 (Ethereum)", "uniswap_v3"),
    ("PancakeSwap v3 (BSC)", "pancakeswap_v3"),
    ("Raydium (CLMM)", "raydium_clmm"),
    ("PancakeSwap Infinity CLAMM", "pancakeswap_alpha_clamm"),
]

# Slugs to exclude due to CMC data quality issues.
# tether (USDT) is mislabeled as BTCUSDT on KuCoin in CMC's market-pairs data,
# causing a fatal symbol collision with actual Bitcoin data.
EXCLUDED_SLUGS = {"tether"}

# Map human-readable interval → TradingView series interval + resample info
INTERVAL_MAP = {
    "1m": ("1", 1),
    "5m": ("5", 5),
    "15m": ("15", 15),
    "30m": ("30", 30),
    "1h": ("60", 60),
    "2h": ("120", 120),
    "4h": ("240", 240),
    "1d": ("1D", 1440),
    "1w": ("1W", 10080),
    "1M": ("1M", 43200),
}

l.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=l.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

# concurrency control — set in main()
_cmc_sem: Optional[asyncio.Semaphore] = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _make_headers() -> Dict[str, str]:
    return {**BROWSER_HEADERS, "x-request-id": uuid4().hex}


class TransientError(Exception):
    pass


async def exponential_backoff(fn, *args, retries: int = 5, **kwargs):
    base_delay = 1.0
    growth = 2.0
    max_delay = 30.0
    for attempt in range(retries):
        try:
            return await fn(*args, **kwargs)
        except TransientError:
            delay = min(base_delay * (growth**attempt), max_delay)
            l.info(f"Transient error, retrying in {delay:.1f}s …")
            await asyncio.sleep(delay)
    raise TransientError("Max retries exceeded")


# ---------------------------------------------------------------------------
# CMC HTTP helpers
# ---------------------------------------------------------------------------
async def _get_json(
    session: aiohttp.ClientSession, path: str, params: Dict[str, Any]
) -> dict:
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
# CMC listings (top N coins by market-cap rank)
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
    session: aiohttp.ClientSession, total: int = 600, page_size: int = 100
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
# CMC market pairs
# ---------------------------------------------------------------------------
async def fetch_market_pairs(
    session: aiohttp.ClientSession,
    slug: str,
    start: int = 1,
    limit: int = 100,
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
# exchange / pair resolution
# ---------------------------------------------------------------------------
def _pair_volume(p: dict) -> float:
    """Extract 24h volume from a CMC market-pair dict, trying known fields."""
    for field in ("volume24h", "volume24hUsd", "volumeUsd", "volume24hQuote"):
        try:
            v = p.get(field)
            if v is not None:
                return float(v)
        except (TypeError, ValueError):
            pass
    return 0.0


def resolve_all_pairs(
    coin: dict, market_pairs_map: Dict[str, list]
) -> Optional[Tuple[str, str, dict]]:
    """
    Find the best USDC / USDT / USD1 pair per whitelisted exchange.

    For each exchange, the best quote asset is selected according to
    QUOTE_PRIORITY (USDC → USDT → USD1).  Within the same quote asset,
    the pair with the highest 24h volume wins.

    Returns (tradingview_symbol, tradingview_exchange, availability) or None.
    availability is a dict {column_name: pair_name or None} for every
    exchange in EXCHANGE_COLUMNS.
    """
    slug = coin.get("slug", "")
    pairs = market_pairs_map.get(slug, [])
    if not pairs:
        return None

    # Group pairs by (exchange, quote) and keep the highest-volume one.
    # key = (cmc_exchange_name, quote_symbol)
    best_per_quote: Dict[Tuple[str, str], dict] = {}
    for p in pairs:
        ex = p.get("exchangeName", "")
        quote = p.get("quoteSymbol", "")
        base = p.get("baseSymbol", "")
        if not (ex and base and quote in QUOTE_PRIORITY):
            continue
        key = (ex, quote)
        vol = _pair_volume(p)
        prev = best_per_quote.get(key)
        if prev is None or vol > _pair_volume(prev):
            best_per_quote[key] = p

    # For each whitelisted exchange, pick the best quote asset by priority.
    # exchange_best: cmc_name → (base, quote, tv_symbol)
    exchange_best: Dict[str, Tuple[str, str, str]] = {}
    for cmc_name in EXCHANGE_PRIORITY:
        for quote in QUOTE_PRIORITY:
            p = best_per_quote.get((cmc_name, quote))
            if p:
                base = p.get("baseSymbol", "")
                if base:
                    exchange_best[cmc_name] = (base, quote, f"{base}{quote}")
                    break

    # Determine primary exchange (first in priority with ASCII base).
    # Non-ASCII symbols are skipped because TradingView's WebSocket
    # cannot resolve tickers like 币安人生.
    first_exchange: Optional[str] = None
    first_symbol: Optional[str] = None
    for cmc_name in EXCHANGE_PRIORITY:
        tv_ex = CMC_TO_TV_EXCHANGE.get(cmc_name)
        best = exchange_best.get(cmc_name)
        if tv_ex and best and best[0].isascii():
            first_exchange = tv_ex
            first_symbol = best[2]
            break

    if first_exchange is None:
        return None

    # Build per-exchange availability
    availability: Dict[str, Optional[str]] = {}
    for cmc_name, col_name in EXCHANGE_COLUMNS:
        best = exchange_best.get(cmc_name)
        availability[col_name] = best[2] if best else None

    return first_symbol, first_exchange, availability


# ---------------------------------------------------------------------------
# TradingView OHLCV fetching (WebSocket, via tradingview-scraper internals)
# ---------------------------------------------------------------------------
# We use the library's StreamHandler for the WebSocket session and send a
# custom create_series with the correct interval (the library hardcodes "1").
# Imported lazily to avoid the noisy pkg_resources deprecation warning at import time.


def _fetch_tv_ohlcv_sync(
    exchange_symbol: str,
    tv_interval: str,
    n_bars: int,
    jwt_token: str = "unauthorized_user_token",
) -> List[dict]:
    """
    Use tradingview-scraper's StreamHandler for session management, but
    send a create_series with the proper interval. Collect the
    timescale_update, close, and return the list of candles.
    """
    from tradingview_scraper.symbols.stream.stream_handler import StreamHandler

    print(exchange_symbol)

    ws_url = (
        "wss://data.tradingview.com/socket.io/websocket"
        "?from=chart%2FVEPYsueI%2F&type=chart"
    )
    handler = StreamHandler(websocket_url=ws_url, jwt_token=jwt_token)

    try:
        qs = handler.quote_session
        cs = handler.chart_session

        resolve = json.dumps({"adjustment": "splits", "symbol": exchange_symbol})
        handler.send_message("quote_add_symbols", [qs, f"={resolve}"])
        handler.send_message("resolve_symbol", [cs, "sds_sym_1", f"={resolve}"])
        # ---- the key fix: use the actual tv_interval instead of hardcoded "1" ----
        handler.send_message(
            "create_series",
            [cs, "sds_1", "s1", "sds_sym_1", tv_interval, n_bars, ""],
        )
        handler.send_message("quote_fast_symbols", [qs, exchange_symbol])

        # Collect until we get a timescale_update.  Set a read timeout so we
        # don't hang forever on dead connections / unlisted symbols.
        handler.ws.settimeout(10)

        import re as _re
        from websocket import WebSocketTimeoutException

        for _ in range(60):
            try:
                raw = handler.ws.recv()
            except WebSocketTimeoutException:
                raise TransientError(f"timeout waiting for {exchange_symbol}")
            except Exception:
                raise TransientError(f"recv error for {exchange_symbol}")

            if _re.match(r"~m~\d+~m~~h~\d+$", raw):
                handler.ws.send(raw)
                continue
            parts = [x for x in _re.split(r"~m~\d+~m~", raw) if x]
            for part in parts:
                if not part.startswith("{"):
                    continue
                pkt = json.loads(part)
                if pkt.get("m") == "timescale_update":
                    raw_series = pkt.get("p", [{}, {}])[1].get("sds_1", {}).get("s", [])
                    candles = []
                    for entry in raw_series:
                        v = entry["v"]
                        candles.append(
                            {
                                "timestamp": v[0],
                                "open": v[1],
                                "high": v[2],
                                "low": v[3],
                                "close": v[4],
                                "volume": v[5],
                            }
                        )
                    return candles
                elif pkt.get("m") == "critical_error":
                    raise TransientError(
                        f"critical_error for {exchange_symbol}: {str(pkt)[:200]}"
                    )
        raise TransientError("no timescale_update")
    finally:
        try:
            handler.ws.close()
        except Exception:
            pass


async def _fetch_one_symbol(
    exchange_symbol: str,
    tv_interval: str,
    n_bars: int,
    jwt_token: str,
) -> Optional[pl.DataFrame]:
    """Fetch OHLCV for one symbol via TradingView WebSocket, with backoff."""

    async def _inner():
        try:
            candles = await asyncio.to_thread(
                _fetch_tv_ohlcv_sync,
                exchange_symbol,
                tv_interval,
                n_bars,
                jwt_token,
            )
        except Exception as e:
            raise TransientError(f"fetch failed for {exchange_symbol}: {e}") from e

        if not candles:
            raise TransientError(f"empty result for {exchange_symbol}")
        return candles

    try:
        candles = await exponential_backoff(_inner)
    except TransientError:
        l.error(f"failed to fetch {exchange_symbol} after retries")
        return None

    df = pl.DataFrame(candles)
    df = df.with_columns(
        pl.from_epoch("timestamp", time_unit="s")
        .dt.replace_time_zone("UTC")
        .alias("ts")
    ).drop("timestamp")
    return df


async def fetch_ohlcv_for_group(
    symbols: List[str],
    exchange: str,
    interval: str,
    n_bars: int,
    max_concurrent: int,
    jwt_token: str,
) -> Dict[str, pl.DataFrame]:
    """Fetch OHLCV data for a group of symbols on the same exchange."""
    tv_interval = INTERVAL_MAP[interval][0]
    exchange_symbols = [f"{exchange}:{sym}" for sym in symbols]

    l.info(
        f"fetching {len(exchange_symbols)} symbols from {exchange} "
        f"(interval={interval}, n_bars={n_bars}, concurrency={max_concurrent})"
    )

    sem = asyncio.Semaphore(max_concurrent)

    async def _one(esym: str) -> Tuple[str, Optional[pl.DataFrame]]:
        async with sem:
            return esym, await _fetch_one_symbol(
                esym,
                tv_interval,
                n_bars,
                jwt_token,
            )

    fut = [_one(es) for es in exchange_symbols]
    results = await tqdm.gather(*fut, desc=f"fetching {exchange}", unit="sym")

    out: Dict[str, pl.DataFrame] = {}
    for esym, df in results:
        sym = esym.split(":")[1]
        if df is not None:
            out[sym] = df
        else:
            l.warning(f"no data for {sym} on {exchange}")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull OHLCV klines for top CMC coins via TradingView WebSocket."
    )
    parser.add_argument(
        "-n",
        "--num-coins",
        type=int,
        default=600,
        help="Number of top coins to scrape (default: 600).",
    )
    parser.add_argument(
        "-j",
        "--cmc-concurrency",
        type=int,
        default=CMC_CONCURRENCY,
        help="Max concurrent CMC HTTP requests (default: %(default)s).",
    )
    parser.add_argument(
        "--tv-concurrency",
        type=int,
        default=TV_CONCURRENCY,
        help="Max concurrent TradingView WebSocket connections (default: %(default)s).",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        choices=list(INTERVAL_MAP),
        help="OHLCV interval (default: 1d).",
    )
    parser.add_argument(
        "--n-bars",
        type=int,
        default=5000,
        help="Max bars to fetch per coin (default: 5000).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for Parquet files (default: {DEFAULT_OUTPUT_DIR}).",
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
    _cmc_sem = asyncio.Semaphore(args.cmc_concurrency)

    # TradingView JWT token (from env)
    jwt_token = os.getenv("TV_JWT_TOKEN", "unauthorized_user_token")
    if jwt_token == "unauthorized_user_token":
        l.warning("No TV_JWT_TOKEN in env — using anonymous access")

    # ------------------------------------------------------------------
    async with aiohttp.ClientSession() as session:
        # ---- 1. listings ----
        l.info(f"fetching top {args.num_coins} coins by market cap …")
        coins = await fetch_all_listings(session, total=args.num_coins)
        l.info(f"got {len(coins)} coins")

        # ---- 2. market pairs ----
        l.info("fetching spot market pairs …")
        market_pairs_map = await fetch_all_market_pairs(session, coins)

        # ---- 3. resolve TradingView symbols ----
        tv_symbols: Dict[str, List[str]] = {}
        # symbol → list of (coin_dict, availability_dict)
        coin_index: Dict[str, list] = {}
        seen: set = set()
        unmatched = 0

        for coin in coins:
            if coin.get("slug") in EXCLUDED_SLUGS:
                continue
            resolved = resolve_all_pairs(coin, market_pairs_map)
            if resolved is None:
                unmatched += 1
                continue
            symbol, exchange, availability = resolved
            key = (exchange, symbol)
            if key not in seen:
                seen.add(key)
                tv_symbols.setdefault(exchange, []).append(symbol)
            coin_index.setdefault(symbol, []).append((coin, availability))

        total_matched = sum(len(v) for v in tv_symbols.values())
        l.info(
            f"resolved {total_matched} symbols across "
            f"{len(tv_symbols)} exchanges "
            f"({unmatched} coins had no USDC/USDT/USD1 pair on target exchanges)"
        )

        # ---- 4. fetch OHLCV per exchange group ----
        all_data: Dict[str, pl.DataFrame] = {}

        for exchange, symbols in tv_symbols.items():
            group_data = await fetch_ohlcv_for_group(
                symbols,
                exchange,
                args.interval,
                args.n_bars,
                args.tv_concurrency,
                jwt_token,
            )
            all_data.update(group_data)

        l.info(f"fetched OHLCV data for {len(all_data)} symbols")

        # ---- 5. save ----
        frames = []
        seen_coins: set = set()
        for sym, df in all_data.items():
            for coin, avail in coin_index.get(sym, []):
                cid = coin.get("id", sym)
                if cid in seen_coins:
                    continue
                seen_coins.add(cid)
                slug = coin.get("slug", "")
                quotes = coin.get("quotes", [])
                mcap = next(
                    (q.get("marketCap") for q in quotes if q.get("name") == "USD"),
                    None,
                )
                tags = coin.get("tags") or []
                exchange_cols = [
                    pl.lit(avail.get(col), dtype=pl.Utf8).alias(col)
                    for _, col in EXCHANGE_COLUMNS
                ]
                frames.append(
                    df.with_columns(
                        [
                            pl.lit(slug).alias("slug"),
                            pl.lit(mcap, dtype=pl.Float64).alias("market_cap"),
                            pl.lit(tags, dtype=pl.List(pl.Utf8)).alias("tags"),
                        ]
                        + exchange_cols
                    )
                )

        combined = pl.concat(frames).sort("ts", "slug")
        os.makedirs(args.output, exist_ok=True)
        out_path = pjoin(args.output, "all.parquet")
        combined.write_parquet(out_path, compression="zstd")
        l.info(
            f"saved {combined.select(pl.col('slug')).n_unique()} symbols "
            f"({len(combined)} rows) to {out_path}"
        )

        # ---- summary ----
        if args.debug:
            print(f"\nSample (first 5 coins):")
            for coin in coins[:5]:
                sym = coin.get("slug", "?")
                name = coin.get("name", "?")
                rank = coin.get("cmcRank", "?")
                resolved = resolve_all_pairs(coin, market_pairs_map)
                tv_info = f"{resolved[0]} @ {resolved[1]}" if resolved else "—"
                print(f"  {rank:>4}  {sym:<8} {name:<30}  {tv_info}")


if __name__ == "__main__":
    with logging_redirect_tqdm():
        try:
            asyncio.run(main())
        except Exception as e:
            l.exception("Fatal error", exc_info=e)
            sys.exit(1)
