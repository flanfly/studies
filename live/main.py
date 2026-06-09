"""CLI: fetch spot pairs, borrow rates, and daily klines for active
USDT/USDC pairs from HTX, KuCoin, Kraken, and Binance.

Examples::

    uv run live/main.py --num-bars 365 \\
                        --output-klines kl.parquet \\
                        --output-symbols sym.parquet
    uv run python -m live.main --num-bars 365 \\
                               --output-klines kl.parquet \\
                               --output-symbols sym.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import logging
import os
import sys
from typing import Tuple

import polars as pl
import tqdm
from dotenv import load_dotenv
from httpx import AsyncClient
from tqdm.contrib.logging import logging_redirect_tqdm

from live import Binance, Exchange, HTX, Kraken, KuCoin, empty_klines_df

logger = logging.getLogger("live")

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

# ``httpx`` (and ``httpcore`` under it) log every request/response at
# INFO level by default. With ~2000 pairs x 4 exchanges that's
# thousands of noise lines. Suppress them; the retry layer logs
# transient failures at WARNING via the ``live`` logger, so we still
# see what matters.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def _require_env(*keys: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for k in keys:
        v = os.environ.get(k)
        if not v:
            logger.error("missing required env var %s -- set it in .env", k)
            sys.exit(2)
        values[k] = v
    return values


def _build_exchanges() -> list[Exchange]:
    """Construct one instance of each exchange, pulling credentials
    from the environment. The .env file in the project root is loaded
    on import of this module.
    """
    load_dotenv()
    htx_creds = _require_env("HTX_ACCESS_KEY", "HTX_SECRET_KEY")
    kc_creds = _require_env(
        "KUCOIN_API_KEY", "KUCOIN_API_SECRET", "KUCOIN_API_PASSWORD"
    )
    bn_creds = _require_env("BINANCE_API_KEY", "BINANCE_API_SECRET")
    return [
        HTX(
            access_key=htx_creds["HTX_ACCESS_KEY"],
            secret_key=htx_creds["HTX_SECRET_KEY"],
        ),
        KuCoin(
            api_key=kc_creds["KUCOIN_API_KEY"],
            api_secret=kc_creds["KUCOIN_API_SECRET"],
            api_password=kc_creds["KUCOIN_API_PASSWORD"],
        ),
        Kraken(),
        Binance(
            api_key=bn_creds["BINANCE_API_KEY"],
            api_secret=bn_creds["BINANCE_API_SECRET"],
        ),
    ]


async def _klines_for_pairs(
    client: AsyncClient,
    ex: Exchange,
    pairs: pl.DataFrame,
    start: dt.datetime,
    end: dt.datetime,
    outer: tqdm.tqdm,
) -> pl.DataFrame:
    """Fetch ``klines_paged`` for every (symbol, base, quote) row in
    ``pairs``, in parallel across symbols. Each completed symbol
    updates ``outer`` by one unit.

    Concurrency is bounded by ``KLINES_CONCURRENCY`` so a large
    ``pairs`` frame doesn't exhaust httpx's connection pool. The
    default httpx ``Limits`` is 100 per host, so we stay under that
    to avoid ``httpx.PoolTimeout`` storms.
    """
    # ``symbol`` is opaque to consumers; it must be passed back to
    # ``klines()`` exactly as ``pairs()`` returned it. We also need
    # the per-row ``base``/``quote`` for the joinable schema, so we
    # forward the whole row via a dict.
    rows = pairs.select("symbol", "base", "quote").to_dicts()

    semaphore = asyncio.Semaphore(ex.KLINES_CONCURRENCY)

    async def _one(row: dict[str, str]) -> pl.DataFrame:
        async with semaphore:
            try:
                df = await ex.klines_paged(
                    client, row["symbol"], start_time=start, end_time=end
                )
            except Exception:
                logger.exception(
                    "klines_paged failed for %s on %s", row["symbol"], ex.NAME
                )
                df = empty_klines_df()
        if df.height == 0:
            outer.update(1)
            return df
        # ``klines()`` already populates ``base``/``quote``; trust the
        # adapter rather than re-pasting from ``pairs`` (the values
        # must match, but the adapter is the source of truth).
        outer.update(1)
        return df

    parts = await asyncio.gather(*(_one(r) for r in rows))
    if not parts:
        return empty_klines_df()
    return pl.concat(parts)


async def retrieve(
    client: AsyncClient,
    bar: tqdm.tqdm,
    ex: Exchange,
    num_bars: int,
    assets: list[str],
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Fetch all pairs for ``ex`` and all daily klines for those pairs
    over the last ``num_bars`` days. Returns ``(klines, pairs)``.
    """
    # 1. pairs (with retries on transient errors)
    logger.info("[%s] fetching pairs", ex.NAME)
    pairs = await ex.pairs_with_retry(
        client, quote_assets=set(a.lower() for a in assets)
    )
    # The kline pass will iterate per (symbol, base, quote). Re-allocate
    # the outer bar to the right total so progress is meaningful.
    n_pairs = pairs.height
    logger.info("[%s] %d pairs, fetching %d bars each", ex.NAME, n_pairs, num_bars)
    if n_pairs == 0:
        return empty_klines_df(), pairs

    # 2. klines
    # ``end`` is the start of the next not-yet-open daily bar, so the
    # in-progress (still-accumulating) candle is excluded by the
    # half-open ``[start, end)`` contract. Per-exchange alignment
    # (HTX: 16:00 UTC, others: 00:00 UTC) is handled by
    # ``Exchange.closed_klines_end``.
    end = ex.closed_klines_end(dt.datetime.now(dt.timezone.utc))
    start = end - dt.timedelta(days=num_bars)
    klines = await _klines_for_pairs(client, ex, pairs, start, end, bar)
    return klines, pairs


async def amain() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Fetch spot pairs, borrow rates, and daily klines for "
            "HTX, KuCoin, Kraken, and Binance."
        )
    )
    p.add_argument(
        "-n",
        "--num-bars",
        help="Number of days to fetch per pair.",
        type=int,
        default=1000,
    )
    p.add_argument(
        "-k",
        "--output-klines",
        help="Destination for raw klines (parquet).",
        type=str,
        default="klines.parquet",
    )
    p.add_argument(
        "-s",
        "--output-symbols",
        type=str,
        help="Destination for symbols and their borrow rates (parquet).",
        default="symbols.parquet",
    )
    p.add_argument(
        "-q",
        "--quote-assets",
        type=str,
        default="USD,USDT,USD1",
        help="Quote assets to fetch klines for. Comma separated list.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )

    args = p.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    assets = [a.strip() for a in args.quote_assets.split(",") if a.strip()]
    exchanges = _build_exchanges()

    # The progress bar is shared across all exchanges; each completed
    # pair-symbol-kline unit ticks it by 1. The total grows as each
    # exchange reports its pair count.
    bar = tqdm.tqdm(desc="klines", unit="pair", dynamic_ncols=True)

    # ``Limits(max_connections=100, max_keepalive_connections=20)`` is
    # the httpx default; keep it that way and let
    # ``Exchange.KLINES_CONCURRENCY`` cap in-flight tasks per
    # exchange.
    async with AsyncClient(timeout=30.0) as client:
        results = await asyncio.gather(
            *(retrieve(client, bar, ex, args.num_bars, assets) for ex in exchanges),
            return_exceptions=True,
        )

    # Separate successes from failures so one bad exchange doesn't
    # sink the whole run.
    klines_parts: list[pl.DataFrame] = []
    pairs_parts: list[pl.DataFrame] = []
    for ex, res in zip(exchanges, results):
        if isinstance(res, BaseException):
            logger.error("[%s] failed: %r", ex.NAME, res)
            continue
        kl, pp = res
        klines_parts.append(kl)
        pairs_parts.append(pp)

    bar.close()

    klines_df = pl.concat(klines_parts) if klines_parts else empty_klines_df()
    pairs_df = pl.concat(pairs_parts) if pairs_parts else pl.DataFrame()

    logger.info(
        "writing %d kline rows to %s, %d pair rows to %s",
        klines_df.height,
        args.output_klines,
        pairs_df.height,
        args.output_symbols,
    )
    klines_df.write_parquet(args.output_klines)
    pairs_df.write_parquet(args.output_symbols)
    return 0


def main() -> int:
    with logging_redirect_tqdm():
        try:
            return asyncio.run(amain())
        except KeyboardInterrupt:
            logger.warning("interrupted")
            return 130
        except Exception:
            logger.exception("fatal error during sync")
            return 1


if __name__ == "__main__":
    sys.exit(main())
