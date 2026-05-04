import os
import sys
import time
import datetime as dt
import traceback

import polars as pl
import pandas as pd
import numpy as np

import asyncio
import more_itertools as it
from tqdm.asyncio import tqdm

from binance_common.configuration import ConfigurationRestAPI
from binance_common.constants import SPOT_REST_API_PROD_URL
from binance_sdk_spot.spot import Spot
from binance_sdk_spot.rest_api.models import KlinesIntervalEnum

from telegram import Bot

from dotenv import load_dotenv
import logging as l

load_dotenv()

l.basicConfig(
    format="[%(asctime)s] %(levelname)s    %(message)s",
    level=l.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

CONCURRENCY = 10
KLINE_INTERVAL = KlinesIntervalEnum.INTERVAL_15m.value
BUFFER_SIZE = (2 * 24 * 60) // 15  # 2 days of 15-minute klines
UPDATE_INTERVAL = 60  # seconds
QUOTE_ASSET = "USDC"
TELEGRAM_CHAT_ID = "646299665"

KELLY_FRACTION = 0.354  # half kelly
HOLDING_MINUTES = 2400
VOLUME_STDEV_THRESHOLD = 2
MOMENTUM_THRESHOLD = 0.2


binance_connector_sem = asyncio.Semaphore(CONCURRENCY)


def klines_to_df(symbol: str, raw: list[list]) -> pl.DataFrame:
    """Convert raw Binance klines response into a Polars DataFrame."""
    rows = []
    for k in raw:
        rows.append(
            {
                "symbol": symbol,
                "open_time": dt.datetime.fromtimestamp(k[0] / 1000, tz=dt.timezone.utc),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": dt.datetime.fromtimestamp(
                    k[6] / 1000, tz=dt.timezone.utc
                ),
                "quote_vol": float(k[7]),
                "count": int(k[8]),
                "taker_buy_vol": float(k[9]),
                "taker_buy_quote_vol": float(k[10]),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "symbol": pl.Utf8,
            "open_time": pl.Datetime("us", time_zone="UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "close_time": pl.Datetime("us", time_zone="UTC"),
            "quote_vol": pl.Float64,
            "count": pl.Int64,
            "taker_buy_vol": pl.Float64,
            "taker_buy_quote_vol": pl.Float64,
        },
    )


async def step(df: pl.DataFrame) -> str | None:
    last_kline = df.select(pl.col("open_time").max()).item()
    df = (
        df.sort(["symbol", "open_time"])
        .select(
            pl.col("open_time"),
            pl.col("symbol"),
            # rolling volume z-score
            vol=(pl.col("quote_vol") - pl.col("quote_vol").mean().over("symbol"))
            / pl.col("quote_vol").std().over("symbol"),
            # momentum
            mom=pl.col("close").pct_change(1).over("symbol"),
        )
        .with_columns(
            active=(pl.col("mom") >= MOMENTUM_THRESHOLD)
            & (pl.col("vol") > VOLUME_STDEV_THRESHOLD)
        )
        .filter((pl.col("open_time") == last_kline) & pl.col("active"))
    )

    if not df.is_empty():
        print(df)
        return df.select(pl.col("symbol")).to_series().to_list()[0]


async def sell_pair(client: Spot, pair: str) -> None:
    l.info(f"Selling {pair}...")
    pass


async def buy_pair(client: Spot, pair: str, quote_qty: float) -> None:
    l.info(f"Buying ${quote_qty} of {pair}...")

    async with binance_connector_sem:
        try:
            resp = await asyncio.to_thread(
                client.rest_api.new_order,
                symbol=pair,
                side="BUY",
                type="MARKET",
                quote_order_qty=str(quote_qty),
            )
            l.info(f"Buy order response: {resp.data()}")
        except Exception as e:
            l.error(f"Error placing buy order for {pair}: {e}")


async def run() -> None:
    # Initialise Spot client from environment credentials
    config = ConfigurationRestAPI(
        api_key=os.environ["BINANCE_HIFREQ_KEY"],
        api_secret=os.environ["BINANCE_HIFREQ_SECRET"],
        base_path=SPOT_REST_API_PROD_URL,
        # base_path="https://demo-api.binance.com",
    )
    client = Spot(config_rest_api=config)
    bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
    now = None

    while True:
        if now is not None:
            last_iter = dt.datetime.now(tz=dt.timezone.utc) - now
            l.info(f"Last iteration took {last_iter.total_seconds():.2f}s")
            delay = dt.timedelta(seconds=UPDATE_INTERVAL) - last_iter
            if delay.total_seconds() > 0:
                l.info(
                    f"Sleeping for {delay.total_seconds():.2f}s until next iteration..."
                )
                await asyncio.sleep(delay.total_seconds())

        try:
            now = dt.datetime.now(tz=dt.timezone.utc)

            # fetch latest klines
            pairs = await fetch_trading_pairs(client)

            fut = [fetch_klines(client, now, sym) for sym in pairs]
            frag = await tqdm.gather(*fut)
            df = pl.concat([df for df in frag if df is not None])
            latest_kline = df["open_time"].max()

            # selling holdings older than HOLDING_MINUTES
            holdings = await fetch_holdings(client)
            holdings = (
                holdings.with_columns(
                    pair=pl.col("symbol") + QUOTE_ASSET,
                )
                .join(
                    df.filter(pl.col("open_time") == latest_kline).select(
                        pl.col("symbol"), pl.col("close")
                    ),
                    left_on="pair",
                    right_on="symbol",
                    how="left",
                )
                .filter(
                    (
                        pl.col("close").is_null()
                        | (pl.col("free") * pl.col("close") > 1)
                    )  # only count if value > $1
                )
            )

            print(holdings)

            # max_age = now - dt.timedelta(minutes=HOLDING_MINUTES)
            # sell = holdings.filter(pl.col("buy_time") < max_age)
            # if not sell.is_empty():
            #    print(f"Selling {sell.height} holdings...")
            #    fut = [
            #        sell_pair(client, p)
            #        for p in sell.select(pl.col("pair")).to_series().to_list()
            #    ]
            #    await tqdm.gather(*fut)
            #    continue  # skip buying in the same iteration

            # derive trading signals and buy
            pair = await step(df)
            if pair is not None:
                base = pair.removesuffix(QUOTE_ASSET)
                l.info(f"Trying to buy {base}...")

                cash = holdings.filter(pl.col("symbol") == QUOTE_ASSET)
                if cash.is_empty():
                    cash = 0
                else:
                    cash = cash["free"].to_list()[0]

                frac = cash * KELLY_FRACTION
                if frac < 10:  # minimum order size
                    l.info(f"Not enough cash to buy {base}, skipping")
                    continue

                hdf = holdings.with_columns(
                    value=pl.col("free") * pl.col("close")
                ).filter((pl.col("value") > 1) & (pl.col("symbol") == base))

                if not hdf.is_empty():
                    l.info("Already holding assets with >$1 value, skipping buy")
                    continue

                await buy_pair(client, pair, np.trunc(frac))
                async with bot:
                    await bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=f"Bought {base} at {now.isoformat()}",
                    )
                sys.exit(0)

        except Exception as e:
            l.error(f"Error in main loop: {e}")
            traceback.print_exc()
            async with bot:
                await bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=f"Error in main loop: {e}",
                )


async def fetch_holdings(client: Spot) -> pl.DataFrame:
    async with binance_connector_sem:
        l.info("Fetching account holdings...")
        resp = await asyncio.to_thread(
            client.rest_api.get_account, omit_zero_balances=True
        )

    rows = []
    for b in resp.data().balances or []:
        rows.append(
            {
                "symbol": b.asset,
                "free": float(b.free),
                "locked": float(b.locked),
            }
        )

    df = pl.DataFrame(
        rows, schema={"symbol": pl.Utf8, "free": pl.Float64, "locked": pl.Float64}
    )

    fut = [
        fetch_trades(client, f"{r['symbol']}{QUOTE_ASSET}")
        for r in rows
        if r["free"] > 0 and r["symbol"] != QUOTE_ASSET
    ]
    if len(fut) > 0:
        trades = [t for t in await tqdm.gather(*fut) if t is not None]
        if len(trades) > 0:
            return (
                df.join(
                    pl.concat(trades)
                    .filter(pl.col("is_buyer"))
                    .select(
                        symbol=pl.col("symbol").str.replace(QUOTE_ASSET, ""),
                        buy_price=pl.col("price"),
                        buy_time=pl.col("time"),
                    ),
                    on="symbol",
                    how="left",
                )
                .sort(["symbol", "buy_time"])
                .group_by("symbol")
                .last()
            )

    return df.with_columns(
        pl.lit(None).cast(pl.Float64).alias("buy_price"),
        pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("buy_time"),
    ).sort("symbol")


async def fetch_trades(client: Spot, pair: str) -> pl.DataFrame | None:
    async with binance_connector_sem:
        try:
            resp = await asyncio.to_thread(
                client.rest_api.my_trades, symbol=pair, limit=1000
            )
        except Exception as e:
            l.error(f"Error fetching trades for {pair}: {e}")
            return None

    rows = []
    for t in resp.data():
        rows.append(
            {
                "symbol": t.symbol,
                "time": dt.datetime.fromtimestamp(t.time / 1000, tz=dt.timezone.utc),
                "price": float(t.price),
                "is_buyer": t.is_buyer,
                "qty": float(t.qty),
                "quote_qty": float(t.quote_qty),
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "symbol": pl.Utf8,
            "time": pl.Datetime("us", time_zone="UTC"),
            "price": pl.Float64,
            "is_buyer": pl.Boolean,
            "qty": pl.Float64,
            "quote_qty": pl.Float64,
        },
    )


async def fetch_trading_pairs(client: Spot) -> list[str]:
    async with binance_connector_sem:
        l.info("Fetching exchange info...")
        exchange_resp = await asyncio.to_thread(client.rest_api.exchange_info)

    return [
        s.symbol
        for s in exchange_resp.data().symbols or []
        if s.quote_asset == QUOTE_ASSET
        and s.status == "TRADING"
        and s.is_spot_trading_allowed
    ]


async def fetch_klines(client: Spot, now: dt.datetime, sym: str) -> pl.DataFrame | None:
    period_s = pd.to_timedelta(KLINE_INTERVAL).total_seconds()
    start_time = now - dt.timedelta(seconds=period_s * BUFFER_SIZE)

    async with binance_connector_sem:
        resp = await asyncio.to_thread(
            client.rest_api.klines,
            symbol=sym,
            interval=KLINE_INTERVAL,
            start_time=int(start_time.timestamp() * 1000),
            limit=int(BUFFER_SIZE + 1),
        )
        raw = resp.data()

    if not raw:
        l.warning(f"Empty klines response for {sym}")
        return None

    return klines_to_df(sym, raw).filter(pl.col('close_time') <= now)




if __name__ == "__main__":
    asyncio.run(run())
