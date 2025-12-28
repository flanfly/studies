from binance_common.configuration import ConfigurationRestAPI
from binance_common.constants import SPOT_REST_API_PROD_URL
from binance_sdk_spot.spot import Spot
from binance_sdk_spot.rest_api.models import ExchangeInfoResponse, Ticker24hrResponse

import numpy as np
from tabulate import tabulate
from tqdm import tqdm
import pandas as pd
import polars as pl
import dotenv
import matplotlib.pyplot as plt

import re
import logging as l
import os
import itertools as it
import functools as fc
from datetime import datetime

l.basicConfig(level=l.INFO)
dotenv.load_dotenv()

stables = set(
    [
        "USDT",
        "BUSD",
        "USDC",
        "PAX",
        "PAXG",
        "TUSD",
        "DAI",
        "USDP",
        "UST",
        "FDUSD",
        "USD1",
        "EURC",
        "EURI",
        "XUSD",
        "AEUR",
        "USD",
        "EUR",
        "GBP",
        "JPY",
        "AUD",
        "CAD",
        "CHF",
        "NZD",
        "KRW",
    ]
)


def main():
    p = f'cache-{datetime.now().strftime("%Y-%m-%d")}.parquet'
    if os.path.exists(p):
        l.info(f"Loading cache file {p}")
        df = pl.read_parquet(p)
    else:
        df = retrieve_data()
        df.write_parquet(p)
        l.info(f"Wrote cache file {p}")

    build_portfolio(df)


def retrieve_data():
    key = os.getenv("BINANCE_API_KEY")
    if not key:
        l.error("BINANCE_API_KEY environment variable not set")
        return
    secret = os.getenv("BINANCE_API_SECRET")
    if not secret:
        l.error("BINANCE_API_SECRET environment variable not set")
        return
    cfg = ConfigurationRestAPI(
        api_key=key, api_secret=secret, base_path=SPOT_REST_API_PROD_URL
    )
    client = Spot(config_rest_api=cfg).rest_api

    info: ExchangeInfoResponse = client.exchange_info().data()
    pairs = dict(
        map(
            lambda s: [s.symbol, s.base_asset],
            filter(
                lambda s: s.is_spot_trading_allowed
                and s.quote_asset in stables
                and s.base_asset not in stables,
                info.symbols,
            ),
        )
    )

    vol24h: dict[str, float] = {}
    for batch in it.batched(tqdm(pairs.keys(), desc="Fetching 24h volumes"), n=80):
        day: Ticker24hrResponse = (
            client.ticker24hr(symbols=list(batch), type="MINI").data().to_dict()
        )
        vol24h.update({s.symbol: s.quote_volume for s in day})

    assert len(vol24h) == len(pairs)
    assert all(map(lambda p: p in vol24h, pairs))

    coins = dict(
        map(
            lambda sym: [
                pairs[sym],
                sym,
            ],
            sorted(pairs.keys(), key=lambda s: vol24h[s]),
        )
    )

    df_list = []
    for c in tqdm(coins.values(), desc="Fetching historical klines"):
        candles = client.klines(symbol=c, interval="1d", limit=60 + 365).data()
        dfcoin = pl.DataFrame(
            [row[0:1] + row[4:6] for row in candles],
            schema={
                "ts": pl.Int64,
                "close": pl.Float64,
                "volume": pl.Float64,
            },
            orient="row",
        ).with_columns(
            [
                pl.col("close").cast(pl.Float64, strict=False),
                pl.col("volume").cast(pl.Float64, strict=False),
                pl.col("ts").cast(pl.Datetime("ms", "UTC")),
                pl.lit(c.lower()).alias("symbol"),
            ]
        )

        df_list.append(dfcoin)

    return pl.concat(df_list)


def build_portfolio(df: pd.DataFrame):
    today = pd.Timestamp.now(tz="UTC").normalize()
    yesterday = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).normalize()
    labels = list(map(lambda n: f"{n}", range(1, 11)))

    btcregexp = re.compile("^btc(" + "|".join(stables).lower() + ")$")
    btcpair = list(
        filter(lambda s: btcregexp.match(s), df["symbol"].unique().to_list())
    )[0]
    if not btcpair:
        l.error("No BTC stablecoin pair found in data")
        return

    ref = df.filter(pl.col("symbol") == btcpair).select(
        [
            pl.col("ts"),
            (pl.col("close") / pl.col("close").shift(1)).log().alias("ref"),
        ]
    )

    df = (
        df.sort(["symbol", "ts"])
        .filter(
            [
                pl.col("ts") <= today,
                pl.col("ts") >= (today - pd.Timedelta(days=365 + 60)),
                pl.col("volume").rolling_mean(365) < 100e6,
            ]
        )
        .join(
            ref.select([pl.col("ts"), pl.col("ref")]),
            on="ts",
            how="left",
        )
        .with_columns(
            [
                # yesterday's close as today's open (24h market)
                pl.col("close").shift(1).over("symbol").alias("open"),
                pl.col("volume").shift(1).alias("vol"),
            ]
        )
        .with_columns(
            [
                # today's log return
                (pl.col("close") / pl.col("open")).log().alias("ret"),
                # 15-day volume level
                (pl.col("volume") / pl.col("volume").rolling_mean(15))
                .log()
                .over("symbol")
                .alias("vol15d"),
                # 30d momentum
                (pl.col("open") / pl.col("open").shift(30).over("symbol"))
                .log()
                .alias("mom30d"),
                # 1d momentum
                (pl.col("open") / pl.col("open").shift(1).over("symbol"))
                .log()
                .alias("mom1d"),
            ]
        )
        .with_columns(
            [
                (
                    (
                        pl.col("vol15d")
                        - pl.col("vol15d").rolling_mean(window_size=365).over("symbol")
                    )
                    / pl.col("vol15d").rolling_std(window_size=365).over("symbol")
                ).alias("vol15d"),
                (
                    (
                        pl.col("mom30d")
                        - pl.col("mom30d").rolling_mean(window_size=365).over("symbol")
                    )
                    / pl.col("mom30d").rolling_std(window_size=365).over("symbol")
                ).alias("mom30d"),
                (
                    (
                        pl.col("mom1d")
                        - pl.col("mom1d").rolling_mean(window_size=365).over("symbol")
                    )
                    / pl.col("mom1d").rolling_std(window_size=365).over("symbol")
                ).alias("mom1d"),
            ]
        )
    )

    res, df = run_strategy(df)

    resyd = res.loc[yesterday]
    perfyd = np.exp(resyd["perf20d"]) - 1
    perfdx = perfyd - (np.exp(res.loc[yesterday - pd.Timedelta(days=1), "perf20d"]) - 1)
    print(f"20-day performance: {perfyd:.2%} ({perfdx:.2%} change)")

    dftd = df.filter(pl.col("ts") == today).to_pandas()
    dftd["bto"] = (10_000 / dftd.close) * dftd.weight
    dftd.sort_values("weight", ascending=False, inplace=True)

    dftd = dftd[["symbol", "bto", "close", "weight"]]
    dftd["bto"] = dftd["bto"].map("{:,.2f}".format)
    dftd["close"] = dftd["close"].map("{:,.4f}".format)
    dftd["weight"] = dftd["weight"].map("{:.1%}".format)

    print(f"Portfolio weights for {today.date()}:")
    print(tabulate(dftd, headers="keys", tablefmt="psql", showindex=False))


# unmodified code from the notebook
# expects ts, symbol, mom1d, mom30d, ret, vol, vol15d, ref
def run_strategy(df: pl.DataFrame):
    df = (
        df.with_columns(
            [
                pl.col("vol15d")
                .qcut(
                    10,
                    allow_duplicates=True,
                    labels=list(map(lambda n: f"{n}", range(1, 11))),
                )
                .over("ts")
                .alias("volrank"),
                pl.col("mom30d")
                .abs()
                .qcut(
                    10,
                    allow_duplicates=True,
                    labels=list(map(lambda n: f"{n}", range(1, 11))),
                )
                .over("ts")
                .alias("longrank"),
                pl.col("mom1d")
                .qcut(
                    10,
                    allow_duplicates=True,
                    labels=list(map(lambda n: f"{n}", range(1, 11))),
                )
                .over("ts")
                .alias("rank"),
            ]
        )
        .filter(
            [
                (pl.col("volrank").is_in(["10"]).not_())
                & (pl.col("longrank").is_in(["8", "9", "10"]).not_())
            ]
        )
        .filter(
            [
                pl.col("rank") == "1",
            ]
        )
        .with_columns(
            [(pl.col("vol") / pl.col("vol").sum().over(["ts"])).alias("weight")]
        )
        .sort(["ts", "symbol"])
        .with_columns(
            [
                pl.col("weight").alias("initial"),
                (pl.col("weight") * pl.col("ret").exp()).alias("eod"),
            ]
        )
        .with_columns(
            [
                ((pl.col("eod").abs() + pl.col("initial").abs()) * 0.002).alias("fee"),
            ]
        )
        .with_columns(
            [
                (pl.col("eod") - pl.col("initial") - pl.col("fee")).alias("pnl"),
            ]
        )
    )

    res = (
        df.sort("ts")
        .group_by("ts")
        .agg(
            [
                (1 + pl.col("pnl").sum()).log().alias("strategy"),
                pl.col("fee").sum().alias("fee"),
                pl.col("ref").first(),
            ]
        )
        .sort(["ts"])
        .with_columns(
            [
                (pl.col("strategy") - pl.col("ref")).alias("perf"),
                (pl.col("strategy") - pl.col("ref")).rolling_mean(20).alias("perf20d"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("perf20d").shift(1) > 0)
                .then(pl.col("perf"))
                .otherwise(0)
                .alias("cond")
            ]
        )
        .with_columns(
            [
                pl.col("cond").cum_sum().alias("equity"),
            ]
        )
        .to_pandas()
        .set_index("ts")
        .sort_index()
    )

    return res, df


if __name__ == "__main__":
    main()
