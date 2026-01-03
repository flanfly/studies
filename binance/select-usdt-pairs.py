from binance_common.configuration import ConfigurationRestAPI
from binance_common.constants import SPOT_REST_API_PROD_URL
from binance_sdk_spot.spot import Spot
from binance_sdk_spot.rest_api.models import ExchangeInfoResponse, Ticker24hrResponse

import numpy as np
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

access_key = os.getenv("R2_ACCESS_KEY")
secret_key = os.getenv("R2_SECRET_KEY")
account_id = os.getenv("R2_ACCOUNT_ID")

so = {
    "aws_endpoint_url": f"https://{account_id}.r2.cloudflarestorage.com",
    "aws_access_key_id": access_key,
    "aws_secret_access_key": secret_key,
    "aws_region": "auto",
}

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

key = os.getenv("BINANCE_API_KEY")
if not key:
    l.error("BINANCE_API_KEY environment variable not set")
secret = os.getenv("BINANCE_API_SECRET")
if not secret:
    l.error("BINANCE_API_SECRET environment variable not set")
cfg = ConfigurationRestAPI(
    api_key=key, api_secret=secret, base_path=SPOT_REST_API_PROD_URL
)
client = Spot(config_rest_api=cfg).rest_api

info: ExchangeInfoResponse = client.exchange_info().data()
pairs = dict(
    map(
        lambda s: [s.symbol.lower(), s.base_asset.lower()],
        filter(
            lambda s: s.is_spot_trading_allowed
            and s.quote_asset.lower() == 'usdt'
            and s.base_asset.upper() not in stables
            and not s.base_asset.lower().endswith(('bull', 'bear'), 1),
            info.symbols,
        ),
    )
)

if len(os.sys.argv) != 3:
    print("Usage: select-usdt-pairs.py <input-file> <output-file>")
    os.sys.exit(1)

infile = os.sys.argv[1]
outfile = os.sys.argv[2]

print(pl.read_parquet(infile, storage_options=so).filter([
    pl.col("symbol").is_in(list(pairs.keys()))
]).with_columns([
    pl.col("symbol").alias('pair'),
    pl.col('symbol').replace(pairs).alias("symbol"),
]).sort("ts").head())#.write_parquet(outfile, compression="zstd", storage_options=so)
