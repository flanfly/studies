from abc import ABC, abstractmethod
import polars as pl
import datetime as dt
from typing import Tuple, Dict, List
from platformdirs import user_cache_dir
from pathlib import Path
from hashlib import sha256
import yfinance as yf
from tqdm import tqdm
import pyarrow.parquet as pq
import sys
from tqdm import tqdm
from dataclasses import dataclass
import logging as l

from . import Universe


class Manual(Universe):
    def __init__(
        self,
        df: pl.DataFrame,
        timestamp_col: str = "ts",
        symbol_col: str = "symbol",
        price_col: str = "close",
        high_col: str | None = None,
        low_col: str | None = None,
        volume_col: str = "volume",
    ):
        self._df = df
        self._timestamp_col = timestamp_col
        self._symbol_col = symbol_col
        self._price_col = price_col
        self._volume_col = volume_col
        self._high_col = high_col
        self._low_col = low_col

    def df(self) -> pl.DataFrame:
        return self._df

    def timestamp_col(self) -> str:
        return self._timestamp_col

    def symbol_col(self) -> str:
        return self._symbol_col

    def price_col(self) -> str:
        return self._price_col

    def low_col(self) -> str | None:
        return self._low_col

    def high_col(self) -> str | None:
        return self._high_col

    def volume_col(self) -> str:
        return self._volume_col

    def until(self, now: dt.datetime) -> "Universe":
        return Manual(
            self._df.filter(pl.col(self._timestamp_col) <= now),
            timestamp_col=self._timestamp_col,
            symbol_col=self._symbol_col,
            price_col=self._price_col,
            high_col=self._high_col,
            low_col=self._low_col,
            volume_col=self._volume_col,
        )


class YFinance(Universe):
    def __init__(self, tickers: list[str], start: int | str | dt.datetime | None):
        self.tickers = tickers
        if isinstance(start, int):
            self.start = dt.datetime(day=1, month=1, year=start)
        elif isinstance(start, str):
            self.start = dt.datetime.strptime(start, "%Y-%m-%d")
        elif isinstance(start, dt.datetime):
            self.start = start
        else:
            raise TypeError(f"Unsupported type for start: {type(start)}")

        # Hash includes the source file so any change to the download
        # / filter logic invalidates stale cache files automatically.
        src_hash = sha256(Path(__file__).read_bytes()).hexdigest()[:8]
        snowflake = sha256(
            (
                f"{'-'.join(sorted(self.tickers))}|"
                f"{self.start.strftime('%Y-%m-%d')}|"
                f"{src_hash}"
            ).encode("utf-8")
        ).hexdigest()
        cachdir = Path(user_cache_dir("backtest", "seu"))
        cachdir.mkdir(parents=True, exist_ok=True)
        self.path = cachdir / f"yf-{snowflake[:6]}.parquet"

        self._df = None

    def _download(self, fd):
        writer = None
        for t in tqdm(self.tickers, desc="downloading data", unit="ticker"):
            pddf = yf.download(
                t,
                period="max",
                interval="1d",
                multi_level_index=False,
                auto_adjust=True,
                progress=False,
            )
            if pddf is None or pddf.empty:
                l.warning(f"No data for {t}, skipping")
                continue

            df = (
                pl.from_pandas(pddf.reset_index())
                .select(
                    open=pl.col("Open").cast(pl.Float32),
                    high=pl.col("High").cast(pl.Float32),
                    low=pl.col("Low").cast(pl.Float32),
                    close=pl.col("Close").cast(pl.Float32),
                    volume=pl.col("Volume").cast(pl.Int64),
                    ts=pl.col("Date").cast(pl.Datetime),
                    symbol=pl.lit(t.upper()).cast(pl.Utf8),
                )
                .sort("ts")
                .filter(pl.col("ts") >= self.start)
            )

            table = df.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(
                    fd,
                    table.schema,
                    compression="zstd",
                )
            writer.write_table(table)
            df = None
            pddf = None

        if writer is not None:
            writer.close()

    def df(self) -> pl.DataFrame:
        if self._df is not None:
            return self._df

        if self.path.is_file():
            mt = dt.datetime.fromtimestamp(
                self.path.stat().st_mtime, tz=dt.timezone.utc
            )
            if (dt.datetime.now(tz=dt.timezone.utc) - mt) < dt.timedelta(hours=24):
                try:
                    self._df = pl.read_parquet(self.path)
                    # Re-apply the start filter on read so cache files
                    # downloaded before the filter existed don't leak
                    # pre-start history into the universe.
                    self._df = self._df.filter(pl.col("ts") >= self.start)
                    return self._df
                except Exception as e:
                    l.error(e)
                    self.path.unlink()

        with open(self.path, "wb") as fd:
            self._download(fd)

        self._df = pl.read_parquet(self.path)
        return self._df

    def timestamp_col(self) -> str:
        return "ts"

    def symbol_col(self) -> str:
        return "symbol"

    def price_col(self) -> str:
        return "close"

    def low_col(self) -> str | None:
        return "low"

    def high_col(self) -> str | None:
        return "high"

    def volume_col(self) -> str:
        return "volume"

    def until(self, now: dt.datetime) -> "Universe":
        return Manual(
            self.df().filter(pl.col("ts") <= now),
            timestamp_col="ts",
            symbol_col="symbol",
            price_col="close",
            volume_col="volume",
        )


import asyncio
from httpx import AsyncClient
from tqdm.asyncio import tqdm


class Binance(Universe):
    def __init__(
        self,
        quote_symbol: str = "USDT",
        look_back: int = 14,
        _df: pl.DataFrame | None = None,
    ):
        self._lb = look_back
        self._quote = quote_symbol
        self._df = _df

    def df(self) -> pl.DataFrame:
        if self._df is None:
            self._fetch()
        return self._df

    def _fetch(self):
        from tqdm.contrib.logging import logging_redirect_tqdm

        with logging_redirect_tqdm():
            asyncio.run(self._do_fetch())

    async def _do_fetch(self):
        async with AsyncClient() as client:
            symbols = await self._get_pairs(client)

            fut = [self._get_klines(client, sym) for sym in symbols]
            res = await tqdm.gather(*fut, desc="fetching klines")

            self._df = pl.concat([df for df in res if df is not None])

    async def _get_pairs(self, c: AsyncClient) -> list[str]:
        resp = await c.get("https://api.binance.com/api/v3/exchangeInfo")
        resp.raise_for_status()

        return [
            s["symbol"]
            for s in resp.json()["symbols"]
            if s["quoteAsset"] == self._quote and s["status"] == "TRADING"
        ]

    async def _get_klines(self, c: AsyncClient, pair: str) -> pl.DataFrame | None:
        p = {
            "symbol": pair,
            "interval": "1d",
            "limit": self._lb + 1,
        }
        resp = await c.get("https://api.binance.com/api/v3/klines", params=p)
        resp.raise_for_status()

        schema = {
            "open_ts": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "base_volume": pl.Float64,
            "close_ts": pl.Int64,
            "quote_volume": pl.Float64,
            "trades": pl.Int64,
            "taker_buy_base_volume": pl.Float64,
            "taker_buy_quote_volume": pl.Float64,
            "ignore": pl.Float64,
        }

        epoch_s_ms_threshold = 10_000_000_000
        epoch_ms_us_threshold = 20_000_000_000_000
        df = pl.DataFrame(resp.json(), orient="row", schema=schema).select(
            ts=pl.when(pl.col("open_ts") > epoch_ms_us_threshold)
            .then(pl.from_epoch("open_ts", time_unit="us"))
            .when(pl.col("open_ts") > epoch_s_ms_threshold)
            .then(pl.from_epoch("open_ts", time_unit="ms"))
            .otherwise(pl.from_epoch("open_ts", time_unit="s"))
            .dt.replace_time_zone("UTC"),
            symbol=pl.lit(pair),
            open=pl.col("open"),
            high=pl.col("high"),
            low=pl.col("low"),
            close=pl.col("close"),
            volume=pl.col("quote_volume"),
        )

        return df

    def timestamp_col(self) -> str:
        return "ts"

    def symbol_col(self) -> str:
        return "symbol"

    def price_col(self) -> str:
        return "open"

    def volume_col(self) -> str:
        return "volume"

    def until(self, now: dt.datetime) -> "Universe":
        return Binance(
            look_back=self._lb,
            quote_symbol=self._quote,
            _df=self.df().filter(pl.col("ts") <= now),
        )
