from abc import ABC, abstractmethod
import polars as pl
import datetime as dt

import sys

from tqdm import tqdm
from dataclasses import dataclass

from typing import Tuple, Dict, List

from . import Universe


class Manual(Universe):
    def __init__(
        self,
        df: pl.DataFrame,
        timestamp_col: str = "ts",
        symbol_col: str = "symbol",
        price_col: str = "close",
        volume_col: str = "volume",
    ):
        self._df = df
        self._timestamp_col = timestamp_col
        self._symbol_col = symbol_col
        self._price_col = price_col
        self._volume_col = volume_col

    def df(self) -> pl.DataFrame:
        return self._df

    def timestamp_col(self) -> str:
        return self._timestamp_col

    def symbol_col(self) -> str:
        return self._symbol_col

    def price_col(self) -> str:
        return self._price_col

    def volume_col(self) -> str:
        return self._volume_col

    def until(self, now: dt.datetime) -> "Universe":
        return Manual(
            self._df.filter(pl.col(self._timestamp_col) <= now),
            timestamp_col=self._timestamp_col,
            symbol_col=self._symbol_col,
            price_col=self._price_col,
            volume_col=self._volume_col,
        )
