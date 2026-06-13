from abc import ABC, abstractmethod
import polars as pl
import datetime as dt

import sys

from tqdm import tqdm
from dataclasses import dataclass

from typing import Tuple

from . import Universe, AlphaModel, Signal


class Rank(AlphaModel):
    def __init__(self, signal: pl.Expr, gates: Tuple[pl.Expr, pl.Expr]):
        self.signal = signal
        self.gates = gates

    def __call__(self, history: pl.DataFrame, u: Universe) -> list[Signal]:
        today = u.df()[u.timestamp_col()].max()
        long = (
            u.df()
            .filter(self.gates[0] & (pl.col(u.timestamp_col()) == today))
            .sort(self.signal, descending=True)
        )
        short = (
            u.df()
            .filter(self.gates[1] & (pl.col(u.timestamp_col()) == today))
            .sort(self.signal, descending=False)
        )

        return [
            *[
                Signal(
                    symbol=row["symbol"],
                    bullish=True,
                    confidence=(len(long) - i) / len(long),
                )
                for i, row in enumerate(long.iter_rows(named=True))
            ],
            *[
                Signal(
                    symbol=row["symbol"],
                    bullish=False,
                    confidence=(len(short) - i) / len(short),
                )
                for (i, row) in enumerate(short.iter_rows(named=True))
            ],
        ]
