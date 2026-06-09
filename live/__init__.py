"""Exchange adapters for spot pairs, borrow rates, and daily klines.

This package exposes four concrete exchange adapters (HTX, KuCoin,
Kraken, Binance) and the abstract ``Exchange`` base class. Each
adapter implements:

  * ``pairs(client, quote_assets)``
        Returns active spot pairs with cross/isolated margin borrow
        rates. Schema: ``ts, symbol, exchange, base, quote,
        cross_rate, isolated_rate``.

  * ``klines(client, symbol, start_time, end_time)``
        Returns at most ``MAX_KLINES`` daily ohlcv klines in the
        half-open range ``[start_time, end_time)`` (start inclusive,
        end exclusive). Schema: ``open_ts, close_ts, symbol,
        exchange, base, quote, open, high, low, close, base_volume,
        quote_volume``.

The ``MAX_KLINES`` constant on each adapter describes the per-call
cap. Use ``klines_paged()`` (inherited from ``Exchange``) to fetch a
range wider than that cap; it splits the range into ``MAX_KLINES``-sized
chunks and calls ``klines()`` in parallel.
"""

from abc import ABC, abstractmethod
import asyncio
import datetime as dt

import polars as pl
from httpx import AsyncClient


__all__ = [
    "Exchange",
    "HTX",
    "KuCoin",
    "Kraken",
    "Binance",
]


def _split_symbol(symbol: str) -> tuple[str, str]:
    """Split a per-exchange symbol into ``(base, quote)`` in lower case.

    Handles both the HTX-style concatenated form (``btcusdt``) and the
    dash-separated form used by KuCoin (``BTC-USDT``). Quote assets are
    assumed to be 4 characters long (USDT, USDC, BTC, ETH, ...).
    """
    upper = symbol.upper()
    if "-" in upper:
        base, _, quote = upper.partition("-")
        return base.lower(), quote.lower()
    if len(upper) <= 4:
        raise ValueError(f"Cannot split symbol {symbol!r} into base/quote")
    return upper[:-4].lower(), upper[-4:].lower()


class Exchange(ABC):
    """Abstract base class for spot exchange adapters.

    Concrete subclasses must implement ``pairs()`` and ``klines()``, set
    a class-level ``MAX_KLINES`` (the per-call daily-candle cap, also
    used as the chunk size by ``klines_paged()``), and override the
    abstract ``NAME`` used as the ``exchange`` column value.
    """

    NAME: str = ""
    # Max daily candles a single ``klines()`` call can return. The constant
    # describes the API's per-page cap. Use ``klines_paged()`` to fetch
    # a range wider than this window.
    MAX_KLINES: int = 0
    DAILY_SECONDS: int = 24 * 60 * 60

    @abstractmethod
    async def pairs(self, client: AsyncClient, quote_assets: set[str]) -> pl.DataFrame:
        """Return active spot pairs for ``quote_assets`` with margin rates.

        Schema: ``ts`` (fetch time), ``symbol`` (per-exchange proprietary),
        ``exchange``, ``base`` (lower case), ``quote`` (lower case),
        ``cross_rate`` (annual borrow rate for cross margin, ``None`` if
        the asset isn't margin-tradeable), ``isolated_rate`` (same, for
        isolated margin).
        """
        pass

    @abstractmethod
    async def klines(
        self,
        client: AsyncClient,
        symbol: str,
        start_time: dt.datetime,  # inclusive
        end_time: dt.datetime,  # exclusive
    ) -> pl.DataFrame:
        """Return at most ``MAX_KLINES`` daily ohlcv klines for ``symbol``
        in the half-open range ``[start_time, end_time)``.

        Schema: ``open_ts``, ``close_ts`` (inclusive last instant of the
        daily candle), ``symbol``, ``exchange``, ``base``, ``quote``,
        ``open``, ``high``, ``low``, ``close``, ``base_volume``,
        ``quote_volume``.

        ``start_time`` is inclusive and ``end_time`` is exclusive: only
        candles whose ``open_ts`` satisfies ``start_time <= open_ts <
        end_time`` are returned. The result is capped at ``MAX_KLINES``
        candles; use ``klines_paged()`` for a wider range.
        """
        pass

    async def klines_paged(
        self,
        client: AsyncClient,
        symbol: str,
        start_time: dt.datetime,
        end_time: dt.datetime,
    ) -> pl.DataFrame:
        """Fetch daily klines for an arbitrary range by splitting into
        ``MAX_KLINES``-sized chunks and calling ``self.klines()`` in
        parallel. Each chunk is a half-open sub-range of the requested
        window, and the results are concatenated and sorted by
        ``open_ts``.
        """
        if self.MAX_KLINES <= 0:
            raise RuntimeError(
                f"{type(self).__name__} has no MAX_KLINES defined; cannot paginate"
            )
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=dt.timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=dt.timezone.utc)
        if start_time >= end_time:
            return await self.klines(client, symbol, start_time, end_time)

        chunk_seconds = self.MAX_KLINES * self.DAILY_SECONDS
        # Walk the range from the end backwards in ``chunk_seconds``
        # windows; each window is a half-open ``[chunk_start, chunk_end)``
        # slice that fits in one ``klines()`` call.
        sections: list[tuple[dt.datetime, dt.datetime]] = []
        cursor = end_time
        while cursor > start_time:
            chunk_end = cursor
            chunk_start = max(start_time, cursor - dt.timedelta(seconds=chunk_seconds))
            sections.append((chunk_start, chunk_end))
            cursor = chunk_start

        parts = await asyncio.gather(
            *(self.klines(client, symbol, s, e) for s, e in sections)
        )
        if not parts:
            return await self.klines(client, symbol, start_time, end_time)
        return pl.concat(parts).sort("open_ts")


# Concrete adapters. Imported after ``Exchange`` is defined to avoid
# a circular import: each adapter does ``from . import _split_symbol,
# Exchange`` at module load time.
from .htx import HTX  # noqa: E402
from .kucoin import KuCoin  # noqa: E402
from .kraken import Kraken  # noqa: E402
from .binance import Binance  # noqa: E402
