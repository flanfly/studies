"""Exchange adapters for spot pairs, borrow rates, and daily klines.

This package exposes six concrete exchange adapters (HTX, KuCoin,
Kraken, Binance, Hyperliquid, AsterDex) and the abstract ``Exchange``
base class. Each adapter implements:

  * ``pairs(client, quote_assets)``
        Returns active spot pairs with cross/isolated margin borrow
        rates and (where the exchange supports it) the current
        perpetual-futures funding rate annualised as APR. Schema:
        ``ts, symbol, exchange, base, quote, cross_rate,
        isolated_rate, funding_rate``. ``funding_rate`` is ``None``
        for exchanges with no native perpetuals (Kraken) or for
        coins that don't have a perpetual contract. Hyperliquid
        is a perpetuals-only DEX, so ``pairs()`` emits one row
        per perp contract (no spot pairs exist) with ``quote =
        "usd"``; ``klines()`` returns the perp's OHLCV rather
        than spot. AsterDex is a Binance-compatible perps DEX
        on BNB Chain, so ``pairs()`` emits one row per perp
        contract with ``quote = "usdt"`` and ``klines()``
        returns perp OHLCV. MEXC's spot v3 API doesn't expose
        a public borrow-rate endpoint, so ``cross_rate`` and
        ``isolated_rate`` are ``None`` for every MEXC pair;
        ``funding_rate`` is populated from the public USDⓈ-M
        funding-rate endpoint when a perpetual contract exists.

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

Transient errors (rate limits, ``5xx`` responses, transport errors,
exchange-specific error codes) are retried automatically with
exponential backoff + jitter. The two public entry points for that
are ``pairs_with_retry()`` and the ``klines()`` calls that
``klines_paged()`` makes internally. ``is_transient_error(exc)`` and
``RETRY_ATTEMPTS``/``RETRY_BASE_DELAY``/``RETRY_MAX_DELAY`` can be
overridden or adjusted per adapter.
"""

from abc import ABC, abstractmethod
import asyncio
import datetime as dt
import logging
import random
from typing import Iterable

import httpx
import polars as pl
from httpx import AsyncClient


logger = logging.getLogger(__name__)


__all__ = [
    "Exchange",
    "HTX",
    "KuCoin",
    "Kraken",
    "Binance",
    "MEXC",
    "Hyperliquid",
    "AsterDex",
    # Schema dicts and helpers shared across adapters and tests.
    "PAIRS_SCHEMA",
    "KLINES_SCHEMA",
    "empty_pairs_df",
    "empty_klines_df",
    "validate_pairs_df",
    "validate_klines_df",
    # Retry policy
    "TransientError",
    "RETRY_ATTEMPTS",
    "RETRY_BASE_DELAY",
    "RETRY_MAX_DELAY",
    "is_transient_http_status",
]


# ---------------------------------------------------------------------
# Shared schema for the two return types.
#
# ``PAIRS_SCHEMA`` is the schema of the DataFrame returned by
# ``Exchange.pairs()``. ``KLINES_SCHEMA`` is the schema of the
# DataFrame returned by ``Exchange.klines()`` (and ``klines_paged``).
# Every adapter builds and validates its output against these, so
# consumers can rely on a uniform schema across exchanges.
# ---------------------------------------------------------------------

PAIRS_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Datetime("us", time_zone="UTC"),
    "symbol": pl.Utf8,
    "exchange": pl.Utf8,
    "base": pl.Utf8,
    "quote": pl.Utf8,
    "cross_rate": pl.Float64,
    "isolated_rate": pl.Float64,
    "funding_rate": pl.Float64,
}


KLINES_SCHEMA: dict[str, pl.DataType] = {
    "open_ts": pl.Datetime("us", time_zone="UTC"),
    "close_ts": pl.Datetime("us", time_zone="UTC"),
    "symbol": pl.Utf8,
    "exchange": pl.Utf8,
    "base": pl.Utf8,
    "quote": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "base_volume": pl.Float64,
    "quote_volume": pl.Float64,
}


def empty_pairs_df() -> pl.DataFrame:
    """Return a 0-row ``pairs`` DataFrame with the canonical schema."""
    return pl.DataFrame(schema=PAIRS_SCHEMA)


def empty_klines_df() -> pl.DataFrame:
    """Return a 0-row ``klines`` DataFrame with the canonical schema."""
    return pl.DataFrame(schema=KLINES_SCHEMA)


def _check_schema(df: pl.DataFrame, expected: dict[str, pl.DataType], name: str) -> None:
    actual = dict(zip(df.columns, df.dtypes))
    if actual != expected:
        raise ValueError(
            f"{name} schema mismatch: got columns/dtypes {actual}, "
            f"expected {expected}"
        )


def validate_pairs_df(df: pl.DataFrame) -> None:
    """Raise ``ValueError`` if ``df`` doesn't have the ``PAIRS_SCHEMA``
    columns and dtypes."""
    _check_schema(df, PAIRS_SCHEMA, "pairs")


def validate_klines_df(df: pl.DataFrame) -> None:
    """Raise ``ValueError`` if ``df`` doesn't have the ``KLINES_SCHEMA``
    columns and dtypes."""
    _check_schema(df, KLINES_SCHEMA, "klines")


# ---------------------------------------------------------------------
# Transient errors and retry policy.
#
# Adapters raise ``TransientError`` (or let an ``httpx.HTTPStatusError``
# propagate) when the cause of the failure is likely to clear up on its
# own: per-IP rate limits, exchange-side 5xx, transport errors, etc.
# The base ``Exchange`` class wraps every public method call in
# ``_retry()`` which re-runs the call with exponential backoff + jitter
# while ``is_transient_error(exc)`` is true.
# ---------------------------------------------------------------------


class TransientError(Exception):
    """Raised by an adapter when a request failed for a reason that's
    likely to clear up on its own (rate limiting, exchange-side
    outage, etc.) and the caller should retry."""


def is_transient_http_status(status: int) -> bool:
    """Return ``True`` for HTTP status codes that warrant a retry:
    ``429`` (rate limit) and any ``5xx`` (server error)."""
    return status == 429 or (isinstance(status, int) and 500 <= status < 600)


# Default retry policy used by ``Exchange._retry`` and applied uniformly
# to ``klines()`` and ``pairs()`` calls.
#
# Backoff is exponential starting at 100ms and capped at 10s. With ±25%
# jitter, the per-attempt delay sequence (before jitter) is:
#   100ms, 200ms, 400ms, 800ms, 1.6s, 3.2s, 6.4s, 10s, 10s, 10s, ...
# (8 doublings from 100ms to 12.8s get capped to 10s at attempt 8).
#
# Attempts are **infinite** (``_retry`` runs ``while True``). This is a
# deliberate "never give up on a transient failure" choice: a sustained
# outage or a misconfigured host will keep retrying every ~10s forever
# rather than propagating an error after a fixed N. Callers that need
# a hard upper bound should cancel the surrounding ``asyncio.Task``.
RETRY_BASE_DELAY = 0.1   # seconds; first backoff is ``base * 2**0``
RETRY_MAX_DELAY = 10.0   # cap on the backoff interval (before jitter)
# ``RETRY_ATTEMPTS`` is kept as a module-level symbol for backward
# compatibility with code that referenced it, but is no longer used
# by ``_retry`` (which now loops ``while True``). Setting it has no
# effect on retry behaviour.
RETRY_ATTEMPTS = None


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

    # Hour-of-day in UTC at which the exchange's daily candle
    # starts. Most exchanges (Binance, KuCoin, Kraken) align to
    # midnight UTC; HTX aligns to 16:00 UTC. ``closed_klines_end()``
    # uses this to compute the latest bar boundary that is strictly
    # in the past (i.e. fully closed) at a given moment.
    DAILY_ALIGN_HOUR_UTC: int = 0

    # Max in-flight ``klines()`` requests for one exchange's pass.
    # Stays comfortably under httpx's default 100-connection pool cap
    # per host, so a large ``pairs`` frame doesn't trigger
    # ``PoolTimeout`` storms. Override on a per-exchange basis if the
    # exchange advertises a higher per-IP limit.
    KLINES_CONCURRENCY: int = 50

    # Schema dicts and helpers exposed on every concrete class as
    # class-level constants/methods so adapters can call
    # ``self.empty_pairs_df()`` and ``self.validate_klines_df(df)``.
    PAIRS_SCHEMA: dict[str, pl.DataType] = PAIRS_SCHEMA
    KLINES_SCHEMA: dict[str, pl.DataType] = KLINES_SCHEMA

    # Default retry policy. ``_retry()`` runs the wrapped call up to
    # ``RETRY_ATTEMPTS`` times, sleeping ``RETRY_BASE_DELAY * 2**(n-1)``
    # seconds (capped at ``RETRY_MAX_DELAY``) between attempts, with
    # ±25% jitter to avoid thundering-herd retries.
    # ``RETRY_ATTEMPTS`` is kept as an ``int | None`` symbol for
    # backward compatibility with code (and tests) that referenced
    # it. The current ``_retry`` implementation is infinite (``while
    # True``), so this attribute is informational only: setting it
    # has no effect on retry behaviour. The active knobs are
    # ``RETRY_BASE_DELAY`` and ``RETRY_MAX_DELAY``.
    RETRY_ATTEMPTS: int | None = RETRY_ATTEMPTS
    RETRY_BASE_DELAY: float = RETRY_BASE_DELAY
    RETRY_MAX_DELAY: float = RETRY_MAX_DELAY

    # ``httpx`` exception types that ``is_transient_error`` treats as
    # transient. Used both for direct checks and for walking
    # ``__cause__``/``__context__`` so that an adapter that wraps an
    # ``httpx`` error in a ``RuntimeError`` is still retried.
    _TRANSIENT_HTTPX: tuple[type[BaseException], ...] = (
        httpx.TimeoutException,
        httpx.ConnectError,
        httpx.ReadError,
        httpx.WriteError,
        httpx.RemoteProtocolError,
        httpx.PoolTimeout,
        httpx.NetworkError,
    )

    @classmethod
    def empty_pairs_df(cls) -> pl.DataFrame:
        return empty_pairs_df()

    @classmethod
    def empty_klines_df(cls) -> pl.DataFrame:
        return empty_klines_df()

    @classmethod
    def validate_pairs_df(cls, df: pl.DataFrame) -> None:
        validate_pairs_df(df)

    @classmethod
    def validate_klines_df(cls, df: pl.DataFrame) -> None:
        validate_klines_df(df)

    async def _fetch_funding_rates(
        self, client: AsyncClient, bases: Iterable[str]
    ) -> dict[str, float]:
        """``{base_lower: annualised_funding_rate}`` for each base asset
        that has a perpetual-futures contract on this exchange.

        The default implementation returns an empty dict: the base
        class is exchange-agnostic and exchanges that don't support
        perpetuals (Kraken) don't override this. Adapters that do
        support perpetuals (Binance, HTX, KuCoin) override this to
        hit their public funding-rate endpoint.

        ``funding_rate`` in the pairs DataFrame is ``None`` for
        coins whose key is missing from the returned dict.
        """
        return {}

    def annualize_funding_rate(
        self, per_interval_rate: float, interval_hours: float
    ) -> float:
        """Convert a single funding payment (paid every
        ``interval_hours`` hours) into an annualised rate (APR).

        ``per_interval_rate`` is the funding rate as reported by
        the exchange for the most recent interval, e.g. ``0.0001``
        (= 1 bp per 8h interval). The result is

            per_interval_rate * (24 / interval_hours) * 365

        For an 8h funding schedule (Binance, HTX, KuCoin) this is
        ``per_interval_rate * 3 * 365 = per_interval_rate * 1095``.
        """
        if interval_hours <= 0:
            raise ValueError(
                f"interval_hours must be > 0, got {interval_hours}"
            )
        return per_interval_rate * (24.0 / interval_hours) * 365.0

    def is_transient_error(self, exc: BaseException) -> bool:
        """Return ``True`` if ``exc`` describes a failure that's likely
        to clear up on its own and is therefore worth retrying.

        The default implementation recognises:

        * ``TransientError`` (raised by adapters)
        * ``httpx.HTTPStatusError`` whose response has status ``429``
          or any ``5xx``
        * ``httpx`` transport-layer errors (``ReadError``,
          ``ConnectError``, ``TimeoutException``, ``PoolTimeout``,
          etc.)

        Adapters that wrap an ``httpx`` exception in a ``RuntimeError``
        for the message do not lose the retry behaviour: the
        classifier walks ``__cause__`` and ``__context__`` before
        giving up.

        Concrete adapters override this to also recognise their own
        exchange-specific error envelopes (e.g. HTX's
        ``err-msg: "request limit"``). Permanent failures like a
        malformed symbol or a bad signature are intentionally NOT
        retried: a 4xx that's not 429 will not be retried.
        """
        seen: set[int] = set()
        cur: BaseException | None = exc
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if isinstance(cur, TransientError):
                return True
            if isinstance(cur, httpx.HTTPStatusError):
                if is_transient_http_status(cur.response.status_code):
                    return True
            elif isinstance(cur, self._TRANSIENT_HTTPX):
                return True
            nxt = cur.__cause__ or cur.__context__
            cur = nxt if nxt is not cur else None
        return False

    async def _retry(self, method_name: str, /, *args, **kwargs):
        """Call ``self.<method_name>(*args, **kwargs)`` with
        exponential backoff and ±25% jitter while
        ``self.is_transient_error(exc)`` is true.

        **Attempts are infinite.** The loop runs ``while True``
        and only stops on a successful return. Non-transient
        exceptions are propagated immediately without further
        attempts; transient exceptions (429, 5xx, transport
        errors, ``TransientError``) trigger an exponential
        backoff starting at ``RETRY_BASE_DELAY`` (100ms by
        default) and capped at ``RETRY_MAX_DELAY`` (10s by
        default).

        This means a sustained outage or a misconfigured host
        (typo in ``HOST``, expired API key, etc.) will keep
        retrying every ~10s forever. Callers that need a hard
        upper bound on retry duration should run the call
        inside an ``asyncio.wait_for`` / task-cancel wrapper.

        Per-attempt delay sequence (before jitter, with
        defaults): 100ms, 200ms, 400ms, 800ms, 1.6s, 3.2s,
        6.4s, **10s**, 10s, 10s, ... (8 doublings from 100ms
        reach 12.8s, which is then capped to 10s on attempt 8
        and beyond).

        ``method_name`` must name an ``async`` method on
        ``self``. The first positional argument is the method
        name; ``*args`` and ``**kwargs`` are forwarded verbatim.
        """
        method = getattr(self, method_name)
        attempt = 0
        while True:
            attempt += 1
            try:
                return await method(*args, **kwargs)
            except BaseException as exc:
                if not self.is_transient_error(exc):
                    raise
                # Exponential backoff with ±25% jitter, capped at
                # ``RETRY_MAX_DELAY``. ``2 ** (attempt - 1)`` doubles
                # on every retry; the cap is reached on the 8th
                # retry at the defaults (100ms → 12.8s → 10s).
                delay = min(
                    self.RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                    self.RETRY_MAX_DELAY,
                )
                delay = delay * (0.75 + random.random() * 0.5)
                logger.warning(
                    "%s.%s transient error (attempt %d): %s; "
                    "retrying in %.1fs (infinite retries)",
                    type(self).__name__,
                    method_name,
                    attempt,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

    @abstractmethod
    async def pairs(
        self,
        client: AsyncClient,
        quote_assets: set[str],
        limit: int | None = None,
    ) -> pl.DataFrame:
        """Return active spot pairs for ``quote_assets`` with margin rates
        and (where applicable) the current perpetual-futures funding
        rate annualised as APR.

        Schema: ``ts`` (fetch time), ``symbol`` (per-exchange proprietary),
        ``exchange``, ``base`` (lower case), ``quote`` (lower case),
        ``cross_rate`` (annual borrow rate for cross margin, ``None`` if
        the asset isn't margin-tradeable), ``isolated_rate`` (same, for
        isolated margin), ``funding_rate`` (annualised current funding
        rate for the perpetual contract on the same base/quote, ``None``
        if the exchange has no native perpetuals -- currently Kraken --
        or if the coin has no perpetual contract).

        ``limit`` is an optional cap on the number of pairs returned:
        when ``None`` (the default), the entire pair universe is
        enumerated. When set to a positive integer, only the first
        ``limit`` pairs (sorted by exchange-specific ordering, usually
        alphabetical by base) are returned, and any per-pair follow-up
        calls (funding rates, etc.) are also bounded. This is useful
        for tests that want to exercise the full ``pairs()`` path on a
        small subset without paying the cost of fetching data for the
        entire exchange. The CLI never passes ``limit``; it always
        fetches the full universe.
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
            df = await self._retry("klines", client, symbol, start_time, end_time)
            self.validate_klines_df(df)
            return df

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
            *(
                self._retry("klines", client, symbol, s, e)
                for s, e in sections
            )
        )
        if not parts:
            df = await self._retry("klines", client, symbol, start_time, end_time)
            self.validate_klines_df(df)
            return df
        out = pl.concat(parts).sort("open_ts")
        self.validate_klines_df(out)
        return out

    async def pairs_with_retry(
        self,
        client: AsyncClient,
        quote_assets: set[str],
        limit: int | None = None,
    ) -> pl.DataFrame:
        """Public wrapper around ``pairs()`` that retries on
        transient errors. Equivalent to ``self._retry("pairs", ...)``
        but exposed as a method so callers don't have to know the
        internal ``_retry`` name.

        ``limit`` is forwarded verbatim to ``pairs()``: when set
        to a positive integer, only the first ``limit`` pairs are
        returned. See ``Exchange.pairs`` for details.
        """
        df = await self._retry("pairs", client, quote_assets, limit)
        self.validate_pairs_df(df)
        return df

    async def klines_with_retry(
        self,
        client: AsyncClient,
        symbol: str,
        start_time: dt.datetime,
        end_time: dt.datetime,
    ) -> pl.DataFrame:
        """Public wrapper around ``klines()`` that retries on
        transient errors. Equivalent to
        ``self._retry("klines", ...)`` but exposed as a method so
        callers don't have to know the internal ``_retry`` name.
        Use this when fetching a single ``MAX_KLINES``-sized window
        of klines; use ``klines_paged()`` for wider ranges (which
        internally retries per page).
        """
        df = await self._retry("klines", client, symbol, start_time, end_time)
        self.validate_klines_df(df)
        return df

    def closed_klines_end(self, now: dt.datetime) -> dt.datetime:
        """Return the ``end_time`` to pass to ``klines()`` /
        ``klines_paged()`` so the in-progress daily candle is
        excluded.

        The ``klines`` contract is half-open ``[start, end)``:
        any candle whose ``open_ts == end`` is dropped. So ``end``
        must equal the *current* in-progress bar's ``open_ts`` --
        which is the most recent past alignment boundary at
        ``DAILY_ALIGN_HOUR_UTC:00:00 UTC``, strictly before ``now``.

        * For midnight-aligned exchanges (Binance, KuCoin, Kraken),
          the result is today's 00:00 UTC if ``now > 00:00``,
          otherwise yesterday's.
        * For HTX (16:00 UTC aligned), the result is 16:00 UTC of the
          current or previous day.

        ``now`` must be timezone-aware (UTC). Naive datetimes are
        treated as UTC.
        """
        if now.tzinfo is None:
            now = now.replace(tzinfo=dt.timezone.utc)
        align_h = self.DAILY_ALIGN_HOUR_UTC
        # ``now``'s alignment boundary (the start of the bar that
        # *contains* ``now``).
        candidate = now.replace(
            hour=align_h, minute=0, second=0, microsecond=0
        )
        # If ``now`` is strictly before the candidate, fall back to
        # the previous day. Use strict ``<`` so that exactly-at-the-
        # boundary ``now`` is treated as inside the new bar (the
        # in-progress bar) and gets excluded.
        if now < candidate:
            candidate = candidate - dt.timedelta(days=1)
        return candidate


# Concrete adapters. Imported after ``Exchange`` is defined to avoid
# a circular import: each adapter does ``from . import _split_symbol,
# Exchange`` at module load time.
from .htx import HTX  # noqa: E402
from .kucoin import KuCoin  # noqa: E402
from .kraken import Kraken  # noqa: E402
from .binance import Binance  # noqa: E402
from .mexc import MEXC  # noqa: E402
from .hyperliquid import Hyperliquid  # noqa: E402
from .asterdex import AsterDex  # noqa: E402
