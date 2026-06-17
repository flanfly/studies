"""MEXC spot exchange adapter.

MEXC's spot market endpoints are unauthenticated. The
``/api/v3/exchangeInfo`` and ``/api/v3/klines`` endpoints work
without API keys.

Public endpoints used:

  * ``GET /api/v3/exchangeInfo``                 -- pair list
  * ``GET /api/v3/klines``                       -- daily candles
  * ``GET /api/v1/contract/funding_rate/{contract}``
      -- current funding rate of a USDT-margined
         linear-swap perpetual

MEXC does not currently expose a public spot-margin borrow-rate
endpoint in v3 of their API (only authenticated ``sapi`` endpoints
that return order-side margin info). Borrow rates are therefore
left as ``None`` for every MEXC pair, matching the convention used
on Kraken. If/when MEXC adds a public borrow-rate endpoint, both
``cross_rate`` and ``isolated_rate`` can be populated from it.

Kline response: ``[open_ts, open, high, low, close, base_volume,
close_ts, quote_volume]`` -- open_ts and close_ts are
millisecond-resolution epoch (or close_ts is just open_ts +
24h-1ms, identical to HTX/KuCoin/Kraken). MEXC returns midnight-
UTC daily candles (verified live: a candle with ``open_ts ==
1700006400000`` corresponds to ``2023-11-15 00:00:00 UTC``).

One quirk of MEXC's kline endpoint: it ignores ``startTime`` and
``endTime`` entirely when ``limit`` is set, and returns the
**most recent** ``limit`` candles. To walk further back than
``limit`` candles, we have to issue successive calls and trim the
overlap. The kline fetch loop in :meth:`klines` handles this
manually because the standard ``klines_paged`` chunked-by-time
logic doesn't fit MEXC's "no date params" semantics.
"""

import asyncio
import datetime as dt
import logging as l
from typing import Iterable

import httpx
import polars as pl
from httpx import AsyncClient

from . import _split_symbol, Exchange


__all__ = ["MEXC"]


class MEXC(Exchange):
    """MEXC spot exchange adapter.

    Public-only: no API key required. Borrow rates are not
    populated (MEXC's spot v3 API doesn't expose a public
    borrow-rate endpoint); only funding rates are populated
    from the public ``/api/v1/contract/funding_rate`` endpoint.
    """

    HOST = "api.mexc.com"
    # Spot and futures funding live on the same host -- no
    # separate domain. We use ``HOST`` for both.
    NAME = "mexc"

    # MEXC kline endpoint caps ``limit`` at 500 in practice: the
    # docs say "Default 500; max 1000", but requesting ``limit=1000``
    # returns 500 rows regardless. We use 500 as the actual cap.
    MAX_KLINES = 500

    # MEXC's daily candles open at midnight UTC (verified live).
    # This matches the Binance / KuCoin / Kraken convention.
    DAILY_ALIGN_HOUR_UTC = 0

    # MEXC USDⓈ-M perpetuals pay funding every 8h. The funding
    # rate returned by the API is per-payment; the APR is
    # ``rate * 3 * 365``.
    _FUNDING_INTERVAL_HOURS: float = 8.0
    # Cap on in-flight ``_fetch_funding_rate`` calls within a
    # single batch. The ``/api/v1/contract/funding_rate``
    # endpoint is rate-limited at 20 requests/second per IP;
    # we empirically hit 429 storms at concurrency=8 when
    # fetching ~2000 bases back-to-back, so 4-wide per batch
    # is the safe default.
    _FUNDING_CONCURRENCY: int = 4
    # Pause between consecutive batches. With 4-wide batches
    # of ~1s each, 0.5s of breathing room keeps us at ~2.7
    # requests/second sustained, well under MEXC's 20 req/s
    # ceiling. (``_fetch_funding_rates`` processes
    # ``bases`` in batches of ``_FUNDING_CONCURRENCY``
    # followed by this delay.)
    _FUNDING_BATCH_DELAY: float = 0.5

    # MEXC contract-API error code returned when a perpetual
    # contract doesn't exist for the given ``base``/``quote``.
    # Treated as a silent "no contract" -- ``funding_rate=None``.
    _MEXC_NO_CONTRACT_CODE = 1001

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
    ):
        # MEXC spot v3 endpoints we use are public; credentials
        # are accepted and stored for forward-compatibility -- if
        # MEXC ever deprecates public access to ``/exchangeInfo``
        # or ``/klines``, or if a future margin / borrow rate
        # endpoint goes behind auth, the adapter can sign requests
        # via :meth:`_sign` below. As of writing, neither endpoint
        # requires auth, so passing ``None`` for both keys is
        # equivalent to passing the user's API keys.
        self._api_key = api_key
        self._api_secret = api_secret

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def _sign(self, params: dict) -> dict:
        """Add ``timestamp`` and ``signature`` to a query-string param
        dict. MEXC uses HMAC-SHA256 over the canonical query string
        (``key=value&key=value...``), the same convention Binance uses.

        Currently unused: all endpoints we hit are public. Kept as a
        forward-compatible helper in case MEXC tightens access to
        ``/exchangeInfo`` / ``/klines`` or a future margin / borrow
        rate endpoint goes behind auth.
        """
        import hashlib
        import hmac
        import time as _time
        from urllib.parse import urlencode

        params = dict(params)
        params["timestamp"] = int(_time.time() * 1000)
        params["recvWindow"] = 5000
        query = urlencode(params)
        sig = hmac.new(
            (self._api_secret or "").encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = sig
        return params

    async def _spot_get(
        self, client: AsyncClient, path: str, params: dict | None = None
    ) -> object:
        """Make a public ``GET`` against MEXC's spot v3 API.

        The response body is JSON; the spot v3 endpoints return
        either a bare value (klines: ``list``) or an object with
        a ``symbols`` / ``serverTime`` field. Errors are
        signalled via 4xx/5xx and a JSON body shaped like
        ``{"code": <int>, "msg": <str>}`` -- ``resp.raise_for_status``
        surfaces those via ``httpx.HTTPStatusError``.
        """
        url = f"https://{self.HOST}{path}"
        try:
            resp = await client.get(url, params=params or {}, timeout=30.0)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise
        except Exception as e:
            raise RuntimeError(
                f"MEXC {path} request failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    # pairs
    # ------------------------------------------------------------------
    async def pairs(
        self,
        client: AsyncClient,
        quote_assets: set[str],
        limit: int | None = None,
    ) -> pl.DataFrame:
        """Returns active spot pairs with funding rate APR.

        Columns: ts, symbol, exchange, base, quote, cross_rate,
        isolated_rate, funding_rate

        ``cross_rate`` and ``isolated_rate`` are ``None`` for every
        pair -- **MEXC does not support spot-margin (borrowing)
        trading.** Every active pair on the exchange has
        ``isMarginTradingAllowed == false`` in
        ``/api/v3/exchangeInfo`` (verified: 0 of 2378 active pairs
        on 2026-06-17), and MEXC's only "leverage" product is the
        USDⓈ-M linear-swap perpetual market, whose cost of carry
        is the perpetual funding rate -- not a borrow rate. We
        populate the latter as ``funding_rate`` below.

        ``funding_rate`` is the most recent MEXC USDⓈ-M perpetual
        funding payment, annualised as ``rate * 3 * 365`` (MEXC
        pays funding every 8h). It's ``None`` for bases that have
        no USDT/USDC perp contract on MEXC.

        ``limit`` is an optional cap on the number of pairs
        returned: when ``None`` (the default), the entire
        ``/exchangeInfo`` universe is enumerated and funding rates
        are probed for every base. When set to a positive integer,
        only the first ``limit`` pairs are returned (in the order
        MEXC's ``exchangeInfo`` returns them -- typically
        alphabetical by base) and only those bases' funding rates
        are probed. The CLI never passes ``limit``; it always
        fetches the full universe.
        """
        now = dt.datetime.now(dt.timezone.utc)

        payload = await self._spot_get(client, "/api/v3/exchangeInfo")
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"MEXC exchangeInfo: unexpected payload type "
                f"{type(payload).__name__}"
            )

        quote_assets_upper = {q.upper() for q in quote_assets}

        rows: list[dict] = []
        bases: set[str] = set()
        for s in payload.get("symbols", []):
            if s.get("status") != "1":
                # Status 1 = online; 2 = paused, 3 = offline. Skip
                # paused / offline pairs so downstream consumers
                # don't try to trade against stale pairs.
                continue
            base = s.get("baseAsset", "").lower()
            quote = s.get("quoteAsset", "").lower()
            if not base or not quote:
                continue
            if quote.upper() not in quote_assets_upper:
                continue
            rows.append(
                {
                    "ts": now,
                    "symbol": s["symbol"],
                    "exchange": self.NAME,
                    "base": base,
                    "quote": quote,
                }
            )
            bases.add(base)
            # ``limit`` bounds the result: stop collecting once
            # we've reached the cap. Note we still iterate the
            # full ``exchangeInfo`` response (it's a single
            # in-memory call) -- only the funding-rate fetch and
            # the returned rows are bounded.
            if limit is not None and len(rows) >= limit:
                break

        if not rows:
            return self.empty_pairs_df()

        # Borrow rates are unavailable on the public spot v3 API;
        # leave them null for every pair.
        for r in rows:
            r["cross_rate"] = None
            r["isolated_rate"] = None

        # Funding rates: probe each base for USDT- and USDC-
        # settled perpetual contracts concurrently.
        funding_rates = await self._fetch_funding_rates(client, bases)
        for r in rows:
            r["funding_rate"] = funding_rates.get(r["base"])

        df = pl.DataFrame(rows)
        df = df.select(
            pl.col("ts").cast(pl.Datetime("us", time_zone="UTC")),
            "symbol",
            "exchange",
            "base",
            "quote",
            pl.col("cross_rate").cast(pl.Float64),
            pl.col("isolated_rate").cast(pl.Float64),
            pl.col("funding_rate").cast(pl.Float64),
        )
        self.validate_pairs_df(df)
        return df

    # ------------------------------------------------------------------
    # funding rate fetcher
    # ------------------------------------------------------------------
    async def _funding_request(
        self, client: AsyncClient, contract: str
    ) -> dict | None:
        """Make a single MEXC funding-rate HTTP request with
        per-call retry on transient errors (429, 5xx, transport).

        Returns the parsed JSON payload on success, ``None`` on a
        permanent failure. The funding-rate endpoint returns
        ``{"success": true, "code": 0, "data": {...}}`` on
        success and ``{"success": false, "code": <int>, "message": ...}``
        on a missing contract -- both are 200 OK, so we always
        return the parsed dict and let the caller decide.

        ``Retry-After`` headers from 429 responses are honoured
        so a single contract that hit the rate limit doesn't get
        discarded as a permanent failure on the first attempt.
        """
        url = (
            f"https://{self.HOST}/api/v1/contract/funding_rate/{contract}"
        )
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = await client.get(url, timeout=30.0)
                if resp.status_code == 429:
                    # ``Retry-After`` is in seconds (per RFC 7231).
                    # Default to 1s if MEXC omits it.
                    retry_after = float(resp.headers.get("Retry-After", "1"))
                    # Cap to avoid a hostile / missing header
                    # stalling the whole adapter.
                    retry_after = min(max(retry_after, 0.0), 10.0)
                    delay = retry_after * (0.75 + __import__("random").random() * 0.5)
                    l.warning(
                        "MEXC funding_rate(%s) 429 rate-limited "
                        "(attempt %d); honouring Retry-After=%.1fs",
                        contract,
                        attempt,
                        retry_after,
                    )
                    await asyncio.sleep(delay)
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    delay = min(
                        self.RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                        self.RETRY_MAX_DELAY,
                    )
                    delay = delay * (0.75 + __import__("random").random() * 0.5)
                    l.warning(
                        "MEXC funding_rate(%s) transient error "
                        "(attempt %d): %s; retrying in %.1fs "
                        "(infinite retries)",
                        contract,
                        attempt,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                l.warning(f"MEXC funding_rate({contract}) HTTP error: {e}")
                return None
            except Exception as e:
                if not self.is_transient_error(e):
                    raise RuntimeError(
                        f"MEXC funding_rate request failed: {e}"
                    ) from e
                delay = min(
                    self.RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                    self.RETRY_MAX_DELAY,
                )
                delay = delay * (0.75 + __import__("random").random() * 0.5)
                l.warning(
                    "MEXC funding_rate(%s) transient error "
                    "(attempt %d): %s; retrying in %.1fs "
                    "(infinite retries)",
                    contract,
                    attempt,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)

    async def _fetch_funding_rate(
        self, client: AsyncClient, base: str, quote: str
    ) -> float | None:
        """Return the most recent funding rate (annualised APR) for the
        MEXC USDⓈ-M perpetual on ``base``/``quote``, or ``None`` if
        no contract exists or the request fails for a non-transient
        reason.

        Endpoint:
        ``GET /api/v1/contract/funding_rate/{contract}``
        where ``contract`` is ``{BASE}_{QUOTE}`` (with an underscore,
        not a dash). Response shape::

            {
              "success": true,
              "code": 0,
              "data": {
                "symbol": "BTC_USDT",
                "fundingRate": -0.000066,
                "collectCycle": 8,   // hours between payments
                "nextSettleTime": ...,
                "timestamp": ...
              }
            }

        On a missing contract the response is::

            {
              "success": false,
              "code": 1001,
              "message": "Contract does not exist"
            }

        We probe USDT first (most common), then USDC. If both
        settle currencies return ``code == 1001`` we return
        ``None`` (no contract for this base).
        """
        for settle in (quote.upper(), "USDC" if quote.upper() == "USDT" else "USDT"):
            contract = f"{base.upper()}_{settle}"
            payload = await self._funding_request(client, contract)
            if payload is None:
                # Network / permanent failure: give up on this
                # base rather than waste time on the fallback.
                return None

            if not isinstance(payload, dict):
                continue
            # MEXC's contract-API envelopes look like
            # ``{"success": <bool>, "code": <int>, "data"|"message": ...}``.
            code = payload.get("code")
            success = payload.get("success")
            if success is False or code == self._MEXC_NO_CONTRACT_CODE:
                # No contract for this settle currency -- try the
                # next one.
                continue
            if isinstance(code, int) and code != 0:
                # Real error: log once, give up on this base.
                l.warning(f"MEXC funding_rate({contract}) error: {payload}")
                return None

            data = payload.get("data")
            if not isinstance(data, dict):
                return None
            try:
                rate = float(data["fundingRate"])
            except (KeyError, TypeError, ValueError):
                return None
            return self.annualize_funding_rate(
                rate, self._FUNDING_INTERVAL_HOURS
            )

        # Both settle currencies returned "no contract".
        return None

    async def _fetch_funding_rates(
        self, client: AsyncClient, bases: Iterable[str]
    ) -> dict[str, float]:
        """``{base_lower: annual_funding_rate}`` for each base asset
        that has a MEXC USDⓈ-M perpetual.

        Concurrency strategy: fire ``_FUNDING_CONCURRENCY``
        requests at a time, then sleep ``_FUNDING_BATCH_DELAY``
        seconds before the next batch. This shaped-burst pattern
        keeps us under MEXC's 20-req/s per-IP limit even at peak,
        because each burst is followed by a quiet period that lets
        their rate limiter recover. A pure semaphore-based approach
        produces continuous bursts that MEXC's WAF throttles.
        """
        bases = sorted({b.lower() for b in bases})
        if not bases:
            return {}

        semaphore = asyncio.Semaphore(self._FUNDING_CONCURRENCY)

        async def _one(b: str) -> tuple[str, float | None]:
            async with semaphore:
                try:
                    rate = await self._fetch_funding_rate(
                        client, base=b, quote="usdt"
                    )
                except Exception as e:
                    l.warning(f"MEXC funding_rate({b}) failed: {e}")
                    return b, None
            return b, rate

        results: list[tuple[str, float | None]] = []
        for i in range(0, len(bases), self._FUNDING_CONCURRENCY):
            chunk = bases[i:i + self._FUNDING_CONCURRENCY]
            chunk_results = await asyncio.gather(*(_one(b) for b in chunk))
            results.extend(chunk_results)
            # Sleep between batches so MEXC's IP-level rate
            # limiter can recover. We use 0.5s because each
            # batch of ~4 requests takes ~1s to complete.
            if i + self._FUNDING_CONCURRENCY < len(bases):
                await asyncio.sleep(self._FUNDING_BATCH_DELAY)

        return {b: r for b, r in results if r is not None}

    # ------------------------------------------------------------------
    # klines
    # ------------------------------------------------------------------
    async def _fetch_kline_window(
        self, client: AsyncClient, symbol: str, limit: int
    ) -> pl.DataFrame:
        """Fetch a single batch of up to ``limit`` daily candles for
        ``symbol`` from MEXC's spot v3 kline endpoint.

        Returns a typed polars frame with columns:
        ``open_ts, open, high, low, close, base_volume, quote_volume``
        (all timestamps ``Datetime(time_zone='UTC')``).

        MEXC's kline endpoint ignores ``startTime`` / ``endTime``
        when ``limit`` is set: it always returns the most recent
        ``limit`` candles, regardless of what you ask for. This
        method just makes a single request; :meth:`klines` is
        responsible for walking back through history by issuing
        successive calls and trimming the overlap.

        The response is JSON: ``[open_ts, open, high, low, close,
        base_volume, close_ts, quote_volume]`` where ``open_ts`` /
        ``close_ts`` are millisecond-resolution epoch and the OHLC
        fields are strings (MEXC's convention).
        """
        params = {"symbol": symbol, "interval": "1d", "limit": limit}
        payload = await self._spot_get(
            client, "/api/v3/klines", params=params
        )
        if not isinstance(payload, list):
            raise RuntimeError(
                f"MEXC klines: expected list payload, got {type(payload).__name__}"
            )

        if not payload:
            return pl.DataFrame(
                schema={
                    "open_ts": pl.Datetime("us", time_zone="UTC"),
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "base_volume": pl.Float64,
                    "quote_volume": pl.Float64,
                }
            )

        rows: list[dict] = []
        for k in payload:
            # The shape is fixed at 8 fields. Index-by-position
            # is robust against MEXC ever switching to keyed
            # objects, since they use arrays.
            try:
                open_ts_ms = int(k[0])
                open_v = float(k[1])
                high = float(k[2])
                low = float(k[3])
                close_v = float(k[4])
                base_volume = float(k[5])
                quote_volume = float(k[7])
            except (KeyError, IndexError, TypeError, ValueError) as e:
                raise RuntimeError(
                    f"MEXC klines: malformed row {k!r}: {e}"
                ) from e
            rows.append(
                {
                    "open_ts": dt.datetime.fromtimestamp(
                        open_ts_ms / 1000.0, tz=dt.timezone.utc
                    ),
                    "open": open_v,
                    "high": high,
                    "low": low,
                    "close": close_v,
                    "base_volume": base_volume,
                    "quote_volume": quote_volume,
                }
            )
        return (
            pl.DataFrame(rows)
            .with_columns(pl.col("open_ts").cast(pl.Datetime("us", time_zone="UTC")))
        )

    async def klines(
        self,
        client: AsyncClient,
        symbol: str,
        start_time: dt.datetime,
        end_time: dt.datetime,
    ) -> pl.DataFrame:
        """Fetch daily klines for ``symbol`` in the half-open range
        ``[start_time, end_time)``.

        MEXC's kline endpoint ignores ``startTime`` / ``endTime``
        entirely -- it always returns the most recent ``limit``
        candles. To get candles further back than ``MAX_KLINES``
        rows, we issue successive calls: each call returns
        ``MAX_KLINES`` candles; we drop the ones we've already
        seen (``open_ts >= last_seen_open_ts``) and keep fetching
        until the oldest returned ``open_ts`` falls below
        ``start_time``.

        Columns: open_ts, close_ts, symbol, exchange, base, quote,
        open, high, low, close, base_volume, quote_volume

        ``start_time`` is inclusive, ``end_time`` is exclusive:
        only candles whose open time satisfies
        ``start_time <= open_ts < end_time`` are returned.
        ``close_ts`` is the inclusive last instant of the daily
        candle (``open_ts + 24h - 1us``).
        """
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=dt.timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=dt.timezone.utc)

        base, quote = _split_symbol(symbol)

        # Walk back through history in MAX_KLINES-sized windows.
        # Each call returns the most recent MAX_KLINES candles;
        # we keep only the ones we haven't seen yet, and stop
        # when the oldest returned candle is at-or-before
        # ``start_time``.
        collected: list[pl.DataFrame] = []
        last_seen_open_ts: dt.datetime | None = None
        while True:
            window = await self._fetch_kline_window(
                client, symbol, limit=self.MAX_KLINES
            )
            if window.height == 0:
                break

            if last_seen_open_ts is not None:
                # Drop anything at-or-after the most recent
                # ``open_ts`` we've already collected -- those
                # are the duplicates from the next page of the
                # API's most-recent-first stream.
                window = window.filter(pl.col("open_ts") < last_seen_open_ts)

            if window.height == 0:
                break

            collected.append(window)
            oldest = window["open_ts"].min()
            # If the oldest candle in this batch is already at
            # or before ``start_time``, we've reached the
            # beginning of the requested range.
            if oldest <= start_time:
                break
            # If we got fewer than MAX_KLINES rows back, the
            # API has no more history to give us.
            if window.height < self.MAX_KLINES:
                break

            last_seen_open_ts = oldest

        if not collected:
            return self.empty_klines_df()

        df = pl.concat(collected).sort("open_ts")

        # Half-open filter ``[start_time, end_time)`` on the
        # ``open_ts``. This excludes the in-progress today
        # candle when ``end_time`` is today's 00:00 UTC.
        df = df.filter(
            (pl.col("open_ts") >= start_time) & (pl.col("open_ts") < end_time)
        )
        if df.height == 0:
            return self.empty_klines_df()

        # Project onto the canonical schema. ``close_ts`` is the
        # inclusive last instant of the daily candle:
        # ``open_ts + 24h - 1us``.
        df = df.with_columns(
            pl.col("open_ts")
            .dt.offset_by(f"{self.DAILY_SECONDS * 1_000_000 - 1}us")
            .alias("close_ts"),
        ).select(
            pl.col("open_ts").cast(pl.Datetime("us", time_zone="UTC")),
            pl.col("close_ts").cast(pl.Datetime("us", time_zone="UTC")),
            pl.lit(symbol).alias("symbol"),
            pl.lit(self.NAME).alias("exchange"),
            pl.lit(base).alias("base"),
            pl.lit(quote).alias("quote"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("base_volume").cast(pl.Float64),
            pl.col("quote_volume").cast(pl.Float64),
        )
        self.validate_klines_df(df)
        return df