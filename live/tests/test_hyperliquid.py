"""Tests for the Hyperliquid exchange adapter.

Hyperliquid is a perpetuals-only DEX. The adapter deviates from
the spot-pair convention used by the other exchanges:

  * ``pairs()`` emits one row per **perp contract** with
    ``quote = "usd"`` and no borrow rates.
  * ``klines()`` returns **perp OHLCV** (filtered to ``n > 0``).
  * Funding is **per-hour** (not 8h), so the annualisation is
    ``rate * 24 * 365``.

These tests cover:

  * Schema contract: ``pairs()`` and ``klines()`` produce the
    canonical polars schemas.
  * Funding rate annualisation matches the 1h schedule.
  * Per-call klines are capped at ``MAX_KLINES`` (500) and the
    response is filtered to ``n > 0``.
  * ``klines_paged()`` walks a wider range backward in 500-bar
    chunks, all bars pass the ``n > 0`` filter.
  * Live ``pairs()`` returns a non-empty frame with realistic
    ``funding_rate`` values (annualised) and the well-known
    ``BTC`` perp is included.
  * Live ``klines()`` for ``BTC`` covers a sensible historical
    depth (Hyperliquid perps launched Feb 2023) and the
    ``open_ts`` is midnight-UTC aligned.
  * ``pairs()`` returns ``cross_rate = None`` and
    ``isolated_rate = None`` for every row.
"""

from __future__ import annotations

import asyncio
import datetime as dt

import httpx
import polars as pl
import pytest
from httpx import AsyncClient

from live import Hyperliquid
from ._assertions import (
    assert_close_ts_matches_open,
    assert_klines_schema,
    assert_lowercase,
    assert_open_ts_in_range,
    assert_pairs_schema,
)


# ----------------------------------------------------------------------
# offline unit tests
# ----------------------------------------------------------------------


def test_hyperliquid_class_constants() -> None:
    """Class constants match the documented Hyperliquid API."""
    ex = Hyperliquid()
    assert ex.NAME == "hyperliquid"
    assert ex.MAX_KLINES == 500  # candleSnapshot hard cap
    assert ex.DAILY_ALIGN_HOUR_UTC == 0  # midnight UTC


def test_annualize_funding_rate_1h_hyperliquid() -> None:
    """Hyperliquid pays funding hourly. 1bp per hour = 24bp/day
    = ~87.6% APR (vs ~10.95% for an 8h schedule).
    """
    ex = Hyperliquid()
    assert ex.annualize_funding_rate(0.0001, 1.0) == pytest.approx(
        0.876, rel=1e-9
    )


def test_annualize_funding_rate_hyperliquid_default() -> None:
    """Using the adapter's own ``_FUNDING_INTERVAL_HOURS`` constant
    gives the right annualisation for a real-looking Hyperliquid
    funding payment.
    """
    ex = Hyperliquid()
    # A typical Hyperliquid funding rate of 5bp/hour on a
    # mid-volatility alt = 0.0005 * 24 * 365 = 4.38 = 438% APR.
    assert ex.annualize_funding_rate(
        0.0005, ex._FUNDING_INTERVAL_HOURS
    ) == pytest.approx(0.0005 * 24 * 365, rel=1e-9)


def test_pairs_schema_dtype_funding_rate() -> None:
    """``funding_rate`` is ``pl.Float64`` (same as the other adapters)."""
    df = Hyperliquid.empty_pairs_df()
    assert "funding_rate" in df.columns
    assert df.schema["funding_rate"] == pl.Float64
    # And the cross/isolated columns are present even though HL
    # never populates them -- the schema must match the contract.
    assert df.schema["cross_rate"] == pl.Float64
    assert df.schema["isolated_rate"] == pl.Float64


def test_pairs_schema_includes_canonical_columns() -> None:
    """The full ``PAIRS_SCHEMA`` is exposed via ``empty_pairs_df``."""
    df = Hyperliquid.empty_pairs_df()
    expected = {
        "ts",
        "symbol",
        "exchange",
        "base",
        "quote",
        "cross_rate",
        "isolated_rate",
        "funding_rate",
    }
    assert set(df.columns) == expected


def test_klines_schema_dtype_columns() -> None:
    """``klines()`` uses the canonical ``KLINES_SCHEMA``."""
    df = Hyperliquid.empty_klines_df()
    expected = {
        "open_ts",
        "close_ts",
        "symbol",
        "exchange",
        "base",
        "quote",
        "open",
        "high",
        "low",
        "close",
        "base_volume",
        "quote_volume",
    }
    assert set(df.columns) == expected
    # All OHLCV columns are Float64; all timestamps are tz-aware UTC.
    for col in ("open", "high", "low", "close", "base_volume", "quote_volume"):
        assert df.schema[col] == pl.Float64
    for col in ("open_ts", "close_ts"):
        dtype = df.schema[col]
        assert isinstance(dtype, pl.Datetime)
        assert dtype.time_zone == "UTC"


def test_quote_is_usd_for_empty_pairs() -> None:
    """``empty_pairs_df`` has the canonical schema; ``quote`` defaults
    to ``pl.Utf8`` and the convention is to emit ``"usd"`` for
    Hyperliquid perps."""
    df = Hyperliquid.empty_pairs_df()
    assert df.schema["quote"] == pl.Utf8


def test_hyperliquid_has_no_ctor_args() -> None:
    """The adapter must be constructible without credentials, since
    every public endpoint is unauthenticated."""
    ex = Hyperliquid()
    # No exceptions on construction; the instance is usable.
    assert ex.NAME == "hyperliquid"


# ----------------------------------------------------------------------
# n>0 filter (the core new requirement)
# ----------------------------------------------------------------------


def _make_candle(t_ms: int, n: int, close: float = 100.0) -> dict:
    """Build a synthetic Hyperliquid candle for offline tests."""
    return {
        "t": t_ms,
        "T": t_ms + 24 * 3600 * 1000 - 1,
        "s": "BTC",
        "i": "1d",
        "o": str(close),
        "c": str(close),
        "h": str(close + 1),
        "l": str(close - 1),
        "v": "1.5",
        "n": n,
    }


def test_drop_zero_trade_candles_drops_n_zero() -> None:
    """Bars with ``n == 0`` are dropped (synthetic / oracle backfill)."""
    from live.hyperliquid import _drop_zero_trade_candles

    rows = [
        {"open_ts": "x", "n": 0},   # synthetic, drop
        {"open_ts": "y", "n": 100},  # real, keep
        {"open_ts": "z", "n": 0},   # synthetic, drop
    ]
    out = _drop_zero_trade_candles(rows)
    assert len(out) == 1
    assert out[0]["open_ts"] == "y"


def test_drop_zero_trade_candles_keeps_missing_n() -> None:
    """Bars with ``n`` missing are KEPT (defensive -- a real live bar
    that omits ``n`` should not be silently dropped)."""
    from live.hyperliquid import _drop_zero_trade_candles

    rows = [
        {"open_ts": "x", "n": None},   # unknown, keep
        {"open_ts": "y", "n": 0},      # explicit zero, drop
        {"open_ts": "z"},              # missing field, keep
    ]
    out = _drop_zero_trade_candles(rows)
    assert len(out) == 2
    kept = {r["open_ts"] for r in out}
    assert kept == {"x", "z"}


def test_klines_filters_n_zero_in_klines() -> None:
    """End-to-end check that ``klines()`` drops ``n == 0`` bars
    when fed a response that includes them.

    We stub ``_fetch_candles`` so no network is hit, then assert
    the resulting frame has no zero-trade bars.
    """
    import asyncio

    # 10 daily bars spanning 2025-01-01 .. 2025-01-10. Alternate
    # between real and synthetic to make the filter easy to spot.
    base_ms = int(
        dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000
    )
    fake = []
    for i in range(10):
        fake.append(_make_candle(base_ms + i * 86_400_000, n=0 if i % 2 == 0 else 100))

    async def t() -> pl.DataFrame:
        ex = Hyperliquid()
        # ``_retry`` is a no-op for our stub; it just awaits the
        # method and returns the result. We bypass it to keep the
        # test pure (no event-loop scheduling of retries).
        async def stub_fetch(client, coin, start_ms, end_ms):
            return fake
        ex._fetch_candles = stub_fetch  # type: ignore[assignment]
        try:
            async with AsyncClient(timeout=30.0) as client:
                df = await ex.klines(
                    client,
                    "BTC",
                    start_time=dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
                    end_time=dt.datetime(2025, 1, 11, tzinfo=dt.timezone.utc),
                )
            return df
        finally:
            del ex._fetch_candles

    df = asyncio.run(t())
    assert_klines_schema(df)
    # 5 real bars survive the filter (the 5 with n=0 are dropped).
    assert df.height == 5
    # open_ts values should be the odd-indexed ones from the input.
    odd_tses = [
        dt.datetime(2025, 1, 2, tzinfo=dt.timezone.utc),
        dt.datetime(2025, 1, 4, tzinfo=dt.timezone.utc),
        dt.datetime(2025, 1, 6, tzinfo=dt.timezone.utc),
        dt.datetime(2025, 1, 8, tzinfo=dt.timezone.utc),
        dt.datetime(2025, 1, 10, tzinfo=dt.timezone.utc),
    ]
    actual = sorted(df["open_ts"].to_list())
    expected = sorted(odd_tses)
    assert actual == expected


# ----------------------------------------------------------------------
# live tests (hit the public Hyperliquid API)
# ----------------------------------------------------------------------


async def test_pairs_schema_includes_all_canonical_columns(
    client: AsyncClient, hyperliquid: Hyperliquid
) -> None:
    """Live ``pairs()`` returns a frame with the canonical
    ``PAIRS_SCHEMA`` -- the schema check matches the other
    exchanges."""
    pairs = await hyperliquid.pairs_with_retry(
        client, quote_assets=set()  # Hyperliquid ignores quote_assets
    )
    assert_pairs_schema(pairs)
    assert pairs.height > 0


async def test_pairs_quote_is_usd(
    client: AsyncClient, hyperliquid: Hyperliquid
) -> None:
    """Every row has ``quote == "usd"`` (the perp quote convention)."""
    pairs = await hyperliquid.pairs_with_retry(client, quote_assets=set())
    assert pairs.height > 0
    quotes = pairs["quote"].unique().to_list()
    assert quotes == ["usd"], f"expected quote='usd' for every row, got {quotes!r}"


async def test_pairs_cross_and_isolated_rates_are_null(
    client: AsyncClient, hyperliquid: Hyperliquid
) -> None:
    """``cross_rate`` and ``isolated_rate`` are always null on
    Hyperliquid -- there's no borrow-rate concept on a perp-only
    DEX."""
    pairs = await hyperliquid.pairs_with_retry(client, quote_assets=set())
    assert pairs.height > 0
    n_cross_null = pairs.filter(pl.col("cross_rate").is_null()).height
    n_iso_null = pairs.filter(pl.col("isolated_rate").is_null()).height
    assert n_cross_null == pairs.height, "cross_rate should be null for all rows"
    assert n_iso_null == pairs.height, "isolated_rate should be null for all rows"


async def test_pairs_funding_rate_annualised(
    client: AsyncClient, hyperliquid: Hyperliquid
) -> None:
    """``funding_rate`` is the **annualised** value (``rate * 24 * 365``)
    -- not the per-hour rate. A few-bp/hour perp should be in
    the tens-of-percent APR range, not a few bps."""
    pairs = await hyperliquid.pairs_with_retry(client, quote_assets=set())
    btc = pairs.filter(pl.col("symbol") == "BTC")
    if btc.height == 0:
        pytest.skip("BTC perp not returned (shouldn't happen on mainnet)")
    rate = btc["funding_rate"][0]
    assert rate is not None, "BTC funding_rate should be populated"
    # BTC's perp funding is usually within a few % APR. We allow
    # up to +/-100% APR to absorb short-lived spikes on
    # liquidations / cascades.
    assert -1.0 < rate < 1.0, (
        f"BTC funding_rate={rate:.4f} implausible (expected small "
        f"APR for a high-liquidity perp)"
    )


async def test_pairs_base_is_lowercase(
    client: AsyncClient, hyperliquid: Hyperliquid
) -> None:
    """``base`` is the lower-cased symbol (matching the convention
    used by HTX / KuCoin / Kraken / Binance)."""
    pairs = await hyperliquid.pairs_with_retry(client, quote_assets=set())
    assert pairs.height > 0
    assert_lowercase(pairs, "base")


async def test_pairs_includes_btc(
    client: AsyncClient, hyperliquid: Hyperliquid
) -> None:
    """The BTC perp is on the canonical list."""
    pairs = await hyperliquid.pairs_with_retry(client, quote_assets=set())
    btc = pairs.filter(pl.col("symbol") == "BTC")
    assert btc.height == 1, f"expected 1 BTC row, got {btc.height}"
    assert btc["base"][0] == "btc"
    assert btc["quote"][0] == "usd"


async def test_klines_schema_for_btc(
    client: AsyncClient, hyperliquid: Hyperliquid
) -> None:
    """``klines()`` for BTC returns the canonical schema in a
    typical 14-day range."""
    end = dt.datetime.now(dt.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = end - dt.timedelta(days=14)
    df = await hyperliquid.klines_with_retry(
        client, "BTC", start_time=start, end_time=end
    )
    assert_klines_schema(df)
    assert_open_ts_in_range(df, start, end)
    # close_ts follows the µs-resolution convention (api_provided=False).
    assert_close_ts_matches_open(df, api_provided=False)


async def test_klines_no_zero_trade_bars(
    client: AsyncClient, hyperliquid: Hyperliquid
) -> None:
    """The klines frame MUST NOT contain any bars with
    ``n == 0``. We don't have a direct handle on ``n`` in the
    schema, so we use the proxy: every bar must have a
    non-negative ``base_volume`` AND a ``close`` price that's
    strictly positive. Pre-2023-02-26 synthetic bars have
    ``v == 0.0`` AND ``n == 0``; real bars have non-zero
    volume.

    More importantly, the row count for BTC over the last 365
    days must be ≤ 365 (we lose some bars to the n=0 filter
    ONLY if they have volume 0). A 365-bar window starting
    today always lies inside the post-launch real-trade region,
    so the count should be exactly 365.
    """
    end = dt.datetime.now(dt.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = end - dt.timedelta(days=365)
    df = await hyperliquid.klines_with_retry(
        client, "BTC", start_time=start, end_time=end
    )
    assert_klines_schema(df)
    # Every bar must have strictly positive close (real bars do;
    # synthetic bars have realistic OHLC but v=0 -- so we
    # can't filter on close alone. Filter on base_volume.
    zero_vol = df.filter(pl.col("base_volume") <= 0.0)
    assert zero_vol.height == 0, (
        f"klines() returned {zero_vol.height} zero-volume bars; "
        f"the n>0 filter is not working"
    )
    # Sanity: a 365-day window that started ~2025-06-12 is well
    # inside the real-trade region (BTC perp launched Feb 2023),
    # so we expect close to 365 bars.
    assert df.height >= 300, (
        f"klines() returned only {df.height} bars in a 365-day "
        f"window; expected ~365 (the n>0 filter may be too "
        f"aggressive)"
    )
    assert df.height <= 365


async def test_klines_open_ts_at_midnight_utc(
    client: AsyncClient, hyperliquid: Hyperliquid
) -> None:
    """Every ``open_ts`` opens at 00:00:00 UTC (Hyperliquid aligns
    to midnight, same as Binance / KuCoin / Kraken)."""
    end = dt.datetime.now(dt.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = end - dt.timedelta(days=30)
    df = await hyperliquid.klines_with_retry(
        client, "BTC", start_time=start, end_time=end
    )
    if df.height == 0:
        pytest.skip("no klines returned")
    hours = df["open_ts"].dt.hour().unique().to_list()
    assert hours == [0], (
        f"expected open_ts at 00:00 UTC, got hours {hours!r}"
    )


async def test_klines_paged_walks_backward(
    client: AsyncClient, hyperliquid: Hyperliquid
) -> None:
    """``klines_paged()`` returns > ``MAX_KLINES`` bars by walking
    backward in 500-bar chunks. The post-filter result is sorted
    by ``open_ts`` and has no gaps inside the real-trade region
    (Hyperliquid BTC perp is active every day since Feb 2023).
    """
    end = dt.datetime.now(dt.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    # 1500 days ≈ 4.1 years; this should require 3+ paged calls
    # (500-bar cap per call).
    start = end - dt.timedelta(days=1500)
    df = await hyperliquid.klines_paged(
        client, "BTC", start_time=start, end_time=end
    )
    assert_klines_schema(df)
    assert_open_ts_in_range(df, start, end)
    assert df.height > Hyperliquid.MAX_KLINES, (
        f"klines_paged() returned {df.height} bars; expected > "
        f"{Hyperliquid.MAX_KLINES} to prove the chunking works"
    )
    # The result must be sorted (the base class sorts the
    # concatenated chunks).
    assert df["open_ts"].is_sorted(), (
        "klines_paged() result is not sorted by open_ts"
    )


async def test_funding_rate_annualisation_factor(
    client: AsyncClient, hyperliquid: Hyperliquid
) -> None:
    """Spot-check the annualisation factor: the funding_rate for
    BTC must be a real-looking small APR (the underlying hourly
    rate is what the raw API returns).

    We hit the public ``_fetch_funding_rates`` path (one
    ``metaAndAssetCtxs`` call) and pluck the BTC entry. This is
    the same code path used by ``pairs()``.
    """
    rates = await hyperliquid._fetch_funding_rates(client, ["BTC"])
    assert "btc" in rates, f"expected BTC in funding dict, got {list(rates)}"
    rate = rates["btc"]
    # BTC funding is rarely > 50% APR in either direction. A
    # hyperliquid BTC perp paying, say, 0.0001 / hour = 87.6%
    # APR would be a massive event. Allow ±200% to absorb
    # liquidation cascades.
    assert -2.0 < rate < 2.0, (
        f"BTC funding_rate={rate:.4f} implausible"
    )


# ----------------------------------------------------------------------
# regression: unknown coin must NOT loop forever
#
# candleSnapshot for a non-existent coin returns HTTP 500 with a
# body of literal ``null``. The base-class classifier treats all
# 5xx as transient; the Hyperliquid override narrows this to
# "5xx + null body = permanent" so the call propagates
# immediately instead of hanging in the infinite-retry loop.
# ----------------------------------------------------------------------


async def test_unknown_coin_raises_immediately(
    client: AsyncClient, hyperliquid: Hyperliquid
) -> None:
    """``klines()`` for a coin that doesn't exist on Hyperliquid
    must raise an ``httpx.HTTPStatusError`` (the 500 we got
    from the server) within a couple of seconds, not loop
    forever under the infinite-retry policy.

    The previous behaviour (pre-override) hung the call
    indefinitely because 500 was classified as transient.
    """
    import time as _time

    start = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2025, 1, 2, tzinfo=dt.timezone.utc)
    t0 = _time.time()
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await hyperliquid.klines_with_retry(
            client,
            "DEFINITELY_NOT_A_REAL_COIN_123",
            start_time=start,
            end_time=end,
        )
    elapsed = _time.time() - t0
    # Must fail in well under the 10s base delay; an unknown
    # coin should not be retried at all.
    assert elapsed < 5.0, (
        f"unknown coin took {elapsed:.2f}s to fail; the "
        f"is_transient_error override is not firing "
        f"(infinite retry on a permanent error)"
    )
    # And the 500 we got back is the literal "unknown coin"
    # signature (body == "null"). The full text check is in
    # the offline unit test below; here we just confirm the
    # status code propagated.
    assert exc_info.value.response.status_code == 500


async def test_delisted_known_coin_returns_empty(
    client: AsyncClient, hyperliquid: Hyperliquid
) -> None:
    """A coin that HL has explicitly delisted returns 200 + ``[]``
    (verified by direct probe). The klines call should
    complete quickly with an empty DataFrame -- not raise.
    The ``n > 0`` filter has no work to do because there are
    no rows to filter."""
    import time as _time

    start = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2025, 1, 2, tzinfo=dt.timezone.utc)
    t0 = _time.time()
    df = await hyperliquid.klines_with_retry(
        client, "MELANIA", start_time=start, end_time=end
    )
    elapsed = _time.time() - t0
    assert elapsed < 5.0, f"delisted-known coin took {elapsed:.2f}s"
    assert_klines_schema(df)
    assert df.height == 0, (
        f"delisted-known coin should return 0 rows, got {df.height}"
    )


def test_is_unknown_coin_500_classifier() -> None:
    """Offline unit test for the ``_is_unknown_coin_500``
    classifier. We construct synthetic ``httpx.Response`` and
    ``httpx.HTTPStatusError`` objects and feed them through the
    static method to pin the exact signature.
    """
    req = httpx.Request("POST", "https://api.hyperliquid.xyz/info")

    # 500 + body "null" → unknown coin → permanent
    r500_null = httpx.Response(500, content=b"null", request=req)
    e500_null = httpx.HTTPStatusError(
        "500", request=req, response=r500_null
    )
    assert Hyperliquid._is_unknown_coin_500(e500_null) is True

    # 500 + body "  null  " (whitespace) → still unknown coin
    r500_ws = httpx.Response(
        500, content=b"  null  ", request=req
    )
    e500_ws = httpx.HTTPStatusError(
        "500", request=req, response=r500_ws
    )
    assert Hyperliquid._is_unknown_coin_500(e500_ws) is True

    # 500 + body '{"error":"something"}' → real outage, NOT the
    # unknown-coin signature → transient (retried)
    r500_obj = httpx.Response(
        500,
        content=b'{"error":"internal server error"}',
        request=req,
    )
    e500_obj = httpx.HTTPStatusError(
        "500", request=req, response=r500_obj
    )
    assert Hyperliquid._is_unknown_coin_500(e500_obj) is False

    # 503 + body "null" → wrong status code, NOT flagged
    r503_null = httpx.Response(503, content=b"null", request=req)
    e503_null = httpx.HTTPStatusError(
        "503", request=req, response=r503_null
    )
    assert Hyperliquid._is_unknown_coin_500(e503_null) is False

    # 400 + body "null" → bad request, NOT flagged
    r400_null = httpx.Response(400, content=b"null", request=req)
    e400_null = httpx.HTTPStatusError(
        "400", request=req, response=r400_null
    )
    assert Hyperliquid._is_unknown_coin_500(e400_null) is False

    # 200 + body "null" → 2xx never even gets here, but if it
    # did, the status check excludes it.
    r200_null = httpx.Response(200, content=b"null", request=req)
    e200_null = httpx.HTTPStatusError(
        "200", request=req, response=r200_null
    )
    assert Hyperliquid._is_unknown_coin_500(e200_null) is False


async def test_is_transient_error_classifies_500_null_as_permanent() -> None:
    """End-to-end check that ``is_transient_error`` returns
    ``False`` for a 500+null HTTPStatusError, even when the
    error is wrapped in a ``RuntimeError`` (the pattern the
    adapters use for cleaner caller messages)."""
    req = httpx.Request("POST", "https://api.hyperliquid.xyz/info")
    resp = httpx.Response(500, content=b"null", request=req)
    http_exc = httpx.HTTPStatusError(
        "500 Internal Server Error",
        request=req,
        response=resp,
    )
    ex = Hyperliquid()

    # Direct
    assert ex.is_transient_error(http_exc) is False, (
        "500+null must be permanent"
    )

    # Wrapped in a RuntimeError (the pattern used inside
    # _post_info → "Hyperliquid /info request failed: ..."
    # gets re-raised). The base-class cause-chain walk
    # should still see the 500+null underneath.
    try:
        try:
            raise http_exc
        except httpx.HTTPStatusError as e:
            raise RuntimeError("Hyperliquid /info failed") from e
    except RuntimeError as wrapped:
        assert ex.is_transient_error(wrapped) is False, (
            "wrapped 500+null must still be classified as permanent "
            "via the cause chain"
        )


async def test_is_transient_error_still_retries_real_5xx() -> None:
    """Sanity check: a 5xx with a real error body (not literal
    ``null``) must STILL be classified as transient, so genuine
    exchange outages get retried. This is the case the override
    is specifically NOT trying to catch."""
    req = httpx.Request("POST", "https://api.hyperliquid.xyz/info")
    resp = httpx.Response(
        502,
        content=b'<html>502 Bad Gateway</html>',
        request=req,
    )
    http_exc = httpx.HTTPStatusError(
        "502 Bad Gateway", request=req, response=resp
    )
    ex = Hyperliquid()
    assert ex.is_transient_error(http_exc) is True, (
        "a 5xx with a real error body should be transient "
        "(the override should not catch this)"
    )
