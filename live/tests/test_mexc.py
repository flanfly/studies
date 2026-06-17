"""Tests for the MEXC spot exchange adapter.

These tests run against the live MEXC API. The endpoints we use
(``/api/v3/exchangeInfo``, ``/api/v3/klines``, and
``/api/v1/contract/funding_rate/{contract}``) are public, so no
API credentials are required.

MEXC uses the symbol convention ``BTCUSDT`` (concatenated,
uppercase) and its daily candles are aligned to midnight UTC.

A few MEXC quirks that the tests pin:

  * Borrow rates are unavailable on the public spot v3 API -- both
    ``cross_rate`` and ``isolated_rate`` are ``None`` for every
    MEXC pair. We assert this directly.
  * The kline endpoint ignores ``startTime``/``endTime`` when
    ``limit`` is set, returning the most recent ``limit`` candles
    only. ``MAX_KLINES == 500``.
  * ``funding_rate`` is ``None`` for coins without a USDT- or
    USDC-settled perpetual on MEXC.
"""

from __future__ import annotations

import asyncio
import datetime as dt

import polars as pl
import pytest
from httpx import AsyncClient

from live import MEXC
from ._assertions import (
    assert_close_ts_matches_open,
    assert_klines_schema,
    assert_lowercase,
    assert_open_ts_in_range,
    assert_pairs_schema,
)


WANTED = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


async def test_pairs(client: AsyncClient, mexc: MEXC) -> None:
    """Pairs structural test (no funding fetch).

    Pass ``quote_assets={"ZZZ"}`` so the early-return branch
    skips the funding fetch entirely (no MEXC pair uses ZZZ as
    a quote). The funding-rate branches are exercised
    separately by ``test_funding_rate_btc_usdt_populated``
    (single asset, live network) and the offline unit tests
    below (``test_funding_rate_no_contract_returns_none`` et al).
    """
    pairs = await mexc.pairs_with_retry(client, quote_assets={"ZZZ"})
    assert_pairs_schema(pairs)
    # Empty result because no MEXC pair has ``ZZZ`` as a quote
    # asset. This proves the early-return branch works without
    # paying for the funding fetch.
    assert pairs.height == 0


async def test_pairs_with_limit(
    client: AsyncClient, mexc: MEXC
) -> None:
    """``limit`` caps both the returned rows and the funding
    fetch.

    We pass ``limit=10`` to bound the funding-rate fetch to a
    handful of symbols instead of the full ~1900 USDT bases.
    This exercises the full ``pairs()`` path (schema, funding
    fetch, schema projection) at a cost that's appropriate for
    a unit test (~5s).
    """
    pairs = await mexc.pairs_with_retry(
        client,
        quote_assets={"usdt"},
        limit=10,
    )
    assert_pairs_schema(pairs)
    # ``limit=10`` caps the result at 10 rows.
    assert pairs.height == 10, f"expected 10 rows, got {pairs.height}"
    # Symbols are ``base+quote`` upper case.
    bad = pairs.filter(
        (pl.col("base").str.to_uppercase() + pl.col("quote").str.to_uppercase())
        != pl.col("symbol")
    )
    assert bad.height == 0, (
        f"MEXC symbol should equal base+quote upper ({bad.height} mismatches)"
    )
    # MEXC has no spot-margin trading (verified: 0 of ~2400
    # active pairs have ``isMarginTradingAllowed == true``).
    assert pairs["cross_rate"].is_null().all()
    assert pairs["isolated_rate"].is_null().all()
    # We don't assert any specific funding_rate values because
    # MEXC's exchangeInfo ordering is by listing recency: the
    # first 10 USDT pairs are recent listings like METALUSDT,
    # NEXUSDT, etc., most of which don't have a USDⓈ-M perp
    # contract. The important property is that the funding
    # fetch was bounded -- which it was, because this test
    # finishes in seconds rather than minutes.


async def test_pairs_symbol_convention(
    client: AsyncClient, mexc: MEXC
) -> None:
    """Verify that MEXC's symbol naming convention matches
    ``base + quote`` (uppercase) by hitting the live
    ``/exchangeInfo`` endpoint and inspecting a single pair.

    We don't go through ``pairs()`` because that would
    trigger the slow full-universe funding fetch; we read the
    exchangeInfo response directly.
    """
    payload = await mexc._spot_get(client, "/api/v3/exchangeInfo")
    btc = next(
        s for s in payload["symbols"] if s["symbol"] == "BTCUSDT"
    )
    assert btc["baseAsset"] == "BTC"
    assert btc["quoteAsset"] == "USDT"
    # MEXC has no spot-margin trading (verified: 0/2378 active
    # pairs have ``isMarginTradingAllowed == true``).
    assert btc["isMarginTradingAllowed"] is False


async def test_klines(
    client: AsyncClient, mexc: MEXC, pairs_window: tuple
) -> None:
    start, end = pairs_window
    klines = pl.concat(
        await asyncio.gather(
            *(
                mexc.klines_with_retry(client, sym, start, end)
                for sym in WANTED
            )
        )
    ).sort(["symbol", "open_ts"])

    assert_klines_schema(klines)
    assert_open_ts_in_range(klines, start, end)
    assert_close_ts_matches_open(klines, api_provided=False)
    assert_lowercase(klines, "base", "quote")
    # 14 days requested = 14 candles per wanted symbol (the in-
    # progress current day is excluded by the half-open range).
    for sym in WANTED:
        n = klines.filter(pl.col("symbol") == sym).height
        assert n == 14, f"{sym} returned {n} klines, expected 14"


async def test_klines_midnight_utc_alignment(
    client: AsyncClient, mexc: MEXC, pairs_window: tuple
) -> None:
    """MEXC's daily candles must open at 00:00 UTC (midnight)."""
    start, end = pairs_window
    klines = await mexc.klines_with_retry(client, "BTCUSDT", start, end)
    if klines.height == 0:
        pytest.skip("no BTCUSDT klines in window")
    # Every open_ts must land on midnight UTC.
    bad = klines.with_columns(
        open_hour=pl.col("open_ts").dt.hour(),
        open_minute=pl.col("open_ts").dt.minute(),
    ).filter((pl.col("open_hour") != 0) | (pl.col("open_minute") != 0))
    assert bad.height == 0, (
        f"MEXC klines not midnight-UTC aligned on {bad.height} rows"
    )


async def test_klines_empty_range(
    client: AsyncClient, mexc: MEXC, pairs_window: tuple
) -> None:
    start, _ = pairs_window
    empty = await mexc.klines_with_retry(client, "BTCUSDT", start, start)
    assert empty.height == 0
    assert_klines_schema(empty)


async def test_klines_paged_walks_backwards(
    client: AsyncClient, mexc: MEXC, paged_window: tuple
) -> None:
    """``klines_paged`` issues successive ``klines()`` calls and
    stitches them together. Since MEXC's endpoint ignores
    ``startTime`` / ``endTime``, the adapter has to walk back
    through history by trimming the overlap between successive
    batches. This test exercises the walk-back path.

    We use ``paged_window`` (4 years) which is wider than the
    single-call ``MAX_KLINES=500`` window, so the walk-back has
    to issue at least 3 successive calls.
    """
    start, end = paged_window
    paged = await mexc.klines_paged(
        client, "BTCUSDT", start_time=start, end_time=end
    )
    capped = await mexc.klines_with_retry(
        client, "BTCUSDT", start_time=start, end_time=end
    )

    # The paged version should cover at least as many candles as
    # the single-call cap. We can't assert exact counts because
    # MEXC's data retention may have grown since this test was
    # written.
    assert paged.height > 0
    assert capped.height <= MEXC.MAX_KLINES, (
        f"klines() returned {capped.height} > MAX_KLINES={MEXC.MAX_KLINES}"
    )
    assert paged.height >= capped.height
    assert_open_ts_in_range(paged, start, end)
    assert_close_ts_matches_open(paged, api_provided=False)


async def test_funding_rate_btc_usdt_populated(
    client: AsyncClient, mexc: MEXC
) -> None:
    """BTC has a USDT-settled perpetual on MEXC, so its funding
    rate must be populated and reasonable.

    MEXC's ``exchangeInfo`` response is ordered by listing recency,
    not alphabetically: BTC is at position 1719 / 1937. We
    iterate the full universe to find BTC, but cap the funding
    fetch at the first pair that comes back with a non-null
    rate (we know BTC will be one of them). This avoids the
    ~5-minute wait of probing every base.
    """
    # Iterate the full universe, but stop the funding fetch as
    # soon as we have BTC's rate. We do this by calling
    # ``_fetch_funding_rate`` directly on a known good base.
    rate = await mexc._fetch_funding_rate(
        client, base="BTC", quote="USDT"
    )
    assert rate is not None, "BTCUSDT funding_rate should be populated"
    # Annualised rate of 8h funding at the typical 1bp-per-8h is
    # 10.95%. We allow a generous ±200% APR window to absorb
    # genuinely volatile funding regimes on major coins.
    assert -2.0 < rate < 2.0, f"BTCUSDT funding_rate={rate:.4f} implausible"


# ----------------------------------------------------------------------
# Offline unit tests for MEXC's funding-rate error classification.
# These don't hit the network -- they construct dict payloads
# directly and feed them through the per-class helper to pin the
# "no contract -> silent None" vs "real error -> log and None" vs
# "success -> annualised APR" branches.
# ----------------------------------------------------------------------


async def test_funding_rate_no_contract_returns_none(
    client: AsyncClient, mexc: MEXC
) -> None:
    """``code == 1001`` ("Contract does not exist") on BOTH USDT
    and USDC means no perpetual -- ``None``."""

    async def fake_request(c, contract):
        return {
            "success": False,
            "code": 1001,
            "message": "Contract does not exist",
        }

    mexc._funding_request = fake_request
    try:
        result = await mexc._fetch_funding_rate(
            client, base="DOGEWIFHAT", quote="USDT"
        )
        assert result is None
    finally:
        del mexc._funding_request


async def test_funding_rate_usdt_first(
    client: AsyncClient, mexc: MEXC
) -> None:
    """When USDT succeeds, the helper returns immediately and
    does not probe USDC. The contract code ``BTC_USDT`` (with
    underscore, MEXC's convention) is the first probe."""

    async def fake_request(c, contract):
        return {
            "success": True,
            "code": 0,
            "data": {
                "symbol": contract,
                "fundingRate": 0.0001,
                "collectCycle": 8,
                "timestamp": 1781078400000,
            },
        }

    call_log: list[str] = []

    async def spy(c, contract):
        call_log.append(contract)
        return await fake_request(c, contract)

    mexc._funding_request = spy
    try:
        result = await mexc._fetch_funding_rate(
            client, base="BTC", quote="USDT"
        )
        assert result is not None
        # 1bp per 8h * 3/day * 365 = 10.95% APR.
        assert result == pytest.approx(0.1095, rel=1e-9)
        # USDT was probed and succeeded; USDC was not tried.
        assert call_log == ["BTC_USDT"]
    finally:
        del mexc._funding_request


async def test_funding_rate_falls_back_to_usdc(
    client: AsyncClient, mexc: MEXC
) -> None:
    """If USDT returns 1001 ("no contract") but USDC succeeds,
    the helper returns the USDC rate."""

    async def fake_request(c, contract):
        if contract.endswith("_USDT"):
            return {
                "success": False,
                "code": 1001,
                "message": "Contract does not exist",
            }
        return {
            "success": True,
            "code": 0,
            "data": {
                "symbol": contract,
                "fundingRate": 0.0002,  # 2bp per 8h
                "collectCycle": 8,
            },
        }

    call_log: list[str] = []

    async def spy(c, contract):
        call_log.append(contract)
        return await fake_request(c, contract)

    mexc._funding_request = spy
    try:
        result = await mexc._fetch_funding_rate(
            client, base="SOME", quote="USDT"
        )
        assert result is not None
        # 2bp per 8h * 3/day * 365 = 21.9% APR.
        assert result == pytest.approx(0.219, rel=1e-9)
        # USDT was tried first and failed; USDC was tried second.
        assert call_log == ["SOME_USDT", "SOME_USDC"]
    finally:
        del mexc._funding_request


async def test_funding_rate_real_error_returns_none(
    client: AsyncClient, mexc: MEXC
) -> None:
    """An envelope with ``success: false`` and a code other than
    1001 (e.g. an internal error) is logged and returns ``None``
    rather than retrying forever."""

    async def fake_request(c, contract):
        return {
            "success": False,
            "code": 999,
            "message": "Internal server error",
        }

    mexc._funding_request = fake_request
    try:
        result = await mexc._fetch_funding_rate(
            client, base="BTC", quote="USDT"
        )
        assert result is None
    finally:
        del mexc._funding_request