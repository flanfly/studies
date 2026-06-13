"""Tests for the ``funding_rate`` column on ``Exchange.pairs()``.

The ``funding_rate`` column carries the **current** funding rate of
the stablecoin-settled perpetual contract on the same base/quote,
annualised as APR. It's ``None`` for exchanges with no native
perpetuals (Kraken) or for coins whose perp doesn't exist on the
exchange that does support perps.

These tests cover:

  * Schema contract: the column is present, is ``pl.Float64``, and
    is nullable.
  * Annualisation helper: ``Exchange.annualize_funding_rate`` is
    the right scaling formula for an 8h funding schedule.
  * Per-exchange online coverage: each exchange that supports
    perps populates a meaningful fraction of ``funding_rate``
    values; Kraken populates none.
  * Kraken's offline default: ``_fetch_funding_rates`` returns
    ``{}`` (no perpetuals).
  * Funding rate values are reasonable annual rates (the test
    bounds are intentionally loose to allow for legitimately
    extreme perp funding on illiquid markets).
"""

from __future__ import annotations

import asyncio
import datetime as dt

import httpx
import polars as pl
import pytest
from httpx import AsyncClient

from live import (
    Binance,
    HTX,
    Kraken,
    KuCoin,
    Exchange,
)
from ._assertions import assert_pairs_schema


# ----------------------------------------------------------------------
# offline tests (no network)
# ----------------------------------------------------------------------


def test_annualize_funding_rate_8h() -> None:
    """8h funding schedule (Binance, HTX, KuCoin): 1bp per
    interval = 3bp/day = ~10.95% APR."""
    ex = Kraken()  # any concrete instance; the helper is on the base class
    assert ex.annualize_funding_rate(0.0001, 8.0) == pytest.approx(0.1095, rel=1e-9)


def test_annualize_funding_rate_1h() -> None:
    """1h funding schedule (Kraken-Futures-style, hypothetical):
    1bp per interval = 24bp/day = ~87.6% APR."""
    ex = Kraken()
    assert ex.annualize_funding_rate(0.0001, 1.0) == pytest.approx(0.876, rel=1e-9)


def test_annualize_funding_rate_4h() -> None:
    """4h schedule: 1bp per interval = 6bp/day = ~21.9% APR."""
    ex = Kraken()
    assert ex.annualize_funding_rate(0.0001, 4.0) == pytest.approx(0.219, rel=1e-9)


def test_annualize_funding_rate_negative() -> None:
    """Negative funding rates (shorts paying longs) annualise
    the same way -- just a sign flip."""
    ex = Kraken()
    assert ex.annualize_funding_rate(-0.0002, 8.0) == pytest.approx(-0.219, rel=1e-9)


def test_annualize_funding_rate_rejects_zero_interval() -> None:
    """``interval_hours == 0`` is degenerate and would divide by
    zero -- the helper rejects it explicitly."""
    ex = Kraken()
    with pytest.raises(ValueError, match="interval_hours"):
        ex.annualize_funding_rate(0.0001, 0.0)


def test_annualize_funding_rate_rejects_negative_interval() -> None:
    ex = Kraken()
    with pytest.raises(ValueError, match="interval_hours"):
        ex.annualize_funding_rate(0.0001, -1.0)


def test_pairs_schema_includes_funding_rate() -> None:
    """The ``PAIRS_SCHEMA`` on the base class lists
    ``funding_rate`` as the last column with dtype
    ``pl.Float64``."""
    assert "funding_rate" in Exchange.PAIRS_SCHEMA
    assert Exchange.PAIRS_SCHEMA["funding_rate"] == pl.Float64


def test_kraken_funding_rates_default_empty() -> None:
    """Kraken has no native perpetuals. The base-class default
    ``_fetch_funding_rates`` returns ``{}`` and is used as-is."""
    import asyncio

    async def t() -> dict[str, float]:
        kr = Kraken()
        return await kr._fetch_funding_rates(None, ["btc", "eth", "sol"])

    assert asyncio.run(t()) == {}


def test_empty_pairs_df_includes_funding_rate_column() -> None:
    """``empty_pairs_df()`` carries the new column. Used by the
    ``pairs()`` fast path when no rows match the quote filter."""
    df = Binance.empty_pairs_df()
    assert "funding_rate" in df.columns
    assert df.schema["funding_rate"] == pl.Float64


# ----------------------------------------------------------------------
# live tests (hit the exchanges)
# ----------------------------------------------------------------------


# Bound: a meaningful fraction of active USDT-margined pairs on
# each perp-supporting exchange should have a non-null funding
# rate. The bound is conservative -- the perp coverage is much
# higher in practice (e.g. Binance: 649/719, KuCoin: 556/945) but
# we don't want a flaky test if a new spot coin is listed
# faster than its perp.
MIN_PERP_COVERAGE_FRACTION = 0.05


async def test_binance_pairs_have_funding_rate(
    client: AsyncClient, binance: Binance
) -> None:
    """Binance spot pairs should have a meaningful fraction of
    non-null ``funding_rate`` values (USDT- or USDC-margined
    perps exist for most active spot pairs)."""
    pairs = await binance.pairs_with_retry(
        client, quote_assets={"usdt", "usdc"}
    )
    assert_pairs_schema(pairs)
    n_total = pairs.height
    assert n_total > 0
    n_with = pairs.filter(pl.col("funding_rate").is_not_null()).height
    assert n_with / n_total >= MIN_PERP_COVERAGE_FRACTION, (
        f"Binance funding coverage too low: "
        f"{n_with}/{n_total} = {n_with / n_total:.1%}"
    )
    # Well-known pairs definitely have perps.
    btc = pairs.filter(pl.col("symbol") == "BTCUSDT")
    assert btc.height == 1
    assert btc["funding_rate"][0] is not None
    # And the rate is in a reasonable annual range. We allow up
    # to ±500% APR to absorb short-lived funding spikes on
    # illiquid markets, but BTC's perp is liquid enough to be
    # within ±50%.
    rate = btc["funding_rate"][0]
    assert -0.5 < rate < 0.5, f"BTCUSDT funding_rate={rate:.4f} implausible"


async def test_kucoin_pairs_have_funding_rate(
    client: AsyncClient, kucoin: KuCoin
) -> None:
    """KuCoin spot pairs should have a meaningful fraction of
    non-null ``funding_rate`` values."""
    pairs = await kucoin.pairs_with_retry(
        client, quote_assets={"usdt", "usdc"}
    )
    assert_pairs_schema(pairs)
    n_total = pairs.height
    assert n_total > 0
    n_with = pairs.filter(pl.col("funding_rate").is_not_null()).height
    assert n_with / n_total >= MIN_PERP_COVERAGE_FRACTION, (
        f"KuCoin funding coverage too low: "
        f"{n_with}/{n_total} = {n_with / n_total:.1%}"
    )
    # BTC spot pair should have a USDT-margined perp on KuCoin.
    btc = pairs.filter(pl.col("symbol") == "BTC-USDT")
    assert btc.height == 1
    assert btc["funding_rate"][0] is not None


async def test_htx_pairs_have_funding_rate(
    client: AsyncClient, htx: HTX
) -> None:
    """HTX spot pairs should have a meaningful fraction of
    non-null ``funding_rate`` values from the USDT-margined
    linear-swap funding-rate endpoint."""
    pairs = await htx.pairs_with_retry(
        client, quote_assets={"usdt", "usdc"}
    )
    assert_pairs_schema(pairs)
    n_total = pairs.height
    assert n_total > 0
    n_with = pairs.filter(pl.col("funding_rate").is_not_null()).height
    assert n_with / n_total >= MIN_PERP_COVERAGE_FRACTION, (
        f"HTX funding coverage too low: "
        f"{n_with}/{n_total} = {n_with / n_total:.1%}"
    )
    btc = pairs.filter(pl.col("symbol") == "btcusdt")
    assert btc.height == 1
    assert btc["funding_rate"][0] is not None


async def test_kraken_funding_rate_is_all_null(
    client: AsyncClient, kraken: Kraken
) -> None:
    """Kraken has no native perpetuals (only dated futures,
    which settle and don't have a funding rate). The
    ``funding_rate`` column on Kraken's pairs output is
    always ``None``."""
    pairs = await kraken.pairs_with_retry(
        client, quote_assets={"usdt", "usdc"}
    )
    assert_pairs_schema(pairs)
    n_total = pairs.height
    if n_total == 0:
        pytest.skip("no Kraken pairs returned (likely no USDT/USDC pairs)")
    n_with = pairs.filter(pl.col("funding_rate").is_not_null()).height
    assert n_with == 0, (
        f"Kraken returned {n_with} non-null funding_rate values, "
        f"expected all-null (Kraken has no perpetuals)"
    )


async def test_pairs_funding_rate_dtype_is_float() -> None:
    """All four exchanges produce a ``funding_rate`` column of
    dtype ``pl.Float64``. The schema check is per-exchange; we
    use the static schema for a single round-trip."""
    df = Binance.empty_pairs_df()
    assert df.schema["funding_rate"] == pl.Float64
    df = HTX.empty_pairs_df()
    assert df.schema["funding_rate"] == pl.Float64
    df = KuCoin.empty_pairs_df()
    assert df.schema["funding_rate"] == pl.Float64
    df = Kraken.empty_pairs_df()
    assert df.schema["funding_rate"] == pl.Float64


# ----------------------------------------------------------------------
# offline unit tests for the funding fetchers' error classification.
# These don't hit the network -- they construct ``httpx.Response``
# objects directly and feed them through the per-exchange fetch
# helpers. The goal is to pin the "no contract -> silent None"
# vs "rate limit -> TransientError" vs "real failure -> warn
# and return None" branches.
# ----------------------------------------------------------------------


def _make_response(
    status_code: int, body: dict | str
) -> httpx.Response:
    """Build a minimal ``httpx.Response`` from ``body``. ``body`` can
    be a ``dict`` (serialised to JSON) or a raw string."""
    import json as _json

    if isinstance(body, dict):
        body = _json.dumps(body)
    return httpx.Response(status_code, content=body.encode())


# --- Binance -------------------------------------------------------------


async def test_binance_funding_rate_invalid_symbol_returns_none(
    client: AsyncClient, binance: Binance
) -> None:
    """A 400 with ``-1121`` ("Invalid symbol") is the expected
    response for a base with no USDT- or USDC-margined
    perpetual. We return ``None`` silently after exhausting
    both settle currencies."""

    async def fake_request(c, contract):
        return {"code": -1121, "msg": "Invalid symbol."}

    binance._funding_request = fake_request
    try:
        # The fetcher returns None only after BOTH settle
        # currencies are tried. We pass base="DOGEWIFHAT" which
        # is unlikely to have a perpetual.
        result = await binance._fetch_funding_rate(
            client, base="DOGEWIFHAT", quote="USDT"
        )
        assert result is None
    finally:
        del binance._funding_request


async def test_binance_funding_rate_success(
    client: AsyncClient, binance: Binance
) -> None:
    """A 200 response with ``lastFundingRate`` annualises to
    ``rate * 3 * 365`` (8h funding schedule). Both settle-
    currency probes succeed; the USDT one is returned."""
    call_log: list[str] = []

    async def fake_request(c, contract):
        call_log.append(contract)
        return {
            "symbol": contract,
            "markPrice": "60000.0",
            "lastFundingRate": "0.0001",
            "nextFundingTime": 1781078400000,
        }

    binance._funding_request = fake_request
    try:
        result = await binance._fetch_funding_rate(
            client, base="BTC", quote="USDT"
        )
        assert result is not None
        # 1bp per 8h interval * 3/day * 365 = 10.95% APR.
        assert result == pytest.approx(0.1095, rel=1e-9)
        # USDT was probed first; the helper returns immediately on
        # the first successful settle currency.
        assert call_log == ["BTCUSDT"]
    finally:
        del binance._funding_request


# --- KuCoin --------------------------------------------------------------


async def test_kucoin_funding_rate_415000_returns_none(
    client: AsyncClient, kucoin: KuCoin
) -> None:
    """A ``code == "415000"`` ("funding rate is not supported")
    on BOTH USDT and USDC means no contract -- ``None``."""
    call_count = 0

    async def fake_request(c, contract):
        nonlocal call_count
        call_count += 1
        return {
            "code": "415000",
            "msg": "funding rate is not supported",
        }

    kucoin._funding_request = fake_request
    try:
        result = await kucoin._fetch_funding_rate(
            client, base="DOGEWIFHAT", quote="USDT"
        )
        assert result is None
        # Both USDT and USDC settle currencies are tried.
        assert call_count == 2
    finally:
        del kucoin._funding_request


async def test_kucoin_funding_rate_uses_granularity_from_response(
    client: AsyncClient, kucoin: KuCoin
) -> None:
    """If the response reports a non-default ``granularity`` (in
    ms), the helper uses that to annualise. KuCoin occasionally
    moves individual contracts to a 4h schedule."""
    async def fake_request(c, contract):
        # 4h = 14_400_000 ms.
        return {
            "code": "200000",
            "data": {
                "symbol": contract,
                "granularity": 14_400_000,
                "value": 0.0001,
                "timePoint": 1781049600000,
            },
        }

    kucoin._funding_request = fake_request
    try:
        result = await kucoin._fetch_funding_rate(
            client, base="BTC", quote="USDT"
        )
        assert result is not None
        # 1bp per 4h * 6/day * 365 = 21.9% APR.
        assert result == pytest.approx(0.219, rel=1e-9)
    finally:
        del kucoin._funding_request


# --- HTX -----------------------------------------------------------------


async def test_htx_funding_rate_1332_returns_none(
    client: AsyncClient, htx: HTX
) -> None:
    """``err_code == 1332`` "The perpetual contract does not
    exist" is the expected response for a base with no USDT-
    margined linear swap. We return ``None`` silently."""
    async def fake_request(c, contract):
        return {
            "status": "error",
            "err_code": 1332,
            "err_msg": "The perpetual contract does not exist.",
            "ts": 1781077880852,
        }

    htx._funding_request = fake_request
    try:
        result = await htx._fetch_funding_rate(
            client, base="DOGEWIFHAT", quote="usdt"
        )
        assert result is None
    finally:
        del htx._funding_request


async def test_htx_funding_rate_skips_non_usdt_partition(
    client: AsyncClient, htx: HTX
) -> None:
    """HTX also lists USDC- and inverse-margined linear swaps.
    We skip those (``trade_partition != "USDT"``) to honour
    the stablecoin-settled-only convention."""
    async def fake_request(c, contract):
        return {
            "status": "ok",
            "data": {
                "contract_code": contract,
                "symbol": "BTC",
                "funding_rate": "-0.0001",
                "fee_asset": "USDC",
                "trade_partition": "USDC",  # NOT USDT
            },
        }

    htx._funding_request = fake_request
    try:
        result = await htx._fetch_funding_rate(
            client, base="BTC", quote="usdt"
        )
        assert result is None
    finally:
        del htx._funding_request


async def test_htx_funding_rate_success(
    client: AsyncClient, htx: HTX
) -> None:
    """A ``status: ok`` envelope with ``trade_partition: USDT``
    annualises the funding rate to 3 * 365 = 1095x the per-
    interval value."""
    async def fake_request(c, contract):
        return {
            "status": "ok",
            "data": {
                "contract_code": contract,
                "symbol": "BTC",
                "funding_rate": "0.0001",
                "fee_asset": "USDT",
                "trade_partition": "USDT",
            },
        }

    htx._funding_request = fake_request
    try:
        result = await htx._fetch_funding_rate(
            client, base="BTC", quote="usdt"
        )
        assert result is not None
        assert result == pytest.approx(0.1095, rel=1e-9)
    finally:
        del htx._funding_request
