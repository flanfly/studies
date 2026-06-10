"""Tests for the Binance exchange adapter.

These tests run against the live Binance API. They require
``BINANCE_API_KEY`` and ``BINANCE_API_SECRET`` in ``.env``; the test
is skipped if either is missing.

Binance uses the symbol convention ``BTCUSDT`` (concatenated, upper
case) and the klines endpoint returns an explicit ``close_time`` per
candle, which we forward as ``close_ts`` (in microsecond resolution).

Binance borrow rates are per-asset (the same rate applies regardless
of pair, and the same for cross and isolated margin) and require the
API key to have margin permissions. If the key doesn't have that
permission, the rates come back as null and the test prints a note
but does not fail.
"""

from __future__ import annotations

import asyncio

import polars as pl
import pytest
import httpx
from httpx import AsyncClient

from live import Binance
from ._assertions import (
    assert_close_ts_matches_open,
    assert_klines_schema,
    assert_kline_candles_per_symbol,
    assert_lowercase,
    assert_open_ts_in_range,
    assert_pairs_joinable_to_klines,
    assert_pairs_schema,
)


WANTED = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


async def test_pairs(client: AsyncClient, binance: Binance) -> None:
    pairs = await binance.pairs_with_retry(client, quote_assets={"usdt", "usdc"})
    assert_pairs_schema(pairs)
    assert pairs.height > 0, "no pairs returned"

    # Symbols are ``UPPER(base)UPPER(quote)``.
    bad = pairs.filter(
        (pl.col("base").str.to_uppercase() + pl.col("quote").str.to_uppercase())
        != pl.col("symbol")
    )
    assert bad.height == 0, (
        f"Binance symbol should be UPPER(base)UPPER(quote) "
        f"({bad.height} mismatches)"
    )

    # ``cross_rate`` and ``isolated_rate`` are populated from the same
    # per-asset endpoint, so they should match in count.
    n_with_cross = pairs.filter(pl.col("cross_rate").is_not_null()).height
    n_with_iso = pairs.filter(pl.col("isolated_rate").is_not_null()).height
    assert n_with_cross == n_with_iso, (
        f"cross_rate ({n_with_cross}) and isolated_rate ({n_with_iso}) "
        f"counts differ"
    )

    sample = pairs.filter(pl.col("symbol").is_in(WANTED))
    assert sample.height == len(WANTED), (
        f"missing well-known pairs (got {sample.height}/{len(WANTED)})"
    )


async def test_klines(
    client: AsyncClient, binance: Binance, pairs_window: tuple
) -> None:
    start, end = pairs_window
    klines = pl.concat(
        await asyncio.gather(
            *(
                binance.klines_with_retry(client, sym, start, end)
                for sym in WANTED
            )
        )
    ).sort(["symbol", "open_ts"])

    assert_klines_schema(klines)
    assert_open_ts_in_range(klines, start, end)
    # Binance returns the close timestamp itself (ms resolution), so we
    # only assert that ``close_ts > open_ts`` -- not the exact ``+24h-1us``
    # relation.
    assert_close_ts_matches_open(klines, api_provided=True)
    assert_lowercase(klines, "base", "quote")
    assert_kline_candles_per_symbol(klines, 14, *WANTED)


async def test_klines_empty_range(
    client: AsyncClient, binance: Binance, pairs_window: tuple
) -> None:
    start, _ = pairs_window
    empty = await binance.klines_with_retry(client, "BTCUSDT", start, start)
    assert empty.height == 0
    assert_klines_schema(empty)


async def test_klines_paged(
    client: AsyncClient, binance: Binance, paged_window: tuple
) -> None:
    start, end = paged_window
    paged = await binance.klines_paged(
        client, "BTCUSDT", start_time=start, end_time=end
    )
    capped = await binance.klines_with_retry(
        client, "BTCUSDT", start_time=start, end_time=end
    )

    assert paged.height > 0
    assert capped.height <= Binance.MAX_KLINES, (
        f"klines() returned {capped.height} > MAX_KLINES={Binance.MAX_KLINES}"
    )
    assert paged.height >= capped.height
    assert_open_ts_in_range(paged, start, end)
    assert_close_ts_matches_open(paged, api_provided=True)


async def test_pairs_joinable_to_klines(
    client: AsyncClient, binance: Binance, pairs_window: tuple
) -> None:
    start, end = pairs_window
    pairs = await binance.pairs_with_retry(client, quote_assets={"usdt", "usdc"})
    klines = pl.concat(
        await asyncio.gather(
            *(
                binance.klines_with_retry(client, sym, start, end)
                for sym in WANTED
            )
        )
    )
    assert_pairs_joinable_to_klines(binance, pairs, klines)


# ----------------------------------------------------------------------
# offline tests for the borrow-rate error handling. These don't hit
# the network -- they construct ``httpx.Response`` objects directly
# and feed them through the helpers.
# ----------------------------------------------------------------------


def _make_response(
    status_code: int, body: dict | str
) -> httpx.Response:
    """Build a minimal ``httpx.Response`` from ``body``. ``body`` can
    be a ``dict`` (serialised to JSON) or a raw string."""
    if isinstance(body, dict):
        import json as _json
        body = _json.dumps(body)
    return httpx.Response(status_code, content=body.encode())


def test_is_asset_not_supported_matches_canonical_envelope() -> None:
    """The ``-11027`` "asset X is not supported" envelope is the
    case the helper is designed to catch. Both 400 and 404 are
    accepted (Binance currently uses 400 but we accept 404
    defensively)."""
    from live.binance import _is_asset_not_supported

    resp = _make_response(400, {"code": -11027, "msg": "asset AEUR is not supported"})
    assert _is_asset_not_supported(resp, "AEUR") is True
    resp = _make_response(404, {"code": -11027, "msg": "asset AEUR is not supported"})
    assert _is_asset_not_supported(resp, "AEUR") is True


def test_is_asset_not_supported_rejects_other_codes() -> None:
    """Other error codes (e.g. ``-1022`` "Signature is not valid",
    ``-1102`` "Mandatory parameter missing") are NOT silently
    swallowed -- they indicate real failures (auth, programming
    error) and should propagate to the warning log."""
    from live.binance import _is_asset_not_supported

    for body in [
        {"code": -1022, "msg": "Signature for this request is not valid."},
        {"code": -1102, "msg": "Mandatory parameter 'asset' was not sent"},
        {"code": -1021, "msg": "Timestamp outside recvWindow"},
    ]:
        resp = _make_response(400, body)
        assert _is_asset_not_supported(resp, "BTC") is False, body


def test_is_asset_not_supported_rejects_wrong_asset() -> None:
    """The ``msg`` is checked against the asset being queried: a
    ``-11027`` envelope for a *different* asset should not
    accidentally silence the current request."""
    from live.binance import _is_asset_not_supported

    resp = _make_response(400, {"code": -11027, "msg": "asset AEUR is not supported"})
    assert _is_asset_not_supported(resp, "BTC") is False


def test_is_asset_not_supported_rejects_non_json() -> None:
    """Non-JSON bodies (or empty bodies) return ``False`` so the
    warning still fires -- those are real failures (auth,
    network, etc.)."""
    from live.binance import _is_asset_not_supported

    resp = _make_response(400, "not json")
    assert _is_asset_not_supported(resp, "BTC") is False
    resp = _make_response(400, "")
    assert _is_asset_not_supported(resp, "BTC") is False
    # A non-dict JSON body (an array) also doesn't match.
    resp = _make_response(400, '["not", "a", "dict"]')
    assert _is_asset_not_supported(resp, "BTC") is False


async def test_fetch_borrow_rate_silences_unsupported_asset(
    client: AsyncClient, binance: Binance
) -> None:
    """A 400 / -11027 envelope for ``asset`` returns ``None``
    silently (no warning, no exception)."""
    from live.binance import _is_asset_not_supported

    # Simulate the response via a monkey-patch on the sapi helper.
    class FakeResp(httpx.Response):
        def __init__(self):
            super().__init__(
                status_code=400,
                content=b'{"code": -11027, "msg": "asset FOO is not supported"}',
            )
        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError(
                "400 Bad Request",
                request=httpx.Request("GET", "https://x/y"),
                response=self,
            )

    orig_sapi = binance._sapi_get
    async def fake_sapi(c, path, params):
        raise httpx.HTTPStatusError(
            "400",
            request=httpx.Request("GET", "https://x/y"),
            response=FakeResp(),
        )
    binance._sapi_get = fake_sapi
    try:
        result = await binance._fetch_borrow_rate(client, "FOO")
        assert result is None  # silent
    finally:
        binance._sapi_get = orig_sapi


async def test_fetch_borrow_rate_warns_on_other_4xx(
    client: AsyncClient, binance: Binance, caplog
) -> None:
    """A 4xx that's NOT the ``-11027`` envelope still logs a
    warning -- it's a real failure (auth, bad signature, etc.)."""
    class FakeResp(httpx.Response):
        def __init__(self):
            super().__init__(
                status_code=400,
                content=b'{"code": -1022, "msg": "Signature for this request is not valid."}',
            )
        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError(
                "400",
                request=httpx.Request("GET", "https://x/y"),
                response=self,
            )

    orig_sapi = binance._sapi_get
    async def fake_sapi(c, path, params):
        raise httpx.HTTPStatusError(
            "400",
            request=httpx.Request("GET", "https://x/y"),
            response=FakeResp(),
        )
    binance._sapi_get = fake_sapi
    try:
        with caplog.at_level("WARNING"):
            result = await binance._fetch_borrow_rate(client, "BTC")
        assert result is None
        # At least one Binance warning was emitted.
        binance_warnings = [r for r in caplog.records if "Binance borrowRate" in r.message]
        assert len(binance_warnings) >= 1
    finally:
        binance._sapi_get = orig_sapi


def test_borrow_rate_concurrency_default() -> None:
    """``BORROW_RATE_CONCURRENCY`` defaults to 50, which is well
    under sapi's per-minute rate limit and stops the
    ``-1021`` recvWindow storm we saw when running with 400+
    concurrent requests."""
    assert Binance.BORROW_RATE_CONCURRENCY == 50


def test_parse_binance_error_code() -> None:
    """``_parse_binance_error_code`` returns the ``code`` field as
    an ``int`` for valid Binance error envelopes, ``None``
    otherwise. Used by the retry classifier and the
    silent-skip path to inspect the response body without
    re-parsing it twice."""
    from live.binance import _parse_binance_error_code

    assert _parse_binance_error_code(_make_response(400, {"code": -11027, "msg": "x"})) == -11027
    assert _parse_binance_error_code(_make_response(400, {"code": -1021, "msg": "x"})) == -1021
    # Non-int code: ignored.
    assert _parse_binance_error_code(_make_response(400, {"code": "abc", "msg": "x"})) is None
    # Non-dict body: ignored.
    assert _parse_binance_error_code(_make_response(400, '["a", "b"]')) is None
    # Non-JSON body: ignored.
    assert _parse_binance_error_code(_make_response(400, "plain text")) is None
    # Empty body: ignored.
    assert _parse_binance_error_code(_make_response(400, "")) is None


async def test_fetch_borrow_rate_retries_transient_code(
    client: AsyncClient, binance: Binance, caplog, monkeypatch
) -> None:
    """A 4xx with a code in ``TRANSIENT_CODES`` is retried with
    backoff. We patch ``_sapi_get`` to fail twice with -1021 and
    then succeed, and assert the call returns the rate rather than
    logging a "gave up" warning.
    """
    import json as _json

    call_count = 0

    async def fake_sapi(c, path, params):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            # Transient failure.
            resp = httpx.Response(
                400,
                content=_json.dumps({"code": -1021, "msg": "Timestamp outside recvWindow."}).encode(),
            )
            raise httpx.HTTPStatusError(
                "400",
                request=httpx.Request("GET", "https://x/y"),
                response=resp,
            )
        # Success.
        return [{"asset": "BTC", "timestamp": 1781049600000, "dailyInterestRate": "0.00001069", "vipLevel": 0}]

    # Speed up the retry for the test: make backoff near-zero.
    binance.RETRY_BASE_DELAY = 0.001
    binance.RETRY_MAX_DELAY = 0.001
    binance._sapi_get = fake_sapi
    try:
        result = await binance._fetch_borrow_rate(client, "BTC")
    finally:
        binance._sapi_get = lambda c, p, params: None
    assert call_count == 3, f"expected 3 calls (2 failures + 1 success), got {call_count}"
    assert result is not None and result > 0
    # The two retry warnings should be present, but NOT the
    # final "gave up" warning.
    gave_up = [r for r in caplog.records if "gave up" in r.message]
    assert not gave_up, f"unexpected gave-up warning: {[r.message for r in gave_up]}"


async def test_fetch_borrow_rate_gives_up_after_max_attempts(
    client: AsyncClient, binance: Binance, caplog
) -> None:
    """If all ``RETRY_ATTEMPTS`` attempts return a transient code,
    we log a final ``gave up`` warning and return ``None``. The
    surrounding batch is unaffected.
    """
    import json as _json

    call_count = 0

    async def fake_sapi(c, path, params):
        nonlocal call_count
        call_count += 1
        resp = httpx.Response(
            400,
            content=_json.dumps({"code": -1021, "msg": "Timestamp outside recvWindow."}).encode(),
        )
        raise httpx.HTTPStatusError(
            "400",
            request=httpx.Request("GET", "https://x/y"),
            response=resp,
        )

    binance.RETRY_BASE_DELAY = 0.001
    binance.RETRY_MAX_DELAY = 0.001
    orig = binance._sapi_get
    binance._sapi_get = fake_sapi
    try:
        with caplog.at_level("WARNING"):
            result = await binance._fetch_borrow_rate(client, "BTC")
    finally:
        binance._sapi_get = orig
    assert call_count == Binance.RETRY_ATTEMPTS
    assert result is None
    gave_up = [r for r in caplog.records if "gave up" in r.message and "BTC" in r.message]
    assert len(gave_up) == 1, f"expected 1 gave-up warning, got {len(gave_up)}"
