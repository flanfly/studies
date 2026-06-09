"""Tests for the base-class retry logic in :class:`Exchange`.

These tests don't hit any exchange API. They instantiate concrete
adapters (``Kraken`` doesn't need credentials; the others are skipped
if ``.env`` is missing) and call ``_retry`` with a stub coroutine that
fails a controllable number of times before succeeding.
"""

from __future__ import annotations

import httpx
import pytest
from httpx import AsyncClient

from live import Kraken, TransientError


async def test_retry_succeeds_on_first_attempt() -> None:
    ex = Kraken()
    failures: list[BaseException] = []
    calls = {"n": 0}

    async def stub(client: AsyncClient, quote_assets: set[str]) -> str:
        calls["n"] += 1
        if failures:
            raise failures.pop(0)
        return "ok"

    ex.pairs = stub  # type: ignore[assignment]
    result = await ex._retry("pairs", client=None, quote_assets=set())  # type: ignore[arg-type]
    assert result == "ok"
    assert calls["n"] == 1


async def test_retry_succeeds_after_two_transient_failures() -> None:
    ex = Kraken()

    # Replace ``pairs`` with a stub that fails twice then succeeds.
    failures: list[BaseException] = [
        TransientError("rate limit"),
        TransientError("rate limit"),
    ]

    async def stub(client: AsyncClient, quote_assets: set[str]) -> str:
        if failures:
            raise failures.pop(0)
        return "ok"

    ex.pairs = stub  # type: ignore[assignment]
    # Speed the test up by tightening the retry policy.
    ex.RETRY_BASE_DELAY = 0.0
    ex.RETRY_MAX_DELAY = 0.0
    result = await ex._retry("pairs", client=None, quote_assets=set())  # type: ignore[arg-type]
    assert result == "ok"


async def test_retry_propagates_non_transient_immediately() -> None:
    ex = Kraken()

    calls = {"n": 0}

    async def stub(client: AsyncClient, quote_assets: set[str]) -> str:
        calls["n"] += 1
        raise ValueError("nope")

    ex.pairs = stub  # type: ignore[assignment]
    with pytest.raises(ValueError, match="nope"):
        await ex._retry("pairs", client=None, quote_assets=set())  # type: ignore[arg-type]
    assert calls["n"] == 1, "non-transient error should not be retried"


async def test_retry_exhausts_attempts_then_raises() -> None:
    ex = Kraken()

    calls = {"n": 0}

    async def stub(client: AsyncClient, quote_assets: set[str]) -> str:
        calls["n"] += 1
        raise TransientError("permanent transient")

    ex.pairs = stub  # type: ignore[assignment]
    # Tighten policy so the test runs fast.
    ex.RETRY_BASE_DELAY = 0.0
    ex.RETRY_MAX_DELAY = 0.0
    ex.RETRY_ATTEMPTS = 3
    with pytest.raises(TransientError, match="permanent transient"):
        await ex._retry("pairs", client=None, quote_assets=set())  # type: ignore[arg-type]
    assert calls["n"] == 3, f"expected 3 attempts, got {calls['n']}"


def test_default_is_transient_error_recognises_common_cases() -> None:
    """The base implementation should classify ``TransientError``,
    ``httpx`` 429/5xx, and ``httpx`` transport errors as transient;
    plain ``RuntimeError`` and 4xx HTTP errors should not be retried.
    """
    import httpx

    ex = Kraken()
    assert ex.is_transient_error(TransientError("x"))

    # 429
    resp_429 = httpx.Response(429, request=httpx.Request("GET", "https://x"))
    assert ex.is_transient_error(
        httpx.HTTPStatusError("429", request=resp_429.request, response=resp_429)
    )

    # 5xx
    resp_500 = httpx.Response(500, request=httpx.Request("GET", "https://x"))
    assert ex.is_transient_error(
        httpx.HTTPStatusError("500", request=resp_500.request, response=resp_500)
    )

    # 4xx (not 429) -- should NOT be retried
    resp_400 = httpx.Response(400, request=httpx.Request("GET", "https://x"))
    assert not ex.is_transient_error(
        httpx.HTTPStatusError("400", request=resp_400.request, response=resp_400)
    )

    # Plain RuntimeError -- not transient by default
    assert not ex.is_transient_error(RuntimeError("bad symbol"))

    # Transport errors
    assert ex.is_transient_error(httpx.ConnectError("nope"))
    assert ex.is_transient_error(httpx.ReadError("nope"))
    assert ex.is_transient_error(httpx.PoolTimeout("nope"))


async def test_retry_recovers_via_wrapped_httpx_error() -> None:
    """An adapter that wraps a transient ``httpx`` error in a
    ``RuntimeError`` should still get retried: the classifier walks
    ``__cause__``.
    """
    ex = Kraken()

    calls = {"n": 0}

    async def stub(client: AsyncClient, quote_assets: set[str]) -> str:
        calls["n"] += 1
        if calls["n"] <= 2:
            try:
                raise httpx.PoolTimeout("pool full")
            except httpx.PoolTimeout as e:
                raise RuntimeError("wrapped") from e
        return "ok"

    ex.pairs = stub  # type: ignore[assignment]
    ex.RETRY_BASE_DELAY = 0.0
    ex.RETRY_MAX_DELAY = 0.0
    ex.RETRY_ATTEMPTS = 5
    result = await ex._retry("pairs", client=None, quote_assets=set())  # type: ignore[arg-type]
    assert result == "ok"
    assert calls["n"] == 3, f"expected 3 attempts (2 fails + 1 success), got {calls['n']}"


def test_is_transient_error_walks_cause_chain() -> None:
    """If an adapter wraps an ``httpx`` transport error in a
    ``RuntimeError`` (so its message reads "HTX klines request
    failed: <httpx>"), the classifier should still recognise it as
    transient by walking ``__cause__``."""
    ex = Kraken()

    # ``raise X from Y`` sets ``__cause__ = Y``.
    try:
        try:
            raise httpx.PoolTimeout("pool full")
        except httpx.PoolTimeout as e:
            raise RuntimeError("HTX klines request failed") from e
    except RuntimeError as wrapped:
        assert ex.is_transient_error(wrapped), (
            "wrapped PoolTimeout should be retried via __cause__"
        )

    # ``__context__`` (implicit chain from re-raise inside ``except``)
    # should also be walked.
    try:
        try:
            raise httpx.ConnectError("nope")
        except httpx.ConnectError:
            raise RuntimeError("HTX klines request failed")
    except RuntimeError as wrapped:
        assert ex.is_transient_error(wrapped), (
            "wrapped ConnectError should be retried via __context__"
        )

    # A 4xx HTTPStatusError wrapped in a RuntimeError is *not*
    # transient, regardless of chain -- bad request should not retry.
    resp_400 = httpx.Response(400, request=httpx.Request("GET", "https://x"))
    http_exc = httpx.HTTPStatusError(
        "400", request=resp_400.request, response=resp_400
    )
    try:
        try:
            raise http_exc
        except httpx.HTTPStatusError as e:
            raise RuntimeError("HTX klines bad request") from e
    except RuntimeError as wrapped:
        assert not ex.is_transient_error(wrapped), (
            "400 should not be retried even if wrapped"
        )
