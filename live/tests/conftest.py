"""Shared fixtures for the ``live`` exchange adapter tests.

The fixtures load ``.env`` from the project root once per session, and
provide:

  * ``client``           – an ``httpx.AsyncClient`` with a generous timeout.
  * ``pairs_window``     – midnight-UTC ``(start, end)`` boundaries for
    the 14-day ``pairs()`` + ``klines()`` tests.
  * ``paged_window``     – 4-year range for the ``klines_paged()`` tests
    (the oldest end is shared with ``pairs_window`` so the requests
    are independent).
  * ``htx``/``kucoin``/``binance`` – concrete adapter instances; the
    test is skipped if the corresponding ``.env`` credentials are
    missing. ``kraken``, ``hyperliquid``, and ``asterdex`` are
    always available because their public endpoints are
    unauthenticated.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from httpx import AsyncClient

from live import AsterDex, Binance, HTX, Hyperliquid, Kraken, KuCoin


def _find_env(start: Path) -> Path | None:
    """Walk up the directory tree from ``start`` until a ``.env`` file
    is found, or return ``None`` if we hit the filesystem root.
    """
    for d in (start, *start.parents):
        candidate = d / ".env"
        if candidate.is_file():
            return candidate
    return None


# Tests live at ``live/tests/conftest.py``; the project root sits a few
# levels up, and ``.env`` may live anywhere along the way. Walk up the
# tree so the same conftest works whether tests are at the project
# root, in ``tests/``, or in ``live/tests/`` as they are now.
ENV_PATH = _find_env(Path(__file__).resolve().parent)


@pytest.fixture(scope="session", autouse=True)
def _load_env() -> None:
    """Load ``.env`` from the project root once per test session."""
    if ENV_PATH is not None:
        load_dotenv(ENV_PATH)


@pytest.fixture
def client() -> AsyncClient:
    """An ``httpx.AsyncClient`` whose default timeout is generous enough
    for slow public exchange endpoints.
    """
    return AsyncClient(timeout=30.0)


@pytest.fixture
def pairs_window() -> tuple[dt.datetime, dt.datetime]:
    """Midnight-UTC ``(start, end)`` for the standard 14-day range:
    ``start = end - 14 days``, with both snapped to the previous
    midnight so the candle count is predictable.
    """
    end = dt.datetime.now(dt.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = end - dt.timedelta(days=14)
    return start, end


@pytest.fixture
def paged_window() -> tuple[dt.datetime, dt.datetime]:
    """4-year range ending at the same midnight as ``pairs_window`` so
    the ``klines()`` and ``klines_paged()`` requests don't overlap.
    """
    end = dt.datetime.now(dt.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = end - dt.timedelta(days=4 * 365)
    return start, end


# ----------------------------------------------------------------------
# exchange fixtures
# ----------------------------------------------------------------------


def _require_env(*keys: str) -> dict[str, str]:
    """Read ``keys`` from the environment, skipping the test if any are
    missing.
    """
    values: dict[str, str] = {}
    for k in keys:
        v = os.environ.get(k)
        if not v:
            where = ENV_PATH or "<no .env found above tests/>"
            pytest.skip(f"{k} not set in {where}")
        values[k] = v
    return values


@pytest.fixture
def htx() -> HTX:
    creds = _require_env("HTX_ACCESS_KEY", "HTX_SECRET_KEY")
    return HTX(access_key=creds["HTX_ACCESS_KEY"], secret_key=creds["HTX_SECRET_KEY"])


@pytest.fixture
def kucoin() -> KuCoin:
    creds = _require_env("KUCOIN_API_KEY", "KUCOIN_API_SECRET", "KUCOIN_API_PASSWORD")
    return KuCoin(
        api_key=creds["KUCOIN_API_KEY"],
        api_secret=creds["KUCOIN_API_SECRET"],
        api_password=creds["KUCOIN_API_PASSWORD"],
    )


@pytest.fixture
def kraken() -> Kraken:
    # Public endpoints only; no credentials needed.
    return Kraken()


@pytest.fixture
def binance() -> Binance:
    creds = _require_env("BINANCE_API_KEY", "BINANCE_API_SECRET")
    return Binance(api_key=creds["BINANCE_API_KEY"], api_secret=creds["BINANCE_API_SECRET"])


@pytest.fixture
def hyperliquid() -> Hyperliquid:
    # Public endpoints only; no credentials needed.
    return Hyperliquid()


@pytest.fixture
def asterdex() -> AsterDex:
    # Public endpoints only; no credentials needed.
    return AsterDex()
