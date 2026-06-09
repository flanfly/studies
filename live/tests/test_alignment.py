"""Tests for the daily-candle alignment logic.

``Exchange.closed_klines_end(now)`` returns the ``end_time`` argument
to pass to ``klines()`` / ``klines_paged()`` so the in-progress
candle is excluded. The contract is half-open ``[start, end)`` on
``end``: the returned value is the ``open_ts`` of the *current*
in-progress bar at ``now``, so the half-open range drops that bar
and keeps every fully-closed one before it.

Most exchanges align to midnight UTC. HTX aligns to 16:00 UTC.
"""

from __future__ import annotations

import datetime as dt

import pytest

from live import HTX, Kraken


def test_kraken_alignment_midnight() -> None:
    """Kraken is a midnight-UTC exchange."""
    ex = Kraken()
    # 05:00 UTC: in-progress bar started at today's 00:00, so
    # ``end`` is today's 00:00 (the in-progress bar's open_ts).
    now = dt.datetime(2026, 6, 9, 5, 0, 0, tzinfo=dt.timezone.utc)
    assert ex.closed_klines_end(now) == dt.datetime(
        2026, 6, 9, 0, 0, 0, tzinfo=dt.timezone.utc
    )
    # 00:00:00.000001 UTC: the new bar has just started; we exclude
    # it (it's in-progress), so ``end`` is still today's 00:00.
    now = dt.datetime(2026, 6, 9, 0, 0, 0, 1, tzinfo=dt.timezone.utc)
    assert ex.closed_klines_end(now) == dt.datetime(
        2026, 6, 9, 0, 0, 0, tzinfo=dt.timezone.utc
    )
    # Exactly 00:00:00 UTC: the new bar has just begun; we still
    # exclude it, so ``end`` is today's 00:00.
    now = dt.datetime(2026, 6, 9, 0, 0, 0, 0, tzinfo=dt.timezone.utc)
    assert ex.closed_klines_end(now) == dt.datetime(
        2026, 6, 9, 0, 0, 0, tzinfo=dt.timezone.utc
    )
    # 23:59:59.999999 UTC: in-progress bar still started at today's
    # 00:00.
    now = dt.datetime(2026, 6, 9, 23, 59, 59, 999999, tzinfo=dt.timezone.utc)
    assert ex.closed_klines_end(now) == dt.datetime(
        2026, 6, 9, 0, 0, 0, tzinfo=dt.timezone.utc
    )


def test_kraken_naive_datetime_treated_as_utc() -> None:
    """A naive ``now`` is treated as UTC (the result has UTC tzinfo)."""
    ex = Kraken()
    now = dt.datetime(2026, 6, 9, 5, 0, 0)
    out = ex.closed_klines_end(now)
    assert out.tzinfo == dt.timezone.utc
    assert out == dt.datetime(2026, 6, 9, 0, 0, 0, tzinfo=dt.timezone.utc)


def test_htx_alignment_16_utc() -> None:
    """HTX daily candles start at 16:00 UTC, not midnight."""
    ex = HTX(access_key="x", secret_key="y")
    # 05:00 UTC: in-progress bar started at 2026-06-08 16:00 UTC.
    now = dt.datetime(2026, 6, 9, 5, 0, 0, tzinfo=dt.timezone.utc)
    assert ex.closed_klines_end(now) == dt.datetime(
        2026, 6, 8, 16, 0, 0, tzinfo=dt.timezone.utc
    )
    # 15:59 UTC: still inside the bar that started at 2026-06-08
    # 16:00. ``end`` is that boundary.
    now = dt.datetime(2026, 6, 9, 15, 59, 0, tzinfo=dt.timezone.utc)
    assert ex.closed_klines_end(now) == dt.datetime(
        2026, 6, 8, 16, 0, 0, tzinfo=dt.timezone.utc
    )
    # 20:00 UTC: in-progress bar started at 2026-06-09 16:00.
    now = dt.datetime(2026, 6, 9, 20, 0, 0, tzinfo=dt.timezone.utc)
    assert ex.closed_klines_end(now) == dt.datetime(
        2026, 6, 9, 16, 0, 0, tzinfo=dt.timezone.utc
    )
    # Exactly 16:00 UTC: the new bar has just begun; we still
    # exclude it.
    now = dt.datetime(2026, 6, 9, 16, 0, 0, 0, tzinfo=dt.timezone.utc)
    assert ex.closed_klines_end(now) == dt.datetime(
        2026, 6, 9, 16, 0, 0, tzinfo=dt.timezone.utc
    )


@pytest.mark.parametrize(
    "now,expected",
    [
        # Exactly at the 16:00 boundary on HTX: in-progress bar is
        # the one starting now; we exclude it.
        (
            dt.datetime(2026, 6, 9, 16, 0, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 6, 9, 16, 0, 0, tzinfo=dt.timezone.utc),
        ),
        # 1us before the next boundary on HTX: in-progress bar still
        # started at 2026-06-08 16:00.
        (
            dt.datetime(2026, 6, 9, 15, 59, 59, 999999, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 6, 8, 16, 0, 0, tzinfo=dt.timezone.utc),
        ),
    ],
)
def test_htx_boundary_conditions(now, expected) -> None:
    ex = HTX(access_key="x", secret_key="y")
    assert ex.closed_klines_end(now) == expected
