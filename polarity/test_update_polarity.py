import importlib

import polars as pl

update_polarity = importlib.import_module("polarity.update-polarity")
merge_metrics = update_polarity.merge_metrics


def test_merge_same_timestamps():
    """Two metrics for the same coin with identical timestamp ranges."""
    df1 = pl.DataFrame(
        {
            "timestamp": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "asset": ["btc", "btc", "btc"],
            "m1": [1.0, 2.0, 3.0],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())

    df2 = pl.DataFrame(
        {
            "timestamp": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "asset": ["btc", "btc", "btc"],
            "m2": [4.0, 5.0, 6.0],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())

    merged = merge_metrics([df1, df2])
    expected = (
        pl.DataFrame(
            {
                "timestamp": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "asset": ["btc", "btc", "btc"],
                "m1": [1.0, 2.0, 3.0],
                "m2": [4.0, 5.0, 6.0],
            }
        )
        .with_columns(pl.col("timestamp").str.to_datetime())
        .sort("timestamp")
    )
    assert merged.equals(expected)


def test_merge_overlapping_timestamps():
    """Metrics with partially overlapping timestamp ranges."""
    df1 = pl.DataFrame(
        {
            "timestamp": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "asset": ["btc", "btc", "btc"],
            "m1": [1.0, 2.0, 3.0],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())

    df2 = pl.DataFrame(
        {
            "timestamp": ["2023-01-02", "2023-01-03", "2023-01-04"],
            "asset": ["btc", "btc", "btc"],
            "m2": [10.0, 20.0, 30.0],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())

    merged = merge_metrics([df1, df2])
    expected = (
        pl.DataFrame(
            {
                "timestamp": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                ],
                "asset": ["btc", "btc", "btc", "btc"],
                "m1": [1.0, 2.0, 3.0, None],
                "m2": [None, 10.0, 20.0, 30.0],
            }
        )
        .with_columns(pl.col("timestamp").str.to_datetime())
        .sort("timestamp")
    )
    assert merged.equals(expected)


def test_merge_three_metrics():
    """Three metrics with staggered timestamp ranges."""
    df1 = pl.DataFrame(
        {
            "timestamp": ["2023-01-01", "2023-01-02"],
            "asset": ["btc", "btc"],
            "m1": [1.0, 2.0],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())

    df2 = pl.DataFrame(
        {
            "timestamp": ["2023-01-02", "2023-01-03"],
            "asset": ["btc", "btc"],
            "m2": [10.0, 20.0],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())

    df3 = pl.DataFrame(
        {
            "timestamp": ["2023-01-03", "2023-01-04"],
            "asset": ["btc", "btc"],
            "m3": [100.0, 200.0],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())

    merged = merge_metrics([df1, df2, df3])
    assert merged.height == 4
    assert set(merged.columns) == {"timestamp", "asset", "m1", "m2", "m3"}
    # No duplicate timestamps
    dupes = merged.group_by("timestamp").len().filter(pl.col("len") > 1)
    assert dupes.height == 0
