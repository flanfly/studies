# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
days_holding = "30"
top_percentile = "0.9"
bottom_percentile = None
min_volume = "1_000_000"

# %%
days_holding_P = int(days_holding)
top_percentile_P = float(top_percentile)
bottom_percentile_P = (
    float(bottom_percentile) if bottom_percentile is not None else None
)
min_volume_P = int(min_volume)


# %%
import datetime as dt
import polars as pl
import scrapbook as sb

momoff = [3, 5, 7, 14, 21, 30, 40, 50, 60, 80, 100, 120, 150, 220]
momcols = [f"mom{n}d" for n in momoff]


if bottom_percentile is not None:
    assert (
        0 <= bottom_percentile_P < top_percentile_P <= 1
    ), "Percentiles must be between 0 and 1, and bottom must be less than top."

    spread_exprs = {
        f"spread{n}d": pl.col("fwdret")
        .filter(pl.col(f"rank{n}d") >= top_percentile_P)
        .mean()
        - pl.col("fwdret").filter(pl.col(f"rank{n}d") <= bottom_percentile_P).mean()
        for n in momoff
    }
else:
    assert 0 <= top_percentile_P <= 1, "Top percentile must be between 0 and 1."

    spread_exprs = {
        f"spread{n}d": pl.col("fwdret")
        .filter(pl.col(f"rank{n}d") >= top_percentile_P)
        .mean()
        for n in momoff
    }

df = (
    pl.read_parquet("polarity/latest-data/*parquet")
    .rename(
        {
            "asset": "symbol",
            "price": "close",
        }
    )
    .with_columns(ts=pl.col("ts").dt.cast_time_unit("us"))
    .sort(["symbol", "ts"])
    .with_columns(vol=pl.col('total_volume').rolling_mean(days_holding_P).over("symbol"))
    .with_columns(
        **{f"mom{n}d": pl.col("close").pct_change(n).over("symbol") for n in momoff}
    )
    .with_columns(fwdret=((pl.col("close").shift(-days_holding_P).over("symbol") / pl.col("close")) - 1))
    .with_columns(
        **{
            f"rank{n}d": (
                pl.when(
                    pl.col('vol') > min_volume_P
                ).then(
                    pl.col(f"mom{n}d").rank(method="ordinal").over("ts") / pl.col(f"mom{n}d").count().over("ts")
                ).otherwise(0)
            )
            for n in momoff
        }
    )
    .group_by("ts")
    .agg(**spread_exprs)
    .filter(pl.col("ts").dt.year() >= 2020)
    .drop_nulls()
    .unpivot(index="ts", variable_name="parameter", value_name="spread")
    .group_by("parameter")
    .agg(
        mean=(
            ((1 + pl.col("spread").mean()) ** (365 / days_holding_P) - 1) * 100
        ).round(2),
        vol=(pl.col("spread").std() * (365 / days_holding_P) ** 0.5 * 100).round(2),
    )
    .with_columns(
        ir=pl.col("mean") / pl.col("vol"),
    )
    .sort("ir")
)

for row in [f"spread{n}d" for n in momoff]:
    sb.glue(
        row,
        df.filter(pl.col("parameter") == row)["ir"][0],
    )

df

# %%
df.write_csv('res.csv')
