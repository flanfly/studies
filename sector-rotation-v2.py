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
max_long = "4"
max_short = "1"
period = "21d"
stop_long = "0.5"
stop_short = "0.05"
leverage = "2"

# %%
import backtest_ng as bt
import polars as pl

from typing import Dict

num_short_positions_P = int(max_short)
num_long_positions_P = int(max_long)
leverage_P = float(leverage)

# Yang-Zhang variance estimator constants — same as
# sector-rotation-prod-v1.py so the ``var`` column in the universe
# matches what the vol-weighted portfolio model expects by default.
yz_k, yz_win = 0.34, 25


sector_etfs = [
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]
start_year = 2015

class Alpha(bt.AlphaModel):
    def _build(self, history: pl.DataFrame, u: bt.Universe) -> pl.DataFrame:
        """Compute the per-symbol universe with Yang-Zhang ``var``,
        momentum, score, and rank columns.  Shared between
        ``__call__`` (signal generation) and ``augment`` (publishing
        the universe to the portfolio model) so the vol-weighted
        model sees the same ``var`` the alpha uses."""
        tscol = u.timestamp_col()
        symcol = u.symbol_col()
        # Drop any date on which the close is missing for *some* symbol
        # — typically the in-progress trading day, which YFinance has
        # only populated for a subset of tickers.  Without this, the
        # rolling Yang-Zhang window has null ``close`` on the last
        # bar and ``var`` is null, so the vol-weighted portfolio model
        # silently falls back to equal weight.  Using the prior
        # complete bar gives a meaningful inverse-vol signal even
        # mid-session.
        partial_dates = (
            u.df()
            .filter(pl.col("close").is_null())[tscol]
            .unique()
        )
        df_src = u.df().filter(~pl.col(tscol).is_in(partial_dates))
        return (
            df_src
            .filter(pl.col(symcol).is_in(sector_etfs))
            .with_columns(
                # Yang-Zhang variance components.
                o=pl.col("open").log() - pl.col("close").shift(1).over(symcol).log(),
                u=pl.col("high").log() - pl.col("open").log(),
                d=pl.col("low").log() - pl.col("open").log(),
                c=pl.col("close").log() - pl.col("open").log(),
            )
            .with_columns(
                rs=pl.col("u") * (pl.col("u") - pl.col("c"))
                + pl.col("d") * (pl.col("d") - pl.col("c"))
            )
            .with_columns(
                var=(
                    pl.col("o").rolling_var(yz_win)
                    + yz_k * pl.col("c").rolling_var(yz_win)
                    + ((1 - yz_k) * pl.col("rs").rolling_mean(yz_win))
                ).over(symcol)
            )
            .with_columns(
                mom1m=pl.col("close").pct_change(21).over(symcol),
                mom6m=pl.col("close").pct_change(21*6).over(symcol),
                mom12m=pl.col("close").pct_change(21*12).over(symcol),
            )
            .with_columns(
                score=pl.col("mom12m") - pl.col("mom1m")
            )
            .with_columns(
                lrank=pl.when((pl.col("mom12m") > 0) | (pl.col("mom6m") > 0))
                    .then((pl.col("score").rank(descending=True) / pl.len()).over(tscol))
                    .otherwise(None),
                srank=pl.when((pl.col("mom12m") < 0) & (pl.col("mom6m") < 0))
                    .then((pl.col("score").rank(descending=False) / pl.len()).over(tscol))
                    .otherwise(None),
            )
        )

    def __call__(self, history: pl.DataFrame, u: bt.Universe) -> list[bt.Signal]:
        symcol = u.symbol_col()
        tscol = u.timestamp_col()
        df = self._build(history, u)
        dfnow = df.filter(pl.col(tscol) == df[tscol].max())
        # Drop gate-failing rows (null ``lrank``/``srank``) *before* sorting
        # and slicing.  In Polars, nulls sort last by default, so without
        # this filter the bottom of the list would be filled with symbols
        # that didn't pass the gate whenever fewer than ``num_long_positions_P``
        # (or ``num_short_positions_P``) symbols qualified.  v1's reference
        # does the same: ``df_now.filter(pl.col("long_rank").is_null().not_())``.
        l = dfnow.filter(pl.col("lrank").is_not_null()).sort("lrank")
        s = dfnow.filter(pl.col("srank").is_not_null()).sort("srank")

        ls = [
            bt.Signal(r[symcol], True, 1.0)
            for r in l.iter_rows(named=True)
        ]
        ss = [
            bt.Signal(r[symcol], False, -1.0)
            for r in s.iter_rows(named=True)
        ]

        return ls[:num_long_positions_P] + ss[:num_short_positions_P]

    def augment(self, history: pl.DataFrame, u: bt.Universe) -> pl.DataFrame:
        """Publish the universe with ``var`` (and friends) so the
        ``VolatilityWeighted`` portfolio model can read the per-symbol
        volatility on the latest bar.  Without this hook the portfolio
        model only ever sees the raw YFinance columns and falls back
        to equal weight."""
        return self._build(history, u)

test = bt.Backtest(
    universe=bt.YFinance(
        tickers=sector_etfs + ['SPY'],
        start=start_year,
    ),
    alpha=Alpha(),
    portfolio=bt.VolatilityWeighted(volatility_col="var", leverage=leverage_P),
    risk=bt.MaxRisk(.5),
    #risk=bt.Fixed(
    #    long_trailing_pct=stop_long_P,
    #    short_trailing_pct=stop_short_P,
    #),
    #execution=bt.Schwab(
    #    base_rate_pct=10,
    #    maintainance_pct=30,
    #),
    benchmark='SPY',
    period=21,
)

test.run()
test.report(plot=True)

# %%
test.live(equity=1_000)
