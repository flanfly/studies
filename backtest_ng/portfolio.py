from abc import ABC, abstractmethod
import polars as pl
import datetime as dt

from typing import Dict

from . import PortfolioModel, Signal, Portfolio, Universe, Target


class EqualWeight(PortfolioModel):
    def __call__(
        self,
        history: pl.DataFrame,
        u: Universe,
        signals: list[Signal],
        portfolio: Portfolio,
    ) -> list[Target]:
        if len(signals) == 0:
            return []

        w = 1.0 / len(signals)
        by_symbol: Dict[str, float] = {}
        for s in signals:
            by_symbol.setdefault(s.symbol, 0.0)
            by_symbol[s.symbol] += w * (1 if s.bullish else -1)

        return [Target(symbol=k, weight=v) for k, v in by_symbol.items()]


class VolatilityWeighted(PortfolioModel):
    """Inverse-volatility (risk-parity style) portfolio construction.

    Weights signals in inverse proportion to their realised target
    volatility, with a *single* shared normalisation across both sides
    so that the gross exposure of the basket is ``leverage`` and the
    long:short ratio is determined by the number of signals on each
    side.  This mirrors the manual construction in
    ``sector-rotation-prod-v1.py``::

        inv_vol[s]  = 1 / sqrt(var[s])        (or 0 if var is missing/<=0)
        weight[s]   = (inv_vol[s] / tiv) * leverage      (signed by side)

    where ``tiv = sum(inv_vol[s] for s in selected)`` is taken over
    the *combined* long + short basket, not per side.  This is the
    key reason the live production strategy is long-biased: with
    ``max_long=4, max_short=1, leverage=2`` the typical gross is
    ~2.0× equity, of which ~1.6× is long and ~0.4× is short.

    The volatility estimate is read from ``u.df()[volatility_col]`` at
    the most recent bar, per symbol.  Symbols whose volatility is
    missing or non-positive get zero weight (and are silently dropped
    rather than penalising the rest of the basket).

    Parameters
    ----------
    volatility_col:
        Column in ``u.df()`` carrying a *per-bar variance* estimate
        for each symbol.  Defaults to ``"var"`` to match the
        Yang-Zhang variance computed in ``sector-rotation-prod-v1.py``.
        Any other variance-style column works the same way (rolling
        close-to-close variance, Parkinson, Garman-Klass, …).
    leverage:
        Total gross exposure of the basket.  Defaults to ``1.0``.
        Note that the gross, not each side, sums to ``leverage`` —
        a 4-long / 1-short basket with ``leverage=2.0`` produces
        ~1.6× long and ~0.4× short, not 2.0×/2.0×.
    """

    def __init__(self, volatility_col: str = "var", leverage: float = 1.0):
        self.volatility_col = volatility_col
        self.leverage = leverage

    def __call__(
        self,
        history: pl.DataFrame,
        u: Universe,
        signals: list[Signal],
        portfolio: Portfolio,
    ) -> list[Target]:
        if len(signals) == 0:
            return []

        # Pull the most recent row of the universe and index variance
        # by symbol.  Per-bar variance → per-bar stdev → inverse vol.
        df = u.df()
        tscol = u.timestamp_col()
        symcol = u.symbol_col()
        latest = df.filter(pl.col(tscol) == df[tscol].max())

        if self.volatility_col in latest.columns:
            var_rows = latest.select(symcol, self.volatility_col).iter_rows(named=True)
            # Polars surfaces a null ``var`` as Python ``None`` here;
            # normalise to 0.0 so the ``<= 0`` check below correctly
            # drops a NaN-vol symbol from the basket.
            var_dict: Dict[str, float] = {
                r[symcol]: (0.0 if r[self.volatility_col] is None else float(r[self.volatility_col]))
                for r in var_rows
            }
        else:
            # The volatility column hasn't been computed (yet) — fall
            # back to equal weight so the model still emits targets.
            var_dict = {s.symbol: 0.0 for s in signals}

        # Compute inverse vols over the *combined* long + short basket
        # and normalise by a single ``tiv``.  The v1 reference
        # intentionally does this so a small short basket (e.g. 1 name)
        # doesn't get an outsized weight by being normalised alone
        # — instead the long:short ratio is driven by the count and
        # by relative vol.
        #
        # We key by the signal itself, not by symbol, so a name that
        # appears on both sides keeps separate long and short entries
        # — the v1 reference iterates over ``long_bkt`` and ``short_bkt``
        # independently for the same reason.
        def _inv_vol(sym: str) -> float:
            var = var_dict.get(sym, 0.0)
            if var is None or var <= 0.0:
                return 0.0
            return 1.0 / (var ** 0.5)

        inv_vols = [(s, _inv_vol(s.symbol)) for s in signals]
        tiv = sum(v for _, v in inv_vols)
        n = len(signals)

        targets: list[Target] = []
        for s, iv in inv_vols:
            if tiv > 0.0:
                w = (iv / tiv) * self.leverage
            else:
                # All symbols have missing/non-positive vol — fall
                # back to per-name equal weight so the basket is
                # still well-defined.
                w = self.leverage / n
            sign = 1.0 if s.bullish else -1.0
            targets.append(Target(symbol=s.symbol, weight=sign * w))

        return targets


# class VolumeWeighted(PortfolioModel):
#    def __init__(
#        self,
#        timestamp_col="ts",
#        symbol_col="symbol",
#        price_col="close",
#        volume_col="volume",
#    ):
#        self.ts_col = timestamp_col
#        self.symbol_col = symbol_col
#        self.price_col = price_col
#        self.volume_col = volume_col
#
#    def __call__(
#        self,
#        df: pl.DataFrame,
#        signals: list[Signal],
#        folio: list[Position],
#        equity: float,
#    ) -> list[Order]:
#        today = df[self.ts_col].max()
#        prices = dict(
#            df.filter(pl.col(self.ts_col) == today)
#            .select([self.symbol_col, self.price_col])
#            .iter_rows()
#        )
#        vols = dict(
#            df.filter(pl.col(self.ts_col) == today)
#            .select([self.symbol_col, self.volume_col])
#            .iter_rows()
#        )
#
#        if not signals or equity <= 0:
#            return [Order(pos.symbol, -pos.shares) for pos in folio]
#
#        total_vol = sum(vols.get(signal.symbol, 0) for signal in signals)
#        if total_vol <= 0:
#            return [Order(pos.symbol, -pos.shares) for pos in folio]
#
#        target_shares = {}
#        for signal in signals:
#            price = prices.get(signal.symbol)
#            if price is not None and price > 0:
#                target_shares[signal.symbol] = (
#                    (vols.get(signal.symbol, 0) / total_vol)
#                    * equity
#                    / price
#                    * (1 if signal.bullish else -1)
#                )
#
#        orders = []
#        current_positions = {pos.symbol: pos.shares for pos in folio}
#        all_symbols = set(current_positions.keys()) | set(target_shares.keys())
#
#        for sym in all_symbols:
#            current = current_positions.get(sym, 0.0)
#            target = target_shares.get(sym, 0.0)
#            delta = target - current
#            if abs(delta) > EPSILON:
#                orders.append(Order(sym, delta))
#
#        return orders
#
#
# class SimpleLeverage(PortfolioModel):
#    def __init__(self, leverage: float, inner: PortfolioModel = EqualWeight()):
#        self.inner = inner
#        self.leverage = leverage
#
#    def __call__(
#        self,
#        df: pl.DataFrame,
#        signals: list[Signal],
#        folio: list[Position],
#        equity: float,
#    ) -> list[Order]:
#        return self.inner(df, signals, folio, equity * self.leverage)
#
#
# class TopN(PortfolioModel):
#    def __init__(self, max_long=5, max_short=5, portfolio_model=None):
#        self.max_long = max_long
#        self.max_short = max_short
#        self.portfolio_model = portfolio_model or EqualWeight()
#
#    def __call__(
#        self,
#        df: pl.DataFrame,
#        signals: list[Signal],
#        folio: list[Position],
#        equity: float,
#    ) -> list[Order]:
#        if not signals:
#            return []
#
#        # Sort signals by confidence for long and short separately
#        long_signals = sorted(
#            [s for s in signals if s.bullish], key=lambda s: s.confidence, reverse=True
#        )
#        short_signals = sorted(
#            [s for s in signals if not s.bullish],
#            key=lambda s: s.confidence,
#            reverse=True,
#        )
#
#        top_long = long_signals[: self.max_long]
#        top_short = short_signals[: self.max_short]
#
#        selected_signals = top_long + top_short
#        return self.portfolio_model(df, selected_signals, folio, equity)
