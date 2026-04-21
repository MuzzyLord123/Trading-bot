"""Regime-aware wrapper that filters entry signals in bad conditions."""
from __future__ import annotations

import pandas as pd

from ..indicators import adx, atr, ema
from .base import Signal, Strategy, StrategyContext


class FilteredStrategy(Strategy):
    """Wrap an inner strategy and drop entry signals that fail regime checks.

    Exits (SHORT signals on a long position) are always passed through so we
    get out of losers quickly. Only entries are filtered – that is the side
    we want to be selective about.
    """

    name = "filtered"

    def __init__(
        self,
        inner: Strategy,
        trend_ema: int = 200,
        min_adx: float = 20.0,
        adx_period: int = 14,
        min_atr_pct: float = 0.001,
        atr_period: int = 14,
        require_uptrend: bool = True,
    ) -> None:
        self.inner = inner
        self.trend_ema = trend_ema
        self.min_adx = min_adx
        self.adx_period = adx_period
        self.min_atr_pct = min_atr_pct
        self.atr_period = atr_period
        self.require_uptrend = require_uptrend

    def min_history(self) -> int:
        base = self.inner.min_history()
        return max(base, self.trend_ema + 5, self.adx_period * 3, self.atr_period * 3)

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        sig = self.inner.generate(df, ctx)
        if sig <= 0:  # Always let exits/flat through – be selective on entries only.
            return sig
        if len(df) < self.min_history():
            return Signal.FLAT

        last_close = float(df["close"].iloc[-1])

        if self.require_uptrend:
            trend = ema(df["close"], self.trend_ema).iloc[-1]
            if pd.isna(trend) or last_close < trend:
                return Signal.FLAT

        if self.min_adx > 0:
            trend_strength = adx(df, self.adx_period).iloc[-1]
            if pd.isna(trend_strength) or trend_strength < self.min_adx:
                return Signal.FLAT

        if self.min_atr_pct > 0:
            atr_val = atr(df, self.atr_period).iloc[-1]
            if pd.isna(atr_val) or atr_val / last_close < self.min_atr_pct:
                return Signal.FLAT

        return Signal.LONG
