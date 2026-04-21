from __future__ import annotations

import pandas as pd

from ..indicators import rsi
from .base import Signal, Strategy, StrategyContext


class RsiReversionStrategy(Strategy):
    name = "rsi_reversion"

    def __init__(
        self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0
    ) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def min_history(self) -> int:
        return self.period * 3

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        if len(df) < self.min_history():
            return Signal.FLAT
        r = rsi(df["close"], self.period)
        prev, last = r.iloc[-2], r.iloc[-1]
        if pd.isna(prev) or pd.isna(last):
            return Signal.FLAT
        if prev < self.oversold <= last:
            return Signal.LONG
        if prev > self.overbought >= last:
            return Signal.SHORT
        return Signal.FLAT
