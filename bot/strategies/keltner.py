"""Keltner-channel breakout strategy.

Similar shape to Bollinger bands but uses ATR instead of standard deviation
for the channel width. A close above the upper band signals that trend
momentum has overpowered recent volatility - a classic continuation entry.
A close below the lower band triggers an exit.

Narrower and more trend-respectful than Bollinger in volatile markets
because true range is less spiky than price stdev around a mean.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import keltner
from .base import Signal, Strategy, StrategyContext


class KeltnerStrategy(Strategy):
    name = "keltner"

    def __init__(
        self,
        period: int = 20,
        multiplier: float = 2.0,
        atr_period: int = 10,
    ) -> None:
        self.period = period
        self.multiplier = multiplier
        self.atr_period = atr_period

    def min_history(self) -> int:
        return max(self.period, self.atr_period) + 5

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        if len(df) < self.min_history():
            return Signal.FLAT
        kc = keltner(df, self.period, self.multiplier, self.atr_period)
        upper = kc["upper"].iloc[-1]
        lower = kc["lower"].iloc[-1]
        last = float(df["close"].iloc[-1])
        if pd.notna(upper) and last > float(upper):
            return Signal.LONG
        if pd.notna(lower) and last < float(lower):
            return Signal.SHORT
        return Signal.FLAT
