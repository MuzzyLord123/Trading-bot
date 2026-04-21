from __future__ import annotations

import pandas as pd

from ..indicators import macd
from .base import Signal, Strategy, StrategyContext


class MacdStrategy(Strategy):
    name = "macd"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def min_history(self) -> int:
        return self.slow + self.signal_period + 5

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        if len(df) < self.min_history():
            return Signal.FLAT
        m = macd(df["close"], self.fast, self.slow, self.signal_period)
        prev_hist = m["hist"].iloc[-2]
        last_hist = m["hist"].iloc[-1]
        if pd.isna(prev_hist) or pd.isna(last_hist):
            return Signal.FLAT
        if prev_hist <= 0 < last_hist:
            return Signal.LONG
        if prev_hist >= 0 > last_hist:
            return Signal.SHORT
        return Signal.FLAT
