from __future__ import annotations

import pandas as pd

from ..indicators import ema
from .base import Signal, Strategy, StrategyContext


class MaCrossoverStrategy(Strategy):
    name = "ma_crossover"

    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        if fast >= slow:
            raise ValueError("fast period must be smaller than slow period")
        self.fast = fast
        self.slow = slow

    def min_history(self) -> int:
        return self.slow + 2

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        if len(df) < self.min_history():
            return Signal.FLAT
        close = df["close"]
        fast_ma = ema(close, self.fast)
        slow_ma = ema(close, self.slow)
        prev_fast, prev_slow = fast_ma.iloc[-2], slow_ma.iloc[-2]
        last_fast, last_slow = fast_ma.iloc[-1], slow_ma.iloc[-1]
        if prev_fast <= prev_slow and last_fast > last_slow:
            return Signal.LONG
        if prev_fast >= prev_slow and last_fast < last_slow:
            return Signal.SHORT
        return Signal.FLAT
