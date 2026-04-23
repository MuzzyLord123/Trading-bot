"""CCI (Commodity Channel Index) strategy.

Entry:  CCI crosses up through ``entry_level`` (default -100, the classic
        oversold-reversal trigger).
Exit:   CCI crosses down through ``exit_level`` (default +100).

The +/-100 thresholds are the original Lambert settings. More extreme
levels (+/-200) filter harder but trade less. Best in range-bound
markets; a trend filter keeps it from fading strong moves.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import cci
from .base import Signal, Strategy, StrategyContext


class CciStrategy(Strategy):
    name = "cci"

    def __init__(
        self,
        period: int = 20,
        entry_level: float = -100.0,
        exit_level: float = 100.0,
    ) -> None:
        self.period = period
        self.entry_level = entry_level
        self.exit_level = exit_level

    def min_history(self) -> int:
        return self.period + 2

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        if len(df) < self.min_history():
            return Signal.FLAT
        c = cci(df, self.period).dropna()
        if len(c) < 2:
            return Signal.FLAT
        prev, last = float(c.iloc[-2]), float(c.iloc[-1])
        if prev <= self.entry_level < last:
            return Signal.LONG
        if prev >= self.exit_level > last:
            return Signal.SHORT
        return Signal.FLAT
