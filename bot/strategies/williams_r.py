"""Williams %R oversold-recovery strategy.

Williams %R sits in [-100, 0] - inverted Stochastic. We enter on the
classic recovery: %R crosses up through ``oversold`` (default -80)
back into the body of its range. We exit when %R crosses down through
``overbought`` (default -20) - the bar has run out of upside room.

Pairs naturally with a trend filter. On its own this strategy will
fade strong trends and gets chopped up; behind a trend gate it earns
its keep on healthy pullbacks within an uptrend.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import williams_r
from .base import Signal, Strategy, StrategyContext


class WilliamsRStrategy(Strategy):
    name = "williams_r"

    def __init__(
        self,
        period: int = 14,
        oversold: float = -80.0,
        overbought: float = -20.0,
    ) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def min_history(self) -> int:
        return self.period + 2

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        if len(df) < self.min_history():
            return Signal.FLAT
        wr = williams_r(df, self.period).dropna()
        if len(wr) < 2:
            return Signal.FLAT
        prev, last = float(wr.iloc[-2]), float(wr.iloc[-1])
        if prev <= self.oversold < last:
            return Signal.LONG
        if prev >= self.overbought > last:
            return Signal.SHORT
        return Signal.FLAT
