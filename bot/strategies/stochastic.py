"""Stochastic oscillator crossover strategy.

Entry:  %K crosses up through %D while both are below ``oversold``.
Exit:   %K crosses down through %D while both are above ``overbought``.

This is a classic momentum-reversal strategy. Works best in range-bound
markets; pairs well with a trend filter so it only fires while the bigger
picture is constructive.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import stochastic
from .base import Signal, Strategy, StrategyContext


class StochasticStrategy(Strategy):
    name = "stochastic"

    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0,
    ) -> None:
        self.k_period = k_period
        self.d_period = d_period
        self.smooth = smooth
        self.oversold = oversold
        self.overbought = overbought

    def min_history(self) -> int:
        return self.k_period + self.d_period + self.smooth + 2

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        if len(df) < self.min_history():
            return Signal.FLAT
        st = stochastic(df, self.k_period, self.d_period, self.smooth).dropna()
        if len(st) < 2:
            return Signal.FLAT
        prev, last = st.iloc[-2], st.iloc[-1]
        if last["k"] > last["d"] and prev["k"] <= prev["d"] and last["k"] < self.overbought:
            if prev["k"] <= self.oversold + 5:  # cross happened near the oversold zone
                return Signal.LONG
        if last["k"] < last["d"] and prev["k"] >= prev["d"] and last["k"] > self.oversold:
            if prev["k"] >= self.overbought - 5:
                return Signal.SHORT
        return Signal.FLAT
