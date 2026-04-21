from __future__ import annotations

import pandas as pd

from ..indicators import bollinger
from .base import Signal, Strategy, StrategyContext


class BollingerStrategy(Strategy):
    name = "bollinger"

    def __init__(self, period: int = 20, std: float = 2.0) -> None:
        self.period = period
        self.std = std

    def min_history(self) -> int:
        return self.period + 5

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        if len(df) < self.min_history():
            return Signal.FLAT
        bands = bollinger(df["close"], self.period, self.std)
        prev_close = df["close"].iloc[-2]
        last_close = df["close"].iloc[-1]
        prev_lower, last_lower = bands["lower"].iloc[-2], bands["lower"].iloc[-1]
        prev_upper, last_upper = bands["upper"].iloc[-2], bands["upper"].iloc[-1]
        if pd.isna(prev_lower) or pd.isna(last_lower):
            return Signal.FLAT
        if prev_close < prev_lower and last_close >= last_lower:
            return Signal.LONG
        if prev_close > prev_upper and last_close <= last_upper:
            return Signal.SHORT
        return Signal.FLAT
