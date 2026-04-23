"""Donchian-channel breakout strategy (the classic Turtle trader entry).

Entry:  close breaks above the highest high of the last ``period`` bars.
Exit:   close breaks below the lowest low of the last ``exit_period`` bars.

The asymmetric windows (wider entry, narrower exit) are deliberate: they
give breakouts room to run while cutting losers fast. Strong-trend edge;
gets chopped up in sideways markets - stack with a trend or ADX filter.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import donchian
from .base import Signal, Strategy, StrategyContext


class DonchianStrategy(Strategy):
    name = "donchian"

    def __init__(self, period: int = 20, exit_period: int = 10) -> None:
        self.period = period
        self.exit_period = exit_period

    def min_history(self) -> int:
        return max(self.period, self.exit_period) + 2

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        if len(df) < self.min_history():
            return Signal.FLAT
        entry = donchian(df.iloc[:-1], self.period)  # exclude current bar to avoid self-reference
        exit_ = donchian(df.iloc[:-1], self.exit_period)
        last_close = float(df["close"].iloc[-1])
        if pd.notna(entry["upper"].iloc[-1]) and last_close > float(entry["upper"].iloc[-1]):
            return Signal.LONG
        if pd.notna(exit_["lower"].iloc[-1]) and last_close < float(exit_["lower"].iloc[-1]):
            return Signal.SHORT
        return Signal.FLAT
