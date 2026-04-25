"""Aroon trend-strength strategy.

Aroon Up and Aroon Down each measure how recently the highest high or
lowest low printed within a window. We enter long when:

  * Aroon Up >= ``up_threshold`` (recent high),
  * AND Aroon Down <= ``down_threshold`` (the recent low is no longer
    fresh - a reasonable proxy for "downtrend has stalled"),
  * AND the Up-Down oscillator is clearly positive.

Exits fire on the inverse - Aroon Down dominates again. Aroon is a
classic trend-identification tool that works well on hourly or daily
bars where rolling extremes are meaningful.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import aroon
from .base import Signal, Strategy, StrategyContext


class AroonStrategy(Strategy):
    name = "aroon"

    def __init__(
        self,
        period: int = 25,
        up_threshold: float = 70.0,
        down_threshold: float = 30.0,
        min_oscillator: float = 30.0,
    ) -> None:
        self.period = period
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.min_oscillator = min_oscillator

    def min_history(self) -> int:
        return self.period + 2

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        if len(df) < self.min_history():
            return Signal.FLAT
        a = aroon(df, self.period)
        if pd.isna(a["up"].iloc[-1]):
            return Signal.FLAT
        up = float(a["up"].iloc[-1])
        down = float(a["down"].iloc[-1])
        osc = float(a["oscillator"].iloc[-1])

        if up >= self.up_threshold and down <= self.down_threshold and osc >= self.min_oscillator:
            return Signal.LONG
        if down >= self.up_threshold and up <= self.down_threshold and osc <= -self.min_oscillator:
            return Signal.SHORT
        return Signal.FLAT
