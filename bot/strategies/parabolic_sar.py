"""Parabolic SAR flip strategy.

The Parabolic SAR returns a directional state at every bar (+1 up,
-1 down). We trade on direction CHANGES: when the SAR flips up we go
long; when it flips down we exit / signal short. SAR also makes for
a natural trailing stop, but the engine's risk manager already owns
that decision so we only emit entry/exit signals here.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import parabolic_sar
from .base import Signal, Strategy, StrategyContext


class ParabolicSarStrategy(Strategy):
    name = "parabolic_sar"

    def __init__(self, step: float = 0.02, max_step: float = 0.2) -> None:
        self.step = step
        self.max_step = max_step

    def min_history(self) -> int:
        # SAR seeds from bar 2 onwards but needs a few bars of context
        # before its acceleration factor is meaningful.
        return 30

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        if len(df) < self.min_history():
            return Signal.FLAT
        sar = parabolic_sar(df, self.step, self.max_step)
        last_dir = int(sar["direction"].iloc[-1])
        prev_dir = int(sar["direction"].iloc[-2])
        if prev_dir != 1 and last_dir == 1:
            return Signal.LONG
        if prev_dir != -1 and last_dir == -1:
            return Signal.SHORT
        return Signal.FLAT
