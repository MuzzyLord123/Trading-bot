from __future__ import annotations

from typing import Any

import pandas as pd

from .base import Signal, Strategy, StrategyContext


class EnsembleStrategy(Strategy):
    """Majority-vote combiner over any number of sub-strategies."""

    name = "ensemble"

    def __init__(
        self,
        members: list[Strategy] | None = None,
        min_agreement: int = 2,
        **_: Any,
    ) -> None:
        self.members: list[Strategy] = members or []
        self.min_agreement = min_agreement

    def min_history(self) -> int:
        if not self.members:
            return 50
        return max(m.min_history() for m in self.members)

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        if not self.members or len(df) < self.min_history():
            return Signal.FLAT
        votes = [int(m.generate(df, ctx)) for m in self.members]
        longs = sum(1 for v in votes if v > 0)
        shorts = sum(1 for v in votes if v < 0)
        if longs >= self.min_agreement and longs > shorts:
            return Signal.LONG
        if shorts >= self.min_agreement and shorts > longs:
            return Signal.SHORT
        return Signal.FLAT
