from __future__ import annotations

from typing import Any

import pandas as pd

from .base import Signal, Strategy, StrategyContext


class EnsembleStrategy(Strategy):
    """Weighted-vote combiner with a lookback window.

    Each sub-strategy's vote is considered "live" for ``window_bars`` bars
    after it fires. This matters because most technical signals are events
    (a crossover prints on one bar and is gone). Without a window, two
    strategies almost never agree on the same bar and the ensemble never
    trades.

    An entry fires LONG when:
      - the number of members with a live LONG vote is >= ``min_agreement``
      - AND the weighted score exceeds ``min_score``

    Exits fire symmetrically with SHORT.
    """

    name = "ensemble"

    def __init__(
        self,
        members: list[Strategy] | None = None,
        weights: list[float] | None = None,
        min_agreement: int = 2,
        min_score: float = 1.5,
        window_bars: int = 5,
        **_: Any,
    ) -> None:
        self.members: list[Strategy] = members or []
        self.weights: list[float] = weights or [1.0] * len(self.members)
        if len(self.weights) != len(self.members):
            raise ValueError("weights must match members length")
        self.min_agreement = min_agreement
        self.min_score = min_score
        self.window_bars = max(1, window_bars)

    def min_history(self) -> int:
        if not self.members:
            return 50
        return max(m.min_history() for m in self.members) + self.window_bars

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        if not self.members or len(df) < self.min_history():
            return Signal.FLAT

        score = 0.0
        long_count = 0
        short_count = 0

        for member, w in zip(self.members, self.weights):
            vote = self._recent_vote(member, df, ctx)
            score += w * vote
            if vote > 0:
                long_count += 1
            elif vote < 0:
                short_count += 1

        if long_count >= self.min_agreement and score >= self.min_score:
            return Signal.LONG
        if short_count >= self.min_agreement and score <= -self.min_score:
            return Signal.SHORT
        return Signal.FLAT

    def _recent_vote(
        self, member: Strategy, df: pd.DataFrame, ctx: StrategyContext
    ) -> int:
        last_vote = 0
        start = max(member.min_history(), len(df) - self.window_bars)
        for i in range(start, len(df) + 1):
            v = int(member.generate(df.iloc[:i], ctx))
            if v != 0:
                last_vote = v
        return last_vote
