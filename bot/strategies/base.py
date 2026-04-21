from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import pandas as pd


class Signal(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


@dataclass
class StrategyContext:
    symbol: str
    timeframe: str


class Strategy:
    """Base strategy. Subclasses implement ``generate``."""

    name: str = "base"

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        """Return a Signal for the most recent closed candle.

        `df` contains columns: timestamp, open, high, low, close, volume.
        Implementations should only look at data up to the final row to avoid
        look-ahead bias.
        """
        raise NotImplementedError

    def min_history(self) -> int:
        """Minimum candles required before ``generate`` can produce a signal."""
        return 50
