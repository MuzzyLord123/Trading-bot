"""Ichimoku Cloud strategy.

A long entry requires a "stack" of confirmations that traders typically
look for on the Ichimoku chart:

  1. Tenkan crosses up through Kijun (bullish momentum cross).
  2. Price is above the cloud (above max(senkou_a, senkou_b)) - the
     dominant trend is up.
  3. Cloud is bullish (senkou_a > senkou_b) - forward-looking bias.

A short signal fires on the inverse: tenkan/kijun bear cross AND price
below the cloud. This is the textbook Ichimoku entry; if any single
leg is missing, the strategy stays flat.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import ichimoku
from .base import Signal, Strategy, StrategyContext


class IchimokuStrategy(Strategy):
    name = "ichimoku"

    def __init__(
        self,
        tenkan: int = 9,
        kijun: int = 26,
        senkou_b: int = 52,
        displacement: int = 26,
    ) -> None:
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b = senkou_b
        self.displacement = displacement

    def min_history(self) -> int:
        return self.senkou_b + self.displacement + 5

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        if len(df) < self.min_history():
            return Signal.FLAT
        ich = ichimoku(df, self.tenkan, self.kijun, self.senkou_b, self.displacement)
        if pd.isna(ich["senkou_a"].iloc[-1]) or pd.isna(ich["senkou_b"].iloc[-1]):
            return Signal.FLAT

        tenkan_now, tenkan_prev = float(ich["tenkan"].iloc[-1]), float(ich["tenkan"].iloc[-2])
        kijun_now, kijun_prev = float(ich["kijun"].iloc[-1]), float(ich["kijun"].iloc[-2])
        cloud_top = max(float(ich["senkou_a"].iloc[-1]), float(ich["senkou_b"].iloc[-1]))
        cloud_bot = min(float(ich["senkou_a"].iloc[-1]), float(ich["senkou_b"].iloc[-1]))
        bullish_cloud = ich["senkou_a"].iloc[-1] > ich["senkou_b"].iloc[-1]
        last_close = float(df["close"].iloc[-1])

        bull_cross = tenkan_now > kijun_now and tenkan_prev <= kijun_prev
        bear_cross = tenkan_now < kijun_now and tenkan_prev >= kijun_prev

        if bull_cross and last_close > cloud_top and bullish_cloud:
            return Signal.LONG
        if bear_cross and last_close < cloud_bot:
            return Signal.SHORT
        return Signal.FLAT
