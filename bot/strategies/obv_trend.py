"""OBV trend strategy - volume confirming price.

Entry:  OBV is above its ``ema`` *and* rising vs ``lookback`` bars ago
        *and* close is above a ``trend_ema`` of price. All three must
        agree so we avoid chasing moves on fading volume.
Exit:   OBV drops back below its EMA - smart money leaving.

OBV is cumulative so its absolute level is meaningless across symbols;
always compare to a per-symbol moving average, never to a fixed number.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import ema, obv
from .base import Signal, Strategy, StrategyContext


class ObvTrendStrategy(Strategy):
    name = "obv_trend"

    def __init__(
        self,
        ema_period: int = 20,
        lookback: int = 5,
        trend_ema: int = 50,
    ) -> None:
        self.ema_period = ema_period
        self.lookback = lookback
        self.trend_ema = trend_ema

    def min_history(self) -> int:
        return max(self.trend_ema, self.ema_period + self.lookback) + 2

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        if len(df) < self.min_history():
            return Signal.FLAT
        ob = obv(df)
        ob_ema = ema(ob, self.ema_period)
        price_ema = ema(df["close"], self.trend_ema)
        last_ob = float(ob.iloc[-1])
        last_ob_ema = float(ob_ema.iloc[-1])
        prev_ob = float(ob.iloc[-1 - self.lookback]) if len(ob) > self.lookback else last_ob
        last_close = float(df["close"].iloc[-1])
        last_trend = float(price_ema.iloc[-1]) if pd.notna(price_ema.iloc[-1]) else last_close

        if last_ob > last_ob_ema and last_ob > prev_ob and last_close > last_trend:
            return Signal.LONG
        if last_ob < last_ob_ema:
            return Signal.SHORT
        return Signal.FLAT
