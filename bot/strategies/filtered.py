"""Regime-aware wrapper that filters entry signals in bad conditions."""
from __future__ import annotations

import pandas as pd

from ..indicators import adx, atr, ema, supertrend
from .base import Signal, Strategy, StrategyContext


def _resample_htf(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an OHLCV frame to a higher timeframe."""
    idx = df.set_index("timestamp")
    out = pd.DataFrame(
        {
            "open": idx["open"].resample(rule).first(),
            "high": idx["high"].resample(rule).max(),
            "low": idx["low"].resample(rule).min(),
            "close": idx["close"].resample(rule).last(),
            "volume": idx["volume"].resample(rule).sum(),
        }
    ).dropna()
    return out.reset_index()


class FilteredStrategy(Strategy):
    """Drop entry signals outside favourable market conditions.

    Only entry (LONG) signals are gated. Exit (SHORT) signals pass through
    so we always exit losers cleanly.

    Gates applied in order:
      1. Uptrend: price > ``trend_ema`` of lower timeframe.
      2. Higher-timeframe trend: price above ``htf_ema`` on ``htf_rule``.
      3. Trend strength: ADX >= ``min_adx``.
      4. Volatility floor: ATR / price >= ``min_atr_pct``.
      5. Supertrend direction: +1 (uptrend).
      6. Volume surge: last bar volume >= ``volume_mult`` * rolling mean.
    """

    name = "filtered"

    def __init__(
        self,
        inner: Strategy,
        trend_ema: int = 200,
        require_uptrend: bool = True,
        min_adx: float = 20.0,
        adx_period: int = 14,
        min_atr_pct: float = 0.001,
        atr_period: int = 14,
        htf_rule: str | None = None,
        htf_ema: int = 50,
        use_supertrend: bool = False,
        supertrend_period: int = 10,
        supertrend_multiplier: float = 3.0,
        volume_mult: float = 0.0,
        volume_period: int = 20,
    ) -> None:
        self.inner = inner
        self.trend_ema = trend_ema
        self.require_uptrend = require_uptrend
        self.min_adx = min_adx
        self.adx_period = adx_period
        self.min_atr_pct = min_atr_pct
        self.atr_period = atr_period
        self.htf_rule = htf_rule
        self.htf_ema = htf_ema
        self.use_supertrend = use_supertrend
        self.supertrend_period = supertrend_period
        self.supertrend_multiplier = supertrend_multiplier
        self.volume_mult = volume_mult
        self.volume_period = volume_period

    def min_history(self) -> int:
        base = self.inner.min_history()
        return max(
            base,
            self.trend_ema + 5,
            self.adx_period * 3,
            self.atr_period * 3,
            self.supertrend_period * 3,
            self.volume_period + 5,
        )

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        sig = self.inner.generate(df, ctx)
        if sig <= 0:
            return sig
        if len(df) < self.min_history():
            return Signal.FLAT

        last_close = float(df["close"].iloc[-1])

        if self.require_uptrend:
            trend = ema(df["close"], self.trend_ema).iloc[-1]
            if pd.isna(trend) or last_close < trend:
                return Signal.FLAT

        if self.htf_rule:
            htf = _resample_htf(df, self.htf_rule)
            if len(htf) < self.htf_ema + 2:
                return Signal.FLAT
            htf_trend = ema(htf["close"], self.htf_ema).iloc[-1]
            if pd.isna(htf_trend) or htf["close"].iloc[-1] < htf_trend:
                return Signal.FLAT

        if self.min_adx > 0:
            strength = adx(df, self.adx_period).iloc[-1]
            if pd.isna(strength) or strength < self.min_adx:
                return Signal.FLAT

        if self.min_atr_pct > 0:
            atr_val = atr(df, self.atr_period).iloc[-1]
            if pd.isna(atr_val) or atr_val / last_close < self.min_atr_pct:
                return Signal.FLAT

        if self.use_supertrend:
            st = supertrend(df, self.supertrend_period, self.supertrend_multiplier)
            direction = st["direction"].iloc[-1]
            if direction != 1:
                return Signal.FLAT

        if self.volume_mult > 0:
            vol = df["volume"]
            avg = vol.rolling(self.volume_period).mean().iloc[-1]
            if pd.isna(avg) or vol.iloc[-1] < self.volume_mult * avg:
                return Signal.FLAT

        return Signal.LONG
