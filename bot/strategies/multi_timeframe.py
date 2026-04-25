"""Multi-timeframe confirmation wrapper.

The inner strategy fires on the native (lower) timeframe bars the engine
feeds it. This wrapper additionally resamples the same OHLCV data to a
higher timeframe and requires the same strategy - run against that
resampled frame - to also return a non-negative signal before the inner
LONG is allowed through.

Intuition: the lower timeframe catches the entry, the higher timeframe
confirms the regime. Runs the exact same strategy twice to avoid the
common failure mode of "trend-follower on 1h contradicted by mean-
reverter on daily" giving mixed signals.

Exits (Signal.SHORT from the inner) always pass through; we never want
a missing confirmation to trap us in a losing position.

This is *not* a tunable knob - it's a regime filter. Turn it on for
strategies that chop in range-bound markets (MA crossover, Donchian,
Keltner) and leave it off for mean-reverters that intentionally want
the lower-timeframe noise (RSI reversion, bollinger, stochastic, CCI).
"""
from __future__ import annotations

import logging

import pandas as pd

from ._helpers import resample_ohlcv
from .base import Signal, Strategy, StrategyContext

log = logging.getLogger("bot.strategies.multi_timeframe")


_HTF_RULES = {
    "1m": "15min",
    "5m": "1h",
    "15m": "4h",
    "30m": "4h",
    "1h": "1D",
    "60m": "1D",
    "1d": "1W",
}


class MultiTimeframeStrategy(Strategy):
    """Wrap ``inner`` so LONG entries require agreement from the same
    strategy evaluated on a higher timeframe."""

    name = "multi_timeframe"

    def __init__(
        self,
        inner: Strategy,
        htf_rule: str | None = None,
        require_long_on_htf: bool = True,
    ) -> None:
        self.inner = inner
        self.htf_rule = htf_rule
        self.require_long_on_htf = require_long_on_htf

    def min_history(self) -> int:
        # The HTF pass needs enough bars to run its own indicators. Pick
        # a conservative multiplier of 10 vs the inner min_history - a
        # daily bar is roughly 7 hourly bars of market time, so we round
        # up and pad.
        return max(self.inner.min_history() * 10, 100)

    def _resolve_rule(self, ctx: StrategyContext) -> str | None:
        if self.htf_rule:
            return self.htf_rule
        return _HTF_RULES.get(ctx.timeframe)

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        sig = self.inner.generate(df, ctx)
        # Always pass through exits and flats; only entries need confirming.
        if sig <= 0:
            return sig

        rule = self._resolve_rule(ctx)
        if not rule:
            # Unknown timeframe - fall open rather than blocking all trades.
            return sig

        try:
            htf_df = resample_ohlcv(df, rule)
        except Exception as exc:
            # Resample is deterministic - failure means a bad rule string
            # (e.g. "3z") slipped past validation. Log so the operator
            # can fix the config rather than silently never entering.
            log.warning("HTF resample with rule %r failed: %s", rule, exc)
            return Signal.FLAT
        if len(htf_df) < self.inner.min_history():
            return Signal.FLAT

        htf_ctx = StrategyContext(symbol=ctx.symbol, timeframe=rule)
        htf_sig = self.inner.generate(htf_df, htf_ctx)

        if self.require_long_on_htf:
            return sig if htf_sig > 0 else Signal.FLAT
        # Looser mode: block only explicit SHORT signals on the HTF.
        return sig if htf_sig >= 0 else Signal.FLAT
