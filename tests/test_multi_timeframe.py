from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.strategies import (
    MultiTimeframeStrategy,
    Signal,
    Strategy,
    StrategyContext,
    build_strategy_from_config,
)


class _AlwaysLong(Strategy):
    def generate(self, df, ctx):
        return Signal.LONG

    def min_history(self):
        return 1


class _AlwaysShort(Strategy):
    def generate(self, df, ctx):
        return Signal.SHORT

    def min_history(self):
        return 1


class _AlwaysFlat(Strategy):
    def generate(self, df, ctx):
        return Signal.FLAT

    def min_history(self):
        return 1


class _TrendFollower(Strategy):
    """Returns LONG iff close is above its rolling mean."""
    def __init__(self, period=20):
        self.period = period

    def generate(self, df, ctx):
        if len(df) < self.period + 1:
            return Signal.FLAT
        mean = df["close"].iloc[-self.period:].mean()
        return Signal.LONG if df["close"].iloc[-1] > mean else Signal.FLAT

    def min_history(self):
        return self.period + 1


def _ohlcv(close, freq="h"):
    close = np.asarray(close, dtype=float)
    ts = pd.date_range("2024-01-01", periods=len(close), freq=freq, tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": close, "high": close + 0.5,
        "low": close - 0.5, "close": close, "volume": np.ones_like(close),
    })


CTX = StrategyContext(symbol="AAPL", timeframe="1h")


def test_passes_flat_signal_through_unchanged():
    mtf = MultiTimeframeStrategy(_AlwaysFlat())
    df = _ohlcv(np.linspace(100, 200, 500))
    assert mtf.generate(df, CTX) == Signal.FLAT


def test_passes_short_through_without_htf_check():
    mtf = MultiTimeframeStrategy(_AlwaysShort())
    df = _ohlcv(np.linspace(100, 90, 500))
    assert mtf.generate(df, CTX) == Signal.SHORT


def test_allows_long_when_htf_agrees():
    mtf = MultiTimeframeStrategy(_TrendFollower(period=20))
    # Strong uptrend: both 1h and resampled 1D should be above their means.
    df = _ohlcv(np.linspace(100, 300, 500))
    assert mtf.generate(df, CTX) == Signal.LONG


def test_blocks_long_when_htf_disagrees():
    """LTF and HTF can disagree during a pullback - confirmation kicks in."""
    # Long rally (HTF bullish) but final 60 bars drop below the LTF mean,
    # so LTF _TrendFollower says FLAT - overall result is FLAT, same outcome.
    # Instead: force LTF LONG via AlwaysLong but HTF using trend follower
    # which will be FLAT on a declining series.
    mtf = MultiTimeframeStrategy(_AlwaysLong())
    # Build data where resampled 1D would NOT show an uptrend. AlwaysLong
    # always says LONG on LTF, so the only reason to block is the HTF.
    # We need an inner that goes LTF=LONG HTF=FLAT: use _TrendFollower on
    # a series where late LTF bars are above LTF mean but late daily bars
    # are below daily mean.
    class _MixedSignal(Strategy):
        def __init__(self): pass
        def generate(self, df, ctx):
            # LTF-ish = short window above its mean; HTF-ish = long window
            if len(df) < 200:
                return Signal.FLAT
            if ctx.timeframe == "1h":
                recent = df["close"].iloc[-30:]
                return Signal.LONG if recent.iloc[-1] > recent.mean() else Signal.FLAT
            # anything else counts as HTF
            return Signal.FLAT
        def min_history(self): return 30

    mtf = MultiTimeframeStrategy(_MixedSignal())
    # Series that closes high on recent 30 bars (LTF up) but overall flat
    n = 500
    x = np.concatenate([np.full(n - 30, 100.0), np.linspace(100, 110, 30)])
    df = _ohlcv(x)
    assert mtf.generate(df, CTX) == Signal.FLAT


def test_loose_mode_only_blocks_explicit_short_htf():
    class _LtfLongHtfShort(Strategy):
        def generate(self, df, ctx):
            if ctx.timeframe == "1h":
                return Signal.LONG
            return Signal.SHORT  # HTF
        def min_history(self): return 1

    strict = MultiTimeframeStrategy(_LtfLongHtfShort(), require_long_on_htf=True)
    loose = MultiTimeframeStrategy(_LtfLongHtfShort(), require_long_on_htf=False)
    df = _ohlcv(np.linspace(100, 150, 500))
    # strict: HTF not LONG -> block
    assert strict.generate(df, CTX) == Signal.FLAT
    # loose: HTF is SHORT (< 0) -> still blocks
    assert loose.generate(df, CTX) == Signal.FLAT


def test_builder_wires_multi_timeframe_when_enabled():
    strat = build_strategy_from_config(
        "ma_crossover",
        params={"ma_crossover": {"fast": 5, "slow": 20}},
        multi_timeframe_cfg={"enabled": True, "rule": "1D"},
    )
    assert isinstance(strat, MultiTimeframeStrategy)


def test_builder_skips_multi_timeframe_when_disabled():
    from bot.strategies import MaCrossoverStrategy
    strat = build_strategy_from_config(
        "ma_crossover",
        params={"ma_crossover": {"fast": 5, "slow": 20}},
        multi_timeframe_cfg={"enabled": False},
    )
    # Unwrapped - straight MA crossover
    assert isinstance(strat, MaCrossoverStrategy)


def test_unknown_timeframe_falls_open():
    # Unknown timeframes should let signals through unchanged rather
    # than silently blocking everything.
    mtf = MultiTimeframeStrategy(_AlwaysLong())
    df = _ohlcv(np.linspace(100, 200, 500))
    ctx = StrategyContext(symbol="AAPL", timeframe="3m")  # not in the map
    assert mtf.generate(df, ctx) == Signal.LONG
