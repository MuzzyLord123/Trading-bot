from __future__ import annotations

import numpy as np
import pandas as pd

from bot.indicators import adx
from bot.strategies import (
    EnsembleStrategy,
    FilteredStrategy,
    MaCrossoverStrategy,
    Signal,
    StrategyContext,
)
from bot.strategies.base import Strategy


def _df(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(close), freq="h", tz="UTC"),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones_like(close),
        }
    )


CTX = StrategyContext(symbol="TEST/GBP", timeframe="1h")


def test_adx_is_bounded_and_rises_in_trend():
    trend = np.linspace(100, 200, 400)
    flat = np.full(400, 200.0) + np.sin(np.linspace(0, 5, 400)) * 0.5
    a_trend = adx(_df(trend), 14).dropna()
    a_flat = adx(_df(flat), 14).dropna()
    assert (a_trend >= 0).all() and (a_trend <= 100).all()
    assert a_trend.iloc[-1] > a_flat.iloc[-1]


def test_filtered_strategy_blocks_entries_in_downtrend():
    class AlwaysLong(Strategy):
        def generate(self, df, ctx):
            return Signal.LONG

        def min_history(self):
            return 1

    down = np.linspace(200, 100, 400)
    wrapped = FilteredStrategy(
        inner=AlwaysLong(),
        trend_ema=50,
        require_uptrend=True,
        min_adx=0.0,
        min_atr_pct=0.0,
    )
    assert wrapped.generate(_df(down), CTX) == Signal.FLAT


def test_filtered_strategy_allows_entries_in_uptrend():
    class AlwaysLong(Strategy):
        def generate(self, df, ctx):
            return Signal.LONG

        def min_history(self):
            return 1

    up = np.linspace(100, 300, 400)
    wrapped = FilteredStrategy(
        inner=AlwaysLong(),
        trend_ema=50,
        require_uptrend=True,
        min_adx=0.0,
        min_atr_pct=0.0,
    )
    assert wrapped.generate(_df(up), CTX) == Signal.LONG


def test_filtered_strategy_passes_exits_through():
    class AlwaysShort(Strategy):
        def generate(self, df, ctx):
            return Signal.SHORT

        def min_history(self):
            return 1

    down = np.linspace(200, 100, 400)
    wrapped = FilteredStrategy(
        inner=AlwaysShort(),
        trend_ema=50,
        require_uptrend=True,
    )
    assert wrapped.generate(_df(down), CTX) == Signal.SHORT


def test_ensemble_lookback_window_allows_stale_agreement():
    close = np.concatenate([np.linspace(100, 80, 50), np.linspace(80, 140, 100)])
    fast = MaCrossoverStrategy(fast=5, slow=20)

    class DelayedLong(Strategy):
        def __init__(self):
            self.count = 0

        def generate(self, df, ctx):
            self.count += 1
            return Signal.LONG if self.count % 3 == 0 else Signal.FLAT

        def min_history(self):
            return 1

    ens_no_window = EnsembleStrategy(
        members=[fast, DelayedLong()], min_agreement=2, min_score=0.0, window_bars=1
    )
    ens_with_window = EnsembleStrategy(
        members=[fast, DelayedLong()], min_agreement=2, min_score=0.0, window_bars=10
    )
    df = _df(close)
    results_no = [int(ens_no_window.generate(df.iloc[:i], CTX)) for i in range(60, len(df))]
    results_yes = [int(ens_with_window.generate(df.iloc[:i], CTX)) for i in range(60, len(df))]
    assert sum(1 for r in results_yes if r > 0) >= sum(1 for r in results_no if r > 0)
