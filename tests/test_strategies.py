from __future__ import annotations

import numpy as np
import pandas as pd

from bot.strategies import (
    BollingerStrategy,
    EnsembleStrategy,
    MaCrossoverStrategy,
    MacdStrategy,
    RsiReversionStrategy,
    Signal,
    StrategyContext,
)


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


CTX = StrategyContext(symbol="VUAG.L", timeframe="1h")


def test_ma_crossover_long_on_uptrend():
    up = np.concatenate([np.linspace(100, 80, 100), np.linspace(80, 160, 100)])
    strat = MaCrossoverStrategy(fast=10, slow=30)
    signals = []
    for i in range(strat.min_history(), len(up)):
        signals.append(int(strat.generate(_df(up[: i + 1]), CTX)))
    assert Signal.LONG in signals


def test_rsi_reversion_detects_oversold_bounce():
    rng = np.random.default_rng(0)
    down = 100 - np.linspace(0, 40, 60)
    up = down[-1] + np.linspace(0, 20, 20)
    prices = np.concatenate([down, up]) + rng.normal(0, 0.5, 80)
    strat = RsiReversionStrategy(period=14, oversold=35, overbought=65)
    signals = []
    for i in range(strat.min_history(), len(prices)):
        signals.append(int(strat.generate(_df(prices[: i + 1]), CTX)))
    assert any(s > 0 for s in signals)


def test_macd_strategy_returns_valid_signal_values():
    rng = np.random.default_rng(2)
    prices = 100 + rng.normal(0, 1, 300).cumsum()
    strat = MacdStrategy()
    sig = strat.generate(_df(prices), CTX)
    assert sig in (Signal.LONG, Signal.SHORT, Signal.FLAT)


def test_bollinger_returns_valid_signal_values():
    rng = np.random.default_rng(3)
    prices = 100 + rng.normal(0, 1, 200).cumsum()
    strat = BollingerStrategy()
    sig = strat.generate(_df(prices), CTX)
    assert sig in (Signal.LONG, Signal.SHORT, Signal.FLAT)


def test_ensemble_requires_agreement():
    class AlwaysLong:
        def generate(self, df, ctx):
            return Signal.LONG

        def min_history(self):
            return 1

    class AlwaysFlat:
        def generate(self, df, ctx):
            return Signal.FLAT

        def min_history(self):
            return 1

    prices = np.linspace(100, 110, 50)
    ens = EnsembleStrategy(members=[AlwaysLong(), AlwaysFlat(), AlwaysFlat()], min_agreement=2)
    assert ens.generate(_df(prices), CTX) == Signal.FLAT
    ens2 = EnsembleStrategy(
        members=[AlwaysLong(), AlwaysLong(), AlwaysFlat()], min_agreement=2
    )
    assert ens2.generate(_df(prices), CTX) == Signal.LONG
