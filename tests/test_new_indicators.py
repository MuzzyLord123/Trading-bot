from __future__ import annotations

import numpy as np
import pandas as pd

from bot.indicators import cci, donchian, keltner, obv, stochastic
from bot.strategies import (
    CciStrategy,
    DonchianStrategy,
    KeltnerStrategy,
    ObvTrendStrategy,
    Signal,
    StochasticStrategy,
    StrategyContext,
)


def _ohlcv(close, high=None, low=None, volume=None):
    close = np.asarray(close, dtype=float)
    high = close + 1.0 if high is None else np.asarray(high, dtype=float)
    low = close - 1.0 if low is None else np.asarray(low, dtype=float)
    volume = np.ones_like(close) if volume is None else np.asarray(volume, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(close), freq="h", tz="UTC"),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


CTX = StrategyContext(symbol="AAPL", timeframe="1h")


# ---------------- stochastic ----------------
def test_stochastic_bounded_between_0_and_100():
    rng = np.random.default_rng(1)
    s = 100 + rng.normal(0, 1, 300).cumsum()
    df = _ohlcv(s, s + 2, s - 2)
    k = stochastic(df).dropna()
    assert ((k["k"] >= 0) & (k["k"] <= 100)).all()


def test_stochastic_strategy_returns_signal_enum():
    rng = np.random.default_rng(2)
    s = 100 + rng.normal(0, 1, 200).cumsum()
    df = _ohlcv(s, s + 2, s - 2)
    sig = StochasticStrategy().generate(df, CTX)
    assert sig in (Signal.LONG, Signal.FLAT, Signal.SHORT)


def test_stochastic_strategy_insufficient_history_is_flat():
    df = _ohlcv(np.linspace(100, 110, 5))
    assert StochasticStrategy().generate(df, CTX) == Signal.FLAT


# ---------------- donchian ----------------
def test_donchian_channel_sanity():
    df = _ohlcv(np.linspace(100, 200, 100), np.linspace(101, 201, 100), np.linspace(99, 199, 100))
    ch = donchian(df, 20).dropna()
    assert (ch["upper"] >= ch["lower"]).all()
    assert (ch["mid"] >= ch["lower"]).all()
    assert (ch["mid"] <= ch["upper"]).all()


def test_donchian_strategy_fires_long_on_breakout():
    # flat for 30 bars, then a sharp breakout on the final bar.
    close = list(np.full(30, 100.0)) + [110.0]
    high = list(np.full(30, 100.5)) + [110.5]
    low = list(np.full(30, 99.5)) + [109.5]
    df = _ohlcv(close, high, low)
    assert DonchianStrategy(period=20).generate(df, CTX) == Signal.LONG


def test_donchian_strategy_fires_short_on_breakdown():
    close = list(np.full(30, 100.0)) + [90.0]
    high = list(np.full(30, 100.5)) + [90.5]
    low = list(np.full(30, 99.5)) + [89.5]
    df = _ohlcv(close, high, low)
    assert DonchianStrategy(period=20, exit_period=10).generate(df, CTX) == Signal.SHORT


# ---------------- keltner ----------------
def test_keltner_channel_widens_with_volatility():
    calm = _ohlcv(np.linspace(100, 101, 60), np.linspace(100.1, 101.1, 60), np.linspace(99.9, 100.9, 60))
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 3, 60).cumsum()
    noisy = _ohlcv(100 + noise, 100 + noise + 4, 100 + noise - 4)
    wc = keltner(calm).dropna().iloc[-1]
    wn = keltner(noisy).dropna().iloc[-1]
    assert (wn["upper"] - wn["lower"]) > (wc["upper"] - wc["lower"])


def test_keltner_strategy_long_on_upper_break():
    # mostly flat, final bar closes above the upper band
    close = list(np.full(50, 100.0)) + [120.0]
    high = [c + 0.3 for c in close]
    low = [c - 0.3 for c in close]
    df = _ohlcv(close, high, low)
    assert KeltnerStrategy(period=20, multiplier=2.0, atr_period=10).generate(df, CTX) == Signal.LONG


# ---------------- CCI ----------------
def test_cci_produces_values_on_random_walk():
    rng = np.random.default_rng(4)
    s = 100 + rng.normal(0, 1, 200).cumsum()
    df = _ohlcv(s, s + 1, s - 1)
    out = cci(df).dropna()
    assert not out.empty


def test_cci_strategy_fires_long_on_cross_above_minus_100():
    # Sharp drop (CCI well below -100) followed by recovery should produce an
    # upward cross of -100 somewhere along the recovery path.
    s = np.concatenate([np.full(30, 100.0), np.linspace(100, 70, 30), np.linspace(70, 110, 60)])
    df = _ohlcv(s, s + 0.5, s - 0.5)
    signals = [
        CciStrategy(period=20).generate(df.iloc[:n], CTX)
        for n in range(40, len(df))
    ]
    assert Signal.LONG in signals


# ---------------- OBV ----------------
def test_obv_rises_when_price_rises_on_volume():
    close = np.linspace(100, 110, 50)
    volume = np.full(50, 1000.0)
    df = _ohlcv(close, close + 0.1, close - 0.1, volume)
    ob = obv(df)
    assert ob.iloc[-1] > ob.iloc[0]


def test_obv_trend_strategy_long_in_rising_market():
    close = np.linspace(100, 130, 80)
    volume = np.full(80, 1000.0)
    df = _ohlcv(close, close + 0.1, close - 0.1, volume)
    sig = ObvTrendStrategy(ema_period=20, lookback=5, trend_ema=50).generate(df, CTX)
    assert sig == Signal.LONG


def test_obv_trend_strategy_short_when_volume_flow_reverses():
    # price up but the LATEST segment has OBV falling below its EMA
    close = np.concatenate([np.linspace(100, 130, 60), np.linspace(130, 115, 40)])
    volume = np.concatenate([np.full(60, 1000.0), np.full(40, 2000.0)])  # big volume on the drop
    df = _ohlcv(close, close + 0.1, close - 0.1, volume)
    sig = ObvTrendStrategy(ema_period=20, lookback=5, trend_ema=50).generate(df, CTX)
    assert sig in (Signal.SHORT, Signal.FLAT)  # always exits, may be flat if borderline


# ---------------- registry / builder ----------------
def test_all_new_strategies_registered_and_buildable():
    from bot.strategies import build_strategy_from_config
    for name in ("stochastic", "donchian", "keltner", "cci", "obv_trend"):
        strat = build_strategy_from_config(name, {name: {}})
        assert strat.name == name or strat.name  # built without crashing


def test_ensemble_can_include_new_strategies():
    from bot.strategies import build_strategy_from_config
    strat = build_strategy_from_config(
        "ensemble",
        params={},
        ensemble_cfg={
            "members": ["ma_crossover", "stochastic", "donchian", "cci"],
            "min_agreement": 1,
            "min_score": 0.0,
        },
    )
    # exercise .generate once to make sure construction wired everything up
    rng = np.random.default_rng(7)
    s = 100 + rng.normal(0, 1, 200).cumsum()
    df = _ohlcv(s, s + 0.5, s - 0.5)
    out = strat.generate(df, CTX)
    assert out in (Signal.LONG, Signal.FLAT, Signal.SHORT)
