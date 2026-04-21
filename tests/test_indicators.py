from __future__ import annotations

import numpy as np
import pandas as pd

from bot.indicators import atr, bollinger, ema, macd, rsi, sma


def _series(values):
    return pd.Series(values, dtype=float)


def test_sma_and_ema_produce_expected_length():
    s = _series(range(1, 21))
    assert len(sma(s, 5)) == 20
    assert sma(s, 5).iloc[4] == 3.0
    e = ema(s, 5)
    assert not np.isnan(e.iloc[-1])


def test_rsi_bounds():
    rng = np.random.default_rng(42)
    s = _series(100 + rng.normal(0, 1, 200).cumsum())
    r = rsi(s, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_macd_columns_and_signal_properties():
    s = _series(np.linspace(100, 120, 80))
    m = macd(s, 12, 26, 9)
    assert set(m.columns) == {"macd", "signal", "hist"}
    assert not m["macd"].dropna().empty


def test_bollinger_bounds():
    s = _series(np.linspace(100, 200, 100))
    b = bollinger(s, 20, 2.0).dropna()
    assert (b["upper"] >= b["mid"]).all()
    assert (b["lower"] <= b["mid"]).all()


def test_atr_is_non_negative():
    rng = np.random.default_rng(1)
    close = 100 + rng.normal(0, 1, 100).cumsum()
    df = pd.DataFrame(
        {
            "high": close + 1,
            "low": close - 1,
            "close": close,
        }
    )
    out = atr(df, 14).dropna()
    assert (out >= 0).all()
