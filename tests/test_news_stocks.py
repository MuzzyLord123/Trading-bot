from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from bot.news_stocks import (
    days_until_earnings,
    detect_news_gaps,
    is_bearish_gap,
)
from bot.strategies.filtered import FilteredStrategy
from bot.strategies.base import Signal, Strategy, StrategyContext


class _AlwaysLong(Strategy):
    name = "always_long"

    def min_history(self) -> int:
        return 1

    def generate(self, df: pd.DataFrame, ctx: StrategyContext) -> Signal:
        return Signal.LONG


def _make_df(n: int = 60, base: float = 100.0) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = np.linspace(base, base * 1.10, n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        }
    )


def test_days_until_earnings_returns_correct_value():
    earnings = pd.DataFrame(
        {"date": pd.to_datetime(["2024-02-15", "2024-05-15"], utc=True)}
    )
    days = days_until_earnings(earnings, pd.Timestamp("2024-02-10", tz="UTC"))
    assert days == 5


def test_days_until_earnings_handles_no_upcoming():
    earnings = pd.DataFrame(
        {"date": pd.to_datetime(["2024-02-15"], utc=True)}
    )
    days = days_until_earnings(earnings, pd.Timestamp("2024-03-01", tz="UTC"))
    assert days is None


def test_days_until_earnings_empty_df():
    assert days_until_earnings(pd.DataFrame(columns=["date"]), pd.Timestamp("2024-01-01", tz="UTC")) is None


def test_detect_news_gaps_flags_large_gap_with_volume():
    df = _make_df(n=30)
    # Inject a 7% gap up with 4x volume on bar 25.
    df.loc[25, "open"] = df.loc[24, "close"] * 1.07
    df.loc[25, "volume"] = 5_000_000.0
    flags = detect_news_gaps(df, threshold_pct=0.05, volume_mult=2.0)
    assert flags.iloc[25] is np.True_ or flags.iloc[25] is True or bool(flags.iloc[25])


def test_detect_news_gaps_ignores_small_moves():
    df = _make_df(n=30)
    flags = detect_news_gaps(df, threshold_pct=0.05, volume_mult=2.0)
    assert not flags.any()


def test_is_bearish_gap_only_flags_negative():
    df = _make_df(n=10)
    # Inject 6% gap DOWN.
    df.loc[5, "open"] = df.loc[4, "close"] * 0.94
    flags = is_bearish_gap(df, threshold_pct=0.05)
    assert bool(flags.iloc[5])
    # Now a positive gap of equal magnitude — should NOT flag bearish.
    df.loc[7, "open"] = df.loc[6, "close"] * 1.06
    flags = is_bearish_gap(df, threshold_pct=0.05)
    assert not bool(flags.iloc[7])


def test_filter_blocks_entry_during_earnings_blackout():
    earnings = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-05"], utc=True)}
    )
    df = _make_df(n=10)
    # Final bar is 2024-01-10; inject earnings 2 days ahead i.e. 2024-01-12.
    earnings = pd.DataFrame({"date": pd.to_datetime([df["timestamp"].iloc[-1] + timedelta(days=2)], utc=True)})
    flt = FilteredStrategy(
        inner=_AlwaysLong(),
        trend_ema=2, adx_period=2, atr_period=2,
        supertrend_period=2, volume_period=2,
        require_uptrend=False,
        min_adx=0,
        min_atr_pct=0,
        earnings_calendar={"AAPL": earnings},
        earnings_blackout_days=3,
    )
    sig = flt.generate(df, StrategyContext(symbol="AAPL", timeframe="1d"))
    assert sig == Signal.FLAT


def test_filter_allows_entry_outside_blackout():
    df = _make_df(n=10)
    earnings = pd.DataFrame({"date": pd.to_datetime([df["timestamp"].iloc[-1] + timedelta(days=30)], utc=True)})
    flt = FilteredStrategy(
        inner=_AlwaysLong(),
        trend_ema=2, adx_period=2, atr_period=2,
        supertrend_period=2, volume_period=2,
        require_uptrend=False,
        min_adx=0,
        min_atr_pct=0,
        earnings_calendar={"AAPL": earnings},
        earnings_blackout_days=3,
    )
    sig = flt.generate(df, StrategyContext(symbol="AAPL", timeframe="1d"))
    assert sig == Signal.LONG


def test_filter_blocks_after_bearish_gap():
    df = _make_df(n=10)
    df.loc[9, "open"] = df.loc[8, "close"] * 0.92  # 8% gap down
    flt = FilteredStrategy(
        inner=_AlwaysLong(),
        trend_ema=2, adx_period=2, atr_period=2,
        supertrend_period=2, volume_period=2,
        require_uptrend=False,
        min_adx=0,
        min_atr_pct=0,
        news_gap_threshold=0.05,
    )
    sig = flt.generate(df, StrategyContext(symbol="AAPL", timeframe="1d"))
    assert sig == Signal.FLAT


def test_filter_passes_through_with_no_gap():
    df = _make_df(n=10)
    flt = FilteredStrategy(
        inner=_AlwaysLong(),
        trend_ema=2, adx_period=2, atr_period=2,
        supertrend_period=2, volume_period=2,
        require_uptrend=False,
        min_adx=0,
        min_atr_pct=0,
        news_gap_threshold=0.05,
    )
    sig = flt.generate(df, StrategyContext(symbol="AAPL", timeframe="1d"))
    assert sig == Signal.LONG
