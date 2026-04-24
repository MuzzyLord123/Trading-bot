"""Coverage for bot.news_stocks beyond the existing basic test."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bot import news_stocks
from bot.news_stocks import (
    days_until_earnings,
    detect_news_gaps,
    fetch_earnings_dates,
    is_bearish_gap,
    load_earnings_calendar,
)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(news_stocks, "EARNINGS_CACHE_DIR", tmp_path / "earnings")
    yield


# ---------------- days_until_earnings ----------------
def test_days_until_earnings_on_empty_returns_none():
    assert days_until_earnings(pd.DataFrame(columns=["date"]), pd.Timestamp("2024-01-01")) is None


def test_days_until_earnings_on_none_returns_none():
    assert days_until_earnings(None, pd.Timestamp("2024-01-01")) is None


def test_days_until_earnings_picks_next_upcoming_date():
    df = pd.DataFrame({"date": pd.to_datetime([
        "2023-10-01", "2024-01-20", "2024-04-20"
    ], utc=True)})
    # As of 2024-01-10 the next earnings is 2024-01-20, so 10 days away.
    days = days_until_earnings(df, pd.Timestamp("2024-01-10"))
    assert days == 10


def test_days_until_earnings_returns_none_when_all_in_past():
    df = pd.DataFrame({"date": pd.to_datetime(["2023-01-01"], utc=True)})
    assert days_until_earnings(df, pd.Timestamp("2024-01-10")) is None


def test_days_until_earnings_zero_on_same_day():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-10"], utc=True)})
    assert days_until_earnings(df, pd.Timestamp("2024-01-10")) == 0


def test_days_until_earnings_handles_naive_timestamp():
    # Input timestamp without tz should be coerced to UTC and still work.
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-20"], utc=True)})
    days = days_until_earnings(df, pd.Timestamp("2024-01-10"))
    assert days == 10


# ---------------- detect_news_gaps ----------------
def _gap_frame(opens, closes, volumes):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=len(opens), freq="D", tz="UTC"),
        "open": opens, "high": [max(o, c) + 0.5 for o, c in zip(opens, closes)],
        "low": [min(o, c) - 0.5 for o, c in zip(opens, closes)],
        "close": closes, "volume": volumes,
    })


def test_detect_news_gaps_flags_large_gap_on_high_volume():
    # Bar 22 gaps up 10% on 3x average volume -> flagged.
    n = 30
    closes = [100.0] * n
    opens = list(closes)
    volumes = [1000.0] * n
    opens[22] = 110.0
    volumes[22] = 3000.0
    flags = detect_news_gaps(_gap_frame(opens, closes, volumes),
                             threshold_pct=0.05, volume_mult=2.0)
    assert bool(flags.iloc[22])
    # Bars without a gap are not flagged.
    assert not bool(flags.iloc[10])


def test_detect_news_gaps_requires_both_gap_and_volume():
    n = 30
    closes = [100.0] * n
    opens = list(closes)
    volumes = [1000.0] * n
    # Big gap, normal volume - not flagged.
    opens[22] = 110.0
    flags = detect_news_gaps(_gap_frame(opens, closes, volumes))
    assert not bool(flags.iloc[22])


def test_detect_news_gaps_on_empty_returns_empty():
    assert detect_news_gaps(pd.DataFrame()).empty


def test_detect_news_gaps_on_none_returns_empty():
    assert detect_news_gaps(None).empty


# ---------------- is_bearish_gap ----------------
def test_is_bearish_gap_flags_negative_gaps_only():
    df = _gap_frame(opens=[100.0, 105.0, 90.0], closes=[100.0, 105.0, 90.0], volumes=[1.0, 1.0, 1.0])
    # Bar 2 opens 90 after prev close 105 -> ~-14% gap.
    flags = is_bearish_gap(df, threshold_pct=0.05)
    assert bool(flags.iloc[2])
    # A positive gap is not flagged.
    df2 = _gap_frame(opens=[100.0, 110.0], closes=[100.0, 110.0], volumes=[1.0, 1.0])
    assert not bool(is_bearish_gap(df2).iloc[1])


def test_is_bearish_gap_empty_frame():
    assert is_bearish_gap(pd.DataFrame()).empty


# ---------------- earnings cache + fetch ----------------
def test_fetch_earnings_returns_empty_when_yfinance_missing(monkeypatch):
    # Simulate yfinance not installed.
    import sys
    monkeypatch.setitem(sys.modules, "yfinance", None)
    df = fetch_earnings_dates("AAPL")
    assert df.empty


def test_fetch_earnings_uses_cache(tmp_path, monkeypatch):
    # Pre-populate the cache; patched yfinance shouldn't be hit.
    import types
    fake_yf = types.SimpleNamespace(Ticker=lambda s: (_ for _ in ()).throw(AssertionError("should not be called")))
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    news_stocks._write_earnings_cache("AAPL", pd.DataFrame({"date": pd.to_datetime(["2024-01-20"], utc=True)}))
    df = fetch_earnings_dates("AAPL")
    assert len(df) == 1


def test_fetch_earnings_swallows_yfinance_error(monkeypatch):
    import types
    class _T:
        def __init__(self, s): pass
        @property
        def earnings_dates(self): raise RuntimeError("API down")
    fake_yf = types.SimpleNamespace(Ticker=_T)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    df = fetch_earnings_dates("AAPL")
    assert df.empty


# ---------------- load_earnings_calendar ----------------
def test_load_earnings_calendar_per_symbol_isolation(monkeypatch):
    """A failure on one symbol must not abort the calendar for others."""
    import types

    class _T:
        def __init__(self, s): self.s = s
        @property
        def earnings_dates(self):
            if self.s == "BAD":
                raise RuntimeError("boom")
            # Return a pandas DataFrame shaped like yfinance does.
            return pd.DataFrame({"col": [1]}, index=pd.to_datetime(["2024-01-20"], utc=True))

    fake_yf = types.SimpleNamespace(Ticker=_T)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)

    result = load_earnings_calendar(["AAPL", "BAD", "MSFT"])
    assert set(result) == {"AAPL", "BAD", "MSFT"}
    # BAD still has a DataFrame, just empty.
    assert result["BAD"].empty
