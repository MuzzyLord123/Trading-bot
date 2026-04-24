from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bot.data_cache import OhlcvCache, fetch_with_cache


def _frame(start: str, n: int, freq: str = "h") -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    close = np.linspace(100, 100 + n, n)
    return pd.DataFrame({
        "timestamp": ts,
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.ones(n),
    })


class FakeSource:
    def __init__(self, df_to_return: pd.DataFrame) -> None:
        self._df = df_to_return
        self.calls = 0

    def fetch_ohlcv_range(self, symbol, timeframe, since_ms, until_ms):
        self.calls += 1
        return self._df.copy()


def test_write_then_read_round_trip(tmp_path):
    cache = OhlcvCache(root=tmp_path)
    df = _frame("2024-01-01", 50)
    cache.write("AAPL", "1h", df)
    back = cache.read("AAPL", "1h")
    assert len(back) == 50
    assert (back["close"] == df["close"]).all()


def test_symbol_with_slash_or_dot_is_sanitised(tmp_path):
    cache = OhlcvCache(root=tmp_path)
    cache.write("VUAG.L", "1h", _frame("2024-01-01", 5))
    cache.write("BRK/A", "1h", _frame("2024-01-02", 5))
    assert cache.has("VUAG.L", "1h")
    assert cache.has("BRK/A", "1h")
    # No hidden dotfiles or subdirs leaked out.
    assert list(cache.root.rglob(".*.parquet")) == []


def test_covers_returns_true_only_for_wholly_covered_window(tmp_path):
    cache = OhlcvCache(root=tmp_path)
    cache.write("AAPL", "1h", _frame("2024-01-01", 100))
    spans_all = cache.covers(
        "AAPL", "1h",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    assert spans_all is True
    before_data = cache.covers(
        "AAPL", "1h",
        datetime(2023, 12, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    assert before_data is False


def test_read_window_slices_correctly(tmp_path):
    cache = OhlcvCache(root=tmp_path)
    cache.write("AAPL", "1h", _frame("2024-01-01", 100))
    sub = cache.read_window(
        "AAPL", "1h",
        datetime(2024, 1, 1, 10, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 20, tzinfo=timezone.utc),
    )
    assert len(sub) == 11  # inclusive on both ends


def test_merge_dedups_overlapping_rows(tmp_path):
    cache = OhlcvCache(root=tmp_path)
    cache.write("AAPL", "1h", _frame("2024-01-01", 50))
    # Same window, newer close values.
    updated = _frame("2024-01-01", 50)
    updated["close"] = updated["close"] + 10.0
    cache.write("AAPL", "1h", updated)
    back = cache.read("AAPL", "1h")
    assert len(back) == 50  # no duplicate timestamps
    # "keep=last" means the updated (newer) close survived.
    assert (back["close"] == updated["close"]).all()


def test_fetch_with_cache_hits_only_on_miss(tmp_path):
    df = _frame("2024-01-01", 30)
    source = FakeSource(df)
    cache = OhlcvCache(root=tmp_path)
    since_ms = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    until_ms = int(datetime(2024, 1, 2, 4, tzinfo=timezone.utc).timestamp() * 1000)
    fetch_with_cache(source, "AAPL", "1h", since_ms, until_ms, cache=cache)
    assert source.calls == 1
    # Second call is served from cache; source.calls stays at 1.
    fetch_with_cache(source, "AAPL", "1h", since_ms, until_ms, cache=cache)
    assert source.calls == 1


def test_fetch_with_cache_gracefully_handles_empty_source(tmp_path):
    source = FakeSource(pd.DataFrame())
    cache = OhlcvCache(root=tmp_path)
    out = fetch_with_cache(source, "AAPL", "1h", 0, 1_000_000, cache=cache)
    assert out.empty
    assert not cache.has("AAPL", "1h")


def test_clear_single_symbol_and_all(tmp_path):
    cache = OhlcvCache(root=tmp_path)
    cache.write("AAPL", "1h", _frame("2024-01-01", 5))
    cache.write("MSFT", "1h", _frame("2024-01-01", 5))
    cache.clear(symbol="AAPL", timeframe="1h")
    assert not cache.has("AAPL", "1h")
    assert cache.has("MSFT", "1h")
    cache.clear()  # wipe everything
    assert not cache.has("MSFT", "1h")


def test_coverage_returns_span(tmp_path):
    cache = OhlcvCache(root=tmp_path)
    cache.write("AAPL", "1h", _frame("2024-01-01", 10))
    entry = cache.coverage("AAPL", "1h")
    assert entry is not None
    assert entry.rows == 10
    assert entry.span_start.tzinfo is not None
