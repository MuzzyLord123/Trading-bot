"""Deep coverage for bot.universe beyond the basic hardcoded-fallback
test that already exists."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from bot import universe
from bot.universe import (
    NASDAQ100_HARDCODED,
    _extract_symbols,
    _load_cached,
    _write_cache,
    expand_universe_tokens,
    load_nasdaq100,
    load_sp500,
)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Redirect the SP500/NASDAQ100 caches into tmp_path so tests don't
    collide with the real repo cache."""
    monkeypatch.setattr(universe, "SP500_CACHE", tmp_path / "sp500.csv")
    monkeypatch.setattr(universe, "NASDAQ100_CACHE", tmp_path / "nasdaq100.csv")
    yield


# ---------------- _extract_symbols ----------------
def test_extract_symbols_prefers_first_matching_column():
    html = """
    <table>
      <tr><th>Symbol</th><th>Name</th></tr>
      <tr><td>AAPL</td><td>Apple</td></tr>
      <tr><td>MSFT</td><td>Microsoft</td></tr>
    </table>
    """
    result = _extract_symbols(html, ("Symbol",))
    assert result == ["AAPL", "MSFT"]


def test_extract_symbols_normalises_dot_to_dash_for_yfinance():
    html = """
    <table>
      <tr><th>Symbol</th></tr>
      <tr><td>BRK.B</td></tr>
      <tr><td>BF.B</td></tr>
    </table>
    """
    assert _extract_symbols(html, ("Symbol",)) == ["BRK-B", "BF-B"]


def test_extract_symbols_drops_empty_and_nan():
    html = """
    <table>
      <tr><th>Symbol</th></tr>
      <tr><td>AAPL</td></tr>
      <tr><td></td></tr>
      <tr><td>nan</td></tr>
      <tr><td>MSFT</td></tr>
    </table>
    """
    assert _extract_symbols(html, ("Symbol",)) == ["AAPL", "MSFT"]


def test_extract_symbols_tries_fallback_column():
    html = """
    <table>
      <tr><th>Ticker</th></tr>
      <tr><td>AAPL</td></tr>
    </table>
    """
    assert _extract_symbols(html, ("Symbol", "Ticker")) == ["AAPL"]


def test_extract_symbols_raises_when_no_column_matches():
    html = "<table><tr><th>Something</th></tr><tr><td>X</td></tr></table>"
    with pytest.raises(RuntimeError, match="None of"):
        _extract_symbols(html, ("Symbol",))


# ---------------- cache round-trip ----------------
def test_cache_write_then_read(tmp_path):
    path = tmp_path / "x.csv"
    _write_cache(path, ["AAPL", "MSFT"])
    assert _load_cached(path) == ["AAPL", "MSFT"]


def test_cache_returns_none_when_missing(tmp_path):
    assert _load_cached(tmp_path / "missing.csv") is None


def test_cache_returns_none_when_stale(tmp_path, monkeypatch):
    import os
    path = tmp_path / "stale.csv"
    _write_cache(path, ["AAPL"])
    # Backdate by 8 days - TTL is 7.
    old = __import__("time").time() - 8 * 86400
    os.utime(path, (old, old))
    assert _load_cached(path) is None


# ---------------- load_sp500 ----------------
def test_load_sp500_uses_cache_when_fresh():
    _write_cache(universe.SP500_CACHE, ["AAPL", "MSFT"])
    with patch.object(universe, "_get_url") as get:
        result = load_sp500()
    get.assert_not_called()
    assert result == ["AAPL", "MSFT"]


def test_load_sp500_fetches_when_no_cache():
    html = """<table><tr><th>Symbol</th></tr><tr><td>AAPL</td></tr></table>"""
    with patch.object(universe, "_get_url", return_value=html):
        result = load_sp500()
    assert result == ["AAPL"]
    # Cache now written.
    assert universe.SP500_CACHE.exists()


def test_load_sp500_falls_back_to_stale_cache_on_fetch_failure():
    _write_cache(universe.SP500_CACHE, ["CACHED"])
    # Stale the cache so the TTL check doesn't short-circuit.
    import os, time as _time
    old = _time.time() - 8 * 86400
    os.utime(universe.SP500_CACHE, (old, old))
    with patch.object(universe, "_get_url", side_effect=RuntimeError("403")):
        result = load_sp500()
    assert result == ["CACHED"]


def test_load_sp500_raises_when_no_cache_and_fetch_fails():
    with patch.object(universe, "_get_url", side_effect=RuntimeError("403")):
        with pytest.raises(RuntimeError):
            load_sp500()


# ---------------- load_nasdaq100 ----------------
def test_load_nasdaq100_uses_hardcoded_fallback_when_everything_fails():
    # No cache written. _get_url fails.
    with patch.object(universe, "_get_url", side_effect=RuntimeError("403")):
        result = load_nasdaq100()
    assert result == NASDAQ100_HARDCODED
    # Hardcoded list is now persisted.
    assert universe.NASDAQ100_CACHE.exists()


def test_load_nasdaq100_fetches_from_wikipedia_when_cache_missing():
    html = """<table><tr><th>Ticker</th></tr><tr><td>AAPL</td></tr><tr><td>GOOG</td></tr></table>"""
    with patch.object(universe, "_get_url", return_value=html):
        result = load_nasdaq100()
    assert result == ["AAPL", "GOOG"]


# ---------------- expand_universe_tokens ----------------
def test_expand_tokens_passes_explicit_tickers_through():
    assert expand_universe_tokens(["AAPL", "MSFT"]) == ["AAPL", "MSFT"]


def test_expand_tokens_mixed_universe_and_ticker(tmp_path):
    _write_cache(universe.SP500_CACHE, ["GOOG", "MSFT"])
    result = expand_universe_tokens(["AAPL", "SP500"])
    assert result == ["AAPL", "GOOG", "MSFT"]


def test_expand_tokens_dedups_across_index_union(tmp_path):
    _write_cache(universe.SP500_CACHE, ["AAPL", "MSFT"])
    _write_cache(universe.NASDAQ100_CACHE, ["MSFT", "GOOG"])
    result = expand_universe_tokens(["SP500", "NASDAQ100"])
    # MSFT appears in both - should only be listed once.
    assert result == ["AAPL", "MSFT", "GOOG"]


def test_expand_tokens_is_case_insensitive(tmp_path):
    _write_cache(universe.SP500_CACHE, ["AAPL"])
    result = expand_universe_tokens(["sp500"])
    assert result == ["AAPL"]


def test_expand_tokens_skips_failed_expansion_gracefully():
    # No cache, fetch fails - token is simply skipped.
    with patch.object(universe, "_get_url", side_effect=RuntimeError("403")):
        result = expand_universe_tokens(["AAPL", "SP500"])
    assert result == ["AAPL"]  # SP500 dropped but AAPL survived
