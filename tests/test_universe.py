from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bot import universe


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "SP500_CACHE", tmp_path / "sp500.csv")
    monkeypatch.setattr(universe, "NASDAQ100_CACHE", tmp_path / "nasdaq100.csv")
    yield


def _seed_cache(path: Path, symbols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": symbols}).to_csv(path, index=False)


def test_load_sp500_uses_fresh_cache():
    _seed_cache(universe.SP500_CACHE, ["AAPL", "MSFT", "BRK-B"])
    result = universe.load_sp500()
    assert result == ["AAPL", "MSFT", "BRK-B"]


def test_load_nasdaq100_uses_fresh_cache():
    _seed_cache(universe.NASDAQ100_CACHE, ["AAPL", "MSFT", "NVDA"])
    assert universe.load_nasdaq100() == ["AAPL", "MSFT", "NVDA"]


def test_load_nasdaq100_falls_back_to_hardcoded(monkeypatch):
    # No cache, no network -> must use hardcoded canonical list.
    def _boom(url: str) -> str:
        raise RuntimeError("network down")

    monkeypatch.setattr(universe, "_get_url", _boom)
    result = universe.load_nasdaq100()
    assert len(result) == len(universe.NASDAQ100_HARDCODED)
    # Spot-check a handful of expected names.
    for expected in ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "META", "AMZN"]:
        assert expected in result


def test_nasdaq100_hardcoded_has_no_duplicates():
    # Regression: duplicates would break dedup logic in expand_universe_tokens.
    assert len(universe.NASDAQ100_HARDCODED) == len(set(universe.NASDAQ100_HARDCODED))


def test_nasdaq100_hardcoded_size_sane():
    # The index name is Nasdaq-100 for a reason.
    assert 90 <= len(universe.NASDAQ100_HARDCODED) <= 110


def test_expand_universe_tokens_leaves_regular_symbols_alone():
    _seed_cache(universe.SP500_CACHE, ["AAPL", "MSFT"])
    expanded = universe.expand_universe_tokens(["VWRL.L", "AMD"])
    assert expanded == ["VWRL.L", "AMD"]


def test_expand_universe_tokens_expands_sp500():
    _seed_cache(universe.SP500_CACHE, ["AAPL", "MSFT", "GOOGL"])
    expanded = universe.expand_universe_tokens(["VWRL.L", "SP500"])
    assert expanded == ["VWRL.L", "AAPL", "MSFT", "GOOGL"]


def test_expand_universe_tokens_expands_nasdaq100():
    _seed_cache(universe.NASDAQ100_CACHE, ["NVDA", "TSLA", "AVGO"])
    expanded = universe.expand_universe_tokens(["VWRL.L", "NASDAQ100"])
    assert expanded == ["VWRL.L", "NVDA", "TSLA", "AVGO"]


def test_expand_universe_tokens_deduplicates_across_indices():
    _seed_cache(universe.SP500_CACHE, ["AAPL", "MSFT", "GOOGL", "JPM"])
    _seed_cache(universe.NASDAQ100_CACHE, ["AAPL", "MSFT", "NVDA", "TSLA"])
    expanded = universe.expand_universe_tokens(["SP500", "NASDAQ100"])
    # AAPL/MSFT appear in both indices – listed once each.
    assert expanded == ["AAPL", "MSFT", "GOOGL", "JPM", "NVDA", "TSLA"]


def test_expand_universe_tokens_deduplicates():
    _seed_cache(universe.SP500_CACHE, ["AAPL", "MSFT", "GOOGL"])
    expanded = universe.expand_universe_tokens(["AAPL", "SP500", "MSFT"])
    # AAPL and MSFT appear both explicitly and inside SP500 – each listed once.
    assert expanded == ["AAPL", "MSFT", "GOOGL"]


def test_expand_universe_tokens_case_insensitive():
    _seed_cache(universe.SP500_CACHE, ["AAPL"])
    _seed_cache(universe.NASDAQ100_CACHE, ["NVDA"])
    assert universe.expand_universe_tokens(["sp500"]) == ["AAPL"]
    assert universe.expand_universe_tokens(["S&P500"]) == ["AAPL"]
    assert universe.expand_universe_tokens(["nasdaq100"]) == ["NVDA"]
    assert universe.expand_universe_tokens(["NDX"]) == ["NVDA"]
