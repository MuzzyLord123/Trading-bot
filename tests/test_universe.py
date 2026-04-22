from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bot import universe


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "SP500_CACHE", tmp_path / "sp500.csv")
    yield


def _seed_cache(path: Path, symbols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": symbols}).to_csv(path, index=False)


def test_load_sp500_uses_fresh_cache():
    _seed_cache(universe.SP500_CACHE, ["AAPL", "MSFT", "BRK-B"])
    result = universe.load_sp500()
    assert result == ["AAPL", "MSFT", "BRK-B"]


def test_expand_universe_tokens_leaves_regular_symbols_alone():
    _seed_cache(universe.SP500_CACHE, ["AAPL", "MSFT"])
    expanded = universe.expand_universe_tokens(["VWRL.L", "AMD"])
    assert expanded == ["VWRL.L", "AMD"]


def test_expand_universe_tokens_expands_sp500():
    _seed_cache(universe.SP500_CACHE, ["AAPL", "MSFT", "GOOGL"])
    expanded = universe.expand_universe_tokens(["VWRL.L", "SP500"])
    assert expanded == ["VWRL.L", "AAPL", "MSFT", "GOOGL"]


def test_expand_universe_tokens_deduplicates():
    _seed_cache(universe.SP500_CACHE, ["AAPL", "MSFT", "GOOGL"])
    expanded = universe.expand_universe_tokens(["AAPL", "SP500", "MSFT"])
    # AAPL and MSFT appear both explicitly and inside SP500 – each listed once.
    assert expanded == ["AAPL", "MSFT", "GOOGL"]


def test_expand_universe_tokens_case_insensitive():
    _seed_cache(universe.SP500_CACHE, ["AAPL"])
    assert universe.expand_universe_tokens(["sp500"]) == ["AAPL"]
    assert universe.expand_universe_tokens(["S&P500"]) == ["AAPL"]
