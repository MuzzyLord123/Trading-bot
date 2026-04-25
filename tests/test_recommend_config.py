from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.config import (
    Config,
    ExchangeConfig,
    LoggingConfig,
    NotificationsConfig,
    RiskConfig,
    StrategyConfig,
    TradingConfig,
)
from scripts.recommend_config import PRESETS, _materialise, _rank


def _base() -> Config:
    return Config(
        exchange=ExchangeConfig(),
        trading=TradingConfig(symbols=["AAPL", "MSFT"], starting_capital=10000),
        risk=RiskConfig(),
        strategy=StrategyConfig(),
        logging=LoggingConfig(),
        notifications=NotificationsConfig(),
        secrets={},
    )


def test_every_preset_materialises_to_a_valid_config():
    base = _base()
    for preset in PRESETS:
        cfg = _materialise(base, preset)
        cfg.validate()
        assert cfg.trading.symbols == base.trading.symbols
        assert cfg.trading.starting_capital == base.trading.starting_capital
        assert cfg.strategy.name == preset.strategy["name"]


def test_preset_keys_are_unique():
    keys = [p.key for p in PRESETS]
    assert len(keys) == len(set(keys))


def test_risk_overrides_apply_on_top_of_base():
    base = _base()
    aggressive = next(p for p in PRESETS if p.key == "aggressive_breakout")
    cfg = _materialise(base, aggressive)
    assert cfg.risk.risk_per_trade == 0.015
    assert cfg.risk.max_open_positions == 8
    # Fields not overridden should still match the base.
    assert cfg.risk.taker_fee_pct == base.risk.taker_fee_pct


def test_ranking_prefers_consistency_over_headline_return():
    consistent = {
        "preset": "consistent", "profitable_windows": 4, "n_windows": 4,
        "worst_return_pct": 0.5, "avg_sharpe": 1.5, "avg_return_pct": 8.0,
    }
    lottery = {
        "preset": "lottery", "profitable_windows": 1, "n_windows": 4,
        "worst_return_pct": -25.0, "avg_sharpe": 0.3, "avg_return_pct": 60.0,
    }
    ranked = _rank([lottery, consistent])
    assert ranked[0]["preset"] == "consistent"
    assert ranked[1]["preset"] == "lottery"


def test_ranking_uses_worst_return_as_first_tiebreaker():
    """Two presets equally consistent (3/4 windows positive each).
    The one with the smaller worst-window drawdown should win."""
    floor_minus_2 = {
        "preset": "shallow", "profitable_windows": 3, "n_windows": 4,
        "worst_return_pct": -2.0, "avg_sharpe": 1.0, "avg_return_pct": 5.0,
    }
    floor_minus_15 = {
        "preset": "deep", "profitable_windows": 3, "n_windows": 4,
        "worst_return_pct": -15.0, "avg_sharpe": 1.5, "avg_return_pct": 12.0,
    }
    ranked = _rank([floor_minus_15, floor_minus_2])
    assert ranked[0]["preset"] == "shallow"


def test_ranking_pushes_errored_results_to_bottom():
    good = {
        "preset": "good", "profitable_windows": 2, "n_windows": 4,
        "worst_return_pct": -5.0, "avg_sharpe": 0.5, "avg_return_pct": 3.0,
    }
    broken = {"preset": "broken", "error": "fetch failed"}
    ranked = _rank([broken, good])
    assert ranked[0]["preset"] == "good"
    assert ranked[-1]["preset"] == "broken"
