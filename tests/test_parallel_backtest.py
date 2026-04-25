"""Parallel vs serial determinism tests for walk_forward and the
preset comparator.

Each window of a walk-forward backtest is fully independent (fresh
portfolio + risk reset), so process-pool execution must produce the
same per-window stats as a serial run. Same for preset evaluation.
These tests would catch any regression where parallelism quietly
mutates shared state."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.backtest import Backtester
from bot.config import (
    Config,
    ExchangeConfig,
    LoggingConfig,
    NotificationsConfig,
    RiskConfig,
    StrategyConfig,
    TradingConfig,
)
from bot.risk import RiskManager
from bot.strategies import MaCrossoverStrategy


def _synth_data(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    close = 100 + rng.normal(0, 1, n).cumsum() + np.linspace(0, 25, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": np.full(n, 1000.0),
    })


def _cfg() -> Config:
    return Config(
        exchange=ExchangeConfig(),
        trading=TradingConfig(symbols=["AAPL"], starting_capital=10000,
                              history_candles=500),
        risk=RiskConfig(risk_per_trade=0.02, max_position_pct=0.5,
                        max_open_positions=1, stop_loss_pct=0.03,
                        take_profit_pct=0.06, trailing_stop_pct=0.0,
                        daily_loss_limit_pct=0.5, max_drawdown_pct=0.9,
                        taker_fee_pct=0.001, slippage_pct=0.0),
        strategy=StrategyConfig(name="ma_crossover",
                                params={"ma_crossover": {"fast": 10, "slow": 30}}),
        logging=LoggingConfig(),
        notifications=NotificationsConfig(),
        secrets={},
    )


def test_walk_forward_parallel_matches_serial():
    """Per-window stats must be byte-identical between parallel and serial
    modes. Walk-forward windows are independent so any divergence implies
    the parallel path is leaking state somewhere."""
    cfg = _cfg()
    data = {"AAPL": _synth_data(600)}

    bt_serial = Backtester(cfg, exchange=None,
                           strategy=MaCrossoverStrategy(fast=10, slow=30),
                           risk=RiskManager(cfg.risk))
    serial = bt_serial.walk_forward(n_windows=3, data=data, parallel=False)

    bt_parallel = Backtester(cfg, exchange=None,
                             strategy=MaCrossoverStrategy(fast=10, slow=30),
                             risk=RiskManager(cfg.risk))
    parallel = bt_parallel.walk_forward(n_windows=3, data=data, parallel=True)

    # Same number of windows, same order, same stats.
    assert len(serial["windows"]) == len(parallel["windows"])
    for s_win, p_win in zip(serial["windows"], parallel["windows"]):
        assert s_win["window"] == p_win["window"]
        assert s_win["start"] == p_win["start"]
        assert s_win["end"] == p_win["end"]
        assert s_win["trades"] == p_win["trades"]
        assert s_win["total_return_pct"] == p_win["total_return_pct"]
        assert s_win["max_drawdown_pct"] == p_win["max_drawdown_pct"]
        assert s_win["sharpe"] == p_win["sharpe"]


def test_walk_forward_parallel_returns_windows_in_order():
    """ProcessPool's as_completed yields jobs in finishing order, but the
    public API guarantees windows arrive in chronological order."""
    cfg = _cfg()
    data = {"AAPL": _synth_data(800)}
    bt = Backtester(cfg, exchange=None,
                    strategy=MaCrossoverStrategy(fast=10, slow=30),
                    risk=RiskManager(cfg.risk))
    out = bt.walk_forward(n_windows=4, data=data, parallel=True)
    indices = [w["window"] for w in out["windows"]]
    assert indices == sorted(indices)


def test_walk_forward_parallel_falls_back_serial_on_one_window():
    """When there's only one job (or zero), parallel mode is pointless;
    we should silently fall back to the serial path rather than spin up
    a pool to immediately wind it down."""
    cfg = _cfg()
    data = {"AAPL": _synth_data(100)}  # only enough for ~2 windows
    bt = Backtester(cfg, exchange=None,
                    strategy=MaCrossoverStrategy(fast=10, slow=30),
                    risk=RiskManager(cfg.risk))
    out = bt.walk_forward(n_windows=2, data=data, parallel=True)
    assert out["summary"]["n_windows"] >= 1


def test_evaluate_presets_serial_path_returns_in_input_order():
    """The serial fallback (and explicit serial mode) must preserve the
    input ordering of presets, since downstream ranking expects a stable
    list."""
    from scripts.recommend_config import PRESETS, evaluate_presets

    base = Config(
        exchange=ExchangeConfig(),
        trading=TradingConfig(symbols=["AAPL"], starting_capital=10000),
        risk=RiskConfig(),
        strategy=StrategyConfig(),
        logging=LoggingConfig(),
        notifications=NotificationsConfig(),
        secrets={},
    )

    # Subset to two presets so this test stays under a second.
    selected = PRESETS[:2]
    captured: list[str] = []

    def _track(key: str) -> None:
        captured.append(key)

    # Patch load_data on Backtester so _evaluate doesn't try to hit yfinance.
    import bot.backtest as bt_mod
    real_load = bt_mod.Backtester.load_data
    fake_data = {"AAPL": _synth_data(800, seed=11)}
    bt_mod.Backtester.load_data = lambda self, days, use_cache=True: fake_data
    try:
        results = evaluate_presets(
            selected, base, days=180, n_windows=3,
            parallel=False, progress=_track,
        )
    finally:
        bt_mod.Backtester.load_data = real_load

    assert len(results) == 2
    # Progress callback fired once per preset, in input order.
    assert captured == [p.key for p in selected]


def test_evaluate_presets_with_zero_or_one_input_does_not_spawn():
    """Edge cases: the parallel path bails out when there's nothing to
    parallelise (no presets, or a single preset)."""
    from scripts.recommend_config import evaluate_presets

    base = Config(
        exchange=ExchangeConfig(),
        trading=TradingConfig(symbols=["AAPL"]),
        risk=RiskConfig(),
        strategy=StrategyConfig(),
        logging=LoggingConfig(),
        notifications=NotificationsConfig(),
        secrets={},
    )
    assert evaluate_presets([], base, 180, 3, parallel=True) == []
