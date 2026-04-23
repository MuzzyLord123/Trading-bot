"""Tests for the second round of improvements.

Covers:
  * BacktestResult: benchmark return, excess return, ulcer index, streaks.
  * Config: excluded_symbols filter; min_notional_value validation.
  * Engine/Backtest: min_notional_value gates tiny orders out.
  * stocks._is_transient_http: retry policy on HTTP errors.
"""
from __future__ import annotations

import urllib.error

import numpy as np
import pandas as pd
import pytest

from bot.backtest import BacktestResult, Backtester, _buy_and_hold_return, _streaks
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
from bot.stocks import _is_transient_http
from bot.strategies import MaCrossoverStrategy


def _synthetic(n: int = 400, drift: float = 0.5) -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = np.linspace(100.0, 100.0 + drift * n, n)
    return pd.DataFrame({
        "timestamp": t,
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.ones(n),
    })


def _cfg(**overrides) -> Config:
    risk_kwargs = dict(
        risk_per_trade=0.02, max_position_pct=0.5, max_open_positions=1,
        stop_loss_pct=0.03, take_profit_pct=0.06, trailing_stop_pct=0.0,
        daily_loss_limit_pct=0.5, max_drawdown_pct=0.9,
        taker_fee_pct=0.001, slippage_pct=0.0,
    )
    risk_kwargs.update(overrides.pop("risk", {}))
    return Config(
        exchange=ExchangeConfig(),
        trading=TradingConfig(
            mode="paper", symbols=["FAKE"], timeframe="1h",
            starting_capital=500.0, history_candles=400,
        ),
        risk=RiskConfig(**risk_kwargs),
        strategy=StrategyConfig(name="ma_crossover"),
        logging=LoggingConfig(),
        notifications=NotificationsConfig(),
        secrets={},
    )


# ---------------------------------------------------------------------------
# Streaks.
# ---------------------------------------------------------------------------
def test_streaks_count_longest_runs():
    # W W L W W W L L => longest win=3, longest loss=2
    assert _streaks([1, 1, -1, 1, 1, 1, -1, -1]) == (3, 2)


def test_streaks_handle_all_wins():
    assert _streaks([1, 2, 3]) == (3, 0)


def test_streaks_handle_all_losses():
    assert _streaks([-1, 0, -2]) == (0, 3)  # zero counts as loss


def test_streaks_empty():
    assert _streaks([]) == (0, 0)


# ---------------------------------------------------------------------------
# Buy-and-hold benchmark.
# ---------------------------------------------------------------------------
def test_buy_and_hold_equal_weights_universe():
    a = _synthetic(n=50, drift=0.0)  # flat => 0% return
    b = _synthetic(n=50, drift=2.0)  # end price 200 => 100% return
    ret = _buy_and_hold_return({"a": a, "b": b})
    assert 0.49 < ret < 0.51  # (0 + ~1) / 2


def test_buy_and_hold_empty():
    assert _buy_and_hold_return({}) == 0.0


# ---------------------------------------------------------------------------
# Backtester.run populates benchmark_return + new stats.
# ---------------------------------------------------------------------------
def test_backtest_stats_include_new_fields():
    cfg = _cfg()
    strat = MaCrossoverStrategy(fast=10, slow=30)
    bt = Backtester(cfg, exchange=None, strategy=strat, risk=RiskManager(cfg.risk))
    data = {"FAKE": _synthetic(400, drift=0.2)}
    result = bt.run(data=data, write_csv=False)

    assert result.benchmark_return > 0.0  # uptrend should have positive BH
    stats = result.stats(cfg.trading.starting_capital)
    for key in (
        "benchmark_return_pct", "excess_return_pct",
        "ulcer_index", "longest_win_streak", "longest_loss_streak",
        "median_trade_pnl",
    ):
        assert key in stats


# ---------------------------------------------------------------------------
# Config: excluded_symbols + min_notional_value validation.
# ---------------------------------------------------------------------------
def test_trading_config_excluded_symbols_defaults_empty():
    tc = TradingConfig(mode="paper", symbols=["AAPL"])
    assert tc.excluded_symbols == []


def test_config_rejects_negative_min_notional():
    cfg = Config(
        exchange=ExchangeConfig(),
        trading=TradingConfig(mode="paper", symbols=["AAPL"]),
        risk=RiskConfig(min_notional_value=-1.0),
        strategy=StrategyConfig(),
        logging=LoggingConfig(),
        notifications=NotificationsConfig(),
        secrets={},
    )
    with pytest.raises(ValueError, match="min_notional_value"):
        cfg.validate()


def test_config_load_applies_excluded_symbols(tmp_path, monkeypatch):
    # Make universe expansion deterministic: no network.
    from bot import universe
    monkeypatch.setattr(universe, "load_sp500", lambda: ["AAPL", "MSFT", "NVDA"])
    monkeypatch.setattr(universe, "load_nasdaq100", lambda: ["AAPL", "GOOGL"])

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "exchange: {name: trading212, sandbox: false}\n"
        "trading:\n"
        "  mode: paper\n"
        "  symbols: [SP500]\n"
        "  excluded_symbols: [MSFT, nvda]\n"  # case-insensitive
        "risk: {}\n"
        "strategy: {name: ensemble}\n"
        "logging: {}\n"
        "notifications: {}\n"
    )
    cfg = Config.load(cfg_path)
    assert set(cfg.trading.symbols) == {"AAPL"}


# ---------------------------------------------------------------------------
# min_notional_value in the backtester.
# ---------------------------------------------------------------------------
def test_min_notional_blocks_small_orders():
    # 100% risk over a huge stop would normally mean a small amount. Make
    # it extreme so the gate is guaranteed to fire.
    cfg = _cfg(risk={"min_notional_value": 1_000_000.0})
    strat = MaCrossoverStrategy(fast=10, slow=30)
    bt = Backtester(cfg, exchange=None, strategy=strat, risk=RiskManager(cfg.risk))
    data = {"FAKE": _synthetic(400, drift=0.1)}
    result = bt.run(data=data, write_csv=False)
    # With an insane notional floor no trade can open.
    assert len(result.trades) == 0


# ---------------------------------------------------------------------------
# Transient HTTP retry policy.
# ---------------------------------------------------------------------------
def test_transient_retries_on_429_and_5xx():
    assert _is_transient_http(urllib.error.HTTPError(
        "u", 429, "Too Many Requests", None, None))
    assert _is_transient_http(urllib.error.HTTPError(
        "u", 503, "Service Unavailable", None, None))


def test_transient_does_not_retry_4xx():
    for code in (400, 401, 403, 404, 422):
        assert not _is_transient_http(urllib.error.HTTPError(
            "u", code, "Bad", None, None))


def test_transient_retries_connection_errors():
    assert _is_transient_http(ConnectionError("nope"))
    assert _is_transient_http(TimeoutError())
    assert _is_transient_http(urllib.error.URLError("dns"))


def test_transient_ignores_unrelated_exceptions():
    assert not _is_transient_http(ValueError("programmer error"))
