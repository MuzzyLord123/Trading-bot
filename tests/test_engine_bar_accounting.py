"""Engine bars_held and cooldown should advance per bar, not per tick.

Before the fix, a 5-minute poll on an hourly timeframe made 24 "bars" of
time stop expire in 2 hours; these tests pin the corrected semantics.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from bot.config import (
    Config,
    ExchangeConfig,
    LoggingConfig,
    NotificationsConfig,
    RiskConfig,
    StrategyConfig,
    TradingConfig,
)
from bot.engine import TradingEngine
from bot.portfolio import Position
from bot.risk import RiskManager
from bot.strategies import MaCrossoverStrategy


def _cfg() -> Config:
    return Config(
        exchange=ExchangeConfig(),
        trading=TradingConfig(
            mode="paper", symbols=["FAKE"], timeframe="1h",
            poll_interval_seconds=300, history_candles=100,
        ),
        risk=RiskConfig(
            risk_per_trade=0.01, max_position_pct=1.0, max_open_positions=1,
            stop_loss_pct=0.05, take_profit_pct=0.2, trailing_stop_pct=0.0,
            daily_loss_limit_pct=0.9, max_drawdown_pct=0.9,
            taker_fee_pct=0.0, slippage_pct=0.0,
            cooldown_bars_after_loss=3, time_stop_bars=5,
            min_notional_value=0.0,
        ),
        strategy=StrategyConfig(name="ma_crossover"),
        logging=LoggingConfig(),
        notifications=NotificationsConfig(),
        secrets={},
    )


def _candles(n: int, base_hours_ago: int = 100) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = np.linspace(100.0, 100.0 + n * 0.01, n)
    return pd.DataFrame({
        "timestamp": ts,
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.full(n, 1.0),
    })


def _make_engine(cfg):
    exchange = MagicMock()
    exchange.load_markets.return_value = {"FAKE": {}}
    exchange.amount_to_precision = lambda s, a: round(a, 4)
    exchange.market_min_amount = lambda s: 0.0
    eng = TradingEngine(
        cfg=cfg, exchange=exchange,
        strategy=MaCrossoverStrategy(fast=2, slow=3),
        risk=RiskManager(cfg.risk),
    )
    return eng, exchange


def test_bars_held_advances_once_per_new_candle():
    cfg = _cfg()
    eng, exchange = _make_engine(cfg)
    df = _candles(50)

    # Seed an open position so _bars_held tracking matters.
    eng.portfolio.positions["FAKE"] = Position(
        symbol="FAKE", side="long", amount=1.0, entry_price=100.0,
        stop_loss=50.0, take_profit=500.0,
        peak_price=100.0, trailing_stop_pct=0.0,
        opened_at=datetime.now(timezone.utc),
    )
    # Pre-seed last-seen timestamp so the first tick registers bar N as
    # the "same bar" as the open — matches how engine records an entry.
    eng._last_candle_ts["FAKE"] = df["timestamp"].iloc[-1]

    with patch("bot.engine.is_stock_market_open", return_value=True):
        # 5 consecutive polls of the SAME candle => bars_held must stay 0.
        for _ in range(5):
            exchange.fetch_ohlcv.return_value = df
            eng.tick()
            if "FAKE" not in eng.portfolio.positions:
                break
        assert eng._bars_held.get("FAKE", 0) == 0

        # Now a new candle arrives; bars_held ticks by 1 once.
        df2 = _candles(51)
        exchange.fetch_ohlcv.return_value = df2
        eng.tick()
    assert eng._bars_held.get("FAKE", 0) == 1


def test_cooldown_decays_per_bar_not_per_tick():
    cfg = _cfg()
    eng, exchange = _make_engine(cfg)
    df = _candles(40)
    eng._last_candle_ts["FAKE"] = df["timestamp"].iloc[-1]
    eng._cooldown_ticks_left["FAKE"] = 3

    with patch("bot.engine.is_stock_market_open", return_value=True):
        # Many polls of the SAME candle should NOT decay cooldown.
        for _ in range(10):
            exchange.fetch_ohlcv.return_value = df
            eng.tick()
        assert eng._cooldown_ticks_left.get("FAKE", 0) == 3

        # One new bar decays by 1.
        df2 = _candles(41)
        exchange.fetch_ohlcv.return_value = df2
        eng.tick()
    assert eng._cooldown_ticks_left.get("FAKE", 0) == 2
