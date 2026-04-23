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


def _synthetic_data(n: int = 400) -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    trend = np.linspace(100, 180, n)
    noise = np.sin(np.linspace(0, 20, n)) * 3
    close = trend + noise
    return pd.DataFrame(
        {
            "timestamp": t,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones(n),
        }
    )


def _cfg() -> Config:
    return Config(
        exchange=ExchangeConfig(name="trading212"),
        trading=TradingConfig(
            mode="paper",
            quote_currency="GBP",
            starting_capital=500.0,
            symbols=["VUAG.L"],
            timeframe="1h",
            history_candles=400,
        ),
        risk=RiskConfig(
            risk_per_trade=0.02,
            max_position_pct=0.5,
            max_open_positions=1,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            trailing_stop_pct=0.0,
            daily_loss_limit_pct=0.5,
            max_drawdown_pct=0.9,
            taker_fee_pct=0.001,
            slippage_pct=0.0,
        ),
        strategy=StrategyConfig(name="ma_crossover"),
        logging=LoggingConfig(),
        notifications=NotificationsConfig(),
        secrets={"api_key": "", "api_secret": "", "api_password": ""},
    )


def test_backtest_runs_on_synthetic_uptrend():
    cfg = _cfg()
    strat = MaCrossoverStrategy(fast=10, slow=30)
    bt = Backtester(cfg, exchange=None, strategy=strat, risk=RiskManager(cfg.risk))
    data = {"VUAG.L": _synthetic_data(400)}
    result = bt.run(data=data, write_csv=False)
    stats = result.stats(cfg.trading.starting_capital)
    assert "total_return_pct" in stats
    assert result.equity_curve.shape[0] > 0
