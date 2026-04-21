from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class ExchangeConfig:
    name: str = "kraken"
    requires_password: bool = False
    sandbox: bool = False
    enable_rate_limit: bool = True


@dataclass
class TradingConfig:
    mode: str = "paper"
    quote_currency: str = "GBP"
    starting_capital: float = 500.0
    symbols: list[str] = field(default_factory=lambda: ["BTC/GBP"])
    timeframe: str = "1h"
    poll_interval_seconds: int = 60
    history_candles: int = 500


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01
    max_position_pct: float = 0.25
    max_open_positions: int = 3
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    trailing_stop_pct: float = 0.02
    daily_loss_limit_pct: float = 0.05
    max_drawdown_pct: float = 0.20
    taker_fee_pct: float = 0.0026
    slippage_pct: float = 0.0005
    # Bars to wait before re-entering the same symbol after a losing exit.
    cooldown_bars_after_loss: int = 12
    # Minimum reward:risk ratio required to enter (take_profit/stop distance).
    min_reward_to_risk: float = 1.5
    # Once unrealised profit reaches this % of entry, move stop to breakeven.
    breakeven_trigger_pct: float = 0.02
    # Close position if no net profit after this many bars (0 disables).
    time_stop_bars: int = 48
    # Halve risk per trade while drawdown from peak exceeds this fraction.
    drawdown_risk_reduction_threshold: float = 0.05
    drawdown_risk_reduction_factor: float = 0.5


@dataclass
class StrategyConfig:
    name: str = "ensemble"
    ensemble: dict[str, Any] = field(default_factory=dict)
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    filter: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    trade_log: str = "logs/trades.csv"
    equity_log: str = "logs/equity.csv"


@dataclass
class NotificationsConfig:
    telegram: bool = False


@dataclass
class Config:
    exchange: ExchangeConfig
    trading: TradingConfig
    risk: RiskConfig
    strategy: StrategyConfig
    logging: LoggingConfig
    notifications: NotificationsConfig
    secrets: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        load_dotenv()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Config file '{p}' not found. Copy config.example.yaml to config.yaml."
            )
        raw = yaml.safe_load(p.read_text()) or {}
        return cls(
            exchange=ExchangeConfig(**raw.get("exchange", {})),
            trading=TradingConfig(**raw.get("trading", {})),
            risk=RiskConfig(**raw.get("risk", {})),
            strategy=StrategyConfig(**raw.get("strategy", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
            notifications=NotificationsConfig(**raw.get("notifications", {})),
            secrets={
                "api_key": os.getenv("EXCHANGE_API_KEY", ""),
                "api_secret": os.getenv("EXCHANGE_API_SECRET", ""),
                "api_password": os.getenv("EXCHANGE_API_PASSWORD", ""),
                "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
                "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
            },
        )
