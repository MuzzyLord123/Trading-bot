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
class NewsConfig:
    """Live-mode only: act on major news headlines."""
    enabled: bool = False
    # "defensive": only close positions on bad news.
    # "aggressive": also open positions on good news (see docs).
    mode: str = "defensive"
    major_threshold: float = 5.0
    max_age_minutes: int = 120
    # Block new entries for this many seconds after a bad-news close.
    blackout_seconds: int = 14400  # 4 hours
    # Only act on news that fires at least this many polls before open.
    min_confirmations: int = 1


@dataclass
class Config:
    exchange: ExchangeConfig
    trading: TradingConfig
    risk: RiskConfig
    strategy: StrategyConfig
    logging: LoggingConfig
    notifications: NotificationsConfig
    news: NewsConfig = field(default_factory=NewsConfig)
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
        trading_cfg = TradingConfig(**raw.get("trading", {}))
        # Expand universe tokens ('SP500' etc.) in trading.symbols.
        from .universe import expand_universe_tokens
        trading_cfg.symbols = expand_universe_tokens(trading_cfg.symbols)
        cfg = cls(
            exchange=ExchangeConfig(**raw.get("exchange", {})),
            trading=trading_cfg,
            risk=RiskConfig(**raw.get("risk", {})),
            strategy=StrategyConfig(**raw.get("strategy", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
            notifications=NotificationsConfig(**raw.get("notifications", {})),
            news=NewsConfig(**raw.get("news", {})),
            secrets={
                "api_key": os.getenv("EXCHANGE_API_KEY", ""),
                "api_secret": os.getenv("EXCHANGE_API_SECRET", ""),
                "api_password": os.getenv("EXCHANGE_API_PASSWORD", ""),
                "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
                "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
                "trading212_api_key": os.getenv("TRADING212_API_KEY", ""),
            },
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Raise ValueError if the config contains obviously bad values.

        Catches common misconfigurations (empty symbol list, out-of-range
        percentages, missing live-mode secrets) before the engine starts
        placing orders with nonsensical parameters.
        """
        errors: list[str] = []

        if self.trading.mode not in ("paper", "live"):
            errors.append(f"trading.mode must be 'paper' or 'live', got '{self.trading.mode}'")
        if not self.trading.symbols:
            errors.append("trading.symbols must not be empty")
        if self.trading.starting_capital <= 0:
            errors.append("trading.starting_capital must be > 0")
        if self.trading.poll_interval_seconds <= 0:
            errors.append("trading.poll_interval_seconds must be > 0")
        if self.trading.history_candles <= 0:
            errors.append("trading.history_candles must be > 0")

        r = self.risk
        fractions = {
            "risk_per_trade": r.risk_per_trade,
            "max_position_pct": r.max_position_pct,
            "stop_loss_pct": r.stop_loss_pct,
            "daily_loss_limit_pct": r.daily_loss_limit_pct,
            "max_drawdown_pct": r.max_drawdown_pct,
        }
        for name, value in fractions.items():
            if not 0 < value < 1:
                errors.append(f"risk.{name} must be in (0, 1), got {value}")
        non_negative = {
            "take_profit_pct": r.take_profit_pct,
            "trailing_stop_pct": r.trailing_stop_pct,
            "taker_fee_pct": r.taker_fee_pct,
            "slippage_pct": r.slippage_pct,
            "breakeven_trigger_pct": r.breakeven_trigger_pct,
            "min_reward_to_risk": r.min_reward_to_risk,
        }
        for name, value in non_negative.items():
            if value < 0:
                errors.append(f"risk.{name} must be >= 0, got {value}")
        if r.max_open_positions <= 0:
            errors.append("risk.max_open_positions must be > 0")
        if r.cooldown_bars_after_loss < 0:
            errors.append("risk.cooldown_bars_after_loss must be >= 0")
        if r.time_stop_bars < 0:
            errors.append("risk.time_stop_bars must be >= 0")

        if self.trading.mode == "live":
            if not self.secrets.get("api_key") or not self.secrets.get("api_secret"):
                errors.append(
                    "live mode requires EXCHANGE_API_KEY and EXCHANGE_API_SECRET in env"
                )
            if self.exchange.requires_password and not self.secrets.get("api_password"):
                errors.append("exchange requires password but EXCHANGE_API_PASSWORD is empty")

        if self.notifications.telegram and (
            not self.secrets.get("telegram_bot_token")
            or not self.secrets.get("telegram_chat_id")
        ):
            errors.append(
                "notifications.telegram enabled but TELEGRAM_BOT_TOKEN/CHAT_ID missing"
            )

        if errors:
            raise ValueError("Invalid config:\n  - " + "\n  - ".join(errors))
