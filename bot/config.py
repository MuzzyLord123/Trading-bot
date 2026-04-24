from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class ExchangeConfig:
    # Kept for compatibility with older configs; this bot only trades
    # through Trading 212 now, so the only meaningful field is ``sandbox``
    # (demo vs live T212 environment).
    name: str = "trading212"
    sandbox: bool = False


@dataclass
class TradingConfig:
    mode: str = "paper"
    quote_currency: str = "GBP"
    starting_capital: float = 10000.0
    symbols: list[str] = field(default_factory=lambda: ["VUAG.L"])
    timeframe: str = "1h"
    poll_interval_seconds: int = 300
    history_candles: int = 500
    # Tickers to strip out AFTER universe expansion. Useful for pulling
    # SP500 / NASDAQ100 but skipping specific names (illiquid for you,
    # not tradable on T212, sector avoidance, etc.).
    excluded_symbols: list[str] = field(default_factory=list)


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01
    max_position_pct: float = 0.1
    max_open_positions: int = 5
    stop_loss_pct: float = 0.04
    take_profit_pct: float = 0.10
    trailing_stop_pct: float = 0.03
    daily_loss_limit_pct: float = 0.05
    max_drawdown_pct: float = 0.20
    taker_fee_pct: float = 0.0015
    slippage_pct: float = 0.0005
    # Bars to wait before re-entering the same symbol after a losing exit.
    cooldown_bars_after_loss: int = 4
    # Minimum reward:risk ratio required to enter (take_profit/stop distance).
    min_reward_to_risk: float = 1.5
    # Once unrealised profit reaches this % of entry, move stop to breakeven.
    breakeven_trigger_pct: float = 0.02
    # Close position if no net profit after this many bars (0 disables).
    time_stop_bars: int = 48
    # Halve risk per trade while drawdown from peak exceeds this fraction.
    drawdown_risk_reduction_threshold: float = 0.05
    drawdown_risk_reduction_factor: float = 0.5
    # Volatility-adaptive stop: when True, stop distance = atr * multiplier
    # instead of price * stop_loss_pct. Position size scales inversely so
    # cash-at-risk per trade stays constant across volatility regimes.
    use_atr_stop: bool = False
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0
    # Halt trading after this many losing trades in a row (0 = disabled).
    # Complements daily_loss_limit_pct and max_drawdown_pct.
    max_consecutive_losses: int = 0
    # Scale-out take-profit: when price reaches entry + scale_out_at_r * R
    # (where R = initial stop distance), close scale_out_fraction of the
    # position and move the remaining stop to break-even. 0 disables.
    scale_out_at_r: float = 0.0
    scale_out_fraction: float = 0.5
    # Floor for trade notional (quote currency). T212 rejects sub-minimum
    # orders; failing fast locally keeps us from burning rate budget on
    # orders that will bounce. 0 disables.
    min_notional_value: float = 1.0


@dataclass
class StrategyConfig:
    name: str = "ensemble"
    ensemble: dict[str, Any] = field(default_factory=dict)
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    filter: dict[str, Any] = field(default_factory=dict)
    # Opt-in higher-timeframe confirmation. Wraps the (optionally filtered)
    # strategy so a LONG only fires if the same logic also agrees on the
    # HTF. Keys: enabled (bool), rule (e.g. "1D"; auto-derived if None),
    # require_long_on_htf (bool, default True).
    multi_timeframe: dict[str, Any] = field(default_factory=dict)


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
        trading_cfg = TradingConfig(**raw.get("trading", {}))
        # Expand universe tokens ('SP500' etc.) in trading.symbols, then
        # apply the exclusion list. Exclusions are case-insensitive.
        from .universe import expand_universe_tokens
        expanded = expand_universe_tokens(trading_cfg.symbols)
        excluded = {s.upper() for s in trading_cfg.excluded_symbols}
        trading_cfg.symbols = [s for s in expanded if s.upper() not in excluded]
        # Only pass known keys so legacy configs (which may include fields
        # that no longer exist on ExchangeConfig) still load.
        exchange_raw = {
            k: v for k, v in raw.get("exchange", {}).items()
            if k in {"name", "sandbox"}
        }
        cfg = cls(
            exchange=ExchangeConfig(**exchange_raw),
            trading=trading_cfg,
            risk=RiskConfig(**raw.get("risk", {})),
            strategy=StrategyConfig(**raw.get("strategy", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
            notifications=NotificationsConfig(**raw.get("notifications", {})),
            secrets={
                "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
                "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
                "trading212_api_key": os.getenv("TRADING212_API_KEY", ""),
            },
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Raise ValueError if the config contains obviously bad values."""
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
        if r.atr_period < 2:
            errors.append("risk.atr_period must be >= 2")
        if r.atr_stop_multiplier <= 0:
            errors.append("risk.atr_stop_multiplier must be > 0")
        if r.max_consecutive_losses < 0:
            errors.append("risk.max_consecutive_losses must be >= 0")
        if r.scale_out_at_r < 0:
            errors.append("risk.scale_out_at_r must be >= 0")
        if not 0 <= r.scale_out_fraction < 1:
            errors.append("risk.scale_out_fraction must be in [0, 1)")
        if r.min_notional_value < 0:
            errors.append("risk.min_notional_value must be >= 0")

        if self.trading.mode == "live" and not self.secrets.get("trading212_api_key"):
            errors.append("live mode requires TRADING212_API_KEY in .env")

        if self.notifications.telegram and (
            not self.secrets.get("telegram_bot_token")
            or not self.secrets.get("telegram_chat_id")
        ):
            errors.append(
                "notifications.telegram enabled but TELEGRAM_BOT_TOKEN/CHAT_ID missing"
            )

        if errors:
            raise ValueError("Invalid config:\n  - " + "\n  - ".join(errors))
