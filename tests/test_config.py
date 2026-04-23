from __future__ import annotations

import pytest

from bot.config import (
    Config,
    ExchangeConfig,
    LoggingConfig,
    NotificationsConfig,
    RiskConfig,
    StrategyConfig,
    TradingConfig,
)


def _make(**overrides) -> Config:
    trading = overrides.pop("trading", TradingConfig(mode="paper", symbols=["BTC/GBP"]))
    risk = overrides.pop("risk", RiskConfig())
    return Config(
        exchange=overrides.pop("exchange", ExchangeConfig()),
        trading=trading,
        risk=risk,
        strategy=overrides.pop("strategy", StrategyConfig()),
        logging=overrides.pop("logging", LoggingConfig()),
        notifications=overrides.pop("notifications", NotificationsConfig()),
        secrets=overrides.pop("secrets", {}),
    )


def test_validate_accepts_sensible_paper_config():
    _make().validate()


def test_validate_rejects_empty_symbols():
    cfg = _make(trading=TradingConfig(mode="paper", symbols=[]))
    with pytest.raises(ValueError, match="trading.symbols"):
        cfg.validate()


def test_validate_rejects_bad_mode():
    cfg = _make(trading=TradingConfig(mode="shadow", symbols=["BTC/GBP"]))
    with pytest.raises(ValueError, match="trading.mode"):
        cfg.validate()


def test_validate_rejects_out_of_range_stop_loss():
    cfg = _make(risk=RiskConfig(stop_loss_pct=1.5))
    with pytest.raises(ValueError, match="stop_loss_pct"):
        cfg.validate()


def test_validate_rejects_zero_max_open_positions():
    cfg = _make(risk=RiskConfig(max_open_positions=0))
    with pytest.raises(ValueError, match="max_open_positions"):
        cfg.validate()


def test_validate_requires_secrets_in_live_mode():
    cfg = _make(trading=TradingConfig(mode="live", symbols=["BTC/GBP"]), secrets={})
    with pytest.raises(ValueError, match="EXCHANGE_API_KEY"):
        cfg.validate()


def test_validate_live_mode_ok_with_secrets():
    cfg = _make(
        trading=TradingConfig(mode="live", symbols=["BTC/GBP"]),
        secrets={"api_key": "k", "api_secret": "s"},
    )
    cfg.validate()
