"""Engine-side tests for the severity gate around _notify."""
from __future__ import annotations

from unittest.mock import MagicMock

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
from bot.engine import TradingEngine
from bot.risk import RiskManager
from bot.strategies import MaCrossoverStrategy


def _engine(desktop_severity: str = "warnings", desktop_enabled: bool = True) -> TradingEngine:
    cfg = Config(
        exchange=ExchangeConfig(),
        trading=TradingConfig(symbols=["AAPL"]),
        risk=RiskConfig(),
        strategy=StrategyConfig(),
        logging=LoggingConfig(),
        notifications=NotificationsConfig(
            telegram=False, desktop=desktop_enabled,
            desktop_min_severity=desktop_severity,
        ),
        secrets={},
    )
    eng = TradingEngine(cfg, exchange=MagicMock(), strategy=MaCrossoverStrategy(), risk=RiskManager(cfg.risk))
    # Replace the real desktop notifier with a mock so we can assert calls.
    eng.desktop = MagicMock()
    eng.notifier = MagicMock()
    return eng


def test_trades_severity_gate_at_default_warnings_blocks_trade_popups():
    eng = _engine(desktop_severity="warnings")
    eng._notify("OPEN AAPL", severity="trades")
    eng.desktop.send.assert_not_called()
    # Telegram-equivalent (notifier) still gets the message.
    eng.notifier.send.assert_called_once_with("OPEN AAPL")


def test_warnings_severity_passes_at_default():
    eng = _engine(desktop_severity="warnings")
    eng._notify("Reconciled", severity="warnings", title="Reconciliation")
    eng.desktop.send.assert_called_once()
    eng.notifier.send.assert_called_once()


def test_errors_severity_always_passes():
    eng = _engine(desktop_severity="errors")
    eng._notify("Halted", severity="errors", title="HALT")
    eng.desktop.send.assert_called_once()
    eng.notifier.send.assert_called_once()


def test_errors_threshold_blocks_warnings():
    eng = _engine(desktop_severity="errors")
    eng._notify("Reconciled", severity="warnings")
    eng.desktop.send.assert_not_called()


def test_trades_threshold_passes_everything():
    eng = _engine(desktop_severity="trades")
    eng._notify("OPEN", severity="trades")
    eng._notify("Halted", severity="errors")
    assert eng.desktop.send.call_count == 2


def test_desktop_disabled_never_pops_regardless_of_severity():
    eng = _engine(desktop_enabled=False)
    eng.desktop = None
    eng._notify("OPEN", severity="trades")
    eng._notify("Halted", severity="errors")
    # Should still send via remote notifier (Telegram-like).
    assert eng.notifier.send.call_count == 2


def test_desktop_failure_does_not_crash_notify():
    eng = _engine()
    eng.desktop.send.side_effect = RuntimeError("display server gone")
    # Should not raise.
    eng._notify("Halted", severity="errors")
    eng.notifier.send.assert_called_once()


def test_remote_failure_does_not_block_desktop():
    eng = _engine()
    eng.notifier.send.side_effect = RuntimeError("telegram gone")
    eng._notify("Halted", severity="errors")
    eng.desktop.send.assert_called_once()
