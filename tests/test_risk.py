from __future__ import annotations

from datetime import datetime, timezone

from bot.config import RiskConfig
from bot.portfolio import Portfolio, Position
from bot.risk import RiskManager


def _cfg(**overrides):
    base = dict(
        risk_per_trade=0.01,
        max_position_pct=0.25,
        max_open_positions=3,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        trailing_stop_pct=0.02,
        daily_loss_limit_pct=0.05,
        max_drawdown_pct=0.20,
        taker_fee_pct=0.001,
        slippage_pct=0.0005,
    )
    base.update(overrides)
    return RiskConfig(**base)


def test_size_respects_risk_per_trade():
    rm = RiskManager(_cfg(max_position_pct=1.0))
    result = rm.size("long", price=100.0, equity=500.0, cash=500.0)
    assert result.amount > 0
    expected_risk = 500 * 0.01
    realised_risk = result.amount * (100 - result.stop_loss)
    assert abs(realised_risk - expected_risk) < 1e-6


def test_size_capped_by_max_position_pct():
    rm = RiskManager(_cfg(risk_per_trade=1.0, max_position_pct=0.1))
    result = rm.size("long", price=100.0, equity=1000.0, cash=1000.0)
    assert result.amount * 100.0 <= 100.0 + 1e-9


def test_should_exit_stop_and_target():
    rm = RiskManager(_cfg())
    pos = Position(
        symbol="BTC/GBP",
        side="long",
        amount=1.0,
        entry_price=100.0,
        stop_loss=97.0,
        take_profit=106.0,
        peak_price=100.0,
        trailing_stop_pct=0.02,
        opened_at=datetime.now(timezone.utc),
    )
    assert rm.should_exit(pos, 96.0) == "stop_loss"
    assert rm.should_exit(pos, 106.5) == "take_profit"
    assert rm.should_exit(pos, 101.0) is None


def test_trading_halted_on_daily_loss():
    rm = RiskManager(_cfg(daily_loss_limit_pct=0.05))
    p = Portfolio.new(500.0)
    assert rm.trading_halted(p, 500.0) is None
    assert rm.trading_halted(p, 470.0) is not None


def test_trading_halted_on_max_drawdown():
    rm = RiskManager(_cfg(max_drawdown_pct=0.10))
    p = Portfolio.new(500.0)
    p.peak_equity = 600.0
    assert rm.trading_halted(p, 600.0) is None
    assert rm.trading_halted(p, 530.0) is not None


def test_size_rejects_stop_pct_out_of_range():
    rm = RiskManager(_cfg(stop_loss_pct=1.5))
    result = rm.size("long", price=100.0, equity=500.0, cash=500.0)
    assert result.amount == 0.0


def test_size_enforces_stop_below_entry_below_tp():
    rm = RiskManager(_cfg())
    result = rm.size("long", price=100.0, equity=500.0, cash=500.0)
    assert result.stop_loss < 100.0 < result.take_profit


def test_trailing_stop_ratchets_up():
    pos = Position(
        symbol="BTC/GBP",
        side="long",
        amount=1.0,
        entry_price=100.0,
        stop_loss=97.0,
        take_profit=0.0,
        peak_price=100.0,
        trailing_stop_pct=0.02,
        opened_at=datetime.now(timezone.utc),
    )
    pos.update_trailing(110.0)
    assert pos.stop_loss > 97.0
    assert pos.stop_loss == 110.0 * (1 - 0.02)
    old = pos.stop_loss
    pos.update_trailing(105.0)
    assert pos.stop_loss == old
