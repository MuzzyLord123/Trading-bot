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
        symbol="VUAG.L",
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


def test_size_rejects_when_tp_does_not_cover_costs():
    # tp_pct = 0.001 is smaller than a single-leg fee, let alone round trip.
    rm = RiskManager(
        _cfg(
            take_profit_pct=0.001,
            stop_loss_pct=0.02,
            taker_fee_pct=0.005,
            slippage_pct=0.0,
            min_reward_to_risk=1.0,
        )
    )
    result = rm.size("long", price=100.0, equity=500.0, cash=500.0)
    assert result.amount == 0.0
    assert "cost" in result.reason.lower() or "r:r" in result.reason.lower()


def test_size_cost_aware_tightens_rr_threshold():
    # Without fees gross r:r = 2.0; min=1.6 should pass.
    lenient = RiskManager(
        _cfg(
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            taker_fee_pct=0.0,
            slippage_pct=0.0,
            min_reward_to_risk=1.6,
        )
    )
    assert lenient.size("long", 100.0, 500.0, 500.0).amount > 0

    # Add enough cost that net r:r falls below 1.6.
    strict = RiskManager(
        _cfg(
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            taker_fee_pct=0.005,
            slippage_pct=0.0,
            min_reward_to_risk=1.6,
        )
    )
    assert strict.size("long", 100.0, 500.0, 500.0).amount == 0.0


def test_atr_stop_uses_volatility_distance():
    rm = RiskManager(
        _cfg(
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            taker_fee_pct=0.0,
            slippage_pct=0.0,
            use_atr_stop=True,
            atr_stop_multiplier=2.0,
        )
    )
    atr = 1.5  # absolute price units
    result = rm.size("long", price=100.0, equity=1000.0, cash=1000.0, atr=atr)
    # Stop sits atr * multiplier below entry, not price * stop_pct below.
    assert abs((100.0 - result.stop_loss) - 3.0) < 1e-9
    # TP preserves the tp:sl ratio from config (10%/5% = 2x).
    assert abs((result.take_profit - 100.0) - 6.0) < 1e-9


def test_consecutive_loss_circuit_breaker():
    rm = RiskManager(_cfg(max_consecutive_losses=3))
    p = Portfolio.new(500.0)
    assert rm.trading_halted(p, 500.0) is None
    rm.record_trade_result(-10.0)
    rm.record_trade_result(-10.0)
    assert rm.trading_halted(p, 500.0) is None
    rm.record_trade_result(-10.0)
    assert "consecutive" in (rm.trading_halted(p, 500.0) or "")
    rm.record_trade_result(5.0)  # win resets streak
    assert rm.trading_halted(p, 500.0) is None


def test_should_scale_out_triggers_once_at_r_multiple():
    rm = RiskManager(_cfg(scale_out_at_r=1.0, scale_out_fraction=0.5))
    pos = Position(
        symbol="VUAG.L", side="long", amount=1.0,
        entry_price=100.0, stop_loss=97.0, take_profit=0.0,
        peak_price=100.0, trailing_stop_pct=0.0,
        opened_at=datetime.now(timezone.utc),
    )
    # initial R = 3, so trigger at 103
    assert rm.should_scale_out(pos, 102.0) is False
    assert rm.should_scale_out(pos, 103.0) is True
    pos.scaled_out = True
    assert rm.should_scale_out(pos, 110.0) is False


def test_should_scale_out_disabled_when_stop_above_entry():
    rm = RiskManager(_cfg(scale_out_at_r=1.0))
    pos = Position(
        symbol="VUAG.L", side="long", amount=1.0,
        entry_price=100.0, stop_loss=101.0, take_profit=0.0,
        peak_price=100.0, trailing_stop_pct=0.0,
        opened_at=datetime.now(timezone.utc),
    )
    assert rm.should_scale_out(pos, 200.0) is False


def test_atr_mode_scales_size_inversely_with_volatility():
    rm = RiskManager(
        _cfg(
            risk_per_trade=0.02,
            max_position_pct=1.0,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            taker_fee_pct=0.0,
            slippage_pct=0.0,
            use_atr_stop=True,
            atr_stop_multiplier=2.0,
        )
    )
    calm = rm.size("long", 100.0, 1000.0, 1000.0, atr=0.5)
    noisy = rm.size("long", 100.0, 1000.0, 1000.0, atr=2.0)
    assert calm.amount > noisy.amount > 0


def test_trailing_stop_ratchets_up():
    pos = Position(
        symbol="VUAG.L",
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
