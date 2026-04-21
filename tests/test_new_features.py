from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bot.config import RiskConfig
from bot.indicators import supertrend
from bot.portfolio import Portfolio, Position
from bot.risk import RiskManager


def _cfg(**overrides):
    base = dict(
        risk_per_trade=0.01,
        max_position_pct=1.0,
        max_open_positions=3,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        trailing_stop_pct=0.0,
        daily_loss_limit_pct=0.5,
        max_drawdown_pct=0.9,
        taker_fee_pct=0.001,
        slippage_pct=0.0,
        cooldown_bars_after_loss=0,
        min_reward_to_risk=1.5,
        breakeven_trigger_pct=0.02,
        time_stop_bars=48,
        drawdown_risk_reduction_threshold=0.05,
        drawdown_risk_reduction_factor=0.5,
    )
    base.update(overrides)
    return RiskConfig(**base)


def _pos(entry: float = 100.0, stop: float = 97.0, tp: float = 106.0) -> Position:
    return Position(
        symbol="BTC/GBP",
        side="long",
        amount=1.0,
        entry_price=entry,
        stop_loss=stop,
        take_profit=tp,
        peak_price=entry,
        trailing_stop_pct=0.0,
        opened_at=datetime.now(timezone.utc),
    )


def _df(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(close), freq="h", tz="UTC"),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones_like(close),
        }
    )


def test_supertrend_flips_direction_with_price():
    down = np.linspace(200, 120, 150)
    up = np.linspace(120, 260, 150)
    close = np.concatenate([down, up])
    st = supertrend(_df(close), period=10, multiplier=2.0)
    dir_values = set(st["direction"].dropna().unique())
    assert dir_values.issubset({-1, 1})
    # Indicator must be able to flip both ways given this dataset.
    assert {-1, 1}.issubset(dir_values)
    # End of the uptrend leg should be +1.
    assert st["direction"].iloc[-1] == 1


def test_size_rejects_poor_reward_risk():
    rm = RiskManager(_cfg(take_profit_pct=0.03, stop_loss_pct=0.03, min_reward_to_risk=1.5))
    result = rm.size("long", price=100.0, equity=500.0, cash=500.0)
    assert result.amount == 0.0
    assert "r:r" in result.reason


def test_size_scales_down_in_drawdown():
    rm = RiskManager(_cfg(drawdown_risk_reduction_threshold=0.05, drawdown_risk_reduction_factor=0.5))
    p = Portfolio.new(500.0)
    p.peak_equity = 600.0
    # Same equity both calls – only change is whether we're "in drawdown" relative to peak.
    p.peak_equity = 600.0
    normal = rm.size("long", price=100.0, equity=600.0, cash=1000.0, portfolio=p)
    reduced = rm.size("long", price=100.0, equity=560.0, cash=1000.0, portfolio=p)
    ratio = reduced.amount / normal.amount
    # Position is halved relative to equity – reduced is (560/600) * 0.5 ≈ 0.467 of normal.
    assert 0.4 < ratio < 0.55


def test_breakeven_stop_moves_up():
    rm = RiskManager(_cfg(breakeven_trigger_pct=0.02))
    pos = _pos()
    rm.maybe_move_to_breakeven(pos, price=101.0)
    assert pos.stop_loss == 97.0
    rm.maybe_move_to_breakeven(pos, price=102.5)
    assert pos.stop_loss == 100.0


def test_time_stop_exits_flat_trade():
    rm = RiskManager(_cfg(time_stop_bars=10))
    pos = _pos()
    assert rm.should_exit(pos, price=100.0, bars_held=5) is None
    assert rm.should_exit(pos, price=99.5, bars_held=15) in {"time_stop", "stop_loss"}
    assert rm.should_exit(pos, price=100.0, bars_held=15) == "time_stop"
    assert rm.should_exit(pos, price=105.0, bars_held=15) is None


def test_take_profit_still_triggers():
    rm = RiskManager(_cfg())
    pos = _pos()
    assert rm.should_exit(pos, price=106.5) == "take_profit"
