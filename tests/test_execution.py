from __future__ import annotations

from bot.config import RiskConfig
from bot.execution import buy_fill, sell_fill


def _cfg(**over):
    base = dict(taker_fee_pct=0.001, slippage_pct=0.001)
    base.update(over)
    return RiskConfig(**base)


def test_buy_fill_applies_positive_slippage_and_fee():
    f = buy_fill(price=100.0, amount=1.0, risk_cfg=_cfg())
    assert abs(f.price - 100.1) < 1e-9
    assert abs(f.fee_cost - 100.1 * 1.0 * 0.001) < 1e-9
    assert abs(f.cash_out - (f.price + f.fee_cost)) < 1e-9


def test_sell_fill_applies_negative_slippage_and_fee():
    f = sell_fill(price=100.0, amount=1.0, risk_cfg=_cfg())
    assert abs(f.price - 99.9) < 1e-9
    assert abs(f.cash_in - (f.price - f.fee_cost)) < 1e-9


def test_round_trip_is_symmetric():
    buy = buy_fill(100.0, 1.0, _cfg())
    sell = sell_fill(100.0, 1.0, _cfg())
    # Round-trip cost = slippage (both sides) + fees (both sides), roughly.
    round_trip = buy.cash_out - sell.cash_in
    expected = (buy.price - sell.price) + buy.fee_cost + sell.fee_cost
    assert abs(round_trip - expected) < 1e-9
