"""Shared trade-execution math used by both the live engine and the backtester.

Keeping fee, slippage and PnL calculations in one place prevents the two call
sites from drifting (a classic source of hard-to-spot differences between
backtest and paper/live P&L).
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import RiskConfig


@dataclass
class Fill:
    price: float
    amount: float
    fee_cost: float

    @property
    def notional(self) -> float:
        return self.price * self.amount

    @property
    def cash_out(self) -> float:
        """Cash leaving the account on a buy (notional + fees)."""
        return self.notional + self.fee_cost

    @property
    def cash_in(self) -> float:
        """Cash entering the account on a sell (notional - fees)."""
        return self.notional - self.fee_cost


def buy_fill(price: float, amount: float, risk_cfg: RiskConfig) -> Fill:
    """Simulate a market buy: apply positive slippage, then fees."""
    fill_price = price * (1 + risk_cfg.slippage_pct)
    fee_cost = fill_price * amount * risk_cfg.taker_fee_pct
    return Fill(price=fill_price, amount=amount, fee_cost=fee_cost)


def sell_fill(price: float, amount: float, risk_cfg: RiskConfig) -> Fill:
    """Simulate a market sell: apply negative slippage, then fees."""
    fill_price = price * (1 - risk_cfg.slippage_pct)
    fee_cost = fill_price * amount * risk_cfg.taker_fee_pct
    return Fill(price=fill_price, amount=amount, fee_cost=fee_cost)


def actual_fill(price: float, amount: float, risk_cfg: RiskConfig) -> Fill:
    """Wrap a real broker fill (already-realised price + amount) into a
    :class:`Fill` with the modelled fee cost layered on.

    Use this in live mode where the broker has told us the true fill
    price - we don't want to double-apply our slippage model on top.
    Backtest and paper mode should use :func:`buy_fill` / :func:`sell_fill`
    which add slippage to a notional reference price.
    """
    fee_cost = price * amount * risk_cfg.taker_fee_pct
    return Fill(price=price, amount=amount, fee_cost=fee_cost)


def realised_pnl(entry_price: float, fill: Fill) -> float:
    """PnL for closing ``fill.amount`` of a long at ``fill.price`` net of fees."""
    return (fill.price - entry_price) * fill.amount - fill.fee_cost
