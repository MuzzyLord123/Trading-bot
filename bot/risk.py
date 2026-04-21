from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import RiskConfig
from .portfolio import Portfolio

log = logging.getLogger("bot.risk")


@dataclass
class SizingResult:
    amount: float
    stop_loss: float
    take_profit: float
    reason: str = ""


class RiskManager:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def trading_halted(self, portfolio: Portfolio, equity: float) -> str | None:
        """Return a human-readable reason if trading should halt, else None."""
        if portfolio.day_start_equity > 0:
            daily_dd = 1 - equity / portfolio.day_start_equity
            if daily_dd >= self.cfg.daily_loss_limit_pct:
                return f"daily loss limit hit ({daily_dd:.2%})"
        if portfolio.peak_equity > 0:
            total_dd = 1 - equity / portfolio.peak_equity
            if total_dd >= self.cfg.max_drawdown_pct:
                return f"max drawdown hit ({total_dd:.2%})"
        return None

    def can_open(self, portfolio: Portfolio) -> bool:
        return len(portfolio.positions) < self.cfg.max_open_positions

    def size(
        self,
        side: str,
        price: float,
        equity: float,
        cash: float,
    ) -> SizingResult:
        """Fixed-fractional sizing based on stop distance. Long-only for now."""
        stop_pct = self.cfg.stop_loss_pct
        if stop_pct <= 0 or price <= 0 or side != "long":
            return SizingResult(0.0, 0.0, 0.0, "invalid inputs")

        risk_cash = equity * self.cfg.risk_per_trade
        stop_distance = price * stop_pct
        amount = risk_cash / stop_distance if stop_distance > 0 else 0.0

        max_notional = min(equity * self.cfg.max_position_pct, cash)
        max_amount = max_notional / price if price > 0 else 0.0
        amount = min(amount, max_amount)

        if amount <= 0:
            return SizingResult(0.0, 0.0, 0.0, "sized to zero")

        stop = price * (1 - stop_pct)
        tp = price * (1 + self.cfg.take_profit_pct) if self.cfg.take_profit_pct > 0 else 0.0
        return SizingResult(amount=amount, stop_loss=stop, take_profit=tp)

    def should_exit(self, position, price: float) -> str | None:
        if position.stop_loss and price <= position.stop_loss:
            return "stop_loss"
        if position.take_profit and price >= position.take_profit:
            return "take_profit"
        return None
