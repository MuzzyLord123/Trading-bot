from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import RiskConfig
from .portfolio import Portfolio, Position

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
        self.consecutive_losses = 0

    def record_trade_result(self, pnl: float) -> None:
        """Update the consecutive-loss counter. Called on every trade close."""
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def reset(self) -> None:
        """Clear internal trade-streak state (used between walk-forward windows)."""
        self.consecutive_losses = 0

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
        if (
            self.cfg.max_consecutive_losses > 0
            and self.consecutive_losses >= self.cfg.max_consecutive_losses
        ):
            return f"{self.consecutive_losses} consecutive losses"
        return None

    def can_open(self, portfolio: Portfolio) -> bool:
        return len(portfolio.positions) < self.cfg.max_open_positions

    def _effective_risk_pct(self, portfolio: Portfolio, equity: float) -> float:
        """Scale down risk while in drawdown."""
        base = self.cfg.risk_per_trade
        if portfolio.peak_equity <= 0:
            return base
        dd = 1 - equity / portfolio.peak_equity
        if dd >= self.cfg.drawdown_risk_reduction_threshold:
            return base * self.cfg.drawdown_risk_reduction_factor
        return base

    def size(
        self,
        side: str,
        price: float,
        equity: float,
        cash: float,
        portfolio: Portfolio | None = None,
        atr: float | None = None,
    ) -> SizingResult:
        """Fixed-fractional sizing based on stop distance. Long-only for now.

        When ``cfg.use_atr_stop`` is enabled and ``atr`` is supplied, the stop
        distance is ``atr * atr_stop_multiplier`` instead of a fixed percentage,
        adapting position size to realised volatility.

        Rejects trades whose reward:risk ratio (net of round-trip fees and
        slippage) is below ``min_reward_to_risk``.
        """
        stop_pct = self.cfg.stop_loss_pct
        if stop_pct <= 0 or stop_pct >= 1 or price <= 0 or side != "long":
            return SizingResult(0.0, 0.0, 0.0, "invalid inputs")

        tp_pct = self.cfg.take_profit_pct

        # Resolve absolute stop/target distances. In ATR mode the absolute
        # distance scales with volatility while preserving the tp:sl ratio
        # implied by the config.
        if self.cfg.use_atr_stop and atr is not None and atr > 0:
            stop_distance = atr * self.cfg.atr_stop_multiplier
            tp_distance = stop_distance * (tp_pct / stop_pct) if tp_pct > 0 else 0.0
        else:
            stop_distance = price * stop_pct
            tp_distance = price * tp_pct if tp_pct > 0 else 0.0

        if tp_pct > 0 and self.cfg.min_reward_to_risk > 0:
            cost_cash = price * 2 * (self.cfg.taker_fee_pct + self.cfg.slippage_pct)
            net_reward = tp_distance - cost_cash
            net_risk = stop_distance + cost_cash
            if net_reward <= 0:
                return SizingResult(0.0, 0.0, 0.0, "tp does not cover costs")
            rr = net_reward / net_risk if net_risk > 0 else 0.0
            if rr < self.cfg.min_reward_to_risk:
                return SizingResult(0.0, 0.0, 0.0, f"net r:r {rr:.2f} below min")

        risk_pct = (
            self._effective_risk_pct(portfolio, equity) if portfolio else self.cfg.risk_per_trade
        )
        risk_cash = equity * risk_pct
        amount = risk_cash / stop_distance if stop_distance > 0 else 0.0

        max_notional = min(equity * self.cfg.max_position_pct, cash)
        max_amount = max_notional / price if price > 0 else 0.0
        amount = min(amount, max_amount)

        if amount <= 0:
            return SizingResult(0.0, 0.0, 0.0, "sized to zero")

        stop = price - stop_distance
        tp = price + tp_distance if tp_distance > 0 else 0.0
        # Invariant for long entries: stop < entry < take_profit (if TP is set).
        if stop >= price or stop <= 0 or (tp > 0 and tp <= price):
            return SizingResult(0.0, 0.0, 0.0, "invalid stop/target ordering")
        return SizingResult(amount=amount, stop_loss=stop, take_profit=tp)

    def should_scale_out(self, position: Position, price: float) -> bool:
        """True once per trade, when price reaches entry + R * initial_risk.

        ``R`` = ``self.cfg.scale_out_at_r``. The trigger fires exactly once;
        subsequent calls return False even if price oscillates above the level.
        """
        if position.scaled_out or self.cfg.scale_out_at_r <= 0:
            return False
        initial_risk = position.entry_price - position.stop_loss
        # If the stop has already trailed above entry (e.g. after break-even),
        # we can't infer the original risk – skip the scale-out.
        if initial_risk <= 0:
            return False
        trigger = position.entry_price + self.cfg.scale_out_at_r * initial_risk
        return price >= trigger

    def maybe_move_to_breakeven(self, position: Position, price: float) -> None:
        """Move stop to entry once unrealised profit crosses the trigger."""
        trigger = self.cfg.breakeven_trigger_pct
        if trigger <= 0:
            return
        if price >= position.entry_price * (1 + trigger) and position.stop_loss < position.entry_price:
            position.stop_loss = position.entry_price

    def should_exit(
        self, position: Position, price: float, bars_held: int = 0
    ) -> str | None:
        if position.stop_loss and price <= position.stop_loss:
            return "stop_loss"
        if position.take_profit and price >= position.take_profit:
            return "take_profit"
        if (
            self.cfg.time_stop_bars > 0
            and bars_held >= self.cfg.time_stop_bars
            and price <= position.entry_price
        ):
            return "time_stop"
        return None
