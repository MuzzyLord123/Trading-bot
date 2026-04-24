"""Live-mode broker reconciliation.

The local Portfolio is the bot's view of the world. The Trading 212 API
is the actual truth. These can drift in several ways:

  * Fills the bot sent but never saw the confirmation for (network blip).
  * Orders the bot placed that T212 rejected for reasons our retry logic
    didn't surface (e.g. insufficient cash, unsupported instrument,
    market closed).
  * Manual interventions - you close a position by hand in the app.
  * Corporate actions, stock splits, delistings.

We compare the two views on startup and whenever the engine loop catches
a suspicious gap (no recent fills but broker positions changed, etc.).
The reconciler emits a structured report and updates the local Portfolio
to match the broker. It never places orders. Closing orphaned local
positions or opening "phantom" positions is the operator's call.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .portfolio import Portfolio, Position

if TYPE_CHECKING:  # pragma: no cover
    from .stocks import Trading212Broker

log = logging.getLogger("bot.reconcile")

# Two positions are considered "the same size" if they agree to within
# this fraction. T212 shares can be fractional, so we can't compare for
# exact equality - a 1e-6 rounding difference in quantity is normal.
QUANTITY_TOLERANCE = 1e-4


@dataclass
class ReconciliationReport:
    """What changed during reconciliation. Emitted whether or not anything
    was actually adjusted so operators always get a heartbeat."""
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Broker has a position the bot didn't know about (opened manually or
    # via an order the bot never saw fill). We adopt it - setting a
    # conservative stop equal to the average price so it can't run off
    # forever, but otherwise leave it untouched.
    adopted: list[str] = field(default_factory=list)
    # Bot thought it had a position that the broker does not hold (closed
    # manually, rejected fill). We drop it locally. Any realised P&L from
    # the missed close is unrecoverable from here - log it loudly.
    dropped: list[str] = field(default_factory=list)
    # Both sides have the position but the quantity diverged (partial
    # fill, stock split). We sync to the broker's number; entry price is
    # kept since we don't know what T212's average_price reflects after
    # a split.
    resized: list[tuple[str, float, float]] = field(default_factory=list)
    # Cash balance diverged. We trust the broker and update locally.
    cash_delta: float = 0.0

    @property
    def clean(self) -> bool:
        return (
            not self.adopted
            and not self.dropped
            and not self.resized
            and abs(self.cash_delta) < 0.01
        )

    def summary(self) -> str:
        if self.clean:
            return "reconciliation clean - local state matches broker"
        parts = []
        if self.adopted:
            parts.append(f"adopted {len(self.adopted)} broker position(s): {', '.join(self.adopted)}")
        if self.dropped:
            parts.append(f"dropped {len(self.dropped)} orphaned local position(s): {', '.join(self.dropped)}")
        if self.resized:
            parts.append(f"resized {len(self.resized)} position(s)")
        if abs(self.cash_delta) >= 0.01:
            parts.append(f"cash adjusted by {self.cash_delta:+.2f}")
        return "; ".join(parts)


def reconcile(
    portfolio: Portfolio,
    broker: "Trading212Broker",
    trailing_stop_pct: float = 0.0,
    stop_loss_pct_fallback: float = 0.05,
) -> ReconciliationReport:
    """Align ``portfolio`` with what ``broker`` says is actually held.

    Mutates ``portfolio`` in place. Never places orders. The caller
    should persist state immediately afterwards.

    Parameters
    ----------
    trailing_stop_pct:
        Used when we have to synthesise a Position for a broker-side
        holding we didn't know about. 0.0 = no trailing stop.
    stop_loss_pct_fallback:
        Fraction-of-price fallback stop for adopted positions so
        should_exit has something to work with until the operator sets
        a proper stop. Expressed as e.g. 0.05 for 5%.
    """
    report = ReconciliationReport()

    try:
        broker_positions = {p.ticker: p for p in broker.positions()}
    except Exception as exc:
        log.error("broker.positions() failed: %s - skipping reconciliation", exc)
        return report

    try:
        broker_cash = float(broker.cash())
    except Exception as exc:
        log.error("broker.cash() failed: %s - skipping cash reconciliation", exc)
        broker_cash = None

    local_symbols = set(portfolio.positions)
    broker_symbols = set(broker_positions)

    # Positions the broker holds but we don't know about.
    for ticker in broker_symbols - local_symbols:
        bp = broker_positions[ticker]
        if bp.quantity <= 0:
            continue  # defensive - skip any zero/short reported
        # Synthesize a Position with a conservative fallback stop so the
        # engine has something sensible to evaluate should_exit against.
        entry = bp.average_price or bp.current_price
        stop = entry * (1 - stop_loss_pct_fallback) if entry > 0 else 0.0
        portfolio.positions[ticker] = Position(
            symbol=ticker,
            side="long",
            amount=bp.quantity,
            entry_price=entry,
            stop_loss=stop,
            take_profit=0.0,  # no TP set - operator should review
            peak_price=bp.current_price or entry,
            trailing_stop_pct=trailing_stop_pct,
            opened_at=datetime.now(timezone.utc),
        )
        report.adopted.append(ticker)
        log.warning(
            "adopted broker position %s qty=%.6f avg=%.4f - stop synthesized at %.4f",
            ticker, bp.quantity, entry, stop,
        )

    # Positions we thought we had but the broker does not.
    for ticker in local_symbols - broker_symbols:
        del portfolio.positions[ticker]
        report.dropped.append(ticker)
        log.warning("dropped orphaned local position %s (not held by broker)", ticker)

    # Positions both sides know about. Check the quantities agree.
    for ticker in local_symbols & broker_symbols:
        local = portfolio.positions[ticker]
        bp = broker_positions[ticker]
        if abs(local.amount - bp.quantity) > max(QUANTITY_TOLERANCE, local.amount * 0.001):
            old = local.amount
            local.amount = bp.quantity
            report.resized.append((ticker, old, bp.quantity))
            log.warning(
                "resized %s: local %.6f -> broker %.6f", ticker, old, bp.quantity,
            )

    if broker_cash is not None:
        delta = broker_cash - portfolio.cash
        if abs(delta) >= 0.01:
            report.cash_delta = delta
            portfolio.cash = broker_cash
            log.warning("cash reconciled: %+.2f (now %.2f)", delta, broker_cash)

    return report
