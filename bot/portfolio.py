from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Position:
    symbol: str
    side: str  # currently only "long" – kept for future margin support
    amount: float
    entry_price: float
    stop_loss: float
    take_profit: float
    peak_price: float
    trailing_stop_pct: float
    opened_at: datetime
    # Maximum favourable / adverse excursion (absolute price terms) since
    # entry. Helps diagnose whether stops are too tight or TPs too distant.
    mfe: float = 0.0
    mae: float = 0.0
    # Tracks whether scale-out has already fired so we only take partial
    # profit once per trade.
    scaled_out: bool = False

    def unrealised(self, price: float) -> float:
        return (price - self.entry_price) * self.amount

    def update_trailing(self, price: float) -> None:
        if self.trailing_stop_pct <= 0:
            return
        if price > self.peak_price:
            self.peak_price = price
        new_stop = self.peak_price * (1 - self.trailing_stop_pct)
        if new_stop > self.stop_loss:
            self.stop_loss = new_stop

    def update_excursion(self, price: float) -> None:
        """Track MFE/MAE as price moves. Call once per tick/bar."""
        gain = price - self.entry_price
        if gain > self.mfe:
            self.mfe = gain
        if gain < self.mae:
            self.mae = gain


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    realised_pnl: float = 0.0
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    day_stamp: str = ""

    @classmethod
    def new(cls, starting_cash: float) -> "Portfolio":
        return cls(
            cash=starting_cash,
            peak_equity=starting_cash,
            day_start_equity=starting_cash,
            day_stamp=_today(),
        )

    def equity(self, price_lookup: dict[str, float]) -> float:
        value = self.cash
        for p in self.positions.values():
            price = price_lookup.get(p.symbol, p.entry_price)
            value += p.amount * price
        return value

    def mark_day(self, equity: float) -> None:
        today = _today()
        if today != self.day_stamp:
            self.day_stamp = today
            self.day_start_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
