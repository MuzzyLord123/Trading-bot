from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bot.portfolio import Portfolio, Position
from bot.reconcile import reconcile, ReconciliationReport
from bot.stocks import T212Position


class FakeBroker:
    """Minimal Trading212Broker stand-in for reconciliation tests."""
    def __init__(self, positions: list[T212Position], cash: float = 1000.0,
                 raise_on: str | None = None) -> None:
        self._positions = positions
        self._cash = cash
        self._raise_on = raise_on

    def positions(self) -> list[T212Position]:
        if self._raise_on == "positions":
            raise RuntimeError("boom")
        return list(self._positions)

    def cash(self) -> float:
        if self._raise_on == "cash":
            raise RuntimeError("boom")
        return self._cash


def _local_pos(symbol: str, amount: float, entry: float = 100.0) -> Position:
    return Position(
        symbol=symbol, side="long", amount=amount, entry_price=entry,
        stop_loss=entry * 0.95, take_profit=entry * 1.10,
        peak_price=entry, trailing_stop_pct=0.0,
        opened_at=datetime.now(timezone.utc),
    )


def test_clean_state_reports_clean():
    portfolio = Portfolio.new(1000.0)
    portfolio.positions["AAPL"] = _local_pos("AAPL", 5.0)
    broker = FakeBroker(
        [T212Position(ticker="AAPL", quantity=5.0, average_price=100.0, current_price=102.0)],
        cash=1000.0,
    )
    report = reconcile(portfolio, broker)
    assert report.clean
    assert "clean" in report.summary()


def test_adopts_broker_only_position_with_synthetic_stop():
    portfolio = Portfolio.new(1000.0)
    broker = FakeBroker(
        [T212Position(ticker="MSFT", quantity=2.0, average_price=400.0, current_price=410.0)],
        cash=1000.0,
    )
    report = reconcile(portfolio, broker, stop_loss_pct_fallback=0.05)
    assert report.adopted == ["MSFT"]
    pos = portfolio.positions["MSFT"]
    assert pos.amount == 2.0
    assert pos.entry_price == 400.0
    assert pos.stop_loss == pytest.approx(400.0 * 0.95)


def test_drops_orphaned_local_position():
    portfolio = Portfolio.new(1000.0)
    portfolio.positions["AAPL"] = _local_pos("AAPL", 5.0)
    broker = FakeBroker([], cash=1000.0)
    report = reconcile(portfolio, broker)
    assert report.dropped == ["AAPL"]
    assert "AAPL" not in portfolio.positions


def test_resizes_when_quantities_diverge():
    portfolio = Portfolio.new(1000.0)
    portfolio.positions["AAPL"] = _local_pos("AAPL", 5.0)
    broker = FakeBroker(
        [T212Position(ticker="AAPL", quantity=4.0, average_price=100.0, current_price=100.0)],
        cash=1000.0,
    )
    report = reconcile(portfolio, broker)
    assert len(report.resized) == 1
    assert portfolio.positions["AAPL"].amount == 4.0


def test_tiny_quantity_difference_is_ignored():
    portfolio = Portfolio.new(1000.0)
    portfolio.positions["AAPL"] = _local_pos("AAPL", 5.0)
    broker = FakeBroker(
        [T212Position(ticker="AAPL", quantity=5.0 + 1e-8,
                       average_price=100.0, current_price=100.0)],
        cash=1000.0,
    )
    report = reconcile(portfolio, broker)
    assert report.resized == []
    assert report.clean


def test_cash_reconciled_when_divergent():
    portfolio = Portfolio.new(1000.0)
    broker = FakeBroker([], cash=1234.56)
    report = reconcile(portfolio, broker)
    assert abs(report.cash_delta - 234.56) < 0.01
    assert portfolio.cash == pytest.approx(1234.56)


def test_all_changes_at_once():
    portfolio = Portfolio.new(500.0)
    portfolio.positions["AAPL"] = _local_pos("AAPL", 5.0)
    portfolio.positions["MSFT"] = _local_pos("MSFT", 3.0)  # orphan
    broker = FakeBroker(
        [
            T212Position(ticker="AAPL", quantity=4.5, average_price=100.0, current_price=100.0),  # resized
            T212Position(ticker="GOOG", quantity=1.0, average_price=150.0, current_price=152.0),  # adopted
        ],
        cash=600.0,
    )
    report = reconcile(portfolio, broker)
    assert report.adopted == ["GOOG"]
    assert report.dropped == ["MSFT"]
    assert len(report.resized) == 1
    assert report.cash_delta == pytest.approx(100.0)
    assert set(portfolio.positions) == {"AAPL", "GOOG"}


def test_broker_failure_returns_empty_report_without_mutation():
    portfolio = Portfolio.new(1000.0)
    portfolio.positions["AAPL"] = _local_pos("AAPL", 5.0)
    broker = FakeBroker([], cash=0.0, raise_on="positions")
    report = reconcile(portfolio, broker)
    assert report.clean
    assert "AAPL" in portfolio.positions


def test_cash_failure_still_reconciles_positions():
    portfolio = Portfolio.new(1000.0)
    broker = FakeBroker(
        [T212Position(ticker="MSFT", quantity=2.0, average_price=400.0, current_price=410.0)],
        cash=0.0, raise_on="cash",
    )
    report = reconcile(portfolio, broker)
    assert "MSFT" in portfolio.positions
    assert report.cash_delta == 0.0  # cash untouched when broker cash lookup failed


def test_zero_quantity_broker_position_ignored():
    portfolio = Portfolio.new(1000.0)
    broker = FakeBroker(
        [T212Position(ticker="ZERO", quantity=0.0, average_price=100.0, current_price=100.0)],
        cash=1000.0,
    )
    report = reconcile(portfolio, broker)
    assert "ZERO" not in portfolio.positions
    assert report.adopted == []
