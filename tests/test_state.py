from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot.portfolio import Portfolio, Position
from bot.state import load_portfolio, save_portfolio


def _make_portfolio() -> Portfolio:
    p = Portfolio.new(10000.0)
    p.realised_pnl = 125.50
    p.peak_equity = 10500.0
    p.positions["AAPL"] = Position(
        symbol="AAPL", side="long", amount=5.0, entry_price=150.0,
        stop_loss=145.0, take_profit=165.0, peak_price=155.0,
        trailing_stop_pct=0.02, opened_at=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
        mfe=5.0, mae=-1.2, scaled_out=False,
    )
    p.positions["MSFT"] = Position(
        symbol="MSFT", side="long", amount=10.0, entry_price=400.0,
        stop_loss=388.0, take_profit=424.0, peak_price=410.0,
        trailing_stop_pct=0.02, opened_at=datetime(2024, 1, 16, 11, 0, tzinfo=timezone.utc),
        mfe=10.0, mae=0.0, scaled_out=True,
    )
    return p


def test_round_trip_preserves_every_field(tmp_path):
    path = tmp_path / "portfolio.json"
    original = _make_portfolio()
    save_portfolio(original, path)
    restored = load_portfolio(path)
    assert restored is not None
    assert restored.cash == original.cash
    assert restored.realised_pnl == pytest.approx(125.50)
    assert restored.peak_equity == original.peak_equity
    assert set(restored.positions) == {"AAPL", "MSFT"}
    aapl = restored.positions["AAPL"]
    assert aapl.amount == 5.0
    assert aapl.entry_price == 150.0
    assert aapl.stop_loss == 145.0
    assert aapl.peak_price == 155.0
    assert aapl.mfe == 5.0
    assert aapl.mae == -1.2
    assert aapl.scaled_out is False
    assert aapl.opened_at.tzinfo is not None
    msft = restored.positions["MSFT"]
    assert msft.scaled_out is True


def test_load_returns_none_when_file_missing(tmp_path):
    assert load_portfolio(tmp_path / "does-not-exist.json") is None


def test_load_handles_corrupt_json(tmp_path):
    path = tmp_path / "portfolio.json"
    path.write_text("{ not valid json")
    assert load_portfolio(path) is None


def test_save_is_atomic_write(tmp_path):
    path = tmp_path / "portfolio.json"
    save_portfolio(_make_portfolio(), path)
    # No temp files left behind.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".portfolio-")]
    assert leftovers == []


def test_stale_state_still_loads_but_warns(tmp_path, caplog):
    path = tmp_path / "portfolio.json"
    save_portfolio(_make_portfolio(), path)
    # Backdate the file by two days.
    old = time.time() - 2 * 24 * 3600
    os.utime(path, (old, old))
    with caplog.at_level("WARNING"):
        restored = load_portfolio(path, stale_after_seconds=24 * 3600)
    assert restored is not None
    assert any("offline" in r.message for r in caplog.records)


def test_malformed_position_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "portfolio.json"
    payload = {
        "cash": 500.0, "realised_pnl": 0.0, "peak_equity": 500.0,
        "day_start_equity": 500.0, "day_stamp": "2024-01-15",
        "positions": [
            {"symbol": "AAPL", "amount": "bogus", "entry_price": 1},
            {"symbol": "MSFT", "amount": 1.0, "entry_price": 100.0,
             "stop_loss": 95, "take_profit": 110, "peak_price": 101,
             "trailing_stop_pct": 0.02, "opened_at": "2024-01-15T00:00:00+00:00"},
        ],
    }
    path.write_text(json.dumps(payload))
    restored = load_portfolio(path)
    assert restored is not None
    assert set(restored.positions) == {"MSFT"}


def test_missing_opened_at_defaults_to_now(tmp_path):
    path = tmp_path / "portfolio.json"
    payload = {
        "cash": 500.0, "realised_pnl": 0.0, "peak_equity": 500.0,
        "day_start_equity": 500.0, "day_stamp": "2024-01-15",
        "positions": [
            {"symbol": "AAPL", "amount": 1.0, "entry_price": 100.0,
             "stop_loss": 95, "take_profit": 110, "peak_price": 101,
             "trailing_stop_pct": 0.02},  # no opened_at
        ],
    }
    path.write_text(json.dumps(payload))
    restored = load_portfolio(path)
    assert restored is not None
    assert restored.positions["AAPL"].opened_at.tzinfo is not None
