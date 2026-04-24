"""Durable portfolio state so the engine survives a restart.

Without this, a `Ctrl+C` or crash makes the bot forget every open
position; stops and take-profits go untriggered, risk calculations are
based on zero exposure, and the next entry is computed as if nothing
is in flight. That is unacceptable in a bot that holds overnight.

On every trade open/close we write `state/portfolio.json` atomically.
On startup the engine reloads it, re-populates Position objects with
their MFE/MAE and stop levels intact, and logs a warning if the file
is older than the configured freshness threshold (likely indicating
missed fills while we were offline).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .portfolio import Portfolio, Position

log = logging.getLogger("bot.state")

DEFAULT_PATH = Path("state/portfolio.json")


def _position_to_dict(p: Position) -> dict[str, Any]:
    d = asdict(p)
    d["opened_at"] = p.opened_at.isoformat()
    return d


def _position_from_dict(d: dict[str, Any]) -> Position:
    raw_opened = d.get("opened_at")
    try:
        opened_at = datetime.fromisoformat(raw_opened) if raw_opened else datetime.now(timezone.utc)
    except (TypeError, ValueError):
        opened_at = datetime.now(timezone.utc)
    if opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=timezone.utc)
    return Position(
        symbol=d["symbol"],
        side=d.get("side", "long"),
        amount=float(d["amount"]),
        entry_price=float(d["entry_price"]),
        stop_loss=float(d.get("stop_loss", 0.0)),
        take_profit=float(d.get("take_profit", 0.0)),
        peak_price=float(d.get("peak_price", d["entry_price"])),
        trailing_stop_pct=float(d.get("trailing_stop_pct", 0.0)),
        opened_at=opened_at,
        mfe=float(d.get("mfe", 0.0)),
        mae=float(d.get("mae", 0.0)),
        scaled_out=bool(d.get("scaled_out", False)),
    )


def save_portfolio(portfolio: Portfolio, path: Path = DEFAULT_PATH) -> None:
    """Atomically persist the portfolio to ``path`` (JSON).

    Uses write-then-rename so a crash mid-write can never leave behind a
    half-written file the next startup would choke on.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "cash": portfolio.cash,
        "realised_pnl": portfolio.realised_pnl,
        "peak_equity": portfolio.peak_equity,
        "day_start_equity": portfolio.day_start_equity,
        "day_stamp": portfolio.day_stamp,
        "positions": [_position_to_dict(p) for p in portfolio.positions.values()],
    }
    fd, tmp_name = tempfile.mkstemp(prefix=".portfolio-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_portfolio(
    path: Path = DEFAULT_PATH,
    stale_after_seconds: float = 24 * 3600,
) -> Portfolio | None:
    """Return the persisted portfolio, or None if nothing saved yet.

    If the saved state is older than ``stale_after_seconds`` we load it
    anyway but emit a warning: the market may have moved while we were
    offline, so the caller should consider a fresh broker reconciliation.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.error("portfolio state at %s is unreadable: %s", path, exc)
        return None

    age = time.time() - path.stat().st_mtime
    if age > stale_after_seconds:
        log.warning(
            "portfolio state is %.1f hours old; broker may have filled or closed "
            "positions while offline. Run reconcile before acting.",
            age / 3600,
        )

    portfolio = Portfolio(
        cash=float(payload.get("cash", 0.0)),
        realised_pnl=float(payload.get("realised_pnl", 0.0)),
        peak_equity=float(payload.get("peak_equity", 0.0)),
        day_start_equity=float(payload.get("day_start_equity", 0.0)),
        day_stamp=str(payload.get("day_stamp", "")),
    )
    for raw_pos in payload.get("positions", []):
        try:
            pos = _position_from_dict(raw_pos)
        except (KeyError, ValueError, TypeError) as exc:
            log.warning("skipping malformed position in state: %s (%s)", raw_pos, exc)
            continue
        portfolio.positions[pos.symbol] = pos

    log.info(
        "restored portfolio: cash=%.2f realised_pnl=%.2f positions=%d (saved %s)",
        portfolio.cash, portfolio.realised_pnl, len(portfolio.positions),
        payload.get("saved_at", "?"),
    )
    return portfolio
