"""Bootstrap-CI significance tests for BacktestResult."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.backtest import BacktestResult


def _make_result(pnls: list[float]) -> BacktestResult:
    """Build a BacktestResult with the given closed-trade P&L values.
    The equity curve isn't relevant for bootstrap_ci, but we populate
    a trivial one so other methods don't blow up if they're called."""
    res = BacktestResult(equity_curve=pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=len(pnls) or 1, freq="D", tz="UTC"),
        "equity": [10000 + sum(pnls[:i + 1]) for i in range(max(1, len(pnls)))],
        "cash": [10000.0] * max(1, len(pnls)),
    }))
    res.trades = [{"pnl": p, "symbol": "X", "side": "long", "amount": 1.0,
                    "price": 100.0, "reason": "test", "timestamp": pd.Timestamp("2024-01-01")}
                   for p in pnls]
    return res


def test_bootstrap_ci_returns_zero_for_no_trades():
    res = _make_result([])
    out = res.bootstrap_ci()
    assert out["lo"] == 0.0
    assert out["hi"] == 0.0
    assert out["significant"] is False
    assert out["trades"] == 0


def test_bootstrap_ci_returns_zero_for_single_trade():
    """One trade has nothing to resample meaningfully; we punt."""
    out = _make_result([5.0]).bootstrap_ci()
    assert out["significant"] is False


def test_bootstrap_ci_flags_clearly_winning_strategy():
    # 30 wins of +5, 5 losses of -1. Bootstrap of this should never go
    # below zero - even the worst resamples are dominated by wins.
    pnls = [5.0] * 30 + [-1.0] * 5
    out = _make_result(pnls).bootstrap_ci(n_resamples=2000, confidence=0.95, seed=1)
    assert out["lo"] > 0
    assert out["significant"] is True


def test_bootstrap_ci_does_not_flag_breakeven_noise():
    # Symmetric noise - half wins, half losses, same magnitude. Total is
    # near zero by construction; the lower CI bound should straddle it.
    pnls = [1.0, -1.0] * 50
    out = _make_result(pnls).bootstrap_ci(n_resamples=3000, seed=42)
    assert out["lo"] <= 0  # not significantly positive


def test_bootstrap_ci_lo_hi_contain_median():
    pnls = [2.0, -1.0, 3.0, -2.0, 4.0, -1.0, 2.0, -1.0]
    out = _make_result(pnls).bootstrap_ci(seed=7)
    assert out["lo"] <= out["median"] <= out["hi"]


def test_bootstrap_ci_seed_is_deterministic():
    pnls = [1.5, -0.8, 2.0, -1.2, 0.9]
    a = _make_result(pnls).bootstrap_ci(seed=99, n_resamples=500)
    b = _make_result(pnls).bootstrap_ci(seed=99, n_resamples=500)
    assert a == b


def test_bootstrap_ci_higher_confidence_widens_band():
    pnls = list(np.random.default_rng(11).normal(0.5, 1.0, 100))
    narrow = _make_result(pnls).bootstrap_ci(confidence=0.50, seed=3)
    wide = _make_result(pnls).bootstrap_ci(confidence=0.95, seed=3)
    assert (wide["hi"] - wide["lo"]) >= (narrow["hi"] - narrow["lo"])
