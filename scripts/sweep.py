#!/usr/bin/env python3
"""Parameter sweep: systematically test which indicators, filters, and
risk settings actually deliver high win rate and return.

Runs every config against the same synthetic regimes (trending / choppy /
bear) with multiple seeds for statistical stability, then aggregates the
results and prints a ranked report.

IMPORTANT CAVEAT: synthetic data is not the same as real market data.
These numbers tell us *which configs have an edge at all* and *how WR /
return trade off*. Always validate the winner on real Yahoo Finance /
Trading 212 data with backtest.py.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from bot.backtest import Backtester
from bot.config import (
    Config,
    ExchangeConfig,
    LoggingConfig,
    NotificationsConfig,
    RiskConfig,
    StrategyConfig,
    TradingConfig,
)
from bot.risk import RiskManager
from bot.strategies import build_strategy_from_config


# ---------------------------------------------------------------------------
# Synthetic data generators – 4 regimes × multiple seeds.
# ---------------------------------------------------------------------------
def _to_ohlcv(close: np.ndarray) -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=len(close), freq="h", tz="UTC")
    noise = np.abs(np.diff(close, prepend=close[0])) * 0.5 + 0.1
    return pd.DataFrame(
        {
            "timestamp": t,
            "open": close,
            "high": close + noise,
            "low": close - noise,
            "close": close,
            "volume": np.ones_like(close),
        }
    )


def trending(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = np.linspace(0, 1.2, n)
    noise = rng.normal(0, 0.004, n).cumsum()
    return _to_ohlcv(100 * np.exp(drift + noise))


def choppy(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cycle = 5 * np.sin(np.linspace(0, 30, n))
    noise = rng.normal(0, 1.0, n).cumsum() * 0.3
    return _to_ohlcv(100 + cycle + noise)


def bear(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = np.linspace(0, -0.35, n)
    noise = rng.normal(0, 0.010, n).cumsum()
    return _to_ohlcv(100 * np.exp(drift + noise))


def mixed(n: int, seed: int) -> pd.DataFrame:
    """Most realistic: alternating up/chop/down phases."""
    rng = np.random.default_rng(seed)
    phases = []
    price = 100.0
    remaining = n
    while remaining > 0:
        length = min(rng.integers(150, 400), remaining)
        regime = rng.choice(["up", "chop", "down"])
        drift = {"up": 0.004, "chop": 0.0, "down": -0.003}[regime]
        vol = {"up": 0.006, "chop": 0.012, "down": 0.008}[regime]
        step = rng.normal(drift, vol, length)
        path = price * np.exp(step.cumsum())
        phases.append(path)
        price = path[-1]
        remaining -= length
    return _to_ohlcv(np.concatenate(phases))


REGIMES: dict[str, callable] = {
    "trending": trending,
    "choppy": choppy,
    "bear": bear,
    "mixed": mixed,
}


# ---------------------------------------------------------------------------
# Config builder.
# ---------------------------------------------------------------------------
@dataclass
class SweepConfig:
    name: str
    strategy: str  # "ma_crossover" | "rsi_reversion" | "macd" | "bollinger" | "ensemble"
    tp_pct: float
    sl_pct: float
    # ensemble
    members: list[str] | None = None
    min_agreement: int = 2
    min_score: float = 1.5
    # filter
    filter_enabled: bool = False
    trend_ema: int = 100
    min_adx: float = 15.0
    use_supertrend: bool = False
    htf_rule: str | None = None
    # risk
    risk_per_trade: float = 0.02
    trailing_stop_pct: float = 0.02
    breakeven_trigger_pct: float = 0.015
    time_stop_bars: int = 48
    min_reward_to_risk: float = 1.2


def to_config(s: SweepConfig) -> Config:
    return Config(
        exchange=ExchangeConfig(name="trading212"),
        trading=TradingConfig(
            starting_capital=10000.0,
            symbols=["SYN"],
            timeframe="1h",
            history_candles=500,
        ),
        risk=RiskConfig(
            risk_per_trade=s.risk_per_trade,
            max_position_pct=0.5,
            max_open_positions=1,
            stop_loss_pct=s.sl_pct,
            take_profit_pct=s.tp_pct,
            trailing_stop_pct=s.trailing_stop_pct,
            daily_loss_limit_pct=0.5,
            max_drawdown_pct=0.9,
            taker_fee_pct=0.0026,
            slippage_pct=0.0005,
            cooldown_bars_after_loss=6,
            min_reward_to_risk=s.min_reward_to_risk,
            breakeven_trigger_pct=s.breakeven_trigger_pct,
            time_stop_bars=s.time_stop_bars,
            drawdown_risk_reduction_threshold=0.1,
            drawdown_risk_reduction_factor=0.5,
        ),
        strategy=StrategyConfig(
            name=s.strategy,
            ensemble={
                "min_agreement": s.min_agreement,
                "min_score": s.min_score,
                "members": s.members or ["ma_crossover", "rsi_reversion", "macd", "bollinger"],
                "weights": {"ma_crossover": 1.2, "macd": 1.2, "rsi_reversion": 1.0, "bollinger": 0.8},
            },
            params={
                "ma_crossover": {"fast": 20, "slow": 50},
                "rsi_reversion": {"period": 14, "oversold": 30, "overbought": 70},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "bollinger": {"period": 20, "std": 2.0},
            },
            filter={
                "enabled": s.filter_enabled,
                "trend_ema": s.trend_ema,
                "require_uptrend": True,
                "min_adx": s.min_adx,
                "adx_period": 14,
                "min_atr_pct": 0.0003,
                "atr_period": 14,
                "htf_rule": s.htf_rule,
                "htf_ema": 50,
                "use_supertrend": s.use_supertrend,
                "supertrend_period": 10,
                "supertrend_multiplier": 3.0,
                "volume_mult": 0.0,
                "volume_period": 20,
            },
        ),
        logging=LoggingConfig(),
        notifications=NotificationsConfig(),
        secrets={"api_key": "", "api_secret": "", "api_password": ""},
    )


def run_one(s: SweepConfig, data: pd.DataFrame) -> dict[str, Any]:
    cfg = to_config(s)
    strategy = build_strategy_from_config(
        cfg.strategy.name, cfg.strategy.params, cfg.strategy.ensemble, cfg.strategy.filter
    )
    bt = Backtester(cfg, exchange=None, strategy=strategy, risk=RiskManager(cfg.risk))
    res = bt.run(data={"SYN/GBP": data}, write_csv=False)
    return res.stats(cfg.trading.starting_capital)


# ---------------------------------------------------------------------------
# Build the config grid.
# ---------------------------------------------------------------------------
def build_configs() -> list[SweepConfig]:
    configs: list[SweepConfig] = []

    # --- 1. Each strategy alone, default TP/SL, no filter ---------------
    for name in ["ma_crossover", "rsi_reversion", "macd", "bollinger"]:
        configs.append(SweepConfig(f"solo_{name}", name, tp_pct=0.06, sl_pct=0.03))
        configs.append(
            SweepConfig(f"solo_{name}_filtered", name, tp_pct=0.06, sl_pct=0.03, filter_enabled=True)
        )

    # --- 2. TP/SL ratio sweep on the best solo strategy (MACD, typically) ---
    for tp, sl in [
        (0.02, 0.02),  # 1:1
        (0.03, 0.02),  # 1.5:1
        (0.04, 0.02),  # 2:1
        (0.06, 0.02),  # 3:1
        (0.09, 0.02),  # 4.5:1
        (0.02, 0.04),  # inverted – high WR bet
        (0.015, 0.04), # high-WR / low-RR (scalp-like)
        (0.015, 0.06), # very high-WR / very low-RR
    ]:
        configs.append(
            SweepConfig(f"macd_tp{tp:.3f}_sl{sl:.3f}", "macd", tp_pct=tp, sl_pct=sl)
        )

    # --- 3. Ensemble combinations ---------------------------------------
    member_sets: list[tuple[str, list[str]]] = [
        ("all", ["ma_crossover", "rsi_reversion", "macd", "bollinger"]),
        ("trend", ["ma_crossover", "macd"]),
        ("macd_rsi", ["rsi_reversion", "macd"]),
    ]
    for mname, members in member_sets:
        for min_agree in [1, 2]:
            for min_score in [1.0, 1.8]:
                if min_agree > len(members):
                    continue
                configs.append(
                    SweepConfig(
                        f"ens_{mname}_a{min_agree}_s{min_score:.1f}",
                        "ensemble",
                        tp_pct=0.06,
                        sl_pct=0.03,
                        members=members,
                        min_agreement=min_agree,
                        min_score=min_score,
                    )
                )

    # --- 4. Filter tightness on the ensemble ----------------------------
    for trend_ema in [50, 100, 200]:
        for min_adx in [10, 20]:
            configs.append(
                SweepConfig(
                    f"ens_filt_ema{trend_ema}_adx{min_adx}",
                    "ensemble",
                    tp_pct=0.06,
                    sl_pct=0.03,
                    members=["ma_crossover", "rsi_reversion", "macd", "bollinger"],
                    min_agreement=2,
                    min_score=1.5,
                    filter_enabled=True,
                    trend_ema=trend_ema,
                    min_adx=min_adx,
                )
            )

    # --- 5. Best-guess "high WR scalp" configs --------------------------
    for tp, sl in [(0.01, 0.04), (0.015, 0.05), (0.02, 0.06)]:
        configs.append(
            SweepConfig(
                f"scalp_tp{tp:.3f}_sl{sl:.3f}",
                "ensemble",
                tp_pct=tp,
                sl_pct=sl,
                members=["ma_crossover", "rsi_reversion", "macd", "bollinger"],
                min_agreement=2,
                min_score=1.5,
                filter_enabled=True,
                trend_ema=100,
                min_adx=15,
                min_reward_to_risk=0.1,  # ratio would reject these otherwise
            )
        )

    return configs


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------
def main() -> None:
    import sys
    n_bars = 1200
    seeds = [1, 2]
    configs = build_configs()
    print(
        f"Running {len(configs)} configs x {len(REGIMES)} regimes x {len(seeds)} seeds "
        f"= {len(configs) * len(REGIMES) * len(seeds)} backtests",
        flush=True,
    )

    # Pre-generate all data so every config sees identical markets.
    datasets: dict[tuple[str, int], pd.DataFrame] = {
        (rname, seed): fn(n_bars, seed) for rname, fn in REGIMES.items() for seed in seeds
    }

    rows = []
    for i, cfg in enumerate(configs, 1):
        print(f"  [{i}/{len(configs)}] {cfg.name}", flush=True)
        for (rname, seed), df in datasets.items():
            try:
                stats = run_one(cfg, df)
            except Exception as exc:
                stats = {"trades": 0, "error": str(exc)[:60]}
            rows.append({
                "config": cfg.name,
                "strategy": cfg.strategy,
                "tp_pct": cfg.tp_pct,
                "sl_pct": cfg.sl_pct,
                "filter": cfg.filter_enabled,
                "regime": rname,
                "seed": seed,
                **stats,
            })

    df = pd.DataFrame(rows)
    df.to_csv("reports/sweep_raw.csv", index=False)

    # Aggregate over seeds and regimes per config.
    agg = (
        df.groupby("config")
        .agg(
            trades=("trades", "mean"),
            win_rate=("win_rate_pct", "mean"),
            return_pct=("total_return_pct", "mean"),
            max_dd=("max_drawdown_pct", "mean"),
            profit_factor=("profit_factor", "mean"),
            sharpe=("sharpe", "mean"),
            return_std=("total_return_pct", "std"),
        )
        .round(2)
    )
    # Filter out configs that barely traded — WR is noise.
    meaningful = agg[agg["trades"] >= 5].copy()
    meaningful["score"] = (
        meaningful["win_rate"] / 100 * 0.4
        + meaningful["return_pct"] / 100 * 0.4
        + meaningful["profit_factor"].clip(upper=5) / 5 * 0.2
    )
    meaningful = meaningful.sort_values("score", ascending=False)
    meaningful.to_csv("reports/sweep_summary.csv")

    print("\n=== TOP 15 BY COMPOSITE SCORE (trades >= 5 average) ===")
    print(meaningful.head(15).to_string())

    print("\n=== HIGHEST WIN RATE (trades >= 10) ===")
    top_wr = agg[agg["trades"] >= 10].sort_values("win_rate", ascending=False)
    print(top_wr.head(10).to_string())

    print("\n=== HIGHEST RETURN (trades >= 10) ===")
    top_ret = agg[agg["trades"] >= 10].sort_values("return_pct", ascending=False)
    print(top_ret.head(10).to_string())

    print("\n=== MOST CONSISTENT (low return std, decent return, trades >= 10) ===")
    consistent = agg[(agg["trades"] >= 10) & (agg["return_pct"] > 0)].sort_values("return_std")
    print(consistent.head(10).to_string())


if __name__ == "__main__":
    main()
