#!/usr/bin/env python3
"""Compare the old vs improved ensemble on three synthetic regimes.

Synthetic data is NOT a substitute for real backtesting – it's a quick way to
check that the new filters are directionally improving win rate and not
breaking anything. Always run ``python backtest.py`` on real Kraken data
before trusting the numbers.
"""
from __future__ import annotations

import json

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


def _trending_market(n: int = 2000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = np.linspace(0, 1.8, n)
    noise = rng.normal(0, 0.005, n).cumsum()
    close = 100 * np.exp(drift + noise)
    return _to_ohlcv(close)


def _choppy_market(n: int = 1000, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cycle = 4 * np.sin(np.linspace(0, 30, n))
    noise = rng.normal(0, 1.0, n).cumsum() * 0.3
    close = 100 + cycle + noise
    return _to_ohlcv(close)


def _bear_market(n: int = 1000, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = np.linspace(0, -0.4, n)
    noise = rng.normal(0, 0.012, n).cumsum()
    close = 100 * np.exp(drift + noise)
    return _to_ohlcv(close)


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


def _base_cfg() -> Config:
    return Config(
        exchange=ExchangeConfig(name="kraken"),
        trading=TradingConfig(
            starting_capital=500.0,
            symbols=["SYN/GBP"],
            timeframe="1h",
            history_candles=1000,
        ),
        risk=RiskConfig(
            risk_per_trade=0.01,
            max_position_pct=0.5,
            max_open_positions=1,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            trailing_stop_pct=0.02,
            daily_loss_limit_pct=0.5,
            max_drawdown_pct=0.9,
            taker_fee_pct=0.0026,
            slippage_pct=0.0005,
            cooldown_bars_after_loss=0,
        ),
        strategy=StrategyConfig(
            name="ensemble",
            ensemble={
                "min_agreement": 2,
                "min_score": 0.0,
                "members": ["ma_crossover", "rsi_reversion", "macd", "bollinger"],
            },
            params={
                "ma_crossover": {"fast": 20, "slow": 50},
                "rsi_reversion": {"period": 14, "oversold": 30, "overbought": 70},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "bollinger": {"period": 20, "std": 2.0},
            },
            filter={"enabled": False},
        ),
        logging=LoggingConfig(),
        notifications=NotificationsConfig(),
        secrets={"api_key": "", "api_secret": "", "api_password": ""},
    )


def _improved_cfg() -> Config:
    cfg = _base_cfg()
    cfg.risk.cooldown_bars_after_loss = 12
    cfg.strategy.ensemble = {
        "min_agreement": 2,
        "min_score": 1.8,
        "members": ["ma_crossover", "rsi_reversion", "macd", "bollinger"],
        "weights": {
            "ma_crossover": 1.2,
            "macd": 1.2,
            "rsi_reversion": 1.0,
            "bollinger": 0.8,
        },
    }
    cfg.strategy.filter = {
        "enabled": True,
        "trend_ema": 50,
        "require_uptrend": True,
        "min_adx": 15.0,
        "adx_period": 14,
        "min_atr_pct": 0.0005,
        "atr_period": 14,
    }
    return cfg


def _run(cfg: Config, data: pd.DataFrame) -> dict:
    strategy = build_strategy_from_config(
        cfg.strategy.name,
        cfg.strategy.params,
        cfg.strategy.ensemble,
        cfg.strategy.filter,
    )
    bt = Backtester(cfg, exchange=None, strategy=strategy, risk=RiskManager(cfg.risk))
    result = bt.run(data={"SYN/GBP": data}, write_csv=False)
    return result.stats(cfg.trading.starting_capital)


def main() -> None:
    regimes = {
        "trending_up": _trending_market(),
        "choppy": _choppy_market(),
        "bear": _bear_market(),
    }
    variants = {"old": _base_cfg(), "new": _improved_cfg()}

    rows = []
    for regime_name, data in regimes.items():
        for variant_name, cfg in variants.items():
            stats = _run(cfg, data)
            rows.append({"regime": regime_name, "variant": variant_name, **stats})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    summary = {}
    for regime_name in regimes:
        sub = df[df["regime"] == regime_name].set_index("variant")
        summary[regime_name] = {
            "old_win_rate": sub.loc["old", "win_rate_pct"],
            "new_win_rate": sub.loc["new", "win_rate_pct"],
            "old_return": sub.loc["old", "total_return_pct"],
            "new_return": sub.loc["new", "total_return_pct"],
            "old_trades": int(sub.loc["old", "trades"]),
            "new_trades": int(sub.loc["new", "trades"]),
        }
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
