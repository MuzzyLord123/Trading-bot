#!/usr/bin/env python3
"""Compare a handful of opinionated config presets via walk-forward and
recommend the most STABLE one - not the highest-returning one.

Why not just pick the highest backtest return?
----------------------------------------------
Because that's curve-fitting. The config with the best single-pass
return is the one that happened to fit the noise in your particular
sample. It will almost certainly underperform out of sample.

What this script optimises for instead
--------------------------------------
For each preset we run a walk-forward backtest with N non-overlapping
windows. The "winner" is chosen by these tiebreakers, in order:

  1. Most windows with a positive return (consistency).
  2. Smallest worst-window drawdown (downside floor).
  3. Highest median Sharpe across windows (risk-adjusted skill).

Headline total return is reported but does not pick the winner. A
preset that returned 50% but lost in 3 of 4 windows is worse than one
that returned 10% in every window.

The presets are deliberately a small, opinionated set covering
distinct trading philosophies. Sweeping a parameter grid would land us
back in overfitting territory.

Usage
-----
    python scripts/recommend_config.py --days 365 --windows 4

The script does NOT modify your config.yaml. It prints a table and a
recommendation; you decide whether to adopt it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import click
from rich.table import Table

from bot.backtest import build_backtester
from bot.config import (
    Config,
    ExchangeConfig,
    LoggingConfig,
    NotificationsConfig,
    RiskConfig,
    StrategyConfig,
    TradingConfig,
)
from bot.logger import console, setup_logging


@dataclass
class Preset:
    """A named, self-contained trading philosophy."""
    key: str
    description: str
    strategy: dict[str, Any]
    risk_overrides: dict[str, Any] = field(default_factory=dict)


PRESETS: list[Preset] = [
    Preset(
        key="trend_classical",
        description="Trend-following ensemble (MA crossover + MACD + Donchian) with HTF confirmation.",
        strategy={
            "name": "ensemble",
            "ensemble": {
                "members": ["ma_crossover", "macd", "donchian"],
                "min_agreement": 2,
                "min_score": 1.0,
                "weights": {"ma_crossover": 1.0, "macd": 1.0, "donchian": 1.0},
            },
            "params": {
                "ma_crossover": {"fast": 8, "slow": 21},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "donchian": {"period": 20, "exit_period": 10},
            },
            "filter": {"enabled": True, "trend_ema": 50, "require_uptrend": True,
                        "min_adx": 18, "min_atr_pct": 0.001},
            "multi_timeframe": {"enabled": True, "require_long_on_htf": True},
        },
    ),
    Preset(
        key="mean_reversion",
        description="Mean-reverters (RSI + Bollinger + CCI) without trend confirmation.",
        strategy={
            "name": "ensemble",
            "ensemble": {
                "members": ["rsi_reversion", "bollinger", "cci"],
                "min_agreement": 2,
                "min_score": 1.0,
                "weights": {"rsi_reversion": 1.0, "bollinger": 1.0, "cci": 1.0},
            },
            "params": {
                "rsi_reversion": {"period": 14, "oversold": 30, "overbought": 70},
                "bollinger": {"period": 20, "std": 2.0},
                "cci": {"period": 20, "entry_level": -100, "exit_level": 100},
            },
            "filter": {"enabled": True, "trend_ema": 200, "require_uptrend": True,
                        "min_adx": 0, "min_atr_pct": 0.001},
        },
    ),
    Preset(
        key="all_strategies",
        description="Wide ensemble of all 9 strategies with broad filters.",
        strategy={
            "name": "ensemble",
            "ensemble": {
                "members": ["ma_crossover", "rsi_reversion", "macd", "bollinger",
                             "stochastic", "donchian", "keltner", "cci", "obv_trend"],
                "min_agreement": 3,
                "min_score": 2.0,
            },
            "params": {},
            "filter": {"enabled": True, "trend_ema": 100, "require_uptrend": True,
                        "min_adx": 15, "min_atr_pct": 0.001},
        },
    ),
    Preset(
        key="conservative",
        description="Single-strategy MACD with strict ATR stops, scale-out, max 3 positions.",
        strategy={
            "name": "macd",
            "params": {"macd": {"fast": 12, "slow": 26, "signal": 9}},
            "filter": {"enabled": True, "trend_ema": 200, "require_uptrend": True,
                        "min_adx": 25, "min_atr_pct": 0.002},
        },
        risk_overrides={
            "risk_per_trade": 0.005,
            "max_open_positions": 3,
            "max_position_pct": 0.05,
            "use_atr_stop": True,
            "atr_stop_multiplier": 2.5,
            "scale_out_at_r": 1.0,
            "scale_out_fraction": 0.5,
            "max_consecutive_losses": 3,
        },
    ),
    Preset(
        key="aggressive_breakout",
        description="Donchian + Keltner breakout duo, looser stops, more positions.",
        strategy={
            "name": "ensemble",
            "ensemble": {
                "members": ["donchian", "keltner"],
                "min_agreement": 1,
                "min_score": 0.5,
            },
            "params": {
                "donchian": {"period": 20, "exit_period": 10},
                "keltner": {"period": 20, "multiplier": 2.0, "atr_period": 10},
            },
            "filter": {"enabled": True, "trend_ema": 50, "require_uptrend": True,
                        "min_adx": 20, "min_atr_pct": 0.001},
        },
        risk_overrides={
            "risk_per_trade": 0.015,
            "max_open_positions": 8,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.15,
            "trailing_stop_pct": 0.04,
        },
    ),
    Preset(
        key="baseline_holdout",
        description="Buy-and-hold equivalent: ensemble that almost never exits, very loose stops.",
        strategy={
            "name": "ensemble",
            "ensemble": {
                "members": ["ma_crossover", "obv_trend"],
                "min_agreement": 1, "min_score": 0.5,
            },
            "params": {
                "ma_crossover": {"fast": 50, "slow": 200},
                "obv_trend": {"ema_period": 50, "lookback": 10, "trend_ema": 200},
            },
            "filter": {"enabled": False},
        },
        risk_overrides={
            "stop_loss_pct": 0.15,
            "take_profit_pct": 0.50,
            "trailing_stop_pct": 0.0,
            "time_stop_bars": 0,
            "max_position_pct": 0.20,
            "max_open_positions": 5,
        },
    ),
]


def _materialise(base: Config, preset: Preset) -> Config:
    """Build a fresh Config that combines the user's universe and account
    settings with the preset's strategy and risk overrides."""
    risk_kwargs = {**base.risk.__dict__, **preset.risk_overrides}
    risk = RiskConfig(**risk_kwargs)
    strategy = StrategyConfig(
        name=preset.strategy["name"],
        ensemble=preset.strategy.get("ensemble", {}),
        params=preset.strategy.get("params", {}),
        filter=preset.strategy.get("filter", {}),
        multi_timeframe=preset.strategy.get("multi_timeframe", {}),
    )
    return Config(
        exchange=ExchangeConfig(
            name=base.exchange.name, sandbox=base.exchange.sandbox,
        ),
        trading=TradingConfig(
            mode="paper",
            quote_currency=base.trading.quote_currency,
            starting_capital=base.trading.starting_capital,
            symbols=list(base.trading.symbols),
            timeframe=base.trading.timeframe,
            poll_interval_seconds=base.trading.poll_interval_seconds,
            history_candles=base.trading.history_candles,
            excluded_symbols=list(base.trading.excluded_symbols),
        ),
        risk=risk,
        strategy=strategy,
        logging=LoggingConfig(level="WARNING"),
        notifications=NotificationsConfig(),
        secrets=dict(base.secrets),
    )


def _evaluate(preset: Preset, base: Config, days: int, n_windows: int) -> dict[str, Any]:
    """Run walk-forward on one preset, return summary stats."""
    cfg = _materialise(base, preset)
    bt = build_backtester(cfg)
    try:
        wf = bt.walk_forward(days=days, n_windows=n_windows)
    except Exception as exc:
        return {"error": str(exc)}

    summary = wf["summary"]
    windows = wf["windows"]
    return {
        "preset": preset.key,
        "description": preset.description,
        "avg_return_pct": summary.get("avg_return_pct", 0.0),
        "median_return_pct": summary.get("median_return_pct", 0.0),
        "worst_return_pct": summary.get("worst_window_return_pct", 0.0),
        "profitable_windows": summary.get("profitable_windows", 0),
        "n_windows": summary.get("n_windows", 0),
        "avg_sharpe": summary.get("avg_sharpe", 0.0),
        "max_drawdown_pct": min(
            (w.get("max_drawdown_pct", 0.0) for w in windows), default=0.0,
        ),
        "trades_avg": (
            sum(w.get("trades", 0) for w in windows) / max(1, len(windows))
        ),
    }


def _rank(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort by stability tiebreakers, not headline return.

    Order of importance:
      1. profitable_windows / n_windows (consistency)
      2. worst_return_pct (downside floor)
      3. avg_sharpe (risk-adjusted skill)
      4. avg_return_pct (only as final tiebreaker)
    """
    def key(r: dict[str, Any]) -> tuple:
        if "error" in r:
            return (0, -999.0, -999.0, -999.0)
        consistency = r["profitable_windows"] / max(1, r["n_windows"])
        return (
            consistency,
            r["worst_return_pct"],
            r["avg_sharpe"],
            r["avg_return_pct"],
        )
    return sorted(results, key=key, reverse=True)


@click.command()
@click.option("--config", "config_path", default="config.yaml", show_default=True,
              help="Base config supplying universe, capital and timeframe.")
@click.option("--days", type=int, default=365, show_default=True,
              help="Total history window to walk-forward across.")
@click.option("--windows", type=int, default=4, show_default=True,
              help="Number of equal-sized walk-forward windows.")
@click.option("--only", type=str, default=None,
              help="Comma-separated list of preset keys to evaluate (default: all).")
def main(config_path: str, days: int, windows: int, only: str | None) -> None:
    base = Config.load(config_path)
    setup_logging("WARNING")

    selected = PRESETS
    if only:
        wanted = {k.strip() for k in only.split(",")}
        selected = [p for p in PRESETS if p.key in wanted]
        if not selected:
            raise click.ClickException(
                f"No preset matched {sorted(wanted)}. "
                f"Known: {[p.key for p in PRESETS]}"
            )

    console().print(
        f"\n[bold]Evaluating {len(selected)} preset(s) over {days}d "
        f"in {windows} walk-forward windows.[/bold]\n"
        f"Universe: {len(base.trading.symbols)} symbols. "
        f"This is going to make several yfinance calls per preset; expect "
        f"a couple of minutes.\n"
    )

    results = []
    for preset in selected:
        console().print(f"  evaluating [cyan]{preset.key}[/cyan]...")
        results.append(_evaluate(preset, base, days, windows))

    ranked = _rank(results)

    table = Table(title=f"Walk-forward comparison ({windows} windows over {days}d)")
    table.add_column("rank")
    table.add_column("preset")
    table.add_column("profitable_windows", justify="right")
    table.add_column("worst_window_pct", justify="right")
    table.add_column("avg_sharpe", justify="right")
    table.add_column("avg_return_pct", justify="right")
    table.add_column("avg_trades", justify="right")
    for i, r in enumerate(ranked, start=1):
        if "error" in r:
            table.add_row(str(i), r.get("preset", "?"), "", "ERROR", "", r["error"][:40], "")
            continue
        table.add_row(
            str(i),
            r["preset"],
            f"{r['profitable_windows']}/{r['n_windows']}",
            f"{r['worst_return_pct']:+.2f}%",
            f"{r['avg_sharpe']:.2f}",
            f"{r['avg_return_pct']:+.2f}%",
            f"{r['trades_avg']:.1f}",
        )
    console().print(table)

    winner = ranked[0]
    if "error" in winner:
        console().print("\n[red]No preset completed without error.[/red]\n")
        return

    consistency = winner["profitable_windows"] / max(1, winner["n_windows"])
    console().print(
        f"\n[bold green]Recommendation: {winner['preset']}[/bold green]\n"
        f"  description    : {winner['description']}\n"
        f"  consistency    : {winner['profitable_windows']}/"
        f"{winner['n_windows']} profitable windows ({consistency:.0%})\n"
        f"  worst window   : {winner['worst_return_pct']:+.2f}% return\n"
        f"  avg sharpe     : {winner['avg_sharpe']:.2f}\n"
        f"  headline return: {winner['avg_return_pct']:+.2f}% per window\n"
    )
    console().print(
        "[yellow]Caveat:[/yellow] this picks the preset that held up best "
        "on the last "
        f"{days} days. Markets change. Re-run quarterly. The most stable "
        "preset on history is the BEST GUESS for the near future, not a "
        "guarantee.\n"
    )

    console().print(
        "[dim]Full result JSON:[/dim] "
        + json.dumps([r for r in ranked], indent=2)
    )


if __name__ == "__main__":
    main()
