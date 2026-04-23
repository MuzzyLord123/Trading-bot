#!/usr/bin/env python3
"""Entry point for backtesting a strategy on historical data."""
from __future__ import annotations

import json

import click
from rich.table import Table

from bot.backtest import build_backtester
from bot.config import Config
from bot.logger import console, setup_logging


@click.command()
@click.option("--config", "config_path", default="config.yaml", show_default=True)
@click.option("--days", type=int, default=180, show_default=True,
              help="How many days of history to replay.")
@click.option("--no-csv", is_flag=True, default=False,
              help="Skip writing reports/ CSVs.")
@click.option("--windows", type=int, default=4, show_default=True,
              help="Number of equal-sized windows for stability stats (0 to skip).")
def main(config_path: str, days: int, no_csv: bool, windows: int) -> None:
    cfg = Config.load(config_path)
    setup_logging(cfg.logging.level)
    bt = build_backtester(cfg)
    result = bt.run(days=days, write_csv=not no_csv)
    stats = result.stats(cfg.trading.starting_capital)

    table = Table(title=f"Backtest: {cfg.strategy.name} over {days}d")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in stats.items():
        table.add_row(key, str(value))
    console().print(table)

    if windows > 0:
        per_window = result.window_stats(n_windows=windows)
        if per_window:
            wtable = Table(title=f"Stability: {windows} equal windows")
            for col in ("window", "start", "end", "return_pct", "max_drawdown_pct", "sharpe"):
                wtable.add_column(col)
            for row in per_window:
                wtable.add_row(*[str(row[c]) for c in
                                 ("window", "start", "end", "return_pct", "max_drawdown_pct", "sharpe")])
            console().print(wtable)

    console().print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
