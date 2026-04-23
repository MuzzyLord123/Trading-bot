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
              help="Number of equal-sized windows for equity-curve stability stats (0 to skip).")
@click.option("--walk-forward", type=int, default=0, show_default=True,
              help="If > 1, run a walk-forward backtest with fresh portfolio per window.")
def main(config_path: str, days: int, no_csv: bool, windows: int, walk_forward: int) -> None:
    cfg = Config.load(config_path)
    setup_logging(cfg.logging.level)
    bt = build_backtester(cfg)

    if walk_forward > 1:
        data = bt.load_data(days)
        wf = bt.walk_forward(days=days, n_windows=walk_forward, data=data)
        wtable = Table(title=f"Walk-forward: {walk_forward} windows over {days}d")
        cols = ("window", "start", "end", "total_return_pct", "sharpe", "max_drawdown_pct", "trades")
        for col in cols:
            wtable.add_column(col)
        for row in wf["windows"]:
            wtable.add_row(*[str(row.get(c, "")) for c in cols])
        console().print(wtable)
        console().print(json.dumps(wf["summary"], indent=2))
        return

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
