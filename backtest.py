#!/usr/bin/env python3
"""Entry point for backtesting a strategy on historical data."""
from __future__ import annotations

import json
from pathlib import Path

import click
from rich.table import Table

from bot.backtest import build_backtester
from bot.config import Config
from bot.logger import console, setup_logging

STATS_PATH = Path("reports/backtest_stats.json")


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
@click.option("--no-cache", is_flag=True, default=False,
              help="Skip the on-disk OHLCV cache and re-download everything.")
@click.option("--parallel/--no-parallel", default=True,
              help="Run walk-forward windows in a process pool. Roughly Nx "
                   "faster on N cores; falls back to serial if subprocess "
                   "creation fails. No effect on a single-pass backtest "
                   "(those are inherently sequential).")
@click.option("--workers", type=int, default=None,
              help="Override worker count for --parallel (default: one per CPU).")
def main(config_path: str, days: int, no_csv: bool, windows: int,
         walk_forward: int, no_cache: bool, parallel: bool, workers: int | None) -> None:
    cfg = Config.load(config_path)
    setup_logging(cfg.logging.level)
    bt = build_backtester(cfg)

    if walk_forward > 1:
        data = bt.load_data(days, use_cache=not no_cache)
        wf = bt.walk_forward(
            days=days, n_windows=walk_forward, data=data,
            parallel=parallel, max_workers=workers,
        )
        wtable = Table(title=f"Walk-forward: {walk_forward} windows over {days}d")
        cols = ("window", "start", "end", "total_return_pct", "sharpe", "max_drawdown_pct", "trades")
        for col in cols:
            wtable.add_column(col)
        for row in wf["windows"]:
            wtable.add_row(*[str(row.get(c, "")) for c in cols])
        console().print(wtable)
        console().print(json.dumps(wf["summary"], indent=2))
        return

    result = bt.run(
        days=days, write_csv=not no_csv,
        data=bt.load_data(days, use_cache=not no_cache),
    )
    stats = result.stats(cfg.trading.starting_capital)
    if not no_csv and stats:
        STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATS_PATH.write_text(json.dumps(stats, indent=2))

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

    # Bootstrap-CI on the trade P&L sequence. Tells you whether the
    # backtest's headline return is statistically distinguishable from
    # "lucky ordering of these specific trades". Cheap honest test;
    # nothing replaces walk-forward, but this is a useful add-on.
    boot = result.bootstrap_ci(n_resamples=2000, confidence=0.90)
    if boot["trades"] >= 2:
        verdict = (
            "[green]edge survives resampling[/green]" if boot["significant"]
            else "[yellow]return not distinguishable from luck[/yellow]"
        )
        console().print(
            f"\n[bold]Bootstrap CI[/bold] (90%, {boot['n_resamples']} resamples, "
            f"{boot['trades']} trades): "
            f"lo={boot['lo']:+.2f} median={boot['median']:+.2f} hi={boot['hi']:+.2f} "
            f"{verdict}",
            markup=True,
        )

    console().print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
