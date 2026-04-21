#!/usr/bin/env python3
"""Entry point for paper and live trading."""
from __future__ import annotations

import sys

import click

from bot.config import Config
from bot.engine import build_engine
from bot.logger import console, setup_logging


@click.command()
@click.option("--config", "config_path", default="config.yaml", show_default=True)
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Skip the interactive confirmation when starting live mode.",
)
def main(config_path: str, yes: bool) -> None:
    cfg = Config.load(config_path)
    log = setup_logging(cfg.logging.level)

    if cfg.trading.mode == "live":
        if not cfg.secrets["api_key"] or not cfg.secrets["api_secret"]:
            console().print(
                "[red]Live mode requires EXCHANGE_API_KEY and EXCHANGE_API_SECRET in .env[/red]"
            )
            sys.exit(1)
        if not yes:
            console().print(
                f"[yellow]About to trade LIVE on {cfg.exchange.name} with "
                f"{cfg.trading.starting_capital} {cfg.trading.quote_currency}.[/yellow]"
            )
            answer = input("Type 'LIVE' to continue: ").strip()
            if answer != "LIVE":
                console().print("Aborted.")
                sys.exit(0)
    elif cfg.trading.mode != "paper":
        raise click.BadParameter(
            f"trading.mode must be 'paper' or 'live', got '{cfg.trading.mode}'"
        )

    engine = build_engine(cfg)
    log.info("Loaded strategy: %s", cfg.strategy.name)
    engine.run_forever()


if __name__ == "__main__":
    main()
