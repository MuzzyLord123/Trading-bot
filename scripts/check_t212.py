#!/usr/bin/env python3
"""Read-only smoke test for the Trading 212 integration.

Runs through every API call the live engine makes on startup, in order,
without ever placing an order. Use this once after generating a fresh
T212 API key to confirm:

  * The key is valid (account/cash returns 200).
  * The bot can read your positions.
  * The instrument catalogue loads.
  * Every symbol in your config.yaml resolves to a real T212 ticker.

Exit codes:
  0  everything OK
  1  configuration error (missing key, bad config)
  2  T212 API failure (auth, network, server-side)
  3  one or more configured symbols cannot be resolved on T212

Usage
-----
    python scripts/check_t212.py                   # uses .env + config.yaml
    python scripts/check_t212.py --sandbox         # force demo URL
    python scripts/check_t212.py --symbols AAPL,MSFT,VWRL.L
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.table import Table

from bot.config import Config
from bot.logger import console, setup_logging
from bot.stocks import T212APIError, Trading212Broker


def _resolve_api_key(explicit: str | None) -> str:
    """Pick the API key from the explicit flag, env var, or .env file."""
    if explicit:
        return explicit
    load_dotenv()
    return os.getenv("TRADING212_API_KEY", "").strip()


def _step(label: str) -> None:
    console().print(f"[bold]>[/bold] {label} ...", end="")


def _ok() -> None:
    console().print(" [green]OK[/green]")


def _fail(reason: str) -> None:
    console().print(f" [red]FAIL[/red] {reason}")


@click.command()
@click.option("--config", "config_path", default="config.yaml",
              help="Config file to read trading.symbols and exchange.sandbox from.")
@click.option("--api-key", default=None,
              help="Override the .env TRADING212_API_KEY (handy for CI).")
@click.option("--sandbox/--no-sandbox", default=None,
              help="Force demo (sandbox) or live URL. Defaults to config.exchange.sandbox.")
@click.option("--symbols", default=None,
              help="Comma-separated symbols to verify resolution for "
                   "(default: trading.symbols from config).")
def main(config_path: str, api_key: str | None, sandbox: bool | None, symbols: str | None) -> None:
    setup_logging("WARNING")

    api_key = _resolve_api_key(api_key)
    if not api_key:
        console().print(
            "[red]No API key found.[/red] Set TRADING212_API_KEY in .env or pass --api-key."
        )
        sys.exit(1)

    # Determine sandbox flag without crashing if config.yaml is absent.
    cfg_path = Path(config_path)
    target_symbols: list[str] = []
    sandbox_flag = sandbox
    if cfg_path.exists():
        try:
            cfg = Config.load(cfg_path)
            if sandbox_flag is None:
                sandbox_flag = cfg.exchange.sandbox
            target_symbols = list(cfg.trading.symbols)
        except Exception as exc:
            console().print(f"[yellow]Config read failed ({exc}); proceeding without it.[/yellow]")
    if sandbox_flag is None:
        sandbox_flag = False
    if symbols:
        target_symbols = [s.strip() for s in symbols.split(",") if s.strip()]

    env_label = "DEMO" if sandbox_flag else "LIVE"
    masked = f"{api_key[:4]}{'.' * 8}{api_key[-4:]}" if len(api_key) > 8 else "*" * len(api_key)
    console().print(
        f"\n[bold]Trading 212 connectivity check[/bold]\n"
        f"  environment: {env_label}\n"
        f"  api key    : {masked}\n"
        f"  symbols    : {len(target_symbols) if target_symbols else 0} from "
        f"{cfg_path.name if cfg_path.exists() else 'flag'}\n"
    )

    broker = Trading212Broker(api_key=api_key, sandbox=sandbox_flag, request_interval_s=2.0)

    # ---- Step 1: account/cash. This validates auth + connectivity. ----
    _step("GET /equity/account/cash (auth + connectivity)")
    try:
        cash = broker.cash()
    except T212APIError as exc:
        _fail(f"{exc.status} - {'auth' if exc.is_auth_error else 'API error'}")
        if exc.is_auth_error:
            console().print(
                "[red]The API key was rejected.[/red] Likely causes:\n"
                "  - The key is for the wrong environment (live key on demo URL or vice versa).\n"
                "  - The key was rotated or revoked in the T212 app.\n"
                "  - Approval is still pending (T212 takes a few days to enable API access).\n"
            )
        else:
            console().print(f"[dim]Body:[/dim] {exc.body[:400]}")
        sys.exit(2)
    except Exception as exc:
        _fail(str(exc)[:120])
        sys.exit(2)
    _ok()
    console().print(f"  free cash: [green]{cash:,.2f}[/green]")

    # ---- Step 2: positions. ----
    _step("GET /equity/portfolio (open positions)")
    try:
        positions = broker.positions()
    except T212APIError as exc:
        _fail(f"{exc.status}")
        console().print(f"[dim]Body:[/dim] {exc.body[:400]}")
        sys.exit(2)
    _ok()
    console().print(f"  open positions: {len(positions)}")
    if positions:
        ptable = Table(title="Currently held")
        for col in ("ticker", "quantity", "avg price", "current price", "unrealised"):
            ptable.add_column(col, justify="right" if col != "ticker" else "left")
        for p in positions:
            unreal = (p.current_price - p.average_price) * p.quantity
            ptable.add_row(
                p.ticker,
                f"{p.quantity:.4f}",
                f"{p.average_price:,.4f}",
                f"{p.current_price:,.4f}",
                f"{unreal:+,.2f}",
            )
        console().print(ptable)

    # ---- Step 3: instruments. Slow on first call (2-5MB payload). ----
    _step("GET /equity/metadata/instruments (catalogue)")
    try:
        instruments = broker.load_instruments()
    except T212APIError as exc:
        _fail(f"{exc.status}")
        console().print(f"[dim]Body:[/dim] {exc.body[:400]}")
        sys.exit(2)
    _ok()
    console().print(f"  instruments listed: {len(instruments)}")

    # ---- Step 4: resolve every configured symbol. ----
    if not target_symbols:
        console().print(
            "\n[yellow]No symbols to resolve - skipping resolution check. "
            "Pass --symbols or run with a populated config.yaml.[/yellow]"
        )
        sys.exit(0)

    _step(f"Resolving {len(target_symbols)} symbol(s) to T212 tickers")
    unresolved: list[str] = []
    table = Table(title="Symbol resolution")
    table.add_column("symbol")
    table.add_column("T212 ticker")
    table.add_column("status")
    # Avoid spamming output for huge universes.
    show = target_symbols if len(target_symbols) <= 25 else target_symbols[:25]
    for sym in target_symbols:
        resolved = broker.resolve_ticker(sym)
        ok = resolved != sym or resolved.upper() in broker._ticker_map  # sym is its own T212 ticker
        if not ok:
            unresolved.append(sym)
        if sym in show:
            table.add_row(sym, resolved, "[green]ok[/green]" if ok else "[red]MISSING[/red]")
    if len(target_symbols) > 25:
        table.caption = f"showing first 25 of {len(target_symbols)} symbols"
    _ok() if not unresolved else _fail(f"{len(unresolved)} unresolved")
    console().print(table)

    if unresolved:
        console().print(
            f"\n[red]{len(unresolved)} symbol(s) did not match any T212 instrument:[/red]\n"
            f"  {', '.join(unresolved[:10])}"
            + (" ..." if len(unresolved) > 10 else "")
        )
        console().print(
            "[yellow]Action:[/yellow] either add them to "
            "trading.excluded_symbols in config.yaml or check whether T212 "
            "actually lists them (some Yahoo tickers are not tradable on T212)."
        )
        sys.exit(3)

    console().print(
        "\n[bold green]All checks passed.[/bold green] "
        "The bot can authenticate, read positions and resolve every configured symbol.\n"
        "[dim]This script never places an order. To actually trade, set "
        "trading.mode: live in config.yaml.[/dim]"
    )


if __name__ == "__main__":
    main()
