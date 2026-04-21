from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

_console = Console()


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(console=_console, rich_tracebacks=True, show_path=False)],
    )
    return logging.getLogger("bot")


class CsvLogger:
    """Append-only CSV writer keyed on a fixed header."""

    def __init__(self, path: str | Path, header: list[str]) -> None:
        self.path = Path(path)
        self.header = header
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(header)

    def write(self, row: dict[str, Any]) -> None:
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self.header])


def console() -> Console:
    return _console
