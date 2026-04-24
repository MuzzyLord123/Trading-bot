from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

log = logging.getLogger("bot.logger")

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
    """Append-only CSV writer keyed on a fixed header.

    Automatically handles header-schema migrations: if we're asked to append
    to an existing file whose header doesn't match the columns we now want
    to write, the old file is rotated aside with a dated suffix so a future
    pandas.read_csv can't choke on mismatched field counts, and a fresh
    file is started with the new header.
    """

    def __init__(self, path: str | Path, header: list[str]) -> None:
        self.path = Path(path)
        self.header = header
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_header()
            return
        existing_header = self._read_existing_header()
        if existing_header != header:
            self._rotate_and_reset(existing_header)

    def _write_header(self) -> None:
        with self.path.open("w", newline="") as f:
            csv.writer(f).writerow(self.header)

    def _read_existing_header(self) -> list[str] | None:
        try:
            with self.path.open("r", newline="") as f:
                reader = csv.reader(f)
                return next(reader, None)
        except OSError:
            return None

    def _rotate_and_reset(self, existing_header: list[str] | None) -> None:
        """Move the old file aside and start a fresh one with the new header."""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = self.path.with_name(f"{self.path.stem}.{stamp}{self.path.suffix}")
        try:
            self.path.rename(backup)
            log.warning(
                "CSV header changed for %s (had %s, now %s); rotated old file to %s",
                self.path, existing_header, self.header, backup.name,
            )
        except OSError as exc:
            # Best effort - if we can't rotate, overwrite so we're at least
            # not corrupting future reads. Data loss is the lesser evil vs.
            # a dashboard that crashes on every load.
            log.error("could not rotate %s (%s); overwriting", self.path, exc)
        self._write_header()

    def write(self, row: dict[str, Any]) -> None:
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self.header])


def console() -> Console:
    return _console
