from __future__ import annotations

from pathlib import Path

import pandas as pd

from bot.logger import CsvLogger


def test_new_file_gets_header(tmp_path):
    path = tmp_path / "t.csv"
    CsvLogger(path, ["a", "b"])
    assert path.read_text().strip() == "a,b"


def test_write_appends_row_in_header_order(tmp_path):
    path = tmp_path / "t.csv"
    log = CsvLogger(path, ["ts", "symbol", "pnl"])
    log.write({"symbol": "AAPL", "ts": "2024-01-01", "pnl": 1.25})
    lines = path.read_text().strip().splitlines()
    assert lines[0] == "ts,symbol,pnl"
    assert lines[1] == "2024-01-01,AAPL,1.25"


def test_missing_keys_become_empty_string(tmp_path):
    path = tmp_path / "t.csv"
    log = CsvLogger(path, ["a", "b", "c"])
    log.write({"a": 1})
    assert path.read_text().strip().splitlines()[-1] == "1,,"


def test_reopen_with_matching_header_is_idempotent(tmp_path):
    path = tmp_path / "t.csv"
    CsvLogger(path, ["a", "b"]).write({"a": 1, "b": 2})
    CsvLogger(path, ["a", "b"]).write({"a": 3, "b": 4})
    lines = path.read_text().strip().splitlines()
    assert lines == ["a,b", "1,2", "3,4"]  # no duplicated header, no rotation


def test_header_change_rotates_old_file(tmp_path):
    path = tmp_path / "t.csv"
    # v1 writes 2 columns.
    CsvLogger(path, ["a", "b"]).write({"a": 1, "b": 2})
    # v2 expects 4 columns - rotation happens on construction.
    CsvLogger(path, ["a", "b", "c", "d"]).write({"a": 10, "b": 20, "c": 30, "d": 40})
    # Old data moved aside with a dated backup name.
    backups = [p for p in tmp_path.iterdir() if p.name.startswith("t.") and p.suffix == ".csv" and p != path]
    assert len(backups) == 1
    assert backups[0].read_text().strip().splitlines() == ["a,b", "1,2"]
    # Current file has the new header + the new row only.
    current = path.read_text().strip().splitlines()
    assert current == ["a,b,c,d", "10,20,30,40"]


def test_rotated_file_is_still_pandas_parseable(tmp_path):
    path = tmp_path / "trades.csv"
    # Simulate the real bug: old header with 7 cols, then a new writer
    # upgrades to 10 cols, rotation should kick in.
    v1 = ["timestamp", "symbol", "side", "price", "amount", "pnl", "reason"]
    v2 = v1 + ["bars_held", "mfe", "mae"]
    log_v1 = CsvLogger(path, v1)
    for i in range(3):
        log_v1.write({c: i for c in v1})
    log_v2 = CsvLogger(path, v2)
    log_v2.write({c: 99 for c in v2})
    # Both files parse cleanly.
    new_df = pd.read_csv(path)
    assert list(new_df.columns) == v2
    assert len(new_df) == 1
    backups = [p for p in tmp_path.iterdir() if p.name.startswith("trades.") and p != path]
    assert len(backups) == 1
    old_df = pd.read_csv(backups[0])
    assert list(old_df.columns) == v1
    assert len(old_df) == 3
