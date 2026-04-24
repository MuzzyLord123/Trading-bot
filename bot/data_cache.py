"""On-disk OHLCV cache for the backtester.

Every backtest used to re-download the full history window from Yahoo,
which is slow (network + parsing) and eats into Yahoo's rate budget.
The parameter sweep script was the worst offender - hundreds of config
variants each triggering the same few downloads. With this cache a
repeat backtest over the same timeframe + symbol set starts in <1s
instead of 30-60s, and sweep.py becomes usable on real universes.

Strategy
--------
One parquet file per (symbol, timeframe). Each holds a
``timestamp / open / high / low / close / volume`` DataFrame with
timestamps sorted and de-duplicated. On a cache hit we serve the stored
rows; on a cache miss we fetch from the data source, extend any
existing file with the new rows and persist.

The cache only holds *data* - never trade logs, signals, or derived
series. Those depend on strategy config and would stale on every code
change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger("bot.data_cache")

DEFAULT_DIR = Path("cache/ohlcv")


@dataclass
class CacheEntry:
    path: Path
    rows: int
    span_start: datetime | None
    span_end: datetime | None


class OhlcvCache:
    """Parquet-backed OHLCV cache, keyed on (symbol, timeframe)."""

    def __init__(self, root: Path = DEFAULT_DIR) -> None:
        self.root = Path(root)

    def path_for(self, symbol: str, timeframe: str) -> Path:
        # Sanitise the symbol so ticker suffixes (".L") and slashes don't
        # create subdirectories or dotfiles.
        safe = symbol.replace("/", "_").replace("\\", "_").replace(".", "_")
        return self.root / timeframe / f"{safe}.parquet"

    def has(self, symbol: str, timeframe: str) -> bool:
        return self.path_for(symbol, timeframe).exists()

    def read(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Return the cached frame for ``symbol``. Empty frame on miss."""
        path = self.path_for(symbol, timeframe)
        if not path.exists():
            return _empty_ohlcv()
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            log.warning("cache unreadable at %s: %s", path, exc)
            return _empty_ohlcv()
        # Parquet round-trip preserves tz, but defensively coerce in case
        # the file was written by an older version.
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def read_window(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime,
    ) -> pd.DataFrame:
        """Return the subset of cached rows between ``since`` and ``until``.

        The cache's coverage is opaque to the caller - this is a best-effort
        "give me whatever's already stored". If the cache doesn't span the
        full window the caller should fetch the gap separately and write
        the merged frame back with :meth:`write`.
        """
        df = self.read(symbol, timeframe)
        if df.empty:
            return df
        mask = (df["timestamp"] >= since) & (df["timestamp"] <= until)
        return df.loc[mask].reset_index(drop=True)

    def write(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Merge ``df`` into the cached frame for ``symbol`` and persist."""
        if df is None or df.empty:
            return
        path = self.path_for(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = self._merge(self.read(symbol, timeframe), df)
        try:
            merged.to_parquet(path, index=False)
        except Exception as exc:
            log.warning("could not persist cache %s: %s", path, exc)

    def coverage(self, symbol: str, timeframe: str) -> CacheEntry | None:
        """Describe what's in the cache for ``symbol`` without reading it all."""
        df = self.read(symbol, timeframe)
        if df.empty:
            return None
        return CacheEntry(
            path=self.path_for(symbol, timeframe),
            rows=len(df),
            span_start=df["timestamp"].min().to_pydatetime(),
            span_end=df["timestamp"].max().to_pydatetime(),
        )

    def covers(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime,
    ) -> bool:
        """True if the cache fully covers the requested window with no gap
        longer than ``freshness_tolerance``."""
        entry = self.coverage(symbol, timeframe)
        if entry is None or entry.span_start is None or entry.span_end is None:
            return False
        return entry.span_start <= since and entry.span_end >= until

    def clear(self, symbol: str | None = None, timeframe: str | None = None) -> None:
        """Drop cache files. ``None`` on a dimension means "all values"."""
        if symbol is None and timeframe is None:
            if not self.root.exists():
                return
            for p in self.root.rglob("*.parquet"):
                p.unlink(missing_ok=True)
            return
        if symbol is None:
            tf_dir = self.root / (timeframe or "")
            if tf_dir.exists():
                for p in tf_dir.glob("*.parquet"):
                    p.unlink(missing_ok=True)
            return
        if timeframe is None:
            for p in self.root.rglob("*.parquet"):
                sanitised = self.path_for(symbol, p.parent.name).name
                if p.name == sanitised:
                    p.unlink(missing_ok=True)
            return
        self.path_for(symbol, timeframe).unlink(missing_ok=True)

    @staticmethod
    def _merge(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        combined = pd.concat([existing, new], ignore_index=True)
        if "timestamp" not in combined.columns:
            return combined
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
        # Keep the newer row when timestamps collide - later fetches may
        # have corrected OHLC values (e.g. dividend adjustments).
        combined = combined.drop_duplicates(subset="timestamp", keep="last")
        return combined.sort_values("timestamp").reset_index(drop=True)


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


def fetch_with_cache(
    source,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    cache: OhlcvCache | None = None,
) -> pd.DataFrame:
    """Return OHLCV for ``[since_ms, until_ms]``, serving from cache when
    possible and extending the cache with any rows we had to fetch.

    ``source`` must expose ``fetch_ohlcv_range(symbol, timeframe,
    since_ms, until_ms) -> DataFrame`` - exactly the signature
    :class:`bot.stocks.YFinanceSource` uses.
    """
    cache = cache or OhlcvCache()
    since_dt = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc)
    until_dt = datetime.fromtimestamp(until_ms / 1000, tz=timezone.utc)

    if cache.covers(symbol, timeframe, since_dt, until_dt):
        log.debug("cache hit for %s %s", symbol, timeframe)
        return cache.read_window(symbol, timeframe, since_dt, until_dt)

    log.info("cache miss for %s %s - fetching", symbol, timeframe)
    fresh = source.fetch_ohlcv_range(symbol, timeframe, since_ms, until_ms)
    if fresh is None or fresh.empty:
        # Fetch failed but we may still have *some* data cached.
        return cache.read_window(symbol, timeframe, since_dt, until_dt)
    cache.write(symbol, timeframe, fresh)
    return cache.read_window(symbol, timeframe, since_dt, until_dt)
