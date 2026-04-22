"""Stock-specific news features for backtest and live.

Two free, historical sources of "news"-like signal:

  * :func:`load_earnings_calendar` – upcoming/historical earnings dates
    per ticker via yfinance. Used to gate entries near earnings.

  * :func:`detect_news_gaps` – derives "news event" markers from price
    action: bars that gap > ``threshold_pct`` with above-average volume
    almost always reflect a real news event (earnings, guidance,
    M&A, lawsuit). Cheap proxy that's fully backtestable from OHLCV.

Both are cached on disk where appropriate so re-running a backtest
doesn't re-hit yfinance for every ticker.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger("bot.news_stocks")

EARNINGS_CACHE_DIR = Path("cache/earnings")
EARNINGS_TTL_SECONDS = 24 * 3600  # refresh daily


# ---------------------------------------------------------------------------
# Earnings calendar.
# ---------------------------------------------------------------------------
def _earnings_cache_path(symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace(".", "_")
    return EARNINGS_CACHE_DIR / f"{safe}.json"


def _read_earnings_cache(symbol: str) -> pd.DataFrame | None:
    path = _earnings_cache_path(symbol)
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > EARNINGS_TTL_SECONDS:
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if not data:
        return pd.DataFrame(columns=["date"])
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def _write_earnings_cache(symbol: str, df: pd.DataFrame) -> None:
    EARNINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {"date": pd.Timestamp(d).isoformat()}
        for d in df["date"].tolist()
    ]
    _earnings_cache_path(symbol).write_text(json.dumps(rows))


def fetch_earnings_dates(symbol: str) -> pd.DataFrame:
    """Return a DataFrame with one ``date`` column of earnings timestamps."""
    cached = _read_earnings_cache(symbol)
    if cached is not None:
        return cached
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed – returning empty earnings calendar")
        return pd.DataFrame(columns=["date"])
    try:
        ticker = yf.Ticker(symbol)
        raw = ticker.earnings_dates
    except Exception as exc:
        log.warning("earnings fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame(columns=["date"])
    if raw is None or raw.empty:
        df = pd.DataFrame(columns=["date"])
    else:
        df = pd.DataFrame(
            {"date": pd.to_datetime(raw.index, utc=True)}
        ).dropna().drop_duplicates().sort_values("date").reset_index(drop=True)
    _write_earnings_cache(symbol, df)
    return df


def load_earnings_calendar(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Bulk-load earnings dates for many symbols. Cached per ticker."""
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            out[sym] = fetch_earnings_dates(sym)
        except Exception as exc:
            log.warning("earnings load failed for %s: %s", sym, exc)
            out[sym] = pd.DataFrame(columns=["date"])
    return out


def days_until_earnings(
    earnings_df: pd.DataFrame, ts: pd.Timestamp
) -> int | None:
    """Whole days from ``ts`` to next earnings date in ``earnings_df``."""
    if earnings_df is None or earnings_df.empty:
        return None
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    upcoming = earnings_df[earnings_df["date"] >= ts]
    if upcoming.empty:
        return None
    delta = upcoming["date"].iloc[0] - ts
    return max(0, int(delta.total_seconds() // 86400))


# ---------------------------------------------------------------------------
# Price-action news proxy.
# ---------------------------------------------------------------------------
def detect_news_gaps(
    df: pd.DataFrame,
    threshold_pct: float = 0.05,
    volume_mult: float = 2.0,
    volume_lookback: int = 20,
) -> pd.Series:
    """Boolean series, ``True`` where the bar opens with a large gap on
    above-average volume. Such bars almost always correspond to real news.

    A "gap" is the absolute change between the previous close and this bar's
    open. ``threshold_pct = 0.05`` flags 5%+ moves; ``volume_mult = 2``
    additionally requires volume to be 2× the rolling mean.
    """
    if df is None or df.empty or "close" not in df.columns:
        return pd.Series([], dtype=bool)
    prev_close = df["close"].shift(1)
    gap_pct = (df["open"] - prev_close).abs() / prev_close
    avg_vol = df["volume"].rolling(volume_lookback).mean()
    big_gap = gap_pct >= threshold_pct
    big_vol = df["volume"] >= (avg_vol * volume_mult)
    return (big_gap & big_vol).fillna(False)


def is_bearish_gap(
    df: pd.DataFrame,
    threshold_pct: float = 0.05,
) -> pd.Series:
    """True where the bar opens with a large *negative* gap."""
    if df is None or df.empty:
        return pd.Series([], dtype=bool)
    prev_close = df["close"].shift(1)
    gap_pct = (df["open"] - prev_close) / prev_close
    return (gap_pct <= -abs(threshold_pct)).fillna(False)
