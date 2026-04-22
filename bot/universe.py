"""Stock-universe loaders.

Provides helpers to materialise a list of tickers from a short keyword
(``SP500``) so users can write::

    symbols:
      - SP500

in config.yaml and have the engine trade the whole index.

The S&P 500 list is fetched from Wikipedia and cached on disk. Wikipedia's
table is the canonical free source and updates quickly after rebalances.
Cache TTL is a week; delete the file to force a refresh.
"""
from __future__ import annotations

import logging
import time
from io import StringIO
from pathlib import Path

import pandas as pd

log = logging.getLogger("bot.universe")

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_CACHE = Path("cache/sp500.csv")
SP500_TTL_SECONDS = 7 * 24 * 3600  # one week

# Tokens that users can write in config.yaml -> trading.symbols.
UNIVERSE_TOKENS = {"SP500", "S&P500", "S&P 500", "SANDP500"}


def _fetch_sp500_from_wikipedia() -> list[str]:
    import urllib.request

    req = urllib.request.Request(
        SP500_URL, headers={"User-Agent": "trading-bot/1.0"}
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8")
    tables = pd.read_html(StringIO(html))
    # First table on that page is the constituent list.
    df = tables[0]
    if "Symbol" not in df.columns:
        raise RuntimeError("Unexpected Wikipedia layout – 'Symbol' column missing")
    # Wikipedia uses e.g. "BRK.B"; yfinance uses "BRK-B".
    symbols = (
        df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False).tolist()
    )
    return [s for s in symbols if s]


def load_sp500(force_refresh: bool = False) -> list[str]:
    """Return the current S&P 500 constituent tickers, yfinance-formatted."""
    SP500_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not force_refresh and SP500_CACHE.exists():
        age = time.time() - SP500_CACHE.stat().st_mtime
        if age < SP500_TTL_SECONDS:
            try:
                return pd.read_csv(SP500_CACHE)["symbol"].astype(str).tolist()
            except Exception as exc:
                log.warning("S&P 500 cache unreadable, refetching: %s", exc)
    try:
        symbols = _fetch_sp500_from_wikipedia()
    except Exception as exc:
        log.warning("S&P 500 fetch failed (%s); using cached copy if any", exc)
        if SP500_CACHE.exists():
            return pd.read_csv(SP500_CACHE)["symbol"].astype(str).tolist()
        raise
    pd.DataFrame({"symbol": symbols}).to_csv(SP500_CACHE, index=False)
    log.info("Fetched %d S&P 500 tickers", len(symbols))
    return symbols


def expand_universe_tokens(symbols: list[str]) -> list[str]:
    """Replace universe keywords in a symbol list with real tickers."""
    out: list[str] = []
    seen: set[str] = set()
    for sym in symbols:
        up = str(sym).upper().strip()
        if up in UNIVERSE_TOKENS:
            try:
                expanded = load_sp500()
            except Exception as exc:
                log.error("Could not expand %s: %s", sym, exc)
                continue
            for t in expanded:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
        else:
            if sym not in seen:
                seen.add(sym)
                out.append(sym)
    return out
