"""Stock-universe loaders.

Provides helpers to materialise a list of tickers from a short keyword
(``SP500``, ``NASDAQ100``) so users can write::

    symbols:
      - SP500
      - NASDAQ100

in config.yaml and have the engine trade the whole index.

Lists are fetched from Wikipedia and cached on disk. Wikipedia is the
canonical free source and updates quickly after index rebalances.
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

NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
NASDAQ100_CACHE = Path("cache/nasdaq100.csv")

# Canonical Nasdaq-100 constituent tickers (late-2025 composition).
# Used as a reliable fallback when Wikipedia is unreachable or its page
# layout changes. Refreshed on every successful Wikipedia fetch — this
# list is only consulted when the live fetch fails AND no cache exists.
NASDAQ100_HARDCODED: list[str] = [
    "ADBE", "ADP", "AMD", "ABNB", "ALNY", "GOOGL", "GOOG", "AMZN", "AEP",
    "AMGN", "ADI", "AAPL", "AMAT", "APP", "ARM", "ASML", "TEAM", "ADSK",
    "AXON", "BKR", "BKNG", "AVGO", "CDNS", "CHTR", "CTAS", "CSCO", "CCEP",
    "CTSH", "CMCSA", "CEG", "CPRT", "CSGP", "COST", "CRWD", "CSX", "DDOG",
    "DXCM", "FANG", "DASH", "EA", "EXC", "FAST", "FER", "FTNT", "GEHC",
    "GILD", "HON", "IDXX", "INSM", "INTC", "INTU", "ISRG", "KDP", "KLAC",
    "KHC", "LRCX", "LIN", "MAR", "MRVL", "MELI", "META", "MCHP", "MU",
    "MSFT", "MSTR", "MDLZ", "MPWR", "MNST", "NFLX", "NVDA", "NXPI", "ODFL",
    "ORLY", "PCAR", "PLTR", "PANW", "PAYX", "PYPL", "PDD", "PEP", "QCOM",
    "REGN", "ROP", "ROST", "STX", "SHOP", "SBUX", "SNPS", "TTWO", "TSLA",
    "TXN", "TRI", "TMUS", "VRSK", "VRTX", "WMT", "WBD", "WDC", "WDAY",
    "XEL", "ZS",
]

CACHE_TTL_SECONDS = 7 * 24 * 3600  # one week
# kept for backwards compat with old import sites
SP500_TTL_SECONDS = CACHE_TTL_SECONDS

# Tokens that users can write in config.yaml -> trading.symbols.
_SP500_TOKENS = {"SP500", "S&P500", "S&P 500", "SANDP500"}
_NASDAQ100_TOKENS = {"NASDAQ100", "NDX", "NASDAQ-100"}
UNIVERSE_TOKENS = _SP500_TOKENS | _NASDAQ100_TOKENS


def _get_url(url: str) -> str:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "trading-bot/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8")


def _extract_symbols(html: str, column_candidates: tuple[str, ...]) -> list[str]:
    """Parse HTML table and return a normalised list of yfinance tickers."""
    tables = pd.read_html(StringIO(html))
    for df in tables:
        for col in column_candidates:
            if col in df.columns:
                symbols = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.replace(".", "-", regex=False)
                    .tolist()
                )
                return [s for s in symbols if s and s.lower() != "nan"]
    raise RuntimeError(
        f"None of {column_candidates} found in any table on the page"
    )


def _load_cached(cache_path: Path) -> list[str] | None:
    if not cache_path.exists():
        return None
    if time.time() - cache_path.stat().st_mtime >= CACHE_TTL_SECONDS:
        return None
    try:
        return pd.read_csv(cache_path)["symbol"].astype(str).tolist()
    except Exception as exc:
        log.warning("cache unreadable at %s: %s", cache_path, exc)
        return None


def _write_cache(cache_path: Path, symbols: list[str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": symbols}).to_csv(cache_path, index=False)


def load_sp500(force_refresh: bool = False) -> list[str]:
    """Return the current S&P 500 constituent tickers, yfinance-formatted."""
    if not force_refresh:
        cached = _load_cached(SP500_CACHE)
        if cached is not None:
            return cached
    try:
        html = _get_url(SP500_URL)
        symbols = _extract_symbols(html, ("Symbol",))
    except Exception as exc:
        log.warning("S&P 500 fetch failed (%s); using cached copy if any", exc)
        if SP500_CACHE.exists():
            return pd.read_csv(SP500_CACHE)["symbol"].astype(str).tolist()
        raise
    _write_cache(SP500_CACHE, symbols)
    log.info("Fetched %d S&P 500 tickers", len(symbols))
    return symbols


def load_nasdaq100(force_refresh: bool = False) -> list[str]:
    """Return the current Nasdaq-100 constituent tickers, yfinance-formatted.

    Resolution order:
      1. Fresh on-disk cache (written on any previous successful load).
      2. Wikipedia (primary source – stays current with rebalances).
      3. Stale cache if present (Wikipedia failed).
      4. Hardcoded fallback list (last-resort, always available).
    """
    if not force_refresh:
        cached = _load_cached(NASDAQ100_CACHE)
        if cached is not None:
            return cached
    try:
        html = _get_url(NASDAQ100_URL)
        # Wikipedia's Nasdaq-100 page uses "Ticker" (primary) but falls
        # back to "Symbol" historically.
        symbols = _extract_symbols(html, ("Ticker", "Symbol"))
        _write_cache(NASDAQ100_CACHE, symbols)
        log.info("Fetched %d Nasdaq-100 tickers from Wikipedia", len(symbols))
        return symbols
    except Exception as exc:
        log.warning(
            "Nasdaq-100 Wikipedia fetch failed (%s); using fallback", exc
        )
    if NASDAQ100_CACHE.exists():
        try:
            return pd.read_csv(NASDAQ100_CACHE)["symbol"].astype(str).tolist()
        except Exception:
            pass
    log.info("Using hardcoded Nasdaq-100 list (%d tickers)", len(NASDAQ100_HARDCODED))
    _write_cache(NASDAQ100_CACHE, NASDAQ100_HARDCODED)
    return list(NASDAQ100_HARDCODED)


def _expand_single_token(up: str) -> list[str] | None:
    if up in _SP500_TOKENS:
        return load_sp500()
    if up in _NASDAQ100_TOKENS:
        return load_nasdaq100()
    return None


def expand_universe_tokens(symbols: list[str]) -> list[str]:
    """Replace universe keywords in a symbol list with real tickers.

    Deduplicates across multiple index tokens so, for example,
    ``[SP500, NASDAQ100]`` yields the union with AAPL/MSFT/etc. listed once.
    """
    out: list[str] = []
    seen: set[str] = set()
    for sym in symbols:
        up = str(sym).upper().strip()
        expanded = None
        if up in UNIVERSE_TOKENS:
            try:
                expanded = _expand_single_token(up)
            except Exception as exc:
                log.error("Could not expand %s: %s", sym, exc)
                continue
        if expanded is not None:
            for t in expanded:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
        else:
            if sym not in seen:
                seen.add(sym)
                out.append(sym)
    return out

