"""Per-symbol stock news feed powered by Yahoo Finance.

Yahoo Finance exposes a free, unauthenticated news endpoint per ticker via
``yfinance.Ticker(symbol).news`` – a list of dicts with headline, publisher,
link and publish timestamp. This module wraps that with:

  * An on-disk JSON cache (``cache/news/<symbol>.json``) with a short TTL
    so the dashboard doesn't hammer Yahoo when a user clicks through
    500 tickers.
  * A uniform :class:`NewsItem` dataclass so the rest of the codebase
    doesn't need to know about yfinance's dict shape.
  * Fan-out :func:`fetch_news_bulk` that fetches news for many symbols
    in sequence, swallowing per-symbol failures (one offline ticker
    shouldn't blank the whole dashboard).

The yfinance news payload has drifted between library versions. The
extractor below reads both the legacy flat shape and the newer
``content`` nested shape, so upgrading yfinance won't silently break
the feed.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("bot.news_feed")

CACHE_DIR = Path("cache/news")
DEFAULT_MAX_AGE_MINUTES = 15


@dataclass
class NewsItem:
    symbol: str
    title: str
    publisher: str
    link: str
    published_at: float  # unix seconds, UTC
    summary: str = ""

    @property
    def published_dt(self) -> datetime:
        return datetime.fromtimestamp(self.published_at, tz=timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NewsItem":
        return cls(**d)


def _extract(raw: dict[str, Any], symbol: str) -> NewsItem | None:
    """Coerce a single yfinance news item into a NewsItem.

    Handles both the flat legacy layout (``title``, ``link``,
    ``providerPublishTime``, ``publisher``) and the newer nested
    ``content`` layout (``content.title``, ``content.canonicalUrl.url``,
    ``content.pubDate``, ``content.provider.displayName``).
    """
    content = raw.get("content") or raw

    title = content.get("title") or raw.get("title") or ""
    if not title:
        return None

    publisher = (
        (content.get("provider") or {}).get("displayName")
        or raw.get("publisher")
        or ""
    )

    link = (
        (content.get("canonicalUrl") or {}).get("url")
        or (content.get("clickThroughUrl") or {}).get("url")
        or raw.get("link")
        or ""
    )

    ts_raw = (
        content.get("pubDate")
        or raw.get("providerPublishTime")
        or raw.get("pubDate")
    )
    published_at = _parse_timestamp(ts_raw)

    summary = content.get("summary") or raw.get("summary") or ""

    return NewsItem(
        symbol=symbol,
        title=title.strip(),
        publisher=publisher.strip(),
        link=link.strip(),
        published_at=published_at,
        summary=summary.strip()[:500],
    )


def _parse_timestamp(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        # yfinance legacy returns unix seconds directly.
        return float(value)
    if isinstance(value, str):
        # Newer versions send ISO-8601 strings like "2025-03-12T14:22:00Z".
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _cache_path(symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"{safe}.json"


def _read_cache(symbol: str, max_age_minutes: int) -> list[NewsItem] | None:
    path = _cache_path(symbol)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > max_age_minutes * 60:
        return None
    try:
        raw = json.loads(path.read_text())
        return [NewsItem.from_dict(d) for d in raw]
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        log.debug("dropping bad news cache for %s: %s", symbol, exc)
        return None


def _write_cache(symbol: str, items: list[NewsItem]) -> None:
    path = _cache_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps([i.to_dict() for i in items], indent=0))
    except OSError as exc:
        log.warning("could not persist news cache for %s: %s", symbol, exc)


def fetch_news(
    symbol: str,
    limit: int = 10,
    max_age_minutes: int = DEFAULT_MAX_AGE_MINUTES,
    use_cache: bool = True,
) -> list[NewsItem]:
    """Return up to ``limit`` recent news items for ``symbol``.

    Results are served from the on-disk cache when fresher than
    ``max_age_minutes``. Returns an empty list on fetch failure – the
    caller should treat missing news as a normal condition.
    """
    if use_cache:
        cached = _read_cache(symbol, max_age_minutes)
        if cached is not None:
            return cached[:limit]

    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed – news feed disabled")
        return []

    try:
        raw_items = yf.Ticker(symbol).news or []
    except Exception as exc:
        log.warning("news fetch failed for %s: %s", symbol, exc)
        return []

    items: list[NewsItem] = []
    for raw in raw_items:
        try:
            item = _extract(raw, symbol)
        except Exception as exc:
            log.debug("skipping malformed news item for %s: %s", symbol, exc)
            continue
        if item is not None:
            items.append(item)

    items.sort(key=lambda i: i.published_at, reverse=True)
    _write_cache(symbol, items)
    return items[:limit]


def fetch_news_bulk(
    symbols: list[str],
    per_symbol_limit: int = 5,
    max_age_minutes: int = DEFAULT_MAX_AGE_MINUTES,
    total_limit: int | None = None,
) -> list[NewsItem]:
    """Fetch news for many symbols and merge into a single time-sorted list.

    Useful for a dashboard view over the full trading universe. Individual
    symbol failures are logged but don't abort the run.
    """
    out: list[NewsItem] = []
    for symbol in symbols:
        out.extend(
            fetch_news(symbol, limit=per_symbol_limit, max_age_minutes=max_age_minutes)
        )
    out.sort(key=lambda i: i.published_at, reverse=True)
    if total_limit is not None:
        return out[:total_limit]
    return out


def clear_cache(symbol: str | None = None) -> None:
    """Delete the on-disk news cache for one symbol or all symbols."""
    if symbol is None:
        if not CACHE_DIR.exists():
            return
        for p in CACHE_DIR.glob("*.json"):
            p.unlink(missing_ok=True)
        return
    _cache_path(symbol).unlink(missing_ok=True)
