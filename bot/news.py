"""Crypto news feed + classifier.

Polls RSS from major free crypto news sources and classifies each headline
by (symbol, sentiment, importance). Used by the live engine to act on
major news events. LIVE-MODE ONLY – not used in backtest (historical news
is not available via free feeds).

Sources:
  - CoinDesk: https://www.coindesk.com/arc/outboundfeeds/rss/
  - Cointelegraph: https://cointelegraph.com/rss

Classification is keyword-based (no LLM, no API key). Not perfect, but:
  - Major events ("SEC approves", "hacked for $X", "lawsuit", "halving")
    are reliably detected from headlines alone
  - Ambiguous headlines score as neutral (no action)
"""
from __future__ import annotations

import logging
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("bot.news")

FEEDS: list[str] = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
]

# Asset name -> canonical symbol fragment. Matched case-insensitively in
# headlines. The engine then expands to "<SYM>/<QUOTE>" based on its symbol list.
ASSET_ALIASES: dict[str, list[str]] = {
    "BTC": ["bitcoin", "btc", "satoshi"],
    "ETH": ["ethereum", "ether ", "eth ", "vitalik"],
    "SOL": ["solana", "sol "],
    "ADA": ["cardano", "ada "],
    "DOT": ["polkadot", "dot "],
    "XRP": ["ripple", "xrp"],
    "LINK": ["chainlink", "link "],
    "LTC": ["litecoin", "ltc"],
    "AVAX": ["avalanche", "avax"],
    "DOGE": ["dogecoin", "doge"],
    "MATIC": ["polygon", "matic"],
}

# Bullish and bearish keywords. Weights chosen so that a single strong word
# (hack, approval) clears the 5.0 "major" threshold.
BULL_KEYWORDS: dict[str, float] = {
    "approves": 6.0, "approved": 6.0, "approval": 6.0,
    "etf": 4.0, "greenlight": 5.0, "legalises": 6.0, "legalizes": 6.0,
    "partnership": 3.0, "integrates": 3.0, "adopts": 3.0, "adoption": 3.0,
    "halving": 5.0, "upgrade": 3.0, "launch": 2.0, "launches": 2.0,
    "surge": 2.0, "rally": 2.0, "breakout": 2.0, "all-time high": 4.0,
    "bullish": 2.0, "milestone": 2.0, "acquires": 3.0,
    "rate cut": 4.0, "inflow": 2.0, "inflows": 2.0,
}
BEAR_KEYWORDS: dict[str, float] = {
    "hack": 6.0, "hacked": 6.0, "exploit": 5.0, "exploited": 5.0, "breach": 5.0,
    "stolen": 4.0, "drained": 5.0,
    "lawsuit": 4.0, "sues": 3.0, "sec charges": 6.0, "indicted": 5.0,
    "bankruptcy": 6.0, "bankrupt": 6.0, "insolvent": 6.0, "collapse": 5.0,
    "ban": 4.0, "bans": 4.0, "banned": 4.0, "crackdown": 4.0, "outlaws": 5.0,
    "crash": 3.0, "plunge": 3.0, "plummet": 3.0, "tank": 2.0,
    "rate hike": 4.0, "outflow": 2.0, "outflows": 2.0,
    "rug pull": 6.0, "scam": 3.0, "fraud": 4.0,
    "delisted": 4.0, "halt": 3.0, "halted": 3.0,
}


@dataclass(frozen=True)
class NewsItem:
    id: str  # stable identifier for dedup
    timestamp: datetime
    title: str
    asset: str  # canonical asset (e.g. "BTC"), or "" if unattributable
    sentiment: float  # positive = bullish, negative = bearish
    importance: float  # >= 0. Defensive actions fire above major_threshold.

    @property
    def is_bullish(self) -> bool:
        return self.sentiment > 0

    @property
    def is_bearish(self) -> bool:
        return self.sentiment < 0


def _http_get(url: str, timeout: float = 10.0) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "trading-bot/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        log.warning("news fetch failed %s: %s", url, exc)
        return None


def _parse_rss(xml_text: str, source: str) -> list[NewsItem]:
    out: list[NewsItem] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        log.warning("rss parse failed for %s: %s", source, exc)
        return out
    channel = root.find("channel") or root
    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        link = (item.findtext("link") or "").strip()
        pubdate_raw = (item.findtext("pubDate") or "").strip()
        ts = _parse_rfc822(pubdate_raw) or datetime.now(timezone.utc)
        item_id = f"{source}:{link or title}"
        asset = detect_asset(title)
        sentiment, importance = classify(title)
        out.append(
            NewsItem(
                id=item_id,
                timestamp=ts,
                title=title,
                asset=asset,
                sentiment=sentiment,
                importance=importance,
            )
        )
    return out


def _parse_rfc822(s: str) -> datetime | None:
    if not s:
        return None
    from email.utils import parsedate_to_datetime
    try:
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def detect_asset(title: str) -> str:
    lowered = title.lower()
    for symbol, aliases in ASSET_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                return symbol
    return ""


def classify(title: str) -> tuple[float, float]:
    """Return (sentiment, importance).

    Sentiment is positive for bullish, negative for bearish. Importance is
    the summed keyword weight regardless of direction – filter on this for
    "major" news.
    """
    lowered = " " + title.lower() + " "
    bull = sum(w for kw, w in BULL_KEYWORDS.items() if _kw_in(lowered, kw))
    bear = sum(w for kw, w in BEAR_KEYWORDS.items() if _kw_in(lowered, kw))
    return bull - bear, bull + bear


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _kw_in(text: str, kw: str) -> bool:
    # Simple substring match – keywords in the maps are phrased to avoid
    # false positives (e.g. "halving" not "half").
    return kw in text


def fetch_news(feeds: list[str] = FEEDS) -> list[NewsItem]:
    items: list[NewsItem] = []
    for url in feeds:
        body = _http_get(url)
        if body:
            items.extend(_parse_rss(body, source=url))
    items.sort(key=lambda i: i.timestamp, reverse=True)
    return items


@dataclass
class NewsSignal:
    symbol: str  # "BTC/GBP"
    direction: int  # +1 bullish, -1 bearish
    title: str
    timestamp: datetime
    importance: float


class NewsMonitor:
    """Deduplicated, symbol-aware view of the news feed.

    The engine calls ``scan()`` once per tick; it returns only *new*
    signals above the importance threshold whose asset matches a configured
    trading symbol.
    """

    def __init__(
        self,
        symbols: list[str],
        major_threshold: float = 5.0,
        max_age_minutes: int = 120,
        dedup_cache_path: str | Path = "cache/news_seen.txt",
    ) -> None:
        self.symbols = symbols
        self.major_threshold = major_threshold
        self.max_age = timedelta(minutes=max_age_minutes)
        self.dedup_path = Path(dedup_cache_path)
        self._seen: set[str] = self._load_seen()

    def _load_seen(self) -> set[str]:
        if not self.dedup_path.exists():
            return set()
        try:
            return set(self.dedup_path.read_text().splitlines())
        except Exception:
            return set()

    def _save_seen(self) -> None:
        self.dedup_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Cap to last 2000 ids to avoid unbounded growth.
            lines = list(self._seen)[-2000:]
            self.dedup_path.write_text("\n".join(lines))
        except Exception as exc:
            log.warning("could not persist news dedup cache: %s", exc)

    def _symbol_for_asset(self, asset: str) -> str | None:
        if not asset:
            return None
        for sym in self.symbols:
            base = sym.split("/")[0].upper()
            if base == asset:
                return sym
        return None

    def scan(self) -> list[NewsSignal]:
        now = datetime.now(timezone.utc)
        signals: list[NewsSignal] = []
        new_ids: list[str] = []
        for item in fetch_news():
            if item.id in self._seen:
                continue
            new_ids.append(item.id)
            if now - item.timestamp > self.max_age:
                continue
            if item.importance < self.major_threshold:
                continue
            if item.sentiment == 0:
                continue
            symbol = self._symbol_for_asset(item.asset)
            if symbol is None:
                continue
            signals.append(
                NewsSignal(
                    symbol=symbol,
                    direction=1 if item.is_bullish else -1,
                    title=item.title,
                    timestamp=item.timestamp,
                    importance=item.importance,
                )
            )
        if new_ids:
            self._seen.update(new_ids)
            self._save_seen()
        return signals
