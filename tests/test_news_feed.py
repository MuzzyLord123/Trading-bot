from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from bot import news_feed
from bot.news_feed import (
    NewsItem,
    _extract,
    _parse_timestamp,
    clear_cache,
    fetch_news,
    fetch_news_bulk,
)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(news_feed, "CACHE_DIR", tmp_path / "news")
    yield


def test_parse_timestamp_handles_unix_seconds():
    assert _parse_timestamp(1700000000) == 1700000000.0
    assert _parse_timestamp(1700000000.5) == 1700000000.5


def test_parse_timestamp_handles_iso_8601():
    ts = _parse_timestamp("2024-03-15T12:00:00Z")
    assert ts > 0


def test_parse_timestamp_handles_none_and_junk():
    assert _parse_timestamp(None) == 0.0
    assert _parse_timestamp("not-a-date") == 0.0


def test_extract_reads_legacy_flat_shape():
    raw = {
        "title": "Apple beats earnings",
        "publisher": "Reuters",
        "link": "https://example.com/apple",
        "providerPublishTime": 1700000000,
    }
    item = _extract(raw, "AAPL")
    assert item is not None
    assert item.title == "Apple beats earnings"
    assert item.publisher == "Reuters"
    assert item.link == "https://example.com/apple"
    assert item.published_at == 1700000000.0
    assert item.symbol == "AAPL"


def test_extract_reads_new_nested_content_shape():
    raw = {
        "content": {
            "title": "Tesla announces split",
            "provider": {"displayName": "Bloomberg"},
            "canonicalUrl": {"url": "https://bloomberg.com/tesla"},
            "pubDate": "2024-03-15T12:00:00Z",
            "summary": "Details inside.",
        }
    }
    item = _extract(raw, "TSLA")
    assert item is not None
    assert item.title == "Tesla announces split"
    assert item.publisher == "Bloomberg"
    assert item.link == "https://bloomberg.com/tesla"
    assert item.summary == "Details inside."
    assert item.published_at > 0


def test_extract_returns_none_for_missing_title():
    assert _extract({"publisher": "x"}, "AAPL") is None


def test_fetch_news_uses_cache_when_fresh():
    items = [
        NewsItem(
            symbol="AAPL", title="t", publisher="p", link="l",
            published_at=1700000000.0, summary="",
        )
    ]
    news_feed._write_cache("AAPL", items)
    with patch.object(news_feed, "_extract") as extractor:
        got = fetch_news("AAPL", limit=10, max_age_minutes=60)
        assert extractor.call_count == 0  # cache hit, no parsing done
    assert len(got) == 1
    assert got[0].title == "t"


def test_fetch_news_ignores_stale_cache():
    # Write a cache file with a stale mtime.
    cache = news_feed._cache_path("AAPL")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([{
        "symbol": "AAPL", "title": "old", "publisher": "", "link": "",
        "published_at": 0.0, "summary": "",
    }]))
    old = time.time() - 3600  # 1 hour ago
    import os
    os.utime(cache, (old, old))

    fake_yf = type("F", (), {})()
    class FakeTicker:
        def __init__(self, _sym):
            self.news = [{"title": "fresh", "publisher": "P", "link": "L",
                          "providerPublishTime": 1700000000}]
    fake_yf.Ticker = FakeTicker

    with patch.dict("sys.modules", {"yfinance": fake_yf}):
        got = fetch_news("AAPL", max_age_minutes=1)  # stale > 1 min
    assert len(got) == 1
    assert got[0].title == "fresh"


def test_fetch_news_swallows_network_errors():
    fake_yf = type("F", (), {})()
    class FailingTicker:
        def __init__(self, _sym): pass
        @property
        def news(self):
            raise RuntimeError("network down")
    fake_yf.Ticker = FailingTicker
    with patch.dict("sys.modules", {"yfinance": fake_yf}):
        assert fetch_news("AAPL", use_cache=False) == []


def test_fetch_news_bulk_merges_and_sorts_by_time():
    items_a = [NewsItem("AAPL", "A1", "", "", 1000.0, "")]
    items_b = [NewsItem("TSLA", "T1", "", "", 2000.0, "")]
    news_feed._write_cache("AAPL", items_a)
    news_feed._write_cache("TSLA", items_b)
    merged = fetch_news_bulk(["AAPL", "TSLA"], per_symbol_limit=5)
    assert [i.title for i in merged] == ["T1", "A1"]  # newer first


def test_fetch_news_bulk_respects_total_limit():
    for sym in ("A", "B", "C"):
        news_feed._write_cache(sym, [
            NewsItem(sym, f"{sym}-1", "", "", 1000.0, ""),
            NewsItem(sym, f"{sym}-2", "", "", 900.0, ""),
        ])
    merged = fetch_news_bulk(["A", "B", "C"], per_symbol_limit=5, total_limit=2)
    assert len(merged) == 2


def test_clear_cache_single_symbol():
    news_feed._write_cache("AAPL", [NewsItem("AAPL", "t", "", "", 0.0, "")])
    assert news_feed._cache_path("AAPL").exists()
    clear_cache("AAPL")
    assert not news_feed._cache_path("AAPL").exists()
