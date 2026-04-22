from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bot.news import NewsItem, NewsMonitor, classify, detect_asset


def test_detect_asset_matches_aliases():
    assert detect_asset("Bitcoin surges past 60k") == "BTC"
    assert detect_asset("Vitalik unveils Ethereum roadmap") == "ETH"
    assert detect_asset("Solana mainnet outage") == "SOL"
    assert detect_asset("No crypto mentioned here") == ""


def test_classify_bullish_reaches_major():
    sentiment, importance = classify("SEC approves spot Bitcoin ETF in surprise decision")
    assert sentiment > 0
    assert importance >= 5.0


def test_classify_bearish_reaches_major():
    sentiment, importance = classify("Major exchange hacked for $200M, funds drained")
    assert sentiment < 0
    assert importance >= 5.0


def test_classify_neutral_headline_scores_low():
    sentiment, importance = classify("Bitcoin price continues sideways into weekend")
    assert importance < 5.0


def test_news_monitor_filters_by_symbol_and_threshold(tmp_path):
    symbols = ["BTC/GBP", "ETH/GBP"]
    monitor = NewsMonitor(
        symbols=symbols,
        major_threshold=5.0,
        max_age_minutes=120,
        dedup_cache_path=tmp_path / "seen.txt",
    )
    # Inject items directly via the internal cache test helper pattern –
    # we monkeypatch fetch_news to return our fixtures.
    now = datetime.now(timezone.utc)
    items = [
        NewsItem("a", now - timedelta(minutes=5), "SEC approves Bitcoin ETF", "BTC", 6.0, 6.0),
        NewsItem("b", now - timedelta(minutes=5), "Ethereum hacked for $50M", "ETH", -6.0, 6.0),
        NewsItem("c", now - timedelta(minutes=5), "Solana minor update", "SOL", 0.0, 0.0),
        NewsItem("d", now - timedelta(hours=5), "Bitcoin approves something long ago", "BTC", 5.0, 5.0),
        NewsItem("e", now - timedelta(minutes=5), "Unattributed crypto news hack", "", -6.0, 6.0),
    ]
    from bot import news as news_mod
    orig = news_mod.fetch_news
    news_mod.fetch_news = lambda feeds=None: items
    try:
        signals = monitor.scan()
    finally:
        news_mod.fetch_news = orig

    symbols_seen = {s.symbol for s in signals}
    directions = {(s.symbol, s.direction) for s in signals}
    assert "BTC/GBP" in symbols_seen
    assert "ETH/GBP" in symbols_seen
    assert ("BTC/GBP", 1) in directions
    assert ("ETH/GBP", -1) in directions
    # Stale item d ignored; unattributed e ignored; neutral c ignored.
    assert len(signals) == 2


def test_news_monitor_deduplicates_across_calls(tmp_path):
    monitor = NewsMonitor(
        symbols=["BTC/GBP"],
        major_threshold=5.0,
        max_age_minutes=120,
        dedup_cache_path=tmp_path / "seen.txt",
    )
    from bot import news as news_mod
    now = datetime.now(timezone.utc)
    items = [NewsItem("same", now, "SEC approves Bitcoin ETF", "BTC", 6.0, 6.0)]
    orig = news_mod.fetch_news
    news_mod.fetch_news = lambda feeds=None: items
    try:
        first = monitor.scan()
        second = monitor.scan()
    finally:
        news_mod.fetch_news = orig
    assert len(first) == 1
    assert second == []
