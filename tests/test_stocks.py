from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bot.stocks import (
    Trading212Broker,
    YFinanceSource,
    is_stock_market_open,
)


def test_is_stock_market_open_weekday_hours():
    # Tuesday 14:00 UTC -> US premarket overlap, UK session still open.
    dt = datetime(2026, 4, 21, 14, 0, tzinfo=timezone.utc)
    assert is_stock_market_open(dt) is True


def test_is_stock_market_open_weekend():
    # Saturday midday – no market.
    dt = datetime(2026, 4, 18, 14, 0, tzinfo=timezone.utc)
    assert is_stock_market_open(dt) is False


def test_is_stock_market_open_overnight():
    # Tuesday 03:00 UTC – all major markets closed.
    dt = datetime(2026, 4, 21, 3, 0, tzinfo=timezone.utc)
    assert is_stock_market_open(dt) is False


def test_yfinance_source_rejects_unknown_timeframe():
    src = YFinanceSource()
    with pytest.raises(ValueError):
        src._interval("7h")


def test_yfinance_source_maps_common_timeframes():
    src = YFinanceSource()
    assert src._interval("1h") == "60m"
    assert src._interval("1d") == "1d"
    assert src._interval("1wk") == "1wk"


def test_trading212_broker_resolves_via_instrument_cache():
    broker = Trading212Broker(api_key="dummy")
    broker._instruments = [
        {"ticker": "AAPL_US_EQ", "shortName": "AAPL"},
        {"ticker": "VWRLl_EQ", "shortName": "VWRL"},
    ]
    broker._ticker_map = {
        "AAPL": "AAPL_US_EQ",
        "AAPL_US_EQ": "AAPL_US_EQ",
        "VWRL": "VWRLl_EQ",
        "VWRLL_EQ": "VWRLl_EQ",
    }
    assert broker.resolve_ticker("AAPL") == "AAPL_US_EQ"
    assert broker.resolve_ticker("VWRL") == "VWRLl_EQ"


def test_trading212_broker_falls_back_to_raw_symbol():
    broker = Trading212Broker(api_key="dummy")
    broker._instruments = []
    broker._ticker_map = {}
    # Nothing cached -> return input unchanged so caller can use native tickers.
    assert broker.resolve_ticker("CUSTOM_TICKER") == "CUSTOM_TICKER"
