"""Trading 212 broker tests using a mocked urlopen so we never touch
the real API. Covers every code path the live engine exercises:

  * Auth header is exactly the API key (no Bearer prefix).
  * Live URL vs demo URL based on sandbox flag.
  * Buy/sell sign convention on market orders.
  * 4xx is NOT retried; 429 and 5xx ARE.
  * T212APIError preserves status + body for the caller.
  * Ticker resolution covers US tickers, LSE .L tickers, short names,
    case-insensitive lookup, and graceful fallback for unknown symbols.
"""
from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from bot.stocks import T212APIError, Trading212Broker, _is_transient_http


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _fake_response(payload):
    """Build a context-manager mock that mimics urlopen()'s response."""
    raw = json.dumps(payload).encode() if payload is not None else b""
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = raw
    cm.__exit__.return_value = False
    return cm


def _http_error(status: int, body: str = "") -> urllib.error.HTTPError:
    err = urllib.error.HTTPError(
        url="https://example.com",
        code=status,
        msg="error",
        hdrs={},
        fp=io.BytesIO(body.encode()),
    )
    return err


def _make_broker(sandbox: bool = False) -> Trading212Broker:
    # request_interval_s=0 so tests don't sleep between calls.
    return Trading212Broker(api_key="test-key-1234567890", sandbox=sandbox, request_interval_s=0.0)


# ---------------------------------------------------------------------------
# Auth + URL.
# ---------------------------------------------------------------------------
def test_auth_header_is_just_the_api_key():
    broker = _make_broker()
    with patch("bot.stocks.urllib.request.urlopen", return_value=_fake_response({})) as mock_open:
        broker.cash()
    req = mock_open.call_args[0][0]
    # T212 expects the bare key, NOT "Bearer ...".
    assert req.headers["Authorization"] == "test-key-1234567890"
    assert "Bearer" not in req.headers["Authorization"]


def test_content_type_and_accept_headers_are_json():
    broker = _make_broker()
    with patch("bot.stocks.urllib.request.urlopen", return_value=_fake_response({})) as mock_open:
        broker.cash()
    req = mock_open.call_args[0][0]
    assert req.headers["Content-type"] == "application/json"
    assert req.headers["Accept"] == "application/json"


def test_live_url_used_when_sandbox_false():
    broker = _make_broker(sandbox=False)
    with patch("bot.stocks.urllib.request.urlopen", return_value=_fake_response({})) as mock_open:
        broker.cash()
    assert mock_open.call_args[0][0].full_url.startswith("https://live.trading212.com/api/v0/")


def test_demo_url_used_when_sandbox_true():
    broker = _make_broker(sandbox=True)
    with patch("bot.stocks.urllib.request.urlopen", return_value=_fake_response({})) as mock_open:
        broker.cash()
    assert mock_open.call_args[0][0].full_url.startswith("https://demo.trading212.com/api/v0/")


# ---------------------------------------------------------------------------
# Cash + positions parsing.
# ---------------------------------------------------------------------------
def test_cash_returns_free_field_as_float():
    broker = _make_broker()
    payload = {"free": 1234.56, "total": 9999.0, "blocked": 0.0}
    with patch("bot.stocks.urllib.request.urlopen", return_value=_fake_response(payload)):
        assert broker.cash() == pytest.approx(1234.56)


def test_cash_handles_missing_free_field():
    broker = _make_broker()
    with patch("bot.stocks.urllib.request.urlopen", return_value=_fake_response({})):
        assert broker.cash() == 0.0


def test_positions_parses_full_payload():
    broker = _make_broker()
    payload = [
        {"ticker": "AAPL_US_EQ", "quantity": 10.0, "averagePrice": 150.0, "currentPrice": 175.0},
        {"ticker": "VWRLl_EQ", "quantity": 5.5, "averagePrice": 95.0, "currentPrice": 100.0},
    ]
    with patch("bot.stocks.urllib.request.urlopen", return_value=_fake_response(payload)):
        positions = broker.positions()
    assert len(positions) == 2
    assert positions[0].ticker == "AAPL_US_EQ"
    assert positions[0].quantity == 10.0
    assert positions[0].current_price == 175.0
    assert positions[1].quantity == 5.5


def test_positions_empty_payload_returns_empty_list():
    broker = _make_broker()
    with patch("bot.stocks.urllib.request.urlopen", return_value=_fake_response([])):
        assert broker.positions() == []


# ---------------------------------------------------------------------------
# Order placement.
# ---------------------------------------------------------------------------
def test_buy_sends_positive_quantity():
    broker = _make_broker()
    payload = {"id": "order-1", "status": "FILLED"}
    with patch("bot.stocks.urllib.request.urlopen", return_value=_fake_response(payload)) as mock_open:
        broker.place_market_order("AAPL_US_EQ", 5.0)
    req = mock_open.call_args[0][0]
    body = json.loads(req.data)
    assert body == {"ticker": "AAPL_US_EQ", "quantity": 5.0}
    assert req.method == "POST"
    assert req.full_url.endswith("/equity/orders/market")


def test_sell_sends_negative_quantity():
    broker = _make_broker()
    with patch("bot.stocks.urllib.request.urlopen", return_value=_fake_response({"id": "x"})) as mock_open:
        broker.place_market_order("AAPL_US_EQ", -3.0)
    body = json.loads(mock_open.call_args[0][0].data)
    assert body["quantity"] == -3.0


# ---------------------------------------------------------------------------
# Error handling.
# ---------------------------------------------------------------------------
def test_4xx_raises_T212APIError_with_status_and_body():
    broker = _make_broker()
    err = _http_error(400, '{"code":"BadRequest","message":"invalid ticker"}')
    with patch("bot.stocks.urllib.request.urlopen", side_effect=err):
        with pytest.raises(T212APIError) as ei:
            broker.place_market_order("BOGUS", 1.0)
    assert ei.value.status == 400
    assert "BadRequest" in ei.value.body
    assert ei.value.is_client_error
    assert not ei.value.is_rate_limited


def test_401_marks_auth_error():
    broker = _make_broker()
    with patch("bot.stocks.urllib.request.urlopen", side_effect=_http_error(401, "")):
        with pytest.raises(T212APIError) as ei:
            broker.cash()
    assert ei.value.is_auth_error


@pytest.fixture
def _no_sleep(monkeypatch):
    """Tenacity uses time.sleep between retries with wait_exponential.
    Patch it out so retry tests finish in millis instead of ~14 seconds."""
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _s: None)


def test_429_classified_rate_limited(_no_sleep):
    broker = _make_broker()
    err = _http_error(429, "rate limited")
    with patch("bot.stocks.urllib.request.urlopen", side_effect=err):
        with pytest.raises(T212APIError) as ei:
            broker._request("GET", "/equity/account/cash")
    assert ei.value.is_rate_limited


def test_4xx_is_NOT_retried():
    """A definitive 400 should be raised once, not 4 times. Otherwise we
    burn rate budget and delay the real error."""
    broker = _make_broker()
    err = _http_error(400, "bad ticker")
    with patch("bot.stocks.urllib.request.urlopen", side_effect=err) as mock_open:
        with pytest.raises(T212APIError):
            broker._request("GET", "/equity/account/cash")
    assert mock_open.call_count == 1


def test_429_IS_retried_until_attempts_exhausted(_no_sleep):
    broker = _make_broker()
    err = _http_error(429, "rate limit")
    with patch("bot.stocks.urllib.request.urlopen", side_effect=err) as mock_open:
        with pytest.raises(T212APIError):
            broker._request("GET", "/equity/account/cash")
    # tenacity stop_after_attempt(4) -> 4 calls.
    assert mock_open.call_count == 4


def test_500_IS_retried_until_attempts_exhausted(_no_sleep):
    broker = _make_broker()
    err = _http_error(503, "service unavailable")
    with patch("bot.stocks.urllib.request.urlopen", side_effect=err) as mock_open:
        with pytest.raises(T212APIError):
            broker._request("GET", "/equity/account/cash")
    assert mock_open.call_count == 4


def test_transient_predicate_recognises_t212_api_error():
    err = T212APIError("GET", "/x", 503, "")
    assert _is_transient_http(err) is True
    err2 = T212APIError("GET", "/x", 400, "")
    assert _is_transient_http(err2) is False


# ---------------------------------------------------------------------------
# Ticker resolution.
# ---------------------------------------------------------------------------
def _seed_instruments(broker, instruments):
    """Bypass the network and inject a fake instrument list."""
    broker._instruments = list(instruments)
    for inst in instruments:
        broker._index_instrument(inst)


def test_resolve_us_ticker():
    broker = _make_broker()
    _seed_instruments(broker, [{"ticker": "AAPL_US_EQ", "shortName": "AAPL"}])
    assert broker.resolve_ticker("AAPL") == "AAPL_US_EQ"
    assert broker.resolve_ticker("aapl") == "AAPL_US_EQ"


def test_resolve_lse_ticker_with_dot_l_suffix():
    """The pre-fix bug: VWRL.L resolved to itself because the lookup
    candidate had a lowercase 'l' that didn't match the upper-cased map."""
    broker = _make_broker()
    _seed_instruments(broker, [{"ticker": "VWRLl_EQ", "shortName": "VWRL"}])
    assert broker.resolve_ticker("VWRL.L") == "VWRLl_EQ"
    assert broker.resolve_ticker("vwrl.l") == "VWRLl_EQ"
    assert broker.resolve_ticker("VWRL") == "VWRLl_EQ"  # short-name


def test_resolve_already_t212_ticker_passes_through():
    broker = _make_broker()
    _seed_instruments(broker, [{"ticker": "AAPL_US_EQ", "shortName": "AAPL"}])
    assert broker.resolve_ticker("AAPL_US_EQ") == "AAPL_US_EQ"


def test_resolve_unknown_symbol_returns_input():
    broker = _make_broker()
    _seed_instruments(broker, [{"ticker": "AAPL_US_EQ", "shortName": "AAPL"}])
    # Unknown symbols pass through unchanged so T212 can return the
    # canonical "ticker not found" error rather than the bot lying.
    assert broker.resolve_ticker("XYZ123") == "XYZ123"
