"""Trading 212 Invest integration for equities/ETFs.

Two responsibilities in this module:

  * :class:`YFinanceSource` – free market data (OHLCV) from Yahoo Finance.
    Used for both backtesting and live signal generation.

  * :class:`Trading212Broker` – REST client for placing orders and reading
    balance/positions on Trading 212 Invest.

These are combined in :class:`StocksExchange` which implements the same
interface as the ccxt-backed :class:`bot.exchange.Exchange`, so the engine
and backtester don't need to care which asset class they're on.

Notes on data vs execution split:
  * Trading 212 does NOT expose historical candles via their API. Yahoo
    Finance covers that gap for free, without auth.
  * Prices used for signal logic come from yfinance; actual fills happen
    at T212 prices. There will be a small slip between the two. Acceptable
    for daily/hourly bots, not for sub-minute strategies.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger("bot.stocks")


_YF_INTERVAL = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "60m": "60m",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}


class YFinanceSource:
    """Minimal Yahoo Finance OHLCV fetcher.

    Uses the ``yfinance`` library; we import lazily on first fetch so the
    rest of the code (including unit tests) runs without it installed.
    """

    def _yf(self):
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError(
                "yfinance is required for stock trading. "
                "Install with: pip install yfinance"
            ) from exc
        return yf

    def _interval(self, timeframe: str) -> str:
        try:
            return _YF_INTERVAL[timeframe]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}' for stocks. "
                f"Use one of: {sorted(_YF_INTERVAL)}"
            ) from exc

    def _period_for(self, timeframe: str, limit: int) -> str:
        # Yahoo caps intraday history: 60m → 730d, 5m → 60d, 1m → 7d.
        # Pick the smallest period that likely fits ``limit`` bars.
        days_per_bar = {
            "1m": 1 / 390, "5m": 5 / 390, "15m": 15 / 390, "30m": 30 / 390,
            "60m": 1 / 6.5, "1h": 1 / 6.5,
            "1d": 1, "1wk": 7, "1mo": 30,
        }.get(timeframe, 1)
        need_days = max(5, int(limit * days_per_bar * 1.5))
        if timeframe in {"1m"}:
            return "7d"
        if timeframe in {"5m", "15m", "30m"}:
            return f"{min(need_days, 60)}d"
        if timeframe in {"60m", "1h"}:
            return f"{min(need_days, 730)}d"
        if need_days <= 365:
            return f"{need_days}d"
        years = max(2, need_days // 365 + 1)
        return f"{min(years, 10)}y"

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        since: int | None = None,
    ) -> pd.DataFrame:
        yf = self._yf()
        interval = self._interval(timeframe)
        period = self._period_for(timeframe, limit)
        df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.reset_index().rename(
            columns={
                "Date": "timestamp",
                "Datetime": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        return df.tail(limit).reset_index(drop=True)

    def fetch_ohlcv_batch(
        self,
        symbols: list[str],
        timeframe: str,
        limit: int = 500,
    ) -> dict[str, pd.DataFrame]:
        """Batch download OHLCV for many tickers at once.

        Much faster than calling :meth:`fetch_ohlcv` per symbol when scanning
        a large universe (e.g. the S&P 500). yfinance parallelises the
        download internally.
        """
        if not symbols:
            return {}
        yf = self._yf()
        interval = self._interval(timeframe)
        period = self._period_for(timeframe, limit)
        raw = yf.download(
            tickers=" ".join(symbols),
            period=period,
            interval=interval,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        result: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                sub = raw[sym] if len(symbols) > 1 and sym in raw.columns.get_level_values(0) else raw
            except (KeyError, AttributeError):
                continue
            if sub is None or sub.empty:
                continue
            df = sub.reset_index().rename(
                columns={
                    "Date": "timestamp",
                    "Datetime": "timestamp",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            if "timestamp" not in df.columns:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            cols = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in cols):
                continue
            df = df[cols].dropna()
            if df.empty:
                continue
            result[sym] = df.tail(limit).reset_index(drop=True)
        return result

    def fetch_ohlcv_range(
        self, symbol: str, timeframe: str, since_ms: int, until_ms: int
    ) -> pd.DataFrame:
        yf = self._yf()
        interval = self._interval(timeframe)
        start = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc)
        end = datetime.fromtimestamp(until_ms / 1000, tz=timezone.utc)
        df = yf.Ticker(symbol).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
        )
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.reset_index().rename(
            columns={
                "Date": "timestamp",
                "Datetime": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Trading 212 REST client.
# ---------------------------------------------------------------------------
T212_LIVE_BASE = "https://live.trading212.com/api/v0"
T212_DEMO_BASE = "https://demo.trading212.com/api/v0"


@dataclass
class T212Position:
    ticker: str
    quantity: float
    average_price: float
    current_price: float


class Trading212Broker:
    """REST client for Trading 212 Invest.

    Docs: https://t212public-api-docs.redoc.ly/

    The API key is per-environment (live vs demo) – request one from the
    Trading 212 app under Settings → API. It takes a few days for T212 to
    approve API access.
    """

    def __init__(self, api_key: str, sandbox: bool = False, request_interval_s: float = 2.0) -> None:
        self.api_key = api_key
        self.base = T212_DEMO_BASE if sandbox else T212_LIVE_BASE
        self._min_interval = request_interval_s
        self._last_request = 0.0
        self._instruments: list[dict[str, Any]] | None = None
        self._ticker_map: dict[str, str] = {}

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=16))
    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        # Soft rate-limit: space requests out to avoid 429s.
        delta = time.time() - self._last_request
        if delta < self._min_interval:
            time.sleep(self._min_interval - delta)
        url = f"{self.base}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                self._last_request = time.time()
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            log.error("T212 %s %s -> %s %s", method, path, exc.code, body_text[:400])
            raise

    def load_instruments(self) -> list[dict[str, Any]]:
        """Fetch and cache the full instrument catalogue."""
        if self._instruments is None:
            data = self._request("GET", "/equity/metadata/instruments")
            self._instruments = data or []
            # Build a short-name -> T212 ticker map.
            for inst in self._instruments:
                ticker = inst.get("ticker", "")
                short = inst.get("shortName") or ticker.split("_")[0]
                self._ticker_map.setdefault(short.upper(), ticker)
                self._ticker_map.setdefault(ticker.upper(), ticker)
        return self._instruments or []

    def resolve_ticker(self, symbol: str) -> str:
        """Map a yfinance-style symbol (AAPL, VWRL.L) to a T212 ticker."""
        self.load_instruments()
        key = symbol.upper().replace(".L", "l").replace(".", "")
        # Try full symbol first (yfinance style), then stripped.
        for candidate in (symbol.upper(), symbol.upper().replace(".L", "l"), key):
            if candidate in self._ticker_map:
                return self._ticker_map[candidate]
        # Fallback: assume caller passed the T212 ticker directly.
        return symbol

    def cash(self) -> float:
        data = self._request("GET", "/equity/account/cash") or {}
        return float(data.get("free", 0.0))

    def positions(self) -> list[T212Position]:
        data = self._request("GET", "/equity/portfolio") or []
        return [
            T212Position(
                ticker=p["ticker"],
                quantity=float(p.get("quantity", 0.0)),
                average_price=float(p.get("averagePrice", 0.0)),
                current_price=float(p.get("currentPrice", 0.0)),
            )
            for p in data
        ]

    def place_market_order(self, ticker: str, quantity: float) -> dict[str, Any]:
        """Positive ``quantity`` buys, negative sells."""
        body = {"ticker": ticker, "quantity": float(quantity)}
        return self._request("POST", "/equity/orders/market", body) or {}


# ---------------------------------------------------------------------------
# Exchange-compatible facade.
# ---------------------------------------------------------------------------
@dataclass
class Order:
    id: str
    symbol: str
    side: str
    price: float
    amount: float
    status: str
    cost: float
    timestamp: int


class StocksExchange:
    """Implements the subset of :class:`bot.exchange.Exchange` used by the
    engine and backtester, backed by yfinance data + T212 execution.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.data = YFinanceSource()
        live = cfg.trading.mode == "live"
        api_key = cfg.secrets.get("trading212_api_key", "") if live else ""
        self.broker: Trading212Broker | None = (
            Trading212Broker(api_key, sandbox=cfg.exchange.sandbox) if live and api_key else None
        )

    def load_markets(self) -> dict[str, Any]:
        if self.broker is not None:
            # Populates T212 instrument cache so order placement is fast.
            self.broker.load_instruments()
        return {s: {} for s in self.cfg.trading.symbols}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500, since: int | None = None) -> pd.DataFrame:
        return self.data.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)

    def fetch_ohlcv_batch(self, symbols: list[str], timeframe: str, limit: int = 500) -> dict[str, pd.DataFrame]:
        return self.data.fetch_ohlcv_batch(symbols, timeframe, limit=limit)

    def fetch_ohlcv_range(self, symbol: str, timeframe: str, since_ms: int, until_ms: int) -> pd.DataFrame:
        return self.data.fetch_ohlcv_range(symbol, timeframe, since_ms, until_ms)

    def fetch_balance(self) -> dict[str, float]:
        if self.broker is None:
            return {}
        return {self.cfg.trading.quote_currency: self.broker.cash()}

    def create_market_order(self, symbol: str, side: str, amount: float) -> Order:
        if self.cfg.trading.mode != "live":
            raise RuntimeError("create_market_order called outside live mode")
        if self.broker is None:
            raise RuntimeError("Trading 212 broker not configured (missing API key)")
        ticker = self.broker.resolve_ticker(symbol)
        qty = amount if side == "buy" else -abs(amount)
        resp = self.broker.place_market_order(ticker, qty)
        return Order(
            id=str(resp.get("id", "")),
            symbol=symbol,
            side=side,
            price=float(resp.get("fillPrice") or 0.0),
            amount=abs(float(resp.get("filledQuantity") or amount)),
            status=str(resp.get("status", "unknown")),
            cost=float(resp.get("fillPrice") or 0.0) * abs(float(resp.get("filledQuantity") or amount)),
            timestamp=int(time.time() * 1000),
        )

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        # Trading 212 Invest supports fractional shares down to 0.0001. Keep
        # the default permissive — reject tiny amounts via min-notional instead.
        return round(amount, 4)

    def price_to_precision(self, symbol: str, price: float) -> float:
        return round(price, 4)

    def market_min_amount(self, symbol: str) -> float:
        return 0.0


def is_stock_market_open(now: datetime | None = None) -> bool:
    """Rough window that captures both UK (LSE) and US (NYSE/NASDAQ) hours.

    UK: 08:00–16:30 London (07:00–15:30 UTC in summer, 08:00–16:30 UTC in winter).
    US: 09:30–16:00 ET (13:30–20:00 UTC in summer, 14:30–21:00 UTC in winter).

    Returns True on weekdays between 07:00 and 21:00 UTC. Good enough to
    decide "can we trade at all right now?" without a full trading calendar.
    """
    now = now or datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    return 7 <= now.hour < 21
