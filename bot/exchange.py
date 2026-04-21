from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import ccxt
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Config

log = logging.getLogger("bot.exchange")


@dataclass
class Ticker:
    symbol: str
    last: float
    bid: float
    ask: float
    timestamp: int


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


class Exchange:
    """Thin ccxt wrapper with retries and helpers."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._client = self._build_client()

    def _build_client(self) -> ccxt.Exchange:
        name = self.cfg.exchange.name.lower()
        if not hasattr(ccxt, name):
            raise ValueError(f"Exchange '{name}' is not supported by ccxt")
        params: dict[str, Any] = {
            "enableRateLimit": self.cfg.exchange.enable_rate_limit,
        }
        if self.cfg.trading.mode == "live":
            params["apiKey"] = self.cfg.secrets["api_key"]
            params["secret"] = self.cfg.secrets["api_secret"]
            if self.cfg.exchange.requires_password:
                params["password"] = self.cfg.secrets["api_password"]
        client: ccxt.Exchange = getattr(ccxt, name)(params)
        if self.cfg.exchange.sandbox and hasattr(client, "set_sandbox_mode"):
            client.set_sandbox_mode(True)
        return client

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=16))
    def load_markets(self) -> dict[str, Any]:
        return self._client.load_markets()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=16))
    def fetch_ticker(self, symbol: str) -> Ticker:
        t = self._client.fetch_ticker(symbol)
        return Ticker(
            symbol=symbol,
            last=float(t.get("last") or t.get("close") or 0.0),
            bid=float(t.get("bid") or 0.0),
            ask=float(t.get("ask") or 0.0),
            timestamp=int(t.get("timestamp") or time.time() * 1000),
        )

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=16))
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        since: int | None = None,
    ) -> pd.DataFrame:
        raw = self._client.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def fetch_ohlcv_range(
        self, symbol: str, timeframe: str, since_ms: int, until_ms: int
    ) -> pd.DataFrame:
        """Paginate history between two millisecond timestamps."""
        tf_ms = self._client.parse_timeframe(timeframe) * 1000
        out: list[list[float]] = []
        cursor = since_ms
        while cursor < until_ms:
            batch = self._client.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
            if not batch:
                break
            out.extend(batch)
            last_ts = batch[-1][0]
            next_cursor = last_ts + tf_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if self.cfg.exchange.enable_rate_limit:
                time.sleep(self._client.rateLimit / 1000)
        df = pd.DataFrame(out, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df[df["timestamp"].astype("int64") // 10**6 <= until_ms]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def fetch_balance(self) -> dict[str, float]:
        bal = self._client.fetch_balance()
        free = bal.get("free", {}) or {}
        return {k: float(v) for k, v in free.items() if v}

    def create_market_order(self, symbol: str, side: str, amount: float) -> Order:
        if self.cfg.trading.mode != "live":
            raise RuntimeError("create_market_order called outside live mode")
        o = self._client.create_order(symbol=symbol, type="market", side=side, amount=amount)
        return Order(
            id=str(o.get("id", "")),
            symbol=symbol,
            side=side,
            price=float(o.get("average") or o.get("price") or 0.0),
            amount=float(o.get("filled") or amount),
            status=str(o.get("status", "unknown")),
            cost=float(o.get("cost") or 0.0),
            timestamp=int(o.get("timestamp") or time.time() * 1000),
        )

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        try:
            return float(self._client.amount_to_precision(symbol, amount))
        except Exception:
            return round(amount, 8)

    def price_to_precision(self, symbol: str, price: float) -> float:
        try:
            return float(self._client.price_to_precision(symbol, price))
        except Exception:
            return round(price, 8)

    def market_min_amount(self, symbol: str) -> float:
        market = self._client.market(symbol)
        return float((market.get("limits", {}).get("amount", {}) or {}).get("min") or 0.0)
