"""Pick the right exchange/broker adapter based on config."""
from __future__ import annotations

from typing import Any

from .config import Config


# Platforms routed through the stocks (yfinance + Trading 212) stack.
_STOCK_PLATFORMS = {"trading212", "trading_212", "t212"}


def is_stock_platform(cfg: Config) -> bool:
    return cfg.exchange.name.lower() in _STOCK_PLATFORMS


def build_exchange(cfg: Config) -> Any:
    """Return a ccxt-backed Exchange or a StocksExchange, matching cfg."""
    if is_stock_platform(cfg):
        from .stocks import StocksExchange
        return StocksExchange(cfg)
    from .exchange import Exchange
    return Exchange(cfg)
