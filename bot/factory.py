"""Adapter factory.

This bot is Trading 212 only, so the factory is a thin constructor.
Kept as a separate module to avoid import cycles between config and stocks.
"""
from __future__ import annotations

from typing import Any

from .config import Config


def build_exchange(cfg: Config) -> Any:
    """Return the Trading 212 / yfinance adapter configured by ``cfg``."""
    from .stocks import StocksExchange
    return StocksExchange(cfg)


def is_stock_platform(cfg: Config) -> bool:
    """Back-compat shim – this bot only trades stocks, always True."""
    return True
