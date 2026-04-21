"""Crypto market sentiment via the Fear & Greed index.

Source: https://alternative.me/crypto/fear-and-greed-index/
Free API, updated daily, no auth required.

Usage:
    from bot.sentiment import load_fear_greed
    fng = load_fear_greed(days=400)  # DataFrame[timestamp, value]

The loader caches the response on disk so repeated runs don't hit the API
every time. If the API is unreachable the cached value is used; if no cache
exists an empty DataFrame is returned and callers should treat the filter
as disabled (i.e. don't reject signals).
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path

import pandas as pd

log = logging.getLogger("bot.sentiment")

API_URL = "https://api.alternative.me/fng/"
CACHE_PATH = Path("cache/fear_greed.json")
CACHE_TTL_SECONDS = 3600  # refresh once per hour


def _fetch(limit: int) -> dict | None:
    try:
        url = f"{API_URL}?limit={int(limit)}&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "trading-bot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log.warning("Fear & Greed fetch failed: %s", exc)
        return None


def _parse(data: dict) -> pd.DataFrame:
    rows = data.get("data", []) if data else []
    if not rows:
        return pd.DataFrame(columns=["timestamp", "value"])
    df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(int(r["timestamp"]), unit="s", tz="UTC"),
                "value": int(r["value"]),
            }
            for r in rows
        ]
    ).sort_values("timestamp").reset_index(drop=True)
    return df


def load_fear_greed(days: int = 400, use_cache: bool = True) -> pd.DataFrame:
    """Return a DataFrame with columns [timestamp, value]. Empty on failure."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and CACHE_PATH.exists():
        age = time.time() - CACHE_PATH.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            try:
                return _parse(json.loads(CACHE_PATH.read_text()))
            except Exception:
                pass

    data = _fetch(days)
    if data is not None:
        try:
            CACHE_PATH.write_text(json.dumps(data))
        except Exception as exc:
            log.warning("Could not write F&G cache: %s", exc)
        return _parse(data)

    if CACHE_PATH.exists():
        try:
            return _parse(json.loads(CACHE_PATH.read_text()))
        except Exception:
            pass
    return pd.DataFrame(columns=["timestamp", "value"])


def value_at(df: pd.DataFrame, ts: pd.Timestamp) -> int | None:
    """Latest F&G value at or before ``ts``. Returns None if unavailable."""
    if df.empty:
        return None
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    mask = df["timestamp"] <= ts
    if not mask.any():
        return None
    return int(df.loc[mask, "value"].iloc[-1])
