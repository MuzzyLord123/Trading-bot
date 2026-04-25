"""Shared helpers used by more than one strategy module.

Anything that lives here exists because two or more strategies need it
and we don't want to duplicate the implementation. Keep this module
free of imports from sibling strategy modules so it can't introduce
cycles.
"""
from __future__ import annotations

import pandas as pd


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an OHLCV frame to a higher timeframe.

    Used by FilteredStrategy (htf trend gate) and MultiTimeframeStrategy
    (full HTF confirmation pass) - they used to carry private copies
    that drifted in formatting; consolidating here means a fix to the
    resampling rules can never regress one and not the other.
    """
    idx = df.set_index("timestamp")
    out = pd.DataFrame({
        "open": idx["open"].resample(rule).first(),
        "high": idx["high"].resample(rule).max(),
        "low": idx["low"].resample(rule).min(),
        "close": idx["close"].resample(rule).last(),
        "volume": idx["volume"].resample(rule).sum(),
    }).dropna()
    return out.reset_index()
