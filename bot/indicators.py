"""Technical indicators implemented with pandas/numpy only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    # Textbook RSI handles the one-sided edge cases as:
    #   avg_loss == 0 -> RSI = 100 (pure uptrend, never oversold)
    #   avg_gain == 0 -> RSI = 0   (pure downtrend, never overbought)
    # Returning NaN in those cases (as a naive divide would) would make the
    # indicator useless on strongly trending instruments.
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss > 0)), 0.0)
    return out


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, period)
    std = series.rolling(period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower})


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range – used for volatility-aware position sizing."""
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - close_prev).abs(),
            (low - close_prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """Supertrend indicator. Returns line and direction (+1 uptrend, -1 downtrend)."""
    atr_ = atr(df, period)
    hl2 = ((df["high"] + df["low"]) / 2).values
    atr_a = atr_.values
    upper = hl2 + multiplier * atr_a
    lower = hl2 - multiplier * atr_a
    close = df["close"].values

    n = len(df)
    fu = np.full(n, np.nan)
    fl = np.full(n, np.nan)
    direction = np.ones(n, dtype=int)
    line = np.full(n, np.nan)
    initialised = False

    for i in range(n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue
        if not initialised:
            fu[i] = upper[i]
            fl[i] = lower[i]
            direction[i] = 1
            line[i] = lower[i]
            initialised = True
            continue

        fu[i] = (
            upper[i]
            if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1])
            else fu[i - 1]
        )
        fl[i] = (
            lower[i]
            if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1])
            else fl[i - 1]
        )
        if direction[i - 1] == 1:
            if close[i] < fl[i]:
                direction[i] = -1
                line[i] = fu[i]
            else:
                direction[i] = 1
                line[i] = fl[i]
        else:
            if close[i] > fu[i]:
                direction[i] = 1
                line[i] = fl[i]
            else:
                direction[i] = -1
                line[i] = fu[i]

    return pd.DataFrame({"line": line, "direction": direction}, index=df.index)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index – strength of the trend regardless of direction.

    ADX > 25 generally indicates a strong trend; below 20 indicates chop.
    """
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [
            (high - low),
            (high - close_prev).abs(),
            (low - close_prev).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_ = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean() / atr_.replace(0.0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean() / atr_.replace(0.0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth: int = 3
) -> pd.DataFrame:
    """Stochastic oscillator (%K / %D) in [0, 100].

    Measures where the current close sits within the recent high/low range.
    %K > 80 = near top of range (overbought); %K < 20 = near bottom (oversold).
    %D is an SMA of %K used as a signal line.
    """
    low_n = df["low"].rolling(k_period, min_periods=k_period).min()
    high_n = df["high"].rolling(k_period, min_periods=k_period).max()
    span = (high_n - low_n).replace(0.0, np.nan)
    raw_k = 100 * (df["close"] - low_n) / span
    k = raw_k.rolling(smooth, min_periods=smooth).mean()
    d = k.rolling(d_period, min_periods=d_period).mean()
    return pd.DataFrame({"k": k, "d": d})


def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Donchian channel: highest high and lowest low over ``period`` bars.

    Breakouts above the upper band are a classic trend-following entry; the
    lower band is the mirrored exit / short trigger.
    """
    upper = df["high"].rolling(period, min_periods=period).max()
    lower = df["low"].rolling(period, min_periods=period).min()
    mid = (upper + lower) / 2
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": mid})


def keltner(
    df: pd.DataFrame, period: int = 20, multiplier: float = 2.0, atr_period: int = 10
) -> pd.DataFrame:
    """Keltner channel: EMA of close plus/minus ``multiplier`` * ATR.

    Tighter than Bollinger bands in trending markets because volatility is
    measured from true range rather than stdev. A close outside the channel
    is a strong-trend signal rather than a mean-reversion cue.
    """
    mid = ema(df["close"], period)
    atr_ = atr(df, atr_period)
    upper = mid + multiplier * atr_
    lower = mid - multiplier * atr_
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": mid})


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index: how far the typical price has deviated from
    its rolling mean, scaled by mean deviation. Unbounded but most readings
    sit in [-200, +200]. Crosses above +100 often mark strong uptrends;
    crosses below -100 mark strong downtrends.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    ma = typical.rolling(period, min_periods=period).mean()
    mad = typical.rolling(period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    denom = (0.015 * mad).replace(0.0, np.nan)
    return (typical - ma) / denom


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume: cumulative volume, signed by the direction of the
    close. Rising OBV confirms an uptrend (volume flowing in); divergence
    against price is a classic warning that a trend is weakening.
    """
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()

