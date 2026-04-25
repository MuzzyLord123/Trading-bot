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



def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted moving average. Newer bars carry linearly higher weight,
    so it tracks turns faster than SMA without the EMA's exponential tail.
    """
    weights = np.arange(1, period + 1, dtype=float)
    weight_sum = weights.sum()
    return series.rolling(period, min_periods=period).apply(
        lambda x: np.dot(x, weights) / weight_sum, raw=True,
    )


def hma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average. Hull's recipe combines two WMAs to cut lag
    almost in half versus a comparable EMA while staying smooth. Popular
    among swing traders for its responsiveness on hourly bars.

    Formula: HMA(n) = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    """
    half = max(1, int(period / 2))
    sqrt_n = max(1, int(np.sqrt(period)))
    raw = 2 * wma(series, half) - wma(series, period)
    return wma(raw, sqrt_n)


def vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling Volume-Weighted Average Price.

    Strict VWAP resets each session, but our timeframe is multi-day and
    yfinance bars don't expose a session boundary, so we use a rolling
    ``period``-bar window. Practical equivalent: "average price weighted
    by volume over the last N bars". Above VWAP = buyers in control.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]
    return (
        pv.rolling(period, min_periods=period).sum()
        / df["volume"].rolling(period, min_periods=period).sum().replace(0.0, np.nan)
    )


def parabolic_sar(
    df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2
) -> pd.DataFrame:
    """Wilder's Parabolic SAR (Stop And Reverse).

    Stateful trend-following indicator that flips when price crosses it.
    Returns a DataFrame with columns:
      * ``sar``       - the SAR value at this bar
      * ``direction`` - +1 if uptrend (SAR below price), -1 if downtrend.

    Common use: trail stops with the SAR while in trend; flip position
    when direction changes.
    """
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    sar = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)
    if n < 2:
        return pd.DataFrame({"sar": sar, "direction": direction}, index=df.index)

    # Seed from the first two bars: assume up if close[1] >= close[0].
    up = df["close"].iloc[1] >= df["close"].iloc[0]
    direction[0] = direction[1] = 1 if up else -1
    ep = high[1] if up else low[1]   # extreme point
    af = step                        # acceleration factor
    sar[1] = low[0] if up else high[0]

    for i in range(2, n):
        prev_sar = sar[i - 1]
        prev_dir = direction[i - 1]
        new_sar = prev_sar + af * (ep - prev_sar)
        if prev_dir == 1:
            # Don't let SAR move above the prior two lows.
            new_sar = min(new_sar, low[i - 1], low[i - 2])
            if low[i] < new_sar:
                # Flip to downtrend.
                direction[i] = -1
                sar[i] = ep
                ep = low[i]
                af = step
            else:
                direction[i] = 1
                sar[i] = new_sar
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_step)
        else:
            new_sar = max(new_sar, high[i - 1], high[i - 2])
            if high[i] > new_sar:
                direction[i] = 1
                sar[i] = ep
                ep = high[i]
                af = step
            else:
                direction[i] = -1
                sar[i] = new_sar
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_step)

    return pd.DataFrame({"sar": sar, "direction": direction}, index=df.index)


def ichimoku(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """Ichimoku Kinko Hyo cloud system. Returns a frame with:

      * ``tenkan``    - conversion line (short-term trend bias)
      * ``kijun``     - base line (medium-term trend bias)
      * ``senkou_a``  - cloud upper boundary as visible AT this bar
      * ``senkou_b``  - cloud lower boundary as visible AT this bar
      * ``chikou``    - lagging close (helper - operator compares to
                         close N bars ago when reasoning at bar t)

    Note on lookahead: the cloud at bar t is determined by data
    ``displacement`` bars in the past (it's drawn 26 bars FORWARD of
    when its inputs were known). We compute the inputs and ``shift``
    them so cloud values at index t can be read straight off the row
    without any future data leaking in.
    """
    high = df["high"]
    low = df["low"]

    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a_raw = (tenkan_sen + kijun_sen) / 2
    senkou_b_raw = (high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2

    return pd.DataFrame({
        "tenkan": tenkan_sen,
        "kijun": kijun_sen,
        # Drawn ahead of price by ``displacement`` bars. AT bar t the
        # cloud value is what was computed ``displacement`` bars ago.
        "senkou_a": senkou_a_raw.shift(displacement),
        "senkou_b": senkou_b_raw.shift(displacement),
        # Chikou is just the close. The strategy compares it to close
        # ``displacement`` bars in the past for the lagging-confirmation
        # check; storing the close keeps that calculation explicit.
        "chikou": df["close"],
    }, index=df.index)


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R. Bounded in [-100, 0]. Inverted Stochastic-like
    oscillator: -20 = near top of range (overbought), -80 = near bottom
    (oversold). Crosses up through -50 are a common momentum trigger.
    """
    high_n = df["high"].rolling(period, min_periods=period).max()
    low_n = df["low"].rolling(period, min_periods=period).min()
    span = (high_n - low_n).replace(0.0, np.nan)
    return -100 * (high_n - df["close"]) / span


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index. RSI computed on volume-weighted typical price
    rather than close. Bounded in [0, 100]. Same overbought/oversold
    semantics as RSI but the volume weight makes it harder to fool with
    low-conviction price moves.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    raw_money_flow = typical * df["volume"]
    direction = np.sign(typical.diff()).fillna(0.0)
    positive = raw_money_flow.where(direction > 0, 0.0)
    negative = raw_money_flow.where(direction < 0, 0.0)
    pos_sum = positive.rolling(period, min_periods=period).sum()
    neg_sum = negative.rolling(period, min_periods=period).sum()
    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    out = 100 - (100 / (1 + ratio))
    out = out.where(~((neg_sum == 0) & (pos_sum > 0)), 100.0)
    out = out.where(~((pos_sum == 0) & (neg_sum > 0)), 0.0)
    return out


def aroon(df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
    """Aroon Up/Down + Oscillator. Each in [0, 100].

    Aroon Up  = 100 * (period - bars_since_period_high) / period
    Aroon Down = 100 * (period - bars_since_period_low) / period
    Oscillator = Aroon Up - Aroon Down  (in [-100, +100])

    A high Aroon Up with low Aroon Down marks a strong uptrend.
    """
    def _bars_since_max(x):
        # x is a rolling window of length ``period``. np.argmax returns the
        # *forward* index of the max; bars-since-max measured from the
        # latest bar is (period - 1) - that index. Reversing the window
        # before argmax (the previous implementation) was wrong - it gave
        # bars-since-OLDEST extreme, inverting Aroon's sign in clean
        # trends.
        return (period - 1) - int(np.argmax(x))

    def _bars_since_min(x):
        return (period - 1) - int(np.argmin(x))

    high_idx = df["high"].rolling(period, min_periods=period).apply(_bars_since_max, raw=True)
    low_idx = df["low"].rolling(period, min_periods=period).apply(_bars_since_min, raw=True)
    up = 100 * (period - high_idx) / period
    down = 100 * (period - low_idx) / period
    return pd.DataFrame({"up": up, "down": down, "oscillator": up - down}, index=df.index)


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow. Bounded approximately in [-1, 1].

    Sums (close-position-within-bar * volume) over a window and
    normalises by total volume. Positive readings = buyers absorbed
    selling; negative = sellers in control.
    """
    span = (df["high"] - df["low"]).replace(0.0, np.nan)
    multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / span
    money_flow_volume = multiplier.fillna(0.0) * df["volume"]
    return (
        money_flow_volume.rolling(period, min_periods=period).sum()
        / df["volume"].rolling(period, min_periods=period).sum().replace(0.0, np.nan)
    )


def roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change in percent. Pure momentum; positive = price up
    over the lookback, negative = price down. Often used as a regime
    filter (only take longs when ROC > 0)."""
    return 100 * (series - series.shift(period)) / series.shift(period).replace(0.0, np.nan)
