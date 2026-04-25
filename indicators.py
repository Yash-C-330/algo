"""
Technical indicator calculations used by the regime detector and strategies.
All indicators operate on pandas DataFrames with OHLCV columns.
"""
import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index.
    Returns DataFrame with columns: 'adx', 'plus_di', 'minus_di'
    """
    high, low, close = df["high"], df["low"], df["close"]

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                        index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                         index=df.index)

    atr_val = atr(df, period)

    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() /
                     atr_val.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() /
                      atr_val.replace(0, np.nan))

    dx = 100 * ((plus_di - minus_di).abs() /
                (plus_di + minus_di).replace(0, np.nan))
    adx_val = dx.ewm(alpha=1.0 / period, min_periods=period).mean()

    return pd.DataFrame({
        "adx": adx_val,
        "plus_di": plus_di,
        "minus_di": minus_di
    }, index=df.index)


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Supertrend indicator.
    Returns DataFrame with 'supertrend' (value) and 'st_direction' (+1 up, -1 down).
    """
    hl2 = (df["high"] + df["low"]) / 2
    atr_val = atr(df, period)

    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    st = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1, index=df.index)

    for i in range(period, len(df)):
        if i == period:
            st.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
            continue

        prev_st = st.iloc[i - 1]
        prev_dir = direction.iloc[i - 1]
        curr_close = df["close"].iloc[i]

        # Final upper/lower bands (prevent from going further)
        curr_upper = min(upper_band.iloc[i],
                         prev_st if prev_dir == -1 and upper_band.iloc[i] > prev_st
                         else upper_band.iloc[i])
        curr_lower = max(lower_band.iloc[i],
                         prev_st if prev_dir == 1 and lower_band.iloc[i] < prev_st
                         else lower_band.iloc[i])

        if prev_dir == 1:  # Was bullish
            if curr_close < curr_lower:
                direction.iloc[i] = -1
                st.iloc[i] = curr_upper
            else:
                direction.iloc[i] = 1
                st.iloc[i] = curr_lower
        else:  # Was bearish
            if curr_close > curr_upper:
                direction.iloc[i] = 1
                st.iloc[i] = curr_lower
            else:
                direction.iloc[i] = -1
                st.iloc[i] = curr_upper

    return pd.DataFrame({
        "supertrend": st,
        "st_direction": direction
    }, index=df.index)


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume Weighted Average Price (intraday, resets daily).
    Requires 'timestamp' or datetime index, plus 'high','low','close','volume'.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3

    # If we have a datetime index or 'timestamp' column, group by date
    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.date
    elif "timestamp" in df.columns:
        dates = pd.to_datetime(df["timestamp"]).dt.date
    else:
        # No date info, compute cumulative over entire series
        cum_vol = df["volume"].cumsum()
        cum_tp_vol = (typical_price * df["volume"]).cumsum()
        return cum_tp_vol / cum_vol.replace(0, np.nan)

    result = pd.Series(np.nan, index=df.index)
    for date in pd.unique(dates):
        mask = dates == date
        day_vol = df.loc[mask, "volume"].cumsum()
        day_tp_vol = (typical_price[mask] * df.loc[mask, "volume"]).cumsum()
        result[mask] = day_tp_vol / day_vol.replace(0, np.nan)

    return result


def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands. Returns 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width'."""
    middle = sma(close, period)
    std = close.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    width = (upper - lower) / middle * 100  # As percentage

    return pd.DataFrame({
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "bb_width": width
    }, index=close.index)


def opening_range(df: pd.DataFrame, start: str = "09:15", end: str = "09:30") -> dict:
    """
    Calculate the opening range high/low from the first N minutes.
    Expects datetime index.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        time_filter = df.between_time(start, end)
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        mask = (ts.dt.strftime("%H:%M") >= start) & (ts.dt.strftime("%H:%M") <= end)
        time_filter = df[mask]
    else:
        return {"or_high": np.nan, "or_low": np.nan, "or_range": np.nan}

    if time_filter.empty:
        return {"or_high": np.nan, "or_low": np.nan, "or_range": np.nan}

    or_high = time_filter["high"].max()
    or_low = time_filter["low"].min()
    return {
        "or_high": or_high,
        "or_low": or_low,
        "or_range": or_high - or_low
    }


def pivot_points(prev_high: float, prev_low: float, prev_close: float) -> dict:
    """Standard pivot points from previous day's HLC."""
    pivot = (prev_high + prev_low + prev_close) / 3
    return {
        "pivot": pivot,
        "r1": 2 * pivot - prev_low,
        "s1": 2 * pivot - prev_high,
        "r2": pivot + (prev_high - prev_low),
        "s2": pivot - (prev_high - prev_low),
        "r3": prev_high + 2 * (pivot - prev_low),
        "s3": prev_low - 2 * (prev_high - pivot),
    }
