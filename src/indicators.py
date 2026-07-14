"""
Indicadores técnicos para Trading Signals Lab.
"""

import pandas as pd


def _calc_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _calc_vwap(df: pd.DataFrame) -> pd.Series:
    if "Volume" not in df.columns or "High" not in df.columns or "Low" not in df.columns:
        return pd.Series(float("nan"), index=df.index)

    volume = df["Volume"].fillna(0)
    if volume.sum() == 0:
        return pd.Series(float("nan"), index=df.index)

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_vol = typical_price * volume

    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.date
        cum_tp_vol = tp_vol.groupby(dates).cumsum()
        cum_vol = volume.groupby(dates).cumsum()
        return cum_tp_vol / cum_vol.replace(0, float("nan"))

    cum_tp_vol = tp_vol.cumsum()
    cum_vol = volume.cumsum().replace(0, float("nan"))
    return cum_tp_vol / cum_vol


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Añade todos los indicadores necesarios para las estrategias."""
    result = df.copy()
    close = result["Close"]

    # Medias móviles
    result["SMA_20"] = close.rolling(window=20).mean()
    result["SMA_50"] = close.rolling(window=50).mean()
    result["SMA_200"] = close.rolling(window=200).mean()
    result["EMA_9"] = close.ewm(span=9, adjust=False).mean()
    result["EMA_21"] = close.ewm(span=21, adjust=False).mean()

    # RSI y momentum
    result["RSI_14"] = _calc_rsi(close, window=14)
    result["MOMENTUM_5"] = close - close.shift(5)
    result["MOMENTUM_20"] = close - close.shift(20)

    # Volatilidad
    daily_returns = close.pct_change()
    result["VOLATILITY_10"] = daily_returns.rolling(window=10).std()
    result["VOLATILITY_20"] = daily_returns.rolling(window=20).std()

    # Máximos/mínimos sin lookahead
    if "High" in result.columns:
        result["HIGH_10"] = result["High"].rolling(10).max().shift(1)
    if "Low" in result.columns:
        result["LOW_10"] = result["Low"].rolling(10).min().shift(1)

    # Volumen medio
    if "Volume" in result.columns:
        result["VOLUME_AVG_20"] = result["Volume"].rolling(window=20).mean()
    else:
        result["VOLUME_AVG_20"] = float("nan")

    # ATR
    if "High" in result.columns and "Low" in result.columns:
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                result["High"] - result["Low"],
                (result["High"] - prev_close).abs(),
                (result["Low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        result["ATR_14"] = tr.rolling(window=14).mean()
    else:
        result["ATR_14"] = float("nan")

    result["VWAP"] = _calc_vwap(result)
    return result
