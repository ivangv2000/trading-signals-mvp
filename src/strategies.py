"""
Estrategias de trading simples y comparables.
Cada estrategia devuelve series booleanas de entrada y salida.
"""

import pandas as pd

STRATEGY_NAMES = [
    "EMA Trend",
    "Momentum Breakout",
    "Pullback Trend",
    "Mean Reversion",
    "Weakness / Exit Signal",
]


def _safe_series(df: pd.DataFrame, col: str, default=False) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return df[col]


def ema_trend_signals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Tendencia por medias exponenciales."""
    entry = (
        (_safe_series(df, "EMA_9") > _safe_series(df, "EMA_21"))
        & (_safe_series(df, "RSI_14") >= 45)
        & (_safe_series(df, "RSI_14") <= 75)
        & (_safe_series(df, "MOMENTUM_5") > 0)
    )
    exit_sig = (
        (_safe_series(df, "EMA_9") < _safe_series(df, "EMA_21"))
        | (_safe_series(df, "RSI_14") < 40)
        | (_safe_series(df, "MOMENTUM_5") < 0)
    )
    return entry.fillna(False), exit_sig.fillna(False)


def momentum_breakout_signals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Ruptura de máximos recientes."""
    high_10 = _safe_series(df, "HIGH_10")
    close = _safe_series(df, "Close")
    volume = _safe_series(df, "Volume", default=0)
    vol_avg = _safe_series(df, "VOLUME_AVG_20")

    volume_ok = True
    if "Volume" in df.columns and df["Volume"].notna().any():
        volume_ok = (volume > 0) & (
            vol_avg.isna() | (volume > 0.8 * vol_avg)
        )

    entry = (
        (close > high_10)
        & (_safe_series(df, "MOMENTUM_5") > 0)
        & (_safe_series(df, "RSI_14") >= 50)
        & (_safe_series(df, "RSI_14") <= 75)
        & volume_ok
    )
    exit_sig = (
        (close < _safe_series(df, "EMA_9"))
        | (_safe_series(df, "MOMENTUM_5") < 0)
        | (_safe_series(df, "RSI_14") < 45)
    )
    return entry.fillna(False), exit_sig.fillna(False)


def pullback_trend_signals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Rebote dentro de tendencia positiva."""
    close = _safe_series(df, "Close")
    ema21 = _safe_series(df, "EMA_21")
    near_ema = (close - ema21).abs() / close.replace(0, float("nan")) < 0.03

    entry = (
        (close > _safe_series(df, "SMA_50"))
        & (_safe_series(df, "EMA_9") > ema21)
        & (_safe_series(df, "RSI_14") >= 40)
        & (_safe_series(df, "RSI_14") <= 60)
        & near_ema
    )
    exit_sig = (
        (_safe_series(df, "RSI_14") > 70)
        | (close < ema21)
        | (_safe_series(df, "MOMENTUM_5") < 0)
    )
    return entry.fillna(False), exit_sig.fillna(False)


def mean_reversion_signals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Rebote tras sobreventa."""
    close = _safe_series(df, "Close")
    vol = _safe_series(df, "VOLATILITY_10")
    vol_threshold = vol.quantile(0.85) if vol.notna().any() else float("inf")

    sma200_ok = True
    if "SMA_200" in df.columns and df["SMA_200"].notna().any():
        sma200_ok = close > _safe_series(df, "SMA_200")

    entry = (
        (_safe_series(df, "RSI_14") < 35)
        & sma200_ok
        & (vol.isna() | (vol < vol_threshold))
    )
    exit_sig = (
        (_safe_series(df, "RSI_14") > 50)
        | (close > _safe_series(df, "EMA_21"))
    )
    return entry.fillna(False), exit_sig.fillna(False)


def weakness_exit_signals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Señal de debilidad — no abre posiciones largas.
    entry siempre False; exit activo cuando hay debilidad.
    """
    sell_condition = weakness_active(df)
    entry = pd.Series(False, index=df.index)
    return entry, sell_condition.fillna(False)


def weakness_active(df: pd.DataFrame) -> pd.Series:
    """True cuando el activo muestra debilidad (SELL)."""
    close = _safe_series(df, "Close")
    below_sma50 = True
    if "SMA_50" in df.columns and df["SMA_50"].notna().any():
        below_sma50 = close < _safe_series(df, "SMA_50")

    return (
        (_safe_series(df, "EMA_9") < _safe_series(df, "EMA_21"))
        & (_safe_series(df, "MOMENTUM_5") < 0)
        & (_safe_series(df, "RSI_14") < 45)
        & below_sma50
    )


def get_strategy_signals(df: pd.DataFrame, strategy_name: str) -> tuple[pd.Series, pd.Series]:
    """Devuelve (entry_signal, exit_signal) para una estrategia."""
    strategies = {
        "EMA Trend": ema_trend_signals,
        "Momentum Breakout": momentum_breakout_signals,
        "Pullback Trend": pullback_trend_signals,
        "Mean Reversion": mean_reversion_signals,
        "Weakness / Exit Signal": weakness_exit_signals,
    }
    fn = strategies.get(strategy_name)
    if fn is None:
        raise ValueError(f"Estrategia desconocida: {strategy_name}")
    return fn(df)


def get_strategy_reasons(df: pd.DataFrame, strategy_name: str) -> list[str]:
    """Razones breves para la última vela."""
    if df.empty:
        return ["Sin datos."]
    last = df.iloc[-1]
    reasons = []

    if strategy_name == "EMA Trend":
        if last.get("EMA_9", 0) > last.get("EMA_21", 0):
            reasons.append("Tendencia corta alcista (EMA 9 > EMA 21).")
        else:
            reasons.append("Tendencia corta bajista.")
    elif strategy_name == "Momentum Breakout":
        if pd.notna(last.get("HIGH_10")) and last.get("Close", 0) > last.get("HIGH_10", 0):
            reasons.append("Precio rompe máximo reciente.")
        else:
            reasons.append("Sin ruptura de máximos.")
    elif strategy_name == "Pullback Trend":
        reasons.append("Busca rebote cerca de EMA 21 en tendencia.")
    elif strategy_name == "Mean Reversion":
        rsi = last.get("RSI_14", 50)
        reasons.append(f"RSI en {rsi:.0f} — busca rebote tras caída.")
    elif strategy_name == "Weakness / Exit Signal":
        if weakness_active(df).iloc[-1]:
            reasons.append("Activo con debilidad — mejor salir.")
        else:
            reasons.append("Sin debilidad extrema ahora.")

    if pd.notna(last.get("RSI_14")):
        reasons.append(f"RSI: {last['RSI_14']:.1f}")
    return reasons[:4]
