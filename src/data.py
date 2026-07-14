"""
Descarga y limpieza de datos de precios con yfinance.
"""

import pandas as pd
import yfinance as yf

# Intervalos soportados
VALID_INTERVALS = ("1d", "15m", "30m", "1h")


def download_price_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Descarga datos de precios de un ticker usando yfinance.

    Args:
        ticker: Símbolo del activo (ej. AAPL, MSFT).
        period: Periodo histórico (ej. 6mo, 1y, 2y, 5d, 1mo, 60d).
        interval: Intervalo de velas: 1d, 15m, 30m, 1h.

    Returns:
        DataFrame limpio con columnas OHLCV.

    Raises:
        ValueError: Si no hay datos disponibles o el intervalo no es válido.
    """
    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Debes introducir un ticker válido (ej. AAPL).")

    if interval not in VALID_INTERVALS:
        raise ValueError(
            f"Intervalo '{interval}' no soportado. "
            f"Usa uno de: {', '.join(VALID_INTERVALS)}"
        )

    # Descargar datos con ajuste automático de precios
    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if raw is None or raw.empty:
        raise ValueError(
            f"No se encontraron datos para '{ticker}' "
            f"(periodo={period}, intervalo={interval}). "
            "Comprueba el ticker, el periodo y el intervalo."
        )

    df = raw.copy()

    # yfinance a veces devuelve columnas MultiIndex; las aplanamos
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Renombrar columnas a formato estándar
    column_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(
        columns={c: column_map[c.lower()] for c in df.columns if c.lower() in column_map}
    )

    # Asegurar que existan las columnas OHLCV si están disponibles
    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    available = [col for col in expected_cols if col in df.columns]
    df = df[available]

    # Eliminar filas vacías o sin precio de cierre
    if "Close" in df.columns:
        df = df.dropna(subset=["Close"])
    df = df.dropna(how="all")

    if df.empty:
        raise ValueError(
            f"Los datos de '{ticker}' están vacíos después de limpiarlos. "
            "Prueba con otro periodo, intervalo o ticker."
        )

    return df
