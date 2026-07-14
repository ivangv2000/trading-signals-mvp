"""
Backtest simple long-only sin apalancamiento.
Incluye modo diario clásico y modo corto plazo (swing / intradía).
"""

import numpy as np
import pandas as pd

# Columnas mínimas obligatorias para el backtest de corto plazo
REQUIRED_SHORT_TERM_COLS = ["Close", "EMA_9", "EMA_21", "RSI_14", "MOMENTUM_5"]


def run_backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Backtest diario clásico: tendencia + RSI.

    Regla de entrada (estar comprado):
        - Close > SMA_50, SMA_50 > SMA_200, RSI_14 entre 45 y 70

    Returns:
        Tupla (DataFrame con columnas de backtest, diccionario de métricas).
    """
    result = df.copy()

    in_market = (
        (result["Close"] > result["SMA_50"])
        & (result["SMA_50"] > result["SMA_200"])
        & (result["RSI_14"] >= 45)
        & (result["RSI_14"] <= 70)
    )

    result["position"] = in_market.fillna(False).astype(int)
    result["market_return"] = result["Close"].pct_change()
    result["strategy_return"] = result["position"].shift(1).fillna(0) * result["market_return"]
    result["strategy_equity"] = (1 + result["strategy_return"]).cumprod()
    result["buy_hold_equity"] = (1 + result["market_return"]).cumprod()

    valid = result.dropna(subset=["SMA_200", "RSI_14"])

    if valid.empty:
        return result, _empty_classic_metrics()

    total_return_strategy = (valid["strategy_equity"].iloc[-1] - 1) * 100
    total_return_buy_hold = (valid["buy_hold_equity"].iloc[-1] - 1) * 100

    rolling_max = valid["strategy_equity"].cummax()
    drawdown = (valid["strategy_equity"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    position_changes = valid["position"].diff().fillna(0)
    entries = valid[position_changes == 1].index
    exits = valid[position_changes == -1].index
    num_trades = min(len(entries), len(exits))

    trade_returns = []
    for i in range(num_trades):
        entry_price = valid.loc[entries[i], "Close"]
        exit_price = valid.loc[exits[i], "Close"]
        trade_returns.append((exit_price - entry_price) / entry_price)

    win_rate, avg_trade_return = _trade_stats(trade_returns)

    metrics = {
        "total_return_strategy": float(round(total_return_strategy, 2)),
        "total_return_buy_hold": float(round(total_return_buy_hold, 2)),
        "max_drawdown": float(round(max_drawdown, 2)),
        "num_trades": int(num_trades),
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
    }

    return result, metrics


def _empty_classic_metrics() -> dict:
    return {
        "total_return_strategy": 0.0,
        "total_return_buy_hold": 0.0,
        "max_drawdown": 0.0,
        "num_trades": 0,
        "win_rate": 0.0,
        "avg_trade_return": 0.0,
    }


def _empty_short_term_metrics() -> dict:
    return {
        "strategy_total_return": 0.0,
        "buy_hold_total_return": 0.0,
        "max_drawdown": 0.0,
        "num_trades": 0,
        "win_rate": 0.0,
        "avg_trade_return": 0.0,
        "best_trade": 0.0,
        "worst_trade": 0.0,
        "avg_bars_in_trade": 0.0,
    }


def _trade_stats(trade_returns: list) -> tuple[float, float]:
    """Calcula win rate y rentabilidad media por operación."""
    if not trade_returns:
        return 0.0, 0.0
    wins = sum(1 for r in trade_returns if r > 0)
    win_rate = float(round((wins / len(trade_returns)) * 100, 2))
    avg_trade_return = float(round(float(np.mean(trade_returns)) * 100, 2))
    return win_rate, avg_trade_return


def _check_required_columns(df: pd.DataFrame) -> None:
    """Comprueba que existan las columnas mínimas; lanza ValueError si falta alguna."""
    missing = [col for col in REQUIRED_SHORT_TERM_COLS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas necesarias para el backtest de corto plazo: {', '.join(missing)}. "
            "Asegúrate de calcular los indicadores antes de ejecutar el backtest."
        )


def _swing_entry(row, has_sma50: bool) -> bool:
    """Condición de entrada para modo swing."""
    if not (
        pd.notna(row["EMA_9"])
        and pd.notna(row["EMA_21"])
        and pd.notna(row["RSI_14"])
        and pd.notna(row["MOMENTUM_5"])
        and row["EMA_9"] > row["EMA_21"]
        and 45 <= row["RSI_14"] <= 70
        and row["MOMENTUM_5"] > 0
    ):
        return False

    # SMA_50 es opcional: solo se exige si existe y tiene valor
    if has_sma50:
        sma50 = row.get("SMA_50")
        if pd.notna(sma50):
            return row["Close"] > sma50
    return True


def _intraday_entry(row, has_vwap: bool) -> bool:
    """Condición de entrada para modo intradía."""
    if not (
        pd.notna(row["EMA_9"])
        and pd.notna(row["EMA_21"])
        and pd.notna(row["RSI_14"])
        and pd.notna(row["MOMENTUM_5"])
        and row["EMA_9"] > row["EMA_21"]
        and 45 <= row["RSI_14"] <= 70
        and row["MOMENTUM_5"] > 0
    ):
        return False

    # VWAP es opcional: solo se exige si existe y tiene valor
    if has_vwap:
        vwap = row.get("VWAP")
        if pd.notna(vwap):
            return row["Close"] > vwap
    return True


def _exit_condition(row) -> bool:
    """Condición de salida común para swing e intradía."""
    if not (
        pd.notna(row["EMA_9"])
        and pd.notna(row["EMA_21"])
        and pd.notna(row["RSI_14"])
        and pd.notna(row["MOMENTUM_5"])
    ):
        return False
    return (
        row["EMA_9"] < row["EMA_21"]
        or row["RSI_14"] < 40
        or row["MOMENTUM_5"] < 0
    )


def run_short_term_backtest(
    df: pd.DataFrame,
    mode: str = "swing",
    holding_period: int = 3,
    transaction_cost: float = 0.001,
) -> tuple[pd.DataFrame, dict]:
    """
    Backtest corto plazo con salida por señal o por holding_period velas.

    Evita lookahead bias: la posición se aplica con shift(1) en los retornos.

    Args:
        df: DataFrame con precios e indicadores.
        mode: "swing" o "intraday".
        holding_period: Máximo de velas a mantener la posición.
        transaction_cost: Coste por cambio de posición.

    Returns:
        Tupla (df_backtest, metrics).
    """
    _check_required_columns(df)

    if mode not in ("swing", "intraday"):
        raise ValueError(f"Modo '{mode}' no válido. Usa 'swing' o 'intraday'.")

    df_backtest = df.copy()
    n = len(df_backtest)
    has_sma50 = "SMA_50" in df_backtest.columns
    has_vwap = "VWAP" in df_backtest.columns

    if mode == "swing":
        entry_fn = lambda row: _swing_entry(row, has_sma50)
    else:
        entry_fn = lambda row: _intraday_entry(row, has_vwap)

    # --- Simular posiciones barra a barra ---
    positions = np.zeros(n, dtype=int)
    current_pos = 0
    bars_held = 0
    entry_price = None

    trade_returns = []
    bars_in_trades = []

    for i in range(n):
        positions[i] = current_pos
        row = df_backtest.iloc[i]

        if current_pos == 0:
            if entry_fn(row):
                current_pos = 1
                bars_held = 1
                entry_price = float(row["Close"])
        else:
            bars_held += 1
            should_exit = _exit_condition(row) or bars_held >= holding_period

            if should_exit:
                exit_price = float(row["Close"])
                if entry_price and entry_price > 0:
                    ret = (exit_price - entry_price) / entry_price - 2 * transaction_cost
                    trade_returns.append(ret)
                    bars_in_trades.append(bars_held)
                current_pos = 0
                bars_held = 0
                entry_price = None

    # --- Columnas de backtest (nombres exactos solicitados) ---
    df_backtest["Return"] = df_backtest["Close"].pct_change()
    df_backtest["Position"] = positions

    # Evitar lookahead bias: la posición de ayer determina el retorno de hoy
    df_backtest["Strategy_Return"] = (
        df_backtest["Position"].shift(1).fillna(0) * df_backtest["Return"]
    )

    # Restar coste de transacción en cada cambio de posición
    position_changes = df_backtest["Position"].diff().abs().fillna(0)
    df_backtest["Strategy_Return"] = df_backtest["Strategy_Return"] - position_changes * transaction_cost

    df_backtest["Equity_Curve"] = (1 + df_backtest["Strategy_Return"]).cumprod()
    df_backtest["Buy_Hold_Equity"] = (1 + df_backtest["Return"].fillna(0)).cumprod()

    rolling_max = df_backtest["Equity_Curve"].cummax()
    df_backtest["Drawdown"] = (df_backtest["Equity_Curve"] - rolling_max) / rolling_max

    # --- Métricas ---
    valid = df_backtest.dropna(subset=["EMA_21", "RSI_14"])
    if valid.empty:
        return df_backtest, _empty_short_term_metrics()

    strategy_total_return = float(round((valid["Equity_Curve"].iloc[-1] - 1) * 100, 2))
    buy_hold_total_return = float(round((valid["Buy_Hold_Equity"].iloc[-1] - 1) * 100, 2))
    max_drawdown = float(round(valid["Drawdown"].min() * 100, 2))

    num_trades = len(trade_returns)
    win_rate, avg_trade_return = _trade_stats(trade_returns)

    best_trade = float(round(max(trade_returns) * 100, 2)) if trade_returns else 0.0
    worst_trade = float(round(min(trade_returns) * 100, 2)) if trade_returns else 0.0
    avg_bars_in_trade = float(round(float(np.mean(bars_in_trades)), 1)) if bars_in_trades else 0.0

    metrics = {
        "strategy_total_return": strategy_total_return,
        "buy_hold_total_return": buy_hold_total_return,
        "max_drawdown": max_drawdown,
        "num_trades": int(num_trades),
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "avg_bars_in_trade": avg_bars_in_trade,
    }

    return df_backtest, metrics
