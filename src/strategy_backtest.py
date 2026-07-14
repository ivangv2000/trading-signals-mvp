"""
Backtest multiestrategia con scoring de oportunidad.
"""

import numpy as np
import pandas as pd

from src.strategies import (
    STRATEGY_NAMES,
    get_strategy_reasons,
    get_strategy_signals,
    weakness_active,
)


def backtest_strategy(
    df: pd.DataFrame,
    strategy_name: str,
    entry_signal: pd.Series,
    exit_signal: pd.Series,
    holding_period: int = 3,
    transaction_cost: float = 0.001,
) -> tuple[pd.DataFrame, dict]:
    """
    Backtest long-only con shift(1) para evitar lookahead bias.
    """
    df_backtest = df.copy()
    n = len(df_backtest)

    positions = np.zeros(n, dtype=int)
    current_pos = 0
    bars_held = 0
    entry_price = None
    trade_returns = []
    bars_in_trades = []

    entry_arr = entry_signal.fillna(False).values
    exit_arr = exit_signal.fillna(False).values

    for i in range(n):
        positions[i] = current_pos

        if current_pos == 0:
            if entry_arr[i]:
                current_pos = 1
                bars_held = 1
                entry_price = float(df_backtest.iloc[i]["Close"])
        else:
            bars_held += 1
            if exit_arr[i] or bars_held >= holding_period:
                exit_price = float(df_backtest.iloc[i]["Close"])
                if entry_price and entry_price > 0:
                    ret = (exit_price - entry_price) / entry_price - 2 * transaction_cost
                    trade_returns.append(ret)
                    bars_in_trades.append(bars_held)
                current_pos = 0
                bars_held = 0
                entry_price = None

    df_backtest["Return"] = df_backtest["Close"].pct_change()
    df_backtest["Position"] = positions
    df_backtest["Strategy_Return"] = (
        df_backtest["Position"].shift(1).fillna(0) * df_backtest["Return"]
    )
    trades = df_backtest["Position"].diff().abs().fillna(0)
    df_backtest["Strategy_Return"] -= trades * transaction_cost
    df_backtest["Equity_Curve"] = (1 + df_backtest["Strategy_Return"]).cumprod()
    df_backtest["Buy_Hold_Equity"] = (1 + df_backtest["Return"].fillna(0)).cumprod()

    rolling_max = df_backtest["Equity_Curve"].cummax()
    df_backtest["Drawdown"] = (df_backtest["Equity_Curve"] - rolling_max) / rolling_max

    # Marcadores de entrada/salida para gráfico
    pos_diff = df_backtest["Position"].diff().fillna(0)
    df_backtest["Buy_Signal"] = pos_diff == 1
    df_backtest["Sell_Signal"] = pos_diff == -1

    metrics = _calc_metrics(df_backtest, trade_returns, bars_in_trades)
    metrics["strategy_name"] = strategy_name
    return df_backtest, metrics


def _calc_metrics(df_backtest: pd.DataFrame, trade_returns: list, bars_in_trades: list) -> dict:
    valid = df_backtest.dropna(subset=["Close"])
    if valid.empty:
        return _empty_metrics()

    strat_ret = float(round((valid["Equity_Curve"].iloc[-1] - 1) * 100, 2))
    bh_ret = float(round((valid["Buy_Hold_Equity"].iloc[-1] - 1) * 100, 2))

    win_rate, avg_trade = _trade_stats(trade_returns)
    max_dd = float(round(valid["Drawdown"].min() * 100, 2))

    return {
        "strategy_total_return": strat_ret,
        "buy_hold_total_return": bh_ret,
        "excess_return": float(round(strat_ret - bh_ret, 2)),
        "max_drawdown": max_dd,
        "num_trades": int(len(trade_returns)),
        "win_rate": win_rate,
        "avg_trade_return": avg_trade,
        "best_trade": float(round(max(trade_returns) * 100, 2)) if trade_returns else 0.0,
        "worst_trade": float(round(min(trade_returns) * 100, 2)) if trade_returns else 0.0,
        "avg_bars_in_trade": float(round(float(np.mean(bars_in_trades)), 1)) if bars_in_trades else 0.0,
    }


def _empty_metrics() -> dict:
    return {
        "strategy_total_return": 0.0,
        "buy_hold_total_return": 0.0,
        "excess_return": 0.0,
        "max_drawdown": 0.0,
        "num_trades": 0,
        "win_rate": 0.0,
        "avg_trade_return": 0.0,
        "best_trade": 0.0,
        "worst_trade": 0.0,
        "avg_bars_in_trade": 0.0,
    }


def _trade_stats(trade_returns: list) -> tuple[float, float]:
    if not trade_returns:
        return 0.0, 0.0
    wins = sum(1 for r in trade_returns if r > 0)
    win_rate = float(round((wins / len(trade_returns)) * 100, 2))
    avg_trade = float(round(float(np.mean(trade_returns)) * 100, 2))
    return win_rate, avg_trade


def calculate_score(metrics: dict, df: pd.DataFrame, entry_now: bool) -> int:
    """Calcula score de oportunidad 0-100."""
    score = 50.0
    m = metrics

    if m.get("strategy_total_return", 0) > 0:
        score += 15
    if m.get("excess_return", 0) > 0:
        score += 15
    if m.get("win_rate", 0) > 50:
        score += 10
    if m.get("avg_trade_return", 0) > 0:
        score += 10
    if m.get("num_trades", 0) >= 8:
        score += 10
    if m.get("max_drawdown", -100) > -15:
        score += 10

    if m.get("num_trades", 0) < 5:
        score -= 20
    if m.get("max_drawdown", 0) < -25:
        score -= 15
    if m.get("avg_trade_return", 0) < 0:
        score -= 15

    # Volatilidad alta
    if "VOLATILITY_10" in df.columns and df["VOLATILITY_10"].notna().any():
        vol = df["VOLATILITY_10"].iloc[-1]
        threshold = df["VOLATILITY_10"].quantile(0.80)
        if pd.notna(vol) and pd.notna(threshold) and vol > threshold:
            score -= 10

    # Bonus leve si hay entrada activa ahora
    if entry_now:
        score += 5

    return int(max(0, min(100, round(score))))


def _high_volatility(df: pd.DataFrame) -> bool:
    if "VOLATILITY_10" not in df.columns or not df["VOLATILITY_10"].notna().any():
        return False
    vol = df["VOLATILITY_10"].iloc[-1]
    threshold = df["VOLATILITY_10"].quantile(0.85)
    return pd.notna(vol) and pd.notna(threshold) and vol > threshold


def _extreme_volatility(df: pd.DataFrame) -> bool:
    """Volatilidad extrema (percentil 90)."""
    if "VOLATILITY_10" not in df.columns or not df["VOLATILITY_10"].notna().any():
        return False
    vol = df["VOLATILITY_10"].iloc[-1]
    threshold = df["VOLATILITY_10"].quantile(0.90)
    return pd.notna(vol) and pd.notna(threshold) and vol > threshold


def _determine_signal(
    score: int,
    entry_now: bool,
    num_trades: int,
    high_vol: bool,
    extreme_vol: bool,
    data_ok: bool,
    max_drawdown: float,
) -> tuple[str, str]:
    """Devuelve (signal, avoid_reason). SELL se gestiona aparte con weakness."""
    if not data_ok:
        return "AVOID", "Pocos datos"

    very_bad_dd = max_drawdown < -30

    # AVOID solo en casos malos
    if score < 40:
        return "AVOID", "Backtest débil"
    if num_trades < 3:
        return "AVOID", "Pocas operaciones"
    if extreme_vol:
        return "AVOID", "Volatilidad extrema"
    if very_bad_dd and score < 45:
        return "AVOID", "Drawdown muy malo"

    # BUY con umbral más bajo
    if entry_now and score >= 55:
        return "BUY", ""

    # HOLD si no hay entrada clara pero el score no es desastroso
    if score >= 45:
        return "HOLD", ""

    if high_vol:
        return "HOLD", ""

    return "HOLD", ""


def _risk_level(score: int, high_vol: bool) -> str:
    if high_vol or score < 45:
        return "High"
    if score >= 70:
        return "Low"
    return "Medium"


def run_all_strategies(
    df: pd.DataFrame,
    mode: str = "swing",
    holding_period: int = 3,
    transaction_cost: float = 0.001,
) -> tuple[list[dict], dict]:
    """
    Ejecuta todas las estrategias, calcula scores y elige la mejor.

    Returns:
        (results_list, best_result)
    """
    min_rows = 40 if mode == "swing" else 60
    data_ok = len(df) >= min_rows and "Close" in df.columns

    weakness_now = bool(weakness_active(df).iloc[-1]) if len(df) > 0 else False
    high_vol = _high_volatility(df)
    extreme_vol = _extreme_volatility(df)

    results_list = []

    for name in STRATEGY_NAMES:
        entry_sig, exit_sig = get_strategy_signals(df, name)
        df_bt, metrics = backtest_strategy(
            df, name, entry_sig, exit_sig, holding_period, transaction_cost
        )

        entry_now = bool(entry_sig.iloc[-1]) if len(entry_sig) > 0 else False
        score = calculate_score(metrics, df, entry_now)

        # Weakness no compite como estrategia de entrada
        if name == "Weakness / Exit Signal":
            current_signal = "SELL" if weakness_now else "HOLD"
        else:
            sig, _ = _determine_signal(
                score,
                entry_now,
                metrics["num_trades"],
                high_vol,
                extreme_vol,
                data_ok,
                metrics.get("max_drawdown", 0),
            )
            current_signal = sig

        result = {
            "strategy_name": name,
            "current_signal": current_signal,
            "score": score,
            "risk_level": _risk_level(score, high_vol),
            "metrics": metrics,
            "df_backtest": df_bt,
            "reasons": get_strategy_reasons(df, name),
            "entry_now": entry_now,
        }
        results_list.append(result)

    # Elegir mejor estrategia (excluir Weakness para ranking de oportunidad)
    tradable = [r for r in results_list if r["strategy_name"] != "Weakness / Exit Signal"]
    if not tradable:
        best = results_list[0] if results_list else {}
    else:
        best = max(tradable, key=lambda r: r["score"])

    # Señal final del ticker
    best_entry = best.get("entry_now", False)
    best_score = best.get("score", 0)
    best_metrics = best.get("metrics", {})

    final_signal, avoid_reason = _determine_signal(
        best_score,
        best_entry,
        best_metrics.get("num_trades", 0),
        high_vol,
        extreme_vol,
        data_ok,
        best_metrics.get("max_drawdown", 0),
    )

    # SELL solo si Weakness Signal está activo
    if weakness_now:
        final_signal = "SELL"
        avoid_reason = ""

    best_result = {
        **best,
        "current_signal": final_signal,
        "avoid_reason": avoid_reason,
        "weakness_now": weakness_now,
        "high_volatility": high_vol,
        "data_ok": data_ok,
    }

    return results_list, best_result


def analyze_ticker(
    ticker: str,
    df: pd.DataFrame,
    mode: str,
    holding_period: int,
    transaction_cost: float = 0.001,
) -> dict:
    """Analiza un ticker y devuelve resultados completos."""
    results_list, best_result = run_all_strategies(
        df, mode=mode, holding_period=holding_period, transaction_cost=transaction_cost
    )

    last = df.dropna(subset=["Close"]).iloc[-1] if not df.empty else None
    stop = None
    if last is not None and pd.notna(last.get("ATR_14")) and pd.notna(last.get("Close")):
        stop = float(last["Close"]) - 1.5 * float(last["ATR_14"])

    return {
        "ticker": ticker.upper(),
        "results_list": results_list,
        "best": best_result,
        "df": df,
        "last_price": float(last["Close"]) if last is not None else None,
        "stop_price": stop,
        "last_date": df.index[-1] if len(df) > 0 else None,
    }
