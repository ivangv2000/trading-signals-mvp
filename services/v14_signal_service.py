"""Servicio de señales V14 — reutiliza la lógica existente de portfolio_v14.

Paper trading únicamente. No ejecuta órdenes.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data import download_price_data
from src.portfolio_v14 import (
    build_close_prices,
    calculate_v14_features,
    calculate_v14_weights_schedule,
    generate_v14_signals,
    get_current_v14_portfolio_signal,
    get_required_tickers,
    load_v14_config,
)

STRATEGY_NAME = "V14 R1 Return Engine"
DEFAULT_CAPITAL_REFERENCE = 100.0

SIGNAL_REASONS = {
    "BUY": "entra en la cartera V14 esta semana.",
    "INCREASE": "aumenta peso en la cartera V14.",
    "HOLD": "continúa seleccionada sin cambio de peso.",
    "SELL": "ha salido de la cartera V14.",
    "REDUCE": "reduce peso en la cartera V14.",
    "AVOID": "no está seleccionada en la cartera V14.",
}


def _download_data_dict(tickers: list[str], period: str = "5y") -> tuple[dict, list[str]]:
    data_dict: dict = {}
    errors: list[str] = []
    for ticker in tickers:
        t = ticker.strip().upper()
        if not t or t == "CASH":
            continue
        try:
            data_dict[t] = download_price_data(t, period=period, interval="1d")
        except Exception as exc:
            errors.append(f"{t}: {exc}")
    return data_dict, errors


def _parse_universe(universe_text: str | None) -> list[str] | None:
    if not universe_text or not universe_text.strip():
        return None
    return [t.strip().upper() for t in universe_text.replace("\n", ",").split(",") if t.strip()]


def derive_global_action(signals_df: pd.DataFrame | None) -> str:
    if signals_df is None or signals_df.empty:
        return "NO DATA"

    sigs = signals_df["signal"].astype(str).str.upper()
    relevant = signals_df[
        (signals_df["target_weight"].fillna(0) > 0.001)
        | (signals_df["previous_weight"].fillna(0) > 0.001)
    ]
    if relevant.empty:
        return "NO DATA"

    rel_sigs = relevant["signal"].astype(str).str.upper()
    has_buy = rel_sigs.isin(["BUY"]).any()
    has_increase = rel_sigs.isin(["INCREASE"]).any()
    has_sell = rel_sigs.isin(["SELL"]).any()
    has_reduce = rel_sigs.isin(["REDUCE"]).any()
    has_weight_change = has_increase or has_reduce

    if has_buy and (has_sell or has_reduce or has_increase):
        return "REBALANCE"
    if has_buy:
        return "BUY"
    if has_sell and not has_reduce and not has_increase and not has_buy:
        return "SELL"
    if has_sell or has_reduce or has_increase:
        return "REBALANCE"
    if rel_sigs.eq("HOLD").all():
        return "HOLD"
    return "HOLD"


def normalize_v14_signals(
    raw_signal: dict,
    signal_date: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Normaliza la salida V14 al esquema público del snapshot."""
    config = raw_signal.get("config") or load_v14_config()
    strategy = config.get("strategy_name", STRATEGY_NAME)
    date = signal_date or raw_signal.get("last_date") or raw_signal.get("rebalance_date") or ""

    full_df = raw_signal.get("signals_full")
    if full_df is None or (isinstance(full_df, pd.DataFrame) and full_df.empty):
        full_df = raw_signal.get("signals", pd.DataFrame())

    if full_df is None or full_df.empty:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "ticker",
                "signal",
                "previous_weight",
                "target_weight",
                "strategy",
                "reason",
            ]
        ), {
            "strategy": strategy,
            "signal_date": date,
            "global_action": "NO DATA",
        }

    rows = []
    for _, row in full_df.iterrows():
        sig = str(row.get("signal", "AVOID")).upper()
        ticker = str(row.get("ticker", "")).upper()
        pw = float(row.get("previous_weight", 0) or 0)
        tw = float(row.get("target_weight", 0) or 0)
        reason = row.get("reason", "")
        if isinstance(reason, str) and reason.startswith("A_tsmom_63"):
            reason = SIGNAL_REASONS.get(sig, reason)
        rows.append(
            {
                "signal_date": date,
                "ticker": ticker,
                "signal": sig,
                "previous_weight": round(pw, 4),
                "target_weight": round(tw, 4),
                "strategy": strategy,
                "reason": reason,
            }
        )

    df = pd.DataFrame(rows)
    summary_meta = {
        "strategy": strategy,
        "signal_date": date,
        "global_action": derive_global_action(df),
    }
    return df, summary_meta


def calculate_v14_signals(
    capital: float = DEFAULT_CAPITAL_REFERENCE,
    universe_text: str | None = None,
) -> dict:
    """
    Ejecuta la lógica V14 existente (descarga datos + get_current_v14_portfolio_signal).
    No duplica el algoritmo.
    """
    config = load_v14_config()
    custom = _parse_universe(universe_text)
    tickers = custom if custom else get_required_tickers(config)

    data_dict, dl_errors = _download_data_dict(tickers, period="5y")
    signal = get_current_v14_portfolio_signal(data_dict, capital=capital, config=config)

    if signal.get("error"):
        return {
            "error": signal["error"],
            "download_errors": dl_errors,
            "approved_for_real_money": False,
        }

    try:
        close_prices = build_close_prices(data_dict)
        features = calculate_v14_features(close_prices)
        schedule = calculate_v14_weights_schedule(close_prices, config, features)
        rdates = sorted(schedule.keys())
        last_dt = rdates[-1]
        prev_dt = rdates[-2] if len(rdates) >= 2 else None
        target = schedule[last_dt]
        previous = schedule[prev_dt] if prev_dt else pd.Series(dtype=float)
        signal["signals_full"] = generate_v14_signals(target, previous, features, last_dt)
    except Exception:
        signal["signals_full"] = signal.get("signals", pd.DataFrame())

    signal["download_errors"] = dl_errors
    signal["approved_for_real_money"] = False
    signal["computed_at"] = datetime.utcnow().isoformat() + "Z"
    signal["tickers_analyzed_count"] = len(tickers)
    return signal


def load_latest_v14_snapshot():
    from services.v14_snapshot_service import load_latest_v14_snapshot as _load

    return _load()


def save_v14_snapshot(raw_signal: dict, capital_reference: float = DEFAULT_CAPITAL_REFERENCE) -> dict:
    from services.v14_snapshot_service import save_v14_snapshot as _save

    return _save(raw_signal, capital_reference=capital_reference)
