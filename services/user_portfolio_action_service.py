"""Traduce señales del modelo V14 en acciones personales según la cartera del usuario."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from src.portfolio_v14 import load_v14_config

PortfolioMode = Literal["new_zero", "existing", "manual"]
DEFAULT_TOLERANCE = 0.01

DEFENSIVE_TICKERS = {"SHY", "CASH", "IEF", "TLT", "GLD", "USMV", "QUAL", "SCHD", "XLV", "XLP"}

PORTFOLIO_MODE_LABELS = {
    "new_zero": "Empiezo hoy desde cero",
    "existing": "Ya tengo la cartera anterior",
    "manual": "Introducir mi cartera manualmente",
}

USER_GLOBAL_ACTION_TEXT = {
    "BUY": "Para replicar la cartera V14 desde cero debes abrir las posiciones mostradas.",
    "REBALANCE": "Hay compras, ventas o cambios de peso para ajustar tu cartera personal.",
    "HOLD": (
        "No es necesario realizar cambios porque la cartera anterior ya coincide con "
        "la cartera objetivo."
    ),
    "SELL": "Hay posiciones que debes cerrar o reducir a cero.",
    "NO DATA": "Aún no existe un snapshot de señales V14.",
}

FRACTIONAL_WARNING = (
    "Las cantidades son orientativas. Las acciones y ETF no siempre permiten compras "
    "fraccionadas; los costes de operación y el tipo de cambio pueden alterar el resultado real."
)


def derive_user_action(
    user_weight: float,
    target_weight: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> str:
    user_weight = float(user_weight or 0)
    target_weight = float(target_weight or 0)

    if target_weight <= 0.001:
        if user_weight > tolerance:
            return "SELL"
        return "AVOID"

    if user_weight <= tolerance:
        return "BUY"
    if user_weight < target_weight - tolerance:
        return "BUY"
    if user_weight > target_weight + tolerance:
        return "REDUCE"
    return "HOLD"


def derive_user_global_action(actions_df: pd.DataFrame | None) -> str:
    if actions_df is None or actions_df.empty:
        return "NO DATA"

    relevant = actions_df[actions_df["user_action"].astype(str).str.upper() != "AVOID"]
    if relevant.empty:
        return "NO DATA"

    actions = relevant["user_action"].astype(str).str.upper()
    has_buy = actions.isin(["BUY"]).any()
    has_reduce = actions.isin(["REDUCE"]).any()
    has_sell = actions.isin(["SELL"]).any()

    if has_buy and (has_sell or has_reduce):
        return "REBALANCE"
    if has_buy:
        return "BUY"
    if has_sell and not has_reduce and not has_buy:
        return "SELL"
    if has_sell or has_reduce:
        return "REBALANCE"
    if actions.eq("HOLD").all():
        return "HOLD"
    return "HOLD"


def _user_holdings_for_mode(
    signals_df: pd.DataFrame,
    mode: PortfolioMode,
    capital: float,
    manual_holdings: dict[str, float] | None,
) -> dict[str, float]:
    holdings: dict[str, float] = {}
    if mode == "new_zero":
        return {str(t).upper(): 0.0 for t in signals_df["ticker"].astype(str)}

    if mode == "existing":
        for _, row in signals_df.iterrows():
            ticker = str(row["ticker"]).upper()
            holdings[ticker] = float(row.get("previous_weight", 0) or 0)
        return holdings

    manual_holdings = manual_holdings or {}
    capital = max(float(capital), 1.0)
    for _, row in signals_df.iterrows():
        ticker = str(row["ticker"]).upper()
        euros = float(manual_holdings.get(ticker, 0) or 0)
        holdings[ticker] = euros / capital
    return holdings


def explain_user_action(
    ticker: str,
    model_signal: str,
    user_action: str,
    mode: PortfolioMode,
    user_amount: float,
    target_amount: float,
) -> str:
    model_signal = (model_signal or "").upper()
    user_action = (user_action or "").upper()
    ticker = ticker.upper()

    if user_action == "BUY" and mode == "new_zero" and model_signal == "HOLD":
        return (
            f"El modelo mantiene {ticker} al objetivo indicado, pero como estás empezando "
            f"desde cero necesitas abrir la posición."
        )
    if user_action == "BUY":
        return f"Tu peso actual ({user_amount:.0f} €) está por debajo del objetivo ({target_amount:.0f} €)."
    if user_action == "REDUCE":
        return f"Tu peso actual supera el objetivo del modelo; conviene reducir exposición en {ticker}."
    if user_action == "SELL":
        return f"{ticker} ya no forma parte de la cartera objetivo V14; deberías cerrar la posición."
    if user_action == "HOLD" and mode == "existing":
        return "Tu cartera coincide con el objetivo del modelo para este activo."
    if user_action == "HOLD":
        return "No necesitas mover este activo dentro de la tolerancia configurada."
    if user_action == "AVOID":
        return "No forma parte de la cartera objetivo y no tienes exposición relevante."
    return "Revisa el peso objetivo del modelo antes de actuar."


def build_user_portfolio_actions(
    signals_df: pd.DataFrame,
    capital: float,
    mode: PortfolioMode = "new_zero",
    manual_holdings: dict[str, float] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> pd.DataFrame:
    if signals_df is None or signals_df.empty:
        return pd.DataFrame()

    user_weights = _user_holdings_for_mode(signals_df, mode, capital, manual_holdings)
    rows = []

    for _, row in signals_df.iterrows():
        ticker = str(row["ticker"]).upper()
        model_signal = str(row.get("signal", "AVOID")).upper()
        model_prev = float(row.get("previous_weight", 0) or 0)
        target_w = float(row.get("target_weight", 0) or 0)
        user_w = float(user_weights.get(ticker, 0) or 0)

        user_action = derive_user_action(user_w, target_w, tolerance=tolerance)
        if mode == "new_zero" and target_w <= 0.001:
            user_action = "AVOID"

        user_amount = round(capital * user_w, 2)
        target_amount = round(capital * target_w, 2)
        diff_amount = round(target_amount - user_amount, 2)

        rows.append(
            {
                "ticker": ticker,
                "model_signal": model_signal,
                "model_previous_weight": model_prev,
                "model_target_weight": target_w,
                "user_current_weight": user_w,
                "user_action": user_action,
                "user_amount_eur": user_amount,
                "target_amount_eur": target_amount,
                "diff_amount_eur": diff_amount,
                "is_defensive": ticker in DEFENSIVE_TICKERS,
                "reason": explain_user_action(
                    ticker, model_signal, user_action, mode, user_amount, target_amount
                ),
            }
        )

    return pd.DataFrame(rows)


def count_position_buckets(actions_df: pd.DataFrame) -> dict[str, int]:
    if actions_df is None or actions_df.empty:
        return {"risk_positions": 0, "defensive_positions": 0, "total_assets": 0}

    active = actions_df[actions_df["model_target_weight"].fillna(0) > 0.001]
    defensive = active[active["is_defensive"]]
    risk = active[~active["is_defensive"]]
    return {
        "risk_positions": int(len(risk)),
        "defensive_positions": int(len(defensive)),
        "total_assets": int(len(active)),
    }


def build_universe_stats(raw_signal: dict | None, summary: dict | None = None) -> dict:
    stats = (summary or {}).get("universe_stats") if summary else None
    if stats:
        return stats

    if not raw_signal:
        return {}

    config = raw_signal.get("config") or load_v14_config()
    target_weights = raw_signal.get("target_weights") or {}
    full_df = raw_signal.get("signals_full")
    if full_df is None:
        full_df = raw_signal.get("signals")

    analyzed = raw_signal.get("tickers_analyzed_count")
    if analyzed is None:
        from src.portfolio_v14 import get_required_tickers

        analyzed = len(get_required_tickers(config))

    active_tickers = [t for t, w in target_weights.items() if float(w) > 0.001]
    defensive = [t for t in active_tickers if t in DEFENSIVE_TICKERS]
    risk = [t for t in active_tickers if t not in DEFENSIVE_TICKERS]

    eligible = None
    discarded = None
    if isinstance(full_df, pd.DataFrame) and not full_df.empty:
        eligible = int(
            full_df[
                (full_df["target_weight"].fillna(0) > 0.001)
                | (full_df["previous_weight"].fillna(0) > 0.001)
            ]["ticker"].nunique()
        )
        if analyzed:
            discarded = max(0, int(analyzed) - len(active_tickers))

    return {
        "analyzed": analyzed,
        "eligible": eligible,
        "selected_risk": len(risk),
        "defensive": len(defensive),
        "discarded": discarded,
        "total_assets": len(active_tickers),
    }


def format_universe_stat(value) -> str:
    if value is None:
        return "Dato no disponible en este snapshot."
    return str(value)
