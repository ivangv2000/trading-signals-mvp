"""Componentes UI reutilizables para la web pública V14."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from services.user_portfolio_action_service import (
    FRACTIONAL_WARNING,
    USER_GLOBAL_ACTION_TEXT,
    format_universe_stat,
)
from ui.v14_styles import FRESHNESS_COLORS, SIGNAL_COLORS

SIGNAL_ORDER = {"BUY": 0, "SELL": 1, "REDUCE": 2, "HOLD": 3, "AVOID": 4, "INCREASE": 1}


def parse_signal_date(value: str | None) -> date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(str(value)[:10], fmt).date()
        except ValueError:
            continue
    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


def freshness_status(signal_date: str | None) -> tuple[str, str, int | None]:
    parsed = parse_signal_date(signal_date)
    if parsed is None:
        return "NO DATA", FRESHNESS_COLORS.get("ANTIGUA", "#ef4444"), None
    days = (date.today() - parsed).days
    if days <= 7:
        return "ACTUALIZADA", FRESHNESS_COLORS["ACTUALIZADA"], days
    if days <= 14:
        return "REVISAR", FRESHNESS_COLORS["REVISAR"], days
    return "ANTIGUA", FRESHNESS_COLORS["ANTIGUA"], days


def render_paper_banner() -> None:
    import streamlit as st

    st.markdown(
        """
        <div class="v14-paper-banner">
            PAPER TRADING · NO EJECUTA ÓRDENES · APPROVED_FOR_REAL_MONEY=False
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_header(summary: dict | None) -> None:
    import streamlit as st

    signal_date = (summary or {}).get("signal_date", "—")
    status, color, _days = freshness_status(signal_date)
    strategy = (summary or {}).get("strategy", "V14 R1 Return Engine")
    next_reb = (summary or {}).get("next_rebalance", "Viernes después del cierre")

    st.markdown(
        f"""
        <div class="v14-status-grid">
            <div class="v14-status-item">
                <label>Estado</label>
                <span style="color:{color}">{status}</span>
            </div>
            <div class="v14-status-item">
                <label>Fecha</label>
                <span>{signal_date}</span>
            </div>
            <div class="v14-status-item">
                <label>Próxima revisión</label>
                <span>{next_reb}</span>
            </div>
            <div class="v14-status-item">
                <label>Estrategia</label>
                <span>{strategy}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if status == "ANTIGUA":
        st.markdown(
            '<div class="v14-warning">Esta señal es antigua. Actualízala antes de tomar decisiones.</div>',
            unsafe_allow_html=True,
        )


def render_user_global_action(action: str) -> None:
    import streamlit as st

    action = (action or "NO DATA").upper()
    color = SIGNAL_COLORS.get(action, SIGNAL_COLORS["NO DATA"])
    text = USER_GLOBAL_ACTION_TEXT.get(action, USER_GLOBAL_ACTION_TEXT["NO DATA"])
    st.markdown(
        f"""
        <div class="v14-action-card" style="background:{color}22;border-color:{color}55;">
            <div class="v14-action-title">Acción para ti</div>
            <div class="v14-action-value" style="color:{color}">{action}</div>
            <div class="v14-action-desc">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _weight_pct(w: float) -> str:
    return f"{w * 100:.0f}%"


def _diff_label(diff: float) -> str:
    if diff > 0.01:
        return f"Debes comprar: {diff:.0f} €"
    if diff < -0.01:
        return f"Debes vender o reducir: {abs(diff):.0f} €"
    return "Sin diferencia relevante"


def render_user_asset_card(row: pd.Series) -> None:
    import streamlit as st

    ticker = row.get("ticker", "")
    user_action = str(row.get("user_action", "")).upper()
    model_signal = str(row.get("model_signal", "")).upper()
    color = SIGNAL_COLORS.get(user_action, "#9ca3af")
    user_w = float(row.get("user_current_weight", 0) or 0)
    target_w = float(row.get("model_target_weight", 0) or 0)
    user_amt = float(row.get("user_amount_eur", 0) or 0)
    target_amt = float(row.get("target_amount_eur", 0) or 0)
    diff_amt = float(row.get("diff_amount_eur", 0) or 0)
    reason = row.get("reason", "")

    st.markdown(
        f"""
        <div class="v14-asset-card">
            <div class="v14-asset-ticker">{ticker}</div>
            <div class="v14-asset-signal" style="background:{color}33;color:{color}">
                ACCIÓN PARA TI: {user_action}
            </div>
            <div class="v14-asset-meta">
                Señal del modelo: <strong>{model_signal}</strong><br>
                Peso que tienes: {_weight_pct(user_w)}<br>
                Peso objetivo: {_weight_pct(target_w)}<br>
                Tienes: {user_amt:.0f} €<br>
                Objetivo: {target_amt:.0f} €<br>
                {_diff_label(diff_amt)}<br>
                <em>Motivo:</em> {reason}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_user_signal_sections(actions_df: pd.DataFrame) -> None:
    import streamlit as st

    if actions_df is None or actions_df.empty:
        for title, empty in [
            ("Nuevas compras", "No hay nuevas compras."),
            ("Mantener", "No hay posiciones para mantener."),
            ("Vender o reducir", "No hay posiciones para vender."),
        ]:
            st.markdown(f'<div class="v14-section-title">{title}</div>', unsafe_allow_html=True)
            st.info(empty)
        return

    visible = actions_df[actions_df["user_action"].astype(str).str.upper() != "AVOID"]
    buys = visible[visible["user_action"].isin(["BUY"])]
    holds = visible[visible["user_action"].isin(["HOLD"])]
    sells = visible[visible["user_action"].isin(["SELL", "REDUCE"])]

    st.markdown('<div class="v14-section-title">Nuevas compras</div>', unsafe_allow_html=True)
    if buys.empty:
        st.info("No hay nuevas compras.")
    else:
        for _, row in buys.iterrows():
            render_user_asset_card(row)

    st.markdown('<div class="v14-section-title">Mantener</div>', unsafe_allow_html=True)
    if holds.empty:
        st.info("No hay posiciones para mantener.")
    else:
        for _, row in holds.iterrows():
            render_user_asset_card(row)

    st.markdown('<div class="v14-section-title">Vender o reducir</div>', unsafe_allow_html=True)
    if sells.empty:
        st.info("No hay posiciones para vender.")
    else:
        for _, row in sells.iterrows():
            render_user_asset_card(row)


def render_defensive_section(actions_df: pd.DataFrame) -> None:
    import streamlit as st

    st.markdown('<div class="v14-section-title">Parte defensiva</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="v14-card">
            <p style="color:#cbd5e1;line-height:1.55;margin:0;">
                <strong>SHY</strong> es un ETF de bonos del Tesoro estadounidense de corta duración.
                En esta cartera actúa como parte defensiva, no como una de las tres posiciones
                principales de riesgo.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    defensive = actions_df[actions_df["is_defensive"] & (actions_df["model_target_weight"] > 0.001)]
    if defensive.empty:
        st.info("No hay posición defensiva activa en este snapshot.")
        return
    for _, row in defensive.iterrows():
        render_user_asset_card(row)


def render_universe_section(universe_stats: dict | None) -> None:
    import streamlit as st

    stats = universe_stats or {}
    st.markdown('<div class="v14-section-title">¿Cómo se llega a estas posiciones?</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="color:#94a3b8;margin-bottom:0.75rem;">
        El algoritmo analiza un universo amplio, ordena los activos según sus reglas
        y selecciona únicamente los que forman la cartera objetivo.
        </p>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    labels = [
        ("Activos analizados", "analyzed"),
        ("Activos elegibles", "eligible"),
        ("Posiciones seleccionadas", "selected_risk"),
        ("Activos defensivos", "defensive"),
        ("Activos descartados", "discarded"),
    ]
    for col, (label, key) in zip(cols, labels):
        with col:
            st.markdown(
                f"""
                <div class="v14-metric-card">
                    <div class="v14-metric-value">{format_universe_stat(stats.get(key))}</div>
                    <div class="v14-metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def compute_capital_summary(actions_df: pd.DataFrame, capital: float, position_counts: dict) -> dict:
    if actions_df is None or actions_df.empty:
        return {
            "capital_total": capital,
            "allocated": 0.0,
            "cash_or_defensive": capital,
            "risk_positions": 0,
            "defensive_positions": 0,
            "total_assets": 0,
            "avg_per_position": 0.0,
        }

    active = actions_df[actions_df["model_target_weight"].fillna(0) > 0.001].copy()
    allocated = float(active["target_amount_eur"].sum())
    defensive_amt = float(active[active["is_defensive"]]["target_amount_eur"].sum())
    risk_count = position_counts.get("risk_positions", 0)
    avg = allocated / max(risk_count, 1) if risk_count else 0.0

    return {
        "capital_total": capital,
        "allocated": allocated,
        "cash_or_defensive": defensive_amt if defensive_amt > 0 else max(0.0, capital - allocated),
        "risk_positions": position_counts.get("risk_positions", 0),
        "defensive_positions": position_counts.get("defensive_positions", 0),
        "total_assets": position_counts.get("total_assets", 0),
        "avg_per_position": avg,
    }


def render_capital_summary(summary: dict) -> None:
    import streamlit as st

    cols = st.columns(5)
    labels = [
        ("Capital total", f"{summary['capital_total']:.0f} €"),
        ("Importe asignado", f"{summary['allocated']:.0f} €"),
        ("Efectivo / defensivo", f"{summary['cash_or_defensive']:.0f} €"),
        (
            "Posiciones",
            f"{summary['risk_positions']} riesgo · {summary['defensive_positions']} defensiva · "
            f"{summary['total_assets']} total",
        ),
        ("Importe medio", f"{summary['avg_per_position']:.0f} €"),
    ]
    for col, (lbl, val) in zip(cols, labels):
        with col:
            st.markdown(
                f"""
                <div class="v14-metric-card">
                    <div class="v14-metric-value">{val}</div>
                    <div class="v14-metric-label">{lbl}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_fractional_warning() -> None:
    import streamlit as st

    st.caption(FRACTIONAL_WARNING)


def prepare_actions_table(actions_df: pd.DataFrame, show_avoid: bool) -> pd.DataFrame:
    if actions_df is None or actions_df.empty:
        return pd.DataFrame()

    df = actions_df.copy()
    if not show_avoid:
        df = df[df["user_action"].astype(str).str.upper() != "AVOID"]

    df["_order"] = df["user_action"].map(lambda s: SIGNAL_ORDER.get(str(s).upper(), 99))
    df = df.sort_values(["_order", "ticker"]).drop(columns=["_order"])

    display = df.rename(
        columns={
            "ticker": "Ticker",
            "user_action": "Acción para ti",
            "model_signal": "Señal del modelo",
            "user_current_weight": "Peso que tienes",
            "model_target_weight": "Peso objetivo",
            "user_amount_eur": "Importe que tienes (€)",
            "target_amount_eur": "Importe objetivo (€)",
            "diff_amount_eur": "Diferencia (€)",
            "reason": "Motivo",
        }
    )
    for col in ("Peso que tienes", "Peso objetivo"):
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{float(x) * 100:.1f}%")
    return display[
        [
            "Ticker",
            "Acción para ti",
            "Señal del modelo",
            "Peso que tienes",
            "Peso objetivo",
            "Importe que tienes (€)",
            "Importe objetivo (€)",
            "Diferencia (€)",
            "Motivo",
        ]
    ]
