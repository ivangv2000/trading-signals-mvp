"""Reusable Streamlit components for research status (read-only)."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

DISCLAIMER = (
    "Seguimiento experimental en paper trading. No aprobado para operar con dinero real."
)

BADGE_STYLES = {
    "PUBLIC_MODEL": "background:#1e3a5f;color:#93c5fd;",
    "RESEARCH_ONLY": "background:#3b2f1a;color:#fbbf24;",
    "PENDING_EXECUTION": "background:#3f3f46;color:#e4e4e7;",
    "EXECUTED": "background:#1f2937;color:#cbd5e1;",
    "WAITING_FOR_SIGNAL": "background:#312e81;color:#c7d2fe;",
    "DATA_UNAVAILABLE": "background:#3f3f46;color:#a1a1aa;",
}


def render_research_disclaimer() -> None:
    st.markdown(
        f'<div class="d2-waiting-card" style="margin:0.75rem 0;"><strong>{DISCLAIMER}</strong></div>',
        unsafe_allow_html=True,
    )


def render_status_badge(kind: str, text: str) -> None:
    style = BADGE_STYLES.get(kind, BADGE_STYLES["DATA_UNAVAILABLE"])
    st.markdown(
        f'<span class="d2-model-status" style="{style}">{text}</span>',
        unsafe_allow_html=True,
    )


def _fmt_weight(w: float | None) -> str:
    if w is None or (isinstance(w, float) and pd.isna(w)):
        return "—"
    return f"{float(w) * 100:.0f}%"


def render_strategy_signal_card(card: dict[str, Any]) -> None:
    holdings = card.get("holdings") or []
    assets = ", ".join(
        f"{h['ticker']} {_fmt_weight(h.get('target_weight'))}" for h in holdings
    ) or "Sin señal"
    kind = card.get("label_kind", "RESEARCH_ONLY")
    badge = "MODELO PÚBLICO" if kind == "PUBLIC_MODEL" else "INVESTIGACIÓN · PAPER TRADING"
    style = BADGE_STYLES.get(kind, BADGE_STYLES["RESEARCH_ONLY"])
    st.markdown(
        f"""
        <div class="d2-model-card" style="min-height:11rem;">
            <div class="d2-model-label">{badge}</div>
            <div class="d2-model-name">{card.get('model', '—')}</div>
            <div style="font-size:0.85rem;margin-top:0.4rem;color:#cbd5e1;">
                Fecha: <strong>{card.get('signal_date') or '—'}</strong><br/>
                Activos: {assets}<br/>
                Ejecución: {card.get('execution_status_label') or '—'}
            </div>
            <div class="d2-model-status" style="{style};margin-top:0.55rem;">{card.get('status_tag') or card.get('classification') or '—'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_execution_status(label: str, detail: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="d2-waiting-card">
            <div class="d2-waiting-title">{label}</div>
            <p>{detail or DISCLAIMER}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_comparison_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    if not rows:
        st.info("Sin datos de comparación para esta fecha.")
        return
    frame = pd.DataFrame(rows)
    rename = {src: label for src, label in columns if src in frame.columns}
    show = frame[[c for c, _ in columns if c in frame.columns]].rename(columns=rename)
    for col in show.columns:
        if "weight" in col.lower() or "peso" in col.lower():
            show[col] = show[col].map(lambda x: _fmt_weight(x) if pd.notna(x) and x != "" else "—")
        elif "rank" in col.lower() or "score" in col.lower() or "prediction" in col.lower() or "predicción" in col.lower():
            show[col] = show[col].map(
                lambda x: "—" if x is None or (isinstance(x, float) and pd.isna(x)) or x == ""
                else (f"{float(x):.3f}" if isinstance(x, (int, float)) or str(x).replace(".", "", 1).isdigit() else str(x))
            )
    st.dataframe(show, use_container_width=True, hide_index=True)


def render_system_health_card(infra: dict[str, Any]) -> None:
    norgate = infra.get("norgate") or {}
    st.markdown(
        f"""
        <div class="v14-card">
            <div style="font-weight:700;margin-bottom:0.4rem;">Salud técnica</div>
            <div style="font-size:0.9rem;color:#cbd5e1;line-height:1.55;">
                Repositorio: <strong>{infra.get('repository_status') or '—'}</strong><br/>
                Suite: <strong>{infra.get('suite_passed') if infra.get('suite_passed') is not None else '—'}</strong>
                / <strong>{infra.get('suite_tests') if infra.get('suite_tests') is not None else '—'}</strong>
                · Fallos: <strong>{infra.get('suite_failed') if infra.get('suite_failed') is not None else '—'}</strong><br/>
                Trial count: <strong>{infra.get('trial_count')}</strong><br/>
                Modelo público: <strong>{infra.get('public_model')}</strong><br/>
                APPROVED_FOR_REAL_MONEY: <strong>False</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="v14-card" style="margin-top:0.75rem;">
            <div style="font-weight:700;margin-bottom:0.4rem;">Datos point-in-time (Norgate)</div>
            <div style="font-size:0.9rem;color:#cbd5e1;line-height:1.55;">
                Conexión: <strong>{"Sí" if norgate.get("connected") else "No / no leída"}</strong><br/>
                Componentes históricos: <strong>{"Sí" if norgate.get("historical_constituents") else "No"}</strong><br/>
                Retiradas (delisted): <strong>{"Sí" if norgate.get("delisted_securities") else "No"}</strong><br/>
                S&amp;P 500 PIT: <strong>{"Listo" if norgate.get("sp500_pit_ready") else "No"}</strong><br/>
                TOP100/TOP250 PIT: <strong>{"Disponible" if norgate.get("nested_top_available") else "No disponible"}</strong><br/>
                Histórico completo 1998–2026: <strong>{"Disponible" if norgate.get("full_history_available") else "No disponible (trial)"}</strong><br/>
                Ventana trial: <strong>{norgate.get("first_available_date") or "—"} → {norgate.get("last_available_date") or "—"}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
