"""Estilos globales para la web pública V14."""

from __future__ import annotations

import streamlit as st

FOOTER_TEXT = (
    "Proyecto experimental de paper trading. "
    "No constituye asesoramiento financiero. "
    "No ejecuta órdenes. "
    "Los resultados pasados no garantizan resultados futuros. "
    "APPROVED_FOR_REAL_MONEY=False."
)

SIGNAL_COLORS = {
    "BUY": "#22c55e",
    "REBALANCE": "#a855f7",
    "HOLD": "#3b82f6",
    "SELL": "#ef4444",
    "NO DATA": "#6b7280",
    "INCREASE": "#22c55e",
    "REDUCE": "#f97316",
    "AVOID": "#6b7280",
}

FRESHNESS_COLORS = {
    "ACTUALIZADA": "#22c55e",
    "REVISAR": "#f97316",
    "ANTIGUA": "#ef4444",
}


def apply_v14_global_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #0b1220; color: #e5e7eb; }
        .block-container {
            max-width: 1220px;
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { display: none; }
        section[data-testid="stSidebar"] { display: none; }
        div[data-testid="stNavigation"] { display: none !important; }
        .v14-hero-title {
            font-size: 2.75rem;
            font-weight: 900;
            letter-spacing: -0.02em;
            color: #f9fafb;
            margin: 0 0 0.35rem 0;
            line-height: 1.15;
        }
        .v14-hero-sub {
            font-size: 1.12rem;
            color: #94a3b8;
            margin-bottom: 1.5rem;
            line-height: 1.55;
            max-width: 920px;
        }
        .v14-paper-banner {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            font-size: 0.82rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #fbbf24;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .v14-card {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }
        .v14-card-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #9ca3af;
            margin-bottom: 0.5rem;
        }
        .v14-status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.75rem;
            margin-bottom: 1.25rem;
        }
        .v14-status-item {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 0.9rem 1rem;
        }
        .v14-status-item label {
            display: block;
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #9ca3af;
            margin-bottom: 0.25rem;
        }
        .v14-status-item span {
            font-size: 1rem;
            font-weight: 700;
            color: #f3f4f6;
        }
        .v14-action-card {
            border-radius: 18px;
            padding: 1.75rem 2rem;
            margin: 1rem 0 1.5rem;
            border: 1px solid #1f2937;
        }
        .v14-action-title {
            font-size: 0.8rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            opacity: 0.85;
            margin-bottom: 0.35rem;
        }
        .v14-action-value {
            font-size: 2.8rem;
            font-weight: 900;
            line-height: 1;
            margin: 0.2rem 0 0.6rem;
        }
        .v14-action-desc {
            font-size: 1rem;
            opacity: 0.92;
        }
        .v14-section-title {
            font-size: 1.1rem;
            font-weight: 800;
            color: #f9fafb;
            margin: 1.5rem 0 0.75rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .v14-asset-card {
            background: #0f172a;
            border: 1px solid #243044;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.65rem;
        }
        .v14-asset-ticker {
            font-size: 1.35rem;
            font-weight: 800;
            color: #f9fafb;
        }
        .v14-asset-signal {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 6px;
            margin: 0.35rem 0;
        }
        .v14-asset-meta {
            font-size: 0.88rem;
            color: #cbd5e1;
            line-height: 1.45;
        }
        .v14-footer {
            margin-top: 2.5rem;
            padding-top: 1rem;
            border-top: 1px solid #1f2937;
            font-size: 0.78rem;
            color: #94a3b8;
            line-height: 1.5;
            text-align: center;
        }
        .v14-metric-card {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 14px;
            padding: 1rem;
            text-align: center;
        }
        .v14-metric-value { font-size: 1.4rem; font-weight: 800; color: #f9fafb; }
        .v14-metric-label { font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; }
        .v14-warning {
            background: #451a1a;
            border: 1px solid #7f1d1d;
            color: #fecaca;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            margin: 0.75rem 0;
            font-size: 0.9rem;
        }
        .v14-flow {
            background: #0f172a;
            border: 1px solid #243044;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            font-family: ui-monospace, monospace;
            font-size: 0.9rem;
            color: #cbd5e1;
            line-height: 1.8;
        }
        .d2-experimental-banner {
            background: #312e81;
            border: 1px solid #4c1d95;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            font-size: 0.82rem;
            letter-spacing: 0.06em;
            color: #ddd6fe;
            text-align: center;
            margin-bottom: 1.25rem;
        }
        .d2-model-card {
            border-radius: 16px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
        }
        .d2-model-public {
            background: #0f2f1f;
            border: 1px solid #166534;
        }
        .d2-model-experimental {
            background: #2e1f47;
            border: 1px solid #6d28d9;
        }
        .d2-model-label {
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #9ca3af;
            margin-bottom: 0.35rem;
        }
        .d2-model-name {
            font-size: 1.15rem;
            font-weight: 800;
            color: #f9fafb;
            margin-bottom: 0.35rem;
        }
        .d2-model-status {
            font-size: 0.85rem;
            font-weight: 700;
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 6px;
        }
        .d2-status-active { background: #14532d; color: #86efac; }
        .d2-status-shadow { background: #4c1d95; color: #ddd6fe; }
        .d2-formula-card {
            background: #111827;
            border: 1px solid #374151;
            border-radius: 14px;
            padding: 1rem 1.25rem;
            margin: 0.75rem 0 1.25rem;
        }
        .d2-formula-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #9ca3af;
            margin-bottom: 0.5rem;
        }
        .d2-formula-line { font-size: 1rem; color: #e5e7eb; line-height: 1.6; }
        .d2-waiting-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }
        .d2-waiting-title {
            font-size: 1.2rem;
            font-weight: 800;
            color: #f9fafb;
            margin-bottom: 0.5rem;
        }
        .d2-sample-guard {
            background: #1f2937;
            border: 1px solid #374151;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            color: #d1d5db;
            margin-bottom: 1rem;
        }
        .d2-disclaimer {
            font-size: 0.82rem;
            color: #f87171;
            text-align: center;
            margin-top: 1rem;
        }
        .site-nav-shell { margin-bottom: 1.75rem; }
        .site-nav-shell + div [data-testid="stPageLink-NavLink"] {
            text-decoration: none !important;
            font-weight: 600;
            font-size: 0.82rem;
            color: #94a3b8 !important;
        }
        .site-nav-shell + div [data-testid="stPageLink-NavLink"]:hover { color: #e2e8f0 !important; }
        div[data-testid="stRadio"] > label { font-size: 0.88rem; }
        div[data-testid="stRadio"] label p { font-size: 0.88rem !important; line-height: 1.35; }
        .site-nav-card {
            min-height: 80px;
            border-radius: 14px;
            padding: 0.85rem 1rem 0.65rem;
            border: 1px solid #334155;
            background: #111827;
            transition: border-color 0.15s ease, background 0.15s ease;
        }
        .site-nav-card:hover { border-color: #64748b; background: #1a2332; }
        .site-nav-active { border-color: #60a5fa !important; box-shadow: 0 0 0 1px #3b82f6 inset; }
        .site-nav-v14.site-nav-active { border-color: #22c55e !important; box-shadow: 0 0 0 1px #22c55e inset; }
        .site-nav-d2.site-nav-active { border-color: #a855f7 !important; box-shadow: 0 0 0 1px #a855f7 inset; }
        .site-nav-docs.site-nav-active { border-color: #60a5fa !important; }
        .site-nav-icon { font-size: 1.35rem; margin-bottom: 0.2rem; }
        .site-nav-title {
            font-size: 1.05rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            color: #f9fafb;
            line-height: 1.2;
        }
        .site-nav-subtitle {
            font-size: 0.8rem;
            color: #94a3b8;
            line-height: 1.35;
            margin-top: 0.2rem;
        }
        @media (max-width: 768px) {
            .site-nav-title { font-size: 0.95rem; }
            .site-nav-subtitle { font-size: 0.75rem; }
            .site-nav-card { min-height: 72px; }
            .v14-hero-title { font-size: 2rem; }
        }
        .doc-index-title {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #9ca3af;
            margin-bottom: 0.75rem;
            font-weight: 700;
        }
        .doc-chapter-title {
            font-size: 1.45rem;
            font-weight: 800;
            color: #f9fafb;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #1f2937;
        }
        .doc-flow-track {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            align-items: stretch;
            margin: 1rem 0 1.5rem;
        }
        .doc-flow-step {
            flex: 1 1 140px;
            background: #0f172a;
            border: 1px solid #243044;
            border-radius: 12px;
            padding: 0.85rem;
            min-width: 120px;
        }
        .doc-flow-step-title { font-weight: 700; color: #e2e8f0; font-size: 0.9rem; }
        .doc-flow-step-desc { font-size: 0.78rem; color: #94a3b8; margin-top: 0.35rem; line-height: 1.4; }
        .doc-flow-arrow { color: #64748b; align-self: center; font-size: 1.2rem; padding: 0 0.15rem; }
        .doc-timeline { display: flex; flex-direction: column; gap: 0.65rem; margin-top: 1rem; }
        .doc-timeline-item {
            display: grid;
            grid-template-columns: 90px 1fr auto;
            gap: 0.75rem;
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            align-items: center;
        }
        @media (max-width: 640px) {
            .doc-timeline-item { grid-template-columns: 1fr; }
        }
        .doc-timeline-version { font-weight: 800; color: #f3f4f6; }
        .doc-timeline-desc { color: #cbd5e1; font-size: 0.92rem; }
        .doc-timeline-status { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; }
        .doc-link-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.75rem; margin: 1rem 0; }
        .doc-mini-card {
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }
        .doc-mini-card-title { font-weight: 700; color: #e2e8f0; font-size: 0.95rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_footer_disclaimer() -> None:
    st.markdown(f'<div class="v14-footer">{FOOTER_TEXT}</div>', unsafe_allow_html=True)


def render_d2_experimental_banner() -> None:
    st.markdown(
        '<div class="d2-experimental-banner">'
        "EXPERIMENTO DE INVESTIGACIÓN — NO ES UNA SEÑAL OFICIAL NI RECOMENDACIÓN DE COMPRA"
        "</div>",
        unsafe_allow_html=True,
    )


def render_d2_integrity_alerts(view_model: dict) -> None:
    if not view_model.get("contracts_ok", True):
        st.markdown(
            '<div class="v14-warning">El contrato del experimento no coincide con la configuración congelada.</div>',
            unsafe_allow_html=True,
        )
    if view_model.get("data_revision_detected"):
        st.markdown(
            '<div class="v14-warning">Se detectó una revisión de datos históricos. '
            "El seguimiento está detenido hasta revisar la integridad.</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="v14-warning" style="background:#451a1a;border-color:#7f1d1d;">'
        "Sin aprobación para operar con dinero real.</div>",
        unsafe_allow_html=True,
    )
