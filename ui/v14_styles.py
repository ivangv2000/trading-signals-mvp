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
            max-width: 1250px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] { display: none; }
        section[data-testid="stSidebar"] { display: none; }
        header[data-testid="stHeader"] { background: transparent; }
        .v14-hero-title {
            font-size: 2.6rem;
            font-weight: 900;
            letter-spacing: -0.02em;
            color: #f9fafb;
            margin: 0 0 0.25rem 0;
        }
        .v14-hero-sub {
            font-size: 1.05rem;
            color: #94a3b8;
            margin-bottom: 1.2rem;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_footer_disclaimer() -> None:
    st.markdown(f'<div class="v14-footer">{FOOTER_TEXT}</div>', unsafe_allow_html=True)
