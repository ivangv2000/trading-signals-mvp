"""Estilos globales para la interfaz amigable de trading-signals-mvp."""

from __future__ import annotations

import streamlit as st


def apply_app_styles() -> None:
  """Inyecta CSS: fondo oscuro, tarjetas, colores de señal y layout responsive."""
  st.markdown(
    """
    <style>
      :root {
        --bg: #0b1220;
        --card: #151f32;
        --card-border: #2a3a55;
        --text: #f1f5f9;
        --muted: #94a3b8;
        --buy: #22c55e;
        --sell: #ef4444;
        --hold: #3b82f6;
        --avoid: #f97316;
        --nodata: #9ca3af;
        --max-width: 1200px;
      }

      .stApp {
        background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
      }

      .app-shell {
        max-width: var(--max-width);
        margin: 0 auto;
        padding: 0 0.5rem 2rem;
      }

      .app-header {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1.25rem;
        align-items: flex-start;
      }

      .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text);
        margin: 0;
        line-height: 1.15;
      }

      .app-subtitle {
        color: var(--muted);
        font-size: 1rem;
        margin-top: 0.35rem;
        max-width: 520px;
      }

      .header-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: flex-end;
      }

      .meta-pill {
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 0.55rem 0.85rem;
        min-width: 120px;
      }

      .meta-pill-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
      }

      .meta-pill-value {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--text);
        margin-top: 0.15rem;
      }

      .signal-card {
        background: var(--card);
        border: 2px solid var(--card-border);
        border-radius: 24px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin: 1rem 0 1.25rem;
        box-shadow: 0 12px 40px rgba(0,0,0,0.25);
      }

      .signal-card-label {
        font-size: 0.8rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.5rem;
      }

      .signal-card-icon {
        font-size: 2.5rem;
        margin-bottom: 0.35rem;
      }

      .signal-card-decision {
        font-size: clamp(2.8rem, 8vw, 4.5rem);
        font-weight: 900;
        line-height: 1;
        letter-spacing: 0.04em;
      }

      .signal-card-text {
        font-size: 1.2rem;
        color: var(--text);
        margin-top: 0.85rem;
        line-height: 1.45;
      }

      .action-card, .summary-card, .viability-card, .explain-card, .no-data-card {
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 18px;
        padding: 1.15rem 1.25rem;
        margin-bottom: 1rem;
      }

      .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text);
        margin: 1.5rem 0 0.75rem;
      }

      .card-kicker {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--muted);
        margin-bottom: 0.35rem;
      }

      .card-body {
        color: var(--text);
        font-size: 1.02rem;
        line-height: 1.55;
      }

      .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.85rem;
        margin: 1rem 0 1.25rem;
      }

      .summary-card-title {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
      }

      .summary-card-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text);
        margin-top: 0.35rem;
      }

      .badge {
        display: inline-block;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.06em;
      }

      .badge-ok { background: #14532d; color: #bbf7d0; }
      .badge-warn { background: #78350f; color: #fde68a; }
      .badge-bad { background: #450a0a; color: #fecaca; }
      .badge-muted { background: #1f2937; color: #cbd5e1; }

      .viability-ok {
        border-color: #22c55e;
        background: rgba(34,197,94,0.08);
      }

      .viability-bad {
        border-color: #ef4444;
        background: rgba(239,68,68,0.08);
      }

      .viability-title {
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
      }

      .warn-banner {
        background: #451a03;
        border: 2px solid #f97316;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        color: #ffedd5;
        margin: 0.75rem 0 1rem;
        font-weight: 600;
      }

      .no-data-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 0.75rem;
      }

      .no-data-steps {
        color: var(--text);
        line-height: 1.7;
        margin: 0.5rem 0 0;
        padding-left: 1.2rem;
      }

      .explain-list {
        margin: 0;
        padding-left: 1.1rem;
        color: var(--text);
        line-height: 1.65;
      }

      .order-badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 800;
      }

      .order-buy { background: rgba(34,197,94,0.18); color: #86efac; }
      .order-sell { background: rgba(239,68,68,0.18); color: #fca5a5; }
      .order-hold { background: rgba(59,130,246,0.18); color: #93c5fd; }
      .order-reduce { background: rgba(249,115,22,0.18); color: #fdba74; }
      .order-avoid { background: rgba(156,163,175,0.18); color: #d1d5db; }

      .footer-note {
        color: var(--muted);
        font-size: 0.88rem;
        margin-top: 2rem;
        text-align: center;
      }

      @media (max-width: 768px) {
        .app-header { flex-direction: column; }
        .header-meta { justify-content: flex-start; }
      }

      div[data-testid="stSidebar"] {
        background: #0f172a;
      }
    </style>
    """,
    unsafe_allow_html=True,
  )
