"""
¿Qué hago hoy? — Panel principal moderno para usuarios sin conocimientos de trading.

Solo lectura de señales existentes. No ejecuta órdenes. No modifica algoritmos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ui.action_dashboard import (
  build_human_explanations,
  build_order_plan,
  derive_today_decision,
  fetch_latest_prices_eur,
  load_latest_signals,
  strategy_status_cards,
)
from ui.components import (
  render_action_card,
  render_app_header,
  render_human_explanation,
  render_no_data_state,
  render_order_section,
  render_sidebar_navigation,
  render_signal_card,
  render_summary_row,
  render_warning_banner,
)
from ui.dashboard_tests import run_friendly_dashboard_tests
from ui.styles import apply_app_styles

st.set_page_config(
  page_title="¿Qué hago hoy?",
  page_icon="📋",
  layout="wide",
  initial_sidebar_state="expanded",
)

apply_app_styles()
render_sidebar_navigation()

signals_df, source_label = load_latest_signals(ROOT)
today = derive_today_decision(signals_df)

render_app_header(
  signal_date=today.signal_date,
  data_status=today.data_status,
  next_rebalance=today.next_rebalance,
)

st.markdown('<div class="app-shell">', unsafe_allow_html=True)

if not today.has_data:
  render_no_data_state()
else:
  if today.data_status == "ANTIGUA":
    render_warning_banner(
      "No operes con esta señal hasta generar un archivo actualizado."
    )
  elif today.data_status == "REVISAR FECHA":
    render_warning_banner(
      "La señal tiene más de una semana. Comprueba si existe una versión más reciente."
    )

  render_signal_card(today)
  render_action_card(today)

  capital_default = 100.0
  render_summary_row(
    strategy=today.strategy_name,
    risk=today.risk_level,
    next_rebalance=today.next_rebalance or "—",
    capital=f"{capital_default:.0f} €",
  )

  st.markdown('<div class="section-title">Plan de órdenes</div>', unsafe_allow_html=True)
  st.caption(f"Fuente: {source_label} · Simulación en euros · Sin envío a bróker.")

  c1, c2, c3 = st.columns(3)
  capital_eur = c1.number_input(
    "Capital disponible (€)",
    min_value=1.0,
    value=100.0,
    step=10.0,
    key="capital_eur",
  )
  fractional = c2.radio(
    "Acciones fraccionadas",
    options=["Sí", "No"],
    horizontal=True,
    index=0,
    key="fractional_mode",
  )
  commission_eur = c3.number_input(
    "Comisión por operación (€)",
    min_value=0.0,
    value=0.0,
    step=0.5,
    key="commission_eur",
  )

  c4, c5 = st.columns(2)
  fx_cost_pct = c4.number_input(
    "Coste de cambio de divisa (%)",
    min_value=0.0,
    value=0.0,
    step=0.1,
    key="fx_cost_pct",
  )
  max_cost_pct = c5.number_input(
    "Coste máximo aceptable (%)",
    min_value=0.1,
    value=2.0,
    step=0.5,
    key="max_cost_pct",
  )

  show_avoid = st.checkbox("Mostrar también activos descartados", value=False)

  prices_eur = {}
  if signals_df is not None and not signals_df.empty:
    tickers = signals_df["ticker"].astype(str).tolist()
    with st.spinner("Obteniendo precios orientativos..."):
      prices_eur = fetch_latest_prices_eur(tickers)

  plan = build_order_plan(
    signals_df,
    capital_eur=capital_eur,
    fractional=fractional == "Sí",
    commission_eur=commission_eur,
    fx_cost_pct=fx_cost_pct,
    max_cost_pct=max_cost_pct,
    prices_eur=prices_eur,
  )
  render_order_section(plan, show_avoid=show_avoid)

  explanations = build_human_explanations(signals_df)
  render_human_explanation(explanations)

with st.expander("Ver métricas avanzadas (investigación histórica)", expanded=False):
  st.warning(
    "Estas cifras son de backtests pasados. No uses CAGR, Sharpe, DSR ni drawdown "
    "para decidir qué hacer hoy."
  )
  v14_path = ROOT / "config" / "approved_v14_strategy.json"
  if v14_path.exists():
    v14 = json.loads(v14_path.read_text(encoding="utf-8"))
    bt = v14.get("backtest_summary", {})
    st.markdown("**V14 R1 Return Engine (histórico)**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CAGR", f"{bt.get('CAGR', '—')}%")
    m2.metric("Sharpe", bt.get("sharpe", "—"))
    m3.metric("Sortino", bt.get("sortino", "—"))
    m4.metric("Max drawdown", f"{bt.get('max_drawdown', '—')}%")
  st.markdown(
    "Curvas históricas, comparaciones V14/V6 y tablas técnicas están en "
    "**Trading Signals Lab** (app principal)."
  )

with st.expander("Estrategias del sistema", expanded=False):
  for card in strategy_status_cards():
    st.markdown(
      f"**{card['name']}** — {card['status']} · Dinero real: **NO**"
    )

st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
  st.markdown("---")
  st.subheader("Cargar señal")
  uploaded = st.file_uploader(
    "research_v17_6_current_signals.csv",
    type=["csv"],
    key="dashboard_signals_upload",
  )
  if uploaded and st.button("Usar este CSV", use_container_width=True):
    dest = ROOT / "research_v17_6_current_signals.csv"
    dest.write_bytes(uploaded.getvalue())
    st.success(f"Guardado: {dest.name}")
    st.rerun()

  if st.button("Tests interfaz 12/12", use_container_width=True):
    report = run_friendly_dashboard_tests()
    st.dataframe(report, use_container_width=True)

st.markdown(
  '<div class="footer-note">No es asesoramiento financiero. '
  "No conecta con brókers. No envía órdenes reales.</div>",
  unsafe_allow_html=True,
)
