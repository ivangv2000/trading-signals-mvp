"""
V17.7 — Dual Forward Paper Research (interfaz amigable).

Prospective paper tracking: R0 control, R2 challenger, B1 and SPY benchmarks.
NO REAL MONEY. NO BACKTEST.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from paper_research.forward_paper_tracker import ForwardPaperTracker
from paper_research.paper_metrics import compute_drawdown
from ui.components import render_sidebar_navigation, render_summary_card, render_warning_banner
from ui.styles import apply_app_styles

st.set_page_config(
  page_title="Paper Research",
  page_icon="🧪",
  layout="wide",
)

apply_app_styles()
render_sidebar_navigation()

tracker = ForwardPaperTracker(ROOT)
state = tracker._read_state()

st.markdown('<div class="app-shell">', unsafe_allow_html=True)
st.markdown('<div class="app-title">Paper Research</div>', unsafe_allow_html=True)
st.markdown(
  '<div class="app-subtitle">Seguimiento prospectivo R0 vs R2. '
  "No incluye resultados históricos anteriores al paper tracking.</div>",
  unsafe_allow_html=True,
)

render_warning_banner(
  "Datos prospectivos desde el inicio del paper tracking. "
  "No está aprobado para dinero real. APPROVED_FOR_REAL_MONEY=False"
)

# 1. Estado del tracker
st.markdown('<div class="section-title">1. Estado del tracker</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
  render_summary_card("Última señal", state.get("last_signal_date") or "Sin datos")
with c2:
  render_summary_card("Última actualización", state.get("last_equity_date") or "Sin datos")
with c3:
  render_summary_card("Sesiones forward", str(state.get("n_observations", 0)))
with c4:
  r2_exp = tracker.get_r2_exposure_latest()
  render_summary_card("Exposición R2", f"{r2_exp:.0%}" if r2_exp is not None else "—")

# 2. Acción actual
st.markdown('<div class="section-title">2. Acción necesaria hoy</div>', unsafe_allow_html=True)
if not state.get("closure_validated"):
  action_today = "Importar snapshot semanal (CSV de señales + JSON V17.6.1)"
elif not state.get("last_equity_date"):
  action_today = "Actualizar precios diarios para iniciar el seguimiento forward"
else:
  action_today = "Nada urgente — espera la próxima señal o actualiza precios al cierre"
st.info(action_today)

# 3. R0 vs R2
st.markdown('<div class="section-title">3. R0 frente a R2</div>', unsafe_allow_html=True)
equity_df = tracker.get_daily_equity()
r0_eq = equity_df[equity_df["strategy"] == "R0_BASE_S5"] if not equity_df.empty else pd.DataFrame()
r2_eq = equity_df[equity_df["strategy"] == "R2_SPY_TREND"] if not equity_df.empty else pd.DataFrame()
if not r0_eq.empty and not r2_eq.empty:
  rc1, rc2 = st.columns(2)
  with rc1:
    render_summary_card(
      "R0 rentabilidad acumulada",
      f"{float(r0_eq.sort_values('date').iloc[-1]['cumulative_return']):.2f}%",
    )
  with rc2:
    render_summary_card(
      "R2 rentabilidad acumulada",
      f"{float(r2_eq.sort_values('date').iloc[-1]['cumulative_return']):.2f}%",
    )
else:
  st.caption("Sin datos forward todavía. Importa señal y actualiza precios.")

# 4. Curva forward
st.markdown('<div class="section-title">4. Curva forward</div>', unsafe_allow_html=True)
if not equity_df.empty:
  fig = go.Figure()
  palette = {
    "R0_BASE_S5": "#22c55e",
    "R2_SPY_TREND": "#3b82f6",
    "B1_EQUAL_WEIGHT_CURRENT_ASOF": "#f59e0b",
    "SPY": "#a78bfa",
  }
  for strategy in equity_df["strategy"].astype(str).unique():
    sub = equity_df[equity_df["strategy"] == strategy].sort_values("date")
    fig.add_trace(go.Scatter(
      x=sub["date"], y=sub["equity"], name=strategy,
      line=dict(color=palette.get(strategy, "#e5e7eb"), width=2),
    ))
  fig.update_layout(
    height=380, plot_bgcolor="#151f32", paper_bgcolor="#0b1220",
    font=dict(color="#f1f5f9"), legend=dict(orientation="h", y=1.08),
    margin=dict(l=20, r=20, t=30, b=20),
  )
  st.plotly_chart(fig, use_container_width=True)
else:
  st.caption("La curva aparecerá tras la primera valoración diaria.")

with st.sidebar:
  st.header("Acciones paper")
  signals_file = st.file_uploader("CSV señales", type=["csv"], key="paper_signals")
  config_file = st.file_uploader("JSON cierre", type=["json"], key="paper_config")
  if st.button("Importar snapshot", type="primary", use_container_width=True):
    if not signals_file or not config_file:
      st.error("Sube CSV y JSON.")
    else:
      tmp = ROOT / "paper_research" / "tmp"
      tmp.mkdir(parents=True, exist_ok=True)
      sig_path = tmp / "current_signals.csv"
      cfg_path = tmp / "selected_config.json"
      sig_path.write_bytes(signals_file.getvalue())
      cfg_path.write_bytes(config_file.getvalue())
      try:
        result = tracker.import_signal_snapshot(sig_path, cfg_path)
        st.success(result.get("message", "OK"))
      except Exception as exc:
        st.error(str(exc))
      st.rerun()
  if st.button("Actualizar precios", use_container_width=True):
    try:
      st.info(tracker.update_daily_prices().get("message", "OK"))
    except Exception as exc:
      st.error(str(exc))
    st.rerun()
  if st.button("Exportar resultados", use_container_width=True):
    zip_path = tracker.export_results()
    st.success(str(zip_path))

with st.expander("5. Posiciones actuales", expanded=False):
  positions = tracker.get_positions()
  if positions.empty:
    st.caption("Sin posiciones.")
  else:
    st.dataframe(positions, use_container_width=True)

with st.expander("6. Últimas ejecuciones", expanded=False):
  executions = tracker._read_csv(tracker.EXECUTIONS)
  if executions.empty:
    st.caption("Sin ejecuciones.")
  else:
    st.dataframe(
      executions.sort_values("execution_date", ascending=False).head(30),
      use_container_width=True,
    )

with st.expander("7. Métricas forward", expanded=False):
  metrics_df = tracker.get_forward_metrics()
  if metrics_df.empty:
    st.caption("Métricas tras primera valoración.")
  else:
    st.dataframe(metrics_df, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)
