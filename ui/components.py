"""Componentes reutilizables de la interfaz amigable."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from ui.action_dashboard import (
  ACTION_EXACT_TEXT,
  DECISION_COLORS,
  DECISION_ICONS,
  DECISION_MESSAGES,
  OrderPlanResult,
  TodayDecision,
  filter_order_rows,
)


def _esc(text: str) -> str:
  return html.escape(str(text))


def render_signal_card(decision: TodayDecision) -> None:
  color = DECISION_COLORS.get(decision.decision, DECISION_COLORS["NO DATA"])
  icon = DECISION_ICONS.get(decision.decision, "⚠️")
  headline = DECISION_MESSAGES.get(decision.decision, decision.headline)
  st.markdown(
    f"""
    <div class="signal-card" style="border-color:{color}; background:{color}14;">
      <div class="signal-card-label">Decisión de hoy</div>
      <div class="signal-card-icon">{icon}</div>
      <div class="signal-card-decision" style="color:{color};">{_esc(decision.decision)}</div>
      <div class="signal-card-text">{_esc(headline)}</div>
    </div>
    """,
    unsafe_allow_html=True,
  )


def render_status_badge(status: str) -> str:
  mapping = {
    "ACTUALIZADA": ("badge-ok", "ACTUALIZADA"),
    "REVISAR FECHA": ("badge-warn", "REVISAR FECHA"),
    "ANTIGUA": ("badge-bad", "ANTIGUA"),
    "SIN DATOS": ("badge-muted", "SIN DATOS"),
  }
  css, label = mapping.get(status, ("badge-muted", status))
  return f'<span class="badge {css}">{_esc(label)}</span>'


def render_summary_card(title: str, value: str) -> None:
  st.markdown(
    f"""
    <div class="summary-card">
      <div class="summary-card-title">{_esc(title)}</div>
      <div class="summary-card-value">{_esc(value)}</div>
    </div>
    """,
    unsafe_allow_html=True,
  )


def render_action_card(decision: TodayDecision) -> None:
  text = ACTION_EXACT_TEXT.get(decision.decision, decision.what_to_do)
  st.markdown(
    f"""
    <div class="action-card">
      <div class="card-kicker">Qué tienes que hacer</div>
      <div class="card-body">{_esc(text)}</div>
    </div>
    """,
    unsafe_allow_html=True,
  )


def render_viability_card(plan: OrderPlanResult) -> None:
  css = "viability-ok" if plan.executable else "viability-bad"
  title = "PLAN EJECUTABLE" if plan.executable else "NO EJECUTAR"
  color = "#22c55e" if plan.executable else "#ef4444"
  st.markdown(
    f"""
    <div class="viability-card {css}">
      <div class="viability-title" style="color:{color};">{title}</div>
      <div class="card-body">
        Capital total: <strong>{plan.capital_total:.2f} €</strong><br>
        Importe invertido: <strong>{plan.amount_invested:.2f} €</strong><br>
        Efectivo restante: <strong>{plan.cash_remaining:.2f} €</strong><br>
        Costes estimados: <strong>{plan.total_estimated_cost:.2f} €</strong>
        ({plan.cost_percentage_of_capital:.2f}%)<br>
        Número de operaciones: <strong>{plan.n_operations}</strong>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
  )


def _action_badge(action: str) -> str:
  css = {
    "BUY": "order-buy",
    "SELL": "order-sell",
    "HOLD": "order-hold",
    "REDUCE": "order-reduce",
    "AVOID": "order-avoid",
  }.get(str(action).upper(), "order-avoid")
  return f'<span class="order-badge {css}">{_esc(action)}</span>'


def render_order_table(df: pd.DataFrame) -> None:
  if df.empty:
    st.info("No hay operaciones que mostrar con los filtros actuales.")
    return

  headers = list(df.columns)
  header_html = "".join(f"<th>{_esc(h)}</th>" for h in headers)
  rows_html = []
  for _, row in df.iterrows():
    cells = []
    for col in headers:
      val = row[col]
      if col == "Acción":
        cells.append(f"<td>{_action_badge(val)}</td>")
      else:
        cells.append(f"<td>{_esc(val)}</td>")
    rows_html.append("<tr>" + "".join(cells) + "</tr>")

  st.markdown(
    f"""
    <div style="overflow-x:auto;">
      <table style="width:100%; border-collapse:collapse; font-size:0.92rem;">
        <thead>
          <tr style="border-bottom:1px solid #334155; color:#94a3b8; text-align:left;">
            {header_html}
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    """,
    unsafe_allow_html=True,
  )


def render_human_explanation(lines: list[str]) -> None:
  if not lines:
    return
  items = "".join(f"<li>{_esc(line)}</li>" for line in lines)
  st.markdown(
    f"""
    <div class="explain-card">
      <div class="card-kicker">¿Por qué aparece esta señal?</div>
      <ul class="explain-list">{items}</ul>
    </div>
    """,
    unsafe_allow_html=True,
  )


def render_no_data_state() -> None:
  st.markdown(
    """
    <div class="no-data-card">
      <div class="no-data-title">SIN SEÑALES CARGADAS</div>
      <div class="card-body">Sigue estos pasos para ver la decisión de hoy:</div>
      <ol class="no-data-steps">
        <li>Localiza <strong>research_v17_6_current_signals.csv</strong></li>
        <li>Pulsa <strong>Upload</strong> en el panel lateral</li>
        <li>Selecciona el archivo</li>
        <li>La pantalla se actualizará automáticamente</li>
      </ol>
      <p class="card-body" style="margin-top:1rem;color:#94a3b8;">
        El archivo <strong>V17_6_1_CLOSURE.zip</strong> no contiene las señales operativas.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
  )


def render_warning_banner(message: str) -> None:
  st.markdown(f'<div class="warn-banner">{_esc(message)}</div>', unsafe_allow_html=True)


def render_app_header(
  signal_date: str | None,
  data_status: str,
  next_rebalance: str | None,
) -> None:
  badge = render_status_badge(data_status)
  st.markdown(
    f"""
    <div class="app-shell">
      <div class="app-header">
        <div>
          <div class="app-title">¿Qué hago hoy?</div>
          <div class="app-subtitle">
            Última señal disponible del sistema.
            No ejecuta operaciones automáticamente.
          </div>
        </div>
        <div class="header-meta">
          <div class="meta-pill">
            <div class="meta-pill-label">Última señal</div>
            <div class="meta-pill-value">{_esc(signal_date or "Sin datos")}</div>
          </div>
          <div class="meta-pill">
            <div class="meta-pill-label">Estado datos</div>
            <div class="meta-pill-value">{badge}</div>
          </div>
          <div class="meta-pill">
            <div class="meta-pill-label">Próximo rebalanceo</div>
            <div class="meta-pill-value">{_esc(next_rebalance or "—")}</div>
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
  )


def render_summary_row(
  strategy: str,
  risk: str,
  next_rebalance: str,
  capital: str,
) -> None:
  c1, c2, c3, c4 = st.columns(4)
  with c1:
    render_summary_card("Estrategia activa", strategy)
  with c2:
    render_summary_card("Nivel de riesgo", risk)
  with c3:
    render_summary_card("Próxima revisión", next_rebalance)
  with c4:
    render_summary_card("Capital configurado", capital)


def render_sidebar_navigation() -> None:
  st.sidebar.markdown("### Navegación")
  st.sidebar.page_link("pages/1_Que_Hago_Hoy.py", label="Qué hago hoy", icon="📋")
  st.sidebar.page_link("pages/2_Paper_Research.py", label="Paper Research", icon="🧪")

  with st.sidebar.expander("Investigación avanzada", expanded=False):
    st.caption("No mezclar con la señal operativa de hoy.")
    st.page_link("app.py", label="Trading Signals Lab (V14/V6)", icon="📊")
    st.markdown(
      "- **Portfolio V14**: modo V14 en la app principal\n"
      "- **Backtests**: resultados históricos en notebooks\n"
      "- **Resultados históricos**: carpeta `research_outputs/`"
    )

  st.sidebar.error("APPROVED_FOR_REAL_MONEY = False")


def render_order_section(
  plan: OrderPlanResult,
  show_avoid: bool,
) -> None:
  render_viability_card(plan)
  if plan.warnings:
    for warning in plan.warnings:
      render_warning_banner(warning)
  table_df = filter_order_rows(plan.rows, show_avoid=show_avoid)
  render_order_table(table_df)
