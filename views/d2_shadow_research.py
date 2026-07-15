"""Página pública read-only — Investigación D2 shadow paper tracker."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services.d2_shadow_status_service import build_d2_shadow_view_model
from ui.site_navigation import render_primary_navigation
from ui.v14_styles import render_d2_experimental_banner, render_d2_integrity_alerts


def _positions_table(positions: list[dict]) -> pd.DataFrame:
    rows = []
    for p in positions:
        w = p.get("target_weight")
        rows.append({
            "Ticker": p.get("ticker"),
            "Peso objetivo": f"{w * 100:.1f}%" if w is not None else "—",
            "Rank": p.get("rank"),
            "Ejecución": p.get("execution_status", "—"),
        })
    return pd.DataFrame(rows)


vm = build_d2_shadow_view_model()

render_primary_navigation("d2_shadow_research")

st.markdown('<h1 class="v14-hero-title">🧪 Investigación D2</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="v14-hero-sub">Seguimiento experimental en paper trading. '
    "No sustituye a las señales oficiales V14.</p>",
    unsafe_allow_html=True,
)

render_d2_experimental_banner()
render_d2_integrity_alerts(vm)

col_pub, col_exp = st.columns(2)
with col_pub:
    st.markdown(
        """
        <div class="d2-model-card d2-model-public">
            <div class="d2-model-label">MODELO PÚBLICO</div>
            <div class="d2-model-name">V14 R1 Return Engine</div>
            <div class="d2-model-status d2-status-active">Estado: ACTIVO</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_exp:
    st.markdown(
        f"""
        <div class="d2-model-card d2-model-experimental">
            <div class="d2-model-label">MODELO EXPERIMENTAL</div>
            <div class="d2-model-name">D2 Trend Quality</div>
            <div class="d2-model-status d2-status-shadow">Estado: {vm["D2_paper_status"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="v14-section-title">Qué es D2</div>', unsafe_allow_html=True)
st.markdown(
    """
V14 selecciona los activos con mejor momentum y tendencia relativa.

D2 conserva el 70% del ranking V14 y añade un 30% de calidad de tendencia.
Intenta favorecer subidas estables frente a movimientos bruscos.

En el backtest D2 mejoró varias métricas, pero no superó todos los controles
estadísticos. Por eso se está observando con operaciones simuladas y sin
dinero real.
"""
)
st.markdown(
    """
    <div class="d2-formula-card">
        <div class="d2-formula-title">Fórmula simplificada</div>
        <div class="d2-formula-line"><strong>D2</strong> =</div>
        <div class="d2-formula-line">70% señal V14</div>
        <div class="d2-formula-line">+ 30% calidad de tendencia</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="v14-section-title">Estado actual</div>', unsafe_allow_html=True)

if vm["runtime_status"] == "WAITING_FOR_FIRST_FORWARD_SIGNAL":
    st.markdown(
        """
        <div class="d2-waiting-card">
            <div class="d2-waiting-title">Esperando la primera señal prospectiva</div>
            <p>El seguimiento comenzó el 15/07/2026. La primera señal permitida se
            calculará después del cierre semanal válido del 17/07/2026.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif vm["runtime_status"] == "DATA_REVISION_DETECTED":
    st.error("Seguimiento detenido por revisión de datos históricos.")

status_cols = st.columns(4)
status_cols[0].metric("Señales registradas", vm["completed_signals"])
status_cols[1].metric("Ejecuciones completadas", vm["completed_executions"])
status_cols[2].metric("Próximo checkpoint", f"{vm['next_checkpoint']} ejec.")
status_cols[3].metric("Contratos verificados", "Sí" if vm["contracts_ok"] else "No")

if vm["latest_signal_date"] and vm["completed_signals"] > 0:
    st.markdown(f"**Última señal:** {vm['latest_signal_date']}")

    st.markdown('<div class="v14-section-title">Comparación de carteras</div>', unsafe_allow_html=True)
    diff = vm.get("portfolio_diff", {})
    if diff.get("same_portfolio"):
        st.info("Esta semana V14 y D2 proponen la misma cartera.")
    else:
        if diff.get("only_c0"):
            st.write(f"Solo en V14: {', '.join(diff['only_c0'])}")
        if diff.get("only_d2"):
            st.write(f"Solo en D2: {', '.join(diff['only_d2'])}")

    col_c0, col_d2 = st.columns(2)
    with col_c0:
        st.markdown("#### V14 — Cartera oficial del modelo")
        if vm["latest_C0_positions"]:
            st.dataframe(_positions_table(vm["latest_C0_positions"]), use_container_width=True, hide_index=True)
        else:
            st.caption("Sin posiciones registradas.")
    with col_d2:
        st.markdown("#### D2 — Cartera experimental")
        st.caption("Experimental — no es una recomendación pública")
        if vm["latest_D2_positions"]:
            st.dataframe(_positions_table(vm["latest_D2_positions"]), use_container_width=True, hide_index=True)
        else:
            st.caption("Sin posiciones registradas.")

    if vm["executions_preview"]:
        st.markdown('<div class="v14-section-title">Ejecuciones</div>', unsafe_allow_html=True)
        if vm["pending_execution_count"] > 0:
            st.warning(
                "Hay señales registradas esperando el precio de apertura de la siguiente sesión."
            )
        st.dataframe(pd.DataFrame(vm["executions_preview"]), use_container_width=True, hide_index=True)

st.markdown('<div class="v14-section-title">Métricas prospectivas</div>', unsafe_allow_html=True)

if vm["sample_status"] == "INSUFFICIENT_SAMPLE":
    st.markdown(
        """
        <div class="d2-sample-guard">
            <strong>Muestra insuficiente</strong><br>
            Todavía no hay suficientes datos prospectivos para interpretar métricas anualizadas.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.caption("Resultado provisional de paper trading")

metric_cols = st.columns(3)
if vm["C0_paper_return"] is not None:
    metric_cols[0].metric("Rentabilidad acumulada V14", f"{vm['C0_paper_return']:.2f}%")
if vm["D2_paper_return"] is not None:
    metric_cols[1].metric("Rentabilidad acumulada D2", f"{vm['D2_paper_return']:.2f}%")
if vm["D2_active_return"] is not None:
    metric_cols[2].metric("Diferencia D2 vs V14", f"{vm['D2_active_return']:.2f}%")

if vm["C0_max_drawdown"] is not None or vm["D2_max_drawdown"] is not None:
    dd_cols = st.columns(3)
    if vm["C0_max_drawdown"] is not None:
        dd_cols[0].metric("Drawdown V14", f"{vm['C0_max_drawdown']:.2f}%")
    if vm["D2_max_drawdown"] is not None:
        dd_cols[1].metric("Drawdown D2", f"{vm['D2_max_drawdown']:.2f}%")
    if vm["paper_costs"] is not None:
        dd_cols[2].metric("Costes", f"{vm['paper_costs']:.2f}")

if vm["completed_weeks"]:
    st.metric("Semanas completadas", vm["completed_weeks"])

if vm["sample_status"] == "AVAILABLE" and vm["information_ratio"] is not None:
    st.metric("Information ratio (provisional)", vm["information_ratio"])

st.markdown('<div class="v14-section-title">Entiende el experimento</div>', unsafe_allow_html=True)
st.markdown(
    """
- **Qué añade D2:** un 30% de calidad de tendencia (slope × R²) sobre el ranking V14.
- **Por qué no fue seleccionado:** mejoró varias métricas pero falló G7 (IR) y G13 (PBO).
- **Qué se mide en forward paper:** cartera experimental vs V14, ejecuciones, costes y checkpoints 13/26/52.
"""
)
st.page_link(
    "views/documentation_hub.py",
    label="Ver metodología completa de D2",
    icon="📚",
    query_params={"doc": "d2"},
)

st.markdown("---")
st.caption(
    "Las señales se actualizan mediante el proceso externo del tracker después del cierre semanal."
)
if st.button("Actualizar pantalla", type="secondary"):
    st.rerun()

st.markdown(
    '<p class="d2-disclaimer">Simulación en paper trading · no ejecuta órdenes · '
    "no aprobado para dinero real</p>",
    unsafe_allow_html=True,
)
