"""Página pública read-only — Investigación D2 shadow paper tracker."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services.d2_shadow_status_service import build_d2_shadow_view_model
from services.research_status_service import load_latest_d2_status
from ui.research_status_components import (
    render_execution_status,
    render_model_comparison_table,
    render_research_disclaimer,
    render_status_badge,
)
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
d2 = load_latest_d2_status()

render_primary_navigation("d2_shadow_research")

st.markdown('<h1 class="v14-hero-title">🧪 Investigación D2</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="v14-hero-sub">Seguimiento experimental en paper trading. '
    "No sustituye a las señales oficiales V14.</p>",
    unsafe_allow_html=True,
)

render_d2_experimental_banner()
render_research_disclaimer()
render_status_badge("RESEARCH_ONLY", "INVESTIGACIÓN · PAPER TRADING")
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

st.markdown('<div class="v14-section-title">Estado actual</div>', unsafe_allow_html=True)
runtime = d2.get("runtime_status") or vm.get("runtime_status")
if runtime == "WAITING_FOR_FIRST_FORWARD_SIGNAL":
    first = vm.get("first_permitted_signal") or "—"
    render_execution_status(
        "Esperando la primera señal prospectiva",
        f"La primera señal permitida se calculará tras el cierre semanal válido ({first}).",
    )
elif runtime == "FORWARD_SIGNAL_RECORDED_PENDING_EXECUTION":
    render_execution_status(
        d2.get("runtime_status_label") or "Señal registrada · pendiente de apertura",
        f"Última signal_date: {d2.get('signal_date') or vm.get('latest_signal_date') or '—'}",
    )
elif runtime == "DATA_REVISION_DETECTED":
    st.error("Seguimiento detenido por revisión de datos históricos.")

status_cols = st.columns(4)
status_cols[0].metric("Señales registradas", d2.get("completed_signals", vm["completed_signals"]))
status_cols[1].metric("Ejecuciones completadas", d2.get("completed_executions", vm["completed_executions"]))
status_cols[2].metric("Próximo checkpoint", f"{d2.get('next_checkpoint', vm['next_checkpoint'])} ejec.")
status_cols[3].metric("Runtime", d2.get("runtime_status_label") or runtime or "—")

st.write(f"**Histórico:** {d2.get('historical_status') or vm.get('D2_historical_status') or 'D2_NOT_SELECTED'}")
st.write(f"**Paper:** {d2.get('paper_status') or vm.get('D2_paper_status') or 'RESEARCH_SHADOW_NOT_SELECTED'}")
st.caption("APPROVED_FOR_REAL_MONEY=False · D2 no es una mejora confirmada.")

if (d2.get("signal_date") or vm.get("latest_signal_date")) and int(d2.get("completed_signals", vm.get("completed_signals", 0)) or 0) > 0:
    st.markdown(f"**Última señal:** {d2.get('signal_date') or vm.get('latest_signal_date')}")

    st.markdown('<div class="v14-section-title">Comparación C0/V14 vs D2</div>', unsafe_allow_html=True)
    if d2.get("diff_summary"):
        st.info(d2["diff_summary"])
    render_model_comparison_table(
        [
            {
                "ticker": r["ticker"],
                "c0_weight": r.get("c0_weight", r.get("base_weight")),
                "d2_weight": r.get("d2_weight", r.get("other_weight")),
                "baseline_rank": r.get("baseline_rank"),
                "trend_quality_rank": r.get("trend_quality_rank"),
                "final_score": r.get("final_score"),
                "situation": r.get("situation"),
            }
            for r in (d2.get("comparison") or [])
        ],
        [
            ("ticker", "Ticker"),
            ("c0_weight", "C0 weight"),
            ("d2_weight", "D2 weight"),
            ("baseline_rank", "Baseline rank"),
            ("trend_quality_rank", "Trend quality rank"),
            ("final_score", "Final score"),
            ("situation", "Situación"),
        ],
    )

    col_c0, col_d2p = st.columns(2)
    with col_c0:
        st.markdown("#### C0 / V14 baseline")
        if vm["latest_C0_positions"]:
            st.dataframe(_positions_table(vm["latest_C0_positions"]), use_container_width=True, hide_index=True)
        else:
            st.caption("Sin posiciones registradas.")
    with col_d2p:
        st.markdown("#### D2 — Cartera experimental")
        st.caption("Experimental — no es una recomendación pública")
        if vm["latest_D2_positions"]:
            st.dataframe(_positions_table(vm["latest_D2_positions"]), use_container_width=True, hide_index=True)
        else:
            st.caption("Sin posiciones registradas.")

    if vm["executions_preview"]:
        st.markdown('<div class="v14-section-title">Ejecuciones</div>', unsafe_allow_html=True)
        if vm["pending_execution_count"] > 0:
            st.warning("Hay señales registradas esperando el precio de apertura de la siguiente sesión.")
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

st.markdown('<div class="v14-section-title">Entiende el experimento</div>', unsafe_allow_html=True)
st.markdown(
    """
- **Qué añade D2:** un 30% de calidad de tendencia sobre el ranking V14.
- **Por qué no fue seleccionado:** mejoró varias métricas pero falló controles estadísticos clave.
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
