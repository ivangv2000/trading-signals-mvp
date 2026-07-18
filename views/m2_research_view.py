"""Read-only research page for M2 shadow paper tracker."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services.m2_shadow_status_service import (
    B0_STRATEGY,
    CHECKPOINT_WEEKS,
    FIRST_PERMITTED_SIGNAL,
    M2_STRATEGY,
    MIN_SESSIONS_PRELIMINARY,
    PAPER_START_DATE,
    get_m2_shadow_dashboard_data,
)
from ui.site_navigation import render_primary_navigation
from ui.v14_styles import render_d2_integrity_alerts, render_d2_experimental_banner

RUNTIME_LABELS = {
    "WAITING_FOR_FIRST_FORWARD_SIGNAL": "Esperando la primera senal prospectiva",
    "WAITING_FOR_MARKET_CLOSE": "Esperando el cierre del mercado",
    "NO_NEW_DATA": "Cierre todavia no disponible",
    "SIGNAL_RECORDED": "Nueva senal simulada registrada",
    "PENDING_EXECUTION_PRICE": "Esperando la siguiente apertura",
    "ALREADY_PROCESSED_IDEMPOTENT": "La senal ya estaba registrada",
    "DATA_REVISION_DETECTED": "Revision de datos detectada; tracker detenido",
}


def _fmt_pct(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value) * 100:.1f}%"


def _fmt_pct_points(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value):.2f}%"


def _fmt_float(value, digits: int = 2) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def _render_badges() -> None:
    st.markdown(
        """
        <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin:0.2rem 0 1rem 0;">
            <span class="d2-model-status d2-status-shadow">RESEARCH SHADOW</span>
            <span class="d2-model-status" style="background:#7c2d12;color:#fdba74;">NO SELECCIONADO</span>
            <span class="d2-model-status" style="background:#7f1d1d;color:#fecaca;">NO APROBADO PARA DINERO REAL</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _signal_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out = out[["ticker", "selection_rank", "target_weight", "action", "execution_status", "model_version_id"]]
    out.columns = ["Ticker", "Rank", "Peso objetivo", "Accion", "Ejecucion", "Version"]
    out["Peso objetivo"] = out["Peso objetivo"].map(_fmt_pct)
    return out


def _execution_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["execution_open"] = pd.to_numeric(out["execution_open"], errors="coerce").map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
    out["estimated_cost"] = pd.to_numeric(out["estimated_cost"], errors="coerce").map(lambda x: "—" if pd.isna(x) else f"{x:.6f}")
    out["weight_change"] = pd.to_numeric(out["weight_change"], errors="coerce").map(_fmt_pct)
    return out[[
        "execution_date", "strategy", "ticker", "weight_change",
        "execution_open", "estimated_cost", "execution_status", "model_version_id"
    ]].rename(columns={
        "execution_date": "Fecha",
        "strategy": "Estrategia",
        "ticker": "Ticker",
        "weight_change": "Cambio peso",
        "execution_open": "Open",
        "estimated_cost": "Coste estimado",
        "execution_status": "Estado",
        "model_version_id": "Version",
    })


def _equity_chart_data(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(index="date", columns="strategy", values="equity", aggfunc="last").sort_index()
    cols = [c for c in [B0_STRATEGY, M2_STRATEGY] if c in pivot.columns]
    return pivot[cols] if cols else pd.DataFrame()


def _cost_summary(df: pd.DataFrame) -> tuple[str, str]:
    if df.empty or "cumulative_costs" not in df.columns:
        return "—", "0"
    costs = df.groupby("strategy")["cumulative_costs"].last()
    m2_cost = costs.get(M2_STRATEGY)
    sessions = df["date"].nunique() if "date" in df.columns else 0
    return _fmt_float(m2_cost, 2), str(int(sessions))


def render_m2_research_view() -> None:
    from services.research_status_service import (
        load_latest_m2_status,
        load_latest_m5_status,
    )
    from ui.research_status_components import (
        render_execution_status,
        render_model_comparison_table,
        render_research_disclaimer,
        render_status_badge,
    )

    vm = get_m2_shadow_dashboard_data()
    m2 = load_latest_m2_status()
    m5 = load_latest_m5_status()

    render_primary_navigation("m2_research_view")
    st.markdown('<h1 class="v14-hero-title">Investigación M2 y M5</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="v14-hero-sub">Modelo predictivo semanal y combinación adaptativa en paper trading</p>',
        unsafe_allow_html=True,
    )
    _render_badges()
    st.markdown("**V14 R1 Return Engine continua siendo el modelo publico.**")
    render_research_disclaimer()
    render_status_badge("RESEARCH_ONLY", "INVESTIGACIÓN · PAPER TRADING")

    render_d2_experimental_banner()
    render_d2_integrity_alerts({
        "contracts_ok": vm["tracker"].get("contracts_ok", True),
        "data_revision_detected": vm["tracker"].get("data_revision_detected", False),
    })

    for warning in vm["warnings"]:
        st.caption(f"Aviso: {warning}")

    if not vm["available"]:
        st.info("El tracker M2 todavia no esta inicializado.")

    tracker = vm["tracker"]
    model = vm["model"]

    st.markdown('<div class="v14-section-title">A. Estado actual</div>', unsafe_allow_html=True)
    status_cols = st.columns(4)
    status_cols[0].metric("Ultima signal_date", m2.get("signal_date") or "—")
    status_cols[1].metric("Modelo congelado", m2.get("model_version_id") or tracker.get("current_model_version_id") or "—")
    status_cols[2].metric("Completed executions", m2.get("completed_executions", tracker.get("completed_executions", 0)))
    status_cols[3].metric("Siguiente checkpoint", m2.get("next_checkpoint") or tracker.get("next_checkpoint") or 13)
    st.write(f"**Estado M2:** {m2.get('paper_status') or tracker.get('M2_paper_status') or '—'}")
    st.write(f"**Estado M5:** {m5.get('research_classification') or m5.get('classification') or '—'}")
    st.caption("APPROVED_FOR_REAL_MONEY=False")

    st.markdown('<div class="v14-section-title">B. Comparación V14 vs M2</div>', unsafe_allow_html=True)
    if m2.get("diff_summary"):
        st.info(m2["diff_summary"])
    render_model_comparison_table(
        [
            {
                "ticker": r["ticker"],
                "v14_weight": r["base_weight"],
                "m2_weight": r["other_weight"],
                "m2_prediction": r.get("m2_prediction"),
                "m2_rank": r.get("m2_rank"),
                "situation": r["situation"],
            }
            for r in (m2.get("comparison") or [])
        ],
        [
            ("ticker", "Ticker"),
            ("v14_weight", "V14 weight"),
            ("m2_weight", "M2 weight"),
            ("m2_prediction", "M2 prediction"),
            ("m2_rank", "M2 rank"),
            ("situation", "Situación"),
        ],
    )

    st.markdown('<div class="v14-section-title">C. M5 — combinación adaptativa</div>', unsafe_allow_html=True)
    conv = m5.get("conviction") or {}
    m5_cols = st.columns(4)
    m5_cols[0].metric("Conviction raw", _fmt_float(conv.get("conviction_raw"), 4))
    m5_cols[1].metric("Conviction percentile", _fmt_float(conv.get("conviction_percentile"), 3))
    m5_cols[2].metric("M2 weight", _fmt_pct(conv.get("m2_weight")))
    m5_cols[3].metric("Baseline weight", _fmt_pct(conv.get("baseline_weight")))
    if m5.get("explanation"):
        st.info(m5["explanation"])
    if m5.get("holdings"):
        st.dataframe(
            pd.DataFrame(m5["holdings"]).assign(
                target_weight=lambda d: d["target_weight"].map(_fmt_pct)
            ).rename(columns={"ticker": "Ticker", "target_weight": "Peso"}),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown('<div class="v14-section-title">D. Ejecución</div>', unsafe_allow_html=True)
    render_execution_status(
        m2.get("execution_status_label") or "Pendiente de la próxima apertura",
        "Las filas paper permanecen pendientes hasta que exista precio de apertura.",
    )
    exec_rows = []
    for src in (m2.get("executions") or []):
        exec_rows.append({
            "Fecha": src.get("execution_date") or "—",
            "Ticker": src.get("ticker"),
            "Precio": src.get("execution_open") if src.get("execution_open") not in (None, "") else "—",
            "Coste": src.get("estimated_cost") if src.get("estimated_cost") not in (None, "") else "—",
            "Estado": src.get("execution_status"),
        })
    for src in (m5.get("executions") or []):
        exec_rows.append({
            "Fecha": src.get("execution_date") or "—",
            "Ticker": src.get("ticker"),
            "Precio": src.get("execution_price") if src.get("execution_price") not in (None, "") else "—",
            "Coste": src.get("estimated_cost") if src.get("estimated_cost") not in (None, "") else "—",
            "Estado": src.get("status"),
        })
    if exec_rows:
        st.dataframe(pd.DataFrame(exec_rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="v14-section-title">E. Evidencia</div>', unsafe_allow_html=True)
    weeks = int(m2.get("completed_weeks") or tracker.get("completed_weeks") or 0)
    ecols = st.columns(4)
    ecols[0].metric("Semanas prospectivas", weeks)
    ecols[1].metric("Checkpoint 13", "pendiente" if weeks < 13 else "alcanzado")
    ecols[2].metric("Checkpoint 26", "pendiente" if weeks < 26 else "alcanzado")
    ecols[3].metric("Checkpoint 52", "pendiente" if weeks < 52 else "alcanzado")
    st.warning("Evidencia actual insuficiente. No se calcula rentabilidad antes de ejecuciones y semanas resueltas.")

    runtime_text = RUNTIME_LABELS.get(tracker["runtime_status"], tracker["runtime_status"])
    st.markdown('<div class="v14-section-title">Detalle tecnico del tracker M2</div>', unsafe_allow_html=True)
    meta_cols = st.columns(4)
    meta_cols[0].metric("Estado del tracker", runtime_text)
    meta_cols[1].metric("Senales registradas", tracker["completed_signal_batches"])
    meta_cols[2].metric("Inicio del paper", tracker["paper_start_date"] or PAPER_START_DATE)
    meta_cols[3].metric("Primera senal permitida", tracker["first_permitted_signal"] or FIRST_PERMITTED_SIGNAL)

    progress = 0.0
    if tracker["next_checkpoint"] > 0:
        progress = min(float(tracker["completed_weeks"]) / float(tracker["next_checkpoint"]), 1.0)
    st.progress(progress)
    if tracker["completed_weeks"] < CHECKPOINT_WEEKS:
        st.caption("Muestra prospectiva todavia insuficiente para sacar conclusiones.")

    st.markdown('<div class="v14-section-title">Como funciona M2</div>', unsafe_allow_html=True)
    st.markdown(
        """
M2 intenta ordenar los activos segun su probabilidad relativa de terminar
en el cuartil superior durante la semana siguiente.

Usa diez indicadores historicos y combina:

- 70% ranking V14
- 30% ranking predictivo M2

Selecciona normalmente los tres activos mejor clasificados y utiliza las
mismas reglas de pesos, tramo defensivo, costes y ejecucion que V14.
        """
    )
    st.markdown(
        '<div class="v14-warning">Las probabilidades de M2 se utilizan para ordenar activos. '
        "No deben interpretarse como porcentajes literales de ganar dinero.</div>",
        unsafe_allow_html=True,
    )

    historical = vm["historical"]
    st.markdown('<div class="v14-section-title">Resultados historicos validados</div>', unsafe_allow_html=True)
    st.caption("BACKTEST HISTORICO — NO PROSPECTIVO")
    if historical["available"]:
        hist_cols = st.columns(4)
        hist_cols[0].metric("M2 CAGR", _fmt_pct_points(historical["M2_CAGR"]))
        hist_cols[1].metric("B0/V14 CAGR", _fmt_pct_points(historical["B0_CAGR"]))
        hist_cols[2].metric("Excess CAGR", _fmt_pct_points(historical["M2_excess_CAGR"]))
        hist_cols[3].metric("M2 Sharpe", _fmt_float(historical["M2_sharpe"], 2))
        st.caption(f"Periodo historico comun: {historical['historical_period_start']} a {historical['historical_period_end']}")
        st.caption(f"Trial count: {historical['trial_count']}")
    else:
        st.caption("No se pudieron cargar las metricas historicas validadas.")

    with st.expander("Que sabemos del modelo?"):
        diagnostic = vm["diagnostic"]
        if diagnostic.get("available"):
            dcols = st.columns(4)
            dcols[0].metric("AUC", _fmt_float(diagnostic.get("AUC"), 3))
            dcols[1].metric("Brier Skill", _fmt_float(diagnostic.get("brier_skill"), 3))
            dcols[2].metric("Ranking quality", diagnostic.get("ranking_quality") or "—")
            dcols[3].metric("Calibration status", diagnostic.get("calibration_status") or "—")
        else:
            st.caption("No hay diagnostico adicional disponible.")

    st.markdown('<div class="v14-section-title">Ultima senal simulada</div>', unsafe_allow_html=True)
    latest_signal = vm["latest_signal"]
    if latest_signal.empty:
        st.info("Todavia no hay ninguna senal prospectiva registrada.")
        st.caption(f"Primera senal permitida: {FIRST_PERMITTED_SIGNAL}")
        st.caption(r"La pagina se actualizara cuando se ejecute manualmente: `python paper_research\m2_shadow_tracker.py --update`")
    else:
        batch_id = str(latest_signal["signal_batch_id"].astype(str).iloc[0])
        signal_date = str(latest_signal["signal_date"].astype(str).iloc[0])
        generated_at = str(latest_signal["generated_at_utc"].astype(str).iloc[0])
        st.write(f"**Signal batch:** {batch_id}")
        st.write(f"**Signal date:** {signal_date}")
        st.write(f"**Generated at UTC:** {generated_at}")

        b0_df = latest_signal[latest_signal["strategy"].astype(str) == B0_STRATEGY]
        m2_df = latest_signal[latest_signal["strategy"].astype(str) == M2_STRATEGY]
        col_b0, col_m2 = st.columns(2)
        with col_b0:
            st.markdown("#### B0 / V14")
            st.dataframe(_signal_table(b0_df), use_container_width=True, hide_index=True)
        with col_m2:
            st.markdown("#### M2")
            st.dataframe(_signal_table(m2_df), use_container_width=True, hide_index=True)
        st.caption("Cartera simulada de investigacion, no recomendacion personal.")

    st.markdown('<div class="v14-section-title">Ultimas ejecuciones simuladas</div>', unsafe_allow_html=True)
    executions = vm["latest_executions"]
    if executions.empty:
        st.info("Todavia no hay ejecuciones. Las senales se ejecutan en la siguiente apertura disponible.")
    else:
        st.dataframe(_execution_table(executions), use_container_width=True, hide_index=True)

    st.markdown('<div class="v14-section-title">Seguimiento prospectivo</div>', unsafe_allow_html=True)
    equity = vm["equity"]
    if not vm["equity_ready"]:
        st.info("Todavia no existe una muestra suficiente para representar la evolucion.")
    else:
        if vm["preliminary_sample"]:
            st.caption("Resultados preliminares. La muestra todavia es demasiado pequena.")
        chart_df = _equity_chart_data(equity)
        if not chart_df.empty:
            st.line_chart(chart_df, use_container_width=True)
        costs, sessions = _cost_summary(equity)
        eco = st.columns(3)
        eco[0].metric("Costes acumulados M2", costs)
        eco[1].metric("Sesiones observadas", sessions)
        eco[2].metric("Completed weeks", tracker["completed_weeks"])
        if int(sessions) < MIN_SESSIONS_PRELIMINARY:
            st.caption("No se muestran metricas concluyentes como CAGR, Sharpe o DSR antes de 63 sesiones.")

    st.markdown('<div class="v14-section-title">Riesgos y limitaciones</div>', unsafe_allow_html=True)
    st.markdown(
        """
- backtest no garantiza rentabilidad;
- universo historico fijo y riesgo de survivorship bias;
- cartera concentrada normalmente en tres posiciones;
- la muestra prospectiva todavia es insuficiente;
- APPROVED_FOR_REAL_MONEY=False.
        """
    )
    st.markdown("---")
    st.caption("Pagina de solo lectura. No ejecuta el tracker ni modifica archivos de paper_research.")


render_m2_research_view()
