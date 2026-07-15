"""Página principal pública — Señales V14.

Paper trading únicamente. No ejecuta órdenes.
"""

from __future__ import annotations

import streamlit as st

from services.user_portfolio_action_service import (
    PORTFOLIO_MODE_LABELS,
    build_universe_stats,
    build_user_portfolio_actions,
    count_position_buckets,
    derive_user_global_action,
)
from services.v14_signal_service import calculate_v14_signals
from services.v14_snapshot_service import load_latest_v14_snapshot, save_v14_snapshot
from ui.site_navigation import render_primary_navigation
from ui.v14_components import (
    compute_capital_summary,
    prepare_actions_table,
    render_capital_summary,
    render_defensive_section,
    render_fractional_warning,
    render_paper_banner,
    render_status_header,
    render_universe_section,
    render_user_global_action,
    render_user_signal_sections,
)

render_primary_navigation("signals_v14")

st.markdown('<h1 class="v14-hero-title">SEÑALES V14</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="v14-hero-sub">Cartera semanal generada por V14 R1 Return Engine.</p>',
    unsafe_allow_html=True,
)
render_paper_banner()

if "v14_capital" not in st.session_state:
    st.session_state.v14_capital = 100.0

capital = st.number_input(
    "Capital a simular (€)",
    min_value=1.0,
    value=float(st.session_state.v14_capital),
    step=10.0,
    key="v14_capital_input",
)
st.session_state.v14_capital = capital

col_btn, _ = st.columns([1, 3])
with col_btn:
    update_clicked = st.button("ACTUALIZAR SEÑALES V14", type="primary", use_container_width=True)

if update_clicked:
    with st.spinner("Descargando datos y calculando señales V14..."):
        raw = calculate_v14_signals(capital=capital)
    if raw.get("error"):
        st.error(raw["error"])
        if raw.get("download_errors"):
            with st.expander("Errores de descarga"):
                for err in raw["download_errors"]:
                    st.text(err)
    else:
        save_v14_snapshot(raw, capital_reference=capital)
        st.success("Señales V14 actualizadas.")
        st.rerun()

snapshot = load_latest_v14_snapshot()
summary = snapshot.get("summary")
signals_df = snapshot.get("signals")
has_data = snapshot.get("has_data", False)

render_status_header(summary)

if not has_data:
    render_user_global_action("NO DATA")
    st.markdown(
        """
        <div class="v14-card" style="text-align:center;padding:2rem;">
            <p style="font-size:1.1rem;color:#cbd5e1;">
                Aún no hay snapshot guardado. Pulsa <strong>ACTUALIZAR SEÑALES V14</strong>
                para generar la primera señal semanal.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown("### ¿Cuál es tu situación?")
    mode_labels = list(PORTFOLIO_MODE_LABELS.values())
    mode_keys = list(PORTFOLIO_MODE_LABELS.keys())
    selected_label = st.radio(
        "Modo de cartera",
        mode_labels,
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )
    portfolio_mode = mode_keys[mode_labels.index(selected_label)]

    tolerance = 0.01
    manual_holdings: dict[str, float] = {}
    if portfolio_mode == "manual":
        tolerance = st.slider(
            "Tolerancia de peso (puntos porcentuales)",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5,
        ) / 100.0
        st.markdown("Introduce el valor actual en euros de cada posición objetivo:")
        active_rows = signals_df[signals_df["target_weight"].fillna(0) > 0.001]
        prev_rows = signals_df[signals_df["previous_weight"].fillna(0) > 0.001]
        tickers = sorted(set(active_rows["ticker"].astype(str)) | set(prev_rows["ticker"].astype(str)))
        for ticker in tickers:
            manual_holdings[ticker.upper()] = st.number_input(
                f"{ticker} (€)",
                min_value=0.0,
                value=0.0,
                step=1.0,
                key=f"manual_{ticker}",
            )

    actions_df = build_user_portfolio_actions(
        signals_df,
        capital=capital,
        mode=portfolio_mode,
        manual_holdings=manual_holdings,
        tolerance=tolerance,
    )
    user_global_action = derive_user_global_action(actions_df)
    position_counts = count_position_buckets(actions_df)
    universe_stats = build_universe_stats(None, summary)

    render_user_global_action(user_global_action)
    render_fractional_warning()

    cap_summary = compute_capital_summary(actions_df, capital, position_counts)
    render_capital_summary(cap_summary)
    render_universe_section(universe_stats)

    risk_actions = actions_df[~actions_df["is_defensive"]]
    render_user_signal_sections(risk_actions)
    render_defensive_section(actions_df)

    show_avoid = st.toggle("Mostrar activos no seleccionados", value=False)
    table_df = prepare_actions_table(actions_df, show_avoid)
    st.markdown('<div class="v14-section-title">Tabla completa</div>', unsafe_allow_html=True)
    if table_df.empty:
        st.info("Sin filas para mostrar.")
    else:
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    with st.expander("Ver señal del modelo (información secundaria)"):
        model_action = (summary or {}).get("model_global_action") or (summary or {}).get("global_action", "—")
        st.markdown(f"**Decisión global del modelo:** {model_action}")
        st.dataframe(
            signals_df.rename(
                columns={
                    "signal": "Señal del modelo",
                    "previous_weight": "Peso anterior del modelo",
                    "target_weight": "Peso objetivo del modelo",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

st.markdown("---")
st.markdown('<div class="v14-section-title">Entiende cómo se genera esta señal</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="doc-link-cards">
        <div class="doc-mini-card"><div class="doc-mini-card-title">Indicadores</div>
        <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.35rem;">Momentum y tendencia por activo</div></div>
        <div class="doc-mini-card"><div class="doc-mini-card-title">Ranking</div>
        <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.35rem;">Comparación transversal semanal</div></div>
        <div class="doc-mini-card"><div class="doc-mini-card-title">Construcción de cartera</div>
        <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.35rem;">Top 3, vol target y SHY</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.page_link(
    "views/documentation_hub.py",
    label="Ver metodología completa de V14",
    icon="📚",
    query_params={"doc": "v14"},
)
