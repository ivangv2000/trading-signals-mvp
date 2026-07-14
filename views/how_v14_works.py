"""Página educativa — Cómo funciona V14."""

from __future__ import annotations

import streamlit as st

from content.v14_explanation import (
    A_TSMOM_MATH,
    A_TSMOM_SIMPLE,
    EXAMPLE_TEXT,
    FLOW_DIAGRAM,
    GLOSSARY,
    INTRO_TEXT,
    LIMITATIONS,
    METRIC_EXPLANATIONS,
    PROJECT_HISTORY,
    backtest_metrics_from_config,
    champion_section,
    portfolio_rules_from_config,
)

st.markdown("# Cómo funciona V14")
st.markdown(INTRO_TEXT)

sections = [
    "Qué hace V14",
    "El indicador principal",
    "Cómo crea la cartera",
    "Ejemplo sencillo",
    "Cómo se probó",
    "Por qué V14 es el modelo principal",
    "Limitaciones",
    "Glosario",
]
selected = st.radio("Índice", sections, horizontal=True, label_visibility="collapsed")

if selected == "Qué hace V14":
    st.markdown("## Qué hace V14")
    st.markdown(
        """
        - Analiza un universo amplio de acciones y ETF.
        - Utiliza datos de precios diarios.
        - Genera decisiones semanales (rebalanceo los viernes).
        - Compara la fortaleza relativa de los activos.
        - Selecciona una cartera concentrada (top N).
        - Mantiene posiciones hasta el siguiente rebalanceo.
        - Puede emitir **BUY**, **HOLD**, **REDUCE** y **SELL**.
        """
    )
    st.markdown('<div class="v14-flow">' + FLOW_DIAGRAM.replace("\n", "<br>") + "</div>", unsafe_allow_html=True)

elif selected == "El indicador principal":
    st.markdown("## El indicador principal: A_tsmom_63")
    st.markdown(A_TSMOM_SIMPLE)
    with st.expander("Ver detalle matemático"):
        st.markdown(A_TSMOM_MATH)

elif selected == "Cómo crea la cartera":
    st.markdown("## Cómo crea la cartera")
    rules = portfolio_rules_from_config()
    for key, label in [
        ("universo", "Universo"),
        ("frecuencia", "Frecuencia"),
        ("max_posiciones", "Número máximo de posiciones"),
        ("lookback_dias", "Ventana momentum (días)"),
        ("vol_objetivo", "Volatilidad objetivo"),
        ("seleccion", "Reglas de selección"),
        ("mantenimiento", "Reglas de mantenimiento"),
        ("pesos", "Asignación de pesos"),
        ("defensivo", "Activo defensivo"),
        ("riesgo", "Límites de riesgo"),
        ("ejecucion", "Momento de ejecución"),
        ("costes", "Costes simulados"),
    ]:
        val = rules.get(key, "Dato no disponible en la configuración cargada.")
        st.markdown(f"**{label}:** {val}")

elif selected == "Ejemplo sencillo":
    st.markdown("## Ejemplo sencillo")
    st.markdown(EXAMPLE_TEXT)

elif selected == "Cómo se probó":
    st.markdown("## Cómo se probó")
    metrics = backtest_metrics_from_config()
    if not metrics:
        st.warning("No hay métricas de backtest en la configuración cargada.")
    else:
        cols = st.columns(3)
        cards = [
            ("CAGR", metrics.get("CAGR"), f"{metrics.get('CAGR', '—')}%"),
            ("Sharpe", metrics.get("Sharpe"), metrics.get("Sharpe")),
            ("Sortino", metrics.get("Sortino"), metrics.get("Sortino")),
            ("Max Drawdown", metrics.get("Max Drawdown"), f"{metrics.get('Max Drawdown', '—')}%"),
            ("Rebalances", metrics.get("Rebalances"), metrics.get("Rebalances")),
            ("Overfitting risk", metrics.get("Overfitting risk"), metrics.get("Overfitting risk")),
        ]
        for i, (name, _raw, display) in enumerate(cards):
            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div class="v14-metric-card">
                        <div class="v14-metric-value">{display}</div>
                        <div class="v14-metric-label">{name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(METRIC_EXPLANATIONS.get(name, ""))
    st.markdown(
        """
        El backtest utiliza la configuración aprobada en `config/approved_v14_strategy.json`.
        Incluye rebalanceos semanales, benchmarks implícitos (SPY, QQQ, 60/40 en investigación)
        y métricas de robustez. No re-ejecuta el backtest al abrir esta página.
        """
    )

elif selected == "Por qué V14 es el modelo principal":
    st.markdown("## Por qué V14 es el modelo principal")
    champ = champion_section()
    st.markdown(
        f"""
        Dentro del proyecto, **V14** es el campeón actual de paper trading.
        No significa que sea el mejor algoritmo posible, pero:

        - Pasó las validaciones definidas para paper trading web.
        - Ofrece un equilibrio entre rentabilidad y riesgo en backtest.
        - Es el modelo principal de señales públicas.
        - Sigue siendo **experimental**.
        - **No está aprobado para dinero real.**

        <div class="v14-card" style="margin-top:1rem;">
            <div class="v14-card-label">{champ['title']}</div>
            <div style="font-size:1.5rem;font-weight:800;">{champ['strategy']}</div>
            <div style="color:#fbbf24;margin-top:0.5rem;">APPROVED_FOR_REAL_MONEY=False</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif selected == "Limitaciones":
    st.markdown("## Limitaciones")
    for item in LIMITATIONS:
        st.markdown(f"- {item}")

elif selected == "Glosario":
    st.markdown("## Glosario para principiantes")
    for term, definition in GLOSSARY.items():
        with st.expander(term):
            st.write(definition)

with st.expander("Ver evolución completa del proyecto"):
    st.markdown(PROJECT_HISTORY)

st.page_link("views/advanced_research.py", label="Abrir investigación avanzada", icon="🧪")
