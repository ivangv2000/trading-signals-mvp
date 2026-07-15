"""Hub de documentación — índice lateral y capítulos."""

from __future__ import annotations

import streamlit as st

from content.d2_technical_documentation import render_d2_documentation
from content.doc_utils import (
    DOC_ID_TO_LABEL,
    DOC_LABEL_TO_ID,
    DOC_SECTIONS,
    resolve_doc_param,
)
from content.general_documentation import (
    render_backtesting,
    render_data_universe,
    render_execution_costs,
    render_flow,
    render_forward_paper,
    render_indicators,
    render_integrity,
    render_limitations,
    render_metrics_doc,
    render_overview,
    render_ranking_portfolio,
    render_validation,
)
from content.research_history import render_project_history
from content.trading_glossary import filter_glossary
from content.v14_technical_documentation import render_v14_documentation
from ui.site_navigation import render_primary_navigation


def _current_section() -> str:
    qp = st.query_params.get("doc")
    if isinstance(qp, list):
        qp = qp[0] if qp else None
    return resolve_doc_param(qp)


render_primary_navigation("documentation_hub")

st.markdown('<h1 class="v14-hero-title">📚 Documentación del sistema</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="v14-hero-sub">Explicación completa de los datos, algoritmos, backtests, '
    "controles y limitaciones del proyecto.</p>",
    unsafe_allow_html=True,
)

labels = [label for _, label in DOC_SECTIONS]
default_label = DOC_ID_TO_LABEL.get(_current_section(), labels[0])
default_index = labels.index(default_label) if default_label in labels else 0

col_index, col_content = st.columns([1, 3], gap="large")

with col_index:
    st.markdown('<div class="doc-index-title">Índice documental</div>', unsafe_allow_html=True)
    selected_label = st.radio(
        "Capítulo",
        labels,
        index=default_index,
        label_visibility="collapsed",
        key="doc_chapter_radio",
    )
    section_id = DOC_LABEL_TO_ID[selected_label]
    if st.query_params.get("doc") != section_id:
        st.query_params["doc"] = section_id

with col_content:
    st.markdown(f'<div class="doc-chapter-title">{selected_label}</div>', unsafe_allow_html=True)

    if section_id == "overview":
        st.markdown(render_overview(), unsafe_allow_html=False)
    elif section_id == "flow":
        st.markdown(render_flow(), unsafe_allow_html=True)
    elif section_id == "v14":
        render_v14_documentation()
    elif section_id == "d2":
        render_d2_documentation()
    elif section_id == "data":
        st.markdown(render_data_universe())
    elif section_id == "indicators":
        st.markdown(render_indicators())
    elif section_id == "ranking":
        st.markdown(render_ranking_portfolio())
    elif section_id == "execution":
        st.markdown(render_execution_costs())
    elif section_id == "backtesting":
        st.markdown(render_backtesting())
    elif section_id == "metrics":
        st.markdown(render_metrics_doc())
    elif section_id == "validation":
        st.markdown(render_validation())
    elif section_id == "integrity":
        st.markdown(render_integrity())
    elif section_id == "forward_paper":
        st.markdown(render_forward_paper())
    elif section_id == "limitations":
        st.markdown(render_limitations())
    elif section_id == "history":
        st.markdown(render_project_history(), unsafe_allow_html=True)
    elif section_id == "glossary":
        st.markdown("### Glosario")
        query = st.text_input("Buscar término", placeholder="Ej. momentum, Sharpe, ledger…")
        filtered = filter_glossary(query)
        for term, definition in sorted(filtered.items()):
            st.markdown(f"**{term.capitalize()}** — {definition}")
        if not filtered:
            st.info("Ningún término coincide con la búsqueda.")
    else:
        st.info("Capítulo no disponible.")
