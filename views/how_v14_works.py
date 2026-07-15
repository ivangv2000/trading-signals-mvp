"""Vista de compatibilidad — redirige a la nueva documentación."""

from __future__ import annotations

import streamlit as st

from ui.site_navigation import render_primary_navigation

render_primary_navigation("documentation_hub")

st.markdown("# Esta documentación se ha trasladado")
st.markdown(
    """
La página **Cómo funciona** ya no es una sección principal del sitio.
Todo el contenido ampliado vive ahora en **Documentación**, con capítulos sobre V14,
D2, backtesting, métricas, validación y limitaciones.
"""
)
st.page_link(
    "views/documentation_hub.py",
    label="Abrir nueva documentación",
    icon="📚",
    query_params={"doc": "overview"},
)
