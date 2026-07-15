"""Navegación principal visual del sitio — tarjetas grandes con st.page_link."""

from __future__ import annotations

import streamlit as st

NAV_ITEMS = [
    {
        "id": "signals_v14",
        "page": "views/signals_v14.py",
        "icon": "📈",
        "title": "SEÑALES V14",
        "subtitle": "Cartera pública y acciones para ti",
        "css_class": "site-nav-card site-nav-v14",
    },
    {
        "id": "d2_shadow_research",
        "page": "views/d2_shadow_research.py",
        "icon": "🧪",
        "title": "INVESTIGACIÓN D2",
        "subtitle": "Experimento prospectivo en paper trading",
        "css_class": "site-nav-card site-nav-d2",
    },
    {
        "id": "documentation_hub",
        "page": "views/documentation_hub.py",
        "icon": "📚",
        "title": "DOCUMENTACIÓN",
        "subtitle": "Método, pruebas, métricas y limitaciones",
        "css_class": "site-nav-card site-nav-docs",
    },
]


def render_primary_navigation(active_page: str) -> None:
    """Renderiza tres tarjetas de navegación horizontales."""
    st.markdown('<div class="site-nav-shell">', unsafe_allow_html=True)
    cols = st.columns(3, gap="small")
    for col, item in zip(cols, NAV_ITEMS):
        active = item["id"] == active_page
        active_class = " site-nav-active" if active else ""
        with col:
            st.markdown(
                f'<div class="{item["css_class"]}{active_class}">'
                f'<div class="site-nav-icon">{item["icon"]}</div>'
                f'<div class="site-nav-title">{item["title"]}</div>'
                f'<div class="site-nav-subtitle">{item["subtitle"]}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            if not active:
                st.page_link(item["page"], label="Ir a esta sección", icon=item["icon"])
    st.markdown("</div>", unsafe_allow_html=True)
