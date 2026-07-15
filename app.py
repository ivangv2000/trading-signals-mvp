"""
Router principal — Web pública V14 R1 Return Engine.
"""

import streamlit as st

from ui.v14_styles import apply_v14_global_styles

st.set_page_config(
    page_title="Señales V14",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

apply_v14_global_styles()

signals_page = st.Page(
    "views/signals_v14.py",
    title="Señales V14",
    icon="📈",
    default=True,
)

d2_research_page = st.Page(
    "views/d2_shadow_research.py",
    title="Investigación D2",
    icon="🧪",
)

documentation_page = st.Page(
    "views/documentation_hub.py",
    title="Documentación",
    icon="📚",
)

legacy_how_page = st.Page(
    "views/how_v14_works.py",
    title="Cómo funciona (legacy)",
    icon="🧠",
    visibility="hidden",
)

advanced_page = st.Page(
    "views/advanced_research.py",
    title="Investigación avanzada",
    icon="🧪",
    visibility="hidden",
)

navigation = st.navigation(
    [signals_page, d2_research_page, documentation_page, legacy_how_page, advanced_page],
    position="hidden",
)

navigation.run()

from ui.v14_styles import render_footer_disclaimer

render_footer_disclaimer()
