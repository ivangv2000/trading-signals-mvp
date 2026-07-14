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

explanation_page = st.Page(
    "views/how_v14_works.py",
    title="Cómo funciona",
    icon="🧠",
)

advanced_page = st.Page(
    "views/advanced_research.py",
    title="Investigación avanzada",
    icon="🧪",
    visibility="hidden",
)

navigation = st.navigation(
    [signals_page, explanation_page, advanced_page],
    position="top",
)

navigation.run()

# Paper trading público — no ejecuta órdenes.
from ui.v14_styles import render_footer_disclaimer

render_footer_disclaimer()
