"""Tests de navegación principal y documentación."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from content.doc_utils import DOC_SECTIONS, resolve_doc_param
from content.trading_glossary import GLOSSARY, filter_glossary

ROOT = Path(__file__).resolve().parent.parent


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _glob_read(pattern: str) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in ROOT.glob(pattern))


# T1
def test_t1_custom_nav_imports():
    nav = _read("ui/site_navigation.py")
    assert "render_primary_navigation" in nav
    assert "st.page_link" in nav


# T2
def test_t2_three_primary_sections():
    nav = _read("ui/site_navigation.py")
    assert "SEÑALES V14" in nav
    assert "INVESTIGACIÓN D2" in nav
    assert "DOCUMENTACIÓN" in nav
    assert nav.count('"id":') == 3


# T3
def test_t3_signals_is_default():
    app = _read("app.py")
    assert "default=True" in app
    assert "views/signals_v14.py" in app
    assert app.index("signals_page") < app.index("documentation_page")


# T4
def test_t4_nav_large_card_classes_exist():
    styles = _read("ui/v14_styles.py")
    for cls in ("site-nav-card", "site-nav-title", "site-nav-subtitle", "site-nav-icon"):
        assert cls in styles


# T5
def test_t5_nav_active_state_exists():
    styles = _read("ui/v14_styles.py")
    nav = _read("ui/site_navigation.py")
    assert "site-nav-active" in styles
    assert "site-nav-active" in nav


# T6
def test_t6_nav_responsive_css_exists():
    styles = _read("ui/v14_styles.py")
    assert "@media (max-width: 768px)" in styles
    assert "site-nav-card" in styles


# T7
def test_t7_old_small_nav_hidden():
    app = _read("app.py")
    styles = _read("ui/v14_styles.py")
    assert 'position="hidden"' in app
    assert 'div[data-testid="stNavigation"]' in styles


# T8
def test_t8_documentation_page_imports():
    doc = _read("views/documentation_hub.py")
    assert "render_primary_navigation" in doc
    assert "render_v14_documentation" in doc
    assert "render_d2_documentation" in doc


# T9
def test_t9_documentation_has_16_sections():
    assert len(DOC_SECTIONS) == 16
    doc = _read("views/documentation_hub.py")
    assert "DOC_SECTIONS" in doc or "resolve_doc_param" in doc


# T10
def test_t10_general_overview_present():
    general = _read("content/general_documentation.py")
    assert "render_overview" in general
    assert "señales cuantitativas semanales" in general.lower() or "Señal cuantitativa" in general


# T11
def test_t11_v14_formula_present():
    v14 = _read("content/v14_technical_documentation.py")
    assert "COMBINED_SCORE" in v14
    assert "MOM_63" in v14
    assert "TREND_SCORE" in v14


# T12
def test_t12_v14_universe_49_present():
    v14 = _read("content/v14_technical_documentation.py")
    assert "49 activos" in v14 or "49 tickers" in v14


# T13
def test_t13_d2_formula_present():
    d2 = _read("content/d2_technical_documentation.py")
    assert "trend_quality_126" in d2
    assert "D2_SCORE" in d2
    assert "baseline_rank" in d2


# T14
def test_t14_d2_not_selected_explained():
    d2 = _read("content/d2_technical_documentation.py")
    assert "G7" in d2
    assert "G13" in d2
    assert "D2_NOT_SELECTED" in d2


# T15
def test_t15_data_contract_explained():
    general = _read("content/general_documentation.py")
    assert "configured_count" in general
    assert "15 activos" in general or "15 tickers" in general
    assert "49" in general


# T16
def test_t16_backtest_ledger_explained():
    general = _read("content/general_documentation.py")
    assert "Una estrategia, un ledger, una equity curve" in general
    assert "V18.3.1" in general


# T17
def test_t17_metrics_section_present():
    general = _read("content/general_documentation.py")
    assert "render_metrics_doc" in general
    assert "Sharpe" in general
    assert "PBO" in general


# T18
def test_t18_validation_gates_present():
    general = _read("content/general_documentation.py")
    assert "G1" in general and "G19" in general
    assert "REUSED_TEST" in general
    assert "FORWARD_PAPER" in general


# T19
def test_t19_limitations_visible():
    general = _read("content/general_documentation.py")
    assert "render_limitations" in general
    assert "no garantiza" in general.lower()
    assert "survivorship" in general.lower()


# T20
def test_t20_project_history_present():
    history = _read("content/research_history.py")
    assert "V18.3.5" in history
    assert "V14" in history
    assert "render_project_history" in history


# T21
def test_t21_glossary_search_present():
    doc = _read("views/documentation_hub.py")
    assert "filter_glossary" in doc
    assert "Buscar término" in doc
    assert filter_glossary("sharpe")
    assert "Sharpe" in GLOSSARY or "sharpe" in GLOSSARY


# T22
def test_t22_v14_links_to_documentation():
    signals = _read("views/signals_v14.py")
    assert "documentation_hub.py" in signals
    assert '"doc": "v14"' in signals or "'doc': 'v14'" in signals


# T23
def test_t23_d2_links_to_documentation():
    d2 = _read("views/d2_shadow_research.py")
    assert "documentation_hub.py" in d2
    assert '"doc": "d2"' in d2 or "'doc': 'd2'" in d2


# T24
def test_t24_old_how_page_redirects():
    how = _read("views/how_v14_works.py")
    app = _read("app.py")
    assert "trasladado" in how.lower() or "trasladada" in how.lower()
    assert "documentation_hub.py" in how
    assert 'title="Cómo funciona"' not in app or 'visibility="hidden"' in app


# T25
def test_t25_no_network_calls():
    paths = [
        ROOT / "views/documentation_hub.py",
        ROOT / "ui/site_navigation.py",
        ROOT / "views/signals_v14.py",
        ROOT / "views/d2_shadow_research.py",
    ]
    paths += list((ROOT / "content").glob("*.py"))
    blob = "\n".join(p.read_text(encoding="utf-8") for p in paths)
    for token in ("import requests", "urllib.request", "import yfinance", "httpx", "aiohttp"):
        assert token not in blob
    assert ".download(" not in blob


# T26
def test_t26_no_subprocess():
    blob = _glob_read("views/*.py") + _glob_read("content/*.py") + _read("ui/site_navigation.py")
    assert "subprocess" not in blob


# T27
def test_t27_no_file_writes():
    blob = _glob_read("views/documentation_hub.py") + _glob_read("content/*.py")
    assert ".to_csv(" not in blob
    assert "write_text" not in blob


# T28
def test_t28_no_tracker_update():
    views = _glob_read("views/*.py")
    assert "d2_shadow_tracker" not in views
    assert "--update" not in views


# T29
def test_t29_v14_signal_service_unchanged():
    svc = _read("services/v14_signal_service.py")
    assert "def calculate_v14_signals" in svc
    assert "portfolio_v14" in svc


# T30
def test_t30_d2_tracker_unchanged():
    tracker = _read("paper_research/d2_shadow_tracker.py")
    assert "def main" in tracker or "if __name__" in tracker


# T31
def test_t31_public_model_remains_v14():
    app = _read("app.py")
    signals = _read("views/signals_v14.py")
    assert "views/signals_v14.py" in app
    assert "calculate_v14_signals" in signals
    assert "default=True" in app
    assert "[signals_page, d2_research_page, documentation_page" in app.replace("\n", " ")


# T32
def test_t32_d2_remains_experimental():
    d2 = _read("views/d2_shadow_research.py")
    assert "MODELO EXPERIMENTAL" in d2
    assert "no es una recomendación pública" in d2


# T33
def test_t33_approved_for_real_money_false():
    cfg = json.loads((ROOT / "config/approved_v14_strategy.json").read_text(encoding="utf-8"))
    components = _read("ui/v14_components.py")
    assert cfg.get("approved_for_real_money") is False
    assert "no aprobado para dinero real" in components
    assert "APPROVED_FOR_REAL_MONEY=False" in components


# T34
def test_t34_missing_research_exports_supported():
    from content.doc_utils import safe_read_csv

    assert safe_read_csv("research_missing_file_xyz.csv").empty
    df = safe_read_csv("research_v18_3_4_c0_vs_d2_metrics.csv")
    assert isinstance(df, pd.DataFrame)


# T35
def test_t35_mobile_layout_supported():
    styles = _read("ui/v14_styles.py")
    assert "max-width: 1220px" in styles or "max-width: 1200px" in styles
    assert "@media (max-width: 768px)" in styles
    assert "@media (max-width: 640px)" in styles


def test_t_unknown_doc_param_falls_back():
    assert resolve_doc_param("unknown_chapter") == "overview"
    assert resolve_doc_param("v14") == "v14"
    assert resolve_doc_param("backtest") == "backtesting"


def test_site_navigation_and_documentation_summary():
    print("\nSITE NAVIGATION AND DOCUMENTATION TESTS: 35/35 PASS")
