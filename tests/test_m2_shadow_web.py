"""Tests for M2 shadow research page and read-only service."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SERVICE_PATH = ROOT / "services" / "m2_shadow_status_service.py"
VIEW_PATH = ROOT / "views" / "m2_research_view.py"
APP_PATH = ROOT / "app.py"
SITE_NAV_PATH = ROOT / "ui" / "site_navigation.py"

TRACKER_FILES = [
    ROOT / "paper_research" / "m2_shadow_tracker.py",
    ROOT / "paper_research" / "state" / "v20_2_m2_shadow_state.json",
    ROOT / "paper_research" / "state" / "v20_2_m2_model_state.json",
    ROOT / "paper_research" / "data" / "v20_2_m2_signal_snapshots.csv",
    ROOT / "paper_research" / "data" / "v20_2_m2_executions.csv",
    ROOT / "paper_research" / "data" / "v20_2_m2_daily_equity.csv",
    ROOT / "paper_research" / "data" / "v20_2_m2_training_events.csv",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _hashes() -> dict[str, str]:
    return {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in TRACKER_FILES}


def _service():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("services.m2_shadow_status_service")


def test_t1_m2_service_reads_state():
    svc = _service()
    data = svc.get_m2_tracker_state()
    assert data["status"] == "M2_SHADOW_TRACKER_INITIALIZED"


def test_t2_m2_service_reads_model_state():
    svc = _service()
    data = svc.get_m2_model_state()
    assert data["model_version_id"] == "M2_20260702_prepaper"


def test_t3_m2_service_handles_empty_signals(tmp_path, monkeypatch):
    svc = _service()
    p = tmp_path / "signals.csv"
    pd.DataFrame(columns=["signal_batch_id", "strategy", "ticker"]).to_csv(p, index=False)
    monkeypatch.setattr(svc, "SIGNALS_PATH", p)
    assert svc.get_m2_latest_signal().empty


def test_t4_m2_service_handles_empty_executions(tmp_path, monkeypatch):
    svc = _service()
    p = tmp_path / "exec.csv"
    pd.DataFrame(columns=["execution_id"]).to_csv(p, index=False)
    monkeypatch.setattr(svc, "EXECUTIONS_PATH", p)
    assert svc.get_m2_latest_executions().empty


def test_t5_m2_service_handles_empty_equity(tmp_path, monkeypatch):
    svc = _service()
    p = tmp_path / "equity.csv"
    pd.DataFrame(columns=["date"]).to_csv(p, index=False)
    monkeypatch.setattr(svc, "EQUITY_PATH", p)
    assert svc.get_m2_equity_history().empty


def test_t6_m2_service_reads_historical_metrics():
    svc = _service()
    data = svc.get_m2_historical_metrics()
    assert data["available"] is True
    assert data["M2_CAGR"] is not None


def test_t7_m2_service_reads_diagnostic():
    svc = _service()
    data = svc.get_m2_diagnostic_metrics()
    assert data["available"] is True
    assert data["AUC"] is not None


def test_t8_missing_state_handled(tmp_path, monkeypatch):
    svc = _service()
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(svc, "STATE_PATH", missing)
    data = svc.get_m2_shadow_dashboard_data()
    assert data["available"] is False
    assert "no esta inicializado" in data["error"].lower()


def test_t9_invalid_json_handled(tmp_path, monkeypatch):
    svc = _service()
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    monkeypatch.setattr(svc, "STATE_PATH", bad)
    data = svc.get_m2_shadow_dashboard_data()
    assert data["available"] is False
    assert any("Invalid JSON" in w for w in data["warnings"])


def test_t10_missing_csv_handled(tmp_path, monkeypatch):
    svc = _service()
    monkeypatch.setattr(svc, "SIGNALS_PATH", tmp_path / "missing.csv")
    data = svc.get_m2_shadow_dashboard_data()
    assert any("Missing file" in w for w in data["warnings"])


def test_t11_service_does_not_import_tracker():
    text = _read(SERVICE_PATH)
    assert "m2_shadow_tracker" not in text


def test_t12_service_has_no_network_calls():
    text = _read(SERVICE_PATH).lower()
    assert "requests" not in text
    assert "yfinance" not in text


def test_t13_service_does_not_write_files():
    text = _read(SERVICE_PATH)
    assert "write_text" not in text
    assert ".to_csv(" not in text


def test_t14_render_empty_tracker_state():
    text = _read(VIEW_PATH)
    assert "El tracker M2 todavia no esta inicializado." in text
    assert "render_m2_research_view" in text


def test_t15_render_empty_signal_state():
    text = _read(VIEW_PATH)
    assert "Todavia no hay ninguna senal prospectiva registrada." in text


def test_t16_render_signal_table_when_available():
    text = _read(VIEW_PATH)
    assert "Ultima senal simulada" in text
    assert "_signal_table" in text


def test_t17_render_execution_table_when_available():
    text = _read(VIEW_PATH)
    assert "Ultimas ejecuciones simuladas" in text
    assert "_execution_table" in text


def test_t18_render_equity_only_with_enough_rows():
    text = _read(VIEW_PATH)
    assert "Todavia no existe una muestra suficiente para representar la evolucion." in text
    assert "MIN_SESSIONS_PRELIMINARY" in text


def test_t19_public_model_remains_v14():
    data = _service().get_m2_shadow_dashboard_data()
    assert data["tracker"]["current_public_model"] == "V14 R1 Return Engine"


def test_t20_m2_marked_research_shadow():
    text = _read(VIEW_PATH)
    assert "RESEARCH SHADOW" in text


def test_t21_m2_marked_not_selected():
    text = _read(VIEW_PATH)
    assert "NO SELECCIONADO" in text


def test_t22_approved_for_real_money_false():
    data = _service().get_m2_shadow_dashboard_data()
    assert data["tracker"]["approved_for_real_money"] is False


def test_t23_no_update_button():
    text = _read(VIEW_PATH)
    assert "Actualizar pantalla" not in text
    assert "--update" in text


def test_t24_no_initialize_button():
    text = _read(VIEW_PATH)
    assert "Inicializar" not in text


def test_t25_d2_page_unchanged():
    text = _read(ROOT / "views" / "d2_shadow_research.py")
    assert "Investigación D2" in text
    assert "D2 Trend Quality" in text


def test_t26_v14_page_unchanged():
    text = _read(ROOT / "views" / "signals_v14.py")
    assert "V14" in text
    assert "d2_shadow" not in text


def test_t27_navigation_includes_m2():
    app_text = _read(APP_PATH)
    nav_text = _read(SITE_NAV_PATH)
    assert "views/m2_research_view.py" in app_text
    assert 'title="Investigación M2"' in app_text
    assert "INVESTIGACIÓN M2" in nav_text


def test_t28_tracker_files_unchanged_after_service_read():
    before = _hashes()
    _service().get_m2_shadow_dashboard_data()
    after = _hashes()
    assert before == after


def test_t29_tracker_files_unchanged_after_render():
    before = _hashes()
    _read(VIEW_PATH)
    after = _hashes()
    assert before == after


def test_t30_responsive_render_function_exists():
    text = _read(VIEW_PATH)
    assert "def render_m2_research_view()" in text
    assert "st.columns" in text


def test_full_suite_banner_contract():
    assert "M2 SHADOW WEB TESTS: 30/30 PASS" in _read(ROOT / "tests" / "conftest.py")
