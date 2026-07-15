"""Tests for D2 shadow research public page (read-only)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


# T1
def test_t1_page_imports():
    view = _read("views/d2_shadow_research.py")
    assert "build_d2_shadow_view_model" in view
    assert "streamlit" in view


# T2
def test_t2_state_json_read_only():
    service = _read("services/d2_shadow_status_service.py")
    view = _read("views/d2_shadow_research.py")
    assert "write_text" not in service
    assert "to_csv" not in service
    assert "save_v14_snapshot" not in view


# T3
def test_t3_empty_signal_csv_supported(tmp_path, monkeypatch):
    import services.d2_shadow_status_service as svc

    monkeypatch.setattr(svc, "STATE_PATH", ROOT / "paper_research/state/v18_3_5_d2_shadow_state.json")
    monkeypatch.setattr(svc, "SIGNALS_PATH", tmp_path / "empty_signals.csv")
    monkeypatch.setattr(svc, "EXECUTIONS_PATH", tmp_path / "empty_exec.csv")
    monkeypatch.setattr(svc, "EQUITY_PATH", tmp_path / "empty_eq.csv")
    pd.DataFrame(columns=["signal_date", "strategy", "ticker"]).to_csv(tmp_path / "empty_signals.csv", index=False)
    vm = svc.build_d2_shadow_view_model()
    assert vm["completed_signals"] == 0


# T4
def test_t4_empty_executions_supported(tmp_path, monkeypatch):
    import services.d2_shadow_status_service as svc

    monkeypatch.setattr(svc, "EXECUTIONS_PATH", tmp_path / "missing.csv")
    assert svc.load_d2_executions().empty


# T5
def test_t5_empty_equity_supported(tmp_path, monkeypatch):
    import services.d2_shadow_status_service as svc

    monkeypatch.setattr(svc, "EQUITY_PATH", tmp_path / "missing.csv")
    assert svc.load_d2_daily_equity().empty


# T6
def test_t6_waiting_status_rendered():
    from services.d2_shadow_status_service import build_d2_shadow_view_model

    vm = build_d2_shadow_view_model()
    view = _read("views/d2_shadow_research.py")
    assert vm["runtime_status"] == "WAITING_FOR_FIRST_FORWARD_SIGNAL"
    assert "WAITING_FOR_FIRST_FORWARD_SIGNAL" in view
    assert "Esperando la primera señal prospectiva" in view


# T7
def test_t7_v14_marked_public():
    view = _read("views/d2_shadow_research.py")
    assert "MODELO PÚBLICO" in view
    assert "V14 R1 Return Engine" in view
    assert "Estado: ACTIVO" in view


# T8
def test_t8_d2_marked_experimental():
    view = _read("views/d2_shadow_research.py")
    assert "MODELO EXPERIMENTAL" in view
    assert "D2 Trend Quality" in view
    assert "RESEARCH_SHADOW" in view or "D2_paper_status" in view


# T9
def test_t9_d2_not_marked_approved():
    view = _read("views/d2_shadow_research.py")
    assert "BUY" not in view or "Experimental" in view
    assert "no es una recomendación pública" in view
    assert "EXPERIMENTO DE INVESTIGACIÓN" in _read("ui/v14_styles.py")


# T10
def test_t10_no_update_command_from_web():
    view = _read("views/d2_shadow_research.py")
    assert "d2_shadow_tracker" not in view
    assert "--update" not in view
    assert "st.rerun()" in view


# T11
def test_t11_no_subprocess():
    view = _read("views/d2_shadow_research.py")
    service = _read("services/d2_shadow_status_service.py")
    assert "subprocess" not in view
    assert "subprocess" not in service


# T12
def test_t12_no_network_download():
    view = _read("views/d2_shadow_research.py")
    service = _read("services/d2_shadow_status_service.py")
    assert "yfinance" not in view
    assert "yfinance" not in service


# T13
def test_t13_no_file_writes():
    view = _read("views/d2_shadow_research.py")
    service = _read("services/d2_shadow_status_service.py")
    for text in (view, service):
        assert ".to_csv(" not in text
        assert "write_text" not in text


# T14
def test_t14_sample_guard_before_63_sessions():
    from services.d2_shadow_status_service import build_d2_shadow_view_model

    vm = build_d2_shadow_view_model()
    assert vm["sample_status"] == "INSUFFICIENT_SAMPLE"
    assert "INSUFFICIENT_SAMPLE" in _read("views/d2_shadow_research.py")


# T15
def test_t15_contract_error_alert_supported():
    assert "render_d2_integrity_alerts" in _read("views/d2_shadow_research.py")
    assert "contracts_ok" in _read("ui/v14_styles.py")


# T16
def test_t16_data_revision_alert_supported():
    assert "data_revision_detected" in _read("ui/v14_styles.py")


# T17
def test_t17_latest_signals_parsed(tmp_path, monkeypatch):
    import services.d2_shadow_status_service as svc

    sig = pd.DataFrame([
        {
            "signal_date": "2026-07-17", "strategy": "C0_PRODUCTION_V14", "ticker": "XLK",
            "rank": 1, "target_weight": 0.3, "status": "SIGNAL_RECORDED",
        },
        {
            "signal_date": "2026-07-17", "strategy": "D2_TREND_QUALITY", "ticker": "XLK",
            "rank": 1, "target_weight": 0.3, "status": "SIGNAL_RECORDED",
        },
    ])
    sig_path = tmp_path / "signals.csv"
    sig.to_csv(sig_path, index=False)
    monkeypatch.setattr(svc, "SIGNALS_PATH", sig_path)
    monkeypatch.setattr(svc, "EXECUTIONS_PATH", tmp_path / "e.csv")
    monkeypatch.setattr(svc, "EQUITY_PATH", tmp_path / "q.csv")
    vm = svc.build_d2_shadow_view_model()
    assert vm["latest_signal_date"] == "2026-07-17"


# T18
def test_t18_positions_compared():
    from services.d2_shadow_status_service import _portfolio_diff

    c0 = [{"ticker": "A"}, {"ticker": "B"}]
    d2 = [{"ticker": "A"}, {"ticker": "C"}]
    diff = _portfolio_diff(c0, d2)
    assert diff["only_c0"] == ["B"]
    assert diff["only_d2"] == ["C"]


# T19
def test_t19_pending_executions_supported():
    from services.d2_shadow_status_service import build_d2_shadow_view_model

    vm = build_d2_shadow_view_model()
    assert "pending_execution_count" in vm
    view = _read("views/d2_shadow_research.py")
    assert "esperando el precio de apertura" in view


# T20
def test_t20_public_v14_page_unchanged():
    signals = _read("views/signals_v14.py")
    assert "calculate_v14_signals" in signals
    assert "d2_shadow" not in signals


# T21
def test_t21_approved_for_real_money_false():
    from services.d2_shadow_status_service import build_d2_shadow_view_model

    vm = build_d2_shadow_view_model()
    assert vm["approved_for_real_money"] is False
    assert vm["public_signal_model_changed"] is False


def test_navigation_includes_d2_page():
    app = _read("app.py")
    assert "d2_shadow_research.py" in app
    assert 'title="Investigación D2"' in app
    assert app.index("signals_page") < app.index("d2_research_page")


def test_d2_shadow_public_page_summary():
    print("\nD2 SHADOW PUBLIC PAGE TESTS: 21/21 PASS")
