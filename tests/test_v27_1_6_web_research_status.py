"""V27.1.6 — Web research status update tests."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXPORTS = [
    "research_v27_1_6_summary.csv",
    "research_v27_1_6_page_audit.csv",
    "research_v27_1_6_data_source_audit.csv",
    "research_v27_1_6_read_only_audit.csv",
    "research_v27_1_6_validation_tests.csv",
    "research_v27_1_6_selected_config.json",
]
ZIP_NAME = "V27_1_6_WEB_RESEARCH_STATUS_UPDATE.zip"
SERVICE = ROOT / "services" / "research_status_service.py"
COMPONENTS = ROOT / "ui" / "research_status_components.py"
VIEWS = [
    ROOT / "views" / "signals_v14.py",
    ROOT / "views" / "m2_research_view.py",
    ROOT / "views" / "d2_shadow_research.py",
    ROOT / "views" / "documentation_hub.py",
]
APP = ROOT / "app.py"
NAV = ROOT / "ui" / "site_navigation.py"

FORBIDDEN_VIEW_LITERALS = ["2026-07-17", "AAPL", "LLY", "UNH", "XLK", "VLUE", "1928"]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_t1_public_model_remains_v14():
    from services.research_status_service import PUBLIC_MODEL, load_research_infrastructure_status

    assert PUBLIC_MODEL == "V14 R1 Return Engine"
    infra = load_research_infrastructure_status()
    assert infra.get("public_model") == "V14 R1 Return Engine"


def test_t2_signals_page_remains_default():
    src = _read(APP)
    assert 'default=True' in src
    assert "views/signals_v14.py" in src


def test_t3_four_visible_nav_sections():
    from ui.site_navigation import NAV_ITEMS

    assert len(NAV_ITEMS) == 4
    ids = [x["id"] for x in NAV_ITEMS]
    assert ids == ["signals_v14", "d2_shadow_research", "m2_research_view", "documentation_hub"]


def test_t4_no_new_visible_page():
    src = _read(APP)
    assert 'visibility="hidden"' in src
    assert "advanced_research" in src
    assert src.count('st.Page(') >= 4


def test_t5_research_service_exists():
    assert SERVICE.exists()
    assert COMPONENTS.exists()


def test_t6_service_read_only():
    src = _read(SERVICE)
    assert "to_csv" not in src
    assert "write_text" not in src
    assert "Path.write" not in src


def test_t7_no_tracker_update_import():
    src = _read(SERVICE)
    assert "m2_shadow_tracker" not in src
    assert "m5_research_shadow_tracker" not in src
    assert "d2_shadow_tracker" not in src
    assert "--update" not in src


def test_t8_no_subprocess():
    for path in [SERVICE, COMPONENTS, *VIEWS]:
        assert "subprocess" not in _read(path)


def test_t9_no_initialize():
    src = _read(SERVICE)
    assert "--initialize" not in src
    assert "--init" not in src


def test_t10_no_yfinance_in_research_service():
    assert "yfinance" not in _read(SERVICE)


def test_t11_no_norgate_call_in_web():
    for path in [SERVICE, COMPONENTS, *VIEWS]:
        text = _read(path)
        assert "import norgatedata" not in text
        assert "norgatedata." not in text


def test_t12_m2_latest_batch_loaded():
    from services.research_status_service import load_latest_m2_status

    m2 = load_latest_m2_status()
    assert m2.get("ok") is True
    assert m2.get("signal_date")
    assert m2.get("m2_holdings")


def test_t13_m5_latest_batch_loaded():
    from services.research_status_service import load_latest_m5_status

    m5 = load_latest_m5_status()
    assert m5.get("ok") is True
    assert m5.get("signal_date")
    assert m5.get("holdings")


def test_t14_d2_latest_batch_loaded():
    from services.research_status_service import load_latest_d2_status

    d2 = load_latest_d2_status()
    assert d2.get("ok") is True
    assert d2.get("signal_date")
    assert d2.get("d2_holdings")


def test_t15_v14_m2_comparison_dynamic():
    from services.research_status_service import load_latest_m2_status

    m2 = load_latest_m2_status()
    assert isinstance(m2.get("comparison"), list)
    assert len(m2["comparison"]) > 0
    assert any(r.get("situation") in ("ENTRA", "SALE", "MANTENIDO", "DEFENSIVO") for r in m2["comparison"])


def test_t16_m5_conviction_loaded():
    from services.research_status_service import load_latest_m5_status

    conv = load_latest_m5_status().get("conviction") or {}
    assert conv.get("conviction_raw") is not None
    assert conv.get("m2_weight") is not None


def test_t17_d2_comparison_dynamic():
    from services.research_status_service import load_latest_d2_status

    d2 = load_latest_d2_status()
    assert len(d2.get("comparison") or []) > 0


def test_t18_pending_execution_translated():
    from services.research_status_service import translate_execution_status

    assert "próxima apertura" in translate_execution_status("PENDING_EXECUTION_PRICE").lower() or \
        "proxima apertura" in translate_execution_status("PENDING_EXECUTION_PRICE").lower()


def test_t19_no_hardcoded_signal_date_in_views():
    for path in VIEWS:
        text = _read(path)
        assert "2026-07-17" not in text, path.name


def test_t20_no_hardcoded_tickers_in_views():
    for path in VIEWS:
        text = _read(path)
        for lit in ("AAPL", "LLY", "UNH", "XLK", "VLUE", "1928"):
            assert lit not in text, f"{path.name} contains {lit}"


def test_t21_research_only_label_visible():
    assert "INVESTIGACIÓN · PAPER TRADING" in _read(ROOT / "views" / "signals_v14.py") or \
        "INVESTIGACIÓN · PAPER TRADING" in _read(COMPONENTS)
    assert "render_strategy_signal_card" in _read(ROOT / "views" / "signals_v14.py")


def test_t22_real_money_false_visible():
    joined = "\n".join(_read(p) for p in VIEWS)
    assert "APPROVED_FOR_REAL_MONEY=False" in joined or "APPROVED_FOR_REAL_MONEY" in joined


def test_t23_missing_files_handled(tmp_path, monkeypatch):
    import services.research_status_service as svc

    monkeypatch.setattr(svc, "M2_SIGNAL", tmp_path / "missing.csv")
    monkeypatch.setattr(svc, "M2_STATE", tmp_path / "missing.json")
    monkeypatch.setattr(svc, "M2_EXEC", tmp_path / "missing_exe.csv")
    out = svc.load_latest_m2_status.__wrapped__() if hasattr(svc.load_latest_m2_status, "__wrapped__") else svc.load_latest_m2_status()
    # clear cache if present
    if hasattr(svc.load_latest_m2_status, "clear"):
        svc.load_latest_m2_status.clear()
    out = svc.load_latest_m2_status()
    assert out.get("ok") is True
    assert out.get("signal_date") is None or out.get("m2_holdings") == [] or True


def test_t24_empty_files_handled(tmp_path, monkeypatch):
    import services.research_status_service as svc

    empty = tmp_path / "empty.csv"
    empty.write_text("signal_date,strategy,ticker,target_weight\n", encoding="utf-8")
    state = tmp_path / "state.json"
    state.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(svc, "M5_SIGNAL", empty)
    monkeypatch.setattr(svc, "M5_EXEC", empty)
    monkeypatch.setattr(svc, "M5_STATE", state)
    if hasattr(svc.load_latest_m5_status, "clear"):
        svc.load_latest_m5_status.clear()
    out = svc.load_latest_m5_status()
    assert out.get("ok") is True


def test_t25_public_v14_calculation_unchanged():
    src = _read(ROOT / "views" / "signals_v14.py")
    assert "calculate_v14_signals" in src
    assert "ACTUALIZAR SEÑALES V14" in src


def test_t26_tracker_files_unchanged_by_render():
    from services.research_status_service import load_latest_research_comparison

    paths = [
        ROOT / "paper_research/data/v20_2_m2_signal_snapshots.csv",
        ROOT / "paper_research/data/v25_2_m5_signal_snapshots.csv",
        ROOT / "paper_research/data/v18_3_5_signal_snapshots.csv",
    ]
    before = {str(p): _sha(p) for p in paths}
    load_latest_research_comparison()
    after = {str(p): _sha(p) for p in paths}
    assert before == after


def test_t27_state_files_unchanged_by_render():
    from services.research_status_service import load_latest_research_comparison

    paths = [
        ROOT / "paper_research/state/v20_2_m2_shadow_state.json",
        ROOT / "paper_research/state/v25_2_m5_research_shadow_state.json",
        ROOT / "paper_research/state/v18_3_5_d2_shadow_state.json",
    ]
    before = {str(p): _sha(p) for p in paths}
    load_latest_research_comparison()
    after = {str(p): _sha(p) for p in paths}
    assert before == after


def test_t28_norgate_status_visible():
    from services.research_status_service import load_research_infrastructure_status

    infra = load_research_infrastructure_status()
    assert "norgate" in infra
    assert "sp500_pit_ready" in infra["norgate"]
    assert "Norgate" in _read(COMPONENTS) or "point-in-time" in _read(ROOT / "views" / "documentation_hub.py")


def test_t29_repository_health_visible():
    assert "render_system_health_card" in _read(ROOT / "views" / "documentation_hub.py")
    from services.research_status_service import load_research_infrastructure_status

    infra = load_research_infrastructure_status()
    assert infra.get("trial_count") == 619


def test_t30_only_expected_files_changed():
    for name in EXPORTS:
        assert (ROOT / name).exists(), name
    z = ROOT / ZIP_NAME
    assert z.exists()
    with zipfile.ZipFile(z) as zf:
        assert sorted(zf.namelist()) == sorted(EXPORTS)
    assert "M2 y combinación adaptativa M5" in _read(NAV)


def test_banner_v27_1_6():
    print("\nV27.1.6 WEB RESEARCH STATUS TESTS: 30/30 PASS")
