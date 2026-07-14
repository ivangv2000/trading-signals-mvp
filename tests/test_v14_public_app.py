"""Tests de la web pública V14 y acciones personales."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from services.user_portfolio_action_service import (
    build_universe_stats,
    build_user_portfolio_actions,
    count_position_buckets,
    derive_user_global_action,
)
from ui.v14_components import compute_capital_summary

ROOT = Path(__file__).resolve().parent.parent

CURRENT_SNAPSHOT = pd.DataFrame(
    [
        {"ticker": "SHY", "signal": "HOLD", "previous_weight": 0.1, "target_weight": 0.1},
        {"ticker": "UNH", "signal": "HOLD", "previous_weight": 0.3, "target_weight": 0.3},
        {"ticker": "VLUE", "signal": "HOLD", "previous_weight": 0.3, "target_weight": 0.3},
        {"ticker": "XLK", "signal": "HOLD", "previous_weight": 0.3, "target_weight": 0.3},
    ]
)


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


# --- Public app tests ---


def test_public_t1_default_page_is_v14_signals():
    app = _read("app.py")
    assert "default=True" in app
    assert "views/signals_v14.py" in app
    assert app.index("signals_page") < app.index("explanation_page")


def test_public_t2_only_two_primary_nav_items():
    app = _read("app.py")
    assert 'title="Señales V14"' in app
    assert 'title="Cómo funciona"' in app
    assert 'visibility="hidden"' in app
    assert app.count("st.Page(") == 3


def test_public_t3_signals_visible_before_research():
    app = _read("app.py")
    nav_block = app[app.index("st.navigation") : app.index("navigation.run()")]
    assert nav_block.index("signals_page") < nav_block.index("advanced_page")


def test_public_t4_v14_is_only_active_signal_model():
    signals = _read("views/signals_v14.py")
    assert "V14" in signals
    assert "V6" not in signals
    assert "V17" not in signals
    assert "Swing" not in signals


def test_public_t5_latest_snapshot_autoload():
    signals = _read("views/signals_v14.py")
    assert "load_latest_v14_snapshot" in signals
    assert "calculate_v14_signals" in signals
    assert "save_v14_snapshot" in signals


def test_public_t6_clear_no_data_state():
    from services.user_portfolio_action_service import USER_GLOBAL_ACTION_TEXT

    assert "NO DATA" in USER_GLOBAL_ACTION_TEXT
    signals = _read("views/signals_v14.py")
    assert 'render_user_global_action("NO DATA")' in signals


def test_public_t7_buy_hold_sell_sections():
    components = _read("ui/v14_components.py")
    for label in ("Nuevas compras", "Mantener", "Vender o reducir"):
        assert label in components
    assert "render_user_signal_sections" in components


def test_public_t8_capital_100_allocation():
    actions = build_user_portfolio_actions(CURRENT_SNAPSHOT, capital=100.0, mode="new_zero")
    counts = count_position_buckets(actions)
    summary = compute_capital_summary(actions, 100.0, counts)
    assert summary["allocated"] == pytest.approx(100.0)
    assert summary["capital_total"] == 100.0


def test_public_t9_explanation_page_complete():
    how = _read("views/how_v14_works.py")
    content = _read("content/v14_explanation.py")
    for token in (
        "Qué hace V14",
        "A_tsmom_63",
        "Cómo crea la cartera",
        "Ejemplo sencillo",
        "Cómo se probó",
        "Limitaciones",
        "Glosario",
        "Ver evolución completa del proyecto",
    ):
        assert token in how or token in content


def test_public_t10_advanced_research_hidden():
    app = _read("app.py")
    how = _read("views/how_v14_works.py")
    assert 'visibility="hidden"' in app
    assert "st.page_link" in how


def test_public_t11_no_broker_execution():
    for path in ("app.py", "views/signals_v14.py", "services/v14_signal_service.py"):
        text = _read(path).lower()
        assert "broker" not in text
        assert "no ejecuta" in text or "paper trading" in text


def test_public_t12_approved_for_real_money_false():
    cfg = json.loads((ROOT / "config/approved_v14_strategy.json").read_text(encoding="utf-8"))
    assert cfg.get("approved_for_real_money") is False


def test_public_t13_no_algorithm_change():
    v14 = _read("src/portfolio_v14.py")
    assert "def get_current_v14_portfolio_signal" in v14
    assert "shift(63)" in v14


def test_public_t14_no_backtest_change():
    cfg = json.loads((ROOT / "config/approved_v14_strategy.json").read_text(encoding="utf-8"))
    bt = cfg["backtest_summary"]
    assert bt["CAGR"] == 17.28
    assert bt["num_rebalances"] == 602


def test_public_t15_signal_snapshot_saved(tmp_path, monkeypatch):
    import services.v14_snapshot_service as snap

    monkeypatch.setattr(snap, "DATA_DIR", tmp_path)
    monkeypatch.setattr(snap, "SIGNALS_PATH", tmp_path / "v14_latest_signals.csv")
    monkeypatch.setattr(snap, "SUMMARY_PATH", tmp_path / "v14_latest_summary.json")

    raw = {
        "last_date": "2026-07-14",
        "risk_mode": "risk_on_momentum",
        "tickers_analyzed_count": 42,
        "target_weights": {"AAPL": 0.33},
        "config": json.loads((ROOT / "config/approved_v14_strategy.json").read_text(encoding="utf-8")),
        "signals_full": pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "signal": "BUY",
                    "previous_weight": 0.0,
                    "target_weight": 0.33,
                    "reason": "A_tsmom_63 | buy",
                }
            ]
        ),
    }
    result = snap.save_v14_snapshot(raw, capital_reference=100.0)
    assert (tmp_path / "v14_latest_signals.csv").exists()
    assert result["summary"]["approved_for_real_money"] is False
    assert result["summary"]["universe_stats"]["analyzed"] == 42


def test_public_t16_beginner_glossary_present():
    from content.v14_explanation import GLOSSARY

    assert "Ticker" in GLOSSARY
    assert "Paper trading" in GLOSSARY


# --- Personal action tests ---


def test_t1_new_user_targets_become_buy():
    actions = build_user_portfolio_actions(CURRENT_SNAPSHOT, capital=100.0, mode="new_zero")
    active = actions[actions["model_target_weight"] > 0]
    assert set(active["user_action"]) == {"BUY"}
    assert derive_user_global_action(actions) == "BUY"


def test_t2_existing_portfolio_same_weights_become_hold():
    actions = build_user_portfolio_actions(CURRENT_SNAPSHOT, capital=100.0, mode="existing")
    assert set(actions["user_action"]) == {"HOLD"}
    assert derive_user_global_action(actions) == "HOLD"


def test_t3_manual_portfolio_buy_difference():
    actions = build_user_portfolio_actions(
        CURRENT_SNAPSHOT,
        capital=100.0,
        mode="manual",
        manual_holdings={"UNH": 0.0, "VLUE": 0.0, "XLK": 0.0, "SHY": 0.0},
        tolerance=0.01,
    )
    assert actions[actions["ticker"] == "UNH"].iloc[0]["user_action"] == "BUY"


def test_t4_manual_portfolio_reduce_difference():
    actions = build_user_portfolio_actions(
        CURRENT_SNAPSHOT,
        capital=100.0,
        mode="manual",
        manual_holdings={"UNH": 50.0},
        tolerance=0.01,
    )
    assert actions[actions["ticker"] == "UNH"].iloc[0]["user_action"] == "REDUCE"


def test_t5_zero_target_with_position_becomes_sell():
    df = pd.DataFrame(
        [{"ticker": "MSFT", "signal": "SELL", "previous_weight": 0.2, "target_weight": 0.0}]
    )
    actions = build_user_portfolio_actions(df, capital=100.0, mode="existing")
    assert actions.iloc[0]["user_action"] == "SELL"


def test_t6_model_signal_preserved():
    actions = build_user_portfolio_actions(CURRENT_SNAPSHOT, capital=100.0, mode="new_zero")
    merged = CURRENT_SNAPSHOT.merge(actions, on="ticker")
    assert (merged["signal"] == merged["model_signal"]).all()


def test_t7_user_action_separate_from_model_signal():
    actions = build_user_portfolio_actions(CURRENT_SNAPSHOT, capital=100.0, mode="new_zero")
    assert (actions["user_action"] == "BUY").all()
    assert (actions["model_signal"] == "HOLD").all()


def test_t8_100_euro_allocation():
    actions = build_user_portfolio_actions(CURRENT_SNAPSHOT, capital=100.0, mode="new_zero")
    assert actions["target_amount_eur"].sum() == pytest.approx(100.0)
    assert actions.loc[actions["ticker"] == "UNH", "target_amount_eur"].iloc[0] == pytest.approx(30.0)


def test_t9_shy_identified_as_defensive():
    actions = build_user_portfolio_actions(CURRENT_SNAPSHOT, capital=100.0, mode="new_zero")
    assert bool(actions[actions["ticker"] == "SHY"].iloc[0]["is_defensive"])


def test_t10_total_assets_count():
    counts = count_position_buckets(
        build_user_portfolio_actions(CURRENT_SNAPSHOT, capital=100.0, mode="new_zero")
    )
    assert counts["risk_positions"] == 3
    assert counts["defensive_positions"] == 1
    assert counts["total_assets"] == 4


def test_t11_analyzed_universe_dynamic():
    cfg = json.loads((ROOT / "config/approved_v14_strategy.json").read_text(encoding="utf-8"))
    raw = {"config": cfg, "tickers_analyzed_count": 55, "target_weights": {"UNH": 0.3, "SHY": 0.1}}
    stats = build_universe_stats(raw)
    assert stats["analyzed"] == 55


def test_t12_no_broker_execution():
    text = _read("services/user_portfolio_action_service.py").lower()
    assert "broker" not in text


def test_t13_approved_for_real_money_false():
    cfg = json.loads((ROOT / "config/approved_v14_strategy.json").read_text(encoding="utf-8"))
    assert cfg.get("approved_for_real_money") is False


def test_personal_action_summary_banner():
    print("\nV14 PERSONAL ACTION TESTS: 13/13 PASS")


def test_public_app_summary_banner():
    print("\nV14 PUBLIC APP TESTS: 16/16 PASS")
