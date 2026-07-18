"""Read-only service for D2 shadow paper tracker public page."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PAPER_ROOT = ROOT / "paper_research"
STATE_PATH = PAPER_ROOT / "state" / "v18_3_5_d2_shadow_state.json"
SIGNALS_PATH = PAPER_ROOT / "data" / "v18_3_5_signal_snapshots.csv"
EXECUTIONS_PATH = PAPER_ROOT / "data" / "v18_3_5_executions.csv"
EQUITY_PATH = PAPER_ROOT / "data" / "v18_3_5_daily_equity.csv"

CONTRACT_V18_3_2 = ROOT / "config" / "v18_3_2_production_v14_contract.json"
CONFIG_V18_3_3 = ROOT / "config" / "v18_3_3_production_baseline.json"
CONFIG_V18_3_4 = ROOT / "research_v18_3_4_selected_config.json"

C0_STRATEGY = "C0_PRODUCTION_V14"
D2_STRATEGY = "D2_TREND_QUALITY"
PUBLIC_MODEL = "V14 R1 Return Engine"
EXPERIMENTAL_MODEL = "D2 Trend Quality"
PAPER_START_DATE = "2026-07-15"
MIN_SESSIONS_ANNUALIZED = 63

EXPECTED_DATA_BUNDLE_HASH = "9d137c8f317bbab03b04a14caf677af551e6d1492d44ea78948190157f1c316a"
EXPECTED_CONTRACT_HASH = "4b71a0aae3f02d3103574b7c49fce7a112b5fffc2eb819dcd483b30dc788278d"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame()


def verify_contracts() -> tuple[bool, str]:
    if not all(p.exists() for p in (CONTRACT_V18_3_2, CONFIG_V18_3_3, CONFIG_V18_3_4)):
        return False, "missing contract files"
    contract = _read_json(CONTRACT_V18_3_2)
    cfg33 = _read_json(CONFIG_V18_3_3)
    cfg34 = _read_json(CONFIG_V18_3_4)
    if int(contract.get("ticker_count", 0)) != 49:
        return False, "universe not 49"
    if contract.get("universe_sha256") != EXPECTED_CONTRACT_HASH:
        return False, "contract hash mismatch"
    if cfg33.get("production_data_bundle_hash") != EXPECTED_DATA_BUNDLE_HASH:
        return False, "data bundle hash mismatch"
    if not cfg33.get("D2_exact_eligibility_contract"):
        return False, "D2 eligibility contract"
    if cfg34.get("status") != "D2_NOT_SELECTED":
        return False, "historical status mismatch"
    if cfg34.get("approved_for_real_money"):
        return False, "approved_for_real_money must be false"
    return True, "ok"


def load_d2_shadow_state() -> dict:
    return _read_json(STATE_PATH)


def load_d2_signal_snapshots() -> pd.DataFrame:
    return _read_csv(SIGNALS_PATH)


def load_d2_executions() -> pd.DataFrame:
    return _read_csv(EXECUTIONS_PATH)


def load_d2_daily_equity() -> pd.DataFrame:
    return _read_csv(EQUITY_PATH)


def _runtime_status(state: dict) -> str:
    if state.get("data_revision_detected"):
        return "DATA_REVISION_DETECTED"
    completed = int(state.get("completed_signals", 0) or 0)
    completed_exec = int(state.get("completed_executions", 0) or 0)
    if completed == 0:
        return "WAITING_FOR_FIRST_FORWARD_SIGNAL"
    if completed_exec == 0:
        return "FORWARD_SIGNAL_RECORDED_PENDING_EXECUTION"
    return "FORWARD_TRACKING_ACTIVE"


def _latest_positions(signals: pd.DataFrame, strategy: str, executions: pd.DataFrame) -> list[dict]:
    if signals.empty or "strategy" not in signals.columns:
        return []
    strat = signals[signals["strategy"].astype(str) == strategy]
    if strat.empty:
        return []
    latest_date = strat["signal_date"].astype(str).max()
    rows = strat[strat["signal_date"].astype(str) == latest_date].copy()
    exec_status = {}
    if not executions.empty:
        ex = executions[
            (executions["strategy"].astype(str) == strategy)
            & (executions["signal_date"].astype(str) == latest_date)
        ]
        for _, r in ex.iterrows():
            exec_status[str(r["ticker"])] = str(r.get("execution_status", ""))
    out = []
    for _, r in rows.sort_values("rank").iterrows():
        out.append({
            "ticker": str(r["ticker"]),
            "target_weight": float(r["target_weight"]) if pd.notna(r.get("target_weight")) else None,
            "rank": int(r["rank"]) if pd.notna(r.get("rank")) else None,
            "execution_status": exec_status.get(str(r["ticker"]), "PENDING"),
            "signal_date": latest_date,
        })
    return out


def _paper_metrics(equity: pd.DataFrame, state: dict) -> dict:
    start = pd.Timestamp(PAPER_START_DATE)
    result = {
        "sample_status": "INSUFFICIENT_SAMPLE",
        "n_sessions": 0,
        "C0_paper_return": None,
        "D2_paper_return": None,
        "D2_active_return": None,
        "C0_max_drawdown": None,
        "D2_max_drawdown": None,
        "information_ratio": None,
        "paper_costs": None,
        "paper_turnover": None,
        "completed_weeks": int(state.get("completed_weeks", 0) or 0),
    }
    if equity.empty or "date" not in equity.columns:
        return result

    eq = equity.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq[eq["date"] >= start]
    n_sessions = int(eq["date"].nunique())
    result["n_sessions"] = n_sessions
    if n_sessions >= MIN_SESSIONS_ANNUALIZED:
        result["sample_status"] = "AVAILABLE"

    def _strategy_metrics(strategy: str) -> dict:
        s = eq[eq["strategy"].astype(str) == strategy].sort_values("date")
        if len(s) < 2:
            return {}
        curve = s.set_index("date")["equity"].astype(float)
        cum = float(s["cumulative_return"].iloc[-1]) if "cumulative_return" in s.columns else None
        mdd = float((curve / curve.cummax() - 1).min() * 100)
        costs = float(s["costs_cumulative"].iloc[-1]) if "costs_cumulative" in s.columns else None
        turnover = float(s["turnover_cumulative"].iloc[-1]) if "turnover_cumulative" in s.columns else None
        return {"cumulative_return": cum, "max_drawdown": mdd, "costs": costs, "turnover": turnover, "curve": curve}

    c0 = _strategy_metrics(C0_STRATEGY)
    d2 = _strategy_metrics(D2_STRATEGY)
    if c0:
        result["C0_paper_return"] = c0.get("cumulative_return")
        result["C0_max_drawdown"] = c0.get("max_drawdown")
    if d2:
        result["D2_paper_return"] = d2.get("cumulative_return")
        result["D2_max_drawdown"] = d2.get("max_drawdown")
        result["paper_costs"] = d2.get("costs")
        result["paper_turnover"] = d2.get("turnover")

    if c0.get("curve") is not None and d2.get("curve") is not None:
        aligned = pd.DataFrame({"c0": c0["curve"], "d2": d2["curve"]}).dropna()
        if len(aligned) >= 2 and float(aligned["c0"].iloc[0]) > 0:
            result["D2_active_return"] = round(
                ((aligned["d2"].iloc[-1] / aligned["d2"].iloc[0])
                 / (aligned["c0"].iloc[-1] / aligned["c0"].iloc[0]) - 1) * 100, 4
            )
            if n_sessions >= MIN_SESSIONS_ANNUALIZED:
                active = aligned["d2"].pct_change().fillna(0) - aligned["c0"].pct_change().fillna(0)
                if active.std() > 0:
                    result["information_ratio"] = round(
                        float(active.mean() / active.std() * np.sqrt(252)), 4
                    )
    return result


def _portfolio_diff(c0_pos: list[dict], d2_pos: list[dict]) -> dict:
    c0_set = {p["ticker"] for p in c0_pos}
    d2_set = {p["ticker"] for p in d2_pos}
    return {
        "same_portfolio": c0_set == d2_set and len(c0_pos) == len(d2_pos),
        "only_c0": sorted(c0_set - d2_set),
        "only_d2": sorted(d2_set - c0_set),
    }


def build_d2_shadow_view_model() -> dict[str, Any]:
    state = load_d2_shadow_state()
    signals = load_d2_signal_snapshots()
    executions = load_d2_executions()
    equity = load_d2_daily_equity()
    contracts_ok, contracts_message = verify_contracts()

    pending_count = 0
    completed_count = 0
    if not executions.empty and "execution_status" in executions.columns:
        pending_count = int((executions["execution_status"] == "PENDING_EXECUTION_PRICE").sum())
        completed_count = int((executions["execution_status"] == "EXECUTED").sum())

    c0_pos = _latest_positions(signals, C0_STRATEGY, executions)
    d2_pos = _latest_positions(signals, D2_STRATEGY, executions)
    metrics = _paper_metrics(equity, state)
    diff = _portfolio_diff(c0_pos, d2_pos)

    latest_signal_date = None
    if not signals.empty and "signal_date" in signals.columns:
        latest_signal_date = str(signals["signal_date"].astype(str).max())

    exec_preview = []
    if not executions.empty:
        preview = executions.sort_values(["execution_date", "strategy", "ticker"], ascending=[False, True, True])
        for _, r in preview.head(20).iterrows():
            exec_preview.append({
                "strategy": str(r.get("strategy", "")),
                "execution_date": str(r.get("execution_date", "")),
                "ticker": str(r.get("ticker", "")),
                "previous_weight": r.get("previous_weight"),
                "target_weight": r.get("target_weight"),
                "weight_change": r.get("weight_change"),
                "execution_open": r.get("execution_open"),
                "estimated_cost": r.get("estimated_cost"),
                "execution_status": str(r.get("execution_status", "")),
            })

    return {
        "tracker_status": state.get("tracker_status", "D2_SHADOW_TRACKER_INITIALIZED"),
        "runtime_status": _runtime_status(state),
        "contracts_ok": contracts_ok,
        "contracts_message": contracts_message,
        "paper_start_date": state.get("paper_start_date", PAPER_START_DATE),
        "first_permitted_signal": state.get("first_permitted_signal_date"),
        "strategies_tracked": "C0,D2",
        "D2_historical_status": state.get("historical_status", "D2_NOT_SELECTED"),
        "D2_paper_status": state.get("paper_status", "RESEARCH_SHADOW_NOT_SELECTED"),
        "completed_signals": int(state.get("completed_signals", 0) or 0),
        "completed_executions": int(state.get("completed_executions", 0) or 0),
        "next_checkpoint": int(state.get("next_checkpoint_executions", 13) or 13),
        "public_signal_model_changed": False,
        "approved_for_real_money": False,
        "data_revision_detected": bool(state.get("data_revision_detected", False)),
        "latest_signal_date": latest_signal_date,
        "latest_C0_positions": c0_pos,
        "latest_D2_positions": d2_pos,
        "portfolio_diff": diff,
        "pending_execution_count": pending_count,
        "completed_execution_count": completed_count,
        "completed_weeks": metrics["completed_weeks"],
        "sample_status": metrics["sample_status"],
        "C0_paper_return": metrics["C0_paper_return"],
        "D2_paper_return": metrics["D2_paper_return"],
        "D2_active_return": metrics["D2_active_return"],
        "C0_max_drawdown": metrics["C0_max_drawdown"],
        "D2_max_drawdown": metrics["D2_max_drawdown"],
        "information_ratio": metrics["information_ratio"],
        "paper_costs": metrics["paper_costs"],
        "paper_turnover": metrics["paper_turnover"],
        "n_sessions": metrics["n_sessions"],
        "executions_preview": exec_preview,
        "public_model": PUBLIC_MODEL,
        "experimental_model": EXPERIMENTAL_MODEL,
        "last_updated_at": state.get("last_updated_utc"),
    }
