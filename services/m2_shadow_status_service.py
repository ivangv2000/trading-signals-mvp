"""Read-only service for M2 shadow paper tracker research page."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PAPER_ROOT = ROOT / "paper_research"

STATE_PATH = PAPER_ROOT / "state" / "v20_2_m2_shadow_state.json"
MODEL_STATE_PATH = PAPER_ROOT / "state" / "v20_2_m2_model_state.json"
SIGNALS_PATH = PAPER_ROOT / "data" / "v20_2_m2_signal_snapshots.csv"
EXECUTIONS_PATH = PAPER_ROOT / "data" / "v20_2_m2_executions.csv"
EQUITY_PATH = PAPER_ROOT / "data" / "v20_2_m2_daily_equity.csv"
TRAINING_EVENTS_PATH = PAPER_ROOT / "data" / "v20_2_m2_training_events.csv"

SUMMARY_V212 = ROOT / "research_v20_1_2_summary.csv"
PORTFOLIO_V212 = ROOT / "research_v20_1_2_portfolio_metrics.csv"
PREDICTIVE_V212 = ROOT / "research_v20_1_2_predictive_metrics.csv"
SELECTED_V212 = ROOT / "research_v20_1_2_selected_config.json"
DIAGNOSTIC_V203 = ROOT / "research_v20_3_selected_diagnostic.json"

B0_STRATEGY = "B0_PRODUCTION_V14"
M2_STRATEGY = "M2_LOGISTIC_TOP_QUARTILE"
PUBLIC_MODEL = "V14 R1 Return Engine"
EXPERIMENTAL_MODEL = "M2 Logistic Top Quartile"
PAPER_START_DATE = "2026-07-15"
FIRST_PERMITTED_SIGNAL = "2026-07-17"
CHECKPOINT_WEEKS = 13
MIN_EQUITY_ROWS_FOR_CHART = 2
MIN_SESSIONS_PRELIMINARY = 63


def _read_json(path: Path) -> tuple[dict, str | None]:
    if not path.exists():
        return {}, f"Missing file: {path.name}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        LOGGER.exception("Invalid JSON in %s", path)
        return {}, f"Invalid JSON: {path.name}"


def _read_csv(path: Path) -> tuple[pd.DataFrame, str | None]:
    if not path.exists():
        return pd.DataFrame(), f"Missing file: {path.name}"
    if path.stat().st_size == 0:
        return pd.DataFrame(), None
    try:
        return pd.read_csv(path), None
    except (pd.errors.EmptyDataError, ValueError) as exc:
        LOGGER.exception("Unreadable CSV in %s", path)
        return pd.DataFrame(), f"Unreadable CSV: {path.name}"


def _empty_tracker_state() -> dict[str, Any]:
    return {
        "status": "M2_SHADOW_TRACKER_NOT_INITIALIZED",
        "runtime_status": "WAITING_FOR_FIRST_FORWARD_SIGNAL",
        "paper_start_date": PAPER_START_DATE,
        "first_permitted_signal": FIRST_PERMITTED_SIGNAL,
        "M2_historical_status": "PROMISING_NOT_SELECTABLE_PRED5_AMBIGUOUS",
        "M2_paper_status": "RESEARCH_SHADOW_NOT_SELECTED",
        "completed_signal_batches": 0,
        "completed_executions": 0,
        "completed_weeks": 0,
        "next_checkpoint": CHECKPOINT_WEEKS,
        "current_model_version_id": None,
        "data_revision_detected": False,
        "current_public_model": PUBLIC_MODEL,
        "approved_for_real_money": False,
    }


def _empty_model_state() -> dict[str, Any]:
    return {
        "model_version_id": None,
        "training_date": None,
        "feature_names_ordered": [],
        "parameters": {},
        "training_row_count": None,
        "positive_class_count": None,
        "negative_class_count": None,
        "state_origin": None,
        "reconstruction_status": None,
    }


def _safe_bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def get_m2_tracker_state() -> dict[str, Any]:
    payload, _ = _read_json(STATE_PATH)
    if not payload:
        return _empty_tracker_state()
    return {
        "status": payload.get("status", "M2_SHADOW_TRACKER_INITIALIZED"),
        "runtime_status": payload.get("runtime_status", "WAITING_FOR_FIRST_FORWARD_SIGNAL"),
        "paper_start_date": payload.get("paper_start_date", PAPER_START_DATE),
        "first_permitted_signal": payload.get("first_permitted_signal", FIRST_PERMITTED_SIGNAL),
        "M2_historical_status": payload.get("M2_historical_status", "PROMISING_NOT_SELECTABLE_PRED5_AMBIGUOUS"),
        "M2_paper_status": payload.get("M2_paper_status", "RESEARCH_SHADOW_NOT_SELECTED"),
        "completed_signal_batches": int(payload.get("completed_signal_batches", 0) or 0),
        "completed_executions": int(payload.get("completed_executions", 0) or 0),
        "completed_weeks": int(payload.get("completed_weeks", 0) or 0),
        "next_checkpoint": int(payload.get("next_checkpoint", CHECKPOINT_WEEKS) or CHECKPOINT_WEEKS),
        "current_model_version_id": payload.get("current_model_version_id"),
        "data_revision_detected": _safe_bool(payload.get("data_revision_detected")),
        "current_public_model": payload.get("public_model", PUBLIC_MODEL),
        "approved_for_real_money": _safe_bool(payload.get("approved_for_real_money")),
        "contracts_ok": _safe_bool(payload.get("contracts_ok", True)),
        "contracts_message": payload.get("contracts_message", "OK"),
        "tracker_changed": _safe_bool(payload.get("m2_tracker_changed")),
        "last_successful_update_utc": payload.get("last_successful_update_utc"),
        "snapshot_last_date": payload.get("snapshot_last_date"),
    }


def get_m2_model_state() -> dict[str, Any]:
    payload, _ = _read_json(MODEL_STATE_PATH)
    tracker_payload, _ = _read_json(STATE_PATH)
    if not payload:
        return _empty_model_state()
    return {
        "model_version_id": payload.get("model_version_id"),
        "training_date": payload.get("training_date"),
        "feature_names_ordered": payload.get("feature_names_ordered", []),
        "parameters": payload.get("parameters", {}),
        "training_row_count": payload.get("training_row_count"),
        "positive_class_count": payload.get("positive_class_count"),
        "negative_class_count": payload.get("negative_class_count"),
        "state_origin": payload.get("state_origin"),
        "reconstruction_status": tracker_payload.get("reconstruction_status") or payload.get("reconstruction_status"),
    }


def get_m2_latest_signal() -> pd.DataFrame:
    signals, _ = _read_csv(SIGNALS_PATH)
    if signals.empty or "signal_batch_id" not in signals.columns:
        return pd.DataFrame()
    latest_batch = signals["signal_batch_id"].astype(str).max()
    latest = signals[signals["signal_batch_id"].astype(str) == latest_batch].copy()
    if latest.empty:
        return pd.DataFrame()
    latest["target_weight"] = pd.to_numeric(latest.get("target_weight"), errors="coerce")
    latest["selection_rank"] = pd.to_numeric(latest.get("selection_rank"), errors="coerce")
    keep = (
        latest["target_weight"].fillna(0).ne(0)
        | latest.get("action", pd.Series(index=latest.index, dtype=object)).astype(str).isin(["SELL", "REDUCE"])
    )
    latest = latest[keep].copy()
    return latest.sort_values(["strategy", "target_weight", "ticker"], ascending=[True, False, True])


def get_m2_latest_executions() -> pd.DataFrame:
    executions, _ = _read_csv(EXECUTIONS_PATH)
    if executions.empty:
        return pd.DataFrame()
    return executions.sort_values(
        ["execution_date", "recorded_at_utc", "strategy", "ticker"],
        ascending=[False, False, True, True],
    ).head(20).copy()


def get_m2_equity_history() -> pd.DataFrame:
    equity, _ = _read_csv(EQUITY_PATH)
    if equity.empty:
        return pd.DataFrame()
    if "date" in equity.columns:
        equity["date"] = pd.to_datetime(equity["date"], errors="coerce")
    return equity.dropna(subset=["date"]).sort_values(["date", "strategy"]).copy()


def _read_training_events() -> pd.DataFrame:
    training_events, _ = _read_csv(TRAINING_EVENTS_PATH)
    if training_events.empty:
        return pd.DataFrame()
    return training_events.copy()


def get_m2_historical_metrics() -> dict[str, Any]:
    summary_df, _ = _read_csv(SUMMARY_V212)
    portfolio_df, _ = _read_csv(PORTFOLIO_V212)
    predictive_df, _ = _read_csv(PREDICTIVE_V212)
    selected_json, _ = _read_json(SELECTED_V212)
    result = {
        "available": False,
        "historical_period_start": "2019-06-21",
        "historical_period_end": "2026-07-14",
        "M2_CAGR": None,
        "B0_CAGR": None,
        "M2_excess_CAGR": None,
        "M2_sharpe": None,
        "M2_max_drawdown": None,
        "M2_information_ratio_vs_B0": None,
        "M2_mean_IC": None,
        "PBO": None,
        "trial_count": None,
        "historical_status": None,
        "pred5_contract_status": None,
    }
    if not summary_df.empty:
        row = summary_df.iloc[0]
        result["PBO"] = row.get("PBO")
        result["trial_count"] = row.get("trial_count")
        result["historical_status"] = row.get("M2_historical_status")
        result["pred5_contract_status"] = row.get("PRED5_contract_status")
    if not portfolio_df.empty and "model" in portfolio_df.columns:
        m2 = portfolio_df[portfolio_df["model"].astype(str) == M2_STRATEGY]
        b0 = portfolio_df[portfolio_df["model"].astype(str) == B0_STRATEGY]
        if not m2.empty:
            m2_row = m2.iloc[0]
            result["M2_CAGR"] = m2_row.get("CAGR")
            result["M2_excess_CAGR"] = m2_row.get("excess_CAGR")
            result["M2_sharpe"] = m2_row.get("sharpe")
            result["M2_max_drawdown"] = m2_row.get("max_drawdown")
            result["M2_information_ratio_vs_B0"] = m2_row.get("information_ratio")
        if not b0.empty:
            result["B0_CAGR"] = b0.iloc[0].get("CAGR")
    if not predictive_df.empty and {"model", "period"}.issubset(predictive_df.columns):
        m2_pred = predictive_df[
            (predictive_df["model"].astype(str) == M2_STRATEGY)
            & (predictive_df["period"].astype(str) == "MODEL_FULL_AVAILABLE")
        ]
        if not m2_pred.empty:
            result["M2_mean_IC"] = m2_pred.iloc[0].get("mean_ic")
    if selected_json:
        result["available"] = True
        result["selected_forward_paper_challenger"] = selected_json.get("selected_forward_paper_challenger")
    result["available"] = result["available"] or any(result[k] is not None for k in ["M2_CAGR", "M2_mean_IC", "PBO"])
    return result


def get_m2_diagnostic_metrics() -> dict[str, Any]:
    payload, _ = _read_json(DIAGNOSTIC_V203)
    if not payload:
        return {"available": False}
    return {
        "available": True,
        "AUC": payload.get("AUC"),
        "brier_skill": payload.get("brier_skill"),
        "ranking_quality": payload.get("ranking_quality"),
        "calibration_status": payload.get("calibration_status"),
        "stable_features": payload.get("stable_features"),
        "strongest_features": payload.get("strongest_features"),
        "temporal_excess_2019_2021": payload.get("excess_2019_2021"),
        "temporal_excess_2022_2023": payload.get("excess_2022_2023"),
        "temporal_excess_2024_2026": payload.get("excess_2024_2026"),
        "status": payload.get("status"),
        "public_model": payload.get("public_model"),
    }


def _build_signal_comparison(signal_df: pd.DataFrame) -> dict[str, Any]:
    empty = {
        "same_portfolio": False,
        "only_b0": [],
        "only_m2": [],
        "shared": [],
        "weight_differences": pd.DataFrame(),
    }
    if signal_df.empty or "strategy" not in signal_df.columns:
        return empty
    b0 = signal_df[signal_df["strategy"].astype(str) == B0_STRATEGY].copy()
    m2 = signal_df[signal_df["strategy"].astype(str) == M2_STRATEGY].copy()
    if b0.empty and m2.empty:
        return empty
    b0_set = set(b0["ticker"].astype(str))
    m2_set = set(m2["ticker"].astype(str))
    merged = pd.merge(
        b0[["ticker", "target_weight"]].rename(columns={"target_weight": "b0_target_weight"}),
        m2[["ticker", "target_weight"]].rename(columns={"target_weight": "m2_target_weight"}),
        on="ticker",
        how="outer",
    )
    if not merged.empty:
        merged["b0_target_weight"] = pd.to_numeric(merged["b0_target_weight"], errors="coerce").fillna(0.0)
        merged["m2_target_weight"] = pd.to_numeric(merged["m2_target_weight"], errors="coerce").fillna(0.0)
        merged["target_weight_difference"] = merged["m2_target_weight"] - merged["b0_target_weight"]
        merged = merged.sort_values("target_weight_difference", ascending=False)
    return {
        "same_portfolio": b0_set == m2_set and len(b0) == len(m2),
        "only_b0": sorted(b0_set - m2_set),
        "only_m2": sorted(m2_set - b0_set),
        "shared": sorted(b0_set & m2_set),
        "weight_differences": merged,
    }


def get_m2_shadow_dashboard_data() -> dict[str, Any]:
    warnings: list[str] = []
    state_json, state_error = _read_json(STATE_PATH)
    if state_error:
        warnings.append(state_error)
    model_json, model_error = _read_json(MODEL_STATE_PATH)
    if model_error:
        warnings.append(model_error)
    _, signals_error = _read_csv(SIGNALS_PATH)
    _, executions_error = _read_csv(EXECUTIONS_PATH)
    _, equity_error = _read_csv(EQUITY_PATH)
    _, training_error = _read_csv(TRAINING_EVENTS_PATH)
    for err in (signals_error, executions_error, equity_error, training_error):
        if err:
            warnings.append(err)

    tracker = get_m2_tracker_state()
    model = get_m2_model_state()
    latest_signal = get_m2_latest_signal()
    latest_executions = get_m2_latest_executions()
    equity = get_m2_equity_history()
    training_events = _read_training_events()
    historical = get_m2_historical_metrics()
    diagnostic = get_m2_diagnostic_metrics()
    comparison = _build_signal_comparison(latest_signal)

    available = bool(state_json)
    error = None if available else "El tracker M2 todavia no esta inicializado."
    return {
        "available": available,
        "error": error,
        "tracker": tracker,
        "model": model,
        "historical": historical,
        "diagnostic": diagnostic,
        "latest_signal": latest_signal,
        "latest_executions": latest_executions,
        "equity": equity,
        "training_events": training_events,
        "warnings": warnings,
        "signal_comparison": comparison,
        "equity_ready": len(equity["date"].unique()) >= MIN_EQUITY_ROWS_FOR_CHART if not equity.empty and "date" in equity.columns else False,
        "preliminary_sample": len(equity["date"].unique()) < MIN_SESSIONS_PRELIMINARY if not equity.empty and "date" in equity.columns else True,
        "model_state_raw_present": bool(model_json),
    }
