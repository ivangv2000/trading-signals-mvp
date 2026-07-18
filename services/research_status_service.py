"""Read-only research status for Streamlit pages (no tracker updates)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import streamlit as st

    def _cache_data(**kwargs):
        return st.cache_data(**kwargs)
except Exception:  # pragma: no cover - tests without streamlit runtime
    def _cache_data(**kwargs):
        def deco(fn):
            return fn
        return deco


ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper_research"
DATA = PAPER / "data"
STATE = PAPER / "state"

M2_SIGNAL = DATA / "v20_2_m2_signal_snapshots.csv"
M2_EXEC = DATA / "v20_2_m2_executions.csv"
M2_STATE = STATE / "v20_2_m2_shadow_state.json"

M5_SIGNAL = DATA / "v25_2_m5_signal_snapshots.csv"
M5_EXEC = DATA / "v25_2_m5_executions.csv"
M5_STATE = STATE / "v25_2_m5_research_shadow_state.json"

D2_SIGNAL = DATA / "v18_3_5_signal_snapshots.csv"
D2_EXEC = DATA / "v18_3_5_executions.csv"
D2_STATE = STATE / "v18_3_5_d2_shadow_state.json"

V270_CFG = ROOT / "research_v27_0_selected_config.json"
V271_CFG = ROOT / "research_v27_1_selected_config.json"
V27141_CFG = ROOT / "research_v27_1_4_1_selected_config.json"
V27141_SUMMARY = ROOT / "research_v27_1_4_1_summary.csv"
V2715_CFG = ROOT / "research_v27_1_5_selected_config.json"
V2716_CFG = ROOT / "research_v27_1_6_selected_config.json"

B0_STRATEGY = "B0_PRODUCTION_V14"
M2_STRATEGY = "M2_LOGISTIC_TOP_QUARTILE"
C0_STRATEGY = "C0_PRODUCTION_V14"
D2_STRATEGY = "D2_TREND_QUALITY"
CASH_ASSET = "SHY"
PUBLIC_MODEL = "V14 R1 Return Engine"

EXEC_STATUS_ES = {
    "PENDING_EXECUTION_PRICE": "Pendiente de la próxima apertura",
    "EXECUTED": "Entrada paper registrada",
    "EXECUTION_RESOLVED_NEXT_OPEN": "Entrada paper registrada",
    "WAITING_FOR_M2_SIGNAL": "Esperando señal M2",
    "UP_TO_DATE": "Actualizado",
    "SIGNAL_RECORDED": "Señal registrada",
    "SIGNAL_CREATED_PENDING_EXECUTION": "Pendiente de la próxima apertura",
    "FORWARD_SIGNAL_RECORDED_PENDING_EXECUTION": "Señal registrada · pendiente de apertura",
    "WAITING_FOR_FIRST_FORWARD_SIGNAL": "Esperando la primera señal",
    "FORWARD_TRACKING_ACTIVE": "Seguimiento activo",
}


def translate_execution_status(raw: str | None) -> str:
    if not raw:
        return "Sin datos"
    return EXEC_STATUS_ES.get(str(raw), str(raw))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, ValueError, OSError):
        return pd.DataFrame()


def _safe_float(v: Any, default: float | None = None) -> float | None:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        if isinstance(v, str) and v.strip() == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _latest_date(df: pd.DataFrame, col: str = "signal_date") -> str | None:
    if df.empty or col not in df.columns:
        return None
    dates = df[col].dropna().astype(str)
    dates = dates[dates != ""]
    return str(dates.max()) if len(dates) else None


def _weights_map(df: pd.DataFrame) -> dict[str, float]:
    if df.empty or "ticker" not in df.columns or "target_weight" not in df.columns:
        return {}
    out: dict[str, float] = {}
    for _, r in df.iterrows():
        t = str(r["ticker"]).upper()
        w = _safe_float(r.get("target_weight"), 0.0) or 0.0
        if abs(w) > 1e-12:
            out[t] = w
    return out


def _holdings_list(weights: dict[str, float]) -> list[dict[str, Any]]:
    rows = [{"ticker": t, "target_weight": w} for t, w in weights.items()]
    return sorted(rows, key=lambda x: (-x["target_weight"], x["ticker"]))


def classify_situation(base_w: float, other_w: float, ticker: str) -> str:
    if ticker.upper() == CASH_ASSET:
        return "DEFENSIVO"
    b = abs(base_w) > 1e-12
    o = abs(other_w) > 1e-12
    if b and o:
        return "MANTENIDO"
    if o and not b:
        return "ENTRA"
    if b and not o:
        return "SALE"
    return "—"


def compare_portfolios(base: dict[str, float], other: dict[str, float]) -> list[dict[str, Any]]:
    tickers = sorted(set(base) | set(other))
    rows = []
    for t in tickers:
        bw = float(base.get(t, 0.0) or 0.0)
        ow = float(other.get(t, 0.0) or 0.0)
        if abs(bw) < 1e-12 and abs(ow) < 1e-12:
            continue
        rows.append({
            "ticker": t,
            "base_weight": bw,
            "other_weight": ow,
            "situation": classify_situation(bw, ow, t),
        })
    return sorted(rows, key=lambda r: (r["situation"] != "DEFENSIVO", -max(r["base_weight"], r["other_weight"]), r["ticker"]))


def human_diff_summary(base_label: str, other_label: str, comparison: list[dict[str, Any]]) -> str:
    entra = [r["ticker"] for r in comparison if r["situation"] == "ENTRA"]
    sale = [r["ticker"] for r in comparison if r["situation"] == "SALE"]
    mantiene = [r["ticker"] for r in comparison if r["situation"] == "MANTENIDO"]
    defensivo = [r["ticker"] for r in comparison if r["situation"] == "DEFENSIVO"]
    parts = []
    if entra:
        parts.append(f"{other_label} incorpora {', '.join(entra)} respecto a {base_label}")
    if sale:
        parts.append(f"deja fuera {', '.join(sale)}")
    if mantiene:
        parts.append(f"mantiene {', '.join(mantiene)}")
    if defensivo:
        parts.append(f"{', '.join(defensivo)} actúa como tramo defensivo")
    if not entra and not sale and mantiene:
        return f"{other_label} selecciona la misma cartera que {base_label}."
    if not parts:
        return f"Sin comparación disponible entre {base_label} y {other_label}."
    text = "; ".join(parts) + "."
    return text[0].upper() + text[1:]


@_cache_data(ttl=60)
def load_latest_m2_status() -> dict[str, Any]:
    err = None
    try:
        state = _read_json(M2_STATE)
        sig = _read_csv(M2_SIGNAL)
        exe = _read_csv(M2_EXEC)
        latest = _latest_date(sig)
        day = sig[sig["signal_date"].astype(str) == latest] if latest and not sig.empty else pd.DataFrame()
        b0 = day[day["strategy"].astype(str) == B0_STRATEGY] if not day.empty else pd.DataFrame()
        m2 = day[day["strategy"].astype(str) == M2_STRATEGY] if not day.empty else pd.DataFrame()
        b0_w = _weights_map(b0)
        m2_w = _weights_map(m2)
        comparison = compare_portfolios(b0_w, m2_w)
        # Enrich with M2 prediction columns when present
        pred_map = {}
        if not m2.empty:
            for _, r in m2.iterrows():
                pred_map[str(r["ticker"]).upper()] = {
                    "prediction_probability": _safe_float(r.get("prediction_probability")),
                    "prediction_rank": _safe_float(r.get("prediction_rank")),
                    "final_score": _safe_float(r.get("final_score")),
                    "selection_rank": r.get("selection_rank"),
                }
        for row in comparison:
            extra = pred_map.get(row["ticker"], {})
            row["m2_prediction"] = extra.get("prediction_probability")
            row["m2_rank"] = extra.get("prediction_rank")
            row["m2_final_score"] = extra.get("final_score")
        exe_day = exe[exe["signal_date"].astype(str) == latest] if latest and not exe.empty else pd.DataFrame()
        m2_exe = exe_day[exe_day["strategy"].astype(str) == M2_STRATEGY] if not exe_day.empty else pd.DataFrame()
        statuses = m2_exe["execution_status"].astype(str).tolist() if not m2_exe.empty and "execution_status" in m2_exe.columns else []
        raw_status = statuses[0] if statuses else (b0["execution_status"].iloc[0] if not b0.empty and "execution_status" in b0.columns else None)
        return {
            "ok": True,
            "error": None,
            "signal_date": latest,
            "batch_id": str(day["signal_batch_id"].iloc[0]) if not day.empty and "signal_batch_id" in day.columns else None,
            "b0_weights": b0_w,
            "m2_weights": m2_w,
            "b0_holdings": _holdings_list(b0_w),
            "m2_holdings": _holdings_list(m2_w),
            "comparison": comparison,
            "diff_summary": human_diff_summary("V14", "M2", comparison),
            "execution_status_raw": raw_status,
            "execution_status_label": translate_execution_status(str(raw_status) if raw_status else None),
            "executions": m2_exe.to_dict(orient="records") if not m2_exe.empty else [],
            "completed_executions": int(state.get("completed_executions", 0) or 0),
            "completed_signal_batches": int(state.get("completed_signal_batches", 0) or 0),
            "completed_weeks": int(state.get("completed_weeks", 0) or 0),
            "next_checkpoint": state.get("next_checkpoint", 13),
            "model_version_id": state.get("current_model_version_id"),
            "paper_status": state.get("M2_paper_status"),
            "historical_status": state.get("M2_historical_status"),
            "public_model": state.get("public_model", PUBLIC_MODEL),
            "approved_for_real_money": bool(state.get("approved_for_real_money", False)),
            "runtime_status": state.get("runtime_status"),
            "classification": "RESEARCH_SHADOW_NOT_SELECTED",
            "label": "INVESTIGACIÓN · PAPER TRADING",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"No se pudo cargar M2: {exc}", "signal_date": None, "b0_holdings": [], "m2_holdings": [], "comparison": []}


@_cache_data(ttl=60)
def load_latest_m5_status() -> dict[str, Any]:
    try:
        state = _read_json(M5_STATE)
        sig = _read_csv(M5_SIGNAL)
        exe = _read_csv(M5_EXEC)
        latest = _latest_date(sig) or state.get("latest_signal_date")
        day = sig[sig["signal_date"].astype(str) == latest] if latest and not sig.empty else pd.DataFrame()
        weights = _weights_map(day)
        conviction = {}
        if not day.empty:
            r0 = day.iloc[0]
            conviction = {
                "conviction_raw": _safe_float(r0.get("CONVICTION_RAW")),
                "conviction_percentile": _safe_float(r0.get("CONVICTION_PERCENTILE")),
                "m2_weight": _safe_float(r0.get("M5_M2_WEIGHT")),
                "baseline_weight": _safe_float(r0.get("M5_BASELINE_WEIGHT")),
                "m2_top3": str(r0.get("m2_top3", "")),
                "m5_top3": str(r0.get("m5_top3", "")),
            }
        m2 = load_latest_m2_status()
        b0_w = m2.get("b0_weights") or {}
        comparison = compare_portfolios(b0_w, weights)
        same_as_v14 = set(b0_w.keys()) == set(weights.keys()) and all(
            abs(b0_w.get(t, 0) - weights.get(t, 0)) < 1e-6 for t in set(b0_w) | set(weights)
        )
        w_m2 = conviction.get("m2_weight")
        if same_as_v14 and w_m2 is not None and w_m2 < 0.25:
            explanation = (
                "M5 combina la cartera base V14 con M2. En esta señal dio más peso al "
                "baseline, por lo que terminó seleccionando la misma cartera que V14."
            )
        elif same_as_v14:
            explanation = "M5 seleccionó la misma cartera que V14 en esta señal."
        else:
            explanation = human_diff_summary("V14", "M5", comparison)
        statuses = []
        if not exe.empty and "status" in exe.columns:
            if latest and "signal_id" in exe.columns and not day.empty and "signal_id" in day.columns:
                sid = str(day["signal_id"].iloc[0])
                statuses = exe.loc[exe["signal_id"].astype(str) == sid, "status"].astype(str).tolist()
            else:
                statuses = exe["status"].astype(str).tolist()
        raw_status = statuses[0] if statuses else (day["signal_status"].iloc[0] if not day.empty and "signal_status" in day.columns else None)
        return {
            "ok": True,
            "error": None,
            "signal_date": latest,
            "weights": weights,
            "holdings": _holdings_list(weights),
            "conviction": conviction,
            "comparison_vs_v14": comparison,
            "same_as_v14": same_as_v14,
            "explanation": explanation,
            "execution_status_raw": raw_status,
            "execution_status_label": translate_execution_status(str(raw_status) if raw_status else None),
            "executions": exe.to_dict(orient="records") if not exe.empty else [],
            "signals_count": int(state.get("signals_count", 0) or 0),
            "executions_count": int(state.get("executions_count", 0) or 0),
            "resolved_weeks": int(state.get("resolved_weeks", 0) or 0),
            "research_classification": state.get("research_classification"),
            "historical_selection_status": state.get("historical_selection_status"),
            "runtime_status": state.get("runtime_status"),
            "public_model": state.get("current_public_model", PUBLIC_MODEL),
            "approved_for_real_money": bool(state.get("approved_for_real_money", False)),
            "trial_count": int(state.get("trial_count", 619) or 619),
            "classification": state.get("research_classification") or "RESEARCH_ONLY_REJECTED_HISTORICAL_CHALLENGER",
            "label": "INVESTIGACIÓN · PAPER TRADING",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"No se pudo cargar M5: {exc}", "signal_date": None, "holdings": [], "conviction": {}}


@_cache_data(ttl=60)
def load_latest_d2_status() -> dict[str, Any]:
    try:
        state = _read_json(D2_STATE)
        sig = _read_csv(D2_SIGNAL)
        exe = _read_csv(D2_EXEC)
        latest = _latest_date(sig) or state.get("last_signal_date")
        day = sig[sig["signal_date"].astype(str) == latest] if latest and not sig.empty else pd.DataFrame()
        c0 = day[day["strategy"].astype(str) == C0_STRATEGY] if not day.empty else pd.DataFrame()
        d2 = day[day["strategy"].astype(str) == D2_STRATEGY] if not day.empty else pd.DataFrame()
        c0_w = _weights_map(c0)
        d2_w = _weights_map(d2)
        comparison = compare_portfolios(c0_w, d2_w)
        # Enrich ranks/scores from D2 rows
        meta = {}
        if not d2.empty:
            for _, r in d2.iterrows():
                meta[str(r["ticker"]).upper()] = {
                    "baseline_rank": _safe_float(r.get("baseline_rank")),
                    "trend_quality_rank": _safe_float(r.get("trend_quality_rank")),
                    "final_score": _safe_float(r.get("final_score")),
                    "rank": r.get("rank"),
                }
        for row in comparison:
            extra = meta.get(row["ticker"], {})
            row["baseline_rank"] = extra.get("baseline_rank")
            row["trend_quality_rank"] = extra.get("trend_quality_rank")
            row["final_score"] = extra.get("final_score")
            row["c0_weight"] = row["base_weight"]
            row["d2_weight"] = row["other_weight"]
        completed_signals = int(state.get("completed_signals", 0) or 0)
        completed_exec = int(state.get("completed_executions", 0) or 0)
        if completed_signals == 0:
            runtime = "WAITING_FOR_FIRST_FORWARD_SIGNAL"
        elif completed_exec == 0:
            runtime = "FORWARD_SIGNAL_RECORDED_PENDING_EXECUTION"
        else:
            runtime = "FORWARD_TRACKING_ACTIVE"
        exe_day = exe[exe["signal_date"].astype(str) == latest] if latest and not exe.empty else pd.DataFrame()
        statuses = exe_day["execution_status"].astype(str).tolist() if not exe_day.empty and "execution_status" in exe_day.columns else []
        raw_status = statuses[0] if statuses else None
        return {
            "ok": True,
            "error": None,
            "signal_date": latest,
            "c0_weights": c0_w,
            "d2_weights": d2_w,
            "c0_holdings": _holdings_list(c0_w),
            "d2_holdings": _holdings_list(d2_w),
            "comparison": comparison,
            "diff_summary": human_diff_summary("C0/V14", "D2", comparison),
            "completed_signals": completed_signals,
            "completed_executions": completed_exec,
            "next_checkpoint": state.get("next_checkpoint_executions", 13),
            "runtime_status": runtime,
            "runtime_status_label": translate_execution_status(runtime),
            "execution_status_raw": raw_status,
            "execution_status_label": translate_execution_status(str(raw_status) if raw_status else runtime),
            "historical_status": state.get("historical_status"),
            "paper_status": state.get("paper_status"),
            "public_model": state.get("public_model", PUBLIC_MODEL),
            "approved_for_real_money": bool(state.get("approved_for_real_money", False)),
            "classification": state.get("paper_status") or "RESEARCH_SHADOW_NOT_SELECTED",
            "label": "INVESTIGACIÓN · PAPER TRADING",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"No se pudo cargar D2: {exc}", "signal_date": None, "c0_holdings": [], "d2_holdings": [], "comparison": []}


@_cache_data(ttl=60)
def load_research_infrastructure_status() -> dict[str, Any]:
    try:
        n0 = _read_json(V270_CFG)
        n1 = _read_json(V271_CFG)
        health_src = _read_json(V2716_CFG) or _read_json(V2715_CFG) or _read_json(V27141_CFG)
        summary = _read_csv(V27141_SUMMARY)
        suite_tests = suite_passed = suite_failed = None
        repo_status = health_src.get("status")
        trial = int(health_src.get("trial_count") or n0.get("trial_count") or 619)
        if health_src.get("suite_tests") is not None:
            suite_tests = int(health_src["suite_tests"])
            suite_passed = int(health_src.get("suite_passed", suite_tests))
            suite_failed = int(health_src.get("suite_failed", 0))
        elif not summary.empty:
            suite_tests = int(summary.iloc[0].get("full_suite_tests") or 0)
            suite_passed = int(summary.iloc[0].get("full_suite_passed") or 0)
            suite_failed = int(summary.iloc[0].get("full_suite_failed") or 0)
            if not repo_status:
                repo_status = str(summary.iloc[0].get("status", ""))
        return {
            "ok": True,
            "error": None,
            "public_model": PUBLIC_MODEL,
            "approved_for_real_money": False,
            "trial_count": trial,
            "repository_status": repo_status,
            "suite_tests": suite_tests,
            "suite_passed": suite_passed,
            "suite_failed": suite_failed,
            "norgate": {
                "connected": str(n0.get("norgatedata_import", "")).upper() == "PASS",
                "ndu_connected": str(n0.get("ndu_database_connection", "")).upper() == "PASS",
                "delisted_securities": bool(n0.get("delisted_securities")),
                "historical_constituents": bool(n0.get("historical_index_constituents")),
                "sp500_pit_ready": bool(n0.get("sp500_point_in_time_membership")),
                "first_available_date": n0.get("first_available_date"),
                "last_available_date": n0.get("last_available_date"),
                "full_history_available": bool(n0.get("suitable_for_1998_2026_backtest")),
                "nested_top_available": "UNAVAILABLE" not in str(n1.get("nested_universe_status", "UNAVAILABLE")),
                "nested_universe_status": n1.get("nested_universe_status"),
                "pit_pipeline_status": n1.get("status"),
                "norgate_status": n0.get("status"),
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"No se pudo cargar infraestructura: {exc}", "norgate": {}, "trial_count": 619}


@_cache_data(ttl=60)
def load_latest_research_comparison() -> dict[str, Any]:
    m2 = load_latest_m2_status()
    m5 = load_latest_m5_status()
    d2 = load_latest_d2_status()
    infra = load_research_infrastructure_status()
    dates = [d for d in [m2.get("signal_date"), m5.get("signal_date"), d2.get("signal_date")] if d]
    latest = max(dates) if dates else None
    cards = [
        {
            "id": "v14",
            "model": "V14 / B0",
            "classification": "MODELO PÚBLICO",
            "label_kind": "PUBLIC_MODEL",
            "signal_date": m2.get("signal_date"),
            "holdings": m2.get("b0_holdings") or [],
            "execution_status_label": m2.get("execution_status_label"),
            "status_tag": m2.get("paper_status") or "ACTIVO",
        },
        {
            "id": "m2",
            "model": "M2 Logistic Top Quartile",
            "classification": "INVESTIGACIÓN · PAPER TRADING",
            "label_kind": "RESEARCH_ONLY",
            "signal_date": m2.get("signal_date"),
            "holdings": m2.get("m2_holdings") or [],
            "execution_status_label": m2.get("execution_status_label"),
            "status_tag": m2.get("paper_status"),
        },
        {
            "id": "m5",
            "model": "M5 Conviction Adaptive",
            "classification": "INVESTIGACIÓN · PAPER TRADING",
            "label_kind": "RESEARCH_ONLY",
            "signal_date": m5.get("signal_date"),
            "holdings": m5.get("holdings") or [],
            "execution_status_label": m5.get("execution_status_label"),
            "status_tag": m5.get("research_classification"),
        },
        {
            "id": "d2",
            "model": "D2 Trend Quality",
            "classification": "INVESTIGACIÓN · PAPER TRADING",
            "label_kind": "RESEARCH_ONLY",
            "signal_date": d2.get("signal_date"),
            "holdings": d2.get("d2_holdings") or [],
            "execution_status_label": d2.get("execution_status_label"),
            "status_tag": d2.get("paper_status"),
        },
    ]
    plain_parts = [
        m2.get("diff_summary"),
        m5.get("explanation"),
        d2.get("diff_summary"),
    ]
    return {
        "ok": True,
        "latest_signal_date": latest,
        "cards": cards,
        "plain_language": [p for p in plain_parts if p],
        "m2": m2,
        "m5": m5,
        "d2": d2,
        "infra": infra,
        "approved_for_real_money": False,
        "public_model": PUBLIC_MODEL,
    }
