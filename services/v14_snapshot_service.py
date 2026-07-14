"""Persistencia de snapshots públicos V14."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from services.v14_signal_service import (
    DEFAULT_CAPITAL_REFERENCE,
    STRATEGY_NAME,
    derive_global_action,
    normalize_v14_signals,
)
from services.user_portfolio_action_service import build_universe_stats

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SIGNALS_PATH = DATA_DIR / "v14_latest_signals.csv"
SUMMARY_PATH = DATA_DIR / "v14_latest_summary.json"

SIGNAL_COLUMNS = [
    "signal_date",
    "ticker",
    "signal",
    "previous_weight",
    "target_weight",
    "strategy",
    "reason",
]


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _snapshot_hash(signals_df: pd.DataFrame) -> str:
    payload = signals_df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _count_positions(signals_df: pd.DataFrame) -> int:
    if signals_df is None or signals_df.empty:
        return 0
    active = signals_df[signals_df["target_weight"].fillna(0) > 0.001]
    return int(len(active))


def save_v14_snapshot(raw_signal: dict, capital_reference: float = DEFAULT_CAPITAL_REFERENCE) -> dict:
    """Guarda CSV + JSON del snapshot V14."""
    _ensure_data_dir()

    signal_date = raw_signal.get("last_date") or raw_signal.get("rebalance_date") or ""
    signals_df, meta = normalize_v14_signals(raw_signal, signal_date=signal_date)
    snapshot_hash = _snapshot_hash(signals_df) if not signals_df.empty else ""

    global_action = meta.get("global_action") or derive_global_action(signals_df)
    risk_level = raw_signal.get("risk_mode", "unknown")
    num_positions = _count_positions(signals_df)
    universe_stats = build_universe_stats(raw_signal)

    summary = {
        "strategy": meta.get("strategy", STRATEGY_NAME),
        "signal_date": signal_date,
        "next_rebalance": "Viernes después del cierre",
        "global_action": global_action,
        "model_global_action": global_action,
        "risk_level": risk_level,
        "capital_reference": capital_reference,
        "number_of_positions": num_positions,
        "universe_stats": universe_stats,
        "approved_for_real_money": False,
        "source": "v14_r1_return_engine",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "snapshot_hash": snapshot_hash,
    }

    if signals_df.empty:
        pd.DataFrame(columns=SIGNAL_COLUMNS).to_csv(SIGNALS_PATH, index=False)
    else:
        signals_df[SIGNAL_COLUMNS].to_csv(SIGNALS_PATH, index=False)

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"summary": summary, "signals": signals_df}


def load_latest_v14_snapshot() -> dict:
    """Carga el último snapshot si existe."""
    _ensure_data_dir()

    if not SIGNALS_PATH.exists() or not SUMMARY_PATH.exists():
        return {"signals": pd.DataFrame(columns=SIGNAL_COLUMNS), "summary": None, "has_data": False}

    try:
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except Exception:
        summary = None

    try:
        signals = pd.read_csv(SIGNALS_PATH)
        if signals.empty:
            return {"signals": signals, "summary": summary, "has_data": False}
        return {"signals": signals, "summary": summary, "has_data": True}
    except Exception:
        return {"signals": pd.DataFrame(columns=SIGNAL_COLUMNS), "summary": summary, "has_data": False}
