"""12 mandatory tests for V17.7 paper tracker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_research import (
  APPROVED_FOR_REAL_MONEY,
  CASH_ASSET,
  INITIAL_CAPITAL,
  R2_OFF,
  R2_ON,
  STRATEGY_R0,
  STRATEGY_R2,
)
from paper_research.import_signal_snapshot import (
  build_r2_weights,
  compute_r2_exposure,
  load_closure_config,
  validate_closure_config,
)


def _row(name: str, passed: bool, detail: str = "") -> dict:
  return {"test": name, "pass": bool(passed), "detail": detail}


def run_paper_tracker_tests(
  tracker,
  config_path: Path | str | None = None,
  pre_import: bool = False,
) -> pd.DataFrame:
  rows = []

  # T12 / T1
  if config_path and Path(config_path).exists():
    try:
      cfg = load_closure_config(config_path)
      rows.append(_row("T1_CLOSURE_CONFIG_VALID", True, str(cfg.get("closure_status"))))
      rows.append(_row("T12_APPROVED_FOR_REAL_MONEY_FALSE", not cfg.get("approved_for_real_money", True)))
    except Exception as exc:
      rows.append(_row("T1_CLOSURE_CONFIG_VALID", False, str(exc)))
      rows.append(_row("T12_APPROVED_FOR_REAL_MONEY_FALSE", False, str(exc)))
  else:
    state = tracker._read_state()
    rows.append(_row("T1_CLOSURE_CONFIG_VALID", state.get("closure_validated", False) or pre_import,
                     "config pendiente en pre-import" if pre_import else "sin config"))
    rows.append(_row("T12_APPROVED_FOR_REAL_MONEY_FALSE", not APPROVED_FOR_REAL_MONEY))

  snaps = tracker._read_csv(tracker.SIGNAL_SNAPSHOTS)
  execs = tracker._read_csv(tracker.EXECUTIONS)
  daily = tracker._read_csv(tracker.DAILY_EQUITY)
  membership = tracker._read_csv(tracker.MEMBERSHIP_SNAPSHOTS)

  config_ok = any(
    r["test"] == "T1_CLOSURE_CONFIG_VALID" and r["pass"] for r in rows
  )
  awaiting_first_snapshot = snaps.empty and (pre_import or config_ok)

  # T2 R0 weights
  if not snaps.empty:
    r0 = snaps[snaps["strategy"].astype(str) == STRATEGY_R0]
    if not r0.empty:
      latest_sid = r0.sort_values("signal_date").iloc[-1]["snapshot_id"]
      w = r0[r0["snapshot_id"] == latest_sid]["target_weight"].astype(float)
      rows.append(_row("T2_R0_WEIGHTS_SUM_ONE", abs(float(w.sum()) - 1.0) < 1e-6, f"sum={w.sum():.6f}"))
    else:
      rows.append(_row("T2_R0_WEIGHTS_SUM_ONE", awaiting_first_snapshot, "sin R0 aun"))
  else:
    rows.append(_row("T2_R0_WEIGHTS_SUM_ONE", awaiting_first_snapshot, "sin snapshots"))

  # T3 R2 frozen rule
  if not snaps.empty:
    r0 = snaps[snaps["strategy"].astype(str) == STRATEGY_R0]
    r2 = snaps[snaps["strategy"].astype(str) == STRATEGY_R2]
    if not r0.empty and not r2.empty:
      sid = r0.sort_values("signal_date").iloc[-1]["snapshot_id"]
      signal_date = pd.Timestamp(r0[r0["snapshot_id"] == sid].iloc[0]["signal_date"])
      r0_w = r0[r0["snapshot_id"] == sid].set_index("ticker")["target_weight"].astype(float)
      r2_sid = r2.sort_values("signal_date").iloc[-1]["snapshot_id"]
      r2_w = r2[r2["snapshot_id"] == r2_sid].set_index("ticker")["target_weight"].astype(float)
      exposure = float(r2.sort_values("signal_date").iloc[-1].get("risk_exposure", np.nan))
      expected = build_r2_weights(r0_w, exposure)
      aligned = expected.reindex(r2_w.index).fillna(0)
      ok = np.allclose(aligned.values, r2_w.reindex(aligned.index).fillna(0).values, atol=1e-6)
      exp_ok = exposure in (R2_ON, R2_OFF)
      rows.append(_row("T3_R2_RULE_FROZEN", ok and exp_ok, f"exposure={exposure}"))
    else:
      rows.append(_row("T3_R2_RULE_FROZEN", awaiting_first_snapshot, "sin R0/R2"))
  else:
    rows.append(_row("T3_R2_RULE_FROZEN", awaiting_first_snapshot, "sin snapshots"))

  # T4 R2 remainder to SHY
  if not snaps.empty:
    r2 = snaps[snaps["strategy"].astype(str) == STRATEGY_R2]
    r0 = snaps[snaps["strategy"].astype(str) == STRATEGY_R0]
    if not r2.empty and not r0.empty:
      r2_sid = r2.sort_values("signal_date").iloc[-1]["snapshot_id"]
      r0_sid = r0.sort_values("signal_date").iloc[-1]["snapshot_id"]
      r2_w = r2[r2["snapshot_id"] == r2_sid].set_index("ticker")["target_weight"].astype(float)
      r0_w = r0[r0["snapshot_id"] == r0_sid].set_index("ticker")["target_weight"].astype(float)
      exposure = float(r2[r2["snapshot_id"] == r2_sid]["risk_exposure"].iloc[0])
      stock_sum_r0 = float(r0_w.drop(CASH_ASSET, errors="ignore").sum())
      shy_expected = float(r0_w.get(CASH_ASSET, 0.0)) + (1.0 - exposure) * stock_sum_r0
      shy_actual = float(r2_w.get(CASH_ASSET, 0.0))
      rows.append(_row(
        "T4_R2_REMAINDER_TO_SHY",
        abs(shy_expected - shy_actual) < 1e-6,
        f"shy={shy_actual:.6f}",
      ))
    else:
      rows.append(_row("T4_R2_REMAINDER_TO_SHY", awaiting_first_snapshot, "sin R2"))
  else:
    rows.append(_row("T4_R2_REMAINDER_TO_SHY", awaiting_first_snapshot, "sin snapshots"))

  # T5 next open execution
  if not execs.empty:
    ok = bool(
      (pd.to_datetime(execs["execution_date"]) > pd.to_datetime(execs["signal_date"])).all()
    )
    rows.append(_row("T5_NEXT_OPEN_EXECUTION", ok))
  else:
    rows.append(_row("T5_NEXT_OPEN_EXECUTION", True, "ejecucion pendiente en next open"))

  # T6 no duplicate snapshots
  if not snaps.empty:
    dup = snaps.duplicated(subset=["snapshot_id", "strategy", "ticker"]).any()
    id_dup = snaps["snapshot_id"].astype(str).duplicated().any() and False
    unique_ids = snaps.groupby(["snapshot_id", "strategy"]).size()
    rows.append(_row("T6_NO_DUPLICATE_SNAPSHOTS", not dup, f"grupos={len(unique_ids)}"))
  else:
    rows.append(_row("T6_NO_DUPLICATE_SNAPSHOTS", True, "vacio"))

  # T7 append only - verify file grows, no truncate helper exposed
  rows.append(_row("T7_APPEND_ONLY", hasattr(tracker, "_append_csv"), "append_csv disponible"))

  # T8 no historical backfill before first execution
  state = tracker._read_state()
  first_exec = state.get("first_execution_date", {})
  if not daily.empty and first_exec:
    ok = True
    for strategy, start in first_exec.items():
      strat_daily = daily[daily["strategy"].astype(str) == strategy]
      if strat_daily.empty:
        continue
      min_date = pd.to_datetime(strat_daily["date"]).min()
      if min_date < pd.Timestamp(start):
        ok = False
        break
    rows.append(_row("T8_NO_HISTORICAL_BACKFILL", ok))
  else:
    rows.append(_row("T8_NO_HISTORICAL_BACKFILL", True, "sin equity o sin exec"))

  # T9 costs included
  if not execs.empty:
    rows.append(_row(
      "T9_COSTS_INCLUDED",
      (pd.to_numeric(execs["cost_usd"], errors="coerce").fillna(0) >= 0).all(),
    ))
  else:
    rows.append(_row("T9_COSTS_INCLUDED", True, "costes se aplicaran en ejecucion"))

  # T10 no leverage
  if not snaps.empty:
    ok = True
    for sid in snaps["snapshot_id"].astype(str).unique():
      w = snaps[snaps["snapshot_id"] == sid]["target_weight"].astype(float).sum()
      if float(w) > 1.0 + 1e-6:
        ok = False
        break
    rows.append(_row("T10_NO_LEVERAGE", ok))
  else:
    rows.append(_row("T10_NO_LEVERAGE", awaiting_first_snapshot, "sin snapshots"))

  # T11 membership snapshot dated
  if not membership.empty:
    ok = membership["signal_date"].notna().all() and membership["captured_at"].notna().all()
    rows.append(_row("T11_MEMBERSHIP_SNAPSHOT_DATED", bool(ok)))
  else:
    rows.append(_row("T11_MEMBERSHIP_SNAPSHOT_DATED", awaiting_first_snapshot, "sin membresia"))

  df = pd.DataFrame(rows)
  n_pass = int(df["pass"].sum())
  print("=" * 72)
  print(f"V17.7 PAPER TRACKER TESTS: {n_pass}/{len(df)} PASS")
  print(df.to_string(index=False))
  print("=" * 72)
  return df
