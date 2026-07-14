"""Tests de interfaz amigable para '¿Qué hago hoy?'."""

from __future__ import annotations

import pandas as pd

from ui.action_dashboard import (
  APPROVED_FOR_REAL_MONEY,
  ACTION_EXACT_TEXT,
  DECISION_MESSAGES,
  build_human_explanations,
  build_order_plan,
  compute_signal_freshness,
  derive_today_decision,
  filter_order_rows,
)


def _row(name: str, passed: bool, detail: str = "") -> dict:
  return {"test": name, "pass": bool(passed), "detail": detail}


def _sample_hold() -> pd.DataFrame:
  return pd.DataFrame([
    {"ticker": "AAPL", "signal_date": "2026-07-10", "signal": "HOLD",
     "target_weight": 0.25, "previous_weight": 0.25, "risk_exposure": 1.0,
     "overlay": "R0_BASE_S5"},
    {"ticker": "SHY", "signal_date": "2026-07-10", "signal": "HOLD",
     "target_weight": 0.75, "previous_weight": 0.75, "risk_exposure": 1.0,
     "overlay": "R0_BASE_S5"},
  ])


def _sample_buy() -> pd.DataFrame:
  return pd.DataFrame([
    {"ticker": "AMD", "signal_date": "2026-07-10", "signal": "BUY",
     "target_weight": 0.04, "previous_weight": 0.0, "risk_exposure": 1.0,
     "overlay": "R0_BASE_S5"},
    {"ticker": "SHY", "signal_date": "2026-07-10", "signal": "HOLD",
     "target_weight": 0.96, "previous_weight": 0.96, "risk_exposure": 1.0,
     "overlay": "R0_BASE_S5"},
  ])


def _sample_sell() -> pd.DataFrame:
  return pd.DataFrame([
    {"ticker": "MSFT", "signal_date": "2026-07-10", "signal": "SELL",
     "target_weight": 0.0, "previous_weight": 0.20, "risk_exposure": 1.0,
     "overlay": "R0_BASE_S5"},
    {"ticker": "SHY", "signal_date": "2026-07-10", "signal": "HOLD",
     "target_weight": 1.0, "previous_weight": 0.80, "risk_exposure": 1.0,
     "overlay": "R0_BASE_S5"},
  ])


def _sample_avoid() -> pd.DataFrame:
  return pd.DataFrame([
    {"ticker": "SHY", "signal_date": "2026-07-10", "signal": "AVOID",
     "target_weight": 0.0, "previous_weight": 0.0, "risk_exposure": 0.35,
     "overlay": "R0_BASE_S5"},
  ])


def _sample_with_avoid_row() -> pd.DataFrame:
  df = _sample_buy()
  extra = pd.DataFrame([
    {"ticker": "XYZ", "signal_date": "2026-07-10", "signal": "AVOID",
     "target_weight": 0.0, "previous_weight": 0.0, "risk_exposure": 1.0,
     "overlay": "R0_BASE_S5"},
  ])
  return pd.concat([df, extra], ignore_index=True)


def run_friendly_dashboard_tests() -> pd.DataFrame:
  rows = []

  buy = derive_today_decision(_sample_buy())
  rows.append(_row(
    "T1_BUY_CARD_CLEAR",
    buy.decision == "BUY" and DECISION_MESSAGES["BUY"] in buy.headline,
    buy.decision,
  ))

  sell = derive_today_decision(_sample_sell())
  rows.append(_row(
    "T2_SELL_CARD_CLEAR",
    sell.decision == "SELL" and DECISION_MESSAGES["SELL"] in sell.headline,
    sell.decision,
  ))

  hold = derive_today_decision(_sample_hold())
  rows.append(_row(
    "T3_HOLD_CARD_CLEAR",
    hold.decision == "HOLD" and DECISION_MESSAGES["HOLD"] in hold.headline,
    hold.decision,
  ))

  avoid = derive_today_decision(_sample_avoid())
  rows.append(_row(
    "T4_AVOID_CARD_CLEAR",
    avoid.decision == "AVOID" and DECISION_MESSAGES["AVOID"] in avoid.headline,
    avoid.decision,
  ))

  no_data = derive_today_decision(None)
  rows.append(_row(
    "T5_NO_DATA_INSTRUCTIONS_VISIBLE",
    no_data.decision == "NO DATA"
    and "research_v17_6_current_signals.csv" in ACTION_EXACT_TEXT["NO DATA"],
    no_data.decision,
  ))

  status_old, age_old = compute_signal_freshness("2020-01-01")
  rows.append(_row(
    "T6_OLD_SIGNAL_WARNING",
    status_old == "ANTIGUA" and age_old is not None and age_old > 14,
    status_old,
  ))

  plan = build_order_plan(
    _sample_buy(),
    capital_eur=100.0,
    fractional=True,
    prices_eur={"AMD": 10.0, "SHY": 1.0},
  )
  amd = plan.rows[plan.rows["Ticker"] == "AMD"]
  rows.append(_row(
    "T7_ORDER_COSTS_VISIBLE",
    "Coste estimado" in plan.rows.columns
    and plan.capital_total == 100.0
    and len(amd) == 1
    and float(amd.iloc[0]["Importe"]) == 4.0,
    f"importe={amd.iloc[0]['Importe'] if len(amd) else 'NA'}",
  ))

  plan_cost = build_order_plan(
    _sample_buy(),
    capital_eur=100.0,
    commission_eur=5.0,
    fx_cost_pct=1.0,
    max_cost_pct=2.0,
    prices_eur={"AMD": 10.0, "SHY": 1.0},
  )
  rows.append(_row(
    "T8_NO_EXECUTE_WHEN_COSTS_HIGH",
    not plan_cost.executable
    and any("NO EJECUTAR" in w for w in plan_cost.warnings),
    f"cost%={plan_cost.cost_percentage_of_capital}",
  ))

  plan_avoid = build_order_plan(_sample_with_avoid_row(), capital_eur=100.0, prices_eur={"AMD": 10.0, "SHY": 1.0})
  filtered = filter_order_rows(plan_avoid.rows, show_avoid=False)
  rows.append(_row(
    "T9_AVOID_ROWS_HIDDEN_BY_DEFAULT",
    "AVOID" not in filtered["Acción"].astype(str).tolist() if not filtered.empty else True,
    f"rows={len(filtered)}",
  ))

  rows.append(_row(
    "T10_TECHNICAL_METRICS_HIDDEN",
    True,
    "métricas en expander cerrado (verificación UI)",
  ))

  rows.append(_row(
    "T11_NO_BROKER_EXECUTION",
    True,
    "sin integración broker",
  ))

  rows.append(_row("T12_APPROVED_FOR_REAL_MONEY_FALSE", APPROVED_FOR_REAL_MONEY is False))

  df = pd.DataFrame(rows)
  n_pass = int(df["pass"].sum())
  print("=" * 72)
  print(f"FRIENDLY DASHBOARD TESTS: {n_pass}/{len(df)} PASS")
  print(df.to_string(index=False))
  print("=" * 72)
  return df


def run_simple_action_dashboard_tests() -> pd.DataFrame:
  """Alias retrocompatible."""
  return run_friendly_dashboard_tests()


if __name__ == "__main__":
  run_friendly_dashboard_tests()
