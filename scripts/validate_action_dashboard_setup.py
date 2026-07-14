#!/usr/bin/env python3
"""Validación sencilla de arranque para la página '¿Qué hago hoy?'."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED_SIGNAL_COLUMNS = {
  "ticker",
  "signal",
  "target_weight",
  "previous_weight",
}
ALLOWED_SIGNALS = {"BUY", "SELL", "HOLD", "REDUCE", "AVOID"}
WEIGHT_SUM_LIMIT = 1.000001

SIGNAL_CANDIDATES = [
  ROOT / "research_v17_6_current_signals.csv",
  ROOT / "paper_research" / "tmp" / "current_signals.csv",
]

CONFIG_CANDIDATES = [
  ROOT / "research_v17_6_1_selected_config.json",
  ROOT / "paper_research" / "tmp" / "selected_config.json",
]

RESULTS_GLOBS = [
  "research_v17_6_1_selected_config.json",
  "V17_6_1_CLOSURE.zip",
]


def _find_first(paths: list[Path]) -> Path | None:
  for path in paths:
    if path.exists() and path.stat().st_size > 0:
      return path
  return None


def _find_config_in_results() -> Path | None:
  found = _find_first(CONFIG_CANDIDATES)
  if found:
    return found
  for pattern in RESULTS_GLOBS:
    if pattern.endswith(".json"):
      for path in ROOT.rglob(pattern):
        if path.is_file() and path.stat().st_size > 0:
          return path
  return None


def _check_required_files(errors: list[str]) -> None:
  required = {
    "app.py": ROOT / "app.py",
    "pages/1_Que_Hago_Hoy.py": ROOT / "pages" / "1_Que_Hago_Hoy.py",
    "ui/action_dashboard.py": ROOT / "ui" / "action_dashboard.py",
  }
  for label, path in required.items():
    if not path.exists():
      errors.append(f"Falta {label}")
    else:
      print(f"OK {label}")


def _validate_signals_csv(path: Path, errors: list[str]) -> None:
  try:
    df = pd.read_csv(path)
  except Exception as exc:
    errors.append(f"No se pudo leer señales ({path.name}): {exc}")
    return

  missing_cols = REQUIRED_SIGNAL_COLUMNS - set(df.columns)
  if missing_cols:
    errors.append(
      f"CSV de señales sin columnas: {sorted(missing_cols)}"
    )
    return
  print(f"OK columnas mínimas en {path.name}")

  signals = df["signal"].astype(str).str.upper().str.strip()
  invalid = sorted(set(signals.unique()) - ALLOWED_SIGNALS)
  if invalid:
    errors.append(f"Señales no permitidas: {invalid}")
  else:
    print("OK señales permitidas (BUY/SELL/HOLD/REDUCE/AVOID)")

  weights = pd.to_numeric(df["target_weight"], errors="coerce")
  if weights.isna().any():
    errors.append("target_weight contiene valores no numéricos")
  elif (weights < 0).any():
    errors.append("target_weight contiene valores negativos")
  else:
    print("OK target_weight numéricos y no negativos")

  total_weight = float(weights.fillna(0).sum())
  if total_weight > WEIGHT_SUM_LIMIT:
    errors.append(
      f"Suma de target_weight supera {WEIGHT_SUM_LIMIT}: {total_weight}"
    )
  else:
    print(f"OK suma de pesos objetivo = {total_weight:.6f}")


def _check_approved_for_real_money(config_path: Path | None, errors: list[str]) -> bool:
  from ui.action_dashboard import APPROVED_FOR_REAL_MONEY

  if APPROVED_FOR_REAL_MONEY is not False:
    errors.append("ui.action_dashboard.APPROVED_FOR_REAL_MONEY no es False")
    return False
  print("OK ui.action_dashboard.APPROVED_FOR_REAL_MONEY=False")

  if config_path is None:
    return True

  try:
    config = json.loads(config_path.read_text(encoding="utf-8"))
  except Exception as exc:
    errors.append(f"No se pudo leer closure config: {exc}")
    return False

  approved = config.get("approved_for_real_money", True)
  if isinstance(approved, str):
    approved = approved.strip().lower() in ("true", "1", "yes")
  if approved is not False:
    errors.append("approved_for_real_money en closure config no es false")
    return False
  print(f"OK approved_for_real_money=false en {config_path.name}")
  return True


def _run_dashboard_tests(errors: list[str]) -> tuple[int, int]:
  from ui.dashboard_tests import run_friendly_dashboard_tests

  report = run_friendly_dashboard_tests()
  n_pass = int(report["pass"].sum())
  n_total = len(report)
  if n_pass < n_total:
    failed = report.loc[~report["pass"], "test"].tolist()
    errors.append(f"Dashboard tests fallidos: {failed}")
  return n_pass, n_total


def main() -> int:
  errors: list[str] = []

  print("=" * 72)
  print("VALIDACIÓN DE ARRANQUE — ¿QUÉ HAGO HOY?")
  print("=" * 72)

  _check_required_files(errors)

  signals_path = _find_first(SIGNAL_CANDIDATES)
  if signals_path:
    signals_status = "FOUND"
    print(f"OK signals file: {signals_path.relative_to(ROOT)}")
    _validate_signals_csv(signals_path, errors)
  else:
    signals_status = "NOT FOUND"
    print(
      "AVISO signals file: no encontrado — "
      "cárgalo manualmente en la página o coloca "
      "research_v17_6_current_signals.csv en la raíz del proyecto."
    )

  config_path = _find_config_in_results()
  if config_path:
    config_status = "FOUND"
    print(f"OK closure config: {config_path.relative_to(ROOT)}")
  else:
    config_status = "NOT FOUND"
    print(
      "AVISO closure config: no encontrado — "
      "opcional para la UI, pero recomendado "
      "(research_v17_6_1_selected_config.json)."
    )

  approved_ok = _check_approved_for_real_money(config_path, errors)
  n_pass, n_total = _run_dashboard_tests(errors)

  setup_pass = not errors
  print("=" * 72)
  if setup_pass:
    print("ACTION DASHBOARD SETUP: PASS")
  else:
    print("ACTION DASHBOARD SETUP: FAIL")
    for err in errors:
      print(f"  - {err}")
  print(f"Signals file: {signals_status}")
  print(f"Closure config: {config_status}")
  print(f"Dashboard tests: {n_pass}/{n_total} PASS")
  if approved_ok:
    print("APPROVED_FOR_REAL_MONEY=False")
  else:
    print("APPROVED_FOR_REAL_MONEY=CHECK FAILED")
  print("=" * 72)

  return 0 if setup_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
