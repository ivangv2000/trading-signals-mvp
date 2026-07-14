# %% [markdown]
# # Trading Research V17.6.1 — Result Closure (sin re-ejecutar V17.6)
#
# Valida y cierra los exports de V17.6 leyendo únicamente los CSV/JSON existentes.
# **No descarga precios, no entrena XGB, no simula overlays, no null test, no DSR/PBO.**
# **APPROVED_FOR_REAL_MONEY = False siempre.**

# %%
try:
  get_ipython().run_line_magic("pip", "install pandas numpy -q")
except NameError:
  import subprocess, sys
  subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas", "numpy"])

# %% [markdown]
# ## 1. Subir exports V17.6 (13 archivos)

# %%
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
  import google.colab  # noqa: F401
  IN_COLAB = True
except ImportError:
  IN_COLAB = False

CONTENT_ROOT = Path("/content") if IN_COLAB else Path.cwd()
AUDIT_VERSION = "v17_6_1"
APPROVED_FOR_REAL_MONEY = False
CLOSURE_STATUS = "V17_6_CLOSED_VALIDATED"
ORIGINAL_FINAL_STATUS = "RISK_OVERLAY_NOT_YET_ACCEPTABLE"
OFFICIAL_SELECTED_OVERLAY = "NONE"
PAPER_RESEARCH_CHALLENGER = "R2_SPY_TREND"
NEXT_PHASE = "DUAL_FORWARD_PAPER_RESEARCH"

OFFICIAL_OVERLAYS = [
  "R0_BASE_S5",
  "R1_VOL_TARGET_12",
  "R2_SPY_TREND",
  "R3_COMBINED_FIXED",
]

OVERLAY_CLASSIFICATIONS = {
  "R0_BASE_S5": "HIGH_RETURN_HIGH_RISK_BASELINE",
  "R1_VOL_TARGET_12": "RISK_REDUCED_ALPHA_DESTROYED",
  "R2_SPY_TREND": "NEAR_MISS_FORWARD_PAPER_CHALLENGER",
  "R3_COMBINED_FIXED": "LOWER_DRAWDOWN_ALPHA_DESTROYED",
}

# Umbrales preregistrados (no se ajustan tras observar resultados).
THRESH_DD = -35.0
THRESH_CAGR = 15.0
THRESH_EXCESS = 4.0
THRESH_IR = 0.30
THRESH_YEARS_B1 = 55.0
THRESH_YEARS_SPY = 55.0

R0_EXPECTED_CAGR = 26.14
R0_EXPECTED_DD = -50.96
R0_CAGR_TOL = 1.5
R0_DD_TOL = 2.5

REQUIRED_FILES = [
  "research_v17_6_cost_sensitivity.csv",
  "research_v17_6_current_signals.csv",
  "research_v17_6_drawdown_comparison.csv",
  "research_v17_6_dsr_audit.csv",
  "research_v17_6_execution_tests.csv",
  "research_v17_6_exposure_history.csv",
  "research_v17_6_null_results.csv",
  "research_v17_6_overlay_results.csv",
  "research_v17_6_research_period_results.csv",
  "research_v17_6_reused_test_results.csv",
  "research_v17_6_selected_config.json",
  "research_v17_6_summary.csv",
  "research_v17_6_validation_gates.csv",
]


def _resolve_input_path(name):
  for p in (CONTENT_ROOT / name, Path(name)):
    if p.exists() and p.stat().st_size > 0:
      return p
  return CONTENT_ROOT / name


def _load_required_exports():
  if IN_COLAB:
    from google.colab import files
    print("Sube los 13 exports de V17.6:")
    for f in REQUIRED_FILES:
      print(f"  - {f}")
    uploaded = files.upload()
    for fname, data in uploaded.items():
      dest = CONTENT_ROOT / Path(fname).name
      dest.write_bytes(data)
      print(f"Colocado: {dest}")
  else:
    print("Ejecucion local: se usan exports en el directorio actual.")

  missing = [f for f in REQUIRED_FILES if not _resolve_input_path(f).exists()]
  if missing:
    raise FileNotFoundError("Faltan exports V17.6: " + ", ".join(missing))
  print("13 exports obligatorios encontrados OK")


def _to_bool(value):
  if isinstance(value, (bool, np.bool_)):
    return bool(value)
  if isinstance(value, str):
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes"):
      return True
    if normalized in ("false", "0", "no"):
      return False
  if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
    return bool(int(value))
  raise ValueError(f"No se pudo convertir a bool: {value!r}")


def _normalize_selected_overlay(value):
  if value is None or (isinstance(value, float) and pd.isna(value)):
    return "NONE"
  text = str(value).strip()
  if text.lower() in ("", "none", "nan"):
    return "NONE"
  return text


def _fix_json_types(selected_config):
  """Corrige booleanos-string y enteros sin tocar los exports originales."""
  fixed = json.loads(json.dumps(selected_config))

  raw_gates = fixed.get("validation_gates", {})
  fixed_gates = {k: _to_bool(v) for k, v in raw_gates.items()}
  fixed["validation_gates"] = fixed_gates

  summary_block = fixed.get("summary", {})
  if "gates_pass" in summary_block:
    summary_block["gates_pass"] = int(summary_block["gates_pass"])
  if "gates_total" in summary_block:
    summary_block["gates_total"] = int(summary_block["gates_total"])
  if "APPROVED_FOR_REAL_MONEY" in summary_block:
    summary_block["APPROVED_FOR_REAL_MONEY"] = _to_bool(
      summary_block["APPROVED_FOR_REAL_MONEY"]
    )
  if "POINT_IN_TIME_MEMBERSHIP" in summary_block:
    summary_block["POINT_IN_TIME_MEMBERSHIP"] = _to_bool(
      summary_block["POINT_IN_TIME_MEMBERSHIP"]
    )
  fixed["summary"] = summary_block

  if "approved_for_real_money" in fixed:
    fixed["approved_for_real_money"] = _to_bool(fixed["approved_for_real_money"])

  return fixed


def _cost_2x_positive(cost_df, overlay_id):
  rows = cost_df[
    (cost_df["overlay"].astype(str) == overlay_id)
    & (pd.to_numeric(cost_df["cost_multiplier"], errors="coerce") == 2.0)
  ]
  if rows.empty:
    return False
  return float(rows.iloc[0]["CAGR_pct"]) > 0.0


def _overlay_row(df, overlay_id):
  subset = df[df["overlay"].astype(str) == overlay_id]
  if subset.empty:
    raise KeyError(f"Overlay ausente en export: {overlay_id}")
  return subset.iloc[0]


def _build_gate_matrix(research_df, reused_df, cost_df):
  rows = []
  for overlay_id in OFFICIAL_OVERLAYS:
    research = _overlay_row(research_df, overlay_id)
    reused = _overlay_row(reused_df, overlay_id)

    dd = float(research["max_drawdown_pct"])
    cagr = float(research["CAGR_pct"])
    excess = float(research["excess_CAGR_vs_B1"])
    ir = float(research["information_ratio_vs_B1"])
    years_b1 = float(research["pct_years_beating_B1"])
    years_spy = float(research["pct_years_beating_SPY"])
    recon = _to_bool(research.get("reconciliation_pass", False))
    cost_2x = _cost_2x_positive(cost_df, overlay_id)
    reused_pos = float(reused.get("excess_CAGR_vs_B1", 0.0)) > 0.0

    drawdown_pass = dd > THRESH_DD
    cagr_pass = cagr >= THRESH_CAGR
    excess_pass = excess >= THRESH_EXCESS
    information_ratio_pass = ir >= THRESH_IR
    years_b1_pass = years_b1 >= THRESH_YEARS_B1
    years_spy_pass = years_spy >= THRESH_YEARS_SPY

    all_pre_null_gates_pass = all([
      drawdown_pass,
      cagr_pass,
      excess_pass,
      information_ratio_pass,
      years_b1_pass,
      years_spy_pass,
      recon,
      cost_2x,
    ])

    rows.append({
      "overlay": overlay_id,
      "classification": OVERLAY_CLASSIFICATIONS[overlay_id],
      "drawdown_pass": drawdown_pass,
      "cagr_pass": cagr_pass,
      "excess_pass": excess_pass,
      "information_ratio_pass": information_ratio_pass,
      "years_B1_pass": years_b1_pass,
      "years_SPY_pass": years_spy_pass,
      "reconciliation_pass": recon,
      "cost_2x_positive": cost_2x,
      "reused_test_positive_active_return": reused_pos,
      "all_pre_null_gates_pass": all_pre_null_gates_pass,
      "drawdown_margin": round(dd - THRESH_DD, 4),
      "CAGR_margin": round(cagr - THRESH_CAGR, 4),
      "excess_CAGR_margin": round(excess - THRESH_EXCESS, 4),
      "information_ratio_margin": round(ir - THRESH_IR, 4),
      "research_max_drawdown_pct": round(dd, 4),
      "research_CAGR_pct": round(cagr, 4),
      "research_excess_CAGR_vs_B1": round(excess, 4),
      "research_information_ratio_vs_B1": round(ir, 4),
      "reused_test_excess_CAGR_vs_B1": round(float(reused.get("excess_CAGR_vs_B1", np.nan)), 4),
      "reused_test_information_ratio_vs_B1": round(
        float(reused.get("information_ratio_vs_B1", np.nan)), 4
      ),
    })
  return pd.DataFrame(rows)


def _build_near_miss_analysis(gate_matrix_df, dsr_df):
  r2 = gate_matrix_df[gate_matrix_df["overlay"] == "R2_SPY_TREND"].iloc[0]

  dsr_daily = dsr_df[
    (dsr_df["strategy"].astype(str) == "R2_SPY_TREND")
    & (dsr_df.get("sharpe_frequency", pd.Series(dtype=str)).astype(str) == "daily")
  ]
  daily_dsr = float(dsr_daily.iloc[0]["dsr_probability"]) if len(dsr_daily) else np.nan

  excess_shortfall = max(0.0, THRESH_EXCESS - float(r2["research_excess_CAGR_vs_B1"]))
  ir_shortfall = max(0.0, THRESH_IR - float(r2["research_information_ratio_vs_B1"]))

  rows = []
  for _, row in gate_matrix_df.iterrows():
    rows.append({
      "overlay": row["overlay"],
      "classification": row["classification"],
      "officially_approved": False,
      "paper_research_challenger": row["overlay"] == PAPER_RESEARCH_CHALLENGER,
      "research_max_drawdown_pct": row["research_max_drawdown_pct"],
      "research_CAGR_pct": row["research_CAGR_pct"],
      "research_excess_CAGR_vs_B1": row["research_excess_CAGR_vs_B1"],
      "research_information_ratio_vs_B1": row["research_information_ratio_vs_B1"],
      "excess_CAGR_shortfall_pp": (
        round(excess_shortfall, 4) if row["overlay"] == "R2_SPY_TREND" else np.nan
      ),
      "information_ratio_shortfall": (
        round(ir_shortfall, 4) if row["overlay"] == "R2_SPY_TREND" else np.nan
      ),
      "reused_test_excess_CAGR_vs_B1": row["reused_test_excess_CAGR_vs_B1"],
      "reused_test_information_ratio_vs_B1": row["reused_test_information_ratio_vs_B1"],
      "cost_2x_CAGR_positive": row["cost_2x_positive"],
      "daily_dsr_probability": daily_dsr if row["overlay"] == "R2_SPY_TREND" else np.nan,
      "drawdown_pass": row["drawdown_pass"],
      "excess_pass": row["excess_pass"],
      "information_ratio_pass": row["information_ratio_pass"],
      "all_pre_null_gates_pass": row["all_pre_null_gates_pass"],
      "note": (
        "R2 no aprobado: near-miss en excess CAGR e IR en periodo research"
        if row["overlay"] == "R2_SPY_TREND"
        else ""
      ),
    })
  return pd.DataFrame(rows)


_load_required_exports()

summary_in = pd.read_csv(_resolve_input_path("research_v17_6_summary.csv"))
gates_in = pd.read_csv(_resolve_input_path("research_v17_6_validation_gates.csv"))
research_in = pd.read_csv(_resolve_input_path("research_v17_6_research_period_results.csv"))
reused_in = pd.read_csv(_resolve_input_path("research_v17_6_reused_test_results.csv"))
cost_in = pd.read_csv(_resolve_input_path("research_v17_6_cost_sensitivity.csv"))
null_in = pd.read_csv(_resolve_input_path("research_v17_6_null_results.csv"))
tests_in = pd.read_csv(_resolve_input_path("research_v17_6_execution_tests.csv"))
dsr_in = pd.read_csv(_resolve_input_path("research_v17_6_dsr_audit.csv"))
selected_raw = json.loads(
  _resolve_input_path("research_v17_6_selected_config.json").read_text(encoding="utf-8")
)

# %% [markdown]
# ## 2. Validar el cierre V17.6

# %%
summary_row = summary_in.iloc[0].to_dict()
original_final_status = str(summary_row.get("FINAL_STATUS", ""))
technical_tests = str(summary_row.get("technical_tests_pass", ""))
null_permutations = int(summary_row.get("null_permutations", 0))
selected_overlay_raw = selected_raw.get("selected_overlay")
selected_overlay_norm = _normalize_selected_overlay(selected_overlay_raw)

r0_row = _overlay_row(research_in, "R0_BASE_S5")
r0_cagr = float(r0_row["CAGR_pct"])
r0_dd = float(r0_row["max_drawdown_pct"])

closure_checks = {
  "original_final_status": original_final_status == ORIGINAL_FINAL_STATUS,
  "technical_tests_12_12": technical_tests == "12/12",
  "r0_cagr_approx": abs(r0_cagr - R0_EXPECTED_CAGR) <= R0_CAGR_TOL,
  "r0_drawdown_approx": abs(r0_dd - R0_EXPECTED_DD) <= R0_DD_TOL,
  "no_official_overlay_selected": selected_overlay_norm == "NONE",
  "no_null_permutations": null_permutations == 0 and len(null_in) == 0,
  "approved_for_real_money_false": not _to_bool(summary_row.get("APPROVED_FOR_REAL_MONEY", False)),
  "point_in_time_membership_false": not _to_bool(summary_row.get("POINT_IN_TIME_MEMBERSHIP", True)),
  "execution_tests_all_pass": all(_to_bool(v) for v in tests_in["pass"]),
}

assert len(tests_in) == 12, f"Se esperaban 12 tests tecnicos, hay {len(tests_in)}"

failed_checks = [k for k, v in closure_checks.items() if not v]
if failed_checks:
  raise AssertionError(
    "Validacion de cierre fallida: " + ", ".join(failed_checks)
  )

print("CIERRE V17.6: validacion base PASS")
print(f"  FINAL_STATUS original: {original_final_status}")
print(f"  technical_tests: {technical_tests}")
print(f"  R0 CAGR={r0_cagr:.2f}% DD={r0_dd:.2f}%")
print(f"  selected_overlay: {selected_overlay_norm}")
print(f"  null_permutations: {null_permutations}")

# %% [markdown]
# ## 3. Matriz real de gates por overlay
#
# R6-R13 del JSON original evaluan `selected_overlay`. Como quedo vacio,
# esos gates globales pueden ser False aunque un overlay individual pase reglas.

# %%
gate_matrix_df = _build_gate_matrix(research_in, reused_in, cost_in)
print("\nMatriz de gates por overlay (periodo research):")
print(gate_matrix_df.to_string(index=False))

# %% [markdown]
# ## 4. Clasificacion honesta + near-miss R2

# %%
near_miss_df = _build_near_miss_analysis(gate_matrix_df, dsr_in)
r2_near = near_miss_df[near_miss_df["overlay"] == "R2_SPY_TREND"].iloc[0]
r2_excess_shortfall = float(r2_near["excess_CAGR_shortfall_pp"])
r2_ir_shortfall = float(r2_near["information_ratio_shortfall"])

assert not bool(r2_near["officially_approved"]), "R2 no debe marcarse como aprobado"
assert r2_near["classification"] == "NEAR_MISS_FORWARD_PAPER_CHALLENGER"

print("\nClasificacion honesta:")
for overlay_id, label in OVERLAY_CLASSIFICATIONS.items():
  print(f"  {overlay_id}: {label}")
print(f"  OFFICIAL_SELECTED_OVERLAY: {OFFICIAL_SELECTED_OVERLAY}")
print(f"  PAPER_RESEARCH_CHALLENGER: {PAPER_RESEARCH_CHALLENGER}")

# %% [markdown]
# ## 5. Corregir tipos del JSON (sin sobrescribir originales)

# %%
selected_fixed = _fix_json_types(selected_raw)
gates_fixed = selected_fixed["validation_gates"]

# %% [markdown]
# ## 6. Exports V17.6.1

# %%
closure_summary = {
  "AUDIT_VERSION": AUDIT_VERSION,
  "CLOSURE_STATUS": CLOSURE_STATUS,
  "ORIGINAL_FINAL_STATUS": ORIGINAL_FINAL_STATUS,
  "OFFICIAL_SELECTED_OVERLAY": OFFICIAL_SELECTED_OVERLAY,
  "PAPER_RESEARCH_CHALLENGER": PAPER_RESEARCH_CHALLENGER,
  "NEXT_PHASE": NEXT_PHASE,
  "APPROVED_FOR_REAL_MONEY": False,
  "technical_tests_pass": technical_tests,
  "R0_research_CAGR_pct": round(r0_cagr, 2),
  "R0_research_max_drawdown_pct": round(r0_dd, 2),
  "null_permutations_executed": null_permutations,
  "POINT_IN_TIME_MEMBERSHIP": False,
  "R2_excess_shortfall_pp": round(r2_excess_shortfall, 4),
  "R2_information_ratio_shortfall": round(r2_ir_shortfall, 4),
  "R2_research_max_drawdown_pct": float(r2_near["research_max_drawdown_pct"]),
  "R2_research_CAGR_pct": float(r2_near["research_CAGR_pct"]),
  "R2_reused_test_excess_CAGR_vs_B1": float(r2_near["reused_test_excess_CAGR_vs_B1"]),
  "R2_reused_test_information_ratio_vs_B1": float(
    r2_near["reused_test_information_ratio_vs_B1"]
  ),
  "R2_cost_2x_CAGR_positive": bool(r2_near["cost_2x_CAGR_positive"]),
  "R2_daily_dsr_probability": float(r2_near["daily_dsr_probability"]),
  "backtest_reexecuted": False,
  "model_retrained": False,
  "null_test_reexecuted": False,
  "closure_note": (
    "Matriz por overlay calculada desde exports V17.6; "
    "R6-R13 globales dependian de selected_overlay vacio"
  ),
}

selected_out = {
  "version": AUDIT_VERSION,
  "closure_status": CLOSURE_STATUS,
  "original_final_status": ORIGINAL_FINAL_STATUS,
  "official_selected_overlay": OFFICIAL_SELECTED_OVERLAY,
  "paper_research_challenger": PAPER_RESEARCH_CHALLENGER,
  "next_phase": NEXT_PHASE,
  "approved_for_real_money": False,
  "inherited_config": "research_v17_6_selected_config.json",
  "overlay_classifications": OVERLAY_CLASSIFICATIONS,
  "selection_thresholds_frozen": {
    "max_drawdown_pct_gt": THRESH_DD,
    "min_CAGR_pct": THRESH_CAGR,
    "min_excess_CAGR_vs_B1": THRESH_EXCESS,
    "min_information_ratio": THRESH_IR,
    "min_pct_years_beating_B1": THRESH_YEARS_B1,
    "min_pct_years_beating_SPY": THRESH_YEARS_SPY,
  },
  "validation_gates_original_fixed_types": gates_fixed,
  "overlay_gate_matrix": gate_matrix_df.to_dict(orient="records"),
  "near_miss_analysis": near_miss_df.to_dict(orient="records"),
  "closure_summary": closure_summary,
  "closure_checks": closure_checks,
}

pd.DataFrame([closure_summary]).to_csv("research_v17_6_1_closure_summary.csv", index=False)
gate_matrix_df.to_csv("research_v17_6_1_overlay_gate_matrix.csv", index=False)
near_miss_df.to_csv("research_v17_6_1_near_miss_analysis.csv", index=False)
Path("research_v17_6_1_selected_config.json").write_text(
  json.dumps(selected_out, indent=2, default=str),
  encoding="utf-8",
)

EXPORT_FILES = [
  "research_v17_6_1_closure_summary.csv",
  "research_v17_6_1_overlay_gate_matrix.csv",
  "research_v17_6_1_near_miss_analysis.csv",
  "research_v17_6_1_selected_config.json",
]

ZIP_PATH = CONTENT_ROOT / "V17_6_1_CLOSURE.zip"
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
  for name in EXPORT_FILES:
    p = CONTENT_ROOT / name
    if not p.exists():
      p = Path(name)
    if p.exists():
      zf.write(p, p.name)
      print(f"  + {name}")

print("\n" + "=" * 72)
print("V17.6 CLOSED: VALIDATED")
print(f"Official selected overlay: {OFFICIAL_SELECTED_OVERLAY}")
print(f"Forward paper challenger: {PAPER_RESEARCH_CHALLENGER}")
print(f"R2 excess shortfall: {r2_excess_shortfall:.2f} percentage points")
print(f"R2 IR shortfall: {r2_ir_shortfall:.2f}")
print("Backtest reexecuted: False")
print("Model retrained: False")
print("Null test reexecuted: False")
print("APPROVED_FOR_REAL_MONEY=False")
print("=" * 72)
print(f"ZIP creado: {ZIP_PATH}")

if IN_COLAB:
  from google.colab import files
  files.download(str(ZIP_PATH))
  print("Descarga iniciada: V17_6_1_CLOSURE.zip")
