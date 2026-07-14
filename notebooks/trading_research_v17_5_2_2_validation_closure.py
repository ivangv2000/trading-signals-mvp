# %% [markdown]
# # Trading Research V17.5.2.2 — Validation Closure (sin re-ejecutar backtest)
#
# Re-evalúa P1 con criterio financiero de un céntimo leyendo exports V17.5.2.1.
# **No descarga precios, no entrena XGB, no ejecuta null test.**
# **APPROVED_FOR_REAL_MONEY = False siempre.**

# %%
try:
  get_ipython().run_line_magic("pip", "install pandas numpy -q")
except NameError:
  import subprocess, sys
  subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas", "numpy"])

# %% [markdown]
# ## 1. Subir exports V17.5.2.1 (7 archivos)

# %%
import json
import shutil
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
AUDIT_VERSION = "v17_5_2_2"
APPROVED_FOR_REAL_MONEY = False
FINANCIAL_ATOL = 0.01
FINANCIAL_RTOL = 1e-10
EQUITY_CONTINUITY_TOL = 0.01

REQUIRED_FILES = [
  "research_v17_5_2_1_summary.csv",
  "research_v17_5_2_1_validation_gates.csv",
  "research_v17_5_2_1_contribution_reconciliation.csv",
  "research_v17_5_2_1_strategy_results.csv",
  "research_v17_5_2_1_execution_tests.csv",
  "research_v17_5_2_1_current_signals.csv",
  "research_v17_5_2_1_selected_config.json",
]


def _resolve_input_path(name):
  candidates = [CONTENT_ROOT / name, Path(name)]
  for p in candidates:
    if p.exists() and p.stat().st_size > 0:
      return p
  return CONTENT_ROOT / name


def _load_required_exports():
  if IN_COLAB:
    from google.colab import files
    print("Sube los 7 exports de V17.5.2.1:")
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
    raise FileNotFoundError(
      "Faltan exports V17.5.2.1: " + ", ".join(missing)
    )
  print("7 exports obligatorios encontrados OK")


_load_required_exports()

summary_in = pd.read_csv(_resolve_input_path("research_v17_5_2_1_summary.csv"))
gates_in = pd.read_csv(_resolve_input_path("research_v17_5_2_1_validation_gates.csv"))
recon_in = pd.read_csv(_resolve_input_path("research_v17_5_2_1_contribution_reconciliation.csv"))
strategy_in = pd.read_csv(_resolve_input_path("research_v17_5_2_1_strategy_results.csv"))
tests_in = pd.read_csv(_resolve_input_path("research_v17_5_2_1_execution_tests.csv"))
signals_in = pd.read_csv(_resolve_input_path("research_v17_5_2_1_current_signals.csv"))
selected_in = json.loads(_resolve_input_path("research_v17_5_2_1_selected_config.json").read_text(encoding="utf-8"))

# %% [markdown]
# ## 2. Criterio financiero de reconciliación (P1 corregido)

# %%
def evaluate_financial_reconciliation(recon_df):
  rows = []
  for _, r in recon_df.iterrows():
    expected = float(r.get("expected_pnl", np.nan))
    calculated = float(r.get("calculated_pnl", np.nan))
    abs_err = float(r.get("absolute_error", abs(expected - calculated)))
    rel_err = float(r.get("relative_error", np.nan))
    pass_abs = bool(r.get("pass_absolute", abs_err <= FINANCIAL_ATOL))
    max_eq = float(r.get("max_equity_continuity_error", 0.0))
    fin_pass = (
      bool(np.isclose(expected, calculated, atol=FINANCIAL_ATOL, rtol=FINANCIAL_RTOL))
      and pass_abs
      and max_eq <= EQUITY_CONTINUITY_TOL
    )
    rows.append({
      "strategy": r.get("strategy", ""),
      "expected_pnl": round(expected, 6),
      "calculated_pnl": round(calculated, 6),
      "absolute_error": abs_err,
      "relative_error": rel_err,
      "pass_absolute": pass_abs,
      "max_equity_continuity_error": max_eq,
      "financial_reconciliation_pass": fin_pass,
      "legacy_pass": bool(r.get("pass", False)),
    })
  return pd.DataFrame(rows)


closure_df = evaluate_financial_reconciliation(recon_in)
print("Reconciliación financiera por estrategia:")
print(closure_df.to_string(index=False))

assert bool((closure_df["absolute_error"] <= FINANCIAL_ATOL).all()), (
  "absolute_error > 0.01 detectado: "
  + str(closure_df.loc[closure_df["absolute_error"] > FINANCIAL_ATOL, "strategy"].tolist())
)
assert bool((closure_df["max_equity_continuity_error"] <= EQUITY_CONTINUITY_TOL).all()), (
  "max_equity_continuity_error > 0.01 detectado"
)
s5_row = closure_df[closure_df["strategy"].astype(str) == "S5_XGBRANKER"]
assert len(s5_row) == 1, "S5_XGBRANKER no encontrado en contribution_reconciliation"
assert float(s5_row.iloc[0]["absolute_error"]) <= FINANCIAL_ATOL, "S5 absolute_error > 0.01"

assert len(tests_in) == 11, f"Se esperaban 11 tests sintéticos, hay {len(tests_in)}"
assert bool(tests_in["pass"].all()), (
  "Tests sintéticos fallidos: " + str(tests_in.loc[~tests_in["pass"], "test"].tolist())
)

P1_NEW = bool(closure_df["financial_reconciliation_pass"].all())
print(f"\nP1_contribution_reconciles (criterio financiero): {P1_NEW}")

# %% [markdown]
# ## 3. Gates y estado final

# %%
gates_dict = dict(zip(gates_in["gate"], gates_in["pass"].astype(bool)))
gates_dict["P1_contribution_reconciles"] = P1_NEW

ALLOWED_FAIL_ONLY = {
  "G13_max_drawdown_better_than_minus_35pct",
  "G15_top_sector_pnl_below_35pct",
  "G17_preregistered_dsr_above_095",
  "G20_point_in_time_membership_tested",
}

failed_gates = [g for g, v in gates_dict.items() if not v]
unexpected_fail = [g for g in failed_gates if g not in ALLOWED_FAIL_ONLY]
required_pass = all(v for g, v in gates_dict.items() if g not in ALLOWED_FAIL_ONLY)

if required_pass:
  FINAL_STATUS = "READY_FOR_RISK_OVERLAY_RESEARCH"
else:
  FINAL_STATUS = selected_in.get("final_status", summary_in.iloc[0].get("FINAL_STATUS", "FAILED_STATISTICAL_VALIDATION"))

champion = str(summary_in.iloc[0].get("champion", "S5_XGBRANKER"))
champ_row = strategy_in[strategy_in["strategy"].astype(str) == champion]
champ_metrics = champ_row.iloc[0].to_dict() if len(champ_row) else {}

def _metric(name, fallback):
  if name in summary_in.columns:
    v = summary_in.iloc[0][name]
    if pd.notna(v):
      return v
  return champ_metrics.get(name, fallback)

summary_out = {
  "AUDIT_VERSION": AUDIT_VERSION,
  "FINAL_STATUS": FINAL_STATUS,
  "APPROVED_FOR_REAL_MONEY": False,
  "inherited_from": "v17_5_2_1",
  "inherited_champion": champion,
  "inherited_CAGR_pct": round(float(_metric("v1752_1_champion_CAGR", champ_metrics.get("CAGR_pct", 26.90))), 2),
  "inherited_excess_CAGR_vs_B1": round(float(_metric("v1752_1_excess_CAGR_vs_B1", champ_metrics.get("excess_CAGR_vs_equal_weight", 11.44))), 2),
  "inherited_information_ratio_vs_B1": round(float(_metric("v1752_1_information_ratio_vs_B1", champ_metrics.get("information_ratio_vs_equal_weight", 0.726))), 3),
  "inherited_null_p_vs_B1": round(float(_metric("null_corrected_p_vs_B1", 0.001996)), 6),
  "inherited_max_drawdown_pct": round(float(_metric("v1752_1_max_drawdown_pct", champ_metrics.get("max_drawdown_pct", -51.13))), 2),
  "inherited_preregistered_dsr": round(float(_metric("preregistered_dsr", 0.7056)), 4),
  "backtest_reexecuted": False,
  "model_retrained": False,
  "null_test_reexecuted": False,
  "reconciliation_to_cents": True,
  "P1_financial_criterion_pass": P1_NEW,
  "gates_pass": sum(gates_dict.values()),
  "gates_total": len(gates_dict),
  "failed_gates": ",".join(failed_gates) if failed_gates else "",
  "unexpected_failed_gates": ",".join(unexpected_fail) if unexpected_fail else "",
  "next_phase": "RISK_OVERLAY_RESEARCH",
  "closure_note": "P1 re-evaluado con np.isclose(atol=0.01); sin repetir simulacion",
}

gates_out = pd.DataFrame([{"gate": k, "pass": v} for k, v in sorted(gates_dict.items())])
selected_out = {
  "version": AUDIT_VERSION,
  "final_status": FINAL_STATUS,
  "approved_for_real_money": False,
  "inherited_config": "research_v17_5_2_1_selected_config.json",
  "validation_gates": gates_dict,
  "financial_reconciliation": closure_df.to_dict(orient="records"),
  "closure_summary": summary_out,
}

closure_df.to_csv("research_v17_5_2_2_validation_closure.csv", index=False)
gates_out.to_csv("research_v17_5_2_2_validation_gates.csv", index=False)
pd.DataFrame([summary_out]).to_csv("research_v17_5_2_2_summary.csv", index=False)
Path("research_v17_5_2_2_selected_config.json").write_text(
  json.dumps(selected_out, indent=2, default=str), encoding="utf-8")

print("\n" + "=" * 72)
print("VALIDATION CLOSURE V17.5.2.2")
print("=" * 72)
print(f"FINAL_STATUS: {FINAL_STATUS}")
print(f"P1 financial pass: {P1_NEW}")
print(f"Gates: {summary_out['gates_pass']}/{summary_out['gates_total']}")
if failed_gates:
  print("Failed gates:", ", ".join(failed_gates))
print(f"backtest_reexecuted={summary_out['backtest_reexecuted']} | null_test_reexecuted={summary_out['null_test_reexecuted']}")
print(f"next_phase={summary_out['next_phase']}")
print("No se repitió ninguna simulación. APPROVED_FOR_REAL_MONEY=False.")

# %% [markdown]
# ## 4. ZIP y descarga Colab

# %%
ZIP_PATH = CONTENT_ROOT / "V17_5_2_2_VALIDATION_CLOSURE.zip"

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
  for name in (
    "research_v17_5_2_2_validation_closure.csv",
    "research_v17_5_2_2_validation_gates.csv",
    "research_v17_5_2_2_summary.csv",
    "research_v17_5_2_2_selected_config.json",
  ):
    p = CONTENT_ROOT / name
    if p.exists():
      zf.write(p, p.name)
      print(f"  + {name}")

print(f"ZIP creado: {ZIP_PATH}")

if IN_COLAB:
  from google.colab import files
  files.download(str(ZIP_PATH))
  print("Descarga iniciada: V17_5_2_2_VALIDATION_CLOSURE.zip")
else:
  print(f"ZIP listo en: {ZIP_PATH}")
