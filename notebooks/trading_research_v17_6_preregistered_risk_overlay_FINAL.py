# %% [markdown]
# # Trading Research V17.6 — Preregistered Risk Overlay Research
#
# Aplica overlays R0-R3 sobre cartera congelada S5_XGBRANKER.
# **No tuning. No nuevos overlays. APPROVED_FOR_REAL_MONEY = False.**
#
#
# **Revisión integral aplicada:** importación AST segura, sectores GICS, preflight funcional, live inference corregida y PBO alineado.

# %%
try:
  get_ipython().run_line_magic(
    "pip",
    "install yfinance pandas numpy scipy scikit-learn xgboost tqdm lxml html5lib beautifulsoup4 requests pyarrow -q",
  )
except NameError:
  import subprocess, sys
  subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "yfinance", "pandas", "numpy", "scipy", "scikit-learn", "xgboost",
    "tqdm", "lxml", "html5lib", "beautifulsoup4", "requests", "pyarrow"])

# %% [markdown]
# ## 1. Subir motor y configuraciones

# %%
import json
import shutil
import zipfile
from collections import defaultdict
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
  import google.colab  # noqa: F401
  IN_COLAB = True
except ImportError:
  IN_COLAB = False

CONTENT_ROOT = Path("/content") if IN_COLAB else Path.cwd()
CONFIG_DIR = CONTENT_ROOT / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_UPLOADS = {
  "trading_research_v17_2_backtest_integrity_lab.py": CONTENT_ROOT / "trading_research_v17_2_backtest_integrity_lab.py",
  "trading_research_v17_5_2_1_missing_quote_hotfix.py": CONTENT_ROOT / "trading_research_v17_5_2_1_missing_quote_hotfix.py",
  "v17_4_preregistered_experiment.json": CONFIG_DIR / "v17_4_preregistered_experiment.json",
  "v17_5_frozen_audit_config.json": CONFIG_DIR / "v17_5_frozen_audit_config.json",
  "v17_5_1_integrity_patch.json": CONFIG_DIR / "v17_5_1_integrity_patch.json",
  "v17_5_2_execution_contract.json": CONFIG_DIR / "v17_5_2_execution_contract.json",
  "v17_5_2_1_missing_quote_contract.json": CONFIG_DIR / "v17_5_2_1_missing_quote_contract.json",
  "v17_6_preregistered_risk_overlay.json": CONFIG_DIR / "v17_6_preregistered_risk_overlay.json",
}


def _place_uploaded_file(filename, data, dest_map):
  tmp = CONTENT_ROOT / filename
  tmp.write_bytes(data)
  if filename not in dest_map:
    print(f"Ignorado: {filename}")
    return
  dest = dest_map[filename]
  dest.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(tmp, dest)
  print(f"Colocado: {dest}")


if IN_COLAB:
  from google.colab import files
  print("Sube motor V17.2, motor V17.5.2.1 y configs (8 archivos):")
  for f in REQUIRED_UPLOADS:
    print(f"  - {f}")
  uploaded = files.upload()
  for fname, data in uploaded.items():
    _place_uploaded_file(fname, data, REQUIRED_UPLOADS)
else:
  print("Ejecucion local: se usan archivos del proyecto.")

missing = [n for n, d in REQUIRED_UPLOADS.items() if not d.exists() or d.stat().st_size <= 0]
if missing:
  raise FileNotFoundError("Faltan archivos obligatorios: " + ", ".join(missing))
print("Archivos obligatorios OK")

# %% [markdown]
# ## 2. Configuracion V17.6 + motor congelado

# %%
import ast
import hashlib
import math
import re
import warnings

warnings.filterwarnings("ignore")
from scipy import stats
from tqdm.auto import tqdm

AUDIT_VERSION = "v17_6"
OVERLAY_CONFIG_PATH = Path("config/v17_6_preregistered_risk_overlay.json")
PATCH_PATH = Path("config/v17_5_1_integrity_patch.json")
EXEC_CONTRACT_PATH = Path("config/v17_5_2_execution_contract.json")
MISSING_QUOTE_PATH = Path("config/v17_5_2_1_missing_quote_contract.json")
CONFIG_PATH = Path("config/v17_5_frozen_audit_config.json")
PREREG_PATH = Path("config/v17_4_preregistered_experiment.json")

overlay_cfg = json.loads(OVERLAY_CONFIG_PATH.read_text(encoding="utf-8"))
patch = json.loads(PATCH_PATH.read_text(encoding="utf-8")) if PATCH_PATH.exists() else {}
exec_contract = json.loads(EXEC_CONTRACT_PATH.read_text(encoding="utf-8")) if EXEC_CONTRACT_PATH.exists() else {}
missing_quote_contract = json.loads(MISSING_QUOTE_PATH.read_text(encoding="utf-8")) if MISSING_QUOTE_PATH.exists() else {}
cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
fp = cfg["frozen_parameters"]
rg = cfg["risk_gates"]
sp = overlay_cfg["starting_point"]
sel_rules = overlay_cfg["selection_rules"]
ov_params = overlay_cfg["overlay_parameters"]

PRIMARY_BENCHMARK = patch.get("primary_benchmark", "B1_EQUAL_WEIGHT_ALL_ASOF")
LIVE_STRATEGY_SOURCE = patch.get("live_strategy_source", "S5_XGBRANKER_LIVE")
RECON_ABS_TOL = float(exec_contract.get("reconciliation_tolerance", {}).get("absolute_usd", 0.01))
EQUITY_CONTINUITY_TOL = float(missing_quote_contract.get("equity_continuity_tolerance_usd", 0.01))

START_DATE = fp["START_DATE"]
END_DATE = None
TRANSACTION_COST = fp["TRANSACTION_COST"]
SLIPPAGE = fp["SLIPPAGE"]
FORWARD_HORIZON = fp["FORWARD_HORIZON"]
PURGE_DAYS = fp["PURGE_DAYS"]
EMBARGO_DAYS = fp["EMBARGO_DAYS"]
RANDOM_SEED = fp["RANDOM_SEED"]
APPROVED_FOR_REAL_MONEY = False
INITIAL_CAPITAL = 10000
COST_RATE = TRANSACTION_COST + SLIPPAGE
MARKET = "SPY"
CASH_ASSET = "SHY"
WF_START_YEAR = fp["WF_START_YEAR"]
RESEARCH_START = fp["RESEARCH_START"]
RESEARCH_END = fp["RESEARCH_END"]
REUSED_TEST_START = fp["REUSED_TEST_START"]
PAPER_TRADING_START = fp["PAPER_TRADING_START"]
N_TRIALS_ACCUMULATED = fp["N_TRIALS_ACCUMULATED"]
N_PREREG_STRATEGIES = fp["N_PREREGISTERED_STRATEGIES"]
N_OVERLAY_TRIALS = overlay_cfg["dsr_audit"]["n_overlay_trials_added"]
N_DSR_TRIALS_OVERLAY = overlay_cfg["dsr_audit"]["total_trials_for_overlay_dsr"]
N_NULL_OVERLAY = overlay_cfg["null_test"]["n_permutations"]

MAX_TICKERS_FULL = fp["TOP_K_FULL"] * 10
TOP_K_FULL = fp["TOP_K_FULL"]
TOP_K = TOP_K_FULL
BUY_RANK = fp["BUY_RANK"]
HOLD_UNTIL_RANK = fp["HOLD_UNTIL_RANK"]
MAX_STOCK_WEIGHT = fp["MAX_STOCK_WEIGHT"]
MAX_SECTOR_WEIGHT = fp["MAX_SECTOR_WEIGHT"]
MIN_HISTORY_DAYS = 1000
MIN_DOLLAR_VOLUME = 20_000_000
POINT_IN_TIME_MEMBERSHIP = overlay_cfg.get("POINT_IN_TIME_MEMBERSHIP", False)
SURVIVORSHIP_WARNING = overlay_cfg.get("survivorship_warning", cfg["survivorship_warning"])
FROZEN_CHAMPION = overlay_cfg["frozen_base_strategy"]
OVERLAY_VARIANTS = overlay_cfg["overlays_preregistered"]
PERIOD_RESEARCH = overlay_cfg["period_labels"]["research"]
PERIOD_REUSED = overlay_cfg["period_labels"]["reused_test"]
PERIOD_FULL = overlay_cfg["period_labels"]["full"]
FINAL_STATUS = "PENDING"

DOWNLOAD_CACHE_DIR = Path(cfg["cache_reuse"]["download_dir"])
CHECKPOINT_V1741 = Path(cfg["cache_reuse"]["checkpoint_dir"])
PIT_MEMBERSHIP_PATH = Path("data/sp500_historical_membership.csv")

XGB_PARAMS = {
  "objective": "rank:ndcg", "eval_metric": "ndcg@25", "max_depth": 3,
  "learning_rate": 0.03, "n_estimators": 300, "subsample": 0.80,
  "colsample_bytree": 0.70, "min_child_weight": 10, "reg_alpha": 1.0,
  "reg_lambda": 5.0, "random_state": RANDOM_SEED, "n_jobs": -1, "verbosity": 0,
}

BROAD_MARKET_ETFS = {"SPY", "QQQ", "IWM", "DIA"}
DEFENSIVE_ETFS = {"SHY", "IEF", "TLT", "GLD", "LQD", "USMV", "QUAL"}
SECTOR_ETFS = {"XLK", "XLV", "XLF", "XLE", "XLY", "XLP", "XLI", "XLU", "XLC", "XLB", "XLRE"}
FACTOR_ETFS = {"MTUM", "QUAL", "USMV", "VLUE", "SPLV", "SPHB", "SCHD"}
ALL_ETFS = BROAD_MARKET_ETFS | DEFENSIVE_ETFS | SECTOR_ETFS | FACTOR_ETFS
BENCHMARK_ETFS = sorted(BROAD_MARKET_ETFS | {CASH_ASSET})
SECTOR_MAP = {**{e: "ETF" for e in ALL_ETFS}}
FORBIDDEN_FEATURE_PATTERNS = re.compile(
  r"fwd|future|label|target|exit|next|forward|realized|pnl|return_after|barrier", re.I
)
np.random.seed(RANDOM_SEED)

R1_TARGET_VOL = ov_params["R1_VOL_TARGET_12"]["target_vol_annual"]
R1_LOOKBACK = ov_params["R1_VOL_TARGET_12"]["lookback_sessions"]
R1_MIN_EXP = ov_params["R1_VOL_TARGET_12"]["min_exposure"]
R2_ON = ov_params["R2_SPY_TREND"]["risk_on_exposure"]
R2_OFF = ov_params["R2_SPY_TREND"]["risk_off_exposure"]
R2_SMA_WINDOW = ov_params["R2_SPY_TREND"]["sma_window"]
R2_SMA_SHIFT = ov_params["R2_SPY_TREND"]["sma_shift_days"]
R3_MIN_EXP = ov_params["R3_COMBINED_FIXED"]["min_exposure"]

# %% [markdown]
# ## 3. Cargar motor V17.2 + bloques V17.5.2.1

# %%
try:
  _NB_DIR = Path(__file__).parent
except NameError:
  _NB_DIR = CONTENT_ROOT

V172 = _NB_DIR / "trading_research_v17_2_backtest_integrity_lab.py"
if not V172.exists():
  V172 = CONTENT_ROOT / "trading_research_v17_2_backtest_integrity_lab.py"

V17521 = _NB_DIR / "trading_research_v17_5_2_1_missing_quote_hotfix.py"
if not V17521.exists():
  V17521 = CONTENT_ROOT / "trading_research_v17_5_2_1_missing_quote_hotfix.py"

assert V172.exists(), f"No existe {V172}"
assert V17521.exists(), f"No existe {V17521}"

# Contrato explícito que las funciones del ledger necesitan como globals.
_contract = exec_contract.get("execution_contract", exec_contract)
_recon_cfg = exec_contract.get("reconciliation_tolerance", {})
SIGNAL_TIME = _contract.get("SIGNAL_TIME", "AFTER_CLOSE")
EXECUTION_TIME = _contract.get("EXECUTION_TIME", "NEXT_TRADING_DAY_OPEN")
VALUATION_TIME = _contract.get("VALUATION_TIME", "DAILY_CLOSE")
RECON_REL_TOL = float(_recon_cfg.get("relative", 1e-10))


def _load_named_function_from_source(source_path, function_name):
  """Carga un FunctionDef aislado sin ejecutar código top-level del archivo."""
  src = Path(source_path).read_text(encoding="utf-8")
  mod = ast.parse(src)
  for node in mod.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
      isolated = ast.Module(body=[node], type_ignores=[])
      ast.fix_missing_locations(isolated)
      exec(compile(isolated, str(source_path), "exec"), globals())
      fn = globals().get(function_name)
      if callable(fn):
        return fn
  raise RuntimeError(f"FunctionDef no encontrado en {source_path}: {function_name}")


def _exec_function_range(source_path, start_function, end_function_exclusive, label=""):
  """Carga solo FunctionDef comprendidos entre dos funciones; nunca ejecuta top-level."""
  src = Path(source_path).read_text(encoding="utf-8")
  mod = ast.parse(src)
  names = [n.name for n in mod.body if isinstance(n, ast.FunctionDef)]
  if start_function not in names:
    raise RuntimeError(f"Inicio ausente ({label}): {start_function}")
  if end_function_exclusive not in names:
    raise RuntimeError(f"Fin ausente ({label}): {end_function_exclusive}")
  start_i = names.index(start_function)
  end_i = names.index(end_function_exclusive)
  if end_i <= start_i:
    raise RuntimeError(f"Rango inválido ({label}): {start_function} -> {end_function_exclusive}")
  wanted = set(names[start_i:end_i])
  nodes = [n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
  isolated = ast.Module(body=nodes, type_ignores=[])
  ast.fix_missing_locations(isolated)
  exec(compile(isolated, str(source_path), "exec"), globals())
  print(f"Funciones cargadas ({label}): {len(nodes)}")


# Funciones básicas V17.2, aisladas del código top-level.
_src172 = V172.read_text(encoding="utf-8")
_mod172 = ast.parse(_src172)
_v172_needed = {
  "_clean_symbol", "download_data", "classify_assets", "build_features_and_panel",
  "make_signal_dates", "_master_calendar", "_next_trading_day",
  "calculate_metrics_from_equity",
}
for _node in _mod172.body:
  if isinstance(_node, ast.FunctionDef) and _node.name in _v172_needed:
    isolated = ast.Module(body=[_node], type_ignores=[])
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, str(V172), "exec"), globals())

missing_v172 = sorted(n for n in _v172_needed if not callable(globals().get(n)))
assert not missing_v172, f"Funciones V17.2 no cargadas: {missing_v172}"

# Motor V17.5.2.1: dependencias explícitas y bloques de funciones solamente.
_load_named_function_from_source(V17521, "_sector_cap_key")
_load_named_function_from_source(V17521, "allocate_weights_with_hard_caps")
_load_named_function_from_source(V17521, "audit_weight_caps")
_load_named_function_from_source(V17521, "_sector_label")
_load_named_function_from_source(V17521, "sector_pnl_from_asset")

_exec_function_range(V17521, "_ledger_price", "_synth_df", "single_ledger")
_exec_function_range(V17521, "trailing_dollar_vol_asof", "calculate_period_metrics_from_returns", "universe")
_exec_function_range(V17521, "calculate_period_metrics_from_returns", "_sector_label", "metrics")
_exec_function_range(V17521, "_select_buffered", "build_s6_hardcaps", "s5")
_exec_function_range(
  V17521,
  "build_equal_weight_all_asof_benchmark",
  "build_b2_cap_matched_random_benchmark",
  "benchmark",
)

DSR_FORMULA_VERSION = "bailey_lopez_de_prado_psr_v2_frequency_matched"
DSR_V1741_BUG = (
  "V17.4.1 dsr_prob() pasaba Sharpe ANUALIZADO (mean/std*sqrt(252)) a una formula "
  "derivada para Sharpe POR-PERIODO, inflando SR0 y colapsando DSR (~0.04)."
)

_exec_function_range(V17521, "_returns_to_period_series", "run_dsr_unit_tests", "dsr")
_load_named_function_from_source(V17521, "simplified_pbo")
_load_named_function_from_source(V17521, "corrected_empirical_p")

# Resolver de dependencias de funciones entre V17.2 y V17.5.2.1.
# Solo sigue nombres que son FunctionDef reales; no confunde métodos pandas
# como append/groupby/iloc con dependencias globales.
def _load_recursive_function_dependencies(root_names):
  catalogs = {}

  # V17.2 primero; V17.5.2.1 tiene prioridad si un nombre se repite.
  for source_path in (V172, V17521):
    source = Path(source_path).read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        catalogs[node.name] = (Path(source_path), node)

  selected = set()
  pending = list(root_names)

  while pending:
    name = pending.pop()
    if name in selected or name not in catalogs:
      continue
    selected.add(name)
    _, node = catalogs[name]
    referenced_names = {
      child.id
      for child in ast.walk(node)
      if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }
    pending.extend(sorted(referenced_names.intersection(catalogs)))

  # Las funciones se pueden definir en cualquier orden; las referencias se
  # resuelven al ejecutarlas. Se cargan todas en el mismo namespace global.
  for name in sorted(selected):
    source_path, node = catalogs[name]
    isolated = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, str(source_path), "exec"), globals())

  return sorted(selected)


_DEPENDENCY_ROOTS = {
  "simulate_portfolio_with_single_ledger",
  "allocate_weights_with_hard_caps",
  "audit_weight_caps",
  "build_s5_hardcaps",
  "build_equal_weight_all_asof_benchmark",
  "calculate_metrics_from_equity",
  "period_full_metrics",
  "calculate_dsr_audit",
  "simplified_pbo",
  "corrected_empirical_p",
}

_loaded_dependency_functions = _load_recursive_function_dependencies(
  _DEPENDENCY_ROOTS
)
print(
  "Dependencias funcionales cargadas:",
  len(_loaded_dependency_functions),
)

# Inyección explícita de todos los globals reales del motor.
_LEDGER_GLOBALS = {
  "pd": pd,
  "np": np,
  "math": math,
  "defaultdict": defaultdict,
  "CASH_ASSET": CASH_ASSET,
  "DEFENSIVE_ETFS": DEFENSIVE_ETFS,
  "SECTOR_MAP": SECTOR_MAP,
  "ALL_ETFS": ALL_ETFS,
  "INITIAL_CAPITAL": INITIAL_CAPITAL,
  "TRANSACTION_COST": TRANSACTION_COST,
  "SLIPPAGE": SLIPPAGE,
  "COST_RATE": COST_RATE,
  "RECON_ABS_TOL": RECON_ABS_TOL,
  "RECON_REL_TOL": RECON_REL_TOL,
  "EQUITY_CONTINUITY_TOL": EQUITY_CONTINUITY_TOL,
  "SIGNAL_TIME": SIGNAL_TIME,
  "EXECUTION_TIME": EXECUTION_TIME,
  "VALUATION_TIME": VALUATION_TIME,
  "_master_calendar": _master_calendar,
  "_next_trading_day": _next_trading_day,
  "_sector_label": _sector_label,
  "sector_pnl_from_asset": sector_pnl_from_asset,
  "DSR_FORMULA_VERSION": DSR_FORMULA_VERSION,
  "DSR_V1741_BUG": DSR_V1741_BUG,
}

for _fn_name in sorted(set(_loaded_dependency_functions).union({
    "_ledger_price",
    "simulate_portfolio_with_single_ledger",
    "allocate_weights_with_hard_caps",
    "audit_weight_caps",
    "build_s5_hardcaps",
    "build_equal_weight_all_asof_benchmark",
    "calculate_metrics_from_equity",
    "period_full_metrics",
    "calculate_dsr_audit",
})):
  _fn = globals().get(_fn_name)
  if callable(_fn):
    _fn.__globals__.update(_LEDGER_GLOBALS)
    _fn.__globals__.update({
      "_sector_cap_key": _sector_cap_key,
      "MAX_STOCK_WEIGHT": MAX_STOCK_WEIGHT,
      "MAX_SECTOR_WEIGHT": MAX_SECTOR_WEIGHT,
      "TOP_K": TOP_K,
      "BUY_RANK": BUY_RANK,
      "HOLD_UNTIL_RANK": HOLD_UNTIL_RANK,
      "RESEARCH_START": RESEARCH_START,
      "RESEARCH_END": RESEARCH_END,
      "PRIMARY_BENCHMARK": PRIMARY_BENCHMARK,
      "ASOF_UNIVERSE": globals().get("ASOF_UNIVERSE", {}),
      "stats": stats,
      "RANDOM_SEED": RANDOM_SEED,
    })

_REQUIRED_CALLABLES = [
  "_clean_symbol", "download_data", "classify_assets", "build_features_and_panel",
  "make_signal_dates", "_master_calendar", "_next_trading_day",
  "allocate_weights_with_hard_caps",
  "audit_weight_caps", "simulate_portfolio_with_single_ledger",
  "calculate_metrics_from_equity", "build_s5_hardcaps",
  "build_equal_weight_all_asof_benchmark", "period_full_metrics",
  "calculate_dsr_audit", "simplified_pbo", "corrected_empirical_p",
]
_missing = [n for n in _REQUIRED_CALLABLES if not callable(globals().get(n))]
assert not _missing, f"Motor incompleto antes de descargar datos: {_missing}"

# Smoke test del ledger ANTES de descargar 499 activos o entrenar XGB.
# Así cualquier helper ausente falla en segundos y no después del walk-forward.
_smoke_idx = pd.bdate_range("2020-01-02", periods=5)
_smoke_data = {
  "AAA": pd.DataFrame({
    "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
    "High": [101.0, 102.0, 103.0, 104.0, 105.0],
    "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
    "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
    "Volume": [1_000_000] * 5,
  }, index=_smoke_idx),
  CASH_ASSET: pd.DataFrame({
    "Open": [80.0] * 5,
    "High": [80.0] * 5,
    "Low": [80.0] * 5,
    "Close": [80.0] * 5,
    "Volume": [1_000_000] * 5,
  }, index=_smoke_idx),
}
_smoke_targets = {
  _smoke_idx[0]: pd.Series({"AAA": 0.50, CASH_ASSET: 0.50})
}
_smoke_result = simulate_portfolio_with_single_ledger(
  _smoke_targets,
  _smoke_data,
  initial_capital=10_000,
  transaction_cost=TRANSACTION_COST,
  slippage=SLIPPAGE,
  strategy_name="EARLY_LEDGER_SMOKE",
)
assert len(_smoke_result.get("equity", [])) > 0
assert _smoke_result.get("reconciliation", {}).get(
  "absolute_error", 1.0
) <= RECON_ABS_TOL
print("EARLY LEDGER SMOKE: PASS")

print("MOTOR V17.6 CARGADO: funciones y dependencias explícitas OK")

# %% [markdown]
# ## 4. Datos + S5 congelado (sin modificar modelo)

# %%
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_GITHUB_CSV = (
  "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
)
SP500_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TradingResearchV176/1.0)"}
SP500_HTTP_TIMEOUT = 30


def _pick_sp500_column(df, *names):
  for name in names:
    if name in df.columns:
      return name
  return None


def load_sp500_constituents():
  """Devuelve tickers actuales y mapa GICS. Sigue existiendo survivorship bias."""
  try:
    tables = pd.read_html(SP500_WIKI_URL, storage_options=SP500_HTTP_HEADERS)
    df = tables[0]
  except Exception as e:
    print(f"Wikipedia fallback: {e}")
    r = requests.get(
      SP500_GITHUB_CSV,
      headers=SP500_HTTP_HEADERS,
      timeout=SP500_HTTP_TIMEOUT,
    )
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))

  sym_col = _pick_sp500_column(df, "Symbol", "Ticker symbol", "Ticker")
  sec_col = _pick_sp500_column(df, "GICS Sector", "Sector")
  if sym_col is None:
    raise ValueError("No se encontró la columna de ticker del S&P 500")

  tickers = [_clean_symbol(s) for s in df[sym_col].tolist()]
  tickers = [t for t in tickers if t]

  sector_map = {}
  if sec_col is not None:
    for raw_ticker, raw_sector in zip(df[sym_col], df[sec_col]):
      ticker = _clean_symbol(raw_ticker)
      sector = str(raw_sector).strip() if pd.notna(raw_sector) else "OTHER"
      if ticker:
        sector_map[ticker] = sector or "OTHER"

  return sorted(set(tickers)), sector_map


def load_data_from_cache(tickers, extra=None):
  extra = extra or []
  data, missing = {}, []
  DOWNLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

  for t in sorted(set(tickers) | set(extra)):
    p = DOWNLOAD_CACHE_DIR / f"{t}.parquet"
    if p.exists() and p.stat().st_size > 0:
      try:
        data[t] = pd.read_parquet(p)
        continue
      except Exception as exc:
        print(f"Cache inválida {t}: {exc}")
    missing.append(t)

  if missing:
    fetched, _, _ = download_data(missing, START_DATE, END_DATE)
    data.update(fetched)
    for ticker, df in fetched.items():
      try:
        df.to_parquet(DOWNLOAD_CACHE_DIR / f"{ticker}.parquet")
      except Exception as exc:
        print(f"No se pudo guardar cache {ticker}: {exc}")

  if not data and CHECKPOINT_V1741.exists():
    cp = CHECKPOINT_V1741 / "phase_02_universe.pkl"
    if cp.exists():
      blob = pd.read_pickle(cp)
      data = blob.get("data_dict", {})
      print(f"Checkpoint data_dict ({len(data)} tickers)")

  if not data:
    raise FileNotFoundError(
      "Sin datos. Ejecuta V17.4.1 o coloca parquets en "
      "cache/v17_4_1_full_250/downloads/"
    )
  return data


sp_tickers, sp_sector_map = load_sp500_constituents()
SECTOR_MAP.update(sp_sector_map)
SECTOR_MAP.update({e: "ETF" for e in ALL_ETFS})

data_dict = load_data_from_cache(sp_tickers, BENCHMARK_ETFS)
for t, df in data_dict.items():
  for c in ("Open", "High", "Low", "Close", "Volume"):
    if c in df.columns:
      df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

close_all = pd.DataFrame(
  {k: v["Close"] for k, v in data_dict.items()}
).sort_index().ffill()

print(
  f"data_dict={len(data_dict)} | sectores={len(set(sp_sector_map.values()))} | "
  f"{SURVIVORSHIP_WARNING}"
)

STOCKS, ETFS, n_sectors = classify_assets(list(close_all.columns))
STOCKS = [s for s in STOCKS if s in sp_tickers]
assert len(STOCKS) >= 200, f"Universo insuficiente: {len(STOCKS)} stocks"

# Sincronizar globals de todas las funciones importadas.
_RUNTIME_GLOBALS = {
  "data_dict": data_dict,
  "STOCKS": STOCKS,
  "ETFS": ETFS,
  "SECTOR_MAP": SECTOR_MAP,
  "SIGNAL_FREQUENCY": "W-FRI",
}
globals().update(_RUNTIME_GLOBALS)
for _obj in list(globals().values()):
  if callable(_obj) and hasattr(_obj, "__globals__"):
    _obj.__globals__.update(_RUNTIME_GLOBALS)

print(
  f"Namespace sincronizado: {len(STOCKS)} stocks, "
  f"{len(data_dict)} activos, {n_sectors} sectores"
)

RUNTIME_CACHE_DIR = CONTENT_ROOT / "cache" / "v17_6_runtime"
RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_CACHE = RUNTIME_CACHE_DIR / "live_feature_panel.parquet"
MODELING_CACHE = RUNTIME_CACHE_DIR / "modeling_panel.parquet"
OOS_CACHE = RUNTIME_CACHE_DIR / "oos_xgb.parquet"
OOS_STATS_CACHE = RUNTIME_CACHE_DIR / "oos_stats.json"

# Mantiene exactamente la codificación de relevancia utilizada por V17.5.2.1:
# rank(pct=True).astype(int), es decir, etiquetas 0/1 válidas en XGBoost >= 3.
XGB_RELEVANCE_SCHEMA = "v17_5_2_1_frozen_pct_binary_v1"

if FEATURE_CACHE.exists():
  live_feature_panel = pd.read_parquet(FEATURE_CACHE)
  live_feature_panel["signal_date"] = pd.to_datetime(live_feature_panel["signal_date"])
  print(f"Feature cache: {len(live_feature_panel)} filas")
else:
  live_feature_panel = build_features_and_panel(data_dict, STOCKS)
  live_feature_panel.to_parquet(FEATURE_CACHE, index=False)


def attach_forward_labels(feat_panel, data_dict):
  if feat_panel.empty:
    return feat_panel
  rows = []
  for _, r in feat_panel.iterrows():
    t, sig = r["ticker"], pd.Timestamp(r["signal_date"])
    if t not in data_dict:
      continue
    idx = data_dict[t].index
    entry = _next_trading_day(idx, sig)
    if pd.isna(entry):
      rows.append({
        **r.to_dict(),
        "entry_date": pd.NaT,
        "label_end_date": pd.NaT,
        "fwd_ret_20d": np.nan,
        "feature_date": sig,
      })
      continue
    ep = data_dict[t].loc[entry, "Open"] if entry in data_dict[t].index else np.nan
    pos = idx.searchsorted(entry, side="left")
    exit_i = pos + FORWARD_HORIZON
    if exit_i >= len(idx):
      rows.append({
        **r.to_dict(),
        "entry_date": entry,
        "label_end_date": pd.NaT,
        "fwd_ret_20d": np.nan,
        "feature_date": sig,
      })
      continue
    exit_d = idx[exit_i]
    xp = (
      data_dict[t].loc[exit_d, "Open"]
      if "Open" in data_dict[t].columns
      else data_dict[t].loc[exit_d, "Close"]
    )
    fwd = xp / ep - 1 if ep > 0 and np.isfinite(xp) else np.nan
    rows.append({
      **r.to_dict(),
      "entry_date": entry,
      "label_end_date": exit_d,
      "fwd_ret_20d": fwd,
      "feature_date": sig,
    })
  p = pd.DataFrame(rows)
  if len(p):
    p["fwd_excess_20d"] = (
      p["fwd_ret_20d"]
      - p.groupby("signal_date")["fwd_ret_20d"].transform("mean")
    )
    p["relevance"] = p.groupby("signal_date")["fwd_excess_20d"].rank(pct=True)
  return p


def select_features_train(tr, feats, max_feats=12):
  feats = [f for f in feats if f in tr.columns]
  if not feats:
    raise ValueError("No existen features válidas para XGB")
  med = tr[feats].median()
  tr = tr.copy()
  tr[feats] = tr[feats].fillna(med)
  corr = tr[feats].corr().abs()
  keep = list(feats)
  drop = set()
  for i, a in enumerate(keep):
    for b in keep[i + 1:]:
      if a in corr.columns and b in corr.columns and corr.loc[a, b] > 0.95:
        drop.add(b)
  return [f for f in keep if f not in drop][:max_feats], med


def build_frozen_xgb_relevance_labels(frame):
  """
  Reproduce sin cambios la etiqueta del S5 congelado de V17.5.2.1.

  El código original usaba rank(pct=True).astype(int), produciendo
  relevancias enteras 0/1. XGBoost 3.x rechaza grados >31 cuando NDCG usa
  ganancia exponencial, por lo que NO se debe sustituir por rank(method="first").
  """
  required = {"signal_date", "fwd_excess_20d"}
  missing = required - set(frame.columns)
  if missing:
    raise ValueError(
      "Faltan columnas para relevance labels: " + ", ".join(sorted(missing))
    )

  labels = (
    frame.groupby("signal_date", sort=False)["fwd_excess_20d"]
    .rank(pct=True)
    .fillna(0.0)
    .astype(np.int32)
  )

  if len(labels):
    min_label = int(labels.min())
    max_label = int(labels.max())
    if min_label < 0 or max_label > 31:
      raise ValueError(
        f"Relevance labels incompatibles con XGBoost NDCG: "
        f"min={min_label}, max={max_label}"
      )

  return labels


def run_xgb_relevance_compatibility_test():
  """Falla en segundos si la versión instalada de XGBoost no acepta el contrato."""
  import xgboost as xgb

  probe_x = pd.DataFrame({
    "f1": [0.1, 0.2, 0.3, 0.4, 0.2, 0.1, 0.4, 0.3],
    "f2": [1.0, 0.8, 0.6, 0.4, 0.3, 0.5, 0.7, 0.9],
  })
  probe_y = np.asarray([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.int32)

  probe_params = dict(XGB_PARAMS)
  probe_params.update({
    "n_estimators": 1,
    "n_jobs": 1,
    "verbosity": 0,
  })

  probe_model = xgb.XGBRanker(**probe_params)
  probe_model.fit(
    probe_x,
    probe_y,
    group=np.asarray([4, 4], dtype=np.uint32),
  )

  pred = probe_model.predict(probe_x)
  assert len(pred) == len(probe_x)
  print(
    "XGB RANKING COMPATIBILITY: PASS | "
    f"schema={XGB_RELEVANCE_SCHEMA} | labels=0..1"
  )


def walk_forward_xgb_oos(labeled_df, feature_cols, params=None):
  try:
    import xgboost as xgb
  except ImportError as exc:
    raise RuntimeError("xgboost no está instalado") from exc

  params = params or XGB_PARAMS
  oos_preds, oos_ics = [], []

  for test_year in sorted(labeled_df["signal_date"].dt.year.unique()):
    if test_year < WF_START_YEAR:
      continue
    test_start = pd.Timestamp(f"{test_year}-01-01")
    purge_cut = test_start - pd.Timedelta(days=PURGE_DAYS)
    embargo = test_start - pd.Timedelta(days=EMBARGO_DAYS)

    tr = labeled_df[
      (labeled_df["signal_date"] < embargo)
      & (labeled_df["label_end_date"] < purge_cut)
    ].copy()
    te = labeled_df[labeled_df["signal_date"].dt.year == test_year].copy()

    if len(tr) < 80 or len(te) < 10:
      continue

    tr = tr.sort_values(["signal_date", "ticker"]).reset_index(drop=True)
    te = te.sort_values(["signal_date", "ticker"]).reset_index(drop=True)

    sel_feats, med = select_features_train(tr, feature_cols)
    tr[sel_feats] = tr[sel_feats].fillna(med)
    te[sel_feats] = te[sel_feats].fillna(med)

    # Contrato congelado V17.5.2.1: relevancia binaria 0/1.
    # No usar rank(method="first"), porque genera etiquetas 1..N y
    # XGBoost >=3 limita NDCG exponencial a grados <=31.
    y_tr = build_frozen_xgb_relevance_labels(tr)
    group_sizes = (
      tr.groupby("signal_date", sort=False)
      .size()
      .to_numpy(dtype=np.uint32)
    )

    assert int(group_sizes.sum()) == len(tr)
    assert int(y_tr.min()) >= 0
    assert int(y_tr.max()) <= 31

    mdl = xgb.XGBRanker(**params)
    mdl.fit(
      tr[sel_feats],
      y_tr.to_numpy(dtype=np.int32),
      group=group_sizes,
    )

    pred_te = mdl.predict(te[sel_feats])
    ic = stats.spearmanr(pred_te, te["fwd_excess_20d"]).correlation
    oos_ics.append(ic)

    te["xgb_oos_prediction"] = pred_te
    oos_preds.append(
      te[["ticker", "signal_date", "xgb_oos_prediction", "composite_score"]]
    )

  oos_df = (
    pd.concat(oos_preds, ignore_index=True)
    if oos_preds
    else pd.DataFrame(
      columns=["ticker", "signal_date", "xgb_oos_prediction", "composite_score"]
    )
  )
  stats_out = {
    "ranker_oos_ic_mean": (
      round(float(np.nanmean(oos_ics)), 4)
      if oos_ics
      else np.nan
    ),
    "xgb_relevance_schema": XGB_RELEVANCE_SCHEMA,
    "xgb_relevance_max_label": 1,
  }
  return oos_df, stats_out


if MODELING_CACHE.exists():
  modeling_panel = pd.read_parquet(MODELING_CACHE)
  for _c in ("signal_date", "entry_date", "label_end_date", "feature_date"):
    if _c in modeling_panel.columns:
      modeling_panel[_c] = pd.to_datetime(modeling_panel[_c])
  print(f"Modeling cache: {len(modeling_panel)} filas")
else:
  modeling_panel = attach_forward_labels(live_feature_panel, data_dict)
  modeling_panel.to_parquet(MODELING_CACHE, index=False)

labeled_panel = modeling_panel[modeling_panel["fwd_ret_20d"].notna()].copy()

FEATURE_COLS = [
  c for c in live_feature_panel.columns
  if c not in {
    "ticker", "signal_date", "sector", "entry_date", "label_end_date",
    "feature_date", "fwd_ret_20d", "fwd_excess_20d", "relevance",
  }
  and not FORBIDDEN_FEATURE_PATTERNS.search(c)
]

weekly_dates = make_signal_dates("weekly")
signal_dates = weekly_dates[weekly_dates >= pd.Timestamp(RESEARCH_START)]

asof_pack = build_asof_universe_by_rebalance_date(
  data_dict,
  sp_tickers,
  signal_dates,
  max_tickers=250,
)
ASOF_UNIVERSE = asof_pack["universe_by_date"]

# Actualizar globals que dependen de ASOF_UNIVERSE.
for _fn_name in (
  "build_s5_hardcaps",
  "build_equal_weight_all_asof_benchmark",
):
  globals()[_fn_name].__globals__["ASOF_UNIVERSE"] = ASOF_UNIVERSE

print(
  f"Panel={len(live_feature_panel)} | "
  f"signal_dates={len(signal_dates)} | as-of={len(ASOF_UNIVERSE)}"
)

run_xgb_relevance_compatibility_test()

_oos_cache_valid = False
ranker_stats = {}

if OOS_CACHE.exists() and OOS_STATS_CACHE.exists():
  try:
    ranker_stats = json.loads(OOS_STATS_CACHE.read_text(encoding="utf-8"))
    _oos_cache_valid = (
      ranker_stats.get("xgb_relevance_schema")
      == XGB_RELEVANCE_SCHEMA
    )
    if not _oos_cache_valid:
      print(
        "OOS cache ignorada: esquema de relevance antiguo o desconocido"
      )
  except Exception as exc:
    print(f"OOS cache metadata inválida: {exc}")
    _oos_cache_valid = False

if _oos_cache_valid:
  oos_xgb = pd.read_parquet(OOS_CACHE)
  oos_xgb["signal_date"] = pd.to_datetime(oos_xgb["signal_date"])
  print(
    f"OOS cache compatible: {len(oos_xgb)} filas | "
    f"schema={XGB_RELEVANCE_SCHEMA}"
  )
else:
  print("Walk-forward XGB OOS (congelado)...")
  oos_xgb, ranker_stats = walk_forward_xgb_oos(
    labeled_panel,
    FEATURE_COLS,
  )
  oos_xgb.to_parquet(OOS_CACHE, index=False)
  OOS_STATS_CACHE.write_text(
    json.dumps(ranker_stats, indent=2, default=str),
    encoding="utf-8",
  )

print(
  f"OOS rows={len(oos_xgb)} "
  f"IC={ranker_stats.get('ranker_oos_ic_mean')}"
)
assert len(oos_xgb) > 0, "Walk-forward no generó predicciones OOS"

base_s5_targets = build_s5_hardcaps(signal_dates, oos_xgb)
assert len(base_s5_targets) > 0, "S5 no generó pesos históricos"
print(f"S5 base targets: {len(base_s5_targets)} fechas")


def run_from_targets(name, targets):
  sim = simulate_portfolio_with_single_ledger(
    targets,
    data_dict,
    strategy_name=name,
    transaction_cost=TRANSACTION_COST,
    slippage=SLIPPAGE,
  )
  m = calculate_metrics_from_equity(sim["equity"], sim["periodic_returns"])
  m["strategy"] = name
  return {**sim, "metrics": m, "targets": targets}


# PREFLIGHT FUNCIONAL: ejecuta funciones reales, sin inspección defectuosa.
_sample_sig = sorted(base_s5_targets)[-2]
_sample_w = base_s5_targets[_sample_sig]
_sample_viol = audit_weight_caps(
  _sample_w,
  _sample_sig,
  "PREFLIGHT",
  SECTOR_MAP,
)
assert not _sample_viol, f"Allocator/caps fallan: {_sample_viol[:3]}"
assert abs(float(_sample_w.sum()) - 1.0) < 1e-8

_smoke = run_from_targets(
  "PREFLIGHT_LEDGER",
  {_sample_sig: _sample_w},
)
assert len(_smoke["equity"]) > 0
assert _smoke["reconciliation"].get("absolute_error", 1.0) <= RECON_ABS_TOL

print("PREFLIGHT FUNCIONAL: allocator, caps, S5, ledger y métricas PASS")

# %% [markdown]
# ## 5. Overlays preregistrados R0-R3

# %%
def apply_exposure_to_weights(base_w, exposure):
  w = base_w.copy().astype(float)
  stocks = w.drop(CASH_ASSET, errors="ignore")
  stock_sum = float(stocks.sum())
  shy_orig = float(w.get(CASH_ASSET, 0.0))
  for t in stocks.index:
    w[t] = float(stocks[t]) * exposure
  w[CASH_ASSET] = shy_orig + (1.0 - exposure) * stock_sum
  total = float(w.sum())
  if total <= 0:
    return pd.Series({CASH_ASSET: 1.0})
  if abs(total - 1.0) > 1e-8:
    w = w / total
  return w


def spy_trend_exposure(signal_date, spy_df=None):
  spy_df = spy_df if spy_df is not None else data_dict.get(MARKET)
  if spy_df is None or spy_df.empty:
    return R2_OFF
  hist = spy_df.loc[spy_df.index <= pd.Timestamp(signal_date)].copy()
  if len(hist) < R2_SMA_WINDOW + R2_SMA_SHIFT + 1:
    return R2_OFF
  sma = hist["Close"].rolling(R2_SMA_WINDOW).mean().shift(R2_SMA_SHIFT)
  close = float(hist["Close"].iloc[-1])
  sma_val = float(sma.iloc[-1])
  if not np.isfinite(sma_val):
    return R2_OFF
  return R2_ON if close > sma_val else R2_OFF


def vol_target_exposure(daily_returns, signal_date, target_vol=R1_TARGET_VOL,
                        lookback=R1_LOOKBACK, min_exp=R1_MIN_EXP):
  pr = daily_returns.loc[:pd.Timestamp(signal_date)].dropna()
  tail = pr.tail(lookback)
  if len(tail) < 21:
    return 1.0
  realized = float(tail.std() * math.sqrt(252))
  if realized <= 1e-8:
    return 1.0
  exp = min(1.0, target_vol / realized)
  return max(min_exp, exp)


def build_exposure_history(overlay_id, signal_dates, base_daily_returns):
  rows = []
  for sig in sorted(signal_dates):
    sig = pd.Timestamp(sig)
    exp_vol = vol_target_exposure(base_daily_returns, sig)
    exp_trend = spy_trend_exposure(sig)
    if overlay_id == "R0_BASE_S5":
      exp = 1.0
    elif overlay_id == "R1_VOL_TARGET_12":
      exp = exp_vol
    elif overlay_id == "R2_SPY_TREND":
      exp = exp_trend
    elif overlay_id == "R3_COMBINED_FIXED":
      exp = max(R3_MIN_EXP, min(exp_vol, exp_trend))
    else:
      raise ValueError(f"Overlay no autorizado: {overlay_id}")
    rows.append({
      "signal_date": sig, "overlay": overlay_id,
      "risk_exposure": round(exp, 6),
      "exposure_vol": round(exp_vol, 6),
      "exposure_trend": round(exp_trend, 6),
      "spy_trend_on": exp_trend >= R2_ON - 1e-9,
    })
  return pd.DataFrame(rows)


def build_overlay_targets(base_targets, exposure_df):
  exp_map = dict(zip(exposure_df["signal_date"].astype("datetime64[ns]"), exposure_df["risk_exposure"]))
  out = {}
  for sig, w in base_targets.items():
    ts = pd.Timestamp(sig)
    exp = float(exp_map.get(ts, 1.0))
    out[sig] = apply_exposure_to_weights(w, exp)
  return out


def validate_overlay_weights(targets, overlay_id):
  violations = []
  for sig, w in targets.items():
    violations.extend(audit_weight_caps(w, sig, overlay_id, SECTOR_MAP))
    stocks = w.drop(CASH_ASSET, errors="ignore")
    if (stocks < -1e-9).any():
      violations.append({"signal_date": sig, "strategy": overlay_id, "violation": "negative_weight"})
    if float(w.sum()) > 1.0 + 1e-6:
      violations.append({"signal_date": sig, "strategy": overlay_id, "violation": "leverage"})
  return violations

# %% [markdown]
# ## 6. Tests tecnicos (12/12 obligatorio antes del analisis)

# %%
def _test_row(name, passed, expected, obtained, abs_err=None):
  if abs_err is None:
    try:
      abs_err = abs(float(obtained) - float(expected))
    except (TypeError, ValueError):
      abs_err = np.nan
  return {"test": name, "pass": bool(passed), "expected": expected, "obtained": obtained, "absolute_error": abs_err}


def run_overlay_technical_tests():
  rows = []
  r0_sim_probe = run_from_targets("R0_PROBE", base_s5_targets)
  base_daily = r0_sim_probe["periodic_returns"].fillna(0)

  # T1: R0 reproduce base V17.5.2.2 (tolerancia)
  b1_targets_t1, _, _ = build_equal_weight_all_asof_benchmark(signal_dates)
  bench_b1_t1 = run_from_targets(PRIMARY_BENCHMARK, b1_targets_t1)
  r0_full = period_full_metrics(
    r0_sim_probe["periodic_returns"], bench_b1_t1["periodic_returns"],
    None, RESEARCH_START, RESEARCH_END)
  t1_ok = (
    abs(r0_full.get("CAGR_pct", 0) - sp["inherited_CAGR_pct"]) <= 1.5
    and abs(r0_full.get("excess_CAGR_vs_equal_weight", 0) - sp["inherited_excess_CAGR_vs_B1"]) <= 1.5
    and abs(r0_full.get("max_drawdown_pct", 0) - sp["inherited_max_drawdown_pct"]) <= 2.5
  )
  rows.append(_test_row("T1_R0_REPRODUCES_BASE", t1_ok,
    f"CAGR~{sp['inherited_CAGR_pct']}", f"CAGR={r0_full.get('CAGR_pct')} DD={r0_full.get('max_drawdown_pct')}"))

  # T2: vol usa solo datos pasados
  sig_test = signal_dates[len(signal_dates) // 2]
  exp_a = vol_target_exposure(base_daily, sig_test)
  future_daily = base_daily.copy()
  future_daily.loc[future_daily.index > sig_test] = 0.99
  exp_b = vol_target_exposure(future_daily, sig_test)
  rows.append(_test_row("T2_VOLATILITY_USES_ONLY_PAST_DATA", abs(exp_a - exp_b) < 1e-9, exp_a, exp_b))

  # T3: SPY SMA200 shifted
  spy_hist = data_dict[MARKET]
  hist = spy_hist.loc[spy_hist.index <= pd.Timestamp(sig_test)]
  sma = hist["Close"].rolling(R2_SMA_WINDOW).mean().shift(R2_SMA_SHIFT)
  manual = R2_ON if float(hist["Close"].iloc[-1]) > float(sma.iloc[-1]) else R2_OFF
  got = spy_trend_exposure(sig_test)
  rows.append(_test_row("T3_SPY_SMA200_SHIFTED", abs(manual - got) < 1e-9, manual, got))

  # T4: exposure dentro de limites
  exp_hist_r3 = build_exposure_history("R3_COMBINED_FIXED", signal_dates[:20], base_daily)
  t4_ok = bool((exp_hist_r3["risk_exposure"] >= R3_MIN_EXP - 1e-9).all()
               and (exp_hist_r3["risk_exposure"] <= 1.0 + 1e-9).all())
  rows.append(_test_row("T4_EXPOSURE_WITHIN_LIMITS", t4_ok, "[0.25,1]", exp_hist_r3["risk_exposure"].tolist()[:3]))

  # T5: pesos suman 1
  exp_df = build_exposure_history("R1_VOL_TARGET_12", signal_dates[:10], base_daily)
  ov_tgt = build_overlay_targets(base_s5_targets, exp_df)
  sums = [float(w.sum()) for w in ov_tgt.values()]
  rows.append(_test_row("T5_WEIGHTS_SUM_ONE", all(abs(s - 1.0) < 1e-6 for s in sums), 1.0, sums[:3]))

  # T6: caps accion y sector
  viol = validate_overlay_weights(ov_tgt, "R1_VOL_TARGET_12")
  rows.append(_test_row("T6_STOCK_AND_SECTOR_CAPS", len(viol) == 0, 0, len(viol)))

  # T7: resto a SHY
  sig0 = sorted(base_s5_targets.keys())[0]
  w0 = base_s5_targets[sig0]
  exp0 = 0.5
  adj = apply_exposure_to_weights(w0, exp0)
  stock_sum0 = float(w0.drop(CASH_ASSET, errors="ignore").sum())
  shy_expected = float(w0.get(CASH_ASSET, 0.0)) + (1 - exp0) * stock_sum0
  rows.append(_test_row("T7_REMAINDER_TO_SHY", abs(adj.get(CASH_ASSET, 0) - shy_expected) < 1e-6,
    shy_expected, adj.get(CASH_ASSET, 0)))

  # T8: sin leverage
  t8_ok = all(float(w.sum()) <= 1.0 + 1e-6 and (w >= -1e-9).all() for w in ov_tgt.values())
  rows.append(_test_row("T8_NO_LEVERAGE", t8_ok, True, t8_ok))

  # T9: reconciliacion single-ledger
  recon = r0_sim_probe["reconciliation"]
  t9_ok = bool(recon.get("pass_absolute", False) or recon.get("absolute_error", 1) <= RECON_ABS_TOL)
  rows.append(_test_row("T9_SINGLE_LEDGER_RECONCILIATION", t9_ok, f"<={RECON_ABS_TOL}", recon.get("absolute_error")))

  # T10: determinista
  sim_a = run_from_targets("DET_A", {sig0: base_s5_targets[sig0]})
  sim_b = run_from_targets("DET_B", {sig0: base_s5_targets[sig0]})
  det = float(sim_a["equity"].iloc[-1]) == float(sim_b["equity"].iloc[-1])
  rows.append(_test_row("T10_DETERMINISTIC", det, "equal", det))

  # T11: ejecucion next open
  em = r0_sim_probe.get("execution_map", {})
  t11_ok = all(pd.Timestamp(ex) > pd.Timestamp(sig) for sig, ex in em.items()) if em else True
  rows.append(_test_row("T11_EXECUTION_NEXT_OPEN", t11_ok, True, t11_ok))

  # T12: sin drawdown futuro como input
  overlay_src = " ".join(
    name
    for fn in (vol_target_exposure, spy_trend_exposure, build_exposure_history, apply_exposure_to_weights)
    for name in fn.__code__.co_names
  )
  t12_ok = "drawdown" not in overlay_src.lower() and "cummax" not in overlay_src.lower()
  rows.append(_test_row("T12_NO_FUTURE_DRAWDOWN_INPUT", t12_ok, "no_dd_input", t12_ok))

  df = pd.DataFrame(rows)
  n_pass = int(df["pass"].sum())
  print("=" * 72)
  print("OVERLAY TECHNICAL TESTS (12/12 obligatorio)")
  print(df.to_string(index=False))
  print(f"RESULT: {n_pass}/12 PASS")
  print("=" * 72)
  if n_pass < 12:
    failed = df.loc[~df["pass"], "test"].tolist()
    raise AssertionError(f"Tests fallidos: {failed}. Analisis abortado.")
  return df


execution_tests_df = run_overlay_technical_tests()
TECH_TESTS_PASS = bool(execution_tests_df["pass"].all())
assert TECH_TESTS_PASS, "12/12 tests requeridos antes del analisis completo."

# %% [markdown]
# ## 7. Simulacion R0-R3 + metricas

# %%
b1_targets, b1_holdings_df, b1_audit_df = build_equal_weight_all_asof_benchmark(signal_dates)
bench_b1 = run_from_targets(PRIMARY_BENCHMARK, b1_targets)
bench_spy = run_from_targets("B0_SPY", {signal_dates[0]: pd.Series({MARKET: 1.0})})
bench_b1_pr = bench_b1["periodic_returns"]
bench_spy_pr = bench_spy["periodic_returns"]

r0_sim = run_from_targets("R0_BASE_S5", base_s5_targets)
base_daily_returns = r0_sim["periodic_returns"].fillna(0)

overlay_sims = {"R0_BASE_S5": r0_sim}
overlay_targets = {"R0_BASE_S5": base_s5_targets}
exposure_histories = {}

for ov in OVERLAY_VARIANTS:
  if ov == "R0_BASE_S5":
    exposure_histories[ov] = build_exposure_history(ov, signal_dates, base_daily_returns)
    continue
  exp_df = build_exposure_history(ov, signal_dates, base_daily_returns)
  exposure_histories[ov] = exp_df
  tgt = build_overlay_targets(base_s5_targets, exp_df)
  overlay_targets[ov] = tgt
  overlay_sims[ov] = run_from_targets(ov, tgt)

exposure_history_df = pd.concat(exposure_histories.values(), ignore_index=True)


def _annual_vol(returns, start, end):
  pr = returns.loc[(returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))].dropna()
  return round(float(pr.std() * math.sqrt(252) * 100), 2) if len(pr) else np.nan


def _turnover_pct(sim):
  tl = sim.get("trade_ledger", pd.DataFrame())
  if tl.empty or "turnover" not in tl.columns:
    return np.nan
  return round(float(tl["turnover"].sum()) * 100 / max(len(sim.get("targets", {})), 1), 2)


def _total_costs(sim):
  al = sim.get("daily_asset_ledger", pd.DataFrame())
  if al.empty or "allocated_cost" not in al.columns:
    return np.nan
  return round(float(al["allocated_cost"].sum()), 2)


def _pct_invested(sim):
  al = sim.get("daily_asset_ledger", pd.DataFrame())
  if al.empty:
    return np.nan
  shy = al[al["ticker"] == CASH_ASSET] if "ticker" in al.columns else pd.DataFrame()
  if shy.empty:
    return np.nan
  mv = shy.groupby("date")["market_value_end"].sum()
  eq = sim.get("daily_portfolio_ledger", pd.DataFrame())
  if eq.empty:
    return np.nan
  merged = eq.set_index("date")["equity_end"].reindex(mv.index)
  shy_pct = (mv / merged).fillna(0)
  return round(float((1 - shy_pct).mean() * 100), 2)


def _exposure_bucket_pct(exp_df):
  e = exp_df["risk_exposure"]
  return {
    "pct_days_exposure_25_50": round(float(((e >= 0.25) & (e < 0.50)).mean() * 100), 1),
    "pct_days_exposure_50_75": round(float(((e >= 0.50) & (e < 0.75)).mean() * 100), 1),
    "pct_days_exposure_75_100": round(float((e >= 0.75).mean() * 100), 1),
  }


def overlay_metrics_row(overlay_id, sim, exp_df, start, end, period_label):
  m = period_full_metrics(sim["periodic_returns"], bench_b1_pr, bench_spy_pr, start, end)
  calmar = round(m.get("CAGR_pct", 0) / max(abs(m.get("max_drawdown_pct", 1e-6)), 1e-6), 3)
  buckets = _exposure_bucket_pct(exp_df)
  return {
    "overlay": overlay_id,
    "period_label": period_label,
    "CAGR_pct": m.get("CAGR_pct", np.nan),
    "total_return_pct": m.get("total_return_pct", np.nan),
    "sharpe": m.get("sharpe", np.nan),
    "sortino": m.get("sortino", np.nan),
    "max_drawdown_pct": m.get("max_drawdown_pct", np.nan),
    "active_drawdown_pct": m.get("active_drawdown_pct", np.nan),
    "annual_volatility_pct": _annual_vol(sim["periodic_returns"], start, end),
    "excess_CAGR_vs_B1": m.get("excess_CAGR_vs_equal_weight", np.nan),
    "information_ratio_vs_B1": m.get("information_ratio_vs_equal_weight", np.nan),
    "pct_years_beating_B1": m.get("pct_years_beating_equal_weight", np.nan),
    "pct_years_beating_SPY": m.get("pct_years_beating_SPY", np.nan),
    "turnover_pct": _turnover_pct(sim),
    "total_costs_usd": _total_costs(sim),
    "pct_mean_invested": _pct_invested(sim),
    **buckets,
    "calmar_ratio": calmar,
    "reconciliation_pass": bool(sim["reconciliation"].get("pass_absolute", False)),
  }


def collect_overlay_metrics(overlay_id, sim, exp_df):
  end_full = sim["equity"].index.max()
  rows = [
    overlay_metrics_row(overlay_id, sim, exp_df, RESEARCH_START, RESEARCH_END, PERIOD_RESEARCH),
    overlay_metrics_row(overlay_id, sim, exp_df, REUSED_TEST_START, end_full, PERIOD_REUSED),
    overlay_metrics_row(overlay_id, sim, exp_df, RESEARCH_START, end_full, PERIOD_FULL),
  ]
  return rows


overlay_results_rows = []
research_period_rows = []
reused_test_rows = []
for ov in OVERLAY_VARIANTS:
  sim = overlay_sims[ov]
  exp_df = exposure_histories[ov]
  rows = collect_overlay_metrics(ov, sim, exp_df)
  overlay_results_rows.extend(rows)
  research_period_rows.append(next(r for r in rows if r["period_label"] == PERIOD_RESEARCH))
  reused_test_rows.append(next(r for r in rows if r["period_label"] == PERIOD_REUSED))

overlay_results_df = pd.DataFrame(overlay_results_rows)
research_period_df = pd.DataFrame(research_period_rows)
reused_test_df = pd.DataFrame(reused_test_rows)

# DSR + PBO: solo periodo de investigación y retornos alineados por fecha.
dsr_rows = []
for ov in OVERLAY_VARIANTS:
  sim = overlay_sims[ov]
  research_returns = sim["periodic_returns"].loc[
    pd.Timestamp(RESEARCH_START):pd.Timestamp(RESEARCH_END)
  ]
  for freq in ("daily", "weekly"):
    dsr_rows.append(
      calculate_dsr_audit(
        ov,
        research_returns,
        N_DSR_TRIALS_OVERLAY,
        frequency=freq,
      )
    )
dsr_audit_df = pd.DataFrame(dsr_rows)

_pbo_series = {
  name: overlay_sims[name]["periodic_returns"].loc[
    pd.Timestamp(RESEARCH_START):pd.Timestamp(RESEARCH_END)
  ].dropna()
  for name in OVERLAY_VARIANTS
}
_common_idx = None
for _series in _pbo_series.values():
  _common_idx = _series.index if _common_idx is None else _common_idx.intersection(_series.index)
_common_idx = _common_idx.sort_values() if _common_idx is not None else pd.DatetimeIndex([])
pbo_matrix = (
  np.column_stack([
    _pbo_series[name].reindex(_common_idx).values
    for name in OVERLAY_VARIANTS
  ])
  if len(_common_idx) > 10
  else np.empty((0, len(OVERLAY_VARIANTS)))
)
overlay_pbo = simplified_pbo(pbo_matrix) if len(_common_idx) > 10 else np.nan

for ov in OVERLAY_VARIANTS:
  dsr_d = dsr_audit_df[(dsr_audit_df["strategy"] == ov) & (dsr_audit_df["sharpe_frequency"] == "daily")]
  if len(dsr_d):
    overlay_results_df.loc[
      (overlay_results_df["overlay"] == ov) & (overlay_results_df["period_label"] == PERIOD_RESEARCH),
      "dsr_probability"] = dsr_d.iloc[0]["dsr_probability"]
overlay_results_df["pbo_fraction"] = overlay_pbo

# Drawdown comparison
drawdown_comparison_df = research_period_df[[
  "overlay", "max_drawdown_pct", "active_drawdown_pct", "CAGR_pct", "calmar_ratio",
]].copy()
drawdown_comparison_df["inherited_base_drawdown_pct"] = sp["inherited_max_drawdown_pct"]
drawdown_comparison_df["drawdown_improvement_pct"] = (
  drawdown_comparison_df["max_drawdown_pct"] - sp["inherited_max_drawdown_pct"]
)

# Cost sensitivity selected provisional
def cost_sensitivity(sim, targets, overlay_id):
  rows = []
  for mult in [0.5, 1.0, 2.0]:
    cs = simulate_portfolio_with_single_ledger(
      targets, data_dict, strategy_name=f"{overlay_id}_cost_{mult}",
      transaction_cost=TRANSACTION_COST * mult, slippage=SLIPPAGE * mult)
    cm = calculate_metrics_from_equity(cs["equity"], cs["periodic_returns"])
    rows.append({"overlay": overlay_id, "cost_multiplier": mult, "CAGR_pct": cm.get("CAGR_pct", 0)})
  return rows

# %% [markdown]
# ## 8. Seleccion preregistrada + null test (200 perm, overlay seleccionado)

# %%
def is_selection_candidate(row, null_p=np.nan):
  return (
    row.get("max_drawdown_pct", -100) > sel_rules["max_drawdown_pct_gt"]
    and row.get("CAGR_pct", 0) >= sel_rules["min_CAGR_pct"]
    and row.get("excess_CAGR_vs_B1", 0) >= sel_rules["min_excess_CAGR_vs_B1"]
    and row.get("information_ratio_vs_B1", 0) >= sel_rules["min_information_ratio"]
    and row.get("pct_years_beating_B1", 0) >= sel_rules["min_pct_years_beating_B1"]
    and row.get("pct_years_beating_SPY", 0) >= sel_rules["min_pct_years_beating_SPY"]
    and bool(row.get("reconciliation_pass", False))
    and (np.isnan(null_p) or null_p < sel_rules["max_null_p"])
  )


candidates = research_period_df.copy()
candidates["null_p"] = np.nan
candidates["cost_2x_positive"] = False

provisional = None
for ov in OVERLAY_VARIANTS:
  if ov == "R0_BASE_S5":
    continue
  row = research_period_df[research_period_df["overlay"] == ov].iloc[0]
  cost_rows = cost_sensitivity(overlay_sims[ov], overlay_targets[ov], ov)
  cost_2x_ok = any(r["cost_multiplier"] == 2.0 and r["CAGR_pct"] > 0 for r in cost_rows)
  candidates.loc[candidates["overlay"] == ov, "cost_2x_positive"] = cost_2x_ok

eligible = []
for _, row in candidates.iterrows():
  if row["overlay"] == "R0_BASE_S5":
    continue
  if is_selection_candidate(row) and row["cost_2x_positive"]:
    eligible.append(row)

selected_overlay = None
null_dist_df = pd.DataFrame(columns=["permutation", "overlay", "excess_CAGR_vs_B1", "max_drawdown_pct"])
null_summary = {}
null_p = np.nan

if eligible:
  elig_df = pd.DataFrame(eligible)
  elig_df = elig_df.sort_values(
    ["max_drawdown_pct", "calmar_ratio", "information_ratio_vs_B1", "CAGR_pct"],
    ascending=[False, False, False, False],
  )
  provisional = elig_df.iloc[0]["overlay"]
  selected_overlay = provisional
  print(f"Overlay provisional seleccionado: {selected_overlay}")

  null_rows = []
  for perm in tqdm(range(N_NULL_OVERLAY), desc=f"Null {selected_overlay}"):
    held, tgt_base = set(), {}
    for i, sig in enumerate(signal_dates):
      elig = sorted(ASOF_UNIVERSE.get(pd.Timestamp(sig), set()))
      if not elig:
        continue
      rng = np.random.RandomState(RANDOM_SEED + perm * 997 + i)
      sc = pd.Series(rng.rand(len(elig)), index=elig)
      ranked = sc.sort_values(ascending=False).index.tolist()
      sel = _select_buffered(ranked, held, BUY_RANK, HOLD_UNTIL_RANK, TOP_K)
      w = allocate_weights_with_hard_caps(sel, SECTOR_MAP)
      if len(w):
        tgt_base[sig] = w
        held = set(w.index) - {CASH_ASSET}
    if not tgt_base:
      continue
    ns_base = run_from_targets(f"null_base_{perm}", tgt_base)
    exp_df = build_exposure_history(selected_overlay, signal_dates, ns_base["periodic_returns"].fillna(0))
    tgt_ov = build_overlay_targets(tgt_base, exp_df)
    ns = run_from_targets(f"null_{perm}", tgt_ov)
    nm = period_full_metrics(ns["periodic_returns"], bench_b1_pr, bench_spy_pr, RESEARCH_START, RESEARCH_END)
    null_rows.append({
      "permutation": perm, "overlay": selected_overlay,
      "excess_CAGR_vs_B1": nm.get("excess_CAGR_vs_equal_weight", np.nan),
      "max_drawdown_pct": nm.get("max_drawdown_pct", np.nan),
    })
  null_dist_df = pd.DataFrame(null_rows)
  real_excess = float(research_period_df.loc[
    research_period_df["overlay"] == selected_overlay, "excess_CAGR_vs_B1"].iloc[0])
  null_vals = null_dist_df["excess_CAGR_vs_B1"].dropna().tolist()
  null_p, n_exceed, n_perm = corrected_empirical_p(real_excess, null_vals)
  null_summary = {
    "selected_overlay": selected_overlay,
    "real_excess_CAGR_vs_B1": real_excess,
    "null_mean_excess_CAGR": round(np.mean(null_vals), 4) if null_vals else np.nan,
    "n_permutations": n_perm,
    "number_exceeding_real": n_exceed,
    "corrected_empirical_p_value": round(null_p, 6),
  }
  candidates.loc[candidates["overlay"] == selected_overlay, "null_p"] = null_p

  if is_selection_candidate(
      research_period_df[research_period_df["overlay"] == selected_overlay].iloc[0].to_dict(),
      null_p=null_p):
    selected_overlay = selected_overlay
  else:
    selected_overlay = None

cost_sensitivity_df = pd.DataFrame(
  sum([cost_sensitivity(overlay_sims[ov], overlay_targets[ov], ov) for ov in OVERLAY_VARIANTS], [])
)

# Live signals con overlay del seleccionado o R0
def train_final_xgb_for_live(labeled_df, live_feature_panel, feature_cols, params=None):
  try:
    import xgboost as xgb
  except ImportError:
    return None, {}, pd.DataFrame()
  params = params or XGB_PARAMS
  latest_feature_date = live_feature_panel["signal_date"].max()
  tr = labeled_df[labeled_df["label_end_date"] < latest_feature_date].copy()
  tr = tr.sort_values(["signal_date", "ticker"]).reset_index(drop=True)
  audit = {"latest_feature_date": str(latest_feature_date.date()), "n_training_rows": len(tr)}
  if len(tr) < 80:
    return None, audit, pd.DataFrame()
  sel_feats, med = select_features_train(tr, feature_cols)
  tr[sel_feats] = tr[sel_feats].fillna(med)
  y_tr = build_frozen_xgb_relevance_labels(tr)
  group_sizes = (
    tr.groupby("signal_date", sort=False)
    .size()
    .to_numpy(dtype=np.uint32)
  )
  assert int(group_sizes.sum()) == len(tr)
  assert int(y_tr.max()) <= 31

  mdl = xgb.XGBRanker(**params)
  mdl.fit(
    tr[sel_feats],
    y_tr.to_numpy(dtype=np.int32),
    group=group_sizes,
  )
  live_g = live_feature_panel[live_feature_panel["signal_date"] == latest_feature_date].copy()
  elig = ASOF_UNIVERSE.get(pd.Timestamp(latest_feature_date), set())
  live_g = live_g[live_g["ticker"].isin(elig)]
  live_g[sel_feats] = live_g[sel_feats].fillna(med)
  live_g["live_xgb_score"] = mdl.predict(live_g[sel_feats])
  return mdl, audit, live_g


def derive_live_signal(target_weight, previous_weight, tolerance=1e-8):
  tw, pw = float(target_weight), float(previous_weight)
  if tw > pw + tolerance:
    return "BUY"
  if tw > tolerance and abs(tw - pw) <= tolerance:
    return "HOLD"
  if tw > tolerance and tw < pw - tolerance:
    return "REDUCE"
  if tw <= tolerance and pw > tolerance:
    return "SELL"
  return "AVOID"


live_model, live_audit, live_scored = train_final_xgb_for_live(
  labeled_panel,
  live_feature_panel,
  FEATURE_COLS,
)
latest_sig = pd.Timestamp(live_feature_panel["signal_date"].max())
assert live_model is not None and len(live_scored) > 0, "Inferencia live vacía"

# Construir pesos live con el modelo final actual, no con predicciones OOS históricas.
_prev_bt = sorted(base_s5_targets.keys())[-1] if base_s5_targets else None
_prev_held = (
  set(base_s5_targets[_prev_bt].index) - {CASH_ASSET}
  if _prev_bt is not None
  else set()
)
_live_ranked = (
  live_scored.sort_values(
    ["live_xgb_score", "ticker"],
    ascending=[False, True],
  )["ticker"]
  .tolist()
)
_live_selected = _select_buffered(
  _live_ranked,
  _prev_held,
  BUY_RANK,
  HOLD_UNTIL_RANK,
  TOP_K,
)
live_base_w = allocate_weights_with_hard_caps(
  _live_selected,
  SECTOR_MAP,
)
assert abs(float(live_base_w.sum()) - 1.0) < 1e-8

live_overlay = selected_overlay or "R0_BASE_S5"
live_exp = build_exposure_history(
  live_overlay,
  [latest_sig],
  base_daily_returns,
)
live_w = build_overlay_targets(
  {latest_sig: live_base_w},
  live_exp,
)[latest_sig]

prev_sig = (
  signal_dates[signal_dates < latest_sig][-1]
  if any(signal_dates < latest_sig)
  else None
)
prev_w = (
  overlay_targets.get(live_overlay, base_s5_targets).get(
    prev_sig,
    pd.Series(dtype=float),
  )
)

signal_rows = []
_all_live_tickers = sorted(set(live_w.index).union(prev_w.index))
for t in _all_live_tickers:
  wt = float(live_w.get(t, 0.0))
  pw = float(prev_w.get(t, 0.0))
  signal_rows.append({
    "ticker": t,
    "signal_date": latest_sig,
    "signal": derive_live_signal(wt, pw),
    "target_weight": round(wt, 6),
    "previous_weight": round(pw, 6),
    "overlay": live_overlay,
    "risk_exposure": float(live_exp.iloc[0]["risk_exposure"]),
    "strategy_source": LIVE_STRATEGY_SOURCE,
    "paper_trading_start": PAPER_TRADING_START,
  })

current_signals_df = pd.DataFrame(signal_rows).sort_values(
  ["target_weight", "ticker"],
  ascending=[False, True],
)
assert not (
  (current_signals_df["signal"] == "AVOID")
  & (current_signals_df["target_weight"] > 0)
).any()

# %% [markdown]
# ## 9. Gates R1-R13 + FINAL_STATUS

# %%
r0_row = research_period_df[research_period_df["overlay"] == "R0_BASE_S5"].iloc[0]
sel_row = None
if selected_overlay:
  sel_row = research_period_df[research_period_df["overlay"] == selected_overlay].iloc[0]

reused_sel = None
if selected_overlay:
  reused_sel = reused_test_df[reused_test_df["overlay"] == selected_overlay].iloc[0]

viol_all = []
for ov, tgt in overlay_targets.items():
  viol_all.extend(validate_overlay_weights(tgt, ov))

validation_gates = {
  "R1_base_reproduced": (
    abs(r0_row["CAGR_pct"] - sp["inherited_CAGR_pct"]) <= 1.5
    and abs(r0_row["max_drawdown_pct"] - sp["inherited_max_drawdown_pct"]) <= 2.5
  ),
  "R2_overlay_temporal_contract": TECH_TESTS_PASS,
  "R3_exposure_limits": bool((exposure_history_df["risk_exposure"] >= R1_MIN_EXP - 1e-9).all()
                             and (exposure_history_df["risk_exposure"] <= 1.0 + 1e-9).all()),
  "R4_weights_and_caps": len(viol_all) == 0,
  "R5_single_ledger_reconciles": all(overlay_sims[ov]["reconciliation"].get("pass_absolute", False)
                                       or overlay_sims[ov]["reconciliation"].get("absolute_error", 1) <= RECON_ABS_TOL
                                       for ov in OVERLAY_VARIANTS),
  "R6_drawdown_better_than_minus_35": (
    sel_row is not None and sel_row["max_drawdown_pct"] > sel_rules["max_drawdown_pct_gt"]
  ),
  "R7_cagr_at_least_15": sel_row is not None and sel_row["CAGR_pct"] >= sel_rules["min_CAGR_pct"],
  "R8_excess_cagr_at_least_4": sel_row is not None and sel_row["excess_CAGR_vs_B1"] >= sel_rules["min_excess_CAGR_vs_B1"],
  "R9_information_ratio_at_least_030": (
    sel_row is not None and sel_row["information_ratio_vs_B1"] >= sel_rules["min_information_ratio"]
  ),
  "R10_cost_robustness": (
    selected_overlay is not None and any(
      r["overlay"] == selected_overlay and r["cost_multiplier"] == 2.0 and r["CAGR_pct"] > 0
      for _, r in cost_sensitivity_df.iterrows())
  ),
  "R11_null_p_below_005": selected_overlay is not None and null_p < sel_rules["max_null_p"],
  "R12_no_parameter_tuning": set(OVERLAY_VARIANTS) == set(overlay_cfg["overlays_preregistered"]),
  "R13_reused_test_positive_active_return": (
    reused_sel is not None and reused_sel.get("excess_CAGR_vs_B1", 0) > 0
  ),
  "survivorship_warning_active": True,
  "POINT_IN_TIME_MEMBERSHIP": POINT_IN_TIME_MEMBERSHIP,
  "APPROVED_FOR_REAL_MONEY": False,
}

if not TECH_TESTS_PASS:
  FINAL_STATUS = "FAILED_RISK_OVERLAY_INTEGRITY"
elif selected_overlay is None:
  FINAL_STATUS = "RISK_OVERLAY_NOT_YET_ACCEPTABLE"
elif all(validation_gates[k] for k in [
  "R1_base_reproduced", "R2_overlay_temporal_contract", "R3_exposure_limits", "R4_weights_and_caps",
  "R5_single_ledger_reconciles", "R6_drawdown_better_than_minus_35", "R7_cagr_at_least_15",
  "R8_excess_cagr_at_least_4", "R9_information_ratio_at_least_030", "R10_cost_robustness",
  "R11_null_p_below_005", "R12_no_parameter_tuning", "R13_reused_test_positive_active_return",
]):
  FINAL_STATUS = (
    "READY_FOR_FORWARD_PAPER_TRADING" if POINT_IN_TIME_MEMBERSHIP
    else "RISK_OVERLAY_CANDIDATE_SURVIVORSHIP_BIASED"
  )
else:
  FINAL_STATUS = "RISK_OVERLAY_NOT_YET_ACCEPTABLE"

summary = {
  "AUDIT_VERSION": AUDIT_VERSION,
  "FINAL_STATUS": FINAL_STATUS,
  "APPROVED_FOR_REAL_MONEY": False,
  "frozen_base_strategy": FROZEN_CHAMPION,
  "selected_overlay": selected_overlay or "NONE",
  "provisional_overlay": provisional or "NONE",
  "inherited_from_v17522": True,
  "inherited_CAGR_pct": sp["inherited_CAGR_pct"],
  "inherited_max_drawdown_pct": sp["inherited_max_drawdown_pct"],
  "R0_research_CAGR_pct": r0_row["CAGR_pct"],
  "R0_research_max_drawdown_pct": r0_row["max_drawdown_pct"],
  "overlays_evaluated": ",".join(OVERLAY_VARIANTS),
  "parameter_tuning": False,
  "technical_tests_pass": f"{int(execution_tests_df['pass'].sum())}/12",
  "null_permutations": N_NULL_OVERLAY if selected_overlay else 0,
  "null_p_selected": round(null_p, 6) if np.isfinite(null_p) else np.nan,
  "overlay_pbo": round(overlay_pbo, 4) if np.isfinite(overlay_pbo) else np.nan,
  "gates_pass": sum(v for k, v in validation_gates.items() if k.startswith("R")),
  "gates_total": sum(1 for k in validation_gates if k.startswith("R")),
  "POINT_IN_TIME_MEMBERSHIP": POINT_IN_TIME_MEMBERSHIP,
  "survivorship_warning": SURVIVORSHIP_WARNING,
  "paper_trading_start": PAPER_TRADING_START,
  "next_phase": overlay_cfg.get("next_phase_on_success", "FORWARD_PAPER_TRADING"),
  "period_research": PERIOD_RESEARCH,
  "period_reused_test": PERIOD_REUSED,
}

print("\n" + "=" * 72)
print("V17.6 PREREGISTERED RISK OVERLAY RESEARCH")
print("=" * 72)
print(f"FINAL_STATUS: {FINAL_STATUS}")
print(f"Selected overlay: {selected_overlay or 'NONE'}")
print(f"R0 drawdown: {r0_row['max_drawdown_pct']}% (inherited {sp['inherited_max_drawdown_pct']}%)")
print(f"Technical tests: {summary['technical_tests_pass']}")
print(f"APPROVED_FOR_REAL_MONEY=False | S5 congelado | solo R0-R3")

# %% [markdown]
# ## 10. Exports V17.6

# %%
pd.DataFrame([summary]).to_csv("research_v17_6_summary.csv", index=False)
overlay_results_df.to_csv("research_v17_6_overlay_results.csv", index=False)
research_period_df.to_csv("research_v17_6_research_period_results.csv", index=False)
reused_test_df.to_csv("research_v17_6_reused_test_results.csv", index=False)
exposure_history_df.to_csv("research_v17_6_exposure_history.csv", index=False)
drawdown_comparison_df.to_csv("research_v17_6_drawdown_comparison.csv", index=False)
cost_sensitivity_df.to_csv("research_v17_6_cost_sensitivity.csv", index=False)
dsr_audit_df.to_csv("research_v17_6_dsr_audit.csv", index=False)
null_dist_df.to_csv("research_v17_6_null_results.csv", index=False)
execution_tests_df.to_csv("research_v17_6_execution_tests.csv", index=False)
pd.DataFrame([{"gate": k, "pass": v} for k, v in validation_gates.items()]).to_csv(
  "research_v17_6_validation_gates.csv", index=False)
current_signals_df.to_csv("research_v17_6_current_signals.csv", index=False)
Path("research_v17_6_selected_config.json").write_text(json.dumps({
  "version": AUDIT_VERSION,
  "final_status": FINAL_STATUS,
  "approved_for_real_money": False,
  "frozen_base_strategy": FROZEN_CHAMPION,
  "selected_overlay": selected_overlay,
  "overlays_preregistered": OVERLAY_VARIANTS,
  "overlay_parameters": ov_params,
  "selection_rules": sel_rules,
  "validation_gates": validation_gates,
  "null_summary": null_summary,
  "summary": summary,
  "overlay_config": str(OVERLAY_CONFIG_PATH),
}, indent=2, default=str), encoding="utf-8")

EXPORT_FILES = [
  "research_v17_6_summary.csv",
  "research_v17_6_overlay_results.csv",
  "research_v17_6_research_period_results.csv",
  "research_v17_6_reused_test_results.csv",
  "research_v17_6_exposure_history.csv",
  "research_v17_6_drawdown_comparison.csv",
  "research_v17_6_cost_sensitivity.csv",
  "research_v17_6_dsr_audit.csv",
  "research_v17_6_null_results.csv",
  "research_v17_6_execution_tests.csv",
  "research_v17_6_validation_gates.csv",
  "research_v17_6_current_signals.csv",
  "research_v17_6_selected_config.json",
]

# %% [markdown]
# ## 11. ZIP unico + descarga

# %%
ZIP_PATH = CONTENT_ROOT / "V17_6_RESULTS.zip"
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
  for name in EXPORT_FILES:
    p = CONTENT_ROOT / name
    if not p.exists():
      p = Path(name)
    if p.exists():
      zf.write(p, p.name)
      print(f"  + {name}")

if IN_COLAB:
  print("ZIP creado. Usa la celda final para descargar sin repetir el analisis.")
else:
  print(f"ZIP listo: {ZIP_PATH}")

# %% [markdown]
# ## Descargar resultados sin repetir el analisis
#
# Ejecuta solo esta celda si ya corriste el notebook completo y solo quieres el ZIP.

# %%
def download_v176_results_only():
  zip_path = CONTENT_ROOT / "V17_6_RESULTS.zip"
  if not zip_path.exists():
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
      for name in EXPORT_FILES:
        p = CONTENT_ROOT / name
        if not p.exists():
          p = Path(name)
        if p.exists():
          zf.write(p, p.name)
  if IN_COLAB:
    from google.colab import files
    files.download(str(zip_path))
    print("Descarga: V17_6_RESULTS.zip")
  else:
    print(f"ZIP: {zip_path}")


download_v176_results_only()
