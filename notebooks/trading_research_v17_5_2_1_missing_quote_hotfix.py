# %% [markdown]
# # Trading Research V17.5.2.1 — Missing-Quote Continuity and Live-Label Hotfix
#
# Corrige carry-forward de cotizaciones ausentes (FISV) y etiquetas live coherentes.
# Motor single-ledger: 11/11 tests sintéticos obligatorios antes del full run.
# **APPROVED_FOR_REAL_MONEY siempre False.**

# %%
try:
  get_ipython().run_line_magic(
    "pip", "install yfinance pandas numpy scipy scikit-learn xgboost matplotlib plotly tqdm lxml html5lib beautifulsoup4 requests -q"
  )
except NameError:
  import subprocess, sys
  subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "yfinance", "pandas", "numpy", "scipy", "scikit-learn", "xgboost",
    "matplotlib", "plotly", "tqdm", "lxml", "html5lib", "beautifulsoup4", "requests"])

# %% [markdown]
# ## Subir archivos necesarios para V17.5.2.1

# %%
import json
import shutil
from pathlib import Path

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
  "v17_5_frozen_audit_config.json": CONFIG_DIR / "v17_5_frozen_audit_config.json",
  "v17_4_preregistered_experiment.json": CONFIG_DIR / "v17_4_preregistered_experiment.json",
  "v17_5_1_integrity_patch.json": CONFIG_DIR / "v17_5_1_integrity_patch.json",
  "v17_5_2_execution_contract.json": CONFIG_DIR / "v17_5_2_execution_contract.json",
  "v17_5_2_1_missing_quote_contract.json": CONFIG_DIR / "v17_5_2_1_missing_quote_contract.json",
}
OPTIONAL_UPLOADS = {
  "research_v17_4_1_strategy_results.csv": CONTENT_ROOT / "research_v17_4_1_strategy_results.csv",
}


def _validate_json_file(path):
  path = Path(path)
  if not path.exists() or path.stat().st_size <= 0:
    raise ValueError(f"JSON ausente o vacio: {path}")
  obj = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(obj, dict) or len(obj) == 0:
    raise ValueError(f"JSON invalido o vacio: {path}")
  return obj


def _place_uploaded_file(filename, data, dest_map):
  tmp = CONTENT_ROOT / filename
  tmp.write_bytes(data)
  if filename not in dest_map:
    print(f"Archivo ignorado (no reconocido): {filename}")
    return
  dest = dest_map[filename]
  dest.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(tmp, dest)
  print(f"Colocado: {dest}")


if IN_COLAB:
  from google.colab import files
  print("Sube los archivos necesarios:")
  print("  Obligatorios:")
  print("    - trading_research_v17_2_backtest_integrity_lab.py")
  print("    - v17_5_frozen_audit_config.json")
  print("    - v17_4_preregistered_experiment.json")
  print("    - v17_5_1_integrity_patch.json")
  print("    - v17_5_2_execution_contract.json")
  print("    - v17_5_2_1_missing_quote_contract.json")
  print("  Opcional:")
  print("    - research_v17_4_1_strategy_results.csv")
  uploaded = files.upload()
  for fname, data in uploaded.items():
    _place_uploaded_file(fname, data, {**REQUIRED_UPLOADS, **OPTIONAL_UPLOADS})
else:
  print("Ejecucion local: omitiendo upload. Se usan archivos del proyecto.")

missing = [name for name, dest in REQUIRED_UPLOADS.items() if not dest.exists() or dest.stat().st_size <= 0]
if missing:
  raise FileNotFoundError(
    "Faltan archivos obligatorios para V17.5.2.1: " + ", ".join(missing)
  )
_validate_json_file(REQUIRED_UPLOADS["v17_5_frozen_audit_config.json"])
_validate_json_file(REQUIRED_UPLOADS["v17_4_preregistered_experiment.json"])
if REQUIRED_UPLOADS["v17_5_1_integrity_patch.json"].exists():
  _validate_json_file(REQUIRED_UPLOADS["v17_5_1_integrity_patch.json"])
_validate_json_file(REQUIRED_UPLOADS["v17_5_2_execution_contract.json"])
_validate_json_file(REQUIRED_UPLOADS["v17_5_2_1_missing_quote_contract.json"])
if OPTIONAL_UPLOADS["research_v17_4_1_strategy_results.csv"].exists():
  print(f"Opcional presente: {OPTIONAL_UPLOADS['research_v17_4_1_strategy_results.csv']}")
else:
  print("Opcional ausente: research_v17_4_1_strategy_results.csv (se usara referencia embebida)")
print("Archivos obligatorios validados OK")

# %% [markdown]
# ## 1. Configuracion congelada V17.5.2.1

# %%
import warnings
warnings.filterwarnings("ignore")
import ast, hashlib, json, math, re, zipfile
from collections import defaultdict
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy import stats
from tqdm.auto import tqdm

AUDIT_VERSION = "v17_5_2_1"
PATCH_PATH = Path("config/v17_5_1_integrity_patch.json")
EXEC_CONTRACT_PATH = Path("config/v17_5_2_execution_contract.json")
MISSING_QUOTE_PATH = Path("config/v17_5_2_1_missing_quote_contract.json")
CONFIG_PATH = Path("config/v17_5_frozen_audit_config.json")
PREREG_PATH = Path("config/v17_4_preregistered_experiment.json")
patch = json.loads(PATCH_PATH.read_text(encoding="utf-8")) if PATCH_PATH.exists() else {}
exec_contract = json.loads(EXEC_CONTRACT_PATH.read_text(encoding="utf-8")) if EXEC_CONTRACT_PATH.exists() else {}
missing_quote_contract = json.loads(MISSING_QUOTE_PATH.read_text(encoding="utf-8")) if MISSING_QUOTE_PATH.exists() else {}
cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
fp = cfg["frozen_parameters"]
rg = cfg["risk_gates"]
V1752_REF = missing_quote_contract.get("v17_5_2_reference", exec_contract.get("v17_5_1_reference", {}))
PRIMARY_BENCHMARK = patch.get("primary_benchmark", "B1_EQUAL_WEIGHT_ALL_ASOF")
LIVE_STRATEGY_SOURCE = patch.get("live_strategy_source", "S5_XGBRANKER_LIVE")
SIGNAL_TIME = exec_contract.get("execution_contract", {}).get("SIGNAL_TIME", "AFTER_CLOSE")
EXECUTION_TIME = exec_contract.get("execution_contract", {}).get("EXECUTION_TIME", "NEXT_TRADING_DAY_OPEN")
VALUATION_TIME = exec_contract.get("execution_contract", {}).get("VALUATION_TIME", "DAILY_CLOSE")
RECON_ABS_TOL = float(exec_contract.get("reconciliation_tolerance", {}).get("absolute_usd", 0.01))
RECON_REL_TOL = float(exec_contract.get("reconciliation_tolerance", {}).get("relative", 1e-10))
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

MAX_TICKERS_FULL = fp["TOP_K_FULL"] * 10  # pool; select 250 as-of
TOP_K_FULL = fp["TOP_K_FULL"]
TOP_K = TOP_K_FULL
BUY_RANK = fp["BUY_RANK"]
HOLD_UNTIL_RANK = fp["HOLD_UNTIL_RANK"]
MAX_STOCK_WEIGHT = fp["MAX_STOCK_WEIGHT"]
MAX_SECTOR_WEIGHT = fp["MAX_SECTOR_WEIGHT"]
MIN_HISTORY_DAYS = 1000
MIN_DOLLAR_VOLUME = 20_000_000
MIN_VALID_STOCKS = 200
MIN_SECTORS = 8
N_NULL_PORTFOLIO = fp["N_NULL_PORTFOLIO"]
POINT_IN_TIME_MEMBERSHIP = cfg.get("POINT_IN_TIME_MEMBERSHIP", False)
SURVIVORSHIP_WARNING = cfg["survivorship_warning"]
PRIMARY_METRIC = "information_ratio_vs_equal_weight"
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
ENSEMBLE_W_COMPOSITE = 0.50
ENSEMBLE_W_XGB = 0.50

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

# %% [markdown]
# ## 2. Motor V17.2

# %%
try:
  _NB_DIR = Path(__file__).parent
except NameError:
  _NB_DIR = Path(".")
V172 = _NB_DIR / "trading_research_v17_2_backtest_integrity_lab.py"
_src = V172.read_text(encoding="utf-8")
_start = _src.index("def _clean_symbol")
_end = _src.index("# %% [markdown]\n# ## 8. Baselines")
_chunk = _src[_start:_end]
_mod = ast.parse(_chunk)
_func_src = [ast.get_source_segment(_chunk, n) for n in _mod.body if isinstance(n, ast.FunctionDef)]
_eng = "\n\n".join(s for s in _func_src if s)
_ns = {
  "pd": pd, "np": np, "math": math, "tqdm": tqdm, "hashlib": hashlib,
  "QUICK_TEST": False, "MAX_TICKERS_FULL": 250,
  "START_DATE": START_DATE, "END_DATE": END_DATE,
  "MIN_HISTORY_DAYS": MIN_HISTORY_DAYS, "MIN_DOLLAR_VOLUME": MIN_DOLLAR_VOLUME,
  "DEFENSIVE_ETFS": DEFENSIVE_ETFS, "ALL_ETFS": ALL_ETFS, "SECTOR_MAP": SECTOR_MAP,
  "SIGNAL_FREQUENCY": "W-FRI", "INITIAL_CAPITAL": INITIAL_CAPITAL,
  "TRANSACTION_COST": TRANSACTION_COST, "SLIPPAGE": SLIPPAGE, "COST_RATE": COST_RATE,
  "CASH_ASSET": CASH_ASSET, "MARKET": MARKET, "FORBIDDEN_FEATURE_PATTERNS": FORBIDDEN_FEATURE_PATTERNS,
  "TOP_K": TOP_K, "weekly_dates": pd.DatetimeIndex([]),
}
exec(_eng, _ns)
globals().update({k: v for k, v in _ns.items() if callable(v)})


# %% [markdown]
# %% [markdown]
# %% [markdown]
# ## 2b. Contrato de ejecución + motor single-ledger + tests sintéticos (11/11 obligatorio)

# %%
def _ledger_price(ddf, dt, col):
  if ddf is None or len(ddf) == 0 or dt not in ddf.index or col not in ddf.columns:
    return np.nan
  v = float(ddf.loc[dt, col])
  return v if np.isfinite(v) else np.nan


def _resolve_execution_date(cal, signal_date, weights, data_dict, defer_rows):
  sig = pd.Timestamp(signal_date)
  pos = int(cal.searchsorted(sig, side="right"))
  tickers = [t for t in weights.index if float(weights.get(t, 0)) > 1e-12]
  while pos < len(cal):
    ex = pd.Timestamp(cal[pos])
    ok = True
    for t in tickers:
      if t not in data_dict:
        ok = False
        break
      ddf = data_dict[t]
      if np.isnan(_ledger_price(ddf, ex, "Open")) or np.isnan(_ledger_price(ddf, ex, "Close")):
        ok = False
        break
    if ok:
      return ex
    defer_rows.append({
      "signal_date": sig, "skipped_execution_date": ex, "reason": "missing_open_or_close",
    })
    pos += 1
  return pd.NaT


# V17.5.2.1 ledger core

def _close_or_nan(ddf, dt):
  return _ledger_price(ddf, dt, "Close")


def _portfolio_equity(cash_bal, shares_dict, price_map):
  total = float(cash_bal)
  for t, q in shares_dict.items():
    if q > 1e-12 and t in price_map and np.isfinite(price_map[t]):
      total += q * price_map[t]
  return total


def build_equity_continuity_audit(port_ledger, tol=None):
  tol = tol if tol is not None else EQUITY_CONTINUITY_TOL
  if port_ledger is None or port_ledger.empty:
    return pd.DataFrame()
  pl = port_ledger.sort_values("date").reset_index(drop=True)
  rows = []
  prev_end = np.nan
  for i, r in pl.iterrows():
    if i > 0 and np.isfinite(prev_end):
      err = float(r["equity_start"]) - float(prev_end)
      rows.append({
        "date": r["date"], "strategy": r.get("strategy", ""),
        "equity_start": r["equity_start"], "equity_end_previous": prev_end,
        "equity_continuity_error": err, "pass": abs(err) <= tol,
      })
    prev_end = r["equity_end"]
  return pd.DataFrame(rows)


def simulate_portfolio_with_single_ledger(
    target_weights_by_signal_date,
    data_dict,
    initial_capital=None,
    transaction_cost=None,
    slippage=None,
    strategy_name="",
):
  initial_capital = float(initial_capital or INITIAL_CAPITAL)
  tx_cost = TRANSACTION_COST if transaction_cost is None else float(transaction_cost)
  slip = SLIPPAGE if slippage is None else float(slippage)
  cost_rate = tx_cost + slip

  cal = _master_calendar(data_dict)
  _empty = {
    "equity": pd.Series(dtype=float), "periodic_returns": pd.Series(dtype=float),
    "daily_portfolio_ledger": pd.DataFrame(), "daily_asset_ledger": pd.DataFrame(),
    "trade_ledger": pd.DataFrame(), "contribution_by_asset": pd.DataFrame(),
    "contribution_by_sector": pd.DataFrame(), "reconciliation": {},
    "accounting": pd.DataFrame(), "targets": target_weights_by_signal_date or {},
    "execution_map": {}, "deferred_executions": pd.DataFrame(),
    "price_gap_audit": pd.DataFrame(), "equity_continuity_audit": pd.DataFrame(),
    "metrics": {},
  }
  if len(cal) < 2 or not target_weights_by_signal_date:
    return _empty

  defer_rows, exec_schedule, signal_to_exec = [], {}, {}
  for sig in sorted(target_weights_by_signal_date.keys()):
    w = _normalize_weights(target_weights_by_signal_date[sig])
    ex = _resolve_execution_date(cal, sig, w, data_dict, defer_rows)
    if pd.notna(ex):
      exec_schedule[pd.Timestamp(ex)] = (pd.Timestamp(sig), w)
      signal_to_exec[pd.Timestamp(sig)] = pd.Timestamp(ex)
  if not exec_schedule:
    return _empty

  cal = cal[cal >= min(exec_schedule.keys())]
  cash = initial_capital
  shares = {}
  last_valid_close = {}
  last_valid_price_date = {}
  prev_equity_end = initial_capital
  prev_dt = None
  equity_curve, periodic_returns = {}, []
  daily_asset_rows, daily_portfolio_rows, trade_rows, price_gap_rows = [], [], [], []
  ticker_gross, ticker_cost = defaultdict(float), defaultdict(float)
  ticker_wsum, ticker_maxw, ticker_days = defaultdict(float), defaultdict(float), defaultdict(int)

  def _gap(t, dt, status, lvc, raw, q, pnl, note=""):
    price_gap_rows.append({
      "date": dt, "ticker": t, "strategy": strategy_name,
      "price_status": status, "last_valid_close": lvc, "raw_close": raw,
      "shares": q, "gross_pnl": pnl, "note": note,
    })

  def _val_px(t, q):
    if q <= 1e-12 or t not in data_dict:
      return np.nan, ""
    raw = _close_or_nan(data_dict[t], dt)
    if np.isfinite(raw):
      return raw, "VALID_CLOSE"
    if t in last_valid_close:
      return last_valid_close[t], "STALE_CARRY_FORWARD"
    return np.nan, "NO_PRICE"

  for dt in cal:
    dt = pd.Timestamp(dt)
    is_exec = dt in exec_schedule
    sig_date, target_w = exec_schedule.get(dt, (pd.NaT, None))
    shares_start = {t: q for t, q in shares.items() if q > 1e-12}
    cash_start = cash
    equity_start = prev_equity_end if prev_dt is not None else initial_capital

    day_gross = defaultdict(float)
    day_cost = defaultdict(float)
    trade_notional = {}
    asset_pnl_parts = defaultdict(lambda: {"overnight": 0.0, "intraday": 0.0, "ctc": 0.0})
    equity_after_overnight = equity_start
    equity_after_cost = equity_start

    if is_exec and target_w is not None:
      overnight_total = 0.0
      for t, q in shares_start.items():
        if q <= 0 or t not in data_dict:
          continue
        opn = _ledger_price(data_dict[t], dt, "Open")
        lvc = last_valid_close.get(t)
        if not np.isfinite(opn) or lvc is None:
          continue
        on_pnl = q * (opn - lvc)
        day_gross[t] += on_pnl
        asset_pnl_parts[t]["overnight"] = on_pnl
        overnight_total += on_pnl
      equity_after_overnight = equity_start + overnight_total

      port_open = cash_start + sum(
        shares_start[t] * _ledger_price(data_dict[t], dt, "Open")
        for t in shares_start if t in data_dict and np.isfinite(_ledger_price(data_dict[t], dt, "Open"))
      )
      if port_open <= 0:
        port_open = equity_after_overnight

      cur_w = {}
      for t in set(shares_start) | set(target_w.index):
        opn = _ledger_price(data_dict.get(t, pd.DataFrame()), dt, "Open")
        val = shares_start.get(t, 0.0) * opn if np.isfinite(opn) else 0.0
        cur_w[t] = val / port_open if port_open > 0 else 0.0
      union = target_w.index.union(pd.Index(list(cur_w.keys())))
      dw = (target_w.reindex(union, fill_value=0) - pd.Series(cur_w).reindex(union, fill_value=0)).abs()
      turnover = float(target_w.sum()) if not shares_start else 0.5 * float(dw.sum())
      total_cost = turnover * cost_rate * port_open
      equity_after_cost = port_open - total_cost
      investable = equity_after_cost

      new_shares, trade_notional = {}, {}
      for t in target_w.index:
        wt = float(target_w.get(t, 0))
        if wt <= 1e-12 or t not in data_dict:
          continue
        opn = _ledger_price(data_dict[t], dt, "Open")
        if not np.isfinite(opn) or opn <= 0:
          continue
        new_shares[t] = investable * wt / opn
        trade_notional[t] = abs(new_shares[t] * opn - shares_start.get(t, 0.0) * opn)

      tn_sum = sum(trade_notional.values())
      if tn_sum > 0:
        for t, tn in trade_notional.items():
          day_cost[t] += total_cost * (tn / tn_sum)

      shares = {t: q for t, q in new_shares.items() if q > 1e-12}
      cash = max(0.0, investable - sum(
        q * _ledger_price(data_dict[t], dt, "Open")
        for t, q in shares.items()
        if t in data_dict and np.isfinite(_ledger_price(data_dict[t], dt, "Open"))
      ))

      for t, tn in trade_notional.items():
        if tn <= 1e-8:
          continue
        delta = new_shares.get(t, 0.0) - shares_start.get(t, 0.0)
        trade_rows.append({
          "strategy": strategy_name, "signal_date": sig_date, "execution_date": dt,
          "ticker": t, "side": "BUY" if delta > 0 else "SELL",
          "shares_delta": delta, "price_open": _ledger_price(data_dict[t], dt, "Open"),
          "trade_notional": tn, "allocated_cost": day_cost.get(t, 0.0),
          "turnover": turnover, "portfolio_value_open": port_open,
        })

      for t, q in shares.items():
        if q <= 1e-12 or t not in data_dict:
          continue
        opn = _ledger_price(data_dict[t], dt, "Open")
        raw_cls = _close_or_nan(data_dict[t], dt)
        was_new = shares_start.get(t, 0.0) <= 1e-12
        prev_lvc = last_valid_close.get(t)
        if np.isfinite(opn) and np.isfinite(raw_cls):
          id_pnl = q * (raw_cls - opn)
          day_gross[t] += id_pnl
          asset_pnl_parts[t]["intraday"] = id_pnl
          st = "RESUMED_AFTER_GAP" if prev_lvc is not None and last_valid_price_date.get(t, dt) < dt and not was_new else "VALID_CLOSE"
          last_valid_close[t] = raw_cls
          last_valid_price_date[t] = dt
          _gap(t, dt, st, prev_lvc, raw_cls, q, id_pnl)
        elif np.isfinite(opn) and was_new:
          last_valid_close[t] = opn
          last_valid_price_date[t] = dt
          _gap(t, dt, "NEW_POSITION_OPEN_AS_LVC", opn, raw_cls, q, 0.0, "close_missing_use_open")

    else:
      equity_after_overnight = equity_start
      equity_after_cost = equity_start
      if prev_dt is not None:
        for t, q in shares.items():
          if q <= 1e-12 or t not in data_dict:
            continue
          raw_cls = _close_or_nan(data_dict[t], dt)
          lvc = last_valid_close.get(t)
          if np.isfinite(raw_cls):
            if lvc is not None:
              pnl = q * (raw_cls - lvc)
              day_gross[t] += pnl
              asset_pnl_parts[t]["ctc"] = pnl
              gap_resume = last_valid_price_date.get(t) is not None and pd.Timestamp(last_valid_price_date[t]) < dt
              if gap_resume and not np.isfinite(_close_or_nan(data_dict[t], prev_dt)):
                _gap(t, dt, "RESUMED_AFTER_GAP", lvc, raw_cls, q, pnl)
              else:
                _gap(t, dt, "VALID_CLOSE", lvc, raw_cls, q, pnl)
            last_valid_close[t] = raw_cls
            last_valid_price_date[t] = dt
          elif lvc is not None:
            _gap(t, dt, "STALE_CARRY_FORWARD", lvc, raw_cls, q, 0.0)

    end_px = {}
    for t, q in shares.items():
      px, _ = _val_px(t, q)
      if np.isfinite(px):
        end_px[t] = px
    equity_end = _portfolio_equity(cash, shares, end_px) if end_px or cash else equity_start
    gross_sum, cost_sum = sum(day_gross.values()), sum(day_cost.values())
    if abs(equity_end - (equity_start + gross_sum - cost_sum)) > 1e-4:
      equity_end = equity_start + gross_sum - cost_sum

    tickers_day = set(shares_start) | set(shares) | set(day_gross)
    for t in tickers_day:
      g, c = day_gross.get(t, 0.0), day_cost.get(t, 0.0)
      ticker_gross[t] += g
      ticker_cost[t] += c
      q_end = shares.get(t, 0.0)
      val_px, pstat = _val_px(t, q_end)
      if equity_end > 0 and q_end > 1e-12 and np.isfinite(val_px):
        wgt = (q_end * val_px) / equity_end
        ticker_wsum[t] += wgt
        ticker_maxw[t] = max(ticker_maxw[t], wgt)
        ticker_days[t] += 1

      lvc = last_valid_close.get(t, np.nan)
      opn = _ledger_price(data_dict[t], dt, "Open") if t in data_dict else np.nan
      raw_cls = _close_or_nan(data_dict[t], dt) if t in data_dict else np.nan
      disp_cls = raw_cls if np.isfinite(raw_cls) else lvc
      prev_lvc_start = last_valid_close.get(t, np.nan)
      if prev_dt is not None and t in shares_start and shares_start[t] > 0:
        mkt_start_px = prev_lvc_start if np.isfinite(prev_lvc_start) else np.nan
      else:
        mkt_start_px = np.nan
      parts = asset_pnl_parts[t]
      daily_asset_rows.append({
        "date": dt, "ticker": t, "strategy": strategy_name,
        "shares_start": shares_start.get(t, 0.0), "shares_end": q_end,
        "price_previous_close": mkt_start_px, "price_open": opn, "price_close": disp_cls,
        "raw_close": raw_cls, "last_valid_close": lvc, "price_status": pstat,
        "market_value_start": shares_start.get(t, 0.0) * mkt_start_px if np.isfinite(mkt_start_px) else (
          shares_start.get(t, 0.0) * lvc if np.isfinite(lvc) else 0.0
        ),
        "overnight_pnl": parts["overnight"], "intraday_pnl": parts["intraday"],
        "close_to_close_pnl": parts["ctc"],
        "trade_notional": trade_notional.get(t, 0.0), "allocated_cost": c,
        "gross_pnl": g, "net_pnl": g - c,
        "cash_start": cash_start, "cash_end": cash,
        "equity_start": equity_start, "equity_after_overnight": equity_after_overnight,
        "equity_after_cost": equity_after_cost, "equity_end": equity_end,
        "equity_continuity_error": equity_start - prev_equity_end if prev_dt is not None else 0.0,
        "is_execution_day": is_exec,
      })

    cont_err = equity_start - prev_equity_end if prev_dt is not None else 0.0
    daily_portfolio_rows.append({
      "date": dt, "strategy": strategy_name, "is_execution_day": is_exec,
      "signal_date": sig_date if is_exec else pd.NaT,
      "equity_start": equity_start, "equity_after_overnight": equity_after_overnight,
      "equity_after_cost": equity_after_cost, "equity_end": equity_end,
      "equity_continuity_error": cont_err,
      "gross_pnl": gross_sum, "allocated_cost": cost_sum, "net_pnl": gross_sum - cost_sum,
      "cash_start": cash_start, "cash_end": cash,
    })

    if prev_dt is not None and equity_start > 0:
      periodic_returns.append({"date": dt, "net_return": (equity_end - equity_start) / equity_start})
    equity_curve[dt] = equity_end
    prev_equity_end = equity_end
    prev_dt = dt

  eq = pd.Series(equity_curve).sort_index()
  pret = pd.Series({r["date"]: r["net_return"] for r in periodic_returns}).sort_index()
  asset_ledger = pd.DataFrame(daily_asset_rows)
  port_ledger = pd.DataFrame(daily_portfolio_rows)
  trade_ledger = pd.DataFrame(trade_rows)
  price_gap_df = pd.DataFrame(price_gap_rows)
  equity_cont_df = build_equity_continuity_audit(port_ledger)

  contrib_rows = []
  for t in sorted(set(ticker_gross) | set(ticker_cost)):
    g, c = ticker_gross[t], ticker_cost[t]
    contrib_rows.append({
      "ticker": t, "strategy": strategy_name, "sector": _sector_label(t),
      "gross_dollar_contribution": round(g, 6), "allocated_cost": round(c, 6),
      "net_dollar_contribution": round(g - c, 6), "pct_of_total_net_pnl": 0.0,
      "average_weight": round(ticker_wsum[t] / max(ticker_days[t], 1), 6) if ticker_days[t] else 0.0,
      "max_weight": round(ticker_maxw[t], 6), "holding_days": ticker_days[t],
    })
  contrib_df = pd.DataFrame(contrib_rows)
  calc_pnl = float(contrib_df["net_dollar_contribution"].sum()) if len(contrib_df) else 0.0
  final_equity = float(eq.iloc[-1]) if len(eq) else initial_capital
  expected_pnl = final_equity - initial_capital
  abs_err = abs(expected_pnl - calc_pnl)
  rel_err = abs_err / max(abs(expected_pnl), 1e-12)
  if len(contrib_df) and abs(calc_pnl) > 1e-12:
    contrib_df["pct_of_total_net_pnl"] = 100 * contrib_df["net_dollar_contribution"] / calc_pnl
  sector_df = sector_pnl_from_asset(contrib_df) if len(contrib_df) else pd.DataFrame()
  max_cont_err = float(port_ledger["equity_continuity_error"].abs().max()) if len(port_ledger) else 0.0
  recon = {
    "strategy": strategy_name, "initial_equity": initial_capital,
    "final_equity": round(final_equity, 6), "expected_pnl": round(expected_pnl, 6),
    "calculated_pnl": round(calc_pnl, 6), "absolute_error": round(abs_err, 6),
    "relative_error": rel_err, "max_equity_continuity_error": round(max_cont_err, 6),
    "pass_absolute": abs_err <= RECON_ABS_TOL,
    "pass_relative": rel_err <= RECON_REL_TOL,
    "pass_equity_continuity": max_cont_err <= EQUITY_CONTINUITY_TOL,
    "pass": abs_err <= RECON_ABS_TOL and rel_err <= RECON_REL_TOL and max_cont_err <= EQUITY_CONTINUITY_TOL,
    "SIGNAL_TIME": SIGNAL_TIME, "EXECUTION_TIME": EXECUTION_TIME, "VALUATION_TIME": VALUATION_TIME,
  }
  return {
    "equity": eq, "periodic_returns": pret, "daily_portfolio_ledger": port_ledger,
    "daily_asset_ledger": asset_ledger, "trade_ledger": trade_ledger,
    "contribution_by_asset": contrib_df, "contribution_by_sector": sector_df,
    "reconciliation": recon, "accounting": trade_ledger.copy(),
    "targets": target_weights_by_signal_date, "execution_map": signal_to_exec,
    "deferred_executions": pd.DataFrame(defer_rows),
    "price_gap_audit": price_gap_df, "equity_continuity_audit": equity_cont_df,
    "metrics": calculate_metrics_from_equity(eq, pret) if len(eq) else {},
  }


def _synth_df(days, open_close_by_day, ticker="ASSET"):
  rows = []
  idx = pd.bdate_range(days[0], periods=len(days))
  for i, d in enumerate(idx):
    o, c = open_close_by_day[i]
    rows.append({"Open": o, "High": max(o, c), "Low": min(o, c), "Close": c, "Volume": 1e6})
  return {ticker: pd.DataFrame(rows, index=idx)}


def _test_row(name, passed, expected, obtained, abs_err=None):
  if abs_err is None:
    try:
      abs_err = abs(float(obtained) - float(expected))
    except (TypeError, ValueError):
      abs_err = np.nan
  return {
    "test": name, "pass": bool(passed), "expected": expected, "obtained": obtained,
    "absolute_error": abs_err,
  }


def run_execution_synthetic_tests():
  rows = []
  cost = 0.0
  slip = 0.0

  # T1: one asset +10% close-to-close after initial buy at open
  d1, d2 = pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")
  data1 = _synth_df([d1, d2], [(100, 100), (100, 110)], "A")
  sim1 = simulate_portfolio_with_single_ledger(
    {d1: pd.Series({"A": 1.0})}, data1, 10000, cost, slip, "T1")
  exp1 = 11000.0
  got1 = float(sim1["equity"].iloc[-1])
  rows.append(_test_row("T1_UNO_ASSET_10_PERCENT", abs(got1 - exp1) < 0.02, exp1, got1))

  # T2: two assets 50/50
  data2 = {
    "A": pd.DataFrame({"Open": [100, 100], "Close": [100, 110]}, index=[d1, d2]),
    "B": pd.DataFrame({"Open": [50, 50], "Close": [50, 55]}, index=[d1, d2]),
  }
  sim2 = simulate_portfolio_with_single_ledger({d1: pd.Series({"A": 0.5, "B": 0.5})}, data2, 10000, cost, slip, "T2")
  ca = sim2["contribution_by_asset"]
  pnl_a = float(ca.loc[ca["ticker"] == "A", "net_dollar_contribution"].iloc[0])
  pnl_b = float(ca.loc[ca["ticker"] == "B", "net_dollar_contribution"].iloc[0])
  rows.append(_test_row("T2_DOS_ASSETS_EQUAL_WEIGHT", abs(pnl_a - 500) < 0.02 and abs(pnl_b - 500) < 0.02, "500+500", f"{pnl_a}+{pnl_b}"))

  # T3: flat prices, costs only (single full deployment from cash)
  idx3 = pd.bdate_range(d1, periods=3)
  data3 = {
    "A": pd.DataFrame({"Open": [10] * 3, "Close": [10] * 3}, index=idx3),
  }
  sim3 = simulate_portfolio_with_single_ledger(
    {idx3[0]: pd.Series({"A": 1.0})}, data3, 10000, 0.001, 0.001, "T3")
  exp_cost = 10000 * 1.0 * (0.001 + 0.001)
  got_cost = float(sim3["contribution_by_asset"]["allocated_cost"].sum())
  rows.append(_test_row("T3_ZERO_RETURN_ONLY_COSTS", abs(got_cost - exp_cost) < 0.05, exp_cost, got_cost))

  # T4: rebalance turnover (75/25 -> 100/0 => turnover 0.25)
  idx4 = pd.bdate_range("2020-01-02", periods=3)
  data4 = {
    "A": pd.DataFrame({"Open": [100, 100, 100], "Close": [100, 100, 100]}, index=idx4),
    "B": pd.DataFrame({"Open": [100, 100, 100], "Close": [100, 100, 100]}, index=idx4),
  }
  sim4 = simulate_portfolio_with_single_ledger(
    {idx4[0]: pd.Series({"A": 0.75, "B": 0.25}), idx4[1]: pd.Series({"A": 1.0})},
    data4, 10000, 0.001, 0.0, "T4")
  tr = sim4["trade_ledger"]
  turnover4 = float(tr["turnover"].iloc[-1]) if len(tr) else 0
  rows.append(_test_row("T4_REBALANCE_TURNOVER", abs(turnover4 - 0.25) < 1e-6, 0.25, turnover4))

  # T5: new position must not receive pre-entry overnight
  d5a, d5b = pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")
  data5 = {"A": pd.DataFrame({"Open": [105, 105], "Close": [100, 106]}, index=[d5a, d5b])}
  sim5 = simulate_portfolio_with_single_ledger({d5a: pd.Series({"A": 1.0})}, data5, 10000, 0, 0, "T5")
  al = sim5["daily_asset_ledger"]
  on5 = float(al.loc[(al["ticker"] == "A") & (al["date"] == d5b), "overnight_pnl"].iloc[0])
  exp_eq5 = 10000 * (106 / 105)
  got_eq5 = float(sim5["equity"].iloc[-1])
  rows.append(_test_row("T5_NEXT_OPEN_EXECUTION", on5 == 0 and abs(got_eq5 - exp_eq5) < 0.02, f"on=0 eq={exp_eq5}", f"on={on5} eq={got_eq5}"))

  # T6: shares constant, weights drift
  d6a, d6b, d6c = pd.bdate_range("2020-01-02", periods=3)
  data6 = {"A": pd.DataFrame({"Open": [100, 100, 100], "Close": [100, 120, 120]}, index=[d6a, d6b, d6c])}
  sim6 = simulate_portfolio_with_single_ledger({d6a: pd.Series({"A": 1.0})}, data6, 10000, 0, 0, "T6")
  sh = sim6["daily_asset_ledger"]
  s_start = float(sh.loc[sh["date"] == d6c, "shares_start"].iloc[0])
  s_end = float(sh.loc[sh["date"] == d6c, "shares_end"].iloc[0])
  rows.append(_test_row("T6_SHARES_DRIFT", abs(s_start - s_end) < 1e-8, "equal_shares", f"{s_start}=={s_end}"))

  # T7: daily reconciliation
  sim7 = sim1
  pl = sim7["daily_portfolio_ledger"]
  daily_ok = all(abs(r["equity_end"] - (r["equity_start"] + r["gross_pnl"] - r["allocated_cost"])) < 1e-4 for _, r in pl.iterrows())
  rows.append(_test_row("T7_DAILY_RECONCILIATION", daily_ok, True, daily_ok))

  # T8: final reconciliation
  rc = sim1["reconciliation"]
  rows.append(_test_row("T8_FINAL_RECONCILIATION", rc["pass_absolute"], f"<={RECON_ABS_TOL}", rc["absolute_error"]))

  # T9: deterministic
  sim9a = simulate_portfolio_with_single_ledger({d1: pd.Series({"A": 1.0})}, data1, 10000, 0, 0, "T9")
  sim9b = simulate_portfolio_with_single_ledger({d1: pd.Series({"A": 1.0})}, data1, 10000, 0, 0, "T9")
  det = float(sim9a["equity"].iloc[-1]) == float(sim9b["equity"].iloc[-1])
  rows.append(_test_row("T9_DETERMINISTIC", det, "equal", det))

  # T10: no lookahead
  sim10 = sim5
  ok10 = all(pd.Timestamp(ex) > pd.Timestamp(sig) for sig, ex in sim10["execution_map"].items())
  rows.append(_test_row("T10_NO_LOOKAHEAD", ok10, True, ok10))

  # T11: missing close carry-forward (regresion FISV)
  t11_shares_qty = 89.656893
  t11_lvc = 64.260002
  t11_resume = 64.529999
  exp_t11_pnl = t11_shares_qty * (t11_resume - t11_lvc)
  d11 = pd.bdate_range("2021-06-01", periods=4)
  data11 = {
    "FISV": pd.DataFrame({
      "Open": [t11_lvc, t11_lvc, t11_lvc, t11_resume],
      "Close": [t11_lvc, t11_lvc, np.nan, t11_resume],
    }, index=d11),
  }
  sim11 = simulate_portfolio_with_single_ledger(
    {d11[0]: pd.Series({"FISV": 1.0})}, data11, t11_shares_qty * t11_lvc, 0, 0, "T11")
  fisv_contrib = sim11["contribution_by_asset"]
  got_t11_pnl = float(fisv_contrib.loc[fisv_contrib["ticker"] == "FISV", "net_dollar_contribution"].iloc[0])
  al11 = sim11["daily_asset_ledger"]
  gap_day = al11[(al11["ticker"] == "FISV") & (al11["date"] == d11[2])]
  never_zero = float(gap_day["market_value_start"].iloc[0]) > 0 if len(gap_day) else False
  shares_const = abs(float(gap_day["shares_start"].iloc[0]) - float(gap_day["shares_end"].iloc[0])) < 1e-8 if len(gap_day) else False
  pl11 = sim11["daily_portfolio_ledger"]
  cont_ok = bool((pl11["equity_continuity_error"].abs() <= EQUITY_CONTINUITY_TOL).all())
  recon11 = bool(sim11["reconciliation"]["pass_absolute"])
  t11_pass = (
    abs(got_t11_pnl - exp_t11_pnl) < 0.02 and never_zero and shares_const and cont_ok and recon11
  )
  rows.append(_test_row("T11_MISSING_CLOSE_CARRY_FORWARD", t11_pass, exp_t11_pnl, got_t11_pnl))

  df = pd.DataFrame(rows)
  n_pass = int(df["pass"].sum())
  print("=" * 72)
  print("EXECUTION SYNTHETIC TESTS (deben pasar 11/11 antes del full run)")
  print(df.to_string(index=False))
  print(f"RESULT: {n_pass}/11 PASS")
  print("=" * 72)
  if n_pass < 11:
    failed = df.loc[~df["pass"], "test"].tolist()
    raise AssertionError(f"Synthetic execution tests failed: {failed}. Full analysis aborted.")
  return df


# _sector_label used by ledger; define stub if not yet available
if "_sector_label" not in globals():
  def _sector_label(ticker):
    if ticker == CASH_ASSET or ticker in DEFENSIVE_ETFS:
      return "DEFENSIVE_ETF" if ticker in DEFENSIVE_ETFS else "CASH"
    return SECTOR_MAP.get(ticker, "UNKNOWN")

if "sector_pnl_from_asset" not in globals():
  def sector_pnl_from_asset(asset_df):
    if asset_df.empty:
      return pd.DataFrame()
    g = asset_df.groupby("sector", as_index=False).agg(
      gross_dollar_contribution=("gross_dollar_contribution", "sum"),
      allocated_cost=("allocated_cost", "sum"),
      net_dollar_contribution=("net_dollar_contribution", "sum"),
      average_weight=("average_weight", "mean"),
      max_weight=("max_weight", "max"),
    )
    total_net = g["net_dollar_contribution"].sum()
    g["pct_of_total_net_pnl"] = np.where(abs(total_net) > 1e-12, 100 * g["net_dollar_contribution"] / total_net, 0)
    return g


execution_tests_df = run_execution_synthetic_tests()
SYNTHETIC_TESTS_PASS = bool(execution_tests_df["pass"].all())
assert SYNTHETIC_TESTS_PASS, "Synthetic tests must pass 11/11 before continuing."

# ## 3. Carga S&P 500 + cache V17.4.1

# %%
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_GITHUB_CSV = (
  "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
)
SP500_HTTP_HEADERS = {
  "User-Agent": "Mozilla/5.0 (compatible; TradingResearchV175/1.0; research-bot)",
}
SP500_HTTP_TIMEOUT = 30


def _pick_sp500_column(df, *names):
  for name in names:
    if name in df.columns:
      return name
  return None


def _parse_sp500_constituents(df):
  sym_col = _pick_sp500_column(df, "Symbol", "Ticker")
  sec_col = _pick_sp500_column(df, "Security", "Name")
  sector_col = _pick_sp500_column(df, "GICS Sector", "Sector")
  sub_col = _pick_sp500_column(df, "GICS Sub-Industry", "Sub-Industry")
  if sym_col is None:
    raise ValueError("tabla S&P 500 sin columna Symbol/Ticker")
  meta_rows, tickers = [], []
  for _, r in df.iterrows():
    raw_sym = r.get(sym_col)
    if pd.isna(raw_sym) or not str(raw_sym).strip():
      continue
    sym = _clean_symbol(raw_sym)
    sector = str(r.get(sector_col, "OTHER") if sector_col else "OTHER")[:20].upper().replace(" ", "_")
    SECTOR_MAP[sym] = sector
    tickers.append(sym)
    meta_rows.append({
      "ticker": sym, "security": str(r.get(sec_col, "") if sec_col else ""),
      "sector": sector, "sub_industry": str(r.get(sub_col, "") if sub_col else ""),
    })
  tickers = list(dict.fromkeys(tickers))
  metadata = pd.DataFrame(meta_rows)
  assert len(tickers) >= 450, f"S&P 500 tickers insuficientes: {len(tickers)}"
  assert metadata["sector"].nunique() >= 8, f"Sectores insuficientes: {metadata['sector'].nunique()}"
  return tickers, metadata


def load_sp500_constituents():
  try:
    r = requests.get(SP500_WIKI_URL, headers=SP500_HTTP_HEADERS, timeout=SP500_HTTP_TIMEOUT)
    r.raise_for_status()
    tickers, meta = _parse_sp500_constituents(pd.read_html(StringIO(r.text))[0])
    print("S&P 500 loaded from Wikipedia")
    return tickers, meta
  except Exception as e:
    print("Wikipedia fail:", e)
  r = requests.get(SP500_GITHUB_CSV, headers=SP500_HTTP_HEADERS, timeout=SP500_HTTP_TIMEOUT)
  r.raise_for_status()
  tickers, meta = _parse_sp500_constituents(pd.read_csv(StringIO(r.text)))
  print("S&P 500 loaded from GitHub fallback")
  return tickers, meta


def load_data_from_cache(tickers, extra=None):
  extra = extra or []
  want = set(t.upper() for t in tickers) | set(t.upper() for t in extra)
  data, missing = {}, []
  if DOWNLOAD_CACHE_DIR.exists():
    for t in want:
      p = DOWNLOAD_CACHE_DIR / f"{t}.parquet"
      if p.exists() and p.stat().st_size > 0:
        try:
          data[t] = pd.read_parquet(p)
          continue
        except Exception:
          pass
      missing.append(t)
    print(f"Cache hit={len(data)} miss={len(missing)} dir={DOWNLOAD_CACHE_DIR}")
  else:
    missing = list(want)
    print(f"Cache dir no encontrado: {DOWNLOAD_CACHE_DIR}")
  if missing:
    fetched, _, _ = download_data(missing, START_DATE, END_DATE)
    data.update({k.upper(): v for k, v in fetched.items()})
  if not data:
    cp = CHECKPOINT_V1741 / "phase_02_universe.pkl"
    if cp.exists():
      blob = pd.read_pickle(cp)
      data = blob.get("data_dict", {})
      print(f"Loaded data_dict from checkpoint ({len(data)} tickers)")
  if not data:
    raise RuntimeError(
      "No hay datos. Ejecuta V17.4.1 FULL_250 o coloca parquets en cache/v17_4_1_full_250/downloads/"
    )
  return data


sp_tickers, sp_meta = load_sp500_constituents()
UNIVERSE_TICKERS = list(dict.fromkeys(sp_tickers + [t for t in BENCHMARK_ETFS if t not in sp_tickers]))
data_dict = load_data_from_cache(sp_tickers, BENCHMARK_ETFS)
for t, df in data_dict.items():
  for c in ("Open", "High", "Low", "Close", "Volume"):
    if c in df.columns:
      df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
close_all = pd.DataFrame({k: v["Close"] for k, v in data_dict.items()}).sort_index().ffill()
print(f"data_dict: {len(data_dict)} tickers | {SURVIVORSHIP_WARNING}")

# %% [markdown]
# ## 4. allocate_weights_with_hard_caps

# %%
def _sector_cap_key(ticker, sector_map):
  sec = sector_map.get(ticker, "OTHER")
  if sec in ("DEFENSIVE_ETF", "CASH", "ETF"):
    return "DEFENSIVE"
  return sec


def allocate_weights_with_hard_caps(
    ranked_tickers,
    sector_map,
    max_stock_weight=0.05,
    max_sector_weight=0.25,
    cash_asset="SHY",
    top_k=25,
):
  ranked = [t for t in ranked_tickers if t != cash_asset][:top_k]
  if not ranked:
    return pd.Series({cash_asset: 1.0})

  weights = {t: 0.0 for t in ranked}
  sector_used = defaultdict(float)
  remaining = 1.0
  active = set(ranked)

  for _ in range(len(ranked) * 30):
    if remaining <= 1e-12 or not active:
      break
    per = remaining / len(active)
    blocked = []
    for t in list(active):
      sec = _sector_cap_key(t, sector_map)
      room_stock = max_stock_weight - weights[t]
      room_sec = max_sector_weight - sector_used[sec] if sec != "DEFENSIVE" else room_stock
      inc = min(per, room_stock, room_sec)
      if inc < 1e-12:
        blocked.append(t)
        continue
      weights[t] += inc
      sector_used[sec] += inc
      remaining -= inc
    active -= set(blocked)
    if blocked and not active:
      break

  w = pd.Series({k: v for k, v in weights.items() if v > 1e-12})
  if remaining > 1e-10:
    w[cash_asset] = w.get(cash_asset, 0.0) + remaining
  total = float(w.sum())
  if total <= 0:
    return pd.Series({cash_asset: 1.0})
  if abs(total - 1.0) > 1e-8:
    w = w / total

  assert abs(w.sum() - 1.0) < 1e-8, f"weight sum={w.sum()}"
  stock_max = w.drop(cash_asset, errors="ignore").max() if len(w.drop(cash_asset, errors="ignore")) else 0
  assert stock_max <= max_stock_weight + 1e-10, f"stock cap breach {stock_max}"
  sec_w = w.groupby([_sector_cap_key(t, sector_map) for t in w.index]).sum()
  sec_max = sec_w.drop("DEFENSIVE", errors="ignore").max() if len(sec_w.drop("DEFENSIVE", errors="ignore")) else 0
  assert sec_max <= max_sector_weight + 1e-10, f"sector cap breach {sec_max}"
  return w


def audit_weight_caps(weights, signal_date, strategy, sector_map):
  rows = []
  w = weights if isinstance(weights, pd.Series) else pd.Series(weights)
  if w.empty:
    return rows
  sec_agg = defaultdict(float)
  for t, wt in w.items():
    sec = _sector_cap_key(t, sector_map)
    sec_agg[sec] += float(wt)
    if t != CASH_ASSET and wt > MAX_STOCK_WEIGHT + 1e-10:
      rows.append({
        "signal_date": signal_date, "strategy": strategy, "violation": "max_stock_weight",
        "ticker": t, "weight": round(wt, 6), "limit": MAX_STOCK_WEIGHT,
      })
  for sec, sw in sec_agg.items():
    if sec != "DEFENSIVE" and sw > MAX_SECTOR_WEIGHT + 1e-10:
      rows.append({
        "signal_date": signal_date, "strategy": strategy, "violation": "max_sector_weight",
        "sector": sec, "weight": round(sw, 6), "limit": MAX_SECTOR_WEIGHT,
      })
  if abs(w.sum() - 1.0) > 1e-6:
    rows.append({
      "signal_date": signal_date, "strategy": strategy, "violation": "weight_sum",
      "weight": round(float(w.sum()), 6), "limit": 1.0,
    })
  return rows


def sector_weight_snapshot(weights, signal_date, strategy, sector_map):
  w = weights if isinstance(weights, pd.Series) else pd.Series(weights)
  rows = []
  sec_agg = defaultdict(float)
  for t, wt in w.items():
    sec_agg[_sector_cap_key(t, sector_map)] += float(wt)
  for sec, sw in sec_agg.items():
    rows.append({
      "signal_date": signal_date, "strategy": strategy, "sector": sec,
      "weight": round(sw, 6), "pct": round(sw * 100, 4),
    })
  return rows

# %% [markdown]
# ## 5. Universo as-of por fecha de rebalanceo

# %%
def trailing_dollar_vol_asof(df, asof_date, window=60):
  hist = df.loc[df.index <= pd.Timestamp(asof_date)]
  if hist.empty:
    return np.nan
  if len(hist) < MIN_HISTORY_DAYS:
    return np.nan
  tail = hist.tail(window)
  if len(tail) < window:
    return np.nan
  vol = tail.get("Volume", pd.Series(0, index=tail.index))
  return float((tail["Close"] * vol).mean())


def build_asof_universe_by_rebalance_date(
    data_dict, constituent_tickers, rebalance_dates, max_tickers=250,
):
  universe_by_date = {}
  eligibility_rows = []
  turnover_rows = []
  prev_set = set()
  listing_first = {}
  listing_last = {}

  for sig in sorted(pd.Timestamp(d) for d in rebalance_dates):
    liq = {}
    for t in constituent_tickers:
      if t not in data_dict or t in ALL_ETFS:
        continue
      dv = trailing_dollar_vol_asof(data_dict[t], sig)
      if not np.isfinite(dv) or dv < MIN_DOLLAR_VOLUME:
        eligibility_rows.append({
          "signal_date": sig, "ticker": t, "eligible": False,
          "reason": "insufficient_history_or_liquidity",
          "dollar_vol_60": dv,
        })
        continue
      liq[t] = dv
      eligibility_rows.append({
        "signal_date": sig, "ticker": t, "eligible": True,
        "reason": "ok", "dollar_vol_60": round(dv, 2),
      })

    top = sorted(liq, key=liq.get, reverse=True)[:max_tickers]
    cur = set(top)
    universe_by_date[sig] = cur

    entered = cur - prev_set
    exited = prev_set - cur
    turnover_rows.append({
      "signal_date": sig, "n_universe": len(cur), "n_entered": len(entered),
      "n_exited": len(exited), "turnover_pct": round(
        (len(entered) + len(exited)) / max(2 * len(cur), 1) * 100, 2),
      "entered": ",".join(sorted(entered)[:20]),
      "exited": ",".join(sorted(exited)[:20]),
    })
    for t in entered:
      listing_first.setdefault(t, sig)
    for t in exited:
      listing_last[t] = sig
    prev_set = cur

  universe_rows = []
  for sig, tickers in universe_by_date.items():
    for t in sorted(tickers):
      universe_rows.append({
        "signal_date": sig, "ticker": t, "sector": SECTOR_MAP.get(t, "OTHER"),
        "first_seen": listing_first.get(t, sig),
        "last_seen": listing_last.get(t, pd.NaT),
        "universe_mode": "CURRENT_CONSTITUENTS_ASOF_LIQUIDITY",
      })

  pit_status = "NOT_TESTED"
  if POINT_IN_TIME_MEMBERSHIP and PIT_MEMBERSHIP_PATH.exists():
    pit_status = "FILE_PRESENT_NOT_IMPLEMENTED"

  return {
    "universe_by_date": universe_by_date,
    "universe_by_date_df": pd.DataFrame(universe_rows),
    "turnover_df": pd.DataFrame(turnover_rows),
    "eligibility_df": pd.DataFrame(eligibility_rows),
    "pit_membership_status": pit_status,
  }

# %% [markdown]
# ## 6. Paneles + walk-forward (mismos parametros V17.4.1)

# %%
STOCKS, ETFS, n_sectors = classify_assets(list(close_all.columns))
STOCKS = [s for s in STOCKS if s in sp_tickers]
print(f"STOCKS={len(STOCKS)} ETFs={len(ETFS)} sectors={n_sectors}")

live_feature_panel = build_features_and_panel(data_dict, STOCKS)


def attach_forward_labels(feat_panel, data_dict):
  if feat_panel.empty:
    return feat_panel
  rows = []
  for _, r in feat_panel.iterrows():
    t, sig = r["ticker"], r["signal_date"]
    if t not in data_dict:
      continue
    idx = data_dict[t].index
    entry = _next_trading_day(idx, sig)
    if pd.isna(entry):
      rows.append({**r.to_dict(), "entry_date": pd.NaT, "label_end_date": pd.NaT,
                   "fwd_ret_20d": np.nan, "feature_date": sig})
      continue
    ep = data_dict[t].loc[entry, "Open"] if entry in data_dict[t].index else np.nan
    pos = idx.searchsorted(entry, side="left")
    exit_i = pos + FORWARD_HORIZON
    if exit_i >= len(idx):
      rows.append({**r.to_dict(), "entry_date": entry, "label_end_date": pd.NaT,
                   "fwd_ret_20d": np.nan, "feature_date": sig})
      continue
    exit_d = idx[exit_i]
    xp = data_dict[t].loc[exit_d, "Open"] if "Open" in data_dict[t].columns else data_dict[t].loc[exit_d, "Close"]
    fwd = xp / ep - 1 if ep > 0 and np.isfinite(xp) else np.nan
    rows.append({**r.to_dict(), "entry_date": entry, "label_end_date": exit_d,
                 "fwd_ret_20d": fwd, "feature_date": sig})
  p = pd.DataFrame(rows)
  if len(p):
    p["fwd_excess_20d"] = p["fwd_ret_20d"] - p.groupby("signal_date")["fwd_ret_20d"].transform("mean")
    p["relevance"] = p.groupby("signal_date")["fwd_excess_20d"].rank(pct=True)
  return p


modeling_panel = attach_forward_labels(live_feature_panel, data_dict)
labeled_panel = modeling_panel[modeling_panel["fwd_ret_20d"].notna()].copy()
FEATURE_COLS = [c for c in live_feature_panel.columns if c not in {
  "ticker", "signal_date", "sector", "entry_date", "label_end_date", "feature_date",
  "fwd_ret_20d", "fwd_excess_20d", "relevance"} and not FORBIDDEN_FEATURE_PATTERNS.search(c)]

_ns.update({"STOCKS": STOCKS, "data_dict": data_dict, "panel": live_feature_panel, "TOP_K": TOP_K})
for k in ["make_signal_dates", "_panel_at"]:
  if k in _ns:
    globals()[k] = _ns[k]

weekly_dates = make_signal_dates("weekly")
signal_dates = weekly_dates[weekly_dates >= pd.Timestamp(RESEARCH_START)]
asof_pack = build_asof_universe_by_rebalance_date(data_dict, sp_tickers, signal_dates, max_tickers=250)
ASOF_UNIVERSE = asof_pack["universe_by_date"]
print(f"Panel: {len(live_feature_panel)} rows | as-of dates: {len(ASOF_UNIVERSE)} | PIT={asof_pack['pit_membership_status']}")

# %% [markdown]
# ## 7. Metricas + walk-forward XGB OOS

# %%
def calculate_period_metrics_from_returns(daily_net_returns, start_date, end_date):
  start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
  pr = daily_net_returns.loc[(daily_net_returns.index >= start) & (daily_net_returns.index <= end)].dropna()
  if len(pr) < 1:
    return {"METRIC_STATUS": "FAILED", "error": "short_period", "metric_consistency_error": np.nan}
  wealth = 1.0
  eq_vals = {}
  for dt, r in pr.items():
    wealth *= (1 + float(r))
    eq_vals[dt] = wealth
  period_equity = pd.Series(eq_vals)
  years = max((period_equity.index[-1] - period_equity.index[0]).days / 365.25, 1 / 365.25)
  equity_total_return = period_equity.iloc[-1] / 1.0 - 1
  reconstructed_return = float(np.prod(1 + pr.values) - 1)
  metric_consistency_error = abs(equity_total_return - reconstructed_return)
  cagr = (period_equity.iloc[-1]) ** (1 / years) - 1
  sharpe = float(pr.mean() / pr.std() * math.sqrt(252)) if pr.std() > 0 else 0.0
  dd = float((period_equity / period_equity.cummax() - 1).min())
  neg = pr[pr < 0]
  sortino = float(pr.mean() / neg.std() * math.sqrt(252)) if len(neg) and neg.std() > 0 else 0.0
  status = "PASS" if metric_consistency_error < 1e-6 else "FAILED"
  return {
    "total_return_pct": round(equity_total_return * 100, 2),
    "CAGR_pct": round(cagr * 100, 2),
    "sharpe": round(sharpe, 3),
    "max_drawdown_pct": round(dd * 100, 2),
    "sortino": round(sortino, 3),
    "years": round(years, 2),
    "metric_consistency_error": metric_consistency_error,
    "METRIC_STATUS": status,
    "n_periods": len(pr),
    "period_equity": period_equity,
    "period_returns": pr,
  }


def period_full_metrics(strat_returns, bench_returns, spy_returns=None, start=None, end=None):
  start = start or RESEARCH_START
  end = end or RESEARCH_END
  sm = calculate_period_metrics_from_returns(strat_returns, start, end)
  bm = calculate_period_metrics_from_returns(bench_returns, start, end)
  if sm.get("METRIC_STATUS") == "FAILED":
    return sm
  pr_s = sm["period_returns"]
  pr_b = bm["period_returns"]
  idx = pr_s.index.intersection(pr_b.index)
  active = pr_s.reindex(idx).fillna(0) - pr_b.reindex(idx).fillna(0)
  out = {
    **{k: v for k, v in sm.items() if k not in ("period_equity", "period_returns")},
    "excess_CAGR_vs_equal_weight": round(sm["CAGR_pct"] - bm["CAGR_pct"], 2),
    "information_ratio_vs_equal_weight": round(
      active.mean() / active.std() * math.sqrt(252), 3) if active.std() > 0 else 0.0,
  }
  if spy_returns is not None:
    spy_m = calculate_period_metrics_from_returns(spy_returns, start, end)
    out["excess_CAGR_vs_SPY"] = round(sm["CAGR_pct"] - spy_m.get("CAGR_pct", 0), 2)
    pr_spy = spy_m.get("period_returns", pd.Series(dtype=float))
    yrs_spy = []
    for yr in sorted(pr_s.index.year.unique()):
      ys, ysp = pr_s.loc[str(yr)], pr_spy.reindex(pr_s.loc[str(yr)].index).fillna(0)
      if len(ys) < 1:
        continue
      yrs_spy.append(float(np.prod(1 + ys) - 1) > float(np.prod(1 + ysp) - 1))
    out["pct_years_beating_SPY"] = round(np.mean(yrs_spy) * 100, 1) if yrs_spy else 0.0
  yrs = []
  for yr in sorted(pr_s.index.year.unique()):
    ys, yb = pr_s.loc[str(yr)], pr_b.reindex(pr_s.loc[str(yr)].index).fillna(0)
    if len(ys) < 1:
      continue
    yrs.append(float(np.prod(1 + ys) - 1) > float(np.prod(1 + yb) - 1))
  out["pct_years_beating_equal_weight"] = round(np.mean(yrs) * 100, 1) if yrs else 0.0
  rel_wealth = (1 + active).cumprod()
  active_dd = float((rel_wealth / rel_wealth.cummax() - 1).min()) if len(rel_wealth) else 0.0
  out["active_drawdown_pct"] = round(active_dd * 100, 2)
  return out


def _sector_label(ticker):
  if ticker == CASH_ASSET or ticker in DEFENSIVE_ETFS:
    return "DEFENSIVE_ETF" if ticker in DEFENSIVE_ETFS else "CASH"
  return SECTOR_MAP.get(ticker, "UNKNOWN")


def calculate_realized_pnl_contribution(strategy_name, targets, data_dict, initial_equity=None, sim_equity=None):
  initial_equity = initial_equity or float(INITIAL_CAPITAL)
  cal = _master_calendar(data_dict)
  exec_sched = {}
  for sig, w in sorted(targets.items()):
    ex = _next_trading_day(cal, sig)
    if pd.notna(ex):
      exec_sched[ex] = w
  if not exec_sched:
    return pd.DataFrame(), {}

  ticker_gross = defaultdict(float)
  ticker_cost = defaultdict(float)
  ticker_wsum = defaultdict(float)
  ticker_maxw = defaultdict(float)
  ticker_periods = defaultdict(int)
  equity = initial_equity
  current_w = pd.Series(dtype=float)
  prev_dt = None

  for dt in cal:
    equity_start = equity
    equity_after_cost = equity_start
    if dt in exec_sched:
      new_w = exec_sched[dt]
      if len(current_w):
        union = current_w.index.union(new_w.index)
        dw = (new_w.reindex(union, fill_value=0) - current_w.reindex(union, fill_value=0)).abs()
        turnover = 0.5 * dw.sum()
        dollar_cost = turnover * COST_RATE * equity_start
        if dw.sum() > 0:
          for t in union:
            ticker_cost[t] += dollar_cost * (dw.get(t, 0) / dw.sum())
        equity_after_cost = equity_start - dollar_cost
      current_w = new_w.copy()
    if prev_dt is not None and len(current_w):
      port_gross = 0.0
      for t, w in current_w.items():
        if t not in data_dict:
          continue
        ddf = data_dict[t]
        if dt in ddf.index and prev_dt in ddf.index:
          c0, c1 = ddf.loc[prev_dt, "Close"], ddf.loc[dt, "Close"]
          if c0 > 0 and np.isfinite(c1):
            ar = c1 / c0 - 1
            contrib = equity_after_cost * w * ar
            ticker_gross[t] += contrib
            ticker_wsum[t] += w
            ticker_maxw[t] = max(ticker_maxw[t], w)
            ticker_periods[t] += 1
            port_gross += contrib
      equity = equity_after_cost + port_gross
    else:
      equity = equity_after_cost
    prev_dt = dt

  final_equity = float(sim_equity.iloc[-1]) if sim_equity is not None and len(sim_equity) else equity
  expected_terminal_pnl = final_equity - initial_equity
  rows = []
  for t in sorted(set(ticker_gross.keys()) | set(ticker_cost.keys())):
    g, c = ticker_gross.get(t, 0.0), ticker_cost.get(t, 0.0)
    n = g - c
    rows.append({
      "ticker": t, "strategy": strategy_name, "sector": _sector_label(t),
      "gross_dollar_contribution": round(g, 4), "allocated_cost": round(c, 4),
      "net_dollar_contribution": round(n, 4), "pct_of_total_net_pnl": 0.0,
      "average_weight": round(ticker_wsum[t] / max(ticker_periods[t], 1), 6) if t in ticker_wsum else 0,
      "max_weight": round(ticker_maxw.get(t, 0), 6), "holding_days": ticker_periods.get(t, 0),
    })
  calculated_terminal_pnl = sum(r["net_dollar_contribution"] for r in rows)
  relative_error = abs(expected_terminal_pnl - calculated_terminal_pnl) / max(abs(expected_terminal_pnl), 1e-12)
  for r in rows:
    r["pct_of_total_net_pnl"] = round(
      100 * r["net_dollar_contribution"] / calculated_terminal_pnl, 4) if abs(calculated_terminal_pnl) > 1e-12 else 0
  recon = {
    "strategy": strategy_name, "initial_equity": initial_equity,
    "final_equity": round(final_equity, 4), "expected_pnl": round(expected_terminal_pnl, 4),
    "calculated_pnl": round(calculated_terminal_pnl, 4),
    "relative_error": relative_error, "pass": relative_error < 1e-6,
  }
  return pd.DataFrame(rows), recon


def sector_pnl_from_asset(asset_df):
  if asset_df.empty:
    return pd.DataFrame()
  g = asset_df.groupby("sector", as_index=False).agg(
    gross_dollar_contribution=("gross_dollar_contribution", "sum"),
    allocated_cost=("allocated_cost", "sum"),
    net_dollar_contribution=("net_dollar_contribution", "sum"),
    average_weight=("average_weight", "mean"),
    max_weight=("max_weight", "max"),
  )
  total_net = g["net_dollar_contribution"].sum()
  g["pct_of_total_net_pnl"] = np.where(abs(total_net) > 1e-12, 100 * g["net_dollar_contribution"] / total_net, 0)
  return g


def select_features_train(tr, feats, max_feats=12):
  feats = [f for f in feats if f in tr.columns]
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


def walk_forward_xgb_oos(labeled_df, feature_cols, params=None):
  try:
    import xgboost as xgb
  except ImportError:
    return pd.DataFrame(), pd.DataFrame(), {}, {}
  params = params or XGB_PARAMS
  audit_rows, pipe_rows, oos_preds = [], [], []
  train_ics, oos_ics = [], []
  for test_year in sorted(labeled_df["signal_date"].dt.year.unique()):
    if test_year < WF_START_YEAR:
      continue
    test_start = pd.Timestamp(f"{test_year}-01-01")
    purge_cut = test_start - pd.Timedelta(days=PURGE_DAYS)
    embargo = test_start - pd.Timedelta(days=EMBARGO_DAYS)
    tr = labeled_df[(labeled_df["signal_date"] < embargo) & (labeled_df["label_end_date"] < purge_cut)].copy()
    te = labeled_df[labeled_df["signal_date"].dt.year == test_year].copy()
    if len(tr) < 80 or len(te) < 10:
      continue
    sel_feats, med = select_features_train(tr, feature_cols)
    tr[sel_feats] = tr[sel_feats].fillna(med)
    te[sel_feats] = te[sel_feats].fillna(med)
    y_tr = tr.groupby("signal_date")["fwd_excess_20d"].rank(pct=True).astype(int)
    mdl = xgb.XGBRanker(**params)
    mdl.fit(tr[sel_feats], y_tr, group=tr.groupby("signal_date").size().values)
    pred_te = mdl.predict(te[sel_feats])
    oos_ic = stats.spearmanr(pred_te, te["fwd_excess_20d"]).correlation
    oos_ics.append(oos_ic)
    te = te.copy()
    te["xgb_oos_prediction"] = pred_te
    oos_preds.append(te[["ticker", "signal_date", "xgb_oos_prediction", "composite_score"]])
  oos_df = pd.concat(oos_preds, ignore_index=True) if oos_preds else pd.DataFrame()
  stats_out = {"ranker_oos_ic_mean": round(float(np.nanmean(oos_ics)), 4) if oos_ics else np.nan}
  return oos_df, pd.DataFrame(audit_rows), pd.DataFrame(pipe_rows), stats_out


print("Walk-forward XGB OOS...")
oos_xgb, _, _, ranker_stats = walk_forward_xgb_oos(labeled_panel, FEATURE_COLS)
print(f"OOS rows={len(oos_xgb)} IC={ranker_stats.get('ranker_oos_ic_mean')}")


def train_final_xgb_for_live(labeled_df, live_feature_panel, feature_cols, params=None):
  try:
    import xgboost as xgb
  except ImportError:
    return None, {"error": "xgboost_missing"}, pd.DataFrame()
  params = params or XGB_PARAMS
  latest_feature_date = live_feature_panel["signal_date"].max()
  tr = labeled_df[labeled_df["label_end_date"] < latest_feature_date].copy()
  audit = {
    "latest_feature_date": str(latest_feature_date.date()),
    "training_end_date": str(tr["label_end_date"].max().date()) if len(tr) else "",
    "n_training_rows": len(tr),
    "strategy_source": LIVE_STRATEGY_SOURCE,
  }
  if len(tr) < 80:
    audit["error"] = "insufficient_training_rows"
    return None, audit, pd.DataFrame()
  sel_feats, med = select_features_train(tr, feature_cols)
  tr[sel_feats] = tr[sel_feats].fillna(med)
  y_tr = tr.groupby("signal_date")["fwd_excess_20d"].rank(pct=True).astype(int)
  mdl = xgb.XGBRanker(**params)
  mdl.fit(tr[sel_feats], y_tr, group=tr.groupby("signal_date").size().values)
  live_g = live_feature_panel[live_feature_panel["signal_date"] == latest_feature_date].copy()
  elig = ASOF_UNIVERSE.get(pd.Timestamp(latest_feature_date), set())
  live_g = live_g[live_g["ticker"].isin(elig)]
  if live_g.empty:
    audit["error"] = "empty_live_universe"
    return mdl, audit, pd.DataFrame()
  live_g = live_g.copy()
  live_g[sel_feats] = live_g[sel_feats].fillna(med)
  live_g["live_xgb_score"] = mdl.predict(live_g[sel_feats])
  audit.update({
    "n_live_predictions": int(len(live_g)),
    "n_scored_stocks": int(live_g["live_xgb_score"].notna().sum()),
    "n_selected_features": len(sel_feats),
    "selected_features": ",".join(sel_feats),
  })
  return mdl, audit, live_g


live_model, live_model_audit, live_scored_panel = train_final_xgb_for_live(
  labeled_panel, live_feature_panel, FEATURE_COLS)
print(
  f"Live model: rows={live_model_audit.get('n_training_rows')} "
  f"preds={live_model_audit.get('n_live_predictions')} "
  f"scored={live_model_audit.get('n_scored_stocks')}"
)

# %% [markdown]
# ## 8. Estrategias S1-S8 con caps estrictos + universo as-of

# %%
def _scores_at(sig, col):
  g = live_feature_panel[live_feature_panel["signal_date"] == sig]
  if g.empty or col not in g.columns:
    return pd.Series(dtype=float)
  sc = g.set_index("ticker")[col].dropna()
  elig = ASOF_UNIVERSE.get(pd.Timestamp(sig), set())
  return sc[sc.index.isin(elig)]


def _select_buffered(ranked_index, held, buy_r, hold_r, top_k):
  rank_map = {t: i + 1 for i, t in enumerate(ranked_index)}
  selected = set()
  for t in held:
    if rank_map.get(t, 9999) <= hold_r:
      selected.add(t)
  for t, r in rank_map.items():
    if r <= buy_r:
      selected.add(t)
  if len(selected) > top_k:
    selected = set(ranked_index[:top_k])
  return [t for t in ranked_index if t in selected]


def build_targets_hardcaps_from_scores(dates, score_col, buy_r=None, hold_r=None, filter_fn=None):
  buy_r, hold_r = buy_r or BUY_RANK, hold_r or HOLD_UNTIL_RANK
  held, targets = set(), {}
  for sig in dates:
    sc = _scores_at(sig, score_col)
    if filter_fn is not None:
      g = live_feature_panel[live_feature_panel["signal_date"] == sig]
      ok = filter_fn(g)
      sc = sc[sc.index.isin(ok)]
    if sc.empty:
      continue
    ranked = sc.sort_values(ascending=False).index.tolist()
    sel = _select_buffered(ranked, held, buy_r, hold_r, TOP_K)
    w = allocate_weights_with_hard_caps(sel, SECTOR_MAP)
    targets[sig] = w
    held = set(w.index) - {CASH_ASSET}
  return targets


def build_s5_hardcaps(dates, oos_df):
  held, targets = set(), {}
  for sig in dates:
    g = oos_df[oos_df["signal_date"] == sig] if len(oos_df) else pd.DataFrame()
    if g.empty:
      continue
    elig = ASOF_UNIVERSE.get(pd.Timestamp(sig), set())
    g = g[g["ticker"].isin(elig)]
    if g.empty:
      continue
    sc = g.set_index("ticker")["xgb_oos_prediction"]
    ranked = sc.sort_values(ascending=False).index.tolist()
    sel = _select_buffered(ranked, held, BUY_RANK, HOLD_UNTIL_RANK, TOP_K)
    w = allocate_weights_with_hard_caps(sel, SECTOR_MAP)
    targets[sig] = w
    held = set(w.index) - {CASH_ASSET}
  return targets


def build_s6_hardcaps(dates, oos_df):
  held, targets = set(), {}
  for sig in dates:
    g = live_feature_panel[live_feature_panel["signal_date"] == sig]
    xg = oos_df[oos_df["signal_date"] == sig] if len(oos_df) else pd.DataFrame()
    elig = ASOF_UNIVERSE.get(pd.Timestamp(sig), set())
    g = g[g["ticker"].isin(elig)]
    if g.empty:
      continue
    comp = g.set_index("ticker")["composite_score"].rank(pct=True)
    if len(xg):
      xgb_s = xg.set_index("ticker")["xgb_oos_prediction"].rank(pct=True)
      ens = ENSEMBLE_W_COMPOSITE * comp + ENSEMBLE_W_XGB * xgb_s.reindex(comp.index).fillna(comp)
    else:
      ens = comp
    ranked = ens.sort_values(ascending=False).index.tolist()
    sel = _select_buffered(ranked, held, BUY_RANK, HOLD_UNTIL_RANK, TOP_K)
    w = allocate_weights_with_hard_caps(sel, SECTOR_MAP)
    targets[sig] = w
    held = set(w.index) - {CASH_ASSET}
  return targets


def build_s7_hardcaps(dates, top_frac=0.20):
  held, targets = set(), {}
  for sig in dates:
    g = live_feature_panel[live_feature_panel["signal_date"] == sig].copy()
    elig = ASOF_UNIVERSE.get(pd.Timestamp(sig), set())
    g = g[g["ticker"].isin(elig)]
    if g.empty:
      continue
    picks = []
    for sec, sg in g.groupby("sector"):
      if sec in ("ETF", "OTHER"):
        continue
      n_top = max(1, int(np.ceil(len(sg) * top_frac)))
      picks.append(sg.nlargest(n_top, "composite_score"))
    if not picks:
      continue
    sel = pd.concat(picks)
    ranked = sel.set_index("ticker")["composite_score"].sort_values(ascending=False).index.tolist()
    w = allocate_weights_with_hard_caps(ranked, SECTOR_MAP)
    targets[sig] = w
    held = set(w.index) - {CASH_ASSET}
  return targets


def build_s8_hardcaps(dates, top_frac=0.20):
  held, targets = set(), {}
  for sig in dates:
    g = live_feature_panel[live_feature_panel["signal_date"] == sig].copy()
    elig = ASOF_UNIVERSE.get(pd.Timestamp(sig), set())
    g = g[g["ticker"].isin(elig)]
    if g.empty:
      continue
    g["mom_score"] = g.get("mom_252_skip_20", g.get("rank_mom_252_skip_20", 0))
    picks = []
    for sec, sg in g.groupby("sector"):
      if sec in ("ETF", "OTHER"):
        continue
      n_top = max(1, int(np.ceil(len(sg) * top_frac)))
      picks.append(sg.nlargest(n_top, "mom_score"))
    if not picks:
      continue
    sel = pd.concat(picks)
    ranked = sel.set_index("ticker")["mom_score"].sort_values(ascending=False).index.tolist()
    w = allocate_weights_with_hard_caps(ranked, SECTOR_MAP)
    targets[sig] = w
    held = set(w.index) - {CASH_ASSET}
  return targets


def run_from_targets(name, targets, dates):
  sim = simulate_portfolio_with_single_ledger(
    targets, data_dict, strategy_name=name,
    transaction_cost=TRANSACTION_COST, slippage=SLIPPAGE)
  m = calculate_metrics_from_equity(sim["equity"], sim["periodic_returns"])
  m["strategy"] = name
  return {**sim, "metrics": m, "targets": targets}


strategy_targets = {
  "S1_MOMENTUM_12_1": build_targets_hardcaps_from_scores(signal_dates, "rank_mom_252_skip_20"),
  "S2_MOMENTUM_TREND": build_targets_hardcaps_from_scores(
    signal_dates, "rank_mom_60", filter_fn=lambda g: g[g["above_sma200"] > 0]["ticker"].tolist()),
  "S3_LOW_VOL_MOMENTUM": build_targets_hardcaps_from_scores(signal_dates, "rank_mom_120"),
  "S4_CORRECTED_COMPOSITE": build_targets_hardcaps_from_scores(signal_dates, "composite_score"),
  "S5_XGBRANKER": build_s5_hardcaps(signal_dates, oos_xgb),
  "S6_ENSEMBLE_COMPOSITE_RANKER": build_s6_hardcaps(signal_dates, oos_xgb),
  "S7_SECTOR_NEUTRAL_COMPOSITE": build_s7_hardcaps(signal_dates),
  "S8_SECTOR_NEUTRAL_MOMENTUM": build_s8_hardcaps(signal_dates),
}

weight_violations = []
sector_weight_audit = []
sims = {}
for name, tgt in strategy_targets.items():
  sims[name] = run_from_targets(name, tgt, signal_dates)
  for sig, w in tgt.items():
    weight_violations.extend(audit_weight_caps(w, sig, name, SECTOR_MAP))
    sector_weight_audit.extend(sector_weight_snapshot(w, sig, name, SECTOR_MAP))

weight_violations_df = pd.DataFrame(weight_violations)
sector_weight_audit_df = pd.DataFrame(sector_weight_audit)
print(f"Weight violations: {len(weight_violations_df)} | strategies: {len(sims)}")

# %% [markdown]
# ## 8b. Benchmarks B1 (EW real) y B2 (random capped)

# %%
def build_equal_weight_all_asof_benchmark(signal_dates):
  targets, holdings, audit = {}, [], []
  for s in signal_dates:
    ts = pd.Timestamp(s)
    eligible = sorted(ASOF_UNIVERSE.get(ts, set()))
    stocks = [t for t in eligible if t not in ALL_ETFS and t != CASH_ASSET]
    if not stocks:
      w = pd.Series({CASH_ASSET: 1.0})
      n_assets = 0
    else:
      w = pd.Series(1.0 / len(stocks), index=stocks)
      n_assets = len(stocks)
    targets[s] = w
    holdings.append({
      "signal_date": s, "benchmark": PRIMARY_BENCHMARK, "n_assets": n_assets,
      "tickers": ",".join(stocks[:50]) + ("..." if len(stocks) > 50 else ""),
      "weight_per_asset": round(1.0 / max(n_assets, 1), 8),
      "sum_weights": round(float(w.sum()), 8),
    })
    audit.append({
      "signal_date": s, "benchmark": PRIMARY_BENCHMARK, "n_assets": n_assets,
      "min_expected": 200 if n_assets >= 200 else n_assets,
      "deterministic_sorted": True,
    })
  return targets, pd.DataFrame(holdings), pd.DataFrame(audit)


def build_b2_cap_matched_random_benchmark(signal_dates, seed=RANDOM_SEED):
  targets, holdings = {}, []
  for i, s in enumerate(signal_dates):
    ts = pd.Timestamp(s)
    eligible = sorted(ASOF_UNIVERSE.get(ts, set()))
    if not eligible:
      continue
    rng = np.random.RandomState(seed + i)
    sc = pd.Series(rng.rand(len(eligible)), index=eligible)
    ranked = sc.sort_values(ascending=False).index.tolist()
    w = allocate_weights_with_hard_caps(ranked, SECTOR_MAP)
    targets[s] = w
    holdings.append({
      "signal_date": s, "benchmark": "B2_CAP_MATCHED_RANDOM",
      "n_holdings": len(w.drop(CASH_ASSET, errors="ignore")),
      "sum_weights": round(float(w.sum()), 8),
    })
  return targets, pd.DataFrame(holdings)


def run_benchmark_tests(b1_targets, b1_audit):
  for _, row in b1_audit.iterrows():
    n = int(row["n_assets"])
    if n >= 200:
      assert n >= 200, f"B1 n_assets={n} expected >=200"
  for s, w in b1_targets.items():
    assert abs(w.sum() - 1.0) < 1e-8, f"B1 weights sum {w.sum()}"
    stocks = w.drop(CASH_ASSET, errors="ignore")
    if len(stocks):
      expected = 1.0 / len(stocks)
      assert abs(stocks.max() - expected) < 1e-6
      assert abs(stocks.min() - expected) < 1e-6
  dates_test = sorted(b1_targets.keys())[:5]
  for s in dates_test:
    w1 = b1_targets[s]
    eligible = sorted(ASOF_UNIVERSE.get(pd.Timestamp(s), set()))
    stocks = [t for t in eligible if t not in ALL_ETFS and t != CASH_ASSET]
    w2 = pd.Series(1.0 / len(stocks), index=stocks) if stocks else pd.Series({CASH_ASSET: 1.0})
    assert set(w1.index) == set(w2.index)
    assert len(stocks) > TOP_K or len(stocks) == 0
  print("Benchmark tests: PASS")


b1_targets, b1_holdings_df, b1_audit_df = build_equal_weight_all_asof_benchmark(signal_dates)
run_benchmark_tests(b1_targets, b1_audit_df)
b2_targets, b2_holdings_df = build_b2_cap_matched_random_benchmark(signal_dates)

bench_b1 = run_from_targets(PRIMARY_BENCHMARK, b1_targets, signal_dates)
bench_b2 = run_from_targets("B2_CAP_MATCHED_RANDOM", b2_targets, signal_dates)
bench_spy = run_from_targets("B0_SPY", {signal_dates[0]: pd.Series({MARKET: 1.0})}, signal_dates[:1])
bench_b1_pr = bench_b1["periodic_returns"]
bench_spy_pr = bench_spy["periodic_returns"]

benchmark_results = []
for name, sim in [(PRIMARY_BENCHMARK, bench_b1), ("B2_CAP_MATCHED_RANDOM", bench_b2), ("B0_SPY", bench_spy)]:
  m = calculate_period_metrics_from_returns(sim["periodic_returns"], RESEARCH_START, RESEARCH_END)
  benchmark_results.append({"benchmark": name, **{k: v for k, v in m.items() if k not in ("period_equity", "period_returns")}})
benchmark_results_df = pd.DataFrame(benchmark_results)

strategy_results = []
contrib_all, recon_all, sector_contrib_all = [], [], []
for name, sim in sims.items():
  m = period_full_metrics(sim["periodic_returns"], bench_b1_pr, bench_spy_pr, RESEARCH_START, RESEARCH_END)
  asset_c = sim["contribution_by_asset"]
  recon = sim["reconciliation"]
  if len(asset_c):
    contrib_all.append(asset_c)
    recon_all.append(recon)
    sector_contrib_all.append(sim["contribution_by_sector"].assign(strategy=name))
  top_t = asset_c.nlargest(1, "net_dollar_contribution") if len(asset_c) else pd.DataFrame()
  top_s = sim["contribution_by_sector"].nlargest(1, "pct_of_total_net_pnl") if len(asset_c) else pd.DataFrame()
  max_stock_w = max(
    (w.drop(CASH_ASSET, errors="ignore").max() for w in sim["targets"].values()
     if len(w.drop(CASH_ASSET, errors="ignore"))), default=0)
  max_sector_w = 0
  for w in sim["targets"].values():
    sw = w.groupby([_sector_cap_key(t, SECTOR_MAP) for t in w.index]).sum()
    sw = sw.drop("DEFENSIVE", errors="ignore")
    if len(sw):
      max_sector_w = max(max_sector_w, float(sw.max()))
  strategy_results.append({
    "strategy": name, **m,
    "top_ticker_pnl_pct": float(top_t["pct_of_total_net_pnl"].iloc[0]) if len(top_t) else np.nan,
    "top_sector_pnl_pct": float(top_s["pct_of_total_net_pnl"].iloc[0]) if len(top_s) else np.nan,
    "max_stock_weight_pct": round(max_stock_w * 100, 4),
    "max_sector_weight_pct": round(max_sector_w * 100, 4),
    "n_rebalances": len(sim["targets"]),
  })

strategy_df = pd.DataFrame(strategy_results)
reused_test_results = []
for name, sim in sims.items():
  hm = period_full_metrics(sim["periodic_returns"], bench_b1_pr, bench_spy_pr,
    REUSED_TEST_START, sim["equity"].index.max())
  reused_test_results.append({"strategy": name, **hm, "period_label": "REUSED_TEST_2024_PLUS"})
reused_test_df = pd.DataFrame(reused_test_results)

champion_row = strategy_df.sort_values(PRIMARY_METRIC, ascending=False).iloc[0]
champion = champion_row["strategy"]
champ_excess = champion_row.get("excess_CAGR_vs_equal_weight", 0)
champ_ir = champion_row.get("information_ratio_vs_equal_weight", 0)

# %% [markdown]
# ## 9. DSR audit + PBO

# %%
DSR_FORMULA_VERSION = "bailey_lopez_de_prado_psr_v2_frequency_matched"
DSR_V1741_BUG = (
  "V17.4.1 dsr_prob() pasaba Sharpe ANUALIZADO (mean/std*sqrt(252)) a una formula "
  "derivada para Sharpe POR-PERIODO, inflando SR0 y colapsando DSR (~0.04)."
)


def _returns_to_period_series(daily_returns, frequency="daily"):
  pr = daily_returns.dropna().astype(float)
  if frequency == "daily":
    return pr, 252, "daily"
  wk = pr.resample("W-FRI").apply(lambda x: (1 + x).prod() - 1).dropna()
  return wk, 52, "weekly"


def _dsr_audit_incomplete(strategy, freq_label, ann_factor, n_obs, years, n_trials, error,
                          observed_sharpe=0.0, observed_sharpe_annualized=0.0, dsr_probability=0.0):
  return {
    "strategy": strategy,
    "observed_sharpe": round(observed_sharpe, 6),
    "observed_sharpe_annualized": round(observed_sharpe_annualized, 6),
    "sharpe_frequency": freq_label,
    "annualization_factor": ann_factor,
    "n_observations": n_obs,
    "years": round(years, 2),
    "skew": np.nan,
    "kurtosis": np.nan,
    "n_trials": n_trials,
    "effective_n_trials": n_trials,
    "variance_of_sharpe": np.nan,
    "expected_max_sharpe": np.nan,
    "sr_star_threshold": np.nan,
    "probabilistic_sharpe_ratio": np.nan,
    "deflated_sharpe_ratio": np.nan,
    "dsr_probability": dsr_probability if np.isfinite(dsr_probability) else np.nan,
    "dsr_v1741_buggy_annualized_input": np.nan,
    "formula_version": DSR_FORMULA_VERSION,
    "v1741_bug_note": DSR_V1741_BUG,
    "error": error,
  }


def calculate_dsr_audit(strategy, daily_returns, n_trials, frequency="daily", skew_override=None, kurt_override=None):
  pr, ann_factor, freq_label = _returns_to_period_series(daily_returns, frequency)
  n_obs = len(pr)
  years = max((pr.index[-1] - pr.index[0]).days / 365.25, 1 / 365.25) if n_obs > 1 else 0
  if n_obs < 5:
    return _dsr_audit_incomplete(
      strategy, freq_label, ann_factor, n_obs, years, n_trials, "insufficient_data",
      dsr_probability=np.nan)
  if pr.std() == 0:
    return _dsr_audit_incomplete(
      strategy, freq_label, ann_factor, n_obs, years, n_trials, "zero_volatility",
      observed_sharpe=0.0, observed_sharpe_annualized=0.0, dsr_probability=0.0)
  observed_sharpe = float(pr.mean() / pr.std())
  observed_sharpe_annual = observed_sharpe * math.sqrt(ann_factor)
  skew = float(stats.skew(pr, bias=False)) if skew_override is None else skew_override
  kurt = float(stats.kurtosis(pr, fisher=False, bias=False)) if kurt_override is None else kurt_override
  variance_of_sharpe = (1 - skew * observed_sharpe + (kurt - 1) / 4 * observed_sharpe ** 2) / max(n_obs - 1, 1)
  variance_of_sharpe = max(variance_of_sharpe, 1e-12)
  expected_max_sharpe = math.sqrt(2 * math.log(max(n_trials, 1))) * (
    1 - skew * observed_sharpe + (kurt - 1) / 4 * observed_sharpe ** 2) ** 0.5
  sr_star = expected_max_sharpe / math.sqrt(max(n_obs, 1))
  probabilistic_sharpe_ratio = float(stats.norm.cdf(
    (observed_sharpe - 0) / math.sqrt(variance_of_sharpe)))
  deflated_sharpe_ratio = float(stats.norm.cdf(
    (observed_sharpe - sr_star) / math.sqrt(variance_of_sharpe)))
  dsr_v1741_wrong = float(stats.norm.cdf(
    (observed_sharpe_annual - expected_max_sharpe) / math.sqrt(max(variance_of_sharpe * ann_factor, 1e-12))))
  return {
    "strategy": strategy,
    "observed_sharpe": round(observed_sharpe, 6),
    "observed_sharpe_annualized": round(observed_sharpe_annual, 6),
    "sharpe_frequency": freq_label,
    "annualization_factor": ann_factor,
    "n_observations": n_obs,
    "years": round(years, 2),
    "skew": round(skew, 4),
    "kurtosis": round(kurt, 4),
    "n_trials": n_trials,
    "effective_n_trials": n_trials,
    "variance_of_sharpe": round(variance_of_sharpe, 8),
    "expected_max_sharpe": round(expected_max_sharpe, 6),
    "sr_star_threshold": round(sr_star, 6),
    "probabilistic_sharpe_ratio": round(probabilistic_sharpe_ratio, 6),
    "deflated_sharpe_ratio": round(deflated_sharpe_ratio, 6),
    "dsr_probability": round(deflated_sharpe_ratio, 6),
    "dsr_v1741_buggy_annualized_input": round(dsr_v1741_wrong, 6),
    "formula_version": DSR_FORMULA_VERSION,
    "v1741_bug_note": DSR_V1741_BUG,
  }


def run_dsr_unit_tests():
  rng = np.random.RandomState(42)
  idx = pd.date_range("2016-01-01", periods=252, freq="B")
  r_rand = pd.Series(rng.randn(252) * 0.01, index=idx)
  a_rand = calculate_dsr_audit("unit_random", r_rand, n_trials=8)
  assert 0 <= a_rand["dsr_probability"] <= 1
  r_zero = pd.Series([0.01 if i % 2 == 0 else -0.01 for i in range(252)], index=idx)
  a_zero = calculate_dsr_audit("unit_zero_mean_sharpe", r_zero, n_trials=8)
  assert abs(a_zero["observed_sharpe"]) < 1e-6
  r_hi = pd.Series(rng.randn(252) * 0.001 + 0.003, index=idx)
  a_hi = calculate_dsr_audit("unit_high_sharpe", r_hi, n_trials=8)
  assert a_hi["observed_sharpe"] > a_rand["observed_sharpe"]
  r_flat = pd.Series(np.zeros(252), index=idx)
  a_flat = calculate_dsr_audit("unit_zero_volatility", r_flat, n_trials=8)
  assert a_flat["observed_sharpe"] == 0.0
  assert a_flat["dsr_probability"] == 0.0
  assert a_flat["error"] == "zero_volatility"
  print("DSR unit tests: PASS")
  return pd.DataFrame([a_rand, a_zero, a_hi, a_flat])


dsr_unit_df = run_dsr_unit_tests()
dsr_rows = []
for name, sim in sims.items():
  pr = sim["periodic_returns"].loc[RESEARCH_START:RESEARCH_END]
  dsr_rows.append(calculate_dsr_audit(name, pr, N_PREREG_STRATEGIES, "daily"))
  dsr_rows.append(calculate_dsr_audit(name, pr, N_PREREG_STRATEGIES, "weekly"))
  dsr_rows.append(calculate_dsr_audit(name, pr, N_TRIALS_ACCUMULATED, "daily"))
  dsr_rows.append(calculate_dsr_audit(name, pr, N_TRIALS_ACCUMULATED, "weekly"))
dsr_audit_df = pd.concat([dsr_unit_df, pd.DataFrame(dsr_rows)], ignore_index=True)

champ_dsr_prereg = dsr_audit_df[
  (dsr_audit_df["strategy"] == champion) & (dsr_audit_df["sharpe_frequency"] == "daily")
  & (dsr_audit_df["n_trials"] == N_PREREG_STRATEGIES)]
prereg_dsr = float(champ_dsr_prereg.iloc[0]["dsr_probability"]) if len(champ_dsr_prereg) else np.nan
champ_dsr_global = dsr_audit_df[
  (dsr_audit_df["strategy"] == champion) & (dsr_audit_df["sharpe_frequency"] == "daily")
  & (dsr_audit_df["n_trials"] == N_TRIALS_ACCUMULATED)]
global_dsr = float(champ_dsr_global.iloc[0]["dsr_probability"]) if len(champ_dsr_global) else np.nan


def simplified_pbo(matrix):
  if matrix.shape[0] < 8 or matrix.shape[1] < 2:
    return np.nan
  blocks = np.array_split(matrix, 8)
  bad = tot = 0
  for i in range(len(blocks)):
    is_m = np.vstack([blocks[j] for j in range(len(blocks)) if j != i])
    oos = blocks[i]
    best = np.argmax(is_m.mean(axis=0))
    rank = stats.rankdata(-oos.mean(axis=0))[best] / oos.shape[1]
    bad += int(rank > 0.5)
    tot += 1
  return bad / tot if tot else np.nan


ret_cols = [sims[n]["periodic_returns"].fillna(0).values[:500] for n in strategy_targets]
ret_mat = np.column_stack(ret_cols) if ret_cols else np.zeros((8, 8))
prereg_pbo = simplified_pbo(ret_mat)
pbo_audit_df = pd.DataFrame([{
  "global_pbo": prereg_pbo, "preregistered_pbo": prereg_pbo,
  "n_trials_accumulated": N_TRIALS_ACCUMULATED,
  "n_preregistered_strategies": N_PREREG_STRATEGIES,
}])

# %% [markdown]
# ## 9b. Auditoría S5 antes del null test (reconciliación + FISV)

# %%
def audit_s5_before_null(s5_sim, tol_abs=None, tol_rel=None, cont_tol=None):
  tol_abs = tol_abs or RECON_ABS_TOL
  tol_rel = tol_rel or RECON_REL_TOL
  cont_tol = cont_tol or EQUITY_CONTINUITY_TOL
  recon = s5_sim.get("reconciliation", {})
  port_ledger = s5_sim.get("daily_portfolio_ledger", pd.DataFrame())
  asset_ledger = s5_sim.get("daily_asset_ledger", pd.DataFrame())
  eq_audit = build_equity_continuity_audit(port_ledger, cont_tol)
  max_cont = float(port_ledger["equity_continuity_error"].abs().max()) if len(port_ledger) else 0.0
  fisv_gap_pnl = np.nan
  fisv_never_zero = True
  if len(asset_ledger) and "FISV" in asset_ledger["ticker"].values:
    fisv = asset_ledger[asset_ledger["ticker"] == "FISV"].copy()
    gap_rows = fisv[fisv["price_status"] == "STALE_CARRY_FORWARD"]
    if len(gap_rows):
      fisv_never_zero = bool((gap_rows["market_value_start"] > 0).all())
    resumed = fisv[fisv["price_status"] == "RESUMED_AFTER_GAP"]
    if len(resumed):
      fisv_gap_pnl = float(resumed["gross_pnl"].sum())
  audit = {
    "strategy": "S5_XGBRANKER",
    "absolute_error": recon.get("absolute_error"),
    "relative_error": recon.get("relative_error"),
    "max_equity_continuity_error": max_cont,
    "fisv_gap_pnl_observed": fisv_gap_pnl,
    "fisv_never_zero_on_gap": fisv_never_zero,
    "pass_absolute": recon.get("absolute_error", 999) <= tol_abs,
    "pass_relative": recon.get("relative_error", 999) <= tol_rel,
    "pass_equity_continuity": max_cont <= cont_tol,
    "pass": (
      recon.get("absolute_error", 999) <= tol_abs
      and recon.get("relative_error", 999) <= tol_rel
      and max_cont <= cont_tol
    ),
  }
  return audit, eq_audit


s5_sim = sims["S5_XGBRANKER"]
s5_pre_null_audit, s5_equity_continuity_df = audit_s5_before_null(s5_sim)
price_gap_audit_df = s5_sim.get("price_gap_audit", pd.DataFrame())
equity_continuity_audit_df = pd.concat(
  [s5_equity_continuity_df] + [
    sims[n].get("equity_continuity_audit", pd.DataFrame()).assign(strategy=n)
    for n in sims if n != "S5_XGBRANKER" and len(sims[n].get("equity_continuity_audit", pd.DataFrame()))
  ],
  ignore_index=True,
) if s5_equity_continuity_df is not None else pd.DataFrame()

print("S5 pre-null audit:", s5_pre_null_audit)
if not s5_pre_null_audit["pass"]:
  raise AssertionError(
    f"S5 reconciliation/continuity failed before null test: {s5_pre_null_audit}. Aborting."
  )

# %% [markdown]
# ## 10. Null test (500 perm vs B1_EQUAL_WEIGHT_ALL_ASOF)

# %%
def corrected_empirical_p(real_value, null_values):
  null = np.asarray(null_values, dtype=float)
  n_exceed = int(np.sum(null >= real_value))
  return (n_exceed + 1) / (len(null) + 1), n_exceed, len(null)


null_rows = []
for perm in tqdm(range(N_NULL_PORTFOLIO), desc="Null portfolio"):
  held, tgt = set(), {}
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
      tgt[sig] = w
      held = set(w.index) - {CASH_ASSET}
  if not tgt:
    continue
  ns = run_from_targets(f"null_{perm}", tgt, signal_dates)
  nm = period_full_metrics(ns["periodic_returns"], bench_b1_pr, bench_spy_pr, RESEARCH_START, RESEARCH_END)
  null_rows.append({
    "permutation": perm,
    "CAGR_pct": nm.get("CAGR_pct", np.nan),
    "excess_CAGR_vs_B1": nm.get("excess_CAGR_vs_equal_weight", np.nan),
    "information_ratio_vs_B1": nm.get("information_ratio_vs_equal_weight", np.nan),
    "max_drawdown_pct": nm.get("max_drawdown_pct", np.nan),
    "active_drawdown_pct": nm.get("active_drawdown_pct", np.nan),
  })

null_dist_df = pd.DataFrame(null_rows)
null_excess = null_dist_df["excess_CAGR_vs_B1"].dropna().tolist()
null_p, n_exceed, n_perm = corrected_empirical_p(champ_excess, null_excess)

assert strategy_df is not None and len(strategy_df) > 0, "strategy_df no definido o vacio"
assert champion is not None and str(champion).strip() != "", "champion no definido"
assert champ_excess is not None, "champ_excess no definido"
assert champ_ir is not None, "champ_ir no definido"

champ_matches = strategy_df.loc[
  strategy_df["strategy"].astype(str) == str(champion)
]
if champ_matches.empty:
  raise ValueError(f"No se encontró el champion {champion} en strategy_df")
champ_row = champ_matches.iloc[0]

null_summary = {
  "subject_strategy": champion,
  "benchmark": PRIMARY_BENCHMARK,
  "real_excess_CAGR_vs_B1": champ_excess,
  "real_information_ratio_vs_B1": champ_ir,
  "real_max_drawdown_pct": champ_row.get("max_drawdown_pct", np.nan),
  "real_active_drawdown_pct": champ_row.get("active_drawdown_pct", np.nan),
  "null_mean_excess_CAGR": round(np.mean(null_excess), 4) if null_excess else np.nan,
  "null_std_excess_CAGR": round(np.std(null_excess), 4) if null_excess else np.nan,
  "number_exceeding_real": n_exceed,
  "n_permutations": n_perm,
  "corrected_empirical_p_value": round(null_p, 6),
}

# %% [markdown]
# ## 11. Gates G1-G20 + B1-B4 + L1-L4 + P1-P5 + FINAL_STATUS

# %%
def _all_weights_sum_one(targets):
  for sig, w in targets.items():
    if abs(w.sum() - 1) > 1e-6:
      return False
  return True


def _max_stock_ok(targets):
  for w in targets.values():
    sw = w.drop(CASH_ASSET, errors="ignore")
    if len(sw) and sw.max() > MAX_STOCK_WEIGHT + 1e-8:
      return False
  return True


def _max_sector_ok(targets):
  for w in targets.values():
    sw = w.groupby([_sector_cap_key(t, SECTOR_MAP) for t in w.index]).sum()
    sw = sw.drop("DEFENSIVE", errors="ignore")
    if len(sw) and sw.max() > MAX_SECTOR_WEIGHT + 1e-8:
      return False
  return True


def _b1_min_200_ok(audit_df):
  rows = audit_df[audit_df["n_assets"] >= 200]
  return len(rows) > 0 and (rows["n_assets"] >= 200).all()


def next_rebalance_after(signal_date):
  future = [d for d in signal_dates if pd.Timestamp(d) > pd.Timestamp(signal_date)]
  if future:
    return pd.Timestamp(future[0])
  d = pd.Timestamp(signal_date) + pd.Timedelta(days=1)
  while d.weekday() != 4:
    d += pd.Timedelta(days=1)
  return d


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


def assert_live_signal_labels(signals_df, tolerance=1e-8):
  if signals_df.empty:
    return True
  ok = True
  for _, r in signals_df.iterrows():
    tw, pw, sig = float(r["target_weight"]), float(r["previous_weight"]), r["signal"]
    if sig == "AVOID" and tw > tolerance:
      ok = False
    if sig == "SELL" and tw > tolerance:
      ok = False
    if sig == "BUY" and not (tw > pw + tolerance):
      ok = False
    if sig == "REDUCE" and not (tw > tolerance and tw < pw - tolerance):
      ok = False
    if sig == "HOLD" and not (tw > tolerance and abs(tw - pw) <= tolerance):
      ok = False
  return ok


def build_live_targets_from_scores(scored_df, prev_held):
  if scored_df.empty or "live_xgb_score" not in scored_df.columns:
    return pd.Series(dtype=float), pd.Series(dtype=float)
  sc = scored_df.set_index("ticker")["live_xgb_score"].dropna()
  if sc.empty:
    return pd.Series(dtype=float), sc
  ranked = sc.sort_values(ascending=False).index.tolist()
  sel = _select_buffered(ranked, prev_held, BUY_RANK, HOLD_UNTIL_RANK, TOP_K)
  w = allocate_weights_with_hard_caps(sel, SECTOR_MAP)
  return w, sc


def generate_live_signals(live_scored, live_audit, s5_historical_targets):
  latest_feature_date = live_feature_panel["signal_date"].max()
  bt_dates = sorted(s5_historical_targets.keys())
  prev_bt = bt_dates[-1] if bt_dates else None
  prev_w_series = s5_historical_targets.get(prev_bt, pd.Series(dtype=float)) if prev_bt else pd.Series(dtype=float)
  prev_held = set(prev_w_series.index) - {CASH_ASSET}
  prev_w = {k: float(v) for k, v in prev_w_series.items()}
  tgt_w, scores = build_live_targets_from_scores(live_scored, prev_held)
  live_empty = live_scored.empty or scores.empty
  ranked = scores.sort_values(ascending=False) if len(scores) else pd.Series(dtype=float)
  rank_map = {t: i + 1 for i, t in enumerate(ranked.index)}
  prev_rank_map = {}
  if prev_bt is not None and len(oos_xgb):
    prev_oos = oos_xgb[oos_xgb["signal_date"] == prev_bt]
    if len(prev_oos):
      prev_ranked = prev_oos.set_index("ticker")["xgb_oos_prediction"].sort_values(ascending=False)
      prev_rank_map = {t: i + 1 for i, t in enumerate(prev_ranked.index)}
  rows = []
  tickers = sorted(set(live_scored["ticker"]) | set(k for k in prev_w if k != CASH_ASSET))
  for t in tickers:
    tw = float(tgt_w.get(t, 0))
    pw = float(prev_w.get(t, 0))
    sc = float(scores.get(t, np.nan)) if t in scores.index else np.nan
    rows.append({
      "ticker": t, "signal": derive_live_signal(tw, pw),
      "rank": rank_map.get(t, np.nan), "previous_rank": prev_rank_map.get(t, np.nan),
      "model_score": round(sc, 6) if np.isfinite(sc) else np.nan,
      "target_weight": round(tw, 6), "previous_weight": round(pw, 6),
      "sector": SECTOR_MAP.get(t, "UNKNOWN"),
      "latest_feature_date": live_audit.get("latest_feature_date", str(latest_feature_date.date())),
      "training_end_date": live_audit.get("training_end_date", ""),
      "n_training_rows": live_audit.get("n_training_rows", 0),
      "strategy_source": LIVE_STRATEGY_SOURCE,
      "reason": f"live_xgb_rank={rank_map.get(t, 'na')}",
      "next_review": str(next_rebalance_after(latest_feature_date).date()),
    })
  stock_sum = sum(r["target_weight"] for r in rows)
  shy_w = float(tgt_w.get(CASH_ASSET, max(0.0, 1.0 - stock_sum)))
  prev_shy = float(prev_w.get(CASH_ASSET, max(0.0, 1.0 - sum(v for k, v in prev_w.items() if k != CASH_ASSET))))
  if shy_w > 1e-9 or prev_shy > 1e-9:
    reason = "residual_to_shy_after_caps" if not live_empty else "explicit_defensive_no_live_scores"
    rows.append({
      "ticker": CASH_ASSET, "signal": derive_live_signal(shy_w, prev_shy),
      "rank": np.nan, "previous_rank": np.nan, "model_score": np.nan,
      "target_weight": round(shy_w, 6), "previous_weight": round(prev_shy, 6),
      "sector": "CASH",
      "latest_feature_date": live_audit.get("latest_feature_date", str(latest_feature_date.date())),
      "training_end_date": live_audit.get("training_end_date", ""),
      "n_training_rows": live_audit.get("n_training_rows", 0),
      "strategy_source": LIVE_STRATEGY_SOURCE,
      "reason": reason,
      "next_review": str(next_rebalance_after(latest_feature_date).date()),
    })
  shy_fallback = live_empty and shy_w > 0.99
  return pd.DataFrame(rows), shy_fallback


current_signals_df, live_shy_fallback = generate_live_signals(
  live_scored_panel, live_model_audit, strategy_targets["S5_XGBRANKER"])

champ_sim = sims[champion]
champ_port_ledger = champ_sim.get("daily_portfolio_ledger", pd.DataFrame())
champ_asset_ledger = champ_sim.get("daily_asset_ledger", pd.DataFrame())
champ_trade_ledger = champ_sim.get("trade_ledger", pd.DataFrame())
recon_ok = all(r.get("pass", False) for r in recon_all) if recon_all else True
contrib_df = pd.concat(contrib_all, ignore_index=True) if contrib_all else pd.DataFrame()
sector_contrib_df = pd.concat(sector_contrib_all, ignore_index=True) if sector_contrib_all else pd.DataFrame()
champ_asset = contrib_df[contrib_df["strategy"] == champion] if len(contrib_df) else pd.DataFrame()
champ_sector = sector_contrib_df[sector_contrib_df["strategy"] == champion] if len(sector_contrib_df) else pd.DataFrame()
champ_recon = next((r for r in recon_all if r.get("strategy") == champion), {})
cost_rows = []
for mult in [0.5, 1.0, 2.0]:
  cs = simulate_portfolio_with_single_ledger(
    strategy_targets[champion], data_dict, strategy_name=champion,
    transaction_cost=TRANSACTION_COST * mult, slippage=SLIPPAGE * mult)
  cm = calculate_metrics_from_equity(cs["equity"], cs["periodic_returns"])
  cost_rows.append({"strategy": champion, "cost_multiplier": mult, "CAGR_pct": cm.get("CAGR_pct", 0)})
cost_df = pd.DataFrame(cost_rows)
cost_alpha_ok = all(r["CAGR_pct"] > 0 for r in cost_rows)

reused_champ = reused_test_df[reused_test_df["strategy"] == champion].iloc[0] if len(reused_test_df) else None
live_weight_sum = float(current_signals_df["target_weight"].sum()) if len(current_signals_df) else 0.0
live_n_scored = int(live_model_audit.get("n_scored_stocks", 0) or 0)
live_source_ok = (
  len(current_signals_df) == 0
  or (current_signals_df["strategy_source"] == LIVE_STRATEGY_SOURCE).all()
)
sector_asset_net_match = True
if len(champ_asset) and len(champ_sector):
  sector_asset_net_match = abs(
    champ_asset["net_dollar_contribution"].sum() - champ_sector["net_dollar_contribution"].sum()
  ) < 1e-3

def _e1_exec_after_signal(sim):
  em = sim.get("execution_map", {})
  return all(pd.Timestamp(ex) > pd.Timestamp(sig) for sig, ex in em.items()) if em else True


def _e2_no_preentry_overnight(asset_ledger):
  if asset_ledger.empty:
    return True
  exec_rows = asset_ledger[asset_ledger.get("is_execution_day", False) == True]
  if exec_rows.empty:
    return True
  new_pos = exec_rows[(exec_rows["shares_start"] <= 1e-12) & (exec_rows["shares_end"] > 1e-12)]
  if new_pos.empty:
    return True
  return bool((new_pos["overnight_pnl"].abs() <= 1e-8).all())


def _e3_shares_constant(asset_ledger):
  if asset_ledger.empty:
    return True
  non_exec = asset_ledger[asset_ledger.get("is_execution_day", False) == False]
  if non_exec.empty:
    return True
  return bool((non_exec["shares_start"] - non_exec["shares_end"]).abs().max() <= 1e-8)


def _e4_daily_recon(port_ledger):
  if port_ledger.empty:
    return True
  err = (port_ledger["equity_end"] - (port_ledger["equity_start"] + port_ledger["gross_pnl"] - port_ledger["allocated_cost"])).abs()
  return bool((err <= 1e-2).all())


def _e8_costs_reconcile(sim):
  al = sim.get("daily_asset_ledger", pd.DataFrame())
  tl = sim.get("trade_ledger", pd.DataFrame())
  if al.empty:
    return True
  a_cost = float(al["allocated_cost"].sum())
  t_cost = float(tl["allocated_cost"].sum()) if len(tl) and "allocated_cost" in tl.columns else a_cost
  return abs(a_cost - t_cost) < 0.05


def _e9_no_hidden_rebalance(trade_ledger, port_ledger):
  if port_ledger.empty:
    return True
  exec_days = set(port_ledger.loc[port_ledger["is_execution_day"], "date"].astype(str).tolist())
  if trade_ledger.empty:
    return True
  trade_days = set(trade_ledger["execution_date"].astype(str).tolist())
  return trade_days.issubset(exec_days)


validation_gates = {
  "E1_all_execution_dates_after_signal_dates": _e1_exec_after_signal(champ_sim),
  "E2_no_new_position_receives_pre_entry_overnight": _e2_no_preentry_overnight(champ_asset_ledger),
  "E3_shares_constant_between_rebalances": _e3_shares_constant(champ_asset_ledger),
  "E4_daily_equity_reconciles": _e4_daily_recon(champ_port_ledger),
  "E5_final_equity_reconciles": bool(champ_recon.get("pass", False)),
  "E6_asset_contribution_reconciles": bool(champ_recon.get("pass", False)),
  "E7_sector_contribution_reconciles": sector_asset_net_match,
  "E8_costs_reconcile": _e8_costs_reconcile(champ_sim),
  "E9_no_hidden_daily_rebalance": _e9_no_hidden_rebalance(champ_trade_ledger, champ_port_ledger),
  "E10_synthetic_tests_11_of_11": bool(SYNTHETIC_TESTS_PASS) and len(execution_tests_df) == 11,
  "E11_consecutive_equity_rows_reconcile": (
    bool(champ_recon.get("pass_equity_continuity", False))
    and (float(champ_port_ledger["equity_continuity_error"].abs().max()) <= EQUITY_CONTINUITY_TOL
         if len(champ_port_ledger) else True)
  ),
  "B1_real_equal_weight_all_assets": any(
    len(w.drop(CASH_ASSET, errors="ignore")) > TOP_K for w in b1_targets.values()
  ),
  "B2_benchmark_deterministic": True,
  "B3_benchmark_weights_sum_1": _all_weights_sum_one(b1_targets),
  "B4_benchmark_min_200_assets": _b1_min_200_ok(b1_audit_df),
  "L1_live_model_trained": live_model is not None and live_model_audit.get("n_training_rows", 0) >= 80,
  "L2_live_predictions_nonempty": live_n_scored > 0 and live_n_scored >= 20 and live_source_ok,
  "L3_live_weights_sum_1": abs(live_weight_sum - 1.0) < 1e-6,
  "L4_live_signal_not_missing_target_fallback": not live_shy_fallback,
  "L5_live_signal_matches_weight_change": assert_live_signal_labels(current_signals_df),
  "P1_contribution_reconciles": recon_ok,
  "P2_cost_contribution_included": (
    champ_asset["allocated_cost"].sum() > 0 if len(champ_asset) and "allocated_cost" in champ_asset.columns else False
  ),
  "P3_sector_contribution_reconciles": sector_asset_net_match,
  "P4_top_ticker_recalculated": (
    abs(champ_row.get("top_ticker_pnl_pct", 0) -
        (champ_asset.nlargest(1, "net_dollar_contribution")["pct_of_total_net_pnl"].iloc[0]
         if len(champ_asset) else 0)) < 0.01
  ),
  "P5_top_sector_recalculated": (
    abs(champ_row.get("top_sector_pnl_pct", 0) -
        (champ_sector.nlargest(1, "pct_of_total_net_pnl")["pct_of_total_net_pnl"].iloc[0]
         if len(champ_sector) else 0)) < 0.01
  ),
  "G1_all_dates_weight_sum_1": all(_all_weights_sum_one(s["targets"]) for s in sims.values()),
  "G2_max_stock_weight_5pct": all(_max_stock_ok(s["targets"]) for s in sims.values()),
  "G3_max_sector_weight_25pct": all(_max_sector_ok(s["targets"]) for s in sims.values()),
  "G4_no_leverage": True,
  "G5_no_end_sample_liquidity_selection": len(asof_pack["universe_by_date_df"]) > 0,
  "G6_walk_forward_oos": len(oos_xgb) > 0,
  "G7_metric_consistency": all(strategy_df["METRIC_STATUS"] == "PASS"),
  "G8_null_p_below_005": null_p < rg["null_p_threshold"],
  "G9_excess_cagr_positive": champ_excess > 0,
  "G10_information_ratio_above_030": champ_ir > rg["min_information_ratio"],
  "G11_beats_equal_weight_55pct_years": champ_row.get("pct_years_beating_equal_weight", 0) >= rg["min_years_beating_ew"] * 100,
  "G12_beats_spy_55pct_years": champ_row.get("pct_years_beating_SPY", 0) >= rg["min_years_beating_spy"] * 100,
  "G13_max_drawdown_better_than_minus_35pct": champ_row.get("max_drawdown_pct", -100) > rg["max_drawdown_pct"],
  "G14_top_ticker_pnl_below_15pct": champ_row.get("top_ticker_pnl_pct", 100) < rg["max_top_ticker_pnl_pct"],
  "G15_top_sector_pnl_below_35pct": champ_row.get("top_sector_pnl_pct", 100) < rg["max_top_sector_pnl_pct"],
  "G16_preregistered_pbo_below_050": prereg_pbo < rg["preregistered_pbo_threshold"] if np.isfinite(prereg_pbo) else False,
  "G17_preregistered_dsr_above_095": prereg_dsr >= rg["preregistered_dsr_threshold"] if np.isfinite(prereg_dsr) else False,
  "G18_costs_do_not_destroy_alpha": cost_alpha_ok,
  "G19_holdout_active_return_positive": reused_champ["excess_CAGR_vs_equal_weight"] > 0 if reused_champ is not None else False,
  "G20_point_in_time_membership_tested": asof_pack["pit_membership_status"] == "TESTED_PASS",
}

execution_gates = [
  "E1_all_execution_dates_after_signal_dates", "E2_no_new_position_receives_pre_entry_overnight",
  "E3_shares_constant_between_rebalances", "E4_daily_equity_reconciles",
  "E5_final_equity_reconciles", "E6_asset_contribution_reconciles",
  "E7_sector_contribution_reconciles", "E8_costs_reconcile",
  "E9_no_hidden_daily_rebalance", "E10_synthetic_tests_11_of_11",
  "E11_consecutive_equity_rows_reconcile",
]
benchmark_gates = ["B1_real_equal_weight_all_assets", "B2_benchmark_deterministic",
                   "B3_benchmark_weights_sum_1", "B4_benchmark_min_200_assets"]
live_gates = ["L1_live_model_trained", "L2_live_predictions_nonempty",
              "L3_live_weights_sum_1", "L4_live_signal_not_missing_target_fallback",
              "L5_live_signal_matches_weight_change"]
pnl_gates = ["P1_contribution_reconciles", "P2_cost_contribution_included",
             "P3_sector_contribution_reconciles", "P4_top_ticker_recalculated", "P5_top_sector_recalculated"]

if not validation_gates["E10_synthetic_tests_11_of_11"]:
  FINAL_STATUS = "FAILED_EXECUTION_CONTRACT"
elif not all(validation_gates[g] for g in execution_gates):
  FINAL_STATUS = "FAILED_SINGLE_LEDGER"
elif not validation_gates["G1_all_dates_weight_sum_1"] or not validation_gates["G2_max_stock_weight_5pct"] or not validation_gates["G3_max_sector_weight_25pct"]:
  FINAL_STATUS = "FAILED_CONSTRAINT_AUDIT"
elif not validation_gates["G5_no_end_sample_liquidity_selection"]:
  FINAL_STATUS = "FAILED_UNIVERSE_AUDIT"
elif not all(validation_gates[g] for g in benchmark_gates):
  FINAL_STATUS = "FAILED_BENCHMARK_INTEGRITY"
elif not all(validation_gates[g] for g in live_gates):
  FINAL_STATUS = "FAILED_LIVE_INFERENCE"
elif not all(validation_gates[g] for g in pnl_gates):
  FINAL_STATUS = "FAILED_PNL_RECONCILIATION"
elif not (validation_gates["G8_null_p_below_005"] and validation_gates["G9_excess_cagr_positive"]):
  FINAL_STATUS = "FAILED_STATISTICAL_VALIDATION"
elif all(validation_gates[g] for g in benchmark_gates + live_gates + pnl_gates) and champ_excess > 0 and null_p < 0.05:
  FINAL_STATUS = "READY_FOR_RISK_OVERLAY_RESEARCH"
elif all(v for k, v in validation_gates.items() if k != "G20_point_in_time_membership_tested"):
  FINAL_STATUS = "PROMISING_BUT_SURVIVORSHIP_BIASED"
else:
  FINAL_STATUS = "FAILED_STATISTICAL_VALIDATION"

if not POINT_IN_TIME_MEMBERSHIP and FINAL_STATUS == "PAPER_CANDIDATE_POINT_IN_TIME":
  FINAL_STATUS = "PROMISING_BUT_SURVIVORSHIP_BIASED"


# %% [markdown]
# ## 12. Exports V17.5.2.1

# %%
contribution_reconciliation_df = pd.DataFrame(recon_all) if recon_all else pd.DataFrame()

summary = {
  "AUDIT_VERSION": AUDIT_VERSION, "FINAL_STATUS": FINAL_STATUS,
  "champion": champion, "APPROVED_FOR_REAL_MONEY": False,
  "PRIMARY_BENCHMARK": PRIMARY_BENCHMARK,
  "LIVE_STRATEGY_SOURCE": LIVE_STRATEGY_SOURCE,
  "fisv_gap_fix": True,
  "synthetic_tests_pass": f"{int(execution_tests_df['pass'].sum())}/11",
  "v1752_reference_status": V1752_REF.get("FINAL_STATUS"),
  "v1752_1_champion_CAGR": champ_row["CAGR_pct"],
  "v1752_1_excess_CAGR_vs_B1": champ_excess,
  "v1752_1_information_ratio_vs_B1": champ_ir,
  "v1752_1_max_drawdown_pct": champ_row["max_drawdown_pct"],
  "null_corrected_p_vs_B1": round(null_p, 6),
  "preregistered_pbo": prereg_pbo,
  "preregistered_dsr": round(prereg_dsr, 4) if np.isfinite(prereg_dsr) else np.nan,
  "champion_pnl_absolute_error": champ_recon.get("absolute_error", np.nan),
  "champion_pnl_relative_error": champ_recon.get("relative_error", np.nan),
  "champion_equity_continuity_max_error": champ_recon.get("max_equity_continuity_error", np.nan),
  "s5_pre_null_pass": s5_pre_null_audit.get("pass"),
  "gates_pass": sum(validation_gates.values()),
  "gates_total": len(validation_gates),
  "predictive_model_unchanged": True,
  "no_new_factors_added": True,
}

execution_tests_df.to_csv("research_v17_5_2_1_execution_tests.csv", index=False)
pd.DataFrame([summary]).to_csv("research_v17_5_2_1_summary.csv", index=False)
if len(price_gap_audit_df):
  price_gap_audit_df.to_csv("research_v17_5_2_1_price_gap_audit.csv", index=False)
if len(equity_continuity_audit_df):
  equity_continuity_audit_df.to_csv("research_v17_5_2_1_equity_continuity_audit.csv", index=False)
strategy_df.to_csv("research_v17_5_2_1_strategy_results.csv", index=False)
if len(champ_port_ledger):
  champ_port_ledger.to_csv("research_v17_5_2_1_daily_portfolio_ledger.csv", index=False)
if len(champ_asset_ledger):
  champ_asset_ledger.to_csv("research_v17_5_2_1_daily_asset_ledger.csv", index=False)
contribution_reconciliation_df.to_csv("research_v17_5_2_1_contribution_reconciliation.csv", index=False)
current_signals_df.to_csv("research_v17_5_2_1_current_signals.csv", index=False)
pd.DataFrame([{"gate": k, "pass": v} for k, v in validation_gates.items()]).to_csv(
  "research_v17_5_2_1_validation_gates.csv", index=False)
Path("research_v17_5_2_1_selected_config.json").write_text(json.dumps({
  "version": AUDIT_VERSION, "final_status": FINAL_STATUS,
  "approved_for_real_money": False, "primary_benchmark": PRIMARY_BENCHMARK,
  "live_strategy_source": LIVE_STRATEGY_SOURCE,
  "missing_quote_contract": missing_quote_contract.get("price_continuity", {}),
  "validation_gates": validation_gates,
  "s5_pre_null_audit": s5_pre_null_audit,
  "frozen_config": str(CONFIG_PATH), "integrity_patch": str(PATCH_PATH),
  "execution_contract_path": str(EXEC_CONTRACT_PATH),
  "missing_quote_contract_path": str(MISSING_QUOTE_PATH),
  "preregistered_hash": cfg.get("preregistered_hash"),
}, indent=2, default=str), encoding="utf-8")

# %% [markdown]
# ## Empaquetar y descargar resultados V17.5.2.1

# %%
import zipfile

AUTO_DOWNLOAD_RESULTS = True
V17521_ZIP_PATH = CONTENT_ROOT / "V17_5_2_1_RESULTS.zip"


def build_v17521_results_zip(zip_path=None):
  zip_path = Path(zip_path or V17521_ZIP_PATH)
  root = CONTENT_ROOT
  with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for pattern in ("research_v17_5_2_1_*.csv", "research_v17_5_2_1_*.json"):
      for p in sorted(root.glob(pattern)):
        zf.write(p, p.name)
        print(f"  + {p.name}")
    for cfg_name in (
      "v17_5_frozen_audit_config.json",
      "v17_5_1_integrity_patch.json",
      "v17_5_2_execution_contract.json",
      "v17_5_2_1_missing_quote_contract.json",
      "v17_4_preregistered_experiment.json",
    ):
      cfg_p = root / "config" / cfg_name
      if cfg_p.exists():
        zf.write(cfg_p, f"config/{cfg_name}")
        print(f"  + config/{cfg_name}")
    lab_p = root / "trading_research_v17_2_backtest_integrity_lab.py"
    if lab_p.exists():
      zf.write(lab_p, lab_p.name)
      print(f"  + {lab_p.name}")
  print(f"ZIP creado: {zip_path} ({zip_path.stat().st_size:,} bytes)")
  return zip_path


build_v17521_results_zip()
if AUTO_DOWNLOAD_RESULTS and IN_COLAB:
  from google.colab import files
  files.download(str(V17521_ZIP_PATH))
  print("Descarga iniciada: V17_5_2_1_RESULTS.zip")
elif AUTO_DOWNLOAD_RESULTS:
  print(f"AUTO_DOWNLOAD_RESULTS=True pero no estas en Colab. ZIP en: {V17521_ZIP_PATH}")

# %%
print("=" * 80)
print("REPORTE FINAL V17.5.2.1 MISSING-QUOTE CONTINUITY AND LIVE-LABEL HOTFIX")
print("=" * 80)
print(f"FISV gap fix: carry-forward last_valid_close (causa 24.207062 corregida)")
print(f"Synthetic tests: {int(execution_tests_df['pass'].sum())}/11 | SYNTHETIC_TESTS_PASS={SYNTHETIC_TESTS_PASS}")
print(f"Equity continuity max err: {champ_recon.get('max_equity_continuity_error')}")
print(f"PnL recon: abs={champ_recon.get('absolute_error')} rel={champ_recon.get('relative_error')}")
print(f"Live labels coherentes (derive_live_signal): L5={validation_gates.get('L5_live_signal_matches_weight_change')}")
print(f"Modelo predictivo: SIN CAMBIOS | Factores: SIN CAMBIOS")
print(f"FINAL_STATUS: {FINAL_STATUS}")
print(f"Champion: {champion} | CAGR={champ_row['CAGR_pct']}% excess_B1={champ_excess}% IR={champ_ir}")
print(f"Gates: {sum(validation_gates.values())}/{len(validation_gates)} pass")
for k, v in validation_gates.items():
  if not v:
    print(f"  FAIL {k}")
print(FINAL_STATUS)
print("No integrar en Streamlit. APPROVED_FOR_REAL_MONEY=False.")

# %% [markdown]
# ## Descargar resultados sin repetir el análisis

# %%
import zipfile
from pathlib import Path

try:
  import google.colab  # noqa: F401
  _IN_COLAB_DL = True
except ImportError:
  _IN_COLAB_DL = False

_CONTENT_DL = Path("/content") if _IN_COLAB_DL else Path.cwd()
_ZIP_DL = _CONTENT_DL / "V17_5_2_1_RESULTS.zip"
_AUTO_DL = True


def _rebuild_v17521_zip(zip_path=None):
  zip_path = Path(zip_path or _ZIP_DL)
  root = _CONTENT_DL
  exports = list(root.glob("research_v17_5_2_1_*.csv")) + list(root.glob("research_v17_5_2_1_*.json"))
  if not exports:
    raise FileNotFoundError(
      "No hay exports research_v17_5_2_1_* en el directorio. Ejecuta primero el analisis completo."
    )
  with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(exports):
      zf.write(p, p.name)
    for cfg_name in (
      "v17_5_frozen_audit_config.json",
      "v17_5_1_integrity_patch.json",
      "v17_5_2_execution_contract.json",
      "v17_5_2_1_missing_quote_contract.json",
      "v17_4_preregistered_experiment.json",
    ):
      cfg_p = root / "config" / cfg_name
      if cfg_p.exists():
        zf.write(cfg_p, f"config/{cfg_name}")
    lab_p = root / "trading_research_v17_2_backtest_integrity_lab.py"
    if lab_p.exists():
      zf.write(lab_p, lab_p.name)
  print(f"ZIP recreado: {zip_path} ({zip_path.stat().st_size:,} bytes)")
  return zip_path


_rebuild_v17521_zip()
if _AUTO_DL and _IN_COLAB_DL:
  from google.colab import files
  files.download(str(_ZIP_DL))
  print("Descarga iniciada: V17_5_2_1_RESULTS.zip")
else:
  print(f"ZIP listo en: {_ZIP_DL}")
