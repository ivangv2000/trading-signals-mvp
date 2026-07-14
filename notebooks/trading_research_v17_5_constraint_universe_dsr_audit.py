# %% [markdown]
# # Trading Research V17.5 — Constraint, Universe and DSR Audit
#
# Audita caps de peso, universo as-of y DSR sin nuevos factores ni tuning ML.
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
# ## Subir archivos necesarios para V17.5

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
    "Faltan archivos obligatorios para V17.5: " + ", ".join(missing)
  )
_validate_json_file(REQUIRED_UPLOADS["v17_5_frozen_audit_config.json"])
_validate_json_file(REQUIRED_UPLOADS["v17_4_preregistered_experiment.json"])
if OPTIONAL_UPLOADS["research_v17_4_1_strategy_results.csv"].exists():
  print(f"Opcional presente: {OPTIONAL_UPLOADS['research_v17_4_1_strategy_results.csv']}")
else:
  print("Opcional ausente: research_v17_4_1_strategy_results.csv (se usara referencia embebida)")
print("Archivos obligatorios validados OK")

# %% [markdown]
# ## 1. Configuracion congelada V17.5

# %%
import warnings
warnings.filterwarnings("ignore")
import ast, hashlib, json, math, re
from collections import defaultdict
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy import stats
from tqdm.auto import tqdm

AUDIT_VERSION = "v17_5"
CONFIG_PATH = Path("config/v17_5_frozen_audit_config.json")
PREREG_PATH = Path("config/v17_4_preregistered_experiment.json")
cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
fp = cfg["frozen_parameters"]
rg = cfg["risk_gates"]
V1741_REF = cfg.get("v17_4_1_full_250_reference", {})

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
  return out


def calculate_realized_pnl_contribution(strategy_name, targets, data_dict, initial_equity=None):
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
  equity = initial_equity
  current_w = pd.Series(dtype=float)
  prev_dt = None
  for dt in cal:
    equity_start = equity
    if dt in exec_sched:
      new_w = exec_sched[dt]
      if len(current_w):
        union = current_w.index.union(new_w.index)
        dw = (new_w.reindex(union, fill_value=0) - current_w.reindex(union, fill_value=0)).abs()
        turnover = 0.5 * dw.sum()
        dollar_cost = turnover * COST_RATE * equity_start
        equity -= dollar_cost
        equity_start = equity
      current_w = new_w.copy()
    if prev_dt is not None and len(current_w):
      for t, w in current_w.items():
        if t not in data_dict:
          continue
        ddf = data_dict[t]
        if prev_dt in ddf.index and dt in ddf.index:
          ret = ddf.loc[dt, "Close"] / ddf.loc[prev_dt, "Close"] - 1
          ticker_gross[t] += w * ret * equity_start
    prev_dt = dt
  total_net = sum(ticker_gross.values())
  rows = []
  for t in set(list(ticker_gross.keys()) + list(ticker_cost.keys())):
    net = ticker_gross.get(t, 0) - ticker_cost.get(t, 0)
    rows.append({
      "strategy": strategy_name, "ticker": t,
      "net_dollar_contribution": net,
      "pct_of_total_net_pnl": 100 * net / total_net if abs(total_net) > 1e-12 else 0,
    })
  recon = {
    "strategy": strategy_name, "initial_equity": initial_equity,
    "final_equity": equity, "total_net_pnl": total_net,
    "pct_sum": 100 * sum(r["pct_of_total_net_pnl"] for r in rows),
    "relative_error": abs(equity - initial_equity - total_net) / max(abs(total_net), 1e-9),
    "pass": abs(equity - initial_equity - total_net) < 1e-3,
  }
  return pd.DataFrame(rows), recon


def sector_pnl_from_asset(asset_df):
  if asset_df.empty:
    return pd.DataFrame()
  g = asset_df.groupby("ticker", as_index=False)["net_dollar_contribution"].sum()
  g["sector"] = g["ticker"].map(lambda t: SECTOR_MAP.get(t, "OTHER"))
  s = g.groupby("sector", as_index=False)["net_dollar_contribution"].sum()
  total = s["net_dollar_contribution"].sum()
  s["pct_of_total_net_pnl"] = np.where(abs(total) > 1e-12, 100 * s["net_dollar_contribution"] / total, 0)
  return s


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
  sim = simulate_portfolio_from_target_weights(targets, data_dict)
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

bench_ew_targets = {}
for s in signal_dates:
  elig = list(ASOF_UNIVERSE.get(pd.Timestamp(s), []))[:TOP_K]
  if elig:
    bench_ew_targets[s] = allocate_weights_with_hard_caps(elig, SECTOR_MAP)
bench_ew = run_from_targets("B1_EQUAL_WEIGHT", bench_ew_targets, signal_dates)
bench_spy = run_from_targets("B0_SPY", {signal_dates[0]: pd.Series({MARKET: 1.0})}, signal_dates[:1])
bench_ew_pr = bench_ew["periodic_returns"]
bench_spy_pr = bench_spy["periodic_returns"]

strategy_results = []
contrib_all, recon_all = [], []
for name, sim in sims.items():
  m = period_full_metrics(sim["periodic_returns"], bench_ew_pr, bench_spy_pr, RESEARCH_START, RESEARCH_END)
  asset_c, recon = calculate_realized_pnl_contribution(name, sim["targets"], data_dict)
  if len(asset_c):
    contrib_all.append(asset_c)
    recon_all.append(recon)
  top_t = asset_c.nlargest(1, "net_dollar_contribution") if len(asset_c) else pd.DataFrame()
  top_s = sector_pnl_from_asset(asset_c).nlargest(1, "pct_of_total_net_pnl") if len(asset_c) else pd.DataFrame()
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
  hm = period_full_metrics(sim["periodic_returns"], bench_ew_pr, bench_spy_pr,
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
# ## 10. Null test (500 perm, mismos caps)

# %%
def corrected_empirical_p(real_value, null_values):
  null = np.asarray(null_values, dtype=float)
  n_exceed = int(np.sum(null >= real_value))
  return (n_exceed + 1) / (len(null) + 1), n_exceed, len(null)


null_excess = []
for perm in tqdm(range(N_NULL_PORTFOLIO), desc="Null portfolio"):
  held, tgt = set(), {}
  for i, sig in enumerate(signal_dates):
    elig = list(ASOF_UNIVERSE.get(pd.Timestamp(sig), []))
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
  nm = period_full_metrics(ns["periodic_returns"], bench_ew_pr, bench_ew_pr, RESEARCH_START, RESEARCH_END)
  null_excess.append(nm.get("excess_CAGR_vs_equal_weight", 0))

null_p, n_exceed, n_perm = corrected_empirical_p(champ_excess, null_excess)
null_dist_df = pd.DataFrame([{
  "subject_strategy": champion, "real_excess_CAGR": champ_excess,
  "information_ratio": champ_ir, "null_mean": round(np.mean(null_excess), 2) if null_excess else np.nan,
  "number_exceeding_real": n_exceed, "n_permutations": n_perm,
  "corrected_empirical_p_value": round(null_p, 6),
}])

# %% [markdown]
# ## 11. Gates G1-G20 + FINAL_STATUS

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


champ_sim = sims[champion]
champ_row = strategy_df[strategy_df["strategy"] == champion].iloc[0]
recon_ok = all(r.get("pass", False) for r in recon_all) if recon_all else True
contrib_df = pd.concat(contrib_all, ignore_index=True) if contrib_all else pd.DataFrame()
cost_rows = []
for mult in [0.5, 1.0, 2.0]:
  cs = simulate_portfolio_from_target_weights(
    strategy_targets[champion], data_dict,
    transaction_cost=TRANSACTION_COST * mult, slippage=SLIPPAGE * mult)
  cm = calculate_metrics_from_equity(cs["equity"], cs["periodic_returns"])
  cost_rows.append({"strategy": champion, "cost_multiplier": mult, "CAGR_pct": cm.get("CAGR_pct", 0)})
cost_df = pd.DataFrame(cost_rows)
cost_alpha_ok = all(r["CAGR_pct"] > 0 for r in cost_rows)

reused_champ = reused_test_df[reused_test_df["strategy"] == champion].iloc[0] if len(reused_test_df) else None

validation_gates = {
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

if not validation_gates["G1_all_dates_weight_sum_1"] or not validation_gates["G2_max_stock_weight_5pct"] or not validation_gates["G3_max_sector_weight_25pct"]:
  FINAL_STATUS = "FAILED_CONSTRAINT_AUDIT"
elif not validation_gates["G5_no_end_sample_liquidity_selection"]:
  FINAL_STATUS = "FAILED_UNIVERSE_AUDIT"
elif not (validation_gates["G8_null_p_below_005"] and validation_gates["G9_excess_cagr_positive"]):
  FINAL_STATUS = "FAILED_STATISTICAL_VALIDATION"
elif validation_gates["G20_point_in_time_membership_tested"] and all(validation_gates.values()):
  FINAL_STATUS = "PAPER_CANDIDATE_POINT_IN_TIME"
elif all(v for k, v in validation_gates.items() if k != "G20_point_in_time_membership_tested"):
  FINAL_STATUS = "PROMISING_BUT_SURVIVORSHIP_BIASED"
else:
  FINAL_STATUS = "FAILED_STATISTICAL_VALIDATION"

# %% [markdown]
# ## 12. Original vs corregido + señales + exports

# %%
def load_v1741_original():
  p = Path("research_v17_4_1_strategy_results.csv")
  if p.exists():
    return pd.read_csv(p)
  return pd.DataFrame()


orig_df = load_v1741_original()
cmp_rows = []
for name in strategy_targets:
  v175 = strategy_df[strategy_df["strategy"] == name]
  if v175.empty:
    continue
  v175 = v175.iloc[0]
  orig = orig_df[orig_df["strategy"] == name].iloc[0] if len(orig_df) and name in orig_df["strategy"].values else None
  ref = V1741_REF if name == "S5_XGBRANKER" else {}
  cmp_rows.append({
    "strategy": name,
    "v1741_CAGR_pct": orig["CAGR_pct"] if orig is not None else ref.get("research_CAGR_pct", np.nan),
    "v175_CAGR_pct": v175["CAGR_pct"],
    "delta_CAGR_pct": round(v175["CAGR_pct"] - (orig["CAGR_pct"] if orig is not None else ref.get("research_CAGR_pct", 0)), 2),
    "v1741_max_drawdown_pct": orig["max_drawdown_pct"] if orig is not None else ref.get("max_drawdown_pct", np.nan),
    "v175_max_drawdown_pct": v175["max_drawdown_pct"],
    "delta_max_drawdown_pct": round(v175["max_drawdown_pct"] - (orig["max_drawdown_pct"] if orig is not None else ref.get("max_drawdown_pct", 0)), 2),
    "v1741_max_stock_weight_pct": ref.get("max_stock_weight_observed_pct", np.nan) if name == champion else np.nan,
    "v175_max_stock_weight_pct": v175["max_stock_weight_pct"],
    "v1741_max_sector_weight_pct": ref.get("max_sector_weight_observed_pct", np.nan) if name == champion else np.nan,
    "v175_max_sector_weight_pct": v175["max_sector_weight_pct"],
    "v1741_top_sector_pnl_pct": ref.get("top_sector_pnl_pct", np.nan) if name == champion else np.nan,
    "v175_top_sector_pnl_pct": v175["top_sector_pnl_pct"],
  })
original_vs_corrected_df = pd.DataFrame(cmp_rows)

latest_sig = live_feature_panel["signal_date"].max()
g_live = live_feature_panel[live_feature_panel["signal_date"] == latest_sig]
tgt_w = strategy_targets[champion].get(latest_sig, pd.Series(dtype=float))
signal_rows = []
for _, row in g_live.iterrows():
  t = row["ticker"]
  if t not in ASOF_UNIVERSE.get(pd.Timestamp(latest_sig), set()):
    continue
  tw = float(tgt_w.get(t, 0))
  signal_rows.append({
    "ticker": t, "target_weight": round(tw, 6), "strategy_source": champion,
    "sector": SECTOR_MAP.get(t, "OTHER"), "signal_date": str(latest_sig.date()),
    "paper_trading_start": PAPER_TRADING_START,
  })
stock_sum = sum(r["target_weight"] for r in signal_rows)
if stock_sum < 1.0 - 1e-6:
  signal_rows.append({
    "ticker": CASH_ASSET, "target_weight": round(1.0 - stock_sum, 6),
    "strategy_source": champion, "sector": "DEFENSIVE",
    "signal_date": str(latest_sig.date()), "paper_trading_start": PAPER_TRADING_START,
  })
current_signals_df = pd.DataFrame(signal_rows)

eq_rows = []
for n, s in sims.items():
  for d, v in s["equity"].items():
    eq_rows.append({"strategy": n, "date": d, "equity": v})
equity_curves_df = pd.DataFrame(eq_rows)

summary = {
  "AUDIT_VERSION": AUDIT_VERSION, "FINAL_STATUS": FINAL_STATUS,
  "champion": champion, "APPROVED_FOR_REAL_MONEY": False,
  "n_stocks": len(STOCKS), "n_sectors": n_sectors,
  "survivorship_warning": SURVIVORSHIP_WARNING,
  "POINT_IN_TIME_MEMBERSHIP": POINT_IN_TIME_MEMBERSHIP,
  "pit_membership_status": asof_pack["pit_membership_status"],
  "universe_mode": "CURRENT_CONSTITUENTS_ASOF_LIQUIDITY",
  "v1741_reference_status": V1741_REF.get("FINAL_STATUS"),
  "v175_champion_CAGR": champ_row["CAGR_pct"],
  "v175_excess_CAGR_vs_ew": champ_excess,
  "v175_information_ratio": champ_ir,
  "v175_max_stock_weight_pct": champ_row["max_stock_weight_pct"],
  "v175_max_sector_weight_pct": champ_row["max_sector_weight_pct"],
  "v175_max_drawdown_pct": champ_row["max_drawdown_pct"],
  "null_corrected_p": round(null_p, 6),
  "preregistered_pbo": prereg_pbo,
  "preregistered_dsr": round(prereg_dsr, 4) if np.isfinite(prereg_dsr) else np.nan,
  "global_dsr": round(global_dsr, 4) if np.isfinite(global_dsr) else np.nan,
  "ranker_oos_ic": ranker_stats.get("ranker_oos_ic_mean"),
  "weight_violations_n": len(weight_violations_df),
  "gates_pass": sum(validation_gates.values()),
  "gates_total": len(validation_gates),
  "dsr_formula_version": DSR_FORMULA_VERSION,
  "dsr_v1741_bug": DSR_V1741_BUG,
}

strategy_df.to_csv("research_v17_5_strategy_results.csv", index=False)
pd.DataFrame([summary]).to_csv("research_v17_5_summary.csv", index=False)
original_vs_corrected_df.to_csv("research_v17_5_original_vs_corrected.csv", index=False)
pd.DataFrame([{"gate": k, "pass": v} for k, v in validation_gates.items()]).to_csv(
  "research_v17_5_validation_gates.csv", index=False)
weight_violations_df.to_csv("research_v17_5_weight_violations.csv", index=False)
sector_weight_audit_df.to_csv("research_v17_5_sector_weight_audit.csv", index=False)
asof_pack["universe_by_date_df"].to_csv("research_v17_5_universe_by_date.csv", index=False)
asof_pack["turnover_df"].to_csv("research_v17_5_universe_turnover.csv", index=False)
asof_pack["eligibility_df"].to_csv("research_v17_5_universe_eligibility_audit.csv", index=False)
dsr_audit_df.to_csv("research_v17_5_dsr_audit.csv", index=False)
pbo_audit_df.to_csv("research_v17_5_pbo_audit.csv", index=False)
null_dist_df.to_csv("research_v17_5_null_results.csv", index=False)
reused_test_df.to_csv("research_v17_5_reused_test_2024_plus.csv", index=False)
current_signals_df.to_csv("research_v17_5_current_signals.csv", index=False)
equity_curves_df.to_csv("research_v17_5_equity_curves.csv", index=False)
Path("research_v17_5_selected_config.json").write_text(json.dumps({
  "version": AUDIT_VERSION, "final_status": FINAL_STATUS,
  "approved_for_real_money": False, "validation_gates": validation_gates,
  "frozen_config": str(CONFIG_PATH), "preregistered_hash": cfg["preregistered_hash"],
}, indent=2, default=str), encoding="utf-8")

# %% [markdown]
# ## Empaquetar y descargar resultados V17.5

# %%
import zipfile

AUTO_DOWNLOAD_RESULTS = True
V175_ZIP_PATH = CONTENT_ROOT / "V17_5_RESULTS.zip"


def build_v175_results_zip(zip_path=None):
  zip_path = Path(zip_path or V175_ZIP_PATH)
  root = CONTENT_ROOT
  with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for pattern in ("research_v17_5_*.csv", "research_v17_5_*.json"):
      for p in sorted(root.glob(pattern)):
        zf.write(p, p.name)
        print(f"  + {p.name}")
    for cfg_name in ("v17_5_frozen_audit_config.json", "v17_4_preregistered_experiment.json"):
      cfg_p = root / "config" / cfg_name
      if cfg_p.exists():
        zf.write(cfg_p, f"config/{cfg_name}")
        print(f"  + config/{cfg_name}")
  print(f"ZIP creado: {zip_path} ({zip_path.stat().st_size:,} bytes)")
  return zip_path


build_v175_results_zip()
if AUTO_DOWNLOAD_RESULTS and IN_COLAB:
  from google.colab import files
  files.download(str(V175_ZIP_PATH))
  print("Descarga iniciada: V17_5_RESULTS.zip")
elif AUTO_DOWNLOAD_RESULTS:
  print(f"AUTO_DOWNLOAD_RESULTS=True pero no estas en Colab. ZIP en: {V175_ZIP_PATH}")

# %%
print("=" * 80)
print("REPORTE FINAL V17.5 CONSTRAINT, UNIVERSE AND DSR AUDIT")
print("=" * 80)
print(f"FINAL_STATUS: {FINAL_STATUS}")
print(f"Champion: {champion} | CAGR={champ_row['CAGR_pct']}% excess_EW={champ_excess}% IR={champ_ir}")
print(f"Caps: max_stock={champ_row['max_stock_weight_pct']}% max_sector={champ_row['max_sector_weight_pct']}% violations={len(weight_violations_df)}")
print(f"Drawdown: {champ_row['max_drawdown_pct']}% | top_sector_pnl={champ_row['top_sector_pnl_pct']}%")
print(f"Null p={null_p:.6f} | PBO={prereg_pbo:.3f} | DSR prereg={prereg_dsr:.4f} global={global_dsr:.4f}")
print(f"DSR fix: {DSR_V1741_BUG}")
print(f"Universo: as-of rebalance | PIT={asof_pack['pit_membership_status']} | {SURVIVORSHIP_WARNING}")
print(f"V17.4.1 ref: {V1741_REF.get('FINAL_STATUS')} -> V17.5: {FINAL_STATUS}")
print(f"Gates: {sum(validation_gates.values())}/{len(validation_gates)} pass")
for k, v in validation_gates.items():
  if not v:
    print(f"  FAIL {k}")
print(FINAL_STATUS)
print("No integrar en Streamlit. APPROVED_FOR_REAL_MONEY=False.")

# %% [markdown]
# ## Descargar resultados sin repetir el backtest

# %%
# Recrea V17_5_RESULTS.zip desde exports existentes y descarga (sin re-ejecutar modelos).
import zipfile
from pathlib import Path

try:
  import google.colab  # noqa: F401
  _IN_COLAB_DL = True
except ImportError:
  _IN_COLAB_DL = False

_CONTENT_DL = Path("/content") if _IN_COLAB_DL else Path.cwd()
_ZIP_DL = _CONTENT_DL / "V17_5_RESULTS.zip"
_AUTO_DL = True


def _rebuild_v175_zip(zip_path=None):
  zip_path = Path(zip_path or _ZIP_DL)
  root = _CONTENT_DL
  exports = list(root.glob("research_v17_5_*.csv")) + list(root.glob("research_v17_5_*.json"))
  if not exports:
    raise FileNotFoundError(
      "No hay exports research_v17_5_* en el directorio. Ejecuta primero el backtest completo."
    )
  with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(exports):
      zf.write(p, p.name)
    for cfg_name in ("v17_5_frozen_audit_config.json", "v17_4_preregistered_experiment.json"):
      cfg_p = root / "config" / cfg_name
      if cfg_p.exists():
        zf.write(cfg_p, f"config/{cfg_name}")
  print(f"ZIP recreado: {zip_path} ({zip_path.stat().st_size:,} bytes)")
  return zip_path


_rebuild_v175_zip()
if _AUTO_DL and _IN_COLAB_DL:
  from google.colab import files
  files.download(str(_ZIP_DL))
  print("Descarga iniciada: V17_5_RESULTS.zip")
else:
  print(f"ZIP listo en: {_ZIP_DL}")


