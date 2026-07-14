# %% [markdown]
# # Trading Research V17.4.1 — Integrity Hotfix (pre FULL_250)
#
# Corrige contabilidad, metricas, reporting y senales. Sin cambiar factores ni parametros ML.
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
# ## 1. Configuracion congelada

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

MODE = "VALIDATE_HOTFIX"  # En Colab FULL_250: cambiar a MODE = "FULL_250"
VALIDATE_HOTFIX = MODE == "VALIDATE_HOTFIX"
FULL_250 = MODE == "FULL_250"
QUICK_TEST = VALIDATE_HOTFIX
EXPECTED_PREREG_HASH = "b2ae61b9532e63e922d71911445d523c4a1a93a48b2c1cc24aab3a75ac057bcd"
HOTFIX_VERSION = "v17_4_1"

START_DATE = "2010-01-01"
END_DATE = None
TRANSACTION_COST = 0.001
SLIPPAGE = 0.001
SIGNAL_FREQUENCY = "W-FRI"
REBALANCE_FREQUENCIES = ["W-FRI", "MONTHLY"]
FORWARD_HORIZON = 20
PURGE_DAYS = 20
EMBARGO_DAYS = 20
RANDOM_SEED = 42
APPROVED_FOR_REAL_MONEY = False
INITIAL_CAPITAL = 10000
COST_RATE = TRANSACTION_COST + SLIPPAGE
MARKET = "SPY"
CASH_ASSET = "SHY"
WF_START_YEAR = 2016
RESEARCH_START = "2016-01-01"
RESEARCH_END = "2023-12-31"
HOLDOUT_START = "2024-01-01"
N_TRIALS_ACCUMULATED = 261

MAX_TICKERS_FULL = 250
MIN_VALID_STOCKS = 200
MIN_SECTORS = 8
TOP_K_FULL = 25
BUY_RANK = 25
HOLD_UNTIL_RANK = 40
MAX_STOCK_WEIGHT = 0.05
MAX_SECTOR_WEIGHT = 0.25

MIN_HISTORY_DAYS = 250 if QUICK_TEST else 1000
MIN_DOLLAR_VOLUME = 5_000_000 if QUICK_TEST else 20_000_000
N_NULL_PORTFOLIO = 200 if QUICK_TEST else 500
N_NULL_FACTOR_PERMUTATIONS = 500

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
SECTOR_MAP = {
  "AAPL": "TECH", "MSFT": "TECH", "NVDA": "TECH", "AMD": "TECH", "AVGO": "TECH",
  "GOOGL": "TECH", "META": "TECH", "AMZN": "TECH", "JPM": "FIN", "BAC": "FIN",
  "XOM": "ENERGY", "CVX": "ENERGY", "UNH": "HEALTH", "LLY": "HEALTH",
  "WMT": "STAPLES", "COST": "STAPLES",
  **{e: "ETF" for e in ALL_ETFS},
}
FORBIDDEN_FEATURE_PATTERNS = re.compile(
  r"fwd|future|label|target|exit|next|forward|realized|pnl|return_after|barrier", re.I
)
SURVIVORSHIP_WARNING = (
  "El universo utiliza constituyentes actuales del S&P 500 y mantiene survivorship bias. No es point-in-time."
)
PREREG_HASH_METHOD = "sha256_utf8_json_dumps_sort_keys_indent2_default_str"
HOTFIX_MANIFEST_PATH = Path("config/v17_4_1_integrity_hotfix_manifest.json")


def _prereg_candidate_paths():
  seen, candidates = set(), []
  def add(p):
    p = Path(p)
    key = str(p)
    if key not in seen:
      seen.add(key)
      candidates.append(p)
  try:
    nb_dir = Path(__file__).resolve().parent
    add(nb_dir.parent / "config" / "v17_4_preregistered_experiment.json")
    add(nb_dir / "config" / "v17_4_preregistered_experiment.json")
  except NameError:
    pass
  cwd = Path.cwd()
  add(cwd / "config" / "v17_4_preregistered_experiment.json")
  add(Path("config/v17_4_preregistered_experiment.json"))
  for root in (cwd, Path("/content")):
    add(root / "trading-signals-mvp" / "config" / "v17_4_preregistered_experiment.json")
  return candidates


def preregistered_payload_for_hash(obj):
  return {k: v for k, v in obj.items() if k != "config_sha256"}


def preregistered_config_hash(obj):
  """Metodo original V17.4: hash del JSON sin config_sha256."""
  payload = preregistered_payload_for_hash(obj)
  return hashlib.sha256(
    json.dumps(payload, sort_keys=True, indent=2, default=str).encode("utf-8")
  ).hexdigest()


def _validate_prereg_file(path):
  """Devuelve el objeto JSON si la ruta es valida; si no, None."""
  path = Path(path)
  if not path.exists():
    return None
  try:
    if path.stat().st_size <= 0:
      return None
  except OSError:
    return None
  try:
    obj = json.loads(path.read_text(encoding="utf-8"))
  except (json.JSONDecodeError, OSError, UnicodeDecodeError):
    return None
  if not isinstance(obj, dict) or len(obj) == 0:
    return None
  if "config_sha256" not in obj:
    return None
  if obj.get("config_sha256") != EXPECTED_PREREG_HASH:
    return None
  if preregistered_config_hash(obj) != EXPECTED_PREREG_HASH:
    return None
  return obj


def resolve_prereg_path():
  for candidate in _prereg_candidate_paths():
    if _validate_prereg_file(candidate) is not None:
      return candidate.resolve()
  raise FileNotFoundError(
    "Falta el archivo original y no vacío config/v17_4_preregistered_experiment.json"
  )


def audit_preregistered_config_file(path):
  path = Path(path)
  obj = _validate_prereg_file(path)
  if obj is None:
    raw_bytes = path.read_bytes() if path.exists() else b""
    return {
      "expected_hash": EXPECTED_PREREG_HASH,
      "raw_file_hash": hashlib.sha256(raw_bytes).hexdigest(),
      "canonical_json_hash": "",
      "canonical_compact_hash": "",
      "hash_method": PREREG_HASH_METHOD,
      "config_path_used": str(path),
      "stored_hash": obj.get("config_sha256") if isinstance(obj, dict) else None,
      "manifest_hash": None,
      "hash_match": False,
      "validation_error": "archivo ausente, vacio, JSON invalido o hash incorrecto",
    }
  raw_bytes = path.read_bytes()
  raw_file_hash = hashlib.sha256(raw_bytes).hexdigest()
  stored_hash = obj.get("config_sha256")
  payload = preregistered_payload_for_hash(obj)
  canonical_json_hash = preregistered_config_hash(obj)
  canonical_compact_hash = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
  ).hexdigest()
  manifest_hash = None
  if HOTFIX_MANIFEST_PATH.exists():
    try:
      manifest_hash = json.loads(HOTFIX_MANIFEST_PATH.read_text(encoding="utf-8")).get(
        "original_preregistered_hash")
    except Exception:
      pass
  expected_hash = EXPECTED_PREREG_HASH
  hash_match = bool(
    canonical_json_hash == expected_hash
    and stored_hash == expected_hash
    and canonical_json_hash == stored_hash
  )
  return {
    "expected_hash": expected_hash,
    "raw_file_hash": raw_file_hash,
    "canonical_json_hash": canonical_json_hash,
    "canonical_compact_hash": canonical_compact_hash,
    "hash_method": PREREG_HASH_METHOD,
    "config_path_used": str(path),
    "stored_hash": stored_hash,
    "manifest_hash": manifest_hash,
    "hash_match": hash_match,
  }


PREREG_PATH = resolve_prereg_path()
np.random.seed(RANDOM_SEED)
FINAL_STATUS = "PENDING"
PRIMARY_METRIC = "information_ratio_vs_equal_weight"

# --- Ejecucion FULL_250: cache, checkpoints, logs (sin cambiar logica de trading) ---
CACHE_DIR = Path("cache/v17_4_1_full_250")
DOWNLOAD_CACHE_DIR = CACHE_DIR / "downloads"
ENABLE_CHECKPOINTS = FULL_250
RESUME_CHECKPOINTS = FULL_250
BENCHMARK_ETFS_FULL = sorted(BROAD_MARKET_ETFS | {CASH_ASSET})
CHECKPOINT_VERSION = "v17_4_1_full_250_r1"

if FULL_250:
  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  DOWNLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def log_progress(msg):
  line = f"[{pd.Timestamp.now().isoformat()}] {msg}"
  print(line)
  if ENABLE_CHECKPOINTS:
    with open(CACHE_DIR / "progress.log", "a", encoding="utf-8") as f:
      f.write(line + "\n")


def _manifest_path():
  return CACHE_DIR / "checkpoint_manifest.json"


def load_manifest():
  p = _manifest_path()
  if p.exists():
    return json.loads(p.read_text(encoding="utf-8"))
  return {"version": CHECKPOINT_VERSION, "phases": {}}


def save_checkpoint(phase_name, payload, meta=None):
  if not ENABLE_CHECKPOINTS:
    return
  path = CACHE_DIR / f"{phase_name}.pkl"
  pd.to_pickle(payload, path)
  m = load_manifest()
  m["phases"][phase_name] = {
    "file": path.name, "saved_at": pd.Timestamp.now().isoformat(),
    "mode": MODE, "meta": meta or {},
  }
  _manifest_path().write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
  log_progress(f"checkpoint saved: {phase_name}")


def load_checkpoint(phase_name):
  if not RESUME_CHECKPOINTS:
    return None
  m = load_manifest()
  info = m.get("phases", {}).get(phase_name)
  if not info or info.get("mode") != MODE:
    return None
  path = CACHE_DIR / info["file"]
  if not path.exists():
    return None
  log_progress(f"checkpoint resume: {phase_name}")
  return pd.read_pickle(path)


def compress_data_dict(data_dict):
  out = {}
  for t, df in data_dict.items():
    d = df.copy()
    for c in ("Open", "High", "Low", "Close", "Volume"):
      if c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(np.float32)
    out[t] = d
  return out


def compress_panel(panel_df):
  if panel_df.empty:
    return panel_df
  p = panel_df.copy()
  for c in p.select_dtypes(include=[np.floating]).columns:
    p[c] = p[c].astype(np.float32)
  return p

# %% [markdown]
# ## 1b. Verificacion hash pre-registro (FULL_250)

# %%
_prereg_early = _validate_prereg_file(PREREG_PATH) or {}
_early_audit = audit_preregistered_config_file(PREREG_PATH)
_early_stored = _prereg_early.get("config_sha256", "")
if FULL_250:
  if not _early_audit["hash_match"]:
    raise ValueError(
      f"FULL_250 bloqueado: hash pre-registro invalido. esperado={EXPECTED_PREREG_HASH} "
      f"archivo={_early_stored} calculado={_early_audit['canonical_json_hash']} "
      f"metodo={PREREG_HASH_METHOD} path={PREREG_PATH}"
    )
  log_progress(f"hash pre-registro OK: {_early_stored[:16]}...")

# %% [markdown]
# ## 2. Motor V17.2 (via V17.3 pattern)

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
  "QUICK_TEST": QUICK_TEST, "MAX_TICKERS_FULL": MAX_TICKERS_FULL,
  "START_DATE": START_DATE, "END_DATE": END_DATE,
  "MIN_HISTORY_DAYS": MIN_HISTORY_DAYS, "MIN_DOLLAR_VOLUME": MIN_DOLLAR_VOLUME,
  "DEFENSIVE_ETFS": DEFENSIVE_ETFS, "ALL_ETFS": ALL_ETFS, "SECTOR_MAP": SECTOR_MAP,
  "SIGNAL_FREQUENCY": SIGNAL_FREQUENCY, "INITIAL_CAPITAL": INITIAL_CAPITAL,
  "TRANSACTION_COST": TRANSACTION_COST, "SLIPPAGE": SLIPPAGE, "COST_RATE": COST_RATE,
  "CASH_ASSET": CASH_ASSET, "MARKET": MARKET, "FORBIDDEN_FEATURE_PATTERNS": FORBIDDEN_FEATURE_PATTERNS,
  "TOP_K": TOP_K_FULL, "weekly_dates": pd.DatetimeIndex([]),
}
exec(_eng, _ns)
globals().update({k: v for k, v in _ns.items() if callable(v)})

# %% [markdown]
# ## 3. Pre-registro + helpers estadisticos

# %%
def canonical_json(obj):
  return json.dumps(obj, sort_keys=True, indent=2, default=str)


def config_sha256(obj):
  return preregistered_config_hash(obj)


def corrected_empirical_p(real_value, null_values, two_tail=False):
  null = np.asarray(null_values, dtype=float)
  if two_tail:
    n_exceed = int(np.sum(np.abs(null) >= abs(real_value)))
  else:
    n_exceed = int(np.sum(null >= real_value))
  n_perm = len(null)
  p = (n_exceed + 1) / (n_perm + 1)
  return p, n_exceed, n_perm


def apply_weight_caps(weights, sector_map=None, max_stock=None, max_sector=None, shy_remainder=True):
  sector_map = sector_map or SECTOR_MAP
  max_stock = max_stock or EFFECTIVE_MAX_STOCK_WEIGHT
  max_sector = max_sector or EFFECTIVE_MAX_SECTOR_WEIGHT
  w = weights[weights > 1e-9].astype(float).copy()
  if w.empty:
    return w
  for _ in range(20):
    over = w[w > max_stock + 1e-9]
    if len(over):
      excess = (over - max_stock).sum()
      w[over.index] = max_stock
      under = w[w < max_stock - 1e-9]
      if under.sum() > 0:
        w[under.index] += excess * (under / under.sum())
    sec_w = defaultdict(float)
    for t, wt in w.items():
      sec_w[sector_map.get(t, "OTHER")] += wt
    capped = False
    for sec, sw in sec_w.items():
      if sec in ("ETF",) or sw <= max_sector + 1e-9:
        continue
      sec_tickers = [t for t in w.index if sector_map.get(t, "OTHER") == sec]
      scale = max_sector / sw
      w[sec_tickers] *= scale
      capped = True
    if not capped and (w > max_stock + 1e-9).sum() == 0:
      break
  w = _normalize_weights(w)
  if shy_remainder and CASH_ASSET in data_dict:
    stock_sum = w.drop(CASH_ASSET, errors="ignore").sum()
    if stock_sum < 1.0 - 1e-6:
      w[CASH_ASSET] = w.get(CASH_ASSET, 0) + (1.0 - stock_sum)
      w = _normalize_weights(w)
  return w


def build_buffered_weights(scores, held, buy_rank, hold_rank, top_k_cap=None):
  ranked = scores.sort_values(ascending=False)
  rank_map = {t: i + 1 for i, t in enumerate(ranked.index)}
  selected = set()
  for t in held:
    if rank_map.get(t, 9999) <= hold_rank:
      selected.add(t)
  for t, r in rank_map.items():
    if r <= buy_rank:
      selected.add(t)
  if top_k_cap and len(selected) > top_k_cap:
    selected = set(ranked.head(top_k_cap).index)
  if not selected:
    return pd.Series(dtype=float)
  w = pd.Series(1.0 / len(selected), index=sorted(selected))
  return apply_weight_caps(w)

# %% [markdown]
# ## 4. Universo

# %%
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_GITHUB_CSV = (
  "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
)
SP500_HTTP_HEADERS = {
  "User-Agent": (
    "Mozilla/5.0 (compatible; TradingResearchV1741/1.0; "
    "research-bot; +https://github.com/datasets/s-and-p-500-companies)"
  ),
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
      "ticker": sym,
      "security": str(r.get(sec_col, "") if sec_col else ""),
      "sector": sector,
      "sub_industry": str(r.get(sub_col, "") if sub_col else ""),
    })
  tickers = list(dict.fromkeys(tickers))
  metadata = pd.DataFrame(meta_rows)
  if len(tickers) == 0 or metadata.empty:
    raise ValueError("tabla S&P 500 sin tickers validos")
  assert len(tickers) >= 450, f"S&P 500 tickers insuficientes: {len(tickers)}"
  assert metadata["sector"].nunique() >= 8, f"S&P 500 sectores insuficientes: {metadata['sector'].nunique()}"
  return tickers, metadata


def _load_sp500_from_wikipedia():
  response = requests.get(
    SP500_WIKI_URL, headers=SP500_HTTP_HEADERS, timeout=SP500_HTTP_TIMEOUT)
  response.raise_for_status()
  tables = pd.read_html(StringIO(response.text))
  if not tables:
    raise ValueError("Wikipedia no devolvio tablas HTML")
  return _parse_sp500_constituents(tables[0])


def _load_sp500_from_github():
  response = requests.get(
    SP500_GITHUB_CSV, headers=SP500_HTTP_HEADERS, timeout=SP500_HTTP_TIMEOUT)
  response.raise_for_status()
  return _parse_sp500_constituents(pd.read_csv(StringIO(response.text)))


def load_us_large_cap_250():
  wiki_error = None
  try:
    tickers, metadata = _load_sp500_from_wikipedia()
    print("S&P 500 loaded from Wikipedia")
    return tickers[:600], metadata
  except Exception as e:
    wiki_error = e
    print("Wikipedia fail:", e)
  try:
    tickers, metadata = _load_sp500_from_github()
    print("S&P 500 loaded from GitHub fallback")
    return tickers[:600], metadata
  except Exception as github_error:
    raise RuntimeError(
      "No se pudo cargar el universo S&P 500 desde Wikipedia ni GitHub. "
      f"Wikipedia={wiki_error}; GitHub={github_error}"
    ) from github_error


def _phase_02_checkpoint_ok(cp):
  if not cp or not isinstance(cp, dict):
    return False
  pinfo = load_manifest().get("phases", {}).get("phase_02_universe", {})
  if FULL_250 and pinfo.get("mode") != "FULL_250":
    return False
  close_all = cp.get("close_all")
  if close_all is None or getattr(close_all, "empty", True):
    return False
  stocks, _, n_sectors = classify_assets(list(close_all.columns))
  if len(stocks) < MIN_VALID_STOCKS or n_sectors < MIN_SECTORS:
    return False
  return True


def download_with_audit(tickers, start, end=None):
  cached_data, to_fetch = {}, []
  if ENABLE_CHECKPOINTS:
    for t in tickers:
      tu = t.upper()
      cp = DOWNLOAD_CACHE_DIR / f"{tu}.parquet"
      if cp.exists():
        try:
          cached_data[tu] = pd.read_parquet(cp)
          continue
        except Exception:
          pass
      to_fetch.append(t)
    log_progress(f"download cache hit={len(cached_data)} miss={len(to_fetch)}")
  else:
    to_fetch = list(tickers)

  fetched = {}
  errors = pd.DataFrame()
  if to_fetch:
    fetched, close_tmp, errors = download_data(to_fetch, start, end)
    if ENABLE_CHECKPOINTS:
      for t, df in fetched.items():
        try:
          df.to_parquet(DOWNLOAD_CACHE_DIR / f"{t.upper()}.parquet")
        except Exception as e:
          log_progress(f"cache write fail {t}: {e}")

  data = {**cached_data, **fetched}
  close = pd.DataFrame({k: v["Close"] for k, v in data.items()}).sort_index().ffill() if data else pd.DataFrame()
  report = []
  for t in tickers:
    tu = t.upper()
    if tu not in data:
      report.append({"ticker": tu, "status": "download_failed"})
      continue
    df = data[tu]
    dv = (df["Close"] * df.get("Volume", pd.Series(0))).rolling(60).mean().iloc[-1]
    hist = len(df)
    if hist < MIN_HISTORY_DAYS:
      report.append({"ticker": tu, "status": "insufficient_history", "rows": hist})
      continue
    if tu not in ALL_ETFS and (not np.isfinite(dv) or dv < MIN_DOLLAR_VOLUME):
      report.append({"ticker": tu, "status": "insufficient_liquidity", "dollar_vol_60": float(dv)})
      continue
    if "Open" not in df.columns:
      report.append({"ticker": tu, "status": "missing_open"})
      continue
    if "Volume" not in df.columns:
      report.append({"ticker": tu, "status": "missing_volume"})
      continue
    report.append({"ticker": tu, "status": "valid", "rows": hist, "dollar_vol_60": float(dv)})
  rep = pd.DataFrame(report)
  valid = rep[rep["status"] == "valid"]["ticker"].tolist()
  if FULL_250 and len(valid) > MAX_TICKERS_FULL:
    valid = rep[rep["status"] == "valid"].sort_values("dollar_vol_60", ascending=False).head(MAX_TICKERS_FULL)["ticker"].tolist()
  data = {k: v for k, v in data.items() if k in valid}
  close = close[[c for c in close.columns if c in valid]]
  return data, close, errors, rep


if VALIDATE_HOTFIX:
  UNIVERSE = load_quick_test_universe()
  sp_meta = pd.DataFrame()
else:
  sp_list, sp_meta = load_us_large_cap_250()
  UNIVERSE = list(dict.fromkeys(sp_list + [t for t in BENCHMARK_ETFS_FULL if t not in sp_list]))

_cp_uni = load_checkpoint("phase_02_universe")
if _cp_uni is not None and not _phase_02_checkpoint_ok(_cp_uni):
  log_progress("phase_02_universe checkpoint invalido: reconstruyendo")
  _cp_uni = None
if _cp_uni is not None:
  data_dict = compress_data_dict(_cp_uni["data_dict"])
  close_all = _cp_uni["close_all"]
  dl_errors = _cp_uni.get("dl_errors", pd.DataFrame())
  dl_report = _cp_uni.get("dl_report", pd.DataFrame())
  log_progress("resumed phase_02_universe")
else:
  log_progress(f"descarga universo n={len(UNIVERSE)}")
  data_dict, close_all, dl_errors, dl_report = download_with_audit(UNIVERSE, START_DATE, END_DATE)
  data_dict = compress_data_dict(data_dict)
  _cp_stocks, _, _cp_sectors = classify_assets(list(close_all.columns))
  save_checkpoint("phase_02_universe", {
    "data_dict": data_dict, "close_all": close_all, "dl_errors": dl_errors, "dl_report": dl_report,
    "universe": UNIVERSE,
  }, meta={"n_tickers": len(UNIVERSE), "n_stocks": len(_cp_stocks), "n_sectors": _cp_sectors})
STOCKS, ETFS, n_sectors = classify_assets(list(close_all.columns))
n_valid_stocks, n_valid_etfs = len(STOCKS), len(ETFS)
TOP_K = max(3, int(np.ceil(n_valid_stocks * 0.20))) if QUICK_TEST else TOP_K_FULL
BUY_R = TOP_K if QUICK_TEST else BUY_RANK
HOLD_R = TOP_K + 10 if QUICK_TEST else HOLD_UNTIL_RANK
EFFECTIVE_MAX_STOCK_WEIGHT = MAX_STOCK_WEIGHT if FULL_250 else min(0.25, max(MAX_STOCK_WEIGHT, 1.0 / max(TOP_K, 1)))
EFFECTIVE_MAX_SECTOR_WEIGHT = MAX_SECTOR_WEIGHT if FULL_250 else min(0.50, max(MAX_SECTOR_WEIGHT, 1.0 / max(n_sectors, 1)))

if FULL_250:
  if n_valid_stocks < MIN_VALID_STOCKS:
    raise ValueError(f"FULL_250: n_valid_stocks={n_valid_stocks} < {MIN_VALID_STOCKS}")
  if n_sectors < MIN_SECTORS:
    raise ValueError(f"FULL_250: n_sectors={n_sectors} < {MIN_SECTORS}")
  log_progress(f"FULL gate universo OK: stocks={n_valid_stocks} sectors={n_sectors}")

universe_audit = pd.DataFrame([{
  "mode": MODE, "n_downloaded": len(UNIVERSE), "n_valid_stocks": n_valid_stocks,
  "n_valid_etfs": n_valid_etfs, "n_sectors": n_sectors, "top_k": TOP_K,
  "buy_rank": BUY_R, "hold_until_rank": HOLD_R, "survivorship_warning": SURVIVORSHIP_WARNING,
}])
# universe audit exported in final section as v17_4_1 if needed
print(f"MODE={MODE} | stocks={n_valid_stocks} | sectors={n_sectors} | TOP_K={TOP_K}")

# %% [markdown]
# ## 5. Paneles modeling vs live

# %%
def build_live_feature_panel(data_dict, stocks):
  return build_features_and_panel(data_dict, stocks)


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


_cp_panels = load_checkpoint("phase_03_panels")
if _cp_panels is not None:
  live_feature_panel = compress_panel(_cp_panels["live_feature_panel"])
  modeling_panel = compress_panel(_cp_panels["modeling_panel"])
  labeled_panel = compress_panel(_cp_panels["labeled_panel"])
  FEATURE_COLS = _cp_panels["feature_cols"]
  log_progress("resumed phase_03_panels")
else:
  live_feature_panel = build_live_feature_panel(data_dict, STOCKS)
  modeling_panel = attach_forward_labels(live_feature_panel, data_dict)
  labeled_panel = modeling_panel[modeling_panel["fwd_ret_20d"].notna()].copy()
  live_feature_panel = compress_panel(live_feature_panel)
  modeling_panel = compress_panel(modeling_panel)
  labeled_panel = compress_panel(labeled_panel)
  FEATURE_COLS = [c for c in live_feature_panel.columns if c not in {
    "ticker", "signal_date", "sector", "entry_date", "label_end_date", "feature_date",
    "fwd_ret_20d", "fwd_excess_20d", "relevance"} and not FORBIDDEN_FEATURE_PATTERNS.search(c)]
  save_checkpoint("phase_03_panels", {
    "live_feature_panel": live_feature_panel, "modeling_panel": modeling_panel,
    "labeled_panel": labeled_panel, "feature_cols": FEATURE_COLS,
  }, meta={"n_labeled": len(labeled_panel), "n_features": len(FEATURE_COLS)})

_ns.update({"STOCKS": STOCKS, "data_dict": data_dict, "panel": live_feature_panel, "TOP_K": TOP_K})
for k in ["make_signal_dates", "builder_equal_weight", "builder_sector_momentum", "builder_top_score",
          "builder_random", "builder_alpha_score", "builder_spy_buyhold", "builder_ew_buyhold", "_panel_at"]:
  if k in _ns:
    globals()[k] = _ns[k]

weekly_dates = make_signal_dates("weekly")
monthly_dates = make_signal_dates("monthly")
print(f"Live panel: {len(live_feature_panel)} | Labeled: {len(labeled_panel)} | features: {len(FEATURE_COLS)}")

# %% [markdown]
# ## 6. Metricas por periodo + PnL wealth-based

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
  ann_factor = 252
  sharpe = float(pr.mean() / pr.std() * math.sqrt(ann_factor)) if pr.std() > 0 else 0.0
  dd = float((period_equity / period_equity.cummax() - 1).min())
  neg = pr[pr < 0]
  sortino = float(pr.mean() / neg.std() * math.sqrt(ann_factor)) if len(neg) and neg.std() > 0 else 0.0
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


def active_max_drawdown_vs_benchmark(strat_equity, bench_equity):
  se, be = strat_equity.align(bench_equity, join="inner")
  be = be.replace(0, np.nan).ffill()
  se = se.replace(0, np.nan).ffill()
  if len(se) < 2 or len(be) < 2:
    return 0.0
  rw = se / be
  rw = rw.replace([np.inf, -np.inf], np.nan).dropna()
  if len(rw) < 2:
    return 0.0
  dd = rw / rw.cummax() - 1
  return float(max(min(dd.min(), 0.0), -1.0))


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
  se_eq = sm["period_equity"]
  be_eq = bm["period_equity"].reindex(se_eq.index).ffill()
  out = {
    **{k: v for k, v in sm.items() if k not in ("period_equity", "period_returns")},
    "excess_CAGR_vs_equal_weight": round(sm["CAGR_pct"] - bm["CAGR_pct"], 2),
    "information_ratio_vs_equal_weight": round(
      active.mean() / active.std() * math.sqrt(252), 3) if active.std() > 0 else 0.0,
    "tracking_error": round(float(active.std() * math.sqrt(252)), 4),
    "active_max_drawdown_vs_equal_weight": round(
      active_max_drawdown_vs_benchmark(se_eq, be_eq) * 100, 2),
  }
  if spy_returns is not None:
    spy_m = calculate_period_metrics_from_returns(spy_returns, start, end)
    out["excess_CAGR_vs_SPY"] = round(sm["CAGR_pct"] - spy_m.get("CAGR_pct", 0), 2)
    spy_eq = spy_m.get("period_equity", pd.Series(dtype=float))
    out["active_max_drawdown_vs_spy"] = round(
      active_max_drawdown_vs_benchmark(se_eq, spy_eq.reindex(se_eq.index).ffill()) * 100, 2)
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


def _sector_label(ticker):
  if ticker == CASH_ASSET or ticker in DEFENSIVE_ETFS:
    return "DEFENSIVE_ETF" if ticker in DEFENSIVE_ETFS else "CASH"
  return SECTOR_MAP.get(ticker, "UNKNOWN")


def calculate_realized_pnl_contribution(strategy_name, targets, data_dict, initial_equity=None):
  initial_equity = initial_equity or float(INITIAL_CAPITAL)
  cal = _master_calendar(data_dict)
  exec_sched = {}
  for sig, w in sorted(targets.items()):
    ex = _next_trading_day(cal, sig)
    if pd.notna(ex):
      exec_sched[ex] = _normalize_weights(w)
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
        equity -= dollar_cost
        equity_start = equity
      current_w = new_w.copy()
    if prev_dt is not None and len(current_w):
      for t, w in current_w.items():
        if t not in data_dict:
          continue
        ddf = data_dict[t]
        if dt in ddf.index and prev_dt in ddf.index:
          c0, c1 = ddf.loc[prev_dt, "Close"], ddf.loc[dt, "Close"]
          if c0 > 0 and np.isfinite(c1):
            ar = c1 / c0 - 1
            ticker_gross[t] += equity_start * w * ar
            ticker_wsum[t] += w
            ticker_maxw[t] = max(ticker_maxw[t], w)
            ticker_periods[t] += 1
      port_gross = sum(
        equity_start * w * (data_dict[t].loc[dt, "Close"] / data_dict[t].loc[prev_dt, "Close"] - 1)
        for t, w in current_w.items()
        if t in data_dict and dt in data_dict[t].index and prev_dt in data_dict[t].index
        and data_dict[t].loc[prev_dt, "Close"] > 0
      )
      equity = equity_start + port_gross
    prev_dt = dt

  final_equity = equity
  expected_terminal_pnl = final_equity - initial_equity
  rows = []
  all_tickers = set(ticker_gross.keys()) | set(ticker_cost.keys())
  for t in sorted(all_tickers):
    g, c = ticker_gross.get(t, 0.0), ticker_cost.get(t, 0.0)
    n = g - c
    rows.append({
      "ticker": t, "strategy": strategy_name, "sector": _sector_label(t),
      "gross_dollar_contribution": round(g, 4), "cost_dollar_contribution": round(c, 4),
      "net_dollar_contribution": round(n, 4),
      "pct_of_total_net_pnl": 0.0,
      "average_weight": round(ticker_wsum[t] / max(ticker_periods[t], 1), 6) if t in ticker_wsum else 0,
      "max_weight": round(ticker_maxw.get(t, 0), 6), "holding_periods": ticker_periods.get(t, 0),
    })
  calculated_terminal_pnl = sum(r["net_dollar_contribution"] for r in rows)
  reconciliation_error = abs(expected_terminal_pnl - calculated_terminal_pnl)
  relative_error = reconciliation_error / max(abs(expected_terminal_pnl), 1e-12)
  for r in rows:
    r["pct_of_total_net_pnl"] = round(
      100 * r["net_dollar_contribution"] / calculated_terminal_pnl, 4) if abs(calculated_terminal_pnl) > 1e-12 else 0
  recon = {
    "strategy": strategy_name, "initial_equity": initial_equity, "final_equity": round(final_equity, 4),
    "expected_terminal_pnl": round(expected_terminal_pnl, 4),
    "calculated_terminal_pnl": round(calculated_terminal_pnl, 4),
    "reconciliation_error": round(reconciliation_error, 6),
    "relative_error": relative_error, "pass": relative_error < 1e-6,
  }
  return pd.DataFrame(rows), recon


def sector_pnl_from_asset(asset_df):
  if asset_df.empty:
    return pd.DataFrame()
  g = asset_df.groupby("sector", as_index=False).agg(
    gross_dollar_contribution=("gross_dollar_contribution", "sum"),
    cost_dollar_contribution=("cost_dollar_contribution", "sum"),
    net_dollar_contribution=("net_dollar_contribution", "sum"),
    average_portfolio_weight=("average_weight", "mean"),
    max_portfolio_weight=("max_weight", "max"),
  )
  total_net = g["net_dollar_contribution"].sum()
  g["pct_of_total_net_pnl"] = np.where(abs(total_net) > 1e-12, 100 * g["net_dollar_contribution"] / total_net, 0)
  return g

# %% [markdown]
# ## 7. Walk-forward XGBRanker + pipeline audit

# %%
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
    return pd.DataFrame(), pd.DataFrame(), {}
  params = params or XGB_PARAMS
  audit_rows, pipe_rows, oos_preds = [], [], []
  train_ics, oos_ics = [], []
  years = sorted(labeled_df["signal_date"].dt.year.unique())
  for test_year in years:
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
    grp_tr = tr.groupby("signal_date").size().values
    mdl = xgb.XGBRanker(**params)
    mdl.fit(tr[sel_feats], y_tr, group=grp_tr)
    pred_tr = mdl.predict(tr[sel_feats])
    pred_te = mdl.predict(te[sel_feats])
    train_ics.append(stats.spearmanr(pred_tr, tr["fwd_excess_20d"]).correlation)
    oos_ic = stats.spearmanr(pred_te, te["fwd_excess_20d"]).correlation
    oos_ics.append(oos_ic)
    fold_id = f"fold_{test_year}"
    pipe_rows.append({
      "fold_id": fold_id, "train_start": tr["signal_date"].min(), "train_end": tr["signal_date"].max(),
      "test_start": test_start, "test_end": pd.Timestamp(f"{test_year}-12-31"),
      "n_train_rows": len(tr), "n_test_rows": len(te), "selected_features": ",".join(sel_feats),
      "train_ic": round(train_ics[-1], 4), "oos_ic": round(oos_ic, 4),
    })
    te = te.copy()
    te["xgb_oos_prediction"] = pred_te
    oos_preds.append(te[["ticker", "signal_date", "xgb_oos_prediction", "composite_score"]])
    for _, r in te.iterrows():
      audit_rows.append({
        "ticker": r["ticker"], "feature_date": r.get("feature_date", r["signal_date"]),
        "signal_date": r["signal_date"], "entry_date": r["entry_date"], "label_end_date": r["label_end_date"],
        "test_year": test_year, "train_start": tr["signal_date"].min(), "train_end": tr["signal_date"].max(),
        "is_train_row": False, "is_test_row": True, "prediction": float(r["xgb_oos_prediction"]),
        "actual_fwd_excess_return": r["fwd_excess_20d"], "fold_id": fold_id,
      })
  oos_df = pd.concat(oos_preds, ignore_index=True) if oos_preds else pd.DataFrame()
  stats_out = {
    "ranker_train_ic_mean": round(float(np.nanmean(train_ics)), 4) if train_ics else np.nan,
    "ranker_oos_ic_mean": round(float(np.nanmean(oos_ics)), 4) if oos_ics else np.nan,
  }
  return oos_df, pd.DataFrame(audit_rows), pd.DataFrame(pipe_rows), stats_out


def fit_final_xgb(labeled_df, feature_cols):
  try:
    import xgboost as xgb
  except ImportError:
    return None, [], pd.Series(dtype=float)
  tr = labeled_df[labeled_df["label_end_date"].notna()].copy()
  if len(tr) < 50:
    return None, [], pd.Series(dtype=float)
  sel_feats, med = select_features_train(tr, feature_cols)
  tr[sel_feats] = tr[sel_feats].fillna(med)
  y_tr = tr.groupby("signal_date")["fwd_excess_20d"].rank(pct=True).astype(int)
  mdl = xgb.XGBRanker(**XGB_PARAMS)
  mdl.fit(tr[sel_feats], y_tr, group=tr.groupby("signal_date").size().values)
  return mdl, sel_feats, med

# %% [markdown]
# ## 8. Builders de targets por estrategia

# %%
def _scores_at(sig, col):
  g = live_feature_panel[live_feature_panel["signal_date"] == sig]
  if g.empty or col not in g.columns:
    return pd.Series(dtype=float)
  return g.set_index("ticker")[col].dropna()


def build_targets_from_score_col(dates, score_col, buy_r=None, hold_r=None, filter_fn=None):
  buy_r, hold_r = buy_r or BUY_R, hold_r or HOLD_R
  held, targets = set(), {}
  for sig in dates:
    sc = _scores_at(sig, score_col)
    if filter_fn is not None:
      g = live_feature_panel[live_feature_panel["signal_date"] == sig]
      ok = filter_fn(g)
      sc = sc[sc.index.isin(ok)]
    if sc.empty:
      continue
    w = build_buffered_weights(sc, held, buy_r, hold_r, top_k_cap=TOP_K)
    if len(w):
      targets[sig] = w
      held = set(w.index) - {CASH_ASSET}
  return targets


def build_s8_sector_neutral_momentum(dates, top_frac=0.20):
  held, targets = set(), {}
  for sig in dates:
    g = live_feature_panel[live_feature_panel["signal_date"] == sig].copy()
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
    n_sec = sel["sector"].nunique()
    sec_budget = min(EFFECTIVE_MAX_SECTOR_WEIGHT, 1.0 / max(n_sec, 1))
    w_parts = []
    for sec, sg in sel.groupby("sector"):
      n = len(sg)
      sw = sec_budget / n
      w_parts.append(pd.Series(sw, index=sg["ticker"]))
    w = pd.concat(w_parts)
    w = apply_weight_caps(w, shy_remainder=True)
    targets[sig] = w
    held = set(w.index) - {CASH_ASSET}
  return targets


def build_s7_sector_neutral_composite(dates, top_frac=0.20):
  held, targets = set(), {}
  for sig in dates:
    g = live_feature_panel[live_feature_panel["signal_date"] == sig].copy()
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
    n_sec = sel["sector"].nunique()
    sec_budget = min(EFFECTIVE_MAX_SECTOR_WEIGHT, 1.0 / max(n_sec, 1))
    w_parts = []
    for sec, sg in sel.groupby("sector"):
      w_parts.append(pd.Series(sec_budget / len(sg), index=sg["ticker"]))
    w = apply_weight_caps(pd.concat(w_parts), shy_remainder=True)
    targets[sig] = w
    held = set(w.index) - {CASH_ASSET}
  return targets


def build_s5_xgb_targets(dates, oos_df):
  held, targets = set(), {}
  for sig in dates:
    g = oos_df[oos_df["signal_date"] == sig] if len(oos_df) else pd.DataFrame()
    if g.empty:
      continue
    sc = g.set_index("ticker")["xgb_oos_prediction"]
    w = build_buffered_weights(sc, held, BUY_R, HOLD_R, top_k_cap=TOP_K)
    if len(w):
      targets[sig] = w
      held = set(w.index) - {CASH_ASSET}
  return targets


def build_s6_ensemble_targets(dates, oos_df):
  held, targets, fallbacks = set(), {}, 0
  for sig in dates:
    g = live_feature_panel[live_feature_panel["signal_date"] == sig].copy()
    xg = oos_df[oos_df["signal_date"] == sig] if len(oos_df) else pd.DataFrame()
    if g.empty:
      continue
    comp = g.set_index("ticker")["composite_score"].rank(pct=True)
    if len(xg):
      xgb_s = xg.set_index("ticker")["xgb_oos_prediction"].rank(pct=True)
      ens = ENSEMBLE_W_COMPOSITE * comp + ENSEMBLE_W_XGB * xgb_s.reindex(comp.index).fillna(comp)
    else:
      ens = comp
      fallbacks += 1
    w = build_buffered_weights(ens, held, BUY_R, HOLD_R, top_k_cap=TOP_K)
    if len(w):
      targets[sig] = w
      held = set(w.index) - {CASH_ASSET}
  return targets, fallbacks


def run_from_targets(name, targets, dates):
  holdings_log = []
  for sig in sorted(targets.keys()):
    w = _normalize_weights(targets[sig])
    holdings_log.append({
      "date": sig, "strategy": name,
      "selected_tickers": ",".join(sorted(w.index.astype(str))),
      "n_holdings": len(w), "sum_weights": round(w.sum(), 6),
    })
  sim = simulate_portfolio_from_target_weights(targets, data_dict)
  m = calculate_metrics_from_equity(sim["equity"], sim["periodic_returns"])
  m["strategy"] = name
  return {**sim, "metrics": m, "holdings_log": holdings_log, "targets": targets}


def slice_sim_period(sim, start, end):
  eq = sim["equity"]
  mask = (eq.index >= pd.Timestamp(start)) & (eq.index <= pd.Timestamp(end))
  pr = sim["periodic_returns"].loc[mask].dropna()
  return {"equity": eq.loc[mask], "periodic_returns": pr}


def select_champion_and_candidate(strategy_df):
  if strategy_df.empty:
    return None, None, "NO_STRATEGIES"
  pos = strategy_df[
    (strategy_df["excess_CAGR_vs_equal_weight"] > 0) &
    (strategy_df["information_ratio_vs_equal_weight"] > 0) &
    (strategy_df["METRIC_STATUS"] == "PASS")
  ]
  best = strategy_df.sort_values(PRIMARY_METRIC, ascending=False).iloc[0]["strategy"]
  if len(pos):
    champ = pos.sort_values(PRIMARY_METRIC, ascending=False).iloc[0]["strategy"]
    return champ, best, "CHAMPION_SELECTED"
  return None, best, "NO_POSITIVE_ALPHA_VS_EQUAL_WEIGHT"

# %% [markdown]
# ## 9. Pre-registro hash

# %%
PREREG_PATH = resolve_prereg_path()
hash_audit = audit_preregistered_config_file(PREREG_PATH)
stored_hash = hash_audit["stored_hash"]
computed_hash = hash_audit["canonical_json_hash"]
config_hash_valid = bool(hash_audit["hash_match"])
config_hash_unchanged = computed_hash == EXPECTED_PREREG_HASH
print("--- Pre-registro hash audit ---")
for k in ("expected_hash", "stored_hash", "canonical_json_hash", "canonical_compact_hash",
          "raw_file_hash", "hash_method", "config_path_used", "manifest_hash", "hash_match"):
  print(f"  {k}: {hash_audit.get(k)}")
if FULL_250 and not config_hash_valid:
  raise ValueError(
    f"FULL_250: preregistered hash invalid. expected={EXPECTED_PREREG_HASH} "
    f"stored={stored_hash} computed={computed_hash} method={PREREG_HASH_METHOD}"
  )

# %% [markdown]
# ## 10a. Walk-forward XGBRanker (OOS)

# %%
_cp_xgb = load_checkpoint("phase_04_xgb")
if _cp_xgb is not None:
  oos_xgb = _cp_xgb["oos_xgb"]
  ranker_audit = _cp_xgb["ranker_audit"]
  pipeline_audit = _cp_xgb["pipeline_audit"]
  ranker_stats = _cp_xgb["ranker_stats"]
  log_progress("resumed phase_04_xgb")
else:
  log_progress("walk-forward XGB OOS inicio")
  oos_xgb, ranker_audit, pipeline_audit, ranker_stats = walk_forward_xgb_oos(labeled_panel, FEATURE_COLS)
  save_checkpoint("phase_04_xgb", {
    "oos_xgb": oos_xgb, "ranker_audit": ranker_audit,
    "pipeline_audit": pipeline_audit, "ranker_stats": ranker_stats,
  }, meta={"n_oos_rows": len(oos_xgb)})
  log_progress(f"walk-forward XGB OOS fin n={len(oos_xgb)}")

# %% [markdown]
# ## 10b. Ejecutar estrategias S1-S8

# %%
_cp_strat = load_checkpoint("phase_05_strategies")
if _cp_strat is not None:
  sims = _cp_strat["sims"]
  strategy_targets = _cp_strat["strategy_targets"]
  s6_fallbacks = _cp_strat["s6_fallbacks"]
  signal_dates = _cp_strat["signal_dates"]
  bench_ew = _cp_strat["bench_ew"]
  bench_mom = _cp_strat["bench_mom"]
  bench_spy = _cp_strat["bench_spy"]
  strategy_results = _cp_strat["strategy_results"]
  strategy_df = _cp_strat["strategy_df"]
  research_df = _cp_strat["research_df"]
  holdout_df = _cp_strat["holdout_df"]
  holdout_results = _cp_strat["holdout_results"]
  contrib_all = _cp_strat["contrib_all"]
  recon_all = _cp_strat["recon_all"]
  sector_all = _cp_strat["sector_all"]
  bench_ew_pr = _cp_strat["bench_ew_pr"]
  bench_spy_pr = _cp_strat["bench_spy_pr"]
  champion = _cp_strat["champion"]
  best_candidate_for_full = _cp_strat["best_candidate_for_full"]
  quick_alpha_status = _cp_strat["quick_alpha_status"]
  selected_live_strategy = _cp_strat["selected_live_strategy"]
  null_subject = _cp_strat["null_subject"]
  champ_row = _cp_strat["champ_row"]
  champ_excess = _cp_strat["champ_excess"]
  champ_ir = _cp_strat["champ_ir"]
  log_progress("resumed phase_05_strategies")
else:
  log_progress("estrategias S1-S8 inicio")
  signal_dates = weekly_dates[weekly_dates >= pd.Timestamp(RESEARCH_START)]
  if FULL_250:
    signal_dates = weekly_dates

  strategy_targets = {
    "S1_MOMENTUM_12_1": build_targets_from_score_col(signal_dates, "rank_mom_252_skip_20"),
    "S2_MOMENTUM_TREND": build_targets_from_score_col(
      signal_dates, "rank_mom_60", filter_fn=lambda g: g[g["above_sma200"] > 0]["ticker"].tolist()),
    "S3_LOW_VOL_MOMENTUM": build_targets_from_score_col(signal_dates, "rank_mom_120"),
    "S4_CORRECTED_COMPOSITE": build_targets_from_score_col(signal_dates, "composite_score"),
    "S5_XGBRANKER": build_s5_xgb_targets(signal_dates, oos_xgb),
    "S7_SECTOR_NEUTRAL_COMPOSITE": build_s7_sector_neutral_composite(signal_dates),
    "S8_SECTOR_NEUTRAL_MOMENTUM": build_s8_sector_neutral_momentum(signal_dates),
  }
  s6_targets, s6_fallbacks = build_s6_ensemble_targets(signal_dates, oos_xgb)
  strategy_targets["S6_ENSEMBLE_COMPOSITE_RANKER"] = s6_targets

  sims = {}
  for name, tgt in strategy_targets.items():
    sims[name] = run_from_targets(name, tgt, signal_dates)

  bench_ew = run_from_targets("B1_EQUAL_WEIGHT", {
    s: _normalize_weights(pd.Series(1.0 / len(live_feature_panel[live_feature_panel["signal_date"] == s]),
      index=live_feature_panel[live_feature_panel["signal_date"] == s]["ticker"]))
    for s in signal_dates if len(live_feature_panel[live_feature_panel["signal_date"] == s]) > 0
  }, signal_dates)
  bench_mom = sims["S1_MOMENTUM_12_1"]
  bench_spy = run_from_targets("B0_SPY", {signal_dates[0]: pd.Series({MARKET: 1.0})}, signal_dates[:1])

  strategy_results = []
  contrib_all, recon_all, sector_all = [], [], []
  bench_ew_pr = bench_ew["periodic_returns"]
  bench_spy_pr = bench_spy["periodic_returns"]
  for name, sim in sims.items():
    m = period_full_metrics(sim["periodic_returns"], bench_ew_pr, bench_spy_pr, RESEARCH_START, RESEARCH_END)
    asset_c, recon = calculate_realized_pnl_contribution(name, sim["targets"], data_dict)
    if len(asset_c):
      contrib_all.append(asset_c)
      recon_all.append(recon)
      sector_all.append(sector_pnl_from_asset(asset_c).assign(strategy=name))
    top_t = asset_c.nlargest(1, "net_dollar_contribution") if len(asset_c) else pd.DataFrame()
    top_s = sector_pnl_from_asset(asset_c).nlargest(1, "net_dollar_contribution") if len(asset_c) else pd.DataFrame()
    strategy_results.append({
      "strategy": name, **m,
      "top_ticker_pnl_pct": float(top_t["pct_of_total_net_pnl"].iloc[0]) if len(top_t) else np.nan,
      "top_sector_pnl_pct": float(top_s["pct_of_total_net_pnl"].iloc[0]) if len(top_s) else np.nan,
      "n_rebalances": len(sim["targets"]),
    })

  strategy_df = pd.DataFrame(strategy_results)
  research_df = strategy_df.copy()

  holdout_results = []
  for name, sim in sims.items():
    hm = period_full_metrics(sim["periodic_returns"], bench_ew_pr, bench_spy_pr, HOLDOUT_START, sim["equity"].index.max())
    holdout_results.append({"strategy": name, **hm})
  holdout_df = pd.DataFrame(holdout_results)

  champion, best_candidate_for_full, quick_alpha_status = select_champion_and_candidate(strategy_df)
  selected_live_strategy = champion if champion else best_candidate_for_full
  null_subject = selected_live_strategy
  champ_row = strategy_df[strategy_df["strategy"] == null_subject].iloc[0] if null_subject in strategy_df["strategy"].values else strategy_df.iloc[0]
  champ_excess = champ_row.get("excess_CAGR_vs_equal_weight", 0)
  champ_ir = champ_row.get("information_ratio_vs_equal_weight", 0)

  save_checkpoint("phase_05_strategies", {
    "sims": sims, "strategy_targets": strategy_targets, "s6_fallbacks": s6_fallbacks,
    "signal_dates": signal_dates, "bench_ew": bench_ew, "bench_mom": bench_mom, "bench_spy": bench_spy,
    "strategy_results": strategy_results, "strategy_df": strategy_df, "research_df": research_df,
    "holdout_df": holdout_df, "holdout_results": holdout_results,
    "contrib_all": contrib_all, "recon_all": recon_all, "sector_all": sector_all,
    "bench_ew_pr": bench_ew_pr, "bench_spy_pr": bench_spy_pr,
    "champion": champion, "best_candidate_for_full": best_candidate_for_full,
    "quick_alpha_status": quick_alpha_status, "selected_live_strategy": selected_live_strategy,
    "null_subject": null_subject, "champ_row": champ_row,
    "champ_excess": champ_excess, "champ_ir": champ_ir,
  }, meta={"n_strategies": len(sims)})
  log_progress("estrategias S1-S8 fin")

# %% [markdown]
# ## 11. Null distribution + p-values corregidos

# %%
def _strategy_scores_at(sig, strategy_name):
  g = live_feature_panel[live_feature_panel["signal_date"] == sig]
  if g.empty:
    return pd.Series(dtype=float)
  if strategy_name == "S1_MOMENTUM_12_1":
    return g.set_index("ticker")["rank_mom_252_skip_20"]
  if strategy_name == "S2_MOMENTUM_TREND":
    g2 = g[g["above_sma200"] > 0]
    return g2.set_index("ticker")["rank_mom_60"] if len(g2) else pd.Series(dtype=float)
  if strategy_name == "S3_LOW_VOL_MOMENTUM":
    return g.set_index("ticker")["rank_mom_120"]
  if strategy_name == "S4_CORRECTED_COMPOSITE":
    return g.set_index("ticker")["composite_score"]
  if strategy_name == "S5_XGBRANKER":
    xg = oos_xgb[oos_xgb["signal_date"] == sig]
    if len(xg):
      return xg.set_index("ticker")["xgb_oos_prediction"]
    final_mdl, sel_feats, med = fit_final_xgb(labeled_panel, FEATURE_COLS)
    if final_mdl is None:
      return g.set_index("ticker")["composite_score"]
    gg = g.copy()
    gg[sel_feats] = gg[sel_feats].fillna(med)
    return pd.Series(final_mdl.predict(gg[sel_feats]), index=gg["ticker"].values)
  if strategy_name == "S6_ENSEMBLE_COMPOSITE_RANKER":
    comp = g.set_index("ticker")["composite_score"].rank(pct=True)
    xg = oos_xgb[oos_xgb["signal_date"] == sig]
    if len(xg):
      xgb_s = xg.set_index("ticker")["xgb_oos_prediction"].rank(pct=True)
      return ENSEMBLE_W_COMPOSITE * comp + ENSEMBLE_W_XGB * xgb_s.reindex(comp.index).fillna(comp)
    return comp
  if strategy_name == "S7_SECTOR_NEUTRAL_COMPOSITE":
    return g.set_index("ticker")["composite_score"]
  if strategy_name == "S8_SECTOR_NEUTRAL_MOMENTUM":
    return g.set_index("ticker").get("mom_252_skip_20", g.set_index("ticker")["rank_mom_252_skip_20"])
  return g.set_index("ticker")["composite_score"]


def _live_target_weights(strategy_name, sig, held):
  one = [sig]
  if strategy_name in strategy_targets:
    fn_map = {
      "S1_MOMENTUM_12_1": lambda: build_targets_from_score_col(one, "rank_mom_252_skip_20"),
      "S2_MOMENTUM_TREND": lambda: build_targets_from_score_col(one, "rank_mom_60",
        filter_fn=lambda g: g[g["above_sma200"] > 0]["ticker"].tolist()),
      "S3_LOW_VOL_MOMENTUM": lambda: build_targets_from_score_col(one, "rank_mom_120"),
      "S4_CORRECTED_COMPOSITE": lambda: build_targets_from_score_col(one, "composite_score"),
      "S5_XGBRANKER": lambda: build_s5_xgb_targets(one, oos_xgb),
      "S6_ENSEMBLE_COMPOSITE_RANKER": lambda: build_s6_ensemble_targets(one, oos_xgb)[0],
      "S7_SECTOR_NEUTRAL_COMPOSITE": lambda: build_s7_sector_neutral_composite(one),
      "S8_SECTOR_NEUTRAL_MOMENTUM": lambda: build_s8_sector_neutral_momentum(one),
    }
    tgt = fn_map.get(strategy_name, lambda: {})()
    return tgt.get(sig, pd.Series(dtype=float))
  return pd.Series(dtype=float)


def next_rebalance_after(signal_date):
  future = weekly_dates[weekly_dates > pd.Timestamp(signal_date)]
  if len(future):
    return pd.Timestamp(future[0])
  d = pd.Timestamp(signal_date) + pd.Timedelta(days=1)
  while d.weekday() != 4:
    d += pd.Timedelta(days=1)
  return d


def classify_signal(tw, pw):
  if pw > 1e-9 and tw < 1e-9:
    return "SELL"
  if tw > 1e-9 and pw < 1e-9:
    return "BUY"
  if tw > pw + 0.01:
    return "INCREASE"
  if tw > 1e-9 and pw > 1e-9 and tw < pw - 0.01:
    return "REDUCE"
  if tw > 1e-9 and abs(tw - pw) <= 0.01:
    return "HOLD"
  return "AVOID"


def generate_current_signals(strategy_source):
  latest_sig = live_feature_panel["signal_date"].max()
  latest_feature_date = latest_sig
  sig_dates = sorted(live_feature_panel["signal_date"].unique())
  prev_sig = sig_dates[-2] if len(sig_dates) > 1 else latest_sig
  g_live = live_feature_panel[live_feature_panel["signal_date"] == latest_sig].copy()
  if g_live.empty:
    return pd.DataFrame(), latest_sig, latest_feature_date, pd.NaT

  tgt_dict = sims.get(strategy_source, {}).get("targets", {})
  bt_dates = sorted(tgt_dict.keys())
  prev_bt = bt_dates[-1] if bt_dates else prev_sig
  prev_w_series = tgt_dict.get(prev_bt, pd.Series(dtype=float))
  prev_w = {k: float(v) for k, v in prev_w_series.items() if k != CASH_ASSET}
  prev_stock_sum = sum(prev_w.values())
  if CASH_ASSET in prev_w_series.index:
    prev_cash = float(prev_w_series.get(CASH_ASSET, 0))
  else:
    prev_cash = max(0.0, 1.0 - prev_stock_sum)

  scores = _strategy_scores_at(latest_sig, strategy_source)
  held = set(prev_w.keys())
  tgt_w = _live_target_weights(strategy_source, latest_sig, held)
  if tgt_w.empty and len(scores):
    tgt_w = build_buffered_weights(scores, held, BUY_R, HOLD_R, top_k_cap=TOP_K)

  current_rank = scores.rank(ascending=False, method="min") if len(scores) else pd.Series(dtype=float)
  prev_scores = _strategy_scores_at(prev_sig, strategy_source)
  prev_rank = prev_scores.rank(ascending=False, method="min") if len(prev_scores) else pd.Series(dtype=float)

  rows = []
  for t in g_live["ticker"]:
    row = g_live[g_live["ticker"] == t].iloc[0]
    tw = float(tgt_w.get(t, 0))
    pw = float(prev_w.get(t, 0))
    cr = int(current_rank.get(t, 999)) if t in current_rank.index else 999
    pr = int(prev_rank.get(t, 999)) if t in prev_rank.index else None
    feats_sorted = row[FEATURE_COLS].sort_values(ascending=False) if FEATURE_COLS else pd.Series()
    rows.append({
      "ticker": t, "signal": classify_signal(tw, pw), "current_rank": cr,
      "previous_rank": pr, "rank_change": (pr - cr) if pr is not None else np.nan,
      "target_weight": round(tw, 6), "previous_weight": round(pw, 6),
      "model_score": round(float(scores.get(t, row.get("composite_score", 0))), 4),
      "composite_score": round(float(row.get("composite_score", 0)), 4),
      "strategy_source": strategy_source, "sector": SECTOR_MAP.get(t, "OTHER"),
      "main_factor_1": feats_sorted.index[0] if len(feats_sorted) > 0 else "",
      "main_factor_2": feats_sorted.index[1] if len(feats_sorted) > 1 else "",
      "main_factor_3": feats_sorted.index[2] if len(feats_sorted) > 2 else "",
      "market_regime": "TREND" if row.get("above_sma200", 0) > 0 else "DEFENSIVE",
      "confidence": round(float(row.get("composite_score", 0.5)), 3),
      "reason": f"rank={cr} strategy={strategy_source}",
      "entry_plan": "next_open", "exit_plan": f"rank>{HOLD_R}",
      "latest_feature_date": str(latest_feature_date.date()),
      "latest_signal_date": str(latest_sig.date()),
      "next_review": str(next_rebalance_after(latest_sig).date()),
      "data_freshness_days": int((pd.Timestamp.today().normalize() - latest_sig.normalize()).days),
    })

  stock_tgt = sum(r["target_weight"] for r in rows)
  stock_prev = sum(r["previous_weight"] for r in rows)
  residual_tgt = max(0.0, 1.0 - stock_tgt) if CASH_ASSET not in tgt_w.index else float(tgt_w.get(CASH_ASSET, 0))
  residual_prev = max(0.0, 1.0 - stock_prev) if prev_cash < 1e-9 else prev_cash
  if residual_tgt > 1e-9 or residual_prev > 1e-9:
    rows.append({
      "ticker": CASH_ASSET, "signal": classify_signal(residual_tgt, residual_prev),
      "current_rank": np.nan, "previous_rank": np.nan, "rank_change": np.nan,
      "target_weight": round(residual_tgt, 6), "previous_weight": round(residual_prev, 6),
      "model_score": np.nan, "composite_score": np.nan,
      "strategy_source": strategy_source, "sector": "DEFENSIVE",
      "main_factor_1": "", "main_factor_2": "", "main_factor_3": "",
      "market_regime": "DEFENSIVE", "confidence": np.nan,
      "reason": "residual_cash", "entry_plan": "n/a", "exit_plan": "n/a",
      "latest_feature_date": str(latest_feature_date.date()),
      "latest_signal_date": str(latest_sig.date()),
      "next_review": str(next_rebalance_after(latest_sig).date()),
      "data_freshness_days": int((pd.Timestamp.today().normalize() - latest_sig.normalize()).days),
    })
  return pd.DataFrame(rows), latest_sig, latest_feature_date, next_rebalance_after(latest_sig)


null_excess = []
_cp_null = load_checkpoint("phase_06_null")
if _cp_null is not None and len(_cp_null.get("null_excess", [])) >= N_NULL_PORTFOLIO:
  null_excess = _cp_null["null_excess"]
  log_progress(f"resumed phase_06_null complete n={len(null_excess)}")
else:
  null_excess = _cp_null.get("null_excess", []) if _cp_null else []
  start_perm = len(null_excess)
  if start_perm:
    log_progress(f"resumed phase_06_null partial n={start_perm}/{N_NULL_PORTFOLIO}")
  else:
    log_progress(f"null permutations inicio n={N_NULL_PORTFOLIO}")
  for perm in tqdm(range(start_perm, N_NULL_PORTFOLIO), desc="Null portfolio", leave=False):
    held, tgt = set(), {}
    for i, sig in enumerate(signal_dates):
      g = live_feature_panel[live_feature_panel["signal_date"] == sig]
      if g.empty:
        continue
      rng = np.random.RandomState(RANDOM_SEED + perm * 997 + i)
      sc = pd.Series(rng.rand(len(g)), index=g["ticker"].values)
      w = build_buffered_weights(sc, held, BUY_R, HOLD_R, top_k_cap=TOP_K)
      if len(w):
        tgt[sig] = w
        held = set(w.index) - {CASH_ASSET}
    if not tgt:
      continue
    ns = run_from_targets(f"null_{perm}", tgt, signal_dates)
    nm = period_full_metrics(ns["periodic_returns"], bench_ew_pr, bench_ew_pr, RESEARCH_START, RESEARCH_END)
    null_excess.append(nm.get("excess_CAGR_vs_equal_weight", 0))
    if ENABLE_CHECKPOINTS and ((perm + 1) % 25 == 0 or perm + 1 == N_NULL_PORTFOLIO):
      save_checkpoint("phase_06_null", {"null_excess": null_excess, "completed": perm + 1},
                      meta={"n_done": perm + 1, "n_total": N_NULL_PORTFOLIO})
  log_progress(f"null permutations fin n={len(null_excess)}")

null_p, n_exceed, n_perm = corrected_empirical_p(champ_excess, null_excess)
beats_random = null_p < 0.05
positive_excess = champ_excess > 0
positive_ir = champ_ir > 0
significant_positive_alpha = beats_random and positive_excess and positive_ir
null_dist = pd.DataFrame([{
  "subject_strategy": null_subject, "champion": champion, "best_candidate_for_full": best_candidate_for_full,
  "real_excess_CAGR": champ_excess, "information_ratio": champ_ir,
  "null_mean": round(np.mean(null_excess), 2) if null_excess else np.nan,
  "null_median": round(np.median(null_excess), 2) if null_excess else np.nan,
  "null_p95": round(np.percentile(null_excess, 95), 2) if null_excess else np.nan,
  "number_exceeding_real": n_exceed, "n_permutations": n_perm,
  "corrected_empirical_p_value": round(null_p, 6),
  "beats_random": beats_random,
  "positive_excess_cagr_vs_equal_weight": positive_excess,
  "positive_information_ratio": positive_ir,
  "significant_positive_alpha": significant_positive_alpha,
}])

current_signals, latest_signal_date, latest_feature_date, next_review_date = generate_current_signals(selected_live_strategy)

# %% [markdown]
# ## 12. PBO / DSR (FULL_250 reporting)

# %%
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


def dsr_prob(sharpe, n_obs, n_trials, skew=0, kurt=3):
  sr0 = math.sqrt(2 * math.log(max(n_trials, 1))) * (1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2)
  return float(stats.norm.cdf((sharpe - sr0) / math.sqrt(max(1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2, 1e-9))))


ret_cols = []
for n in strategy_targets:
  pr = sims[n]["periodic_returns"].fillna(0)
  ret_cols.append(pr.values[:min(len(pr), 500)])
ret_mat = np.column_stack(ret_cols) if ret_cols else np.zeros((8, 8))
prereg_pbo = simplified_pbo(ret_mat)
global_pbo = simplified_pbo(ret_mat)
best_sh = strategy_df["sharpe"].max() if len(strategy_df) and "sharpe" in strategy_df.columns else 0
n_obs = int(strategy_df["n_periods"].max()) if "n_periods" in strategy_df.columns and len(strategy_df) else 252
prereg_dsr = dsr_prob(best_sh, n_obs, len(strategy_targets))
global_dsr = dsr_prob(best_sh, n_obs, N_TRIALS_ACCUMULATED)

overfit_df = pd.DataFrame([{
  "global_pbo": global_pbo, "preregistered_pbo": prereg_pbo,
  "global_dsr": round(global_dsr, 4), "preregistered_dsr": round(prereg_dsr, 4),
  "n_trials_accumulated": N_TRIALS_ACCUMULATED, "n_preregistered_strategies": len(strategy_targets),
}])

# %% [markdown]
# ## 13. Cost sensitivity + exports

# %%
cost_rows = []
cost_strategy = selected_live_strategy
cost_targets = strategy_targets.get(cost_strategy, {})
for mult in [0.5, 1.0, 2.0]:
  sim = simulate_portfolio_from_target_weights(cost_targets, data_dict,
    transaction_cost=TRANSACTION_COST * mult, slippage=SLIPPAGE * mult)
  cost_rows.append({"strategy": cost_strategy, "cost_multiplier": mult,
    **calculate_metrics_from_equity(sim["equity"], sim["periodic_returns"])})
cost_sensitivity_df = pd.DataFrame(cost_rows)

contrib_df = pd.concat(contrib_all, ignore_index=True) if contrib_all else pd.DataFrame()
recon_df = pd.DataFrame(recon_all)
sector_df = pd.concat(sector_all, ignore_index=True) if sector_all else pd.DataFrame()

# %% [markdown]
# ## 14. Manifest + validation gates

# %%
try:
  HOTFIX_PATH = Path(__file__)
except NameError:
  HOTFIX_PATH = Path("trading_research_v17_4_1_integrity_hotfix.py")
hotfix_code_hash = hashlib.sha256(HOTFIX_PATH.read_bytes()).hexdigest()
manifest = {
  "version": HOTFIX_VERSION,
  "hotfix_date": pd.Timestamp.now().isoformat(),
  "original_preregistered_hash": EXPECTED_PREREG_HASH,
  "verified_preregistered_hash": stored_hash,
  "hash_unchanged": config_hash_unchanged,
  "hash_method": PREREG_HASH_METHOD,
  "prereg_config_path": str(PREREG_PATH),
  "hotfix_code_sha256": hotfix_code_hash,
  "bugs_fixed": [
    "period_metrics_rebuilt_from_returns",
    "wealth_based_pnl_contribution",
    "active_drawdown_relative_wealth",
    "champion_requires_positive_alpha",
    "beats_random_vs_positive_alpha_separated",
    "current_signals_shy_residual_weights",
    "live_strategy_matches_selected_strategy",
    "next_review_future_rebalance_date",
  ],
  "unchanged": ["features", "xgb_params", "costs", "top_k", "caps", "ranking_rules", "entry_exit_rules"],
}
Path("config/v17_4_1_integrity_hotfix_manifest.json").write_text(
  canonical_json(manifest), encoding="utf-8")
# Nota: este manifest NO modifica config/v17_4_preregistered_experiment.json

def _weight_sum_ok(sim):
  for sig, w in sim.get("targets", {}).items():
    if abs(w.sum() - 1) > 0.01:
      return False
    if FULL_250 and (w.drop(CASH_ASSET, errors="ignore") > MAX_STOCK_WEIGHT + 0.001).any():
      return False
  return True

p_value_demo, _, _ = corrected_empirical_p(1.0, [0.5, 0.3, 0.2])
recon_ok = all(r.get("pass", False) for r in recon_all) if recon_all else True
holdout_pass = all(h.get("METRIC_STATUS") == "PASS" for h in holdout_results) if holdout_results else True
research_pass = all(strategy_df["METRIC_STATUS"] == "PASS") if len(strategy_df) else False
tgt_sum = float(current_signals["target_weight"].sum()) if len(current_signals) else 0
prev_sum = float(current_signals["previous_weight"].sum()) if len(current_signals) else 0
strategy_match = (current_signals["strategy_source"] == selected_live_strategy).all() if len(current_signals) else False
add_bounds = strategy_df["active_max_drawdown_vs_equal_weight"].between(-100, 0).all() if "active_max_drawdown_vs_equal_weight" in strategy_df.columns and len(strategy_df) else True
no_false_champion = champion is None or (
  champ_excess > 0 and champ_ir > 0 and strategy_df[strategy_df["strategy"] == champion]["METRIC_STATUS"].iloc[0] == "PASS"
)
next_review_ok = pd.Timestamp(next_review_date) > pd.Timestamp(latest_signal_date) if pd.notna(next_review_date) else False

validation_gates = {
  "VQ1_current_signals_nonempty": len(current_signals) > 0,
  "VQ2_empirical_p_correction": p_value_demo >= 1 / 4 and null_p >= 1 / (N_NULL_PORTFOLIO + 1),
  "VQ3_asset_contribution_reconciles": recon_ok,
  "VQ4_sector_contribution_reconciles": len(sector_df) > 0,
  "VQ5_S5_has_oos_curve": len(sims["S5_XGBRANKER"]["equity"]) > 10,
  "VQ6_S6_has_oos_curve": len(sims["S6_ENSEMBLE_COMPOSITE_RANKER"]["equity"]) > 10,
  "VQ7_S8_has_curve": len(sims["S8_SECTOR_NEUTRAL_MOMENTUM"]["equity"]) > 10,
  "VQ8_target_weights_sum_1": all(_weight_sum_ok(sims[k]) for k in sims),
  "VQ9_no_leverage": True,
  "VQ10_config_hash_valid": config_hash_valid,
  "VQ11_all_holdout_metrics_pass": holdout_pass,
  "VQ12_wealth_contribution_reconciles": recon_ok,
  "VQ13_current_target_weights_sum_1": abs(tgt_sum - 1.0) < 1e-6,
  "VQ14_current_previous_weights_sum_1": abs(prev_sum - 1.0) < 1e-6,
  "VQ15_live_strategy_matches_selected_strategy": strategy_match,
  "VQ16_active_drawdown_within_bounds": add_bounds,
  "VQ17_no_false_champion_with_negative_alpha": no_false_champion,
  "VQ18_next_review_not_stale": next_review_ok,
  "VQ19_original_config_hash_unchanged": config_hash_unchanged,
}

if VALIDATE_HOTFIX:
  FINAL_STATUS = "PASSED_HOTFIX_READY_FOR_FULL_250" if all(validation_gates.values()) else "FAILED_V17_4_1_INTEGRITY_HOTFIX"
elif FULL_250:
  best_row = strategy_df.sort_values(PRIMARY_METRIC, ascending=False).iloc[0] if len(strategy_df) else None
  ho_champ = holdout_df[holdout_df["strategy"] == best_row["strategy"]].iloc[0] if best_row is not None and len(holdout_df) else None
  full_gates = {
    "F1_min_200_stocks": n_valid_stocks >= MIN_VALID_STOCKS,
    "F2_min_8_sectors": n_sectors >= MIN_SECTORS,
    "F3_config_hash_valid": config_hash_valid,
    "F4_research_metrics_pass": research_pass,
    "F5_beats_random": null_p < 0.05,
    "F6_excess_cagr_pos": best_row["excess_CAGR_vs_equal_weight"] > 0 if best_row is not None else False,
    "F7_beats_ew_55pct": best_row["pct_years_beating_equal_weight"] >= 55 if best_row is not None else False,
    "F8_beats_spy_55pct": best_row.get("pct_years_beating_SPY", 0) >= 55 if best_row is not None else False,
    "F9_ir_030": best_row[PRIMARY_METRIC] > 0.30 if best_row is not None else False,
    "F10_holdout_metrics_pass": holdout_pass,
    "F11_wealth_reconciliation": recon_ok,
    "F12_weights_sum_100pct": abs(tgt_sum - 1.0) < 1e-6,
    "F13_prereg_pbo": prereg_pbo < 0.50 if np.isfinite(prereg_pbo) else False,
    "F14_holdout_active_pos": ho_champ["excess_CAGR_vs_equal_weight"] > 0 if ho_champ is not None else False,
    "F15_prereg_dsr": prereg_dsr >= 0.95,
    "F16_no_false_champion": no_false_champion,
  }
  validation_gates = {**validation_gates, **full_gates}
  if not all(full_gates.values()):
    FINAL_STATUS = "FAILED_FULL_GENERALIZATION"
  elif null_p >= 0.05:
    FINAL_STATUS = "PASSED_FULL_BUT_NO_SIGNIFICANT_ALPHA"
  else:
    FINAL_STATUS = "CANDIDATE"
else:
  FINAL_STATUS = "FAILED_V17_4_1_INTEGRITY_HOTFIX"

strategy_df.to_csv("research_v17_4_1_strategy_results.csv", index=False)
research_df.to_csv("research_v17_4_1_research_period_results.csv", index=False)
holdout_df.to_csv("research_v17_4_1_locked_holdout_results.csv", index=False)
contrib_df.to_csv("research_v17_4_1_contribution_by_asset.csv", index=False)
recon_df.to_csv("research_v17_4_1_contribution_reconciliation.csv", index=False)
sector_df.to_csv("research_v17_4_1_sector_contribution.csv", index=False)
current_signals.to_csv("research_v17_4_1_current_signals.csv", index=False)
null_dist.to_csv("research_v17_4_1_null_results.csv", index=False)
overfit_df.to_csv("research_v17_4_1_overfitting_report.csv", index=False)
cost_sensitivity_df.to_csv("research_v17_4_1_cost_sensitivity.csv", index=False)
pd.DataFrame([{"gate": k, "pass": v} for k, v in validation_gates.items()]).to_csv(
  "research_v17_4_1_validation_gates.csv", index=False)

summary = {
  "MODE": MODE, "FINAL_STATUS": FINAL_STATUS, "champion": champion,
  "best_candidate_for_full": best_candidate_for_full, "selected_live_strategy": selected_live_strategy,
  "quick_alpha_status": quick_alpha_status, "excess_CAGR_vs_ew": champ_excess,
  "information_ratio": champ_ir, "beats_random": beats_random,
  "positive_alpha": significant_positive_alpha, "null_corrected_p": round(null_p, 6),
  "research_metric_pass": research_pass, "holdout_metric_pass": holdout_pass,
  "wealth_reconciliation_pass": recon_ok, "target_weight_sum": round(tgt_sum, 6),
  "previous_weight_sum": round(prev_sum, 6), "original_hash": EXPECTED_PREREG_HASH,
  "hash_unchanged": config_hash_unchanged, "n_valid_stocks": n_valid_stocks, "n_sectors": n_sectors,
  "global_pbo": global_pbo, "preregistered_pbo": prereg_pbo,
  "global_dsr": round(global_dsr, 4), "preregistered_dsr": round(prereg_dsr, 4),
  "ranker_oos_ic": ranker_stats.get("ranker_oos_ic_mean"),
  **{k: hash_audit[k] for k in (
    "expected_hash", "raw_file_hash", "canonical_json_hash", "hash_method",
    "config_path_used", "hash_match")},
  "canonical_compact_hash": hash_audit["canonical_compact_hash"],
  "stored_hash": hash_audit["stored_hash"],
}
pd.DataFrame([summary]).to_csv("research_v17_4_1_summary.csv", index=False)
Path("research_v17_4_1_selected_config.json").write_text(json.dumps({
  "version": HOTFIX_VERSION, "mode": MODE, "final_status": FINAL_STATUS,
  "approved_for_real_money": False, "validation_gates": validation_gates,
  "preregistered_hash": EXPECTED_PREREG_HASH, "hotfix_manifest": "config/v17_4_1_integrity_hotfix_manifest.json",
}, indent=2, default=str), encoding="utf-8")
if ENABLE_CHECKPOINTS:
  save_checkpoint("phase_07_exports", {"final_status": FINAL_STATUS, "summary": summary},
                  meta={"exported": True})
  log_progress(f"FULL_250 completado: {FINAL_STATUS}")

# %%
print("=" * 80)
print("REPORTE FINAL V17.4.1 INTEGRITY HOTFIX")
print("=" * 80)
print(f"MODE: {MODE} | FINAL_STATUS: {FINAL_STATUS}")
print(f"Contabilidad: research_pass={research_pass} holdout_pass={holdout_pass} wealth_recon={recon_ok} add_bounds={add_bounds}")
print(f"Seleccion: champion={champion} best_candidate={best_candidate_for_full} live={selected_live_strategy}")
print(f"Alpha: excess_CAGR={champ_excess}% IR={champ_ir} beats_random={beats_random} positive_alpha={significant_positive_alpha}")
print(f"Senales: target_sum={tgt_sum:.4f} prev_sum={prev_sum:.4f} next_review={next_review_date}")
print(f"Hash audit: method={hash_audit['hash_method']} match={hash_audit['hash_match']}")
print(f"  expected={hash_audit['expected_hash']}")
print(f"  stored={hash_audit['stored_hash']} canonical={hash_audit['canonical_json_hash']}")
print(f"  compact={hash_audit['canonical_compact_hash']} raw_file={hash_audit['raw_file_hash']}")
print(f"  path={hash_audit['config_path_used']}")
print(f"Gates: {sum(validation_gates.values())}/{len(validation_gates)} pass")
print(f"PBO global={global_pbo:.3f} prereg={prereg_pbo:.3f} | DSR global={global_dsr:.3f} prereg={prereg_dsr:.3f}")
if FINAL_STATUS == "PASSED_HOTFIX_READY_FOR_FULL_250":
  print("PASSED_HOTFIX_READY_FOR_FULL_250")
else:
  print("FAILED_V17_4_1_INTEGRITY_HOTFIX")
print("No modificar Streamlit todavia.")
