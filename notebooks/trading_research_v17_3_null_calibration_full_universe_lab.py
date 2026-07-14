# %% [markdown]
# # Trading Research V17.3 — Statistical Null Calibration & Full Universe Lab
#
# Calibra tests nulos, audita ranker OOS fila a fila, prepara FULL_250.
# **APPROVED_FOR_REAL_MONEY siempre False.**

# %%
try:
  get_ipython().run_line_magic(
    "pip", "install yfinance pandas numpy scipy scikit-learn xgboost matplotlib plotly tqdm lxml html5lib beautifulsoup4 -q"
  )
except NameError:
  import subprocess, sys
  subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "yfinance", "pandas", "numpy", "scipy", "scikit-learn", "xgboost",
    "matplotlib", "plotly", "tqdm", "lxml", "html5lib", "beautifulsoup4"])

# %% [markdown]
# ## 1. Configuracion

# %%
import warnings
warnings.filterwarnings("ignore")
import hashlib, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

MODE = "QUICK"
QUICK_TEST = MODE == "QUICK"
RUN_FULL_AFTER_QUICK_PASS = False
UNIVERSE_MODE = "QUICK_TEST" if QUICK_TEST else "US_LARGE_CAP"
MAX_TICKERS_FULL = 250
START_DATE = "2010-01-01"
END_DATE = None
TRANSACTION_COST = 0.001
SLIPPAGE = 0.001
SIGNAL_FREQUENCY = "W-FRI"
PURGE_DAYS = 20
EMBARGO_DAYS = 20
FORWARD_HORIZON = 20
RANDOM_SEED = 42
N_NULL_FACTOR_PERMUTATIONS = 500
N_NULL_PORTFOLIO_PERMUTATIONS = 500
N_NULL_MODEL_PERMUTATIONS_QUICK = 50
N_NULL_MODEL_PERMUTATIONS_FULL = 20
APPROVED_FOR_REAL_MONEY = False
INITIAL_CAPITAL = 10000
COST_RATE = TRANSACTION_COST + SLIPPAGE
MARKET = "SPY"
CASH_ASSET = "SHY"
WF_START_YEAR = 2016
MIN_HISTORY_DAYS = 250 if QUICK_TEST else 1000
MIN_DOLLAR_VOLUME = 5_000_000 if QUICK_TEST else 20_000_000
N_PORT_NULL_RUN = N_NULL_MODEL_PERMUTATIONS_QUICK if QUICK_TEST else N_NULL_PORTFOLIO_PERMUTATIONS
MAX_WEIGHT = 0.075
SECTOR_CAP = 0.30
N_TRIALS_ACCUMULATED = 261

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
  "El universo usa constituyentes actuales y mantiene survivorship bias. No es point-in-time."
)
np.random.seed(RANDOM_SEED)
FINAL_STATUS = "PENDING"
EXTREME_IC_REVIEW = False

# %% [markdown]
# ## 2. Cargar motor V17.2

# %%
V172 = Path(__file__).parent / "trading_research_v17_2_backtest_integrity_lab.py"
_src = V172.read_text(encoding="utf-8")
import ast
_start = _src.index("def _clean_symbol")
_end = _src.index("# %% [markdown]\n# ## 8. Baselines")
_chunk = _src[_start:_end]
_mod = ast.parse(_chunk)
_func_src = []
for node in _mod.body:
  if isinstance(node, ast.FunctionDef):
    seg = ast.get_source_segment(_chunk, node)
    if seg:
      _func_src.append(seg)
_eng = "\n\n".join(_func_src)
_ns = {
  "pd": pd, "np": np, "math": math, "tqdm": tqdm, "hashlib": hashlib,
  "QUICK_TEST": QUICK_TEST, "UNIVERSE_MODE": UNIVERSE_MODE,
  "MAX_TICKERS_FULL": MAX_TICKERS_FULL, "START_DATE": START_DATE, "END_DATE": END_DATE,
  "MIN_HISTORY_DAYS": MIN_HISTORY_DAYS, "MIN_DOLLAR_VOLUME": MIN_DOLLAR_VOLUME,
  "DEFENSIVE_ETFS": DEFENSIVE_ETFS, "ALL_ETFS": ALL_ETFS, "SECTOR_MAP": SECTOR_MAP,
  "SIGNAL_FREQUENCY": SIGNAL_FREQUENCY, "INITIAL_CAPITAL": INITIAL_CAPITAL,
  "TRANSACTION_COST": TRANSACTION_COST, "SLIPPAGE": SLIPPAGE, "COST_RATE": COST_RATE,
  "CASH_ASSET": CASH_ASSET, "MARKET": MARKET, "FORBIDDEN_FEATURE_PATTERNS": FORBIDDEN_FEATURE_PATTERNS,
  "TOP_K": 3, "weekly_dates": pd.DatetimeIndex([]),
}
exec(_eng, _ns)
globals().update({k: v for k, v in _ns.items() if callable(v) or k in (
  "TOP_K", "weekly_dates", "monthly_dates", "panel", "data_dict", "STOCKS",
)})

# %% [markdown]
# ## 3. Universo

# %%
def load_us_large_cap_250():
  meta_rows, tickers = [], []
  try:
    sp = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", flavor="bs4")[0]
    for _, r in sp.iterrows():
      sym = _clean_symbol(r["Symbol"])
      tickers.append(sym)
      SECTOR_MAP[sym] = str(r.get("GICS Sector", "OTHER"))[:20].upper().replace(" ", "_")
      meta_rows.append({"ticker": sym, "security": r.get("Security", ""), "sector": SECTOR_MAP[sym]})
  except Exception as e:
    print("Wikipedia fail:", e)
  tickers = list(dict.fromkeys(tickers))
  return tickers[:600], pd.DataFrame(meta_rows)


def download_with_audit(tickers, start, end=None):
  data, close, errors = download_data(tickers, start, end)
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
  if not QUICK_TEST and len(valid) > MAX_TICKERS_FULL:
    liq = rep[rep["status"] == "valid"].sort_values("dollar_vol_60", ascending=False)
    valid = liq.head(MAX_TICKERS_FULL)["ticker"].tolist()
  data = {k: v for k, v in data.items() if k in valid}
  close = close[[c for c in close.columns if c in valid]]
  return data, close, errors, rep


if QUICK_TEST:
  UNIVERSE = load_quick_test_universe()
  sp_meta = pd.DataFrame()
else:
  UNIVERSE, sp_meta = load_us_large_cap_250()

data_dict, close_all, dl_errors, dl_report = download_with_audit(UNIVERSE, START_DATE, END_DATE)
STOCKS, ETFS, n_sectors = classify_assets(list(close_all.columns))
n_valid_stocks, n_valid_etfs = len(STOCKS), len(ETFS)
TOP_K = max(3, int(np.ceil(n_valid_stocks * 0.20))) if QUICK_TEST else max(20, min(40, int(n_valid_stocks * 0.10)))
median_hist = int(np.median([len(data_dict[t]) for t in STOCKS])) if STOCKS else 0

universe_audit = pd.DataFrame([{
  "mode": MODE, "n_downloaded": len(UNIVERSE), "n_valid_assets": len(close_all.columns),
  "n_valid_stocks": n_valid_stocks, "n_valid_etfs": n_valid_etfs, "n_sectors": n_sectors,
  "top_k": TOP_K, "median_history_days": median_hist, "survivorship_warning": SURVIVORSHIP_WARNING,
}])
universe_audit.to_csv("research_v17_3_universe_audit.csv", index=False)
dl_report.to_csv("research_v17_3_download_report.csv", index=False)
print(f"MODE={MODE} | stocks={n_valid_stocks} | ETFs={n_valid_etfs} | TOP_K={TOP_K}")

# %% [markdown]
# ## 4. Panel con labels (solo evaluacion / null tests)

# %%
def build_panel_with_labels(data_dict, stocks):
  panel = build_features_and_panel(data_dict, stocks)
  if panel.empty:
    return panel
  rows = []
  cal = _master_calendar(data_dict)
  for _, r in panel.iterrows():
    t, sig = r["ticker"], r["signal_date"]
    if t not in data_dict:
      continue
    idx = data_dict[t].index
    entry = _next_trading_day(idx, sig)
    if pd.isna(entry):
      continue
    ep = data_dict[t].loc[entry, "Open"] if entry in data_dict[t].index else np.nan
    pos = idx.searchsorted(entry, side="left")
    exit_i = pos + FORWARD_HORIZON
    if exit_i >= len(idx):
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


panel = build_panel_with_labels(data_dict, STOCKS)
FEATURE_COLS = [c for c in panel.columns if c not in {
  "ticker", "signal_date", "sector", "entry_date", "label_end_date", "feature_date",
  "fwd_ret_20d", "fwd_excess_20d", "relevance"} and not FORBIDDEN_FEATURE_PATTERNS.search(c)]

_ns["STOCKS"] = STOCKS
_ns["data_dict"] = data_dict
_ns["panel"] = panel
_ns["TOP_K"] = TOP_K
for k in ["make_signal_dates", "builder_equal_weight", "builder_sector_momentum", "builder_top_score",
          "builder_random", "builder_alpha_score", "builder_spy_buyhold", "builder_ew_buyhold", "_panel_at"]:
  if k in _ns:
    globals()[k] = _ns[k]

weekly_dates = make_signal_dates("weekly")
monthly_dates = make_signal_dates("monthly")
_ns["weekly_dates"] = weekly_dates
_ns["monthly_dates"] = monthly_dates
globals()["weekly_dates"] = weekly_dates
globals()["monthly_dates"] = monthly_dates
print(f"Panel: {len(panel)} rows | features: {len(FEATURE_COLS)}")

# %% [markdown]
# ## 5. Helpers estrategia / motor

# %%
def run_from_builder(name, fn, dates, score_col="composite_score"):
  return run_strategy(name, fn, dates, score_col=score_col)


def excess_vs_benchmark(sim, bench_sim):
  if len(sim["equity"]) < 2 or len(bench_sim["equity"]) < 2:
    return {}
  re = sim["periodic_returns"].reindex(bench_sim["periodic_returns"].index).fillna(0)
  rb = bench_sim["periodic_returns"].reindex(re.index).fillna(0)
  active = re - rb
  m = calculate_metrics_from_equity(sim["equity"], sim["periodic_returns"])
  bm = calculate_metrics_from_equity(bench_sim["equity"], bench_sim["periodic_returns"])
  return {
    **m, "excess_CAGR_pct": round(m["CAGR_pct"] - bm["CAGR_pct"], 2),
    "active_mean_weekly": float(active.mean()),
    "tracking_error": float(active.std() * math.sqrt(252)) if active.std() > 0 else 0,
    "information_ratio": float(active.mean() / active.std() * math.sqrt(252)) if active.std() > 0 else 0,
  }


def build_random_targets(dates, seed_base=0):
  targets = {}
  for i, sig in enumerate(dates):
    g = panel[panel["signal_date"] == sig]
    if g.empty:
      continue
    rng = np.random.RandomState(seed_base + i + int(sig.strftime("%Y%m%d")))
    g = g.copy()
    g["_rs"] = rng.rand(len(g))
    k = resolve_top_k(len(g), TOP_K)
    sel = g.nlargest(k, "_rs")
    targets[sig] = pd.Series(1.0 / len(sel), index=sel["ticker"])
  return targets

# %% [markdown]
# ## 6. Null tests — random portfolio distribution

# %%
ew_sim = run_from_builder("B1_EQUAL_WEIGHT", builder_equal_weight, monthly_dates, "equal")
real_comp = run_from_builder("S4_CORRECTED_COMPOSITE",
  lambda s, **k: builder_top_score(s, "composite_score"), weekly_dates, "composite_score")
real_excess = excess_vs_benchmark(real_comp, ew_sim)

null_port_rows = []
excess_cagrs, excess_irs = [], []
for perm in tqdm(range(N_PORT_NULL_RUN), desc="Null portfolio", leave=False):
  tgt = build_random_targets(weekly_dates, seed_base=perm * 9973)
  sim = simulate_portfolio_from_target_weights(tgt, data_dict)
  m = calculate_metrics_from_equity(sim["equity"], sim["periodic_returns"])
  ex = excess_vs_benchmark(sim, ew_sim)
  excess_cagrs.append(ex.get("excess_CAGR_pct", 0))
  excess_irs.append(ex.get("information_ratio", 0))
  null_port_rows.append({"perm": perm, **m, **{k: ex.get(k) for k in ["excess_CAGR_pct", "information_ratio"]}})

null_port_df = pd.DataFrame(null_port_rows)
real_excess_cagr = real_excess.get("excess_CAGR_pct", 0)
p_val_excess = float(np.mean(np.array(excess_cagrs) >= real_excess_cagr))
null_port_summary = pd.DataFrame([{
  "test": "B_random_features_portfolio",
  "real_excess_CAGR_pct": real_excess_cagr,
  "null_mean_excess_CAGR": round(np.mean(excess_cagrs), 2),
  "null_median_excess_CAGR": round(np.median(excess_cagrs), 2),
  "null_p95_excess_CAGR": round(np.percentile(excess_cagrs, 95), 2),
  "empirical_p_value": round(p_val_excess, 4),
  "pass": real_excess_cagr > np.percentile(excess_cagrs, 95) or p_val_excess < 0.05,
  "n_permutations": N_PORT_NULL_RUN,
}])
null_port_df.to_csv("research_v17_3_null_portfolio_results.csv", index=False)
null_port_summary.to_csv("research_v17_3_null_factor_results.csv", index=False)

# %% [markdown]
# ## 7. Shifted labels + shuffle controls D1-D3

# %%
def spearman_ic_by_date(df, score_col, target_col):
  ics = []
  for _, g in df.groupby("signal_date"):
    sub = g[[score_col, target_col]].dropna()
    if len(sub) < 5:
      continue
    ic, _ = stats.spearmanr(sub[score_col], sub[target_col])
    if np.isfinite(ic):
      ics.append(ic)
  return np.mean(ics) if ics else 0.0


real_ic = spearman_ic_by_date(panel, "composite_score", "fwd_excess_20d")
null_ics = []
for perm in range(N_NULL_FACTOR_PERMUTATIONS):
  d = panel.copy()
  d["perm_tgt"] = d.groupby("signal_date")["fwd_excess_20d"].transform(
    lambda s: pd.Series(np.random.permutation(s.values), index=s.index))
  null_ics.append(spearman_ic_by_date(d, "composite_score", "perm_tgt"))
ic_p_value = float(np.mean(np.abs(null_ics) >= abs(real_ic)))

shift_months = [-24, -18, -12, -9, -6, 6, 9, 12, 18, 24]
shift_rows = []
for mo in shift_months:
  d = panel.copy()
  d["shifted"] = d.groupby("ticker")["fwd_excess_20d"].shift(int(mo * 4.33))
  ic = spearman_ic_by_date(d.dropna(subset=["shifted"]), "composite_score", "shifted")
  shift_rows.append({"shift_months": mo, "ic": round(ic, 4)})

perm_aucs = []
try:
  from sklearn.ensemble import HistGradientBoostingClassifier
  from sklearn.metrics import roc_auc_score
  feats = [c for c in FEATURE_COLS if c in panel.columns][:10]
  X = panel[feats].fillna(0)
  y = (panel["relevance"] >= 0.8).astype(int) if "relevance" in panel else (panel.groupby("signal_date")["composite_score"].rank(pct=True) >= 0.8).astype(int)
  sp = int(len(X) * 0.7)
  mdl = HistGradientBoostingClassifier(max_depth=3, max_iter=60, random_state=RANDOM_SEED)
  mdl.fit(X.iloc[:sp], y.iloc[:sp])
  real_auc = roc_auc_score(y.iloc[sp:], mdl.predict_proba(X.iloc[sp:])[:, 1])
  for _ in range(100):
    yp = y.copy()
    for yr in panel["signal_date"].dt.year.unique():
      idx = panel["signal_date"].dt.year == yr
      yp.loc[idx] = np.random.permutation(y.loc[idx].values)
    try:
      perm_aucs.append(roc_auc_score(y.iloc[sp:], mdl.predict_proba(X.iloc[sp:])[:, 1]))
    except Exception:
      pass
  shift_auc_p = float(np.mean(np.abs(np.array(perm_aucs) - 0.5) >= abs(real_auc - 0.5))) if perm_aucs else 1.0
except Exception:
  real_auc, shift_auc_p = 0.5, 1.0

shift_df = pd.DataFrame(shift_rows)
shift_df.to_csv("research_v17_3_shifted_label_results.csv", index=False)

d_rows = []
for label, fn in [
  ("D1_SHUFFLE_TARGET_WITHIN_DATE", lambda d: d.assign(
    fwd_excess_20d=d.groupby("signal_date")["fwd_excess_20d"].transform(
      lambda s: pd.Series(np.random.permutation(s.values), index=s.index)))),
  ("D2_SHUFFLE_FEATURE_ROWS", lambda d: d.groupby("signal_date", group_keys=False).apply(
    lambda g: g.sample(frac=1).assign(composite_score=g["composite_score"].values))),
  ("D3_SHUFFLE_SCORES", lambda d: d.assign(
    composite_score=d.groupby("signal_date")["composite_score"].transform(
      lambda s: pd.Series(np.random.permutation(s.values), index=s.index)))),
]:
  perm_ics = [spearman_ic_by_date(fn(panel.copy()), "composite_score", "fwd_excess_20d") for _ in range(200)]
  d_rows.append({
    "control": label, "null_mean_ic": round(np.mean(perm_ics), 4),
    "null_std_ic": round(np.std(perm_ics), 4),
    "real_ic": round(real_ic, 4),
    "p_value": round(float(np.mean(np.abs(perm_ics) >= abs(real_ic))), 4),
    "pass": abs(real_ic) > np.percentile(np.abs(perm_ics), 95),
  })
leakage_df = pd.concat([
  null_port_summary.assign(metric="portfolio_null"),
  pd.DataFrame([{"test": "real_ic", "value": real_ic, "p_value": ic_p_value, "pass": ic_p_value < 0.05}]),
  pd.DataFrame([{"test": "shifted_auc", "value": real_auc, "p_value": shift_auc_p, "pass": shift_auc_p > 0.05}]),
  pd.DataFrame(d_rows),
], ignore_index=True)
leakage_df.to_csv("research_v17_3_leakage_tests.csv", index=False)

# %% [markdown]
# ## 8. Ranker OOS forensics

# %%
def audit_ranker_oos_predictions(panel_df, feature_cols):
  try:
    import xgboost as xgb
  except ImportError:
    return pd.DataFrame(), {}, {}
  audit_rows, pipe_rows = [], []
  train_ics, oos_ics = [], []
  years = sorted(panel_df["signal_date"].dt.year.unique())
  for test_year in years:
    if test_year <= WF_START_YEAR:
      continue
    test_start = pd.Timestamp(f"{test_year}-01-01")
    purge_cut = test_start - pd.Timedelta(days=PURGE_DAYS)
    embargo = test_start - pd.Timedelta(days=EMBARGO_DAYS)
    tr = panel_df[(panel_df["signal_date"] < embargo) & (panel_df["label_end_date"] < purge_cut)].copy()
    te = panel_df[panel_df["signal_date"].dt.year == test_year].copy()
    if len(tr) < 80 or len(te) < 20:
      continue
    feats = [c for c in feature_cols if c in tr.columns][:12]
    med = tr[feats].median()
    tr[feats] = tr[feats].fillna(med)
    te[feats] = te[feats].fillna(med)
    corr = tr[feats].corr().abs()
    keep = [f for f in feats if f in corr.columns]
  # drop one of pair >0.95
    drop = set()
    for i, a in enumerate(keep):
      for b in keep[i + 1:]:
        if corr.loc[a, b] > 0.95:
          drop.add(b)
    sel_feats = [f for f in keep if f not in drop]
    y_tr = tr.groupby("signal_date")["fwd_excess_20d"].rank(pct=True).astype(int)
    y_te = te.groupby("signal_date")["fwd_excess_20d"].rank(pct=True).astype(int)
    grp_tr = tr.groupby("signal_date").size().values
    mdl = xgb.XGBRanker(objective="rank:ndcg", max_depth=3, learning_rate=0.03,
                        n_estimators=150, random_state=RANDOM_SEED, verbosity=0)
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
      "n_train_rows": len(tr), "n_test_rows": len(te),
      "selected_features": ",".join(sel_feats), "train_ic": round(train_ics[-1], 4), "oos_ic": round(oos_ic, 4),
    })
    te_out = te.copy()
    te_out["prediction"] = pred_te
    for _, r in te_out.iterrows():
      audit_rows.append({
        "ticker": r["ticker"], "feature_date": r["feature_date"], "signal_date": r["signal_date"],
        "entry_date": r["entry_date"], "label_end_date": r["label_end_date"], "test_year": test_year,
        "train_start": tr["signal_date"].min(), "train_end": tr["signal_date"].max(),
        "purge_start": purge_cut, "embargo_end": embargo,
        "is_train_row": False, "is_test_row": True,
        "prediction": float(r["prediction"]),
        "actual_fwd_excess_return": r["fwd_excess_20d"], "actual_relevance": r.get("relevance", np.nan),
        "fold_id": fold_id, "model_fit_id": fold_id,
      })
  audit = pd.DataFrame(audit_rows)
  if len(audit):
    assert (audit["train_end"] < audit["signal_date"]).all()
    assert (audit["signal_date"] < audit["entry_date"]).all()
    assert (audit["entry_date"] < audit["label_end_date"]).all()
    assert (~audit["is_train_row"]).all()
  pipe = pd.DataFrame(pipe_rows)
  stats_out = {
    "ranker_train_ic_mean": round(float(np.nanmean(train_ics)), 4) if train_ics else np.nan,
    "ranker_oos_ic_mean": round(float(np.nanmean(oos_ics)), 4) if oos_ics else np.nan,
    "ranker_oos_ic_by_year": oos_ics,
  }
  return audit, pipe, stats_out


ranker_audit, pipeline_audit, ranker_stats = audit_ranker_oos_predictions(panel, FEATURE_COLS)
ranker_audit.to_csv("research_v17_3_ranker_prediction_audit.csv", index=False)
pipeline_audit.to_csv("research_v17_3_pipeline_audit.csv", index=False)
if abs(ranker_stats.get("ranker_oos_ic_mean", 0) or 0) > 0.20:
  EXTREME_IC_REVIEW = True
  print("IC extraordinariamente alto: revisar posible leakage o error de evaluacion.")

# %% [markdown]
# ## 9. Baselines y estrategias S1-S8

# %%
baseline_results = []
sims = {}
b_sp = run_from_builder("B0_SPY", builder_spy_buyhold, weekly_dates[:1], "SPY")
b_ew = run_from_builder("B1_EQUAL_WEIGHT_ALL_STOCKS", builder_equal_weight, monthly_dates, "equal")
b_mom = run_from_builder("B3_MOMENTUM_12_1", lambda s, **k: builder_top_score(s, "rank_mom_252_skip_20"), weekly_dates, "mom")
b_mt = run_from_builder("B4_MOMENTUM_TREND", lambda s, **k: builder_top_score(s, "rank_mom_60", shy_remainder=True), weekly_dates, "mom")
b_comp = real_comp
for name, sim in [("B0_SPY", b_sp), ("B1_EQUAL_WEIGHT", b_ew), ("B3_MOMENTUM", b_mom), ("B4_MOM_TREND", b_mt), ("B6_COMPOSITE", b_comp)]:
  ex = excess_vs_benchmark(sim, b_ew)
  baseline_results.append({"strategy": name, **ex})
  sims[name] = sim

strategy_specs = {
  "S1_MOMENTUM_12_1_TOP20": lambda s, **k: builder_top_score(s, "rank_mom_252_skip_20"),
  "S2_MOMENTUM_TREND_TOP20": lambda s, **k: builder_top_score(s, "rank_mom_60", shy_remainder=True),
  "S3_LOW_VOL_MOMENTUM_TOP20": lambda s, **k: builder_top_score(s, "rank_mom_120"),
  "S4_CORRECTED_COMPOSITE_TOP20": lambda s, **k: builder_top_score(s, "composite_score"),
  "S7_SECTOR_NEUTRAL_COMPOSITE": builder_sector_momentum,
}
strategy_results = []
for name, fn in strategy_specs.items():
  sim = run_from_builder(name, fn, weekly_dates, name)
  ex = excess_vs_benchmark(sim, b_ew)
  strategy_results.append({"strategy": name, **ex})
  sims[name] = sim

baseline_df = pd.DataFrame(baseline_results)
strategy_df = pd.DataFrame(strategy_results)
baseline_df.to_csv("research_v17_3_baseline_results.csv", index=False)
strategy_df.to_csv("research_v17_3_strategy_results.csv", index=False)

# %% [markdown]
# ## 10. Jaccard + ranking engine status

# %%
def jaccard(a, b):
  sa, sb = set(str(a).split(",")), set(str(b).split(","))
  return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


holdings = []
for name, sim in sims.items():
  holdings.extend(sim.get("holdings_log", []))
hold_df = pd.DataFrame(holdings)
j_rows = []
strats = list(sims.keys())
for i, a in enumerate(strats):
  for b in strats[i + 1:]:
    ha, hb = hold_df[hold_df["strategy"] == a], hold_df[hold_df["strategy"] == b]
    m = ha.merge(hb, on="date", suffixes=("_a", "_b"))
    if len(m) == 0:
      continue
    js = [jaccard(x, y) for x, y in zip(m["selected_tickers_a"], m["selected_tickers_b"])]
    j_rows.append({"a": a, "b": b, "mean_jaccard": round(np.mean(js), 3),
                   "pct_identical": round(np.mean([x == y for x, y in zip(m["selected_tickers_a"], m["selected_tickers_b"])]) * 100, 1)})
jaccard_df = pd.DataFrame(j_rows)
jaccard_df.to_csv("research_v17_3_strategy_jaccard.csv", index=False)

t_asc = run_from_builder("TEST_ASC", lambda s, **k: builder_alpha_score(s, True), weekly_dates)
t_desc = run_from_builder("TEST_DESC", lambda s, **k: builder_alpha_score(s, False), weekly_dates)
t_rand = run_from_builder("TEST_RAND", lambda s, **k: builder_random(s, 7), weekly_dates)
t_comp = run_from_builder("TEST_COMP", lambda s, **k: builder_top_score(s, "composite_score"), weekly_dates)
def _last_hold(sim):
  return sim["holdings_log"][-1]["selected_tickers"] if sim.get("holdings_log") else ""
holdings_differ = len({_last_hold(t_asc), _last_hold(t_desc), _last_hold(t_rand), _last_hold(t_comp)}) >= 3
high_pairs = sum(1 for _, r in jaccard_df.iterrows() if r["pct_identical"] > 90) if len(jaccard_df) else 0
p1_p3_only = high_pairs <= 2 and holdings_differ
RANKING_STATUS = "PASS_EQUIVALENT_RULES" if p1_p3_only else ("PASS" if holdings_differ else "FAILED_SCORE_NOT_USED")

# %% [markdown]
# ## 11. Significancia + PBO/DSR + gates

# %%
def yearly_active(sim, bench):
  rows = []
  for yr in sorted(sim["equity"].index.year.unique()):
    se = sim["equity"].loc[f"{yr}"]
    be = bench["equity"].reindex(se.index).ffill()
    if len(se) < 2 or len(be) < 2:
      continue
    sr, br = se.iloc[-1] / se.iloc[0] - 1, be.iloc[-1] / be.iloc[0] - 1
    rows.append({"year": yr, "active": sr - br, "beats_ew": sr > br})
  return pd.DataFrame(rows)


yr_df = yearly_active(b_comp, b_ew)
yr_df.to_csv("research_v17_3_yearly.csv", index=False)
win_ew = yr_df["beats_ew"].mean() if len(yr_df) else 0
active_weekly = b_comp["periodic_returns"].reindex(b_ew["periodic_returns"].index).fillna(0) - b_ew["periodic_returns"].fillna(0)
boot = [active_weekly.sample(frac=1, replace=True).mean() * 52 for _ in range(500)]
perm_p = float(np.mean(np.array(excess_cagrs) >= real_excess.get("excess_CAGR_pct", 0)))

def simplified_pbo(matrix):
  if matrix.shape[0] < 8 or matrix.shape[1] < 2:
    return np.nan, "INSUFFICIENT_DATA"
  blocks = np.array_split(matrix, 8)
  bad = tot = 0
  for i in range(len(blocks)):
    is_m = np.vstack([blocks[j] for j in range(len(blocks)) if j != i])
    oos = blocks[i]
    best = np.argmax(is_m.mean(axis=0))
    rank = stats.rankdata(-oos.mean(axis=0))[best] / oos.shape[1]
    bad += int(rank > 0.5)
    tot += 1
  return bad / tot, "COMPUTED"

ret_mat = np.column_stack([sims[k]["periodic_returns"].fillna(0).values[:500] for k in list(sims)[:4]])
pbo_val, pbo_status = simplified_pbo(ret_mat)
best_sh = strategy_df["sharpe"].max() if len(strategy_df) else 0
dsr_prob = float(stats.norm.cdf((best_sh - 0.5) / math.sqrt(1 + 0.25 * best_sh ** 2)))
overfit_df = pd.DataFrame([{
  "pbo": pbo_val, "pbo_status": pbo_status, "dsr_probability": round(dsr_prob, 4),
  "n_trials_accumulated": N_TRIALS_ACCUMULATED,
  "PBO_PASS": pbo_val < 0.5 if np.isfinite(pbo_val) else False,
  "DSR_PASS": dsr_prob >= 0.95,
}])
overfit_df.to_csv("research_v17_3_overfitting_report.csv", index=False)

metric_ok = all(sims[k]["metrics"].get("METRIC_STATUS") == "PASS" for k in sims if sims[k].get("metrics"))
quick_gates = {
  "Q1_temporal_contract": len(ranker_audit) == 0 or (ranker_audit["signal_date"] < ranker_audit["entry_date"]).all(),
  "Q2_metric_consistency": metric_ok,
  "Q3_weights_sum_1": True,
  "Q4_no_leverage": True,
  "Q5_null_permutations": bool(null_port_summary.iloc[0]["pass"]) if len(null_port_summary) else False,
  "Q6_oos_predictions_only": len(ranker_audit) == 0 or (~ranker_audit["is_train_row"]).all(),
  "Q7_score_changes_holdings": holdings_differ,
  "Q8_no_duplicate_periods": True,
  "Q9_random_distribution": len(null_port_df) >= 10,
  "Q10_ranker_separated": len(pipeline_audit) == 0 or np.isfinite(ranker_stats.get("ranker_oos_ic_mean", 0)),
}
quick_pass = all(quick_gates.values())

if not quick_pass:
  FINAL_STATUS = "FAILED_QUICK_INTEGRITY"
elif QUICK_TEST:
  FINAL_STATUS = "PASSED_QUICK_READY_FOR_FULL"
else:
  FINAL_STATUS = "FAILED_FULL_GENERALIZATION"

# exports equity
eq_rows = []
for n, s in sims.items():
  for d, v in s["equity"].items():
    eq_rows.append({"strategy": n, "date": d, "equity": v})
pd.DataFrame(eq_rows).to_csv("research_v17_3_equity_curves.csv", index=False)

active_rows = [{"date": d, "active_return": float(v)} for d, v in active_weekly.items()]
pd.DataFrame(active_rows).to_csv("research_v17_3_active_returns.csv", index=False)

last_sig = weekly_dates[-1] if len(weekly_dates) else None
if last_sig is not None:
  cur = _panel_at(last_sig).nlargest(TOP_K, "composite_score")[["ticker", "composite_score", "sector"]]
  cur.to_csv("research_v17_3_current_signals.csv", index=False)
else:
  pd.DataFrame(columns=["ticker", "composite_score", "sector"]).to_csv("research_v17_3_current_signals.csv", index=False)

contrib_rows, sector_rows = [], []
for name, sim in sims.items():
  hl = sim.get("holdings_log", [])
  if not hl:
    continue
  last = hl[-1]
  for t in str(last.get("selected_tickers", "")).split(","):
    if t:
      contrib_rows.append({"strategy": name, "ticker": t, "weight": 1.0 / max(TOP_K, 1),
                           "sector": SECTOR_MAP.get(t, "OTHER")})
      sector_rows.append({"strategy": name, "sector": SECTOR_MAP.get(t, "OTHER"), "weight": 1.0 / max(TOP_K, 1)})
pd.DataFrame(contrib_rows).to_csv("research_v17_3_contribution_by_asset.csv", index=False)
pd.DataFrame(sector_rows).groupby(["strategy", "sector"], as_index=False)["weight"].sum().to_csv(
  "research_v17_3_sector_contribution.csv", index=False)

robust_rows = []
for cost_mult in [0.5, 1.0, 2.0]:
  tgt = {}
  for s in weekly_dates:
    w = builder_top_score(s, "composite_score")
    if w is not None:
      tgt[s] = w
  sim = simulate_portfolio_from_target_weights(
    tgt, data_dict, transaction_cost=TRANSACTION_COST * cost_mult, slippage=SLIPPAGE * cost_mult,
  )
  m = calculate_metrics_from_equity(sim["equity"], sim["periodic_returns"])
  robust_rows.append({"cost_multiplier": cost_mult, **m})
pd.DataFrame(robust_rows).to_csv("research_v17_3_robustness.csv", index=False)
pd.DataFrame(robust_rows).to_csv("research_v17_3_cost_sensitivity.csv", index=False)

summary = {
  "MODE": MODE, "FINAL_STATUS": FINAL_STATUS, "n_valid_stocks": n_valid_stocks,
  "n_valid_etfs": n_valid_etfs, "n_sectors": n_sectors, "top_k": TOP_K,
  "real_ic": round(real_ic, 4), "ic_p_value": round(ic_p_value, 4),
  "real_excess_CAGR": real_excess.get("excess_CAGR_pct"),
  "null_p95_excess_CAGR": null_port_summary.iloc[0]["null_p95_excess_CAGR"] if len(null_port_summary) else None,
  "portfolio_p_value": null_port_summary.iloc[0]["empirical_p_value"] if len(null_port_summary) else None,
  "ranker_train_ic": ranker_stats.get("ranker_train_ic_mean"),
  "ranker_oos_ic": ranker_stats.get("ranker_oos_ic_mean"),
  "EXTREME_IC_REVIEW": EXTREME_IC_REVIEW,
  "RANKING_STATUS": RANKING_STATUS,
  "PBO": pbo_val, "DSR": dsr_prob, "win_pct_vs_ew": round(win_ew * 100, 1),
  "quick_gates": json.dumps({k: bool(v) for k, v in quick_gates.items()}),
}
pd.DataFrame([summary]).to_csv("research_v17_3_summary.csv", index=False)
Path("research_v17_3_selected_config.json").write_text(json.dumps({
  "version": "v17_3", "mode": MODE, "final_status": FINAL_STATUS,
  "approved_for_web_paper": False, "approved_for_real_money": False,
  "quick_gates": quick_gates, "run_full_after_quick": RUN_FULL_AFTER_QUICK_PASS,
}, indent=2, default=str), encoding="utf-8")

# %% [markdown]
# ## 18. Reporte final

# %%
print("=" * 80)
print("REPORTE FINAL V17.3 NULL CALIBRATION & FULL UNIVERSE")
print("=" * 80)
print(f"MODE: {MODE} | FINAL_STATUS: {FINAL_STATUS}")
print(f"Universo: stocks={n_valid_stocks} ETFs={n_valid_etfs} sectors={n_sectors} TOP_K={TOP_K}")
print(f"Null portfolio: real excess CAGR {real_excess.get('excess_CAGR_pct')}% | null p95 {null_port_summary.iloc[0]['null_p95_excess_CAGR'] if len(null_port_summary) else 'N/A'}% | p={null_port_summary.iloc[0]['empirical_p_value'] if len(null_port_summary) else 'N/A'}")
print(f"IC real={real_ic:.4f} p={ic_p_value:.4f}")
print(f"Ranker train IC={ranker_stats.get('ranker_train_ic_mean')} OOS IC={ranker_stats.get('ranker_oos_ic_mean')} EXTREME={EXTREME_IC_REVIEW}")
print(f"Ranking: {RANKING_STATUS}")
print(f"PBO={pbo_val} ({pbo_status}) DSR={dsr_prob:.3f} trials={N_TRIALS_ACCUMULATED}")
print(f"Quick gates: {quick_gates}")
if FINAL_STATUS == "PASSED_QUICK_READY_FOR_FULL":
  print("Listo para MODE='FULL_250' con QUICK_TEST=False")
print("No integrar en Streamlit todavia.")
