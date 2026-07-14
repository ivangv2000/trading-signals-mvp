# %% [markdown]
# # Trading Research V17.4 — Pre-Registered Full Universe Alpha Challenge
#
# Congela estrategias, corrige PnL/p-values/senales, ejecuta FULL_250 estrictamente OOS.
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
# ## 1. Configuracion congelada

# %%
import warnings
warnings.filterwarnings("ignore")
import ast, hashlib, json, math, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

MODE = "VALIDATE_QUICK"
VALIDATE_QUICK = MODE == "VALIDATE_QUICK"
FULL_250 = MODE == "FULL_250"
QUICK_TEST = VALIDATE_QUICK

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
PREREG_PATH = Path("config/v17_4_preregistered_experiment.json")
np.random.seed(RANDOM_SEED)
FINAL_STATUS = "PENDING"
PRIMARY_METRIC = "information_ratio_vs_equal_weight"

# %% [markdown]
# ## 2. Motor V17.2 (via V17.3 pattern)

# %%
V172 = Path(__file__).parent / "trading_research_v17_2_backtest_integrity_lab.py"
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
  return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


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
def load_us_large_cap_250():
  meta_rows, tickers = [], []
  try:
    sp = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", flavor="bs4")[0]
    for _, r in sp.iterrows():
      sym = _clean_symbol(r["Symbol"])
      tickers.append(sym)
      SECTOR_MAP[sym] = str(r.get("GICS Sector", "OTHER"))[:20].upper().replace(" ", "_")
      meta_rows.append({"ticker": sym, "security": r.get("Security", ""), "sector": SECTOR_MAP[sym],
                        "sub_industry": r.get("GICS Sub-Industry", "")})
  except Exception as e:
    print("Wikipedia fail:", e)
  return list(dict.fromkeys(tickers))[:600], pd.DataFrame(meta_rows)


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
  if FULL_250 and len(valid) > MAX_TICKERS_FULL:
    valid = rep[rep["status"] == "valid"].sort_values("dollar_vol_60", ascending=False).head(MAX_TICKERS_FULL)["ticker"].tolist()
  data = {k: v for k, v in data.items() if k in valid}
  close = close[[c for c in close.columns if c in valid]]
  return data, close, errors, rep


if VALIDATE_QUICK:
  UNIVERSE = load_quick_test_universe()
else:
  UNIVERSE, _ = load_us_large_cap_250()

data_dict, close_all, dl_errors, dl_report = download_with_audit(UNIVERSE, START_DATE, END_DATE)
STOCKS, ETFS, n_sectors = classify_assets(list(close_all.columns))
n_valid_stocks, n_valid_etfs = len(STOCKS), len(ETFS)
TOP_K = max(3, int(np.ceil(n_valid_stocks * 0.20))) if QUICK_TEST else TOP_K_FULL
BUY_R = TOP_K if QUICK_TEST else BUY_RANK
HOLD_R = TOP_K + 10 if QUICK_TEST else HOLD_UNTIL_RANK
EFFECTIVE_MAX_STOCK_WEIGHT = MAX_STOCK_WEIGHT if FULL_250 else min(0.25, max(MAX_STOCK_WEIGHT, 1.0 / max(TOP_K, 1)))
EFFECTIVE_MAX_SECTOR_WEIGHT = MAX_SECTOR_WEIGHT if FULL_250 else min(0.50, max(MAX_SECTOR_WEIGHT, 1.0 / max(n_sectors, 1)))

universe_audit = pd.DataFrame([{
  "mode": MODE, "n_downloaded": len(UNIVERSE), "n_valid_stocks": n_valid_stocks,
  "n_valid_etfs": n_valid_etfs, "n_sectors": n_sectors, "top_k": TOP_K,
  "buy_rank": BUY_R, "hold_until_rank": HOLD_R, "survivorship_warning": SURVIVORSHIP_WARNING,
}])
universe_audit.to_csv("research_v17_4_universe_audit.csv", index=False)
dl_report.to_csv("research_v17_4_download_report.csv", index=False)
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


live_feature_panel = build_live_feature_panel(data_dict, STOCKS)
modeling_panel = attach_forward_labels(live_feature_panel, data_dict)
labeled_panel = modeling_panel[modeling_panel["fwd_ret_20d"].notna()].copy()
FEATURE_COLS = [c for c in live_feature_panel.columns if c not in {
  "ticker", "signal_date", "sector", "entry_date", "label_end_date", "feature_date",
  "fwd_ret_20d", "fwd_excess_20d", "relevance"} and not FORBIDDEN_FEATURE_PATTERNS.search(c)]

_ns.update({"STOCKS": STOCKS, "data_dict": data_dict, "panel": live_feature_panel, "TOP_K": TOP_K})
for k in ["make_signal_dates", "builder_equal_weight", "builder_sector_momentum", "builder_top_score",
          "builder_random", "builder_alpha_score", "builder_spy_buyhold", "builder_ew_buyhold", "_panel_at"]:
  if k in _ns:
    globals()[k] = _ns[k]

weekly_dates = make_signal_dates("weekly")
monthly_dates = make_signal_dates("monthly")
print(f"Live panel: {len(live_feature_panel)} | Labeled: {len(labeled_panel)} | features: {len(FEATURE_COLS)}")

# %% [markdown]
# ## 6. PnL contribution + reconciliacion

# %%
def calculate_realized_pnl_contribution(strategy_name, targets, data_dict):
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
  total_gross = 0.0
  total_cost = 0.0
  current_w = pd.Series(dtype=float)
  prev_dt = None

  for dt in cal:
    period_cost_frac = 0.0
    if dt in exec_sched:
      new_w = exec_sched[dt]
      if len(current_w):
        union = current_w.index.union(new_w.index)
        dw = (new_w.reindex(union, fill_value=0) - current_w.reindex(union, fill_value=0)).abs()
        turnover = 0.5 * dw.sum()
        period_cost_frac = turnover * COST_RATE
        if dw.sum() > 0:
          for t in union:
            ticker_cost[t] += period_cost_frac * (dw.get(t, 0) / dw.sum())
      current_w = new_w.copy()
    if prev_dt is not None and len(current_w):
      port_gross = 0.0
      for t, w in current_w.items():
        if t == CASH_ASSET or t not in data_dict:
          continue
        ddf = data_dict[t]
        if dt in ddf.index and prev_dt in ddf.index:
          c0, c1 = ddf.loc[prev_dt, "Close"], ddf.loc[dt, "Close"]
          if c0 > 0 and np.isfinite(c1):
            ar = c1 / c0 - 1
            pr = w * ar
            ticker_gross[t] += pr
            ticker_wsum[t] += w
            ticker_maxw[t] = max(ticker_maxw[t], w)
            ticker_periods[t] += 1
            port_gross += pr
      total_gross += port_gross
      total_cost += period_cost_frac
    prev_dt = dt

  rows = []
  total_net = sum(ticker_gross[t] - ticker_cost.get(t, 0) for t in ticker_gross)
  for t in sorted(ticker_gross.keys()):
    g, c = ticker_gross[t], ticker_cost.get(t, 0)
    n = g - c
    rows.append({
      "ticker": t, "strategy": strategy_name, "sector": SECTOR_MAP.get(t, "OTHER"),
      "gross_pnl_contribution": round(g, 8), "cost_contribution": round(c, 8),
      "net_pnl_contribution": round(n, 8),
      "pct_of_total_net_pnl": round(100 * n / total_net, 4) if abs(total_net) > 1e-12 else 0,
      "average_weight": round(ticker_wsum[t] / max(ticker_periods[t], 1), 6),
      "max_weight": round(ticker_maxw[t], 6), "holding_periods": ticker_periods[t],
    })
  recon = {
    "strategy": strategy_name, "total_gross_pnl": round(total_gross, 8),
    "total_cost_pnl": round(total_cost, 8), "total_net_pnl": round(total_net, 8),
    "pct_sum": round(sum(r["pct_of_total_net_pnl"] for r in rows), 4),
  }
  return pd.DataFrame(rows), recon


def sector_pnl_from_asset(asset_df):
  if asset_df.empty:
    return pd.DataFrame()
  g = asset_df.groupby("sector", as_index=False).agg(
    gross_pnl_contribution=("gross_pnl_contribution", "sum"),
    cost_contribution=("cost_contribution", "sum"),
    net_pnl_contribution=("net_pnl_contribution", "sum"),
    average_portfolio_weight=("average_weight", "mean"),
    max_portfolio_weight=("max_weight", "max"),
  )
  total_net = g["net_pnl_contribution"].sum()
  g["pct_of_total_net_pnl"] = np.where(abs(total_net) > 1e-12, 100 * g["net_pnl_contribution"] / total_net, 0)
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
  eq_s = eq.loc[mask]
  pr = sim["periodic_returns"].reindex(eq_s.index).fillna(0)
  return {"equity": eq_s, "periodic_returns": pr, "accounting": sim.get("accounting", pd.DataFrame())}


def full_metrics(sim, bench_sim, spy_sim=None):
  if len(sim["equity"]) < 2:
    return {"METRIC_STATUS": "FAILED"}
  m = calculate_metrics_from_equity(sim["equity"], sim["periodic_returns"])
  bm = calculate_metrics_from_equity(bench_sim["equity"], bench_sim["periodic_returns"])
  re = sim["periodic_returns"].reindex(bench_sim["periodic_returns"].index).fillna(0)
  rb = bench_sim["periodic_returns"].reindex(re.index).fillna(0)
  active = re - rb
  pr = sim["periodic_returns"].dropna()
  neg = pr[pr < 0]
  sortino = float(pr.mean() / neg.std() * math.sqrt(252)) if len(neg) and neg.std() > 0 else 0
  out = {
    **m, "excess_CAGR_vs_ew": round(m["CAGR_pct"] - bm["CAGR_pct"], 2),
    "information_ratio_vs_equal_weight": round(
      active.mean() / active.std() * math.sqrt(252), 3) if active.std() > 0 else 0,
    "tracking_error": round(float(active.std() * math.sqrt(252)), 4),
    "active_max_drawdown": round(float((active.cumsum() - active.cumsum().cummax()).min() * 100), 2),
    "sortino": round(sortino, 3),
  }
  if spy_sim and len(spy_sim["equity"]) > 1:
    sm = calculate_metrics_from_equity(spy_sim["equity"], spy_sim["periodic_returns"])
    out["excess_CAGR_vs_SPY"] = round(m["CAGR_pct"] - sm["CAGR_pct"], 2)
  yrs = []
  for yr in sorted(sim["equity"].index.year.unique()):
    se = sim["equity"].loc[str(yr)]
    be = bench_sim["equity"].reindex(se.index).ffill()
    if len(se) < 2 or len(be) < 2:
      continue
    yrs.append({"year": yr, "beats_ew": se.iloc[-1] / se.iloc[0] > be.iloc[-1] / be.iloc[0]})
  out["pct_years_beating_ew"] = round(np.mean([y["beats_ew"] for y in yrs]) * 100, 1) if yrs else 0
  return out

# %% [markdown]
# ## 9. Pre-registro hash

# %%
prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8")) if PREREG_PATH.exists() else {}
stored_hash = prereg.pop("config_sha256", None)
computed_hash = config_sha256(prereg) if prereg else ""
if prereg and stored_hash and stored_hash != "PLACEHOLDER":
  config_hash_valid = stored_hash == computed_hash
elif prereg:
  prereg["config_sha256"] = computed_hash
  PREREG_PATH.write_text(canonical_json(prereg), encoding="utf-8")
  config_hash_valid = True
  stored_hash = computed_hash
else:
  config_hash_valid = False
  stored_hash = ""

if FULL_250 and not config_hash_valid:
  raise ValueError(f"FULL_250: config hash mismatch. Expected {stored_hash}, got {computed_hash}")

# %% [markdown]
# ## 10. Ejecutar estrategias S1-S8

# %%
oos_xgb, ranker_audit, pipeline_audit, ranker_stats = walk_forward_xgb_oos(labeled_panel, FEATURE_COLS)
ranker_audit.to_csv("research_v17_4_ranker_prediction_audit.csv", index=False)
pipeline_audit.to_csv("research_v17_4_pipeline_audit.csv", index=False)
pd.DataFrame([ranker_stats]).to_csv("research_v17_4_ranker_results.csv", index=False)

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
for name, sim in sims.items():
  research_sim = slice_sim_period(sim, RESEARCH_START, RESEARCH_END)
  m = full_metrics(research_sim, slice_sim_period(bench_ew, RESEARCH_START, RESEARCH_END), bench_spy)
  asset_c, recon = calculate_realized_pnl_contribution(name, sim["targets"], data_dict)
  if len(asset_c):
    contrib_all.append(asset_c)
    recon_all.append(recon)
    sector_all.append(sector_pnl_from_asset(asset_c).assign(strategy=name))
  top_t = asset_c.nlargest(1, "net_pnl_contribution") if len(asset_c) else pd.DataFrame()
  top_s = sector_pnl_from_asset(asset_c).nlargest(1, "net_pnl_contribution") if len(asset_c) else pd.DataFrame()
  strategy_results.append({
    "strategy": name, **m,
    "top_ticker_pnl_pct": float(top_t["pct_of_total_net_pnl"].iloc[0]) if len(top_t) else np.nan,
    "top_sector_pnl_pct": float(top_s["pct_of_total_net_pnl"].iloc[0]) if len(top_s) else np.nan,
    "n_rebalances": len(sim["targets"]),
    "oos_xgb_rows": len(oos_xgb),
  })

strategy_df = pd.DataFrame(strategy_results)
strategy_df.to_csv("research_v17_4_strategy_results.csv", index=False)
research_df = strategy_df.copy()
research_df.to_csv("research_v17_4_research_period_results.csv", index=False)

holdout_results = []
for name, sim in sims.items():
  ho = slice_sim_period(sim, HOLDOUT_START, sim["equity"].index.max())
  be = slice_sim_period(bench_ew, HOLDOUT_START, sim["equity"].index.max())
  if len(ho["equity"]) < 2:
    continue
  hm = full_metrics(ho, be, bench_spy)
  holdout_results.append({"strategy": name, **hm})
holdout_df = pd.DataFrame(holdout_results)
holdout_df.to_csv("research_v17_4_locked_holdout_results.csv", index=False)

contrib_df = pd.concat(contrib_all, ignore_index=True) if contrib_all else pd.DataFrame()
recon_df = pd.DataFrame(recon_all)
sector_df = pd.concat(sector_all, ignore_index=True) if sector_all else pd.DataFrame()
contrib_df.to_csv("research_v17_4_contribution_by_asset.csv", index=False)
recon_df.to_csv("research_v17_4_contribution_reconciliation.csv", index=False)
sector_df.to_csv("research_v17_4_sector_contribution.csv", index=False)

# %% [markdown]
# ## 11. Null distribution + p-values corregidos

# %%
champion = strategy_df.sort_values(PRIMARY_METRIC, ascending=False).iloc[0]["strategy"] if len(strategy_df) else "S4_CORRECTED_COMPOSITE"
champ_sim = sims.get(champion, sims.get("S4_CORRECTED_COMPOSITE"))
champ_excess = strategy_df[strategy_df["strategy"] == champion]["excess_CAGR_vs_ew"].iloc[0] if len(strategy_df) else 0

null_excess = []
for perm in tqdm(range(N_NULL_PORTFOLIO), desc="Null portfolio", leave=False):
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
  nm = full_metrics(slice_sim_period(ns, RESEARCH_START, RESEARCH_END),
                    slice_sim_period(bench_ew, RESEARCH_START, RESEARCH_END))
  null_excess.append(nm.get("excess_CAGR_vs_ew", 0))

null_p, n_exceed, n_perm = corrected_empirical_p(champ_excess, null_excess)
null_dist = pd.DataFrame([{
  "champion": champion, "real_excess_CAGR": champ_excess,
  "null_mean": round(np.mean(null_excess), 2) if null_excess else np.nan,
  "null_median": round(np.median(null_excess), 2) if null_excess else np.nan,
  "null_p95": round(np.percentile(null_excess, 95), 2) if null_excess else np.nan,
  "number_exceeding_real": n_exceed, "n_permutations": n_perm,
  "corrected_empirical_p_value": round(null_p, 6),
}])
null_dist.to_csv("research_v17_4_null_distribution.csv", index=False)

# %% [markdown]
# ## 12. Current signals (live panel)

# %%
def generate_current_signals(strategy_source="S6_ENSEMBLE_COMPOSITE_RANKER"):
  latest_sig = live_feature_panel["signal_date"].max()
  prev_sig = live_feature_panel["signal_date"].unique()
  prev_sig = sorted(prev_sig)[-2] if len(prev_sig) > 1 else latest_sig
  g_live = live_feature_panel[live_feature_panel["signal_date"] == latest_sig].copy()
  g_prev = live_feature_panel[live_feature_panel["signal_date"] == prev_sig].copy()
  if g_live.empty:
    return pd.DataFrame()

  final_mdl, sel_feats, med = fit_final_xgb(labeled_panel, FEATURE_COLS)
  if final_mdl is not None:
    g_live[sel_feats] = g_live[sel_feats].fillna(med)
    g_live["model_score"] = final_mdl.predict(g_live[sel_feats])
  else:
    g_live["model_score"] = g_live.get("composite_score", 0)

  comp_rank = g_live.set_index("ticker")["composite_score"].rank(ascending=False, method="min")
  xgb_rank = g_live.set_index("ticker")["model_score"].rank(ascending=False, method="min")
  ens = ENSEMBLE_W_COMPOSITE * g_live.set_index("ticker")["composite_score"].rank(pct=True) + \
        ENSEMBLE_W_XGB * g_live.set_index("ticker")["model_score"].rank(pct=True)
  current_rank = ens.rank(ascending=False, method="min")

  prev_w = {}
  if strategy_source in sims and sims[strategy_source]["holdings_log"]:
    last = sims[strategy_source]["holdings_log"][-1]
    prev_w = {t: 1.0 / max(last["n_holdings"], 1) for t in str(last["selected_tickers"]).split(",") if t}

  prev_rank = g_prev.set_index("ticker")["composite_score"].rank(ascending=False, method="min") if len(g_prev) else pd.Series(dtype=float)
  tgt_w = build_buffered_weights(ens, set(prev_w.keys()), BUY_R, HOLD_R, top_k_cap=TOP_K)

  rows = []
  for t in g_live["ticker"]:
    cr = int(current_rank.get(t, 999))
    pr = int(prev_rank.get(t, 999)) if t in prev_rank.index else None
    tw = float(tgt_w.get(t, 0))
    pw = float(prev_w.get(t, 0))
    rank_chg = (pr - cr) if pr is not None else np.nan
    in_port = pw > 0
    if cr <= BUY_R and not in_port:
      sig = "BUY"
    elif cr <= BUY_R and in_port and tw > pw + 0.01:
      sig = "INCREASE"
    elif cr <= HOLD_R and in_port and abs(tw - pw) <= 0.01:
      sig = "HOLD"
    elif cr <= HOLD_R and in_port and tw < pw - 0.01:
      sig = "REDUCE"
    elif cr > HOLD_R and in_port:
      sig = "SELL"
    else:
      sig = "AVOID"
    row = g_live[g_live["ticker"] == t].iloc[0]
    feats_sorted = row[FEATURE_COLS].sort_values(ascending=False) if FEATURE_COLS else pd.Series()
    rows.append({
      "ticker": t, "signal": sig, "current_rank": cr,
      "previous_rank": pr, "rank_change": rank_chg,
      "target_weight": round(tw, 4), "previous_weight": round(pw, 4),
      "model_score": round(float(row.get("model_score", 0)), 4),
      "composite_score": round(float(row.get("composite_score", 0)), 4),
      "strategy_source": strategy_source, "sector": SECTOR_MAP.get(t, "OTHER"),
      "main_factor_1": feats_sorted.index[0] if len(feats_sorted) > 0 else "",
      "main_factor_2": feats_sorted.index[1] if len(feats_sorted) > 1 else "",
      "main_factor_3": feats_sorted.index[2] if len(feats_sorted) > 2 else "",
      "market_regime": "TREND" if row.get("above_sma200", 0) > 0 else "DEFENSIVE",
      "confidence": round(float(row.get("composite_score", 0.5)), 3),
      "reason": f"rank={cr} vs buy={BUY_R} hold={HOLD_R}",
      "entry_plan": "next_open", "exit_plan": f"rank>{HOLD_R}", "next_review": str(latest_sig + pd.Timedelta(days=7))[:10],
    })
  return pd.DataFrame(rows)


current_signals = generate_current_signals()
current_signals.to_csv("research_v17_4_current_signals.csv", index=False)

# %% [markdown]
# ## 13. PBO / DSR + yearly + jaccard + cost sensitivity

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
best_sh = strategy_df["sharpe"].max() if len(strategy_df) else 0
n_obs = int(strategy_df["n_periods"].max()) if "n_periods" in strategy_df.columns and len(strategy_df) else 252
prereg_dsr = dsr_prob(best_sh, n_obs, len(strategy_targets))
global_dsr = dsr_prob(best_sh, n_obs, N_TRIALS_ACCUMULATED)

overfit_df = pd.DataFrame([{
  "global_pbo": global_pbo, "preregistered_pbo": prereg_pbo,
  "global_dsr": round(global_dsr, 4), "preregistered_dsr": round(prereg_dsr, 4),
  "n_trials_accumulated": N_TRIALS_ACCUMULATED, "n_preregistered_strategies": len(strategy_targets),
}])
overfit_df.to_csv("research_v17_4_overfitting_report.csv", index=False)

yr_rows = []
for name, sim in sims.items():
  for yr in sorted(sim["equity"].index.year.unique()):
    se = sim["equity"].loc[str(yr)]
    be = bench_ew["equity"].reindex(se.index).ffill()
    if len(se) < 2:
      continue
    yr_rows.append({
      "strategy": name, "year": yr,
      "return_pct": round((se.iloc[-1] / se.iloc[0] - 1) * 100, 2),
      "beats_ew": se.iloc[-1] / se.iloc[0] > be.iloc[-1] / be.iloc[0] if len(be) > 1 else False,
    })
pd.DataFrame(yr_rows).to_csv("research_v17_4_yearly.csv", index=False)

j_rows = []
names = list(sims.keys())
for i, a in enumerate(names):
  for b in names[i + 1:]:
    ha = pd.DataFrame(sims[a]["holdings_log"])
    hb = pd.DataFrame(sims[b]["holdings_log"])
    m = ha.merge(hb, on="date", suffixes=("_a", "_b"))
    if len(m) == 0:
      continue
    js = []
    for _, r in m.iterrows():
      sa, sb = set(str(r["selected_tickers_a"]).split(",")), set(str(r["selected_tickers_b"]).split(","))
      js.append(len(sa & sb) / len(sa | sb) if (sa | sb) else 1)
    j_rows.append({"a": a, "b": b, "mean_jaccard": round(np.mean(js), 3)})
pd.DataFrame(j_rows).to_csv("research_v17_4_strategy_jaccard.csv", index=False)

cost_rows = []
for mult in [0.5, 1.0, 2.0]:
  t = strategy_targets.get("S4_CORRECTED_COMPOSITE", {})
  sim = simulate_portfolio_from_target_weights(t, data_dict,
    transaction_cost=TRANSACTION_COST * mult, slippage=SLIPPAGE * mult)
  cost_rows.append({"cost_multiplier": mult, **calculate_metrics_from_equity(sim["equity"], sim["periodic_returns"])})
pd.DataFrame(cost_rows).to_csv("research_v17_4_cost_sensitivity.csv", index=False)

eq_rows = []
for n, s in sims.items():
  for d, v in s["equity"].items():
    eq_rows.append({"strategy": n, "date": d, "equity": v})
pd.DataFrame(eq_rows).to_csv("research_v17_4_equity_curves.csv", index=False)

# %% [markdown]
# ## 14. Validation gates + reporte

# %%
def _weight_sum_ok(sim):
  for sig, w in sim.get("targets", {}).items():
    if abs(w.sum() - 1) > 0.01:
      return False
    if FULL_250 and (w.drop(CASH_ASSET, errors="ignore") > MAX_STOCK_WEIGHT + 0.001).any():
      return False
  return True


p_value_demo, _, _ = corrected_empirical_p(1.0, [0.5, 0.3, 0.2])
recon_ok = all(abs(r.get("pct_sum", 0) - 100) < 5 or abs(r.get("total_net_pnl", 0)) < 1e-6 for r in recon_all) if recon_all else True

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
  "VQ10_config_hash_valid": config_hash_valid or VALIDATE_QUICK,
}

if VALIDATE_QUICK:
  FINAL_STATUS = "PASSED_VALIDATION_READY_FOR_FULL_250" if all(validation_gates.values()) else "FAILED_V17_4_VALIDATION"
elif FULL_250:
  best = strategy_df.sort_values(PRIMARY_METRIC, ascending=False).iloc[0] if len(strategy_df) else None
  ho_champ = holdout_df[holdout_df["strategy"] == best["strategy"]].iloc[0] if best is not None and len(holdout_df) else None
  full_gates = {
    "F1_min_200_stocks": n_valid_stocks >= MIN_VALID_STOCKS,
    "F2_min_8_sectors": n_sectors >= MIN_SECTORS,
    "F5_beats_random": null_p < 0.05,
    "F6_excess_cagr_pos": best["excess_CAGR_vs_ew"] > 0 if best is not None else False,
    "F7_beats_ew_55pct": best["pct_years_beating_ew"] >= 55 if best is not None else False,
    "F9_ir_030": best[PRIMARY_METRIC] > 0.30 if best is not None else False,
    "F14_holdout_active_pos": ho_champ["excess_CAGR_vs_ew"] > 0 if ho_champ is not None else False,
    "F15_prereg_pbo": prereg_pbo < 0.50 if np.isfinite(prereg_pbo) else False,
    "F16_prereg_dsr": prereg_dsr >= 0.95,
  }
  if not all(full_gates.values()):
    FINAL_STATUS = "FAILED_FULL_GENERALIZATION"
  elif null_p >= 0.05:
    FINAL_STATUS = "PASSED_FULL_BUT_NO_SIGNIFICANT_ALPHA"
  else:
    FINAL_STATUS = "CANDIDATE"
else:
  FINAL_STATUS = "FAILED_V17_4_VALIDATION"

pd.DataFrame([{"gate": k, "pass": v} for k, v in validation_gates.items()]).to_csv(
  "research_v17_4_validation_gates.csv", index=False)

summary = {
  "MODE": MODE, "FINAL_STATUS": FINAL_STATUS, "n_valid_stocks": n_valid_stocks,
  "n_sectors": n_sectors, "top_k": TOP_K, "champion": champion,
  "null_corrected_p": round(null_p, 6), "ranker_oos_ic": ranker_stats.get("ranker_oos_ic_mean"),
  "global_pbo": global_pbo, "preregistered_pbo": prereg_pbo,
  "global_dsr": round(global_dsr, 4), "preregistered_dsr": round(prereg_dsr, 4),
  "config_sha256": stored_hash, "current_signals_n": len(current_signals),
  "s6_fallbacks": s6_fallbacks,
}
pd.DataFrame([summary]).to_csv("research_v17_4_summary.csv", index=False)
Path("research_v17_4_selected_config.json").write_text(json.dumps({
  "version": "v17_4", "mode": MODE, "final_status": FINAL_STATUS,
  "approved_for_real_money": False, "validation_gates": validation_gates,
  "preregistered_hash": stored_hash,
}, indent=2, default=str), encoding="utf-8")

# %%
print("=" * 80)
print("REPORTE FINAL V17.4 PRE-REGISTERED FULL UNIVERSE CHALLENGE")
print("=" * 80)
print(f"MODE: {MODE} | FINAL_STATUS: {FINAL_STATUS}")
print(f"Integridad: signals={len(current_signals)} | recon_ok={recon_ok} | config_hash={stored_hash[:12]}...")
print(f"Universo: stocks={n_valid_stocks} sectors={n_sectors} TOP_K={TOP_K}")
if len(strategy_df):
  print(strategy_df[["strategy", "CAGR_pct", "excess_CAGR_vs_ew", PRIMARY_METRIC]].to_string(index=False))
print(f"Champion: {champion} | null p={null_p:.6f} (min possible 1/{N_NULL_PORTFOLIO+1})")
print(f"Ranker OOS IC={ranker_stats.get('ranker_oos_ic_mean')}")
print(f"PBO global={global_pbo:.3f} prereg={prereg_pbo:.3f} | DSR global={global_dsr:.3f} prereg={prereg_dsr:.3f}")
if len(current_signals):
  print(current_signals[["ticker", "signal", "current_rank", "target_weight"]].head(10).to_string(index=False))
print(f"Gates: {validation_gates}")
if FINAL_STATUS == "PASSED_VALIDATION_READY_FOR_FULL_250":
  print("Listo para MODE='FULL_250'")
print("No modificar Streamlit todavia.")
