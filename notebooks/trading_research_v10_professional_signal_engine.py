# %% [markdown]
# # Trading Research V10 Professional Signal Engine
#
# Motor predictivo por acción: triple barrier, meta-labeling, ranking cross-sectional.
#
# **Disclaimer:** Backtest no garantiza resultados futuros. No es asesoramiento financiero.
# **No conecta broker. No ejecuta órdenes. APPROVED_FOR_REAL_MONEY siempre False.**

# %%
!pip install yfinance pandas numpy matplotlib plotly tqdm scipy scikit-learn -q

# %% [markdown]
# ## 1. Configuracion

# %%
import warnings
warnings.filterwarnings("ignore")

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tqdm.auto import tqdm

try:
  from sklearn.pipeline import Pipeline, make_pipeline
  from sklearn.impute import SimpleImputer
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier,
  )
  from sklearn.metrics import confusion_matrix
  HAS_SKLEARN = True
except Exception:
  HAS_SKLEARN = False

QUICK_TEST = True
START_DATE = "2015-01-01"
END_DATE = None
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001
SLIPPAGE = 0.001
RUN_ML = True
RUN_META_LABELING = True
MAX_CONCURRENT_POSITIONS = 5
POSITION_SIZE_PCT = 0.10
HORIZONS = [5, 10, 20]
TAKE_PROFIT_ATR = [1.5, 2.0, 2.5]
STOP_LOSS_ATR = [1.0, 1.5, 2.0]
WF_START_YEAR = 2019
EMBARGO_DAYS = 20
BUY_THRESHOLD = 0.60
STRONG_BUY_THRESHOLD = 0.70
THRESHOLD_GRID = [0.55, 0.60, 0.65, 0.70]
ML_HORIZON = 10
DEFAULT_TP_ATR = 2.0
DEFAULT_SL_ATR = 1.5
RANDOM_SEED = 42
MARKET = "SPY"

UNIVERSE = [
  "SPY", "QQQ", "IWM",
  "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "INTC", "MU",
  "GOOGL", "META", "AMZN", "NFLX", "ADBE", "CRM", "ORCL",
  "JPM", "BAC", "GS", "MS",
  "XOM", "CVX",
  "UNH", "LLY", "JNJ",
  "WMT", "COST", "HD", "MCD",
  "PANW", "NOW", "SNOW", "SHOP", "UBER",
]

SECTOR_MAP = {
  "AAPL": "mega_tech", "MSFT": "mega_tech", "NVDA": "semis", "AMD": "semis",
  "AVGO": "semis", "INTC": "semis", "MU": "semis", "GOOGL": "internet",
  "META": "internet", "AMZN": "internet", "NFLX": "internet", "ADBE": "software",
  "CRM": "software", "ORCL": "software", "JPM": "banks", "BAC": "banks",
  "GS": "banks", "MS": "banks", "XOM": "energy", "CVX": "energy",
  "UNH": "healthcare", "LLY": "healthcare", "JNJ": "healthcare",
  "WMT": "consumer", "COST": "consumer", "HD": "consumer", "MCD": "consumer",
  "PANW": "software", "NOW": "software", "SNOW": "software",
  "SHOP": "internet", "UBER": "internet",
}

if QUICK_TEST:
  UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META", "AMZN", "JPM"]
  START_DATE = "2020-01-01"
  WF_START_YEAR = 2021
  HORIZONS = [5, 10]
  TAKE_PROFIT_ATR = [2.0]
  STOP_LOSS_ATR = [1.5]
  print("QUICK_TEST activo")

print("V10 Professional Signal Engine | desde", START_DATE)
print("Backtest no garantiza resultados futuros.")

# %% [markdown]
# ## 2. Descarga de datos

# %%
def download_data(tickers, start, end=None):
  data, failed = {}, []
  for ticker in tqdm(sorted(set(tickers)), desc="Download"):
    try:
      raw = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
      if raw is None or raw.empty:
        failed.append(ticker)
        continue
      df = raw.copy()
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
      colmap = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
      df = df.rename(columns={c: colmap.get(str(c).lower(), c) for c in df.columns})
      keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
      df = df[keep].dropna(subset=["Close"])
      if df.empty:
        failed.append(ticker)
        continue
      df.index = pd.DatetimeIndex(df.index)
      if df.index.tz:
        df.index = df.index.tz_localize(None)
      data[ticker.upper()] = df.sort_index()
    except Exception as exc:
      failed.append(f"{ticker}({exc})")
  if failed:
    print("Fallidos:", failed[:15])
  close = pd.DataFrame({k: v["Close"] for k, v in data.items()}).sort_index().ffill()
  close.index = pd.DatetimeIndex(close.index)
  if close.index.tz:
    close.index = close.index.tz_localize(None)
  return data, close


data_dict, close_prices = download_data(UNIVERSE, START_DATE, END_DATE)
print("Tickers:", len(data_dict), "| dias:", len(close_prices))

# %% [markdown]
# ## 3. Features profesionales

# %%
def _sf(x, default=np.nan):
  try:
    if x is None or (isinstance(x, float) and np.isnan(x)):
      return default
    if isinstance(x, pd.Series):
      x = x.iloc[0] if len(x) else default
    v = float(x)
    return default if not np.isfinite(v) else v
  except Exception:
    return default


def _rsi(close, window):
  delta = close.diff()
  gain = delta.clip(lower=0).rolling(window).mean()
  loss = (-delta.clip(upper=0)).rolling(window).mean()
  return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _atr(df, window=14):
  h, l, c = df["High"], df["Low"], df["Close"]
  tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
  return tr.rolling(window).mean()


def _single_ticker_features(df):
  d = df.copy()
  c, h, l, v = d["Close"], d["High"], d["Low"], d.get("Volume", pd.Series(0, index=d.index))
  r1 = c.pct_change(1)
  for n in [1, 3, 5, 10, 20, 60, 120]:
    d[f"RET_{n}D"] = r1 if n == 1 else c.pct_change(n)
  for n in [20, 60, 120]:
    d[f"MOM_{n}"] = c.pct_change(n)
  d["MOM_20_60_COMBO"] = d["MOM_20"] - d["MOM_60"]
  d["MOM_60_120_COMBO"] = d["MOM_60"] - d["MOM_120"]
  d["MOM_SKIP_5"] = c.shift(5) / c.shift(65) - 1
  for n in [20, 50, 100, 200]:
    d[f"SMA_{n}"] = c.rolling(n).mean()
  for span in [9, 21, 50]:
    d[f"EMA_{span}"] = c.ewm(span=span, adjust=False).mean()
  d["DIST_SMA20"] = c / d["SMA_20"] - 1
  d["DIST_SMA50"] = c / d["SMA_50"] - 1
  d["DIST_SMA200"] = c / d["SMA_200"] - 1
  d["TREND_STACK"] = ((c > d["SMA_50"]) & (d["SMA_50"] > d["SMA_200"])).astype(float)
  for n in [10, 20, 60]:
    d[f"VOL_{n}"] = r1.rolling(n).std() * np.sqrt(252)
  d["ATR_14"] = _atr(d, 14)
  d["ATR_PCT"] = d["ATR_14"] / c.replace(0, np.nan)
  d["VOL_RATIO_20_60"] = d["VOL_20"] / d["VOL_60"].replace(0, np.nan)
  roll_max = c.rolling(20).max()
  d["DD_20"] = c / roll_max - 1
  d["DD_60"] = c / c.rolling(60).max() - 1
  d["distance_from_high_20"] = c / h.rolling(20).max() - 1
  d["distance_from_high_60"] = c / h.rolling(60).max() - 1
  for w in [2, 7, 14]:
    d[f"RSI_{w}"] = _rsi(c, w)
  hl = (h - l).replace(0, np.nan)
  d["IBS"] = (c - l) / hl
  for n in [20, 60]:
    d[f"HIGH_{n}_PREV"] = h.rolling(n).max().shift(1)
    d[f"LOW_{n}_PREV"] = l.rolling(n).min().shift(1)
  d["BREAKOUT_20"] = c / d["HIGH_20_PREV"] - 1
  d["BREAKOUT_60"] = c / d["HIGH_60_PREV"] - 1
  d["VOLUME_AVG_20"] = v.rolling(20).mean()
  d["VOLUME_RATIO"] = v / d["VOLUME_AVG_20"].replace(0, np.nan)
  return d


def _market_features(data_dict):
  out = {}
  for mkt, prefix in [(MARKET, "SPY"), ("QQQ", "QQQ")]:
    df = data_dict.get(mkt)
    if df is None:
      continue
    f = _single_ticker_features(df.copy())
    out[f"{prefix}_RET_20"] = f["RET_20D"]
    out[f"{prefix}_RET_60"] = f["RET_60D"]
    out[f"{prefix}_ABOVE_SMA200"] = (f["Close"] > f["SMA_200"]).astype(float)
    out[f"{prefix}_VOL_20"] = f["VOL_20"]
  if out:
    mdf = pd.DataFrame(out)
    mdf["MARKET_REGIME_SCORE"] = (
      mdf.get("SPY_ABOVE_SMA200", 0) * 0.5
      + (mdf.get("SPY_RET_20", 0) > 0).astype(float) * 0.25
      + (mdf.get("QQQ_ABOVE_SMA200", 0) * 0.25 if "QQQ_ABOVE_SMA200" in mdf else 0)
    )
    return mdf
  return pd.DataFrame()


def add_cross_sectional(features_by_ticker, sector_map):
  tickers = [t for t in features_by_ticker if t not in ("SPY", "QQQ", "IWM")]
  if not tickers:
    return features_by_ticker
  sample = features_by_ticker[tickers[0]]
  dates = sample.index
  ret20 = pd.DataFrame({t: features_by_ticker[t]["RET_20D"] for t in tickers}, index=dates)
  ret60 = pd.DataFrame({t: features_by_ticker[t]["RET_60D"] for t in tickers}, index=dates)
  mom_vol = pd.DataFrame({
    t: features_by_ticker[t]["MOM_60"] / features_by_ticker[t]["VOL_20"].replace(0, np.nan)
    for t in tickers
  }, index=dates)
  rank_ret20 = ret20.rank(axis=1, pct=True)
  rank_ret60 = ret60.rank(axis=1, pct=True)
  rank_mom_vol = mom_vol.rank(axis=1, pct=True)
  sector_ret20, sector_ret60, sector_rank = {}, {}, {}
  for dt in dates:
    for t in tickers:
      sec = sector_map.get(t, "other")
      peers = [p for p in tickers if sector_map.get(p) == sec]
      if not peers:
        peers = tickers
      s20 = ret20.loc[dt, peers].mean() if dt in ret20.index else np.nan
      s60 = ret60.loc[dt, peers].mean() if dt in ret60.index else np.nan
      sector_ret20[(dt, t)] = ret20.loc[dt, t] - s20 if dt in ret20.index else np.nan
      sector_ret60[(dt, t)] = ret60.loc[dt, t] - s60 if dt in ret60.index else np.nan
      pr = ret60.loc[dt, peers].rank(pct=True) if dt in ret60.index else pd.Series()
      sector_rank[(dt, t)] = pr.get(t, np.nan) if len(pr) else np.nan
  for t in tickers:
    idx = features_by_ticker[t].index
    features_by_ticker[t]["rank_ret_20"] = rank_ret20[t].reindex(idx)
    features_by_ticker[t]["rank_ret_60"] = rank_ret60[t].reindex(idx)
    features_by_ticker[t]["rank_mom_vol"] = rank_mom_vol[t].reindex(idx)
    features_by_ticker[t]["sector_relative_ret_20"] = pd.Series(
      {dt: sector_ret20.get((dt, t), np.nan) for dt in idx}, index=idx
    )
    features_by_ticker[t]["sector_relative_ret_60"] = pd.Series(
      {dt: sector_ret60.get((dt, t), np.nan) for dt in idx}, index=idx
    )
    features_by_ticker[t]["rank_within_sector"] = pd.Series(
      {dt: sector_rank.get((dt, t), np.nan) for dt in idx}, index=idx
    )
    if MARKET in features_by_ticker:
      spy20 = features_by_ticker[MARKET]["RET_20D"].reindex(idx)
      features_by_ticker[t]["RES_MOM_20"] = features_by_ticker[t]["RET_20D"] - spy20
      features_by_ticker[t]["RES_MOM_60"] = features_by_ticker[t]["RET_60D"] - features_by_ticker[MARKET]["RET_60D"].reindex(idx)
  return features_by_ticker


def add_features(data_dict, close_prices, sector_map=None):
  sector_map = sector_map or SECTOR_MAP
  market_feats = _market_features(data_dict)
  features = {}
  for ticker, df in data_dict.items():
    d = _single_ticker_features(df.copy())
    if len(market_feats):
      for col in market_feats.columns:
        d[col] = market_feats[col].reindex(d.index).ffill()
    features[ticker] = d
  features = add_cross_sectional(features, sector_map)
  return features


features_by_ticker = add_features(data_dict, close_prices, SECTOR_MAP)
print("Features para", len(features_by_ticker), "tickers")

# %% [markdown]
# ## 4. Triple barrier labeling (solo train)

# %%
def create_triple_barrier_labels(df, horizon, take_profit_atr, stop_loss_atr):
  d = df.copy()
  c = d["Close"]
  h, l = d["High"], d["Low"]
  atr = d.get("ATR_14", c * 0.02).fillna(c * 0.02)
  labels, future_rets, exit_days, exit_reasons = [], [], [], []
  idx_list = list(d.index)
  for i, dt in enumerate(idx_list):
    entry = _sf(c.loc[dt])
    if not np.isfinite(entry) or entry <= 0:
      labels.append(0)
      future_rets.append(np.nan)
      exit_days.append(np.nan)
      exit_reasons.append("invalid")
      continue
    a = _sf(atr.loc[dt], entry * 0.02)
    upper = entry + take_profit_atr * a
    lower = entry - stop_loss_atr * a
    label, fut_ret, ex_day, ex_reason = 0, np.nan, horizon, "vertical"
    for j in range(1, min(horizon + 1, len(idx_list) - i)):
      fut_dt = idx_list[i + j]
      hi = _sf(h.loc[fut_dt], entry)
      lo = _sf(l.loc[fut_dt], entry)
      if hi >= upper:
        label, fut_ret, ex_day, ex_reason = 1, upper / entry - 1, j, "take_profit"
        break
      if lo <= lower:
        label, fut_ret, ex_day, ex_reason = -1, lower / entry - 1, j, "stop_loss"
        break
    else:
      end_i = min(i + horizon, len(idx_list) - 1)
      end_px = _sf(c.iloc[end_i], entry)
      fut_ret = end_px / entry - 1
      ex_day = end_i - i
    labels.append(label)
    future_rets.append(fut_ret)
    exit_days.append(ex_day)
    exit_reasons.append(ex_reason)
  prefix = f"tb_{horizon}d"
  d[f"{prefix}_label"] = labels
  d[f"{prefix}_future_return"] = future_rets
  d[f"{prefix}_exit_day"] = exit_days
  d[f"{prefix}_exit_reason"] = exit_reasons
  d[f"target_buy_{horizon}d"] = (np.array(labels) == 1).astype(int)
  return d


for ticker in features_by_ticker:
  for h in HORIZONS:
    features_by_ticker[ticker] = create_triple_barrier_labels(
      features_by_ticker[ticker], h, DEFAULT_TP_ATR, DEFAULT_SL_ATR
    )

# %% [markdown]
# ## 5. Eventos candidatos

# %%
def classify_candidate_event(row):
  types = []
  c = _sf(row.get("Close"))
  if (
    c > _sf(row.get("HIGH_20_PREV"), -1)
    and _sf(row.get("MOM_20"), -1) > 0
    and _sf(row.get("VOLUME_RATIO"), 0) > 0.8
  ):
    types.append("momentum_breakout")
  if (
    c > _sf(row.get("SMA_200"), 0)
    and abs(c - _sf(row.get("EMA_21"), c)) / max(c, 1) < 0.04
    and 35 <= _sf(row.get("RSI_14"), 0) <= 60
  ):
    types.append("pullback_in_trend")
  if (
    _sf(row.get("rank_ret_60"), 0) >= 0.75
    and c > _sf(row.get("SMA_100"), 0)
    and _sf(row.get("SPY_ABOVE_SMA200"), 0) >= 0.5
  ):
    types.append("cross_sectional_strength")
  if (
    c > _sf(row.get("SMA_200"), 0)
    and _sf(row.get("RSI_2"), 50) < 15
    and _sf(row.get("IBS"), 1) < 0.25
    and _sf(row.get("SPY_ABOVE_SMA200"), 0) >= 0.5
  ):
    types.append("mean_reversion_quality")
  if not types:
    return False, ""
  return True, types[0]


def add_candidate_flags(features_by_ticker):
  for ticker, df in features_by_ticker.items():
    if ticker in ("SPY", "QQQ", "IWM"):
      df["candidate_event"] = False
      df["setup_type"] = ""
      continue
    flags, setups = [], []
    for dt, row in df.iterrows():
      ok, st = classify_candidate_event(row)
      flags.append(ok)
      setups.append(st)
    df["candidate_event"] = flags
    df["setup_type"] = setups
    features_by_ticker[ticker] = df
  return features_by_ticker


features_by_ticker = add_candidate_flags(features_by_ticker)

# %% [markdown]
# ## 6. Dataset ML (meta-labeling)

# %%
LABEL_PREFIXES = ("tb_", "target_buy_", "forward_", "barrier_", "future_", "exit_")


def get_feature_cols(df):
  exclude = {"Open", "High", "Low", "Close", "Volume", "candidate_event", "setup_type"}
  cols = []
  for c in df.columns:
    if c in exclude:
      continue
    if any(c.startswith(p) for p in LABEL_PREFIXES):
      continue
    if pd.api.types.is_numeric_dtype(df[c]):
      cols.append(c)
  return cols


def build_ml_dataset(features_by_ticker, tickers=None, horizon=ML_HORIZON):
  rows = []
  tickers = tickers or [t for t in features_by_ticker if t not in ("SPY", "QQQ", "IWM")]
  target = f"target_buy_{horizon}d"
  tb_label = f"tb_{horizon}d_label"
  tb_ret = f"tb_{horizon}d_future_return"
  tb_reason = f"tb_{horizon}d_exit_reason"
  for ticker in tickers:
    df = features_by_ticker.get(ticker)
    if df is None:
      continue
    sub = df[df["candidate_event"] == True]  # noqa: E712
    for dt, row in sub.iterrows():
      rec = row.to_dict()
      rec["ticker"] = ticker
      rec["date"] = dt
      rec["sector"] = SECTOR_MAP.get(ticker, "other")
      rec["true_label"] = int(_sf(row.get(target), 0))
      rec["triple_barrier_label"] = int(_sf(row.get(tb_label), 0))
      rec["forward_return"] = _sf(row.get(tb_ret))
      rec["exit_reason"] = row.get(tb_reason, "")
      rows.append(rec)
  panel = pd.DataFrame(rows)
  if len(panel):
    panel["date"] = pd.to_datetime(panel["date"])
    sector_dummies = pd.get_dummies(panel["sector"], prefix="sec")
    panel = pd.concat([panel, sector_dummies], axis=1)
  return panel


panel_df = build_ml_dataset(features_by_ticker)
FEATURE_COLS = get_feature_cols(features_by_ticker.get("AAPL", pd.DataFrame()))
FEATURE_COLS = [c for c in FEATURE_COLS if c in panel_df.columns] if len(panel_df) else FEATURE_COLS
print("Panel candidatos:", len(panel_df), "| features:", len(FEATURE_COLS))

# %% [markdown]
# ## 7. Meta-labeling
#
# El modelo actua como filtro de calidad sobre setups tecnicos, no como oraculo.
# Primero detectamos candidate_event (setup base). Luego el ML predice si ese setup
# merece tomarse segun triple barrier en train.

# %%
def clean_ml_matrix(X, feature_cols):
  X = X.copy()
  for col in feature_cols:
    if col not in X.columns:
      X[col] = 0.0
  X = X[feature_cols]
  for col in feature_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")
  return X.replace([np.inf, -np.inf], np.nan)


def build_model_pipelines():
  if not HAS_SKLEARN:
    return []
  return [
    ("logreg", make_pipeline(
      SimpleImputer(strategy="median"), StandardScaler(),
      LogisticRegression(max_iter=800, class_weight="balanced", random_state=RANDOM_SEED),
    )),
    ("rf", make_pipeline(
      SimpleImputer(strategy="median"),
      RandomForestClassifier(n_estimators=120, max_depth=6, class_weight="balanced_subsample", random_state=RANDOM_SEED, n_jobs=-1),
    )),
    ("hgb", make_pipeline(
      SimpleImputer(strategy="median"),
      HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, max_iter=150, random_state=RANDOM_SEED),
    )),
    ("et", make_pipeline(
      SimpleImputer(strategy="median"),
      ExtraTreesClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
    )),
  ]


def fit_ensemble_predict(train_df, test_df, feature_cols):
  if not HAS_SKLEARN or len(train_df) < 80 or len(test_df) == 0:
    return pd.DataFrame(), None
  target = f"target_buy_{ML_HORIZON}d"
  if target not in train_df.columns:
    target = "true_label"
  X = clean_ml_matrix(train_df, feature_cols)
  y = pd.to_numeric(train_df[target], errors="coerce").fillna(0).astype(int)
  Xt = clean_ml_matrix(test_df, feature_cols)
  probs, model_scores = [], {}
  last_rf = None
  for name, model in build_model_pipelines():
    try:
      model.fit(X, y)
      if hasattr(model, "predict_proba"):
        p = model.predict_proba(Xt)[:, 1]
      else:
        p = model.predict(Xt)
      probs.append(p)
      model_scores[name] = p
      if name == "rf":
        last_rf = model
    except Exception:
      pass
  if not probs:
    return pd.DataFrame(), None
  out = test_df[["ticker", "date", "setup_type", "true_label", "forward_return"]].copy()
  out["prob_buy"] = np.mean(probs, axis=0)
  out["model_scores"] = [json.dumps({k: round(float(v[i]), 4) for k, v in model_scores.items()}) for i in range(len(out))]
  out["candidate_type"] = out["setup_type"]
  return out, last_rf

# %% [markdown]
# ## 8. Walk-forward ML con embargo

# %%
def select_threshold_on_validation(val_preds, thresholds=None):
  thresholds = thresholds or THRESHOLD_GRID
  best_th, best_score = BUY_THRESHOLD, -999
  for th in thresholds:
    sub = val_preds[val_preds["prob_buy"] >= th]
    if len(sub) < 15:
      continue
    wins = sub[sub["true_label"] == 1]["forward_return"]
    losses = sub[sub["true_label"] != 1]["forward_return"]
    gw = wins.clip(lower=0).sum() if len(wins) else 0
    gl = abs(losses.clip(upper=0).sum()) if len(losses) else 0
    pf = gw / gl if gl > 0 else 0
    score = pf + sub["true_label"].mean() * 0.5 - (0.1 if len(sub) < 30 else 0)
    if score > best_score:
      best_score, best_th = score, th
  return best_th, best_score


def run_walk_forward_ml(panel, feature_cols, start_year=WF_START_YEAR):
  if not RUN_ML or not HAS_SKLEARN or len(panel) == 0:
    return pd.DataFrame(), []
  preds_all, thresholds_by_year = [], []
  current_year = pd.Timestamp.today().year
  for year in tqdm(range(start_year, current_year + 1), desc="WF ML"):
    test_start = pd.Timestamp(f"{year}-01-01")
    test_end = pd.Timestamp(f"{year}-12-31")
    train_end = test_start - pd.Timedelta(days=EMBARGO_DAYS + 1)
    train = panel[panel["date"] <= train_end].copy()
    test = panel[(panel["date"] >= test_start) & (panel["date"] <= test_end)].copy()
    if len(train) < 200 or len(test) < 20:
      continue
    split = train["date"].quantile(0.8)
    train_fit = train[train["date"] < split]
    train_val = train[train["date"] >= split]
    if len(train_fit) < 100:
      train_fit = train
      train_val = train.tail(max(50, len(train) // 5))
    val_preds, _ = fit_ensemble_predict(train_fit, train_val, feature_cols)
    th, th_score = select_threshold_on_validation(val_preds) if len(val_preds) else (BUY_THRESHOLD, 0)
    thresholds_by_year.append({"year": year, "threshold": th, "val_score": round(th_score, 3)})
    test_preds, rf_model = fit_ensemble_predict(train, test, feature_cols)
    if len(test_preds):
      test_preds["year"] = year
      test_preds["threshold_used"] = th
      preds_all.append(test_preds)
  oos = pd.concat(preds_all, ignore_index=True) if preds_all else pd.DataFrame()
  return oos, thresholds_by_year


oos_predictions, threshold_log = run_walk_forward_ml(panel_df, FEATURE_COLS)
print("OOS predictions:", len(oos_predictions))

# %% [markdown]
# ## 9. Backtest event-driven V10

# %%
def backtest_v10_signal_engine(predictions_oos, data_dict, features_by_ticker, initial_capital=INITIAL_CAPITAL):
  if predictions_oos is None or len(predictions_oos) == 0:
    return {}, pd.DataFrame(), pd.Series(dtype=float)
  preds = predictions_oos.copy()
  preds["date"] = pd.to_datetime(preds["date"])
  cost_side = TRANSACTION_COST + SLIPPAGE
  dates = sorted(preds["date"].unique())
  cash = initial_capital
  positions = {}
  trades = []
  equity = []

  for dt in dates:
    day = preds[preds["date"] == dt]
    th = _sf(day["threshold_used"].iloc[0], BUY_THRESHOLD) if "threshold_used" in day.columns else BUY_THRESHOLD
    # Salidas
    to_close = []
    for ticker, pos in positions.items():
      df = data_dict.get(ticker)
      feat = features_by_ticker.get(ticker)
      if df is None or dt not in df.index:
        continue
      row = df.loc[dt]
      low, high, close_px = _sf(row.get("Low")), _sf(row.get("High")), _sf(row.get("Close"))
      holding = (dt - pos["entry_date"]).days
      prob_now = day[day["ticker"] == ticker]["prob_buy"].iloc[0] if ticker in day["ticker"].values else pos["prob_entry"]
      regime = 1.0
      if feat is not None and dt in feat.index:
        regime = _sf(feat.loc[dt].get("MARKET_REGIME_SCORE"), 0.5)
      exit_reason, exit_price = None, None
      if low <= pos["stop"]:
        exit_reason, exit_price = "stop_loss", pos["stop"]
      elif high >= pos["take_profit"]:
        exit_reason, exit_price = "take_profit", pos["take_profit"]
      elif holding >= pos["max_hold"]:
        exit_reason, exit_price = "max_hold", close_px
      elif prob_now < 0.45:
        exit_reason, exit_price = "prob_decay", close_px
      elif regime < 0.25:
        exit_reason, exit_price = "regime_exit", close_px
      if exit_reason:
        proceeds = pos["shares"] * exit_price * (1 - cost_side)
        cash += proceeds
        ret_pct = (exit_price / pos["entry_price"] - 1) * 100 - cost_side * 200
        trades.append({
          "ticker": ticker, "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
          "exit_date": dt, "exit_price": exit_price, "exit_reason": exit_reason,
          "holding_days": holding, "return_pct": round(ret_pct, 3),
          "pnl": round(proceeds - pos["cost_basis"], 2),
          "prob_buy_entry": pos["prob_entry"], "setup_type": pos["setup_type"],
        })
        to_close.append(ticker)
    for t in to_close:
      del positions[t]

    # Entradas
    candidates = day[day["prob_buy"] >= th].sort_values("prob_buy", ascending=False)
    slots = MAX_CONCURRENT_POSITIONS - len(positions)
    for _, sig in candidates.head(slots).iterrows():
      ticker = sig["ticker"]
      if ticker in positions:
        continue
      df = data_dict.get(ticker)
      feat = features_by_ticker.get(ticker)
      if df is None:
        continue
      future = df.index[df.index > dt]
      if len(future) == 0:
        continue
      entry_dt = future[0]
      entry_row = df.loc[entry_dt]
      entry_price = _sf(entry_row.get("Open"), _sf(entry_row.get("Close")))
      if not np.isfinite(entry_price) or entry_price <= 0:
        continue
      atr = _sf(feat.loc[dt, "ATR_14"], entry_price * 0.02) if feat is not None and dt in feat.index else entry_price * 0.02
      alloc = cash * POSITION_SIZE_PCT
      if alloc < 100:
        continue
      shares = alloc / (entry_price * (1 + cost_side))
      cost_basis = shares * entry_price * (1 + cost_side)
      if cost_basis > cash:
        continue
      cash -= cost_basis
      positions[ticker] = {
        "entry_date": entry_dt, "entry_price": entry_price, "shares": shares,
        "cost_basis": cost_basis, "stop": entry_price - DEFAULT_SL_ATR * atr,
        "take_profit": entry_price + DEFAULT_TP_ATR * atr,
        "max_hold": ML_HORIZON, "prob_entry": _sf(sig["prob_buy"]),
        "setup_type": sig.get("setup_type", sig.get("candidate_type", "")),
      }

    port_val = cash
    for ticker, pos in positions.items():
      df = data_dict.get(ticker)
      if df is not None and dt in df.index:
        port_val += pos["shares"] * _sf(df.loc[dt, "Close"])
    equity.append((dt, port_val))

  if len(dates):
    last_dt = dates[-1]
    for ticker, pos in list(positions.items()):
      df = data_dict.get(ticker)
      if df is None:
        continue
      px = _sf(df["Close"].iloc[-1])
      proceeds = pos["shares"] * px * (1 - cost_side)
      cash += proceeds
      trades.append({
        "ticker": ticker, "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
        "exit_date": last_dt, "exit_price": px, "exit_reason": "end_of_sample",
        "holding_days": (last_dt - pos["entry_date"]).days,
        "return_pct": round((px / pos["entry_price"] - 1) * 100 - cost_side * 200, 3),
        "pnl": round(proceeds - pos["cost_basis"], 2),
        "prob_buy_entry": pos["prob_entry"], "setup_type": pos["setup_type"],
      })

  trades_df = pd.DataFrame(trades)
  eq = pd.Series({d: v for d, v in equity}, dtype=float).sort_index()
  if eq.empty:
    eq = pd.Series([initial_capital])
  metrics = compute_trade_metrics(trades_df, eq, initial_capital)
  return metrics, trades_df, eq


def compute_trade_metrics(trades_df, equity, initial_capital):
  if equity.empty or len(equity) < 2:
    return {"num_trades": 0, "total_return": 0, "CAGR": 0, "sharpe": 0, "profit_factor": 0}
  rets = equity.pct_change().fillna(0)
  total_ret = equity.iloc[-1] / initial_capital - 1
  years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
  cagr = (equity.iloc[-1] / initial_capital) ** (1 / years) - 1
  peak = equity.cummax()
  mdd = ((equity - peak) / peak.replace(0, np.nan)).min()
  sh = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
  ds = rets[rets < 0].std()
  so = rets.mean() / ds * np.sqrt(252) if ds and ds > 0 else 0
  wins = trades_df[trades_df["return_pct"] > 0] if len(trades_df) else pd.DataFrame()
  losses = trades_df[trades_df["return_pct"] <= 0] if len(trades_df) else pd.DataFrame()
  pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) and losses["pnl"].sum() != 0 else 0
  return {
    "total_return": round(total_ret * 100, 2),
    "CAGR": round(cagr * 100, 2),
    "sharpe": round(sh, 3),
    "sortino": round(so, 3),
    "max_drawdown": round(mdd * 100, 2),
    "calmar": round(cagr / abs(mdd), 3) if mdd != 0 else 0,
    "num_trades": int(len(trades_df)),
    "win_rate": round(len(wins) / len(trades_df) * 100, 2) if len(trades_df) else 0,
    "avg_trade_return": round(trades_df["return_pct"].mean(), 3) if len(trades_df) else 0,
    "median_trade_return": round(trades_df["return_pct"].median(), 3) if len(trades_df) else 0,
    "best_trade": round(trades_df["return_pct"].max(), 3) if len(trades_df) else 0,
    "worst_trade": round(trades_df["return_pct"].min(), 3) if len(trades_df) else 0,
    "profit_factor": round(pf, 3),
    "expectancy": round(trades_df["pnl"].mean(), 2) if len(trades_df) else 0,
    "exposure_pct": round(POSITION_SIZE_PCT * MAX_CONCURRENT_POSITIONS * 100, 1),
    "avg_holding_days": round(trades_df["holding_days"].mean(), 1) if len(trades_df) else 0,
    "trades_per_year": round(len(trades_df) / years, 1),
  }

# %% [markdown]
# ## 10. Walk-forward completo y validacion

# %%
def benchmark_year(close_prices, year, col):
  if col not in close_prices.columns:
    return 0.0
  s = close_prices[col].loc[f"{year}-01-01":f"{year}-12-31"]
  return (s.iloc[-1] / s.iloc[0] - 1) * 100 if len(s) >= 2 else 0.0


def run_v10_full_pipeline():
  yearly_rows = []
  if len(oos_predictions) == 0:
    return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)
  for year in sorted(oos_predictions["year"].unique()):
    yr_pred = oos_predictions[oos_predictions["year"] == year]
    m, tr, eq = backtest_v10_signal_engine(yr_pred, data_dict, features_by_ticker)
    spy_ret = benchmark_year(close_prices, year, MARKET)
    yearly_rows.append({
      "year": year, "return": m.get("total_return", 0), "SPY_return": round(spy_ret, 2),
      "num_trades": m.get("num_trades", 0), "win_rate": m.get("win_rate", 0),
      "profit_factor": m.get("profit_factor", 0), "sharpe": m.get("sharpe", 0),
      "max_drawdown": m.get("max_drawdown", 0), "beats_spy": m.get("total_return", 0) > spy_ret,
    })
  yearly_df = pd.DataFrame(yearly_rows)
  full_metrics, trades_df, equity_oos = backtest_v10_signal_engine(
    oos_predictions, data_dict, features_by_ticker
  )
  return yearly_df, trades_df, equity_oos, full_metrics


yearly_df, trades_df, equity_oos, full_metrics = run_v10_full_pipeline()
print("WF años:", len(yearly_df), "| trades:", len(trades_df))
if len(full_metrics):
  print("Total return:", full_metrics.get("total_return"), "% | PF:", full_metrics.get("profit_factor"))

# %% [markdown]
# ## 11. Tablas de diagnostico

# %%
def signal_quality_by_bucket(preds):
  if preds is None or len(preds) == 0:
    return pd.DataFrame()
  p = preds.copy()
  p["prob_bucket"] = pd.cut(p["prob_buy"], bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0], include_lowest=True)
  return p.groupby("prob_bucket", observed=True).agg(
    n=("prob_buy", "count"),
    win_rate=("true_label", "mean"),
    avg_forward_return=("forward_return", "mean"),
  ).reset_index()


def performance_by_setup(trades_df):
  if trades_df is None or len(trades_df) == 0:
    return pd.DataFrame()
  return trades_df.groupby("setup_type").agg(
    n=("return_pct", "count"),
    win_rate=("return_pct", lambda x: (x > 0).mean()),
    avg_return=("return_pct", "mean"),
    total_pnl=("pnl", "sum"),
  ).reset_index()


def performance_by_sector(trades_df):
  if trades_df is None or len(trades_df) == 0:
    return pd.DataFrame()
  trades_df = trades_df.copy()
  trades_df["sector"] = trades_df["ticker"].map(SECTOR_MAP).fillna("other")
  return trades_df.groupby("sector").agg(
    n=("return_pct", "count"), avg_return=("return_pct", "mean"), total_pnl=("pnl", "sum"),
  ).reset_index()


def performance_by_ticker(trades_df):
  if trades_df is None or len(trades_df) == 0:
    return pd.DataFrame()
  return trades_df.groupby("ticker").agg(
    n=("return_pct", "count"), avg_return=("return_pct", "mean"), total_pnl=("pnl", "sum"),
  ).reset_index().sort_values("total_pnl", ascending=False)


def confusion_by_year(preds):
  rows = []
  if preds is None or len(preds) == 0:
    return pd.DataFrame()
  for year, grp in preds.groupby("year"):
    y_true = grp["true_label"].astype(int)
    y_pred = (grp["prob_buy"] >= BUY_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    rows.append({"year": year, "tp": tp, "fp": fp, "tn": tn, "fn": fn})
  return pd.DataFrame(rows)


def calibration_table(preds):
  if preds is None or len(preds) == 0:
    return pd.DataFrame()
  p = preds.copy()
  p["prob_bin"] = pd.cut(p["prob_buy"], bins=np.linspace(0, 1, 11), include_lowest=True)
  return p.groupby("prob_bin", observed=True).agg(
    n=("prob_buy", "count"), predicted=("prob_buy", "mean"), actual=("true_label", "mean"),
  ).reset_index()


signal_quality = signal_quality_by_bucket(oos_predictions)
setup_perf = performance_by_setup(trades_df)
sector_perf = performance_by_sector(trades_df)
ticker_perf = performance_by_ticker(trades_df)
confusion_yearly = confusion_by_year(oos_predictions)
calibration_df = calibration_table(oos_predictions)

feature_importance_df = pd.DataFrame()
if HAS_SKLEARN and len(panel_df) > 200:
  _, rf_last = fit_ensemble_predict(
    panel_df[panel_df["date"] < panel_df["date"].quantile(0.85)],
    panel_df[panel_df["date"] >= panel_df["date"].quantile(0.85)],
    FEATURE_COLS,
  )
  if rf_last is not None:
    try:
      rf = rf_last.named_steps["extratreesclassifier"] if "extratreesclassifier" in rf_last.named_steps else rf_last.named_steps.get("randomforestclassifier")
      if rf is not None and hasattr(rf, "feature_importances_"):
        feature_importance_df = pd.DataFrame({
          "feature": FEATURE_COLS[: len(rf.feature_importances_)],
          "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=False).head(30)
    except Exception:
      pass

# %% [markdown]
# ## 12. Score final V10

# %%
def compute_v10_score(yearly_df, trades_df, metrics, preds, ticker_perf_df):
  score = 0
  notes = []
  pf = metrics.get("profit_factor", 0)
  sh = metrics.get("sharpe", 0)
  mdd = metrics.get("max_drawdown", -100)
  n = metrics.get("num_trades", 0)
  avg = metrics.get("avg_trade_return", 0)
  cost_thresh = (TRANSACTION_COST + SLIPPAGE) * 100 * 3
  beats_spy = yearly_df["beats_spy"].mean() if len(yearly_df) else 0

  if pf > 1.2:
    score += 20
  if sh > 0.8:
    score += 15
  if mdd > -25:
    score += 15
  if n >= 100:
    score += 10
  if avg > cost_thresh:
    score += 10
  if beats_spy >= 0.5:
    score += 10
  if len(ticker_perf_df) >= 3:
    top_share = ticker_perf_df["total_pnl"].clip(lower=0).max() / max(ticker_perf_df["total_pnl"].clip(lower=0).sum(), 1)
    if top_share < 0.45:
      score += 10
    else:
      notes.append("depende de pocos tickers")
      score -= 20
  if len(calibration_df) > 2:
    cal_err = abs(calibration_df["predicted"] - calibration_df["actual"]).mean()
    if cal_err < 0.15:
      score += 10
    else:
      notes.append("calibracion debil")

  if mdd < -30:
    score -= 20
  if pf < 1:
    score -= 20
  recent = yearly_df[yearly_df["year"].isin([2025, 2026])] if len(yearly_df) else pd.DataFrame()
  if len(recent) and (recent["return"] < -5).any():
    score -= 20
    notes.append("2025/2026 debiles")
  nvda_share = 0
  if len(trades_df) and "ticker" in trades_df.columns:
    nvda_share = (trades_df["ticker"] == "NVDA").mean()
    if nvda_share > 0.35:
      score -= 20
      notes.append("concentrado en NVDA")

  score = int(np.clip(score, 0, 100))
  if score >= 75:
    status = "APPROVED_FOR_WEB_PAPER"
  elif score >= 60:
    status = "CANDIDATE"
  else:
    status = "REJECTED"
  return score, status, notes


signal_engine_score, signal_status, score_notes = compute_v10_score(
  yearly_df, trades_df, full_metrics, oos_predictions, ticker_perf
)
APPROVED_FOR_WEB_PAPER = signal_status == "APPROVED_FOR_WEB_PAPER"
APPROVED_FOR_REAL_MONEY = False
print(f"V10 score: {signal_engine_score}/100 | {signal_status}")

# %% [markdown]
# ## 13. Señales actuales

# %%
def build_current_signals_v10(features_by_ticker, data_dict, panel, feature_cols):
  last_date = close_prices.index[-1]
  train_end = last_date - pd.Timedelta(days=EMBARGO_DAYS)
  train = panel[panel["date"] <= train_end]
  today_rows = []
  for ticker, df in features_by_ticker.items():
    if ticker in ("SPY", "QQQ", "IWM"):
      continue
    if last_date not in df.index:
      continue
    row = df.loc[last_date]
    ok, setup = classify_candidate_event(row)
    if not ok and not RUN_META_LABELING:
      continue
    today_rows.append({**row.to_dict(), "ticker": ticker, "date": last_date, "setup_type": setup, "candidate_event": ok})
  if not today_rows:
    return pd.DataFrame()
  test = pd.DataFrame(today_rows)
  preds, _ = fit_ensemble_predict(train, test, feature_cols) if len(train) >= 100 else (pd.DataFrame(), None)
  prob_map = dict(zip(preds["ticker"], preds["prob_buy"])) if len(preds) else {}
  rows = []
  for _, tr in test.iterrows():
    ticker = tr["ticker"]
    prob = _sf(prob_map.get(ticker), 0.4)
    close = _sf(tr.get("Close"))
    atr = _sf(tr.get("ATR_14"), close * 0.02)
    stop = round(close - DEFAULT_SL_ATR * atr, 2)
    take = round(close + DEFAULT_TP_ATR * atr, 2)
    conf = int(np.clip(prob * 100, 0, 100))
    if prob >= STRONG_BUY_THRESHOLD:
      signal = "BUY"
      reason = f"momentum relativo + tendencia + modelo ML ({tr.get('setup_type', '')})"
    elif prob >= BUY_THRESHOLD:
      signal = "BUY"
      reason = f"setup {tr.get('setup_type', '')} + prob {prob:.2f}"
    elif prob >= 0.50:
      signal = "WATCH"
      reason = "probabilidad intermedia"
    else:
      signal = "AVOID"
      reason = "baja probabilidad o riesgo alto"
    if _sf(tr.get("VOL_20"), 0) > 0.55:
      signal, reason = "AVOID", "volatilidad extrema"
    rows.append({
      "ticker": ticker,
      "signal": signal,
      "confidence": conf,
      "entry_plan": "entrada proxima apertura" if signal == "BUY" else "-",
      "exit_plan": "salir por stop/take/10 dias" if signal in ("BUY", "WATCH") else "-",
      "stop_loss": stop,
      "take_profit": take,
      "max_holding_days": ML_HORIZON,
      "setup_type": tr.get("setup_type", ""),
      "reason": reason,
      "cash_account_executable": True,
      "approved_for_web_paper": APPROVED_FOR_WEB_PAPER,
    })
  return pd.DataFrame(rows)


current_signals = build_current_signals_v10(features_by_ticker, data_dict, panel_df, FEATURE_COLS)
print("Señales actuales:", len(current_signals))
if len(current_signals):
  print(current_signals[["ticker", "signal", "confidence", "setup_type", "reason"]].sort_values("confidence", ascending=False).head(10).to_string(index=False))

# %% [markdown]
# ## 14. Benchmarks y exportar

# %%
def load_v6_equity():
  for p in [Path("research_outputs/v6/research_v6_equity_curves.csv"), Path("research_v6_equity_curves.csv")]:
    if p.exists():
      try:
        return pd.read_csv(p, index_col=0, parse_dates=True)
      except Exception:
        pass
  return None


v6_eq = load_v6_equity()
v6_col = "blended_champion_weights_alpha_0.5"
v6_total = np.nan
if v6_eq is not None and v6_col in v6_eq.columns:
  s = v6_eq[v6_col].dropna()
  if len(s) >= 2:
    v6_total = (s.iloc[-1] / s.iloc[0] - 1) * 100

spy_total = (close_prices[MARKET].iloc[-1] / close_prices[MARKET].iloc[0] - 1) * 100 if MARKET in close_prices else 0
qqq_total = (close_prices["QQQ"].iloc[-1] / close_prices["QQQ"].iloc[0] - 1) * 100 if "QQQ" in close_prices else 0
ew_cols = [c for c in close_prices.columns if c not in ("SPY", "QQQ", "IWM")]
ew_total = (close_prices[ew_cols].mean(axis=1).iloc[-1] / close_prices[ew_cols].mean(axis=1).iloc[0] - 1) * 100 if ew_cols else 0


def export_csv(df, name):
  if df is None or len(df) == 0:
    pd.DataFrame().to_csv(name, index=False)
  else:
    df.to_csv(name, index=False)
  print("Exportado", name)


summary_df = pd.DataFrame([{
  "lab": "v10_professional_signal_engine",
  "signal_engine_score": signal_engine_score,
  "status": signal_status,
  "approved_for_web_paper": APPROVED_FOR_WEB_PAPER,
  "approved_for_real_money": False,
  **full_metrics,
  "pct_years_beats_spy": round(yearly_df["beats_spy"].mean() * 100, 1) if len(yearly_df) else 0,
  "spy_return": round(spy_total, 2),
  "qqq_return": round(qqq_total, 2),
  "v6_blend_return": round(v6_total, 2) if pd.notna(v6_total) else np.nan,
  "ew_universe_return": round(ew_total, 2),
}])

export_csv(summary_df, "research_v10_summary.csv")
export_csv(trades_df, "research_v10_trades.csv")
export_csv(yearly_df, "research_v10_yearly.csv")
export_csv(oos_predictions, "research_v10_oos_predictions.csv")
export_csv(current_signals, "research_v10_current_signals.csv")
export_csv(feature_importance_df, "research_v10_feature_importance.csv")
export_csv(signal_quality, "research_v10_signal_quality.csv")
if len(equity_oos):
  equity_oos.to_frame("equity").to_csv("research_v10_equity_curve.csv")
  print("Exportado research_v10_equity_curve.csv")
else:
  pd.DataFrame().to_csv("research_v10_equity_curve.csv", index=False)

config_out = {
  "version": "v10_professional_signal_engine",
  "approved_for_web_paper": APPROVED_FOR_WEB_PAPER,
  "approved_for_real_money": False,
  "signal_engine_score": signal_engine_score,
  "status": signal_status,
  "meta_labeling": RUN_META_LABELING,
  "triple_barrier": {"horizon": ML_HORIZON, "tp_atr": DEFAULT_TP_ATR, "sl_atr": DEFAULT_SL_ATR},
  "thresholds": {"buy": BUY_THRESHOLD, "strong_buy": STRONG_BUY_THRESHOLD, "by_year": threshold_log},
  "warnings": [
    "Backtest no garantiza resultados futuros.",
    "No es asesoramiento financiero.",
    "No aprobado para dinero real.",
    "No conecta broker ni ejecuta ordenes.",
    "ML filtra setups candidatos (meta-labeling), no predice desde cero.",
  ],
  "metrics": full_metrics,
}
Path("research_v10_selected_config.json").write_text(json.dumps(config_out, indent=2, default=str), encoding="utf-8")
print("Exportado research_v10_selected_config.json")

# %% [markdown]
# ## 15. Reporte final

# %%
print("=" * 80)
print("REPORTE FINAL V10 PROFESSIONAL SIGNAL ENGINE")
print("=" * 80)
print("Disclaimer: Backtest no garantiza resultados futuros.")
print("")
print(f"Estado V10: {signal_status} | Score: {signal_engine_score}/100")
print(f"Total return: {full_metrics.get('total_return', 0)}% | PF: {full_metrics.get('profit_factor', 0)} | "
      f"Sharpe: {full_metrics.get('sharpe', 0)} | DD: {full_metrics.get('max_drawdown', 0)}%")
print(f"Trades: {full_metrics.get('num_trades', 0)} | Win rate: {full_metrics.get('win_rate', 0)}%")
print(f"¿Supera SPY ({spy_total:.2f}%)? {'SI' if full_metrics.get('total_return', 0) > spy_total else 'NO'}")
print(f"¿Supera QQQ ({qqq_total:.2f}%)? {'SI' if full_metrics.get('total_return', 0) > qqq_total else 'NO'}")
if pd.notna(v6_total):
  print(f"¿Supera V6 ({v6_total:.2f}%)? {'SI' if full_metrics.get('total_return', 0) > v6_total else 'NO'}")
if len(setup_perf):
  print("\nMejores setups:")
  print(setup_perf.sort_values("avg_return", ascending=False).head(3).to_string(index=False))
  print("\nPeores setups:")
  print(setup_perf.sort_values("avg_return").head(3).to_string(index=False))
print(f"\nSeñales actuales BUY: {len(current_signals[current_signals['signal']=='BUY']) if len(current_signals) else 0}")
if score_notes:
  print("Notas score:", "; ".join(score_notes))
print("")
if APPROVED_FOR_WEB_PAPER:
  print("Integrar como V10 Signal Engine Paper Trading.")
else:
  print("No integrar V10. Mantener V6 como champion.")
print("APPROVED_FOR_REAL_MONEY=False (siempre)")
