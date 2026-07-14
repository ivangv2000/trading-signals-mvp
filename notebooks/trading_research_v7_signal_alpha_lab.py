# %% [markdown]
# # Trading Research V7 Signal Alpha Lab
#
# Motor predictivo de señales individuales por acción: BUY / SELL / HOLD / AVOID.
#
# **Disclaimer:** Backtest no garantiza resultados futuros. No es asesoramiento financiero.
# **No conecta broker. No ejecuta órdenes. APPROVED_FOR_REAL_MONEY siempre False.**

# %%
!pip install yfinance pandas numpy matplotlib plotly tqdm scikit-learn scipy -q

# %% [markdown]
# ## 1. Configuracion

# %%
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tqdm.auto import tqdm

try:
  from sklearn.pipeline import make_pipeline
  from sklearn.impute import SimpleImputer
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
  HAS_SKLEARN = True
except Exception:
  HAS_SKLEARN = False

QUICK_TEST = False
START_DATE = "2015-01-01"
END_DATE = None
HORIZONS = [1, 3, 5, 10]
TRANSACTION_COST = 0.001
SLIPPAGE = 0.001
INITIAL_CAPITAL = 10000
TOP_N_SIGNALS = 5
MAX_CONCURRENT_POSITIONS = 5
POSITION_SIZE_PCT = 0.10
RUN_ML = True
RUN_BASELINE_RULES = True
WF_START_YEAR = 2018
EMBARGO_DAYS = 10
RANDOM_SEED = 42
ML_HORIZON = 5

DEFAULT_UNIVERSE = [
  "SPY", "QQQ", "IWM",
  "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN",
  "AVGO", "NFLX", "COST", "ADBE", "CRM", "ORCL", "INTC", "MU", "PANW",
  "JPM", "BAC", "GS", "MS",
  "XOM", "CVX",
  "UNH", "LLY", "JNJ",
  "WMT", "HD", "MCD",
]

if QUICK_TEST:
  DEFAULT_UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "META", "GOOGL", "AMZN", "TSLA"]
  START_DATE = "2020-01-01"
  HORIZONS = [3, 5]
  WF_START_YEAR = 2021
  print("QUICK_TEST activo")

MARKET = "SPY"
print("V7 Signal Alpha Lab | desde", START_DATE)
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
    print("Fallidos:", failed[:20], "..." if len(failed) > 20 else "")
  close = pd.DataFrame({k: v["Close"] for k, v in data.items()}).sort_index().ffill()
  close.index = pd.DatetimeIndex(close.index)
  if close.index.tz:
    close.index = close.index.tz_localize(None)
  return data, close


data_dict, close_prices = download_data(DEFAULT_UNIVERSE, START_DATE, END_DATE)
print("Tickers descargados:", len(data_dict), "| filas:", len(close_prices))

# %% [markdown]
# ## 3. Features / indicadores predictivos

# %%
def _rsi(close, window):
  delta = close.diff()
  gain = delta.clip(lower=0).rolling(window).mean()
  loss = (-delta.clip(upper=0)).rolling(window).mean()
  return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _atr(df, window=14):
  high, low, close = df["High"], df["Low"], df["Close"]
  tr = pd.concat([
    high - low,
    (high - close.shift(1)).abs(),
    (low - close.shift(1)).abs(),
  ], axis=1).max(axis=1)
  return tr.rolling(window).mean()


def add_signal_features(df, spy_features=None):
  """Features sin mirar el futuro."""
  d = df.copy()
  c = d["Close"]
  o = d["Open"] if "Open" in d.columns else c
  h = d["High"] if "High" in d.columns else c
  l = d["Low"] if "Low" in d.columns else c
  v = d["Volume"] if "Volume" in d.columns else pd.Series(0, index=d.index)

  d["close"] = c
  d["open"] = o
  d["high"] = h
  d["low"] = l
  d["volume"] = v

  r1 = c.pct_change(1)
  for n in [1, 2, 3, 5, 10, 20, 60]:
    d[f"RET_{n}D"] = r1 if n == 1 else c.pct_change(n)
  for n in [5, 10, 20, 60, 120]:
    d[f"MOM_{n}"] = c.pct_change(n)
  for n in [10, 20, 50, 100, 200]:
    d[f"SMA_{n}"] = c.rolling(n).mean()
  for span in [9, 21, 50]:
    d[f"EMA_{span}"] = c.ewm(span=span, adjust=False).mean()
  d["DIST_SMA20"] = c / d["SMA_20"] - 1
  d["DIST_SMA50"] = c / d["SMA_50"] - 1
  d["DIST_SMA200"] = c / d["SMA_200"] - 1
  for n in [5, 10, 20, 60]:
    d[f"VOL_{n}"] = r1.rolling(n).std() * np.sqrt(252)
  d["ATR_14"] = _atr(d, 14)
  d["ATR_PCT"] = d["ATR_14"] / c.replace(0, np.nan)
  for w in [2, 7, 14]:
    d[f"RSI_{w}"] = _rsi(c, w)
  for n in [5, 10, 20]:
    d[f"HIGH_{n}_PREV"] = h.rolling(n).max().shift(1)
    d[f"LOW_{n}_PREV"] = l.rolling(n).min().shift(1)
  d["VOLUME_AVG_20"] = v.rolling(20).mean()
  d["VOLUME_RATIO"] = v / d["VOLUME_AVG_20"].replace(0, np.nan)
  hl = (h - l).replace(0, np.nan)
  d["DAILY_RANGE"] = hl / c.replace(0, np.nan)
  d["CLOSE_POSITION_IN_RANGE"] = (c - l) / hl
  d["GAP"] = o / c.shift(1) - 1

  if spy_features is not None:
    for col in ["SPY_RET_5", "SPY_RET_20", "SPY_ABOVE_SMA200", "SPY_VOL_20"]:
      if col in spy_features.columns:
        d[col] = spy_features[col].reindex(d.index).ffill()
  return d


def build_spy_features(data_dict):
  spy = data_dict.get(MARKET)
  if spy is None or spy.empty:
    return pd.DataFrame()
  s = add_signal_features(spy.copy())
  out = pd.DataFrame(index=s.index)
  out["SPY_RET_5"] = s["RET_5D"]
  out["SPY_RET_20"] = s["RET_20D"]
  out["SPY_ABOVE_SMA200"] = (s["Close"] > s["SMA_200"]).astype(float)
  out["SPY_VOL_20"] = s["VOL_20"]
  return out


spy_features = build_spy_features(data_dict)
features_by_ticker = {}
for ticker, df in data_dict.items():
  features_by_ticker[ticker] = add_signal_features(df.copy(), spy_features)

print("Features calculadas para", len(features_by_ticker), "tickers")

# %% [markdown]
# ## 4. Labels predictivos (solo entrenamiento / backtest)

# %%
def add_labels(df, horizons=None):
  horizons = horizons or HORIZONS
  d = df.copy()
  c = d["Close"]
  atr_pct = d.get("ATR_PCT", pd.Series(0.02, index=d.index)).fillna(0.02)
  for h in horizons:
    fwd = c.shift(-h) / c - 1
    d[f"forward_return_{h}d"] = fwd
    thr = np.maximum(0.015, 0.75 * atr_pct)
    d[f"target_buy_{h}d"] = (fwd > thr).astype(int)
    d[f"target_sell_{h}d"] = (fwd < -thr).astype(int)
    d[f"label_{h}d"] = np.where(fwd > thr, "BUY", np.where(fwd < -thr, "SELL", "HOLD"))
  return d


for t in features_by_ticker:
  features_by_ticker[t] = add_labels(features_by_ticker[t])

# %% [markdown]
# ## 5. Dataset panel

# %%
LABEL_PREFIXES = ("forward_return_", "target_", "label_")


def get_feature_cols(df):
  exclude = {"Open", "High", "Low", "Close", "Volume", "close", "open", "high", "low", "volume"}
  cols = []
  for c in df.columns:
    if c in exclude:
      continue
    if any(c.startswith(p) for p in LABEL_PREFIXES):
      continue
    if pd.api.types.is_numeric_dtype(df[c]):
      cols.append(c)
  return cols


def build_panel(features_by_ticker, tickers=None):
  rows = []
  tickers = tickers or list(features_by_ticker.keys())
  for ticker in tickers:
    df = features_by_ticker.get(ticker)
    if df is None or df.empty:
      continue
    for dt, row in df.iterrows():
      rec = row.to_dict()
      rec["ticker"] = ticker
      rec["date"] = dt
      rows.append(rec)
  panel = pd.DataFrame(rows)
  if len(panel):
    panel["date"] = pd.to_datetime(panel["date"])
  return panel


panel_df = build_panel(features_by_ticker)
FEATURE_COLS = get_feature_cols(features_by_ticker[MARKET]) if MARKET in features_by_ticker else get_feature_cols(panel_df)
print("Panel:", len(panel_df), "filas |", len(FEATURE_COLS), "features")

# %%
def _sf(x, default=np.nan):
  try:
    if x is None or (isinstance(x, float) and np.isnan(x)):
      return default
    v = float(x)
    return default if not np.isfinite(v) else v
  except Exception:
    return default

# %% [markdown]
# ## 6. Estrategias baseline sin ML

# %%
def rule_breakout_momentum(row):
  buy = (
    _sf(row.get("Close")) > _sf(row.get("HIGH_20_PREV"), -1)
    and _sf(row.get("MOM_20"), -1) > 0
    and 50 <= _sf(row.get("RSI_14"), 0) <= 75
    and _sf(row.get("VOLUME_RATIO"), 0) > 0.8
  )
  sell = _sf(row.get("Close")) < _sf(row.get("EMA_21"), 1e9) and _sf(row.get("MOM_5"), 0) < 0
  return "BUY" if buy else ("SELL" if sell else "HOLD")


def rule_pullback_trend(row):
  close = _sf(row.get("Close"))
  buy = (
    close > _sf(row.get("SMA_100"), 0)
    and close > _sf(row.get("SMA_200"), 0)
    and 35 <= _sf(row.get("RSI_14"), 0) <= 55
    and abs(close - _sf(row.get("EMA_21"), close)) / max(close, 1) < 0.03
    and _sf(row.get("SPY_ABOVE_SMA200"), 0) >= 0.5
  )
  sell = close < _sf(row.get("EMA_21"), 0) and _sf(row.get("RSI_14"), 0) > 75
  return "BUY" if buy else ("SELL" if sell else "HOLD")


def rule_mean_reversion_rsi2(row):
  buy = (
    _sf(row.get("RSI_2"), 50) < 10
    and _sf(row.get("Close")) > _sf(row.get("SMA_200"), 0)
    and _sf(row.get("SPY_ABOVE_SMA200"), 0) >= 0.5
    and _sf(row.get("CLOSE_POSITION_IN_RANGE"), 1) < 0.3
  )
  sell = _sf(row.get("RSI_2"), 0) > 50 and _sf(row.get("Close")) > _sf(row.get("EMA_9"), 0)
  return "BUY" if buy else ("SELL" if sell else "HOLD")


def rule_vol_squeeze_breakout(row):
  buy = (
    _sf(row.get("VOL_10"), 1) < _sf(row.get("VOL_60"), 0)
    and _sf(row.get("Close")) > _sf(row.get("HIGH_10_PREV"), -1)
    and _sf(row.get("MOM_5"), -1) > 0
    and _sf(row.get("VOLUME_RATIO"), 0) > 1
  )
  sell = _sf(row.get("Close")) < _sf(row.get("EMA_9"), 1e9) and _sf(row.get("MOM_5"), 0) < 0
  return "BUY" if buy else ("SELL" if sell else "HOLD")


BASELINE_RULES = {
  "breakout_momentum": rule_breakout_momentum,
  "pullback_trend": rule_pullback_trend,
  "mean_reversion_rsi2": rule_mean_reversion_rsi2,
  "vol_squeeze_breakout": rule_vol_squeeze_breakout,
}


def baseline_scores(row):
  votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
  triggered = []
  if not RUN_BASELINE_RULES:
    return 50.0, ""
  for name, fn in BASELINE_RULES.items():
    try:
      sig = fn(row)
      votes[sig] = votes.get(sig, 0) + 1
      if sig == "BUY":
        triggered.append(name)
    except Exception:
      pass
  buy_votes = votes["BUY"]
  score = 40 + buy_votes * 12 - votes["SELL"] * 8
  score = float(np.clip(score, 0, 100))
  return score, " + ".join(triggered) if triggered else "sin regla clara"


# %% [markdown]
# ## 7. Machine Learning ranking (walk-forward OOS)

# %%
def clean_ml_matrix(X, feature_cols):
  X = X.copy()
  for col in feature_cols:
    if col not in X.columns:
      X[col] = 0.0
  X = X[feature_cols]
  for col in feature_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")
  X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
  return X.astype(float)


def run_ml_walk_forward(panel, horizon=5, start_year=WF_START_YEAR):
  if not RUN_ML or not HAS_SKLEARN:
    return pd.DataFrame()
  target = f"target_buy_{horizon}d"
  fwd = f"forward_return_{horizon}d"
  if target not in panel.columns:
    return pd.DataFrame()

  preds = []
  current_year = pd.Timestamp.today().year
  feature_cols = [c for c in FEATURE_COLS if c in panel.columns]

  for year in range(start_year, current_year + 1):
    test_start = pd.Timestamp(f"{year}-01-01")
    test_end = pd.Timestamp(f"{year}-12-31")
    train_end = test_start - pd.Timedelta(days=EMBARGO_DAYS + 1)
    train = panel[panel["date"] <= train_end].copy()
    test = panel[(panel["date"] >= test_start) & (panel["date"] <= test_end)].copy()
    if len(train) < 500 or len(test) < 50:
      continue
    X = clean_ml_matrix(train, feature_cols)
    y = pd.to_numeric(train[target], errors="coerce").fillna(0).astype(int)
    Xt = clean_ml_matrix(test, feature_cols)
    models = [
      make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500, random_state=RANDOM_SEED)),
      make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_SEED, n_jobs=-1)),
      make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_depth=5, random_state=RANDOM_SEED)),
    ]
    probs = []
    for model in models:
      try:
        model.fit(X, y)
        if hasattr(model, "predict_proba"):
          p = model.predict_proba(Xt)[:, 1]
        else:
          p = model.predict(Xt)
        probs.append(p)
      except Exception as exc:
        print(f"ML {year} model failed: {exc}")
    if not probs:
      continue
    prob_buy = np.mean(probs, axis=0)
    chunk = test[["ticker", "date"]].copy()
    chunk["probability_buy"] = prob_buy
    chunk["model_score"] = prob_buy
    chunk["forward_return_real"] = test[fwd].values if fwd in test.columns else np.nan
    chunk["horizon"] = horizon
    preds.append(chunk)

  if not preds:
    return pd.DataFrame()
  return pd.concat(preds, ignore_index=True)


ml_predictions_5d = run_ml_walk_forward(panel_df, horizon=5)
ml_predictions_10d = run_ml_walk_forward(panel_df, horizon=10) if 10 in HORIZONS else pd.DataFrame()
ml_predictions = ml_predictions_5d.copy()
if len(ml_predictions_10d):
  ml_predictions = pd.concat([ml_predictions_5d, ml_predictions_10d], ignore_index=True)
print("ML OOS predictions:", len(ml_predictions))

# %% [markdown]
# ## 8. Motor de señales

# %%
def momentum_score(row):
  s = 50.0
  if _sf(row.get("MOM_5"), 0) > 0:
    s += 10
  if _sf(row.get("MOM_20"), 0) > 0:
    s += 15
  if _sf(row.get("Close")) > _sf(row.get("HIGH_20_PREV"), -1):
    s += 15
  if _sf(row.get("VOLUME_RATIO"), 0) > 1:
    s += 10
  return float(np.clip(s, 0, 100))


def trend_score(row):
  s = 40.0
  c = _sf(row.get("Close"))
  if c > _sf(row.get("SMA_50"), 0):
    s += 15
  if c > _sf(row.get("SMA_200"), 0):
    s += 20
  if _sf(row.get("SPY_ABOVE_SMA200"), 0) >= 0.5:
    s += 15
  if c > _sf(row.get("EMA_21"), 0):
    s += 10
  return float(np.clip(s, 0, 100))


def risk_score(row):
  s = 70.0
  vol = _sf(row.get("VOL_20"), 0.2)
  atr_pct = _sf(row.get("ATR_PCT"), 0.02)
  if vol > 0.45:
    s -= 35
  elif vol > 0.30:
    s -= 20
  if atr_pct > 0.05:
    s -= 15
  if _sf(row.get("VOLUME_RATIO"), 1) < 0.3:
    s -= 20
  return float(np.clip(s, 0, 100))


def lookup_ml_score(ticker, date, ml_df):
  if ml_df is None or len(ml_df) == 0:
    return np.nan
  d = pd.Timestamp(date).normalize()
  m = ml_df.copy()
  m["date"] = pd.to_datetime(m["date"]).dt.normalize()
  hit = m[(m["ticker"] == ticker) & (m["date"] == d)]
  if hit.empty:
    return np.nan
  return _sf(hit["model_score"].iloc[-1], np.nan)


def predict_ml_snapshot(panel, horizon=5, as_of_date=None):
  """Entrena con historia hasta as_of_date - embargo y predice ese dia (solo señales actuales)."""
  if not RUN_ML or not HAS_SKLEARN:
    return pd.DataFrame()
  target = f"target_buy_{horizon}d"
  as_of_date = pd.Timestamp(as_of_date or panel["date"].max())
  train_end = as_of_date - pd.Timedelta(days=EMBARGO_DAYS)
  train = panel[panel["date"] <= train_end]
  test = panel[panel["date"] == as_of_date]
  if len(train) < 300 or len(test) == 0:
    return pd.DataFrame()
  feature_cols = [c for c in FEATURE_COLS if c in panel.columns]
  X = clean_ml_matrix(train, feature_cols)
  y = pd.to_numeric(train[target], errors="coerce").fillna(0).astype(int)
  Xt = clean_ml_matrix(test, feature_cols)
  models = [
    make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                  LogisticRegression(max_iter=500, random_state=RANDOM_SEED)),
    make_pipeline(SimpleImputer(strategy="median"),
                  RandomForestClassifier(n_estimators=75, max_depth=5, random_state=RANDOM_SEED, n_jobs=-1)),
  ]
  probs = []
  for model in models:
    try:
      model.fit(X, y)
      probs.append(model.predict_proba(Xt)[:, 1])
    except Exception:
      pass
  if not probs:
    return pd.DataFrame()
  out = test[["ticker", "date"]].copy()
  out["model_score"] = np.mean(probs, axis=0)
  out["probability_buy"] = out["model_score"]
  return out


def generate_daily_signals(date, features_by_ticker, ml_predictions=None, max_holding=5):
  date = pd.Timestamp(date)
  rows = []
  ml_df = ml_predictions

  for ticker, df in features_by_ticker.items():
    if ticker == MARKET:
      continue
    sub = df[df.index <= date]
    if len(sub) < 120:
      rows.append(_avoid_row(ticker, date, "datos insuficientes"))
      continue
    row = sub.iloc[-1]
    if row.isna().sum() > len(row) * 0.5:
      rows.append(_avoid_row(ticker, date, "demasiados NaN"))
      continue

    mom = momentum_score(row)
    trd = trend_score(row)
    rsk = risk_score(row)
    ml_s = lookup_ml_score(ticker, date, ml_df)
    base_score, entry_rule = baseline_scores(row)

    if pd.notna(ml_s):
      combined = 0.40 * (ml_s * 100) + 0.25 * mom + 0.20 * trd + 0.15 * rsk
      entry_rule = f"ml+{entry_rule}" if entry_rule else "ml ranking"
    else:
      combined = 0.40 * mom + 0.35 * trd + 0.25 * rsk

    combined = float(np.clip(combined, 0, 100))
    close = _sf(row.get("Close"))
    atr = _sf(row.get("ATR_14"), close * 0.02)
    stop_price = close - 1.5 * atr
    tp_price = close + 2.5 * atr
    stop_pct = (stop_price / close - 1) * 100 if close > 0 else -4.5
    tp_pct = (tp_price / close - 1) * 100 if close > 0 else 8.0

    extreme_vol = _sf(row.get("VOL_20"), 0) > 0.50
    spy_bad = _sf(row.get("SPY_ABOVE_SMA200"), 1) < 0.5 and _sf(row.get("SPY_RET_20"), 0) < -0.05
    low_liq = _sf(row.get("VOLUME_RATIO"), 1) < 0.25

    if extreme_vol or low_liq:
      signal, reason = "AVOID", "volatilidad extrema o liquidez baja"
    elif combined < 45:
      signal, reason = "AVOID", "score bajo"
    elif spy_bad and combined < 60:
      signal, reason = "HOLD", "mercado debil sin ventaja clara"
    elif combined >= 70 and not extreme_vol and not spy_bad:
      signal = "BUY"
      reason = _buy_reason(row, entry_rule)
    elif combined < 45 or _sf(row.get("Close")) < _sf(row.get("EMA_21"), close):
      signal = "SELL"
      reason = "debilidad tecnica"
    elif 45 <= combined < 70:
      signal, reason = "HOLD", "sin ventaja clara para entrar"
    else:
      signal, reason = "HOLD", "esperar confirmacion"

    rows.append({
      "ticker": ticker,
      "date": date,
      "signal": signal,
      "entry_date": "proxima apertura" if signal == "BUY" else "no entrar",
      "entry_rule": entry_rule if signal == "BUY" else "-",
      "max_holding_days": max_holding if signal == "BUY" else np.nan,
      "stop_loss": round(stop_pct, 2),
      "take_profit": round(tp_pct, 2),
      "stop_loss_price": round(stop_price, 2),
      "take_profit_price": round(tp_price, 2),
      "confidence_score": int(round(combined)),
      "model_score": round(_sf(ml_s, combined / 100), 3),
      "combined_score": round(combined, 2),
      "momentum_score": round(mom, 1),
      "trend_score": round(trd, 1),
      "risk_score": round(rsk, 1),
      "reason": reason,
      "close": close,
    })
  return pd.DataFrame(rows)


def _avoid_row(ticker, date, reason):
  return {
    "ticker": ticker, "date": date, "signal": "AVOID", "entry_date": "no entrar",
    "entry_rule": "-", "max_holding_days": np.nan, "stop_loss": np.nan, "take_profit": np.nan,
    "stop_loss_price": np.nan, "take_profit_price": np.nan, "confidence_score": 20,
    "model_score": 0.0, "combined_score": 20.0, "momentum_score": 0, "trend_score": 0,
    "risk_score": 0, "reason": reason, "close": np.nan,
  }


def _buy_reason(row, entry_rule):
  parts = []
  if _sf(row.get("MOM_20"), 0) > 0:
    parts.append("momentum fuerte")
  if _sf(row.get("VOLUME_RATIO"), 0) > 1:
    parts.append("volumen positivo")
  if entry_rule:
    parts.append(entry_rule)
  return " y ".join(parts) if parts else "senal combinada"

# %% [markdown]
# ## 9. Backtest event-driven de señales

# %%
def backtest_signal_engine(signals_df, data_dict, initial_capital=INITIAL_CAPITAL):
  if signals_df is None or len(signals_df) == 0:
    return {}, pd.DataFrame(), pd.Series(dtype=float)

  signals_df = signals_df.copy()
  signals_df["date"] = pd.to_datetime(signals_df["date"])
  dates = sorted(signals_df["date"].unique())
  cash = initial_capital
  positions = {}
  trades = []
  equity = []

  cost_side = TRANSACTION_COST + SLIPPAGE

  for dt in dates:
    day_sigs = signals_df[signals_df["date"] == dt].copy()
    # Salidas primero
    to_close = []
    for ticker, pos in positions.items():
      df = data_dict.get(ticker)
      if df is None or dt not in df.index:
        continue
      idx = df.index.get_loc(dt)
      row = df.loc[dt]
      low, high, close_px = _sf(row.get("Low")), _sf(row.get("High")), _sf(row.get("Close"))
      sig_row = day_sigs[day_sigs["ticker"] == ticker]
      sig = sig_row["signal"].iloc[0] if len(sig_row) else "HOLD"
      conf = sig_row["confidence_score"].iloc[0] if len(sig_row) else 50
      holding = (dt - pos["entry_date"]).days
      exit_reason, exit_price = None, None
      if low <= pos["stop"]:
        exit_reason, exit_price = "stop_loss", pos["stop"]
      elif high >= pos["take_profit"]:
        exit_reason, exit_price = "take_profit", pos["take_profit"]
      elif holding >= pos["max_holding"]:
        exit_reason, exit_price = "max_holding_days", close_px
      elif sig == "SELL" or conf < 45:
        exit_reason, exit_price = "signal_sell", close_px
      if exit_reason:
        proceeds = pos["shares"] * exit_price * (1 - cost_side)
        cash += proceeds
        ret_pct = (exit_price / pos["entry_price"] - 1) * 100 - cost_side * 200
        trades.append({
          "ticker": ticker, "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
          "exit_date": dt, "exit_price": exit_price, "exit_reason": exit_reason,
          "holding_days": holding, "return_pct": round(ret_pct, 3),
          "pnl": round(proceeds - pos["cost_basis"], 2),
          "confidence_score_entry": pos["confidence"],
        })
        to_close.append(ticker)
    for t in to_close:
      del positions[t]

    # Entradas
    buys = day_sigs[day_sigs["signal"] == "BUY"].sort_values("confidence_score", ascending=False)
    slots = MAX_CONCURRENT_POSITIONS - len(positions)
    for _, sig in buys.head(min(TOP_N_SIGNALS, slots)).iterrows():
      ticker = sig["ticker"]
      if ticker in positions:
        continue
      df = data_dict.get(ticker)
      if df is None:
        continue
      future_idx = df.index[df.index > dt]
      if len(future_idx) == 0:
        continue
      entry_dt = future_idx[0]
      entry_row = df.loc[entry_dt]
      entry_price = _sf(entry_row.get("Open"), _sf(entry_row.get("Close")))
      if not np.isfinite(entry_price) or entry_price <= 0:
        continue
      alloc = cash * POSITION_SIZE_PCT
      if alloc < 100:
        continue
      shares = alloc / (entry_price * (1 + cost_side))
      cost_basis = shares * entry_price * (1 + cost_side)
      if cost_basis > cash:
        continue
      cash -= cost_basis
      atr = _sf(sig.get("stop_loss_price"), entry_price * 0.96)
      stop_p = _sf(sig.get("stop_loss_price"), entry_price - 0.015 * entry_price)
      tp_p = _sf(sig.get("take_profit_price"), entry_price + 0.025 * entry_price)
      positions[ticker] = {
        "entry_date": entry_dt, "entry_price": entry_price, "shares": shares,
        "cost_basis": cost_basis, "stop": stop_p, "take_profit": tp_p,
        "max_holding": int(_sf(sig.get("max_holding_days"), 5)),
        "confidence": int(_sf(sig.get("confidence_score"), 50)),
      }

    # Mark-to-market
    port_val = cash
    for ticker, pos in positions.items():
      df = data_dict.get(ticker)
      if df is None or dt not in df.index:
        continue
      port_val += pos["shares"] * _sf(df.loc[dt, "Close"])
    equity.append((dt, port_val))

  # Cerrar posiciones abiertas al final
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
        "confidence_score_entry": pos["confidence"],
      })

  trades_df = pd.DataFrame(trades)
  eq = pd.Series({d: v for d, v in equity}, dtype=float).sort_index()
  if eq.empty:
    eq = pd.Series([initial_capital], index=[dates[0]] if dates else [pd.Timestamp.today()])
  metrics = compute_trade_metrics(trades_df, eq, initial_capital)
  return metrics, trades_df, eq


def compute_trade_metrics(trades_df, equity, initial_capital):
  if equity.empty:
    return {"num_trades": 0, "total_return": 0, "CAGR": 0, "max_drawdown": 0, "sharpe": 0}
  rets = equity.pct_change().fillna(0)
  total_ret = equity.iloc[-1] / initial_capital - 1
  years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
  cagr = (equity.iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 else 0
  peak = equity.cummax()
  dd = (equity - peak) / peak.replace(0, np.nan)
  mdd = dd.min() if len(dd) else 0
  sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
  downside = rets[rets < 0].std()
  sortino = rets.mean() / downside * np.sqrt(252) if downside and downside > 0 else 0
  wins = trades_df[trades_df["return_pct"] > 0] if len(trades_df) else pd.DataFrame()
  losses = trades_df[trades_df["return_pct"] <= 0] if len(trades_df) else pd.DataFrame()
  win_rate = len(wins) / len(trades_df) if len(trades_df) else 0
  gross_win = wins["pnl"].sum() if len(wins) else 0
  gross_loss = abs(losses["pnl"].sum()) if len(losses) else 0
  pf = gross_win / gross_loss if gross_loss > 0 else np.nan
  avg_ret = trades_df["return_pct"].mean() if len(trades_df) else 0
  med_ret = trades_df["return_pct"].median() if len(trades_df) else 0
  expectancy = trades_df["pnl"].mean() if len(trades_df) else 0
  return {
    "total_return": round(total_ret * 100, 2),
    "CAGR": round(cagr * 100, 2),
    "max_drawdown": round(mdd * 100, 2),
    "sharpe": round(sharpe, 3),
    "sortino": round(sortino, 3),
    "num_trades": int(len(trades_df)),
    "win_rate": round(win_rate * 100, 2),
    "avg_trade_return": round(avg_ret, 3),
    "median_trade_return": round(med_ret, 3),
    "best_trade": round(trades_df["return_pct"].max(), 3) if len(trades_df) else 0,
    "worst_trade": round(trades_df["return_pct"].min(), 3) if len(trades_df) else 0,
    "profit_factor": round(pf, 3) if pd.notna(pf) else 0,
    "expectancy": round(expectancy, 2),
    "exposure_pct": round(POSITION_SIZE_PCT * MAX_CONCURRENT_POSITIONS * 100, 1),
    "avg_holding_days": round(trades_df["holding_days"].mean(), 1) if len(trades_df) else 0,
    "trades_per_year": round(len(trades_df) / years, 1) if years > 0 else 0,
  }

# %% [markdown]
# ## 10. Walk-forward completo

# %%
def benchmark_return_year(close_prices, year, col):
  if col not in close_prices.columns:
    return 0.0
  s = close_prices[col].loc[f"{year}-01-01":f"{year}-12-31"]
  if len(s) < 2:
    return 0.0
  return (s.iloc[-1] / s.iloc[0] - 1) * 100


def run_signal_engine_walk_forward(features_by_ticker, data_dict, close_prices, start_year=WF_START_YEAR):
  yearly_rows, all_trades, all_signals = [], [], []
  current_year = pd.Timestamp.today().year

  for year in tqdm(range(start_year, current_year + 1), desc="WF Signal Engine"):
    try:
      test_start = pd.Timestamp(f"{year}-01-01")
      test_end = pd.Timestamp(f"{year}-12-31")
      train_end = test_start - pd.Timedelta(days=EMBARGO_DAYS + 1)
      train_panel = panel_df[panel_df["date"] <= train_end]
      year_panel = panel_df[(panel_df["date"] >= test_start) & (panel_df["date"] <= test_end)]

      ml_year = pd.DataFrame()
      if RUN_ML and HAS_SKLEARN and len(train_panel) > 500 and len(year_panel) > 0:
        tmp_panel = pd.concat([train_panel, year_panel])
        ml_year = run_ml_walk_forward(tmp_panel, horizon=ML_HORIZON, start_year=year)
        if len(ml_year):
          ml_year = ml_year[ml_year["date"].dt.year == year]

      test_dates = sorted(year_panel["date"].unique()) if len(year_panel) else []
      if not test_dates:
        continue
      # Muestrear dias habiles (cada 1 dia para precision, cada 5 si QUICK_TEST)
      step = 5 if QUICK_TEST else 1
      test_dates = test_dates[::step]

      year_signals = []
      for dt in test_dates:
        sigs = generate_daily_signals(dt, features_by_ticker, ml_year if len(ml_year) else None)
        year_signals.append(sigs)
      if not year_signals:
        continue
      signals_year = pd.concat(year_signals, ignore_index=True)
      all_signals.append(signals_year)

      metrics, trades, eq = backtest_signal_engine(signals_year, data_dict)
      if len(trades):
        all_trades.append(trades.assign(year=year))

      spy_ret = benchmark_return_year(close_prices, year, MARKET)
      qqq_ret = benchmark_return_year(close_prices, year, "QQQ")
      yearly_rows.append({
        "year": year,
        "return": metrics.get("total_return", 0),
        "SPY_return": round(spy_ret, 2),
        "QQQ_return": round(qqq_ret, 2),
        "num_trades": metrics.get("num_trades", 0),
        "win_rate": metrics.get("win_rate", 0),
        "max_drawdown": metrics.get("max_drawdown", 0),
        "profit_factor": metrics.get("profit_factor", 0),
        "sharpe": metrics.get("sharpe", 0),
        "beats_spy": metrics.get("total_return", 0) > spy_ret,
      })
    except Exception as exc:
      print(f"WF year {year} failed: {exc}")
      continue

  yearly_df = pd.DataFrame(yearly_rows)
  trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
  signals_oos = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()

  # Equity OOS concatenando anos
  if len(yearly_df):
    eq_parts = []
    cap = INITIAL_CAPITAL
    for year in yearly_df["year"]:
      yr_sigs = signals_oos[signals_oos["date"].dt.year == year] if len(signals_oos) else pd.DataFrame()
      if len(yr_sigs) == 0:
        continue
      m, _, eq = backtest_signal_engine(yr_sigs, data_dict, initial_capital=cap)
      if len(eq):
        eq_scaled = eq / eq.iloc[0] * cap
        cap = eq_scaled.iloc[-1]
        eq_parts.append(eq_scaled)
    equity_oos = pd.concat(eq_parts).sort_index() if eq_parts else pd.Series(dtype=float)
  else:
    equity_oos = pd.Series(dtype=float)

  return yearly_df, trades_df, signals_oos, equity_oos


yearly_df, trades_df, signals_oos, equity_oos = run_signal_engine_walk_forward(
  features_by_ticker, data_dict, close_prices, WF_START_YEAR
)
print("Walk-forward años:", len(yearly_df), "| trades:", len(trades_df))

# %% [markdown]
# ## 11. Validacion anti-overfitting

# %%
def compute_signal_engine_score(yearly_df, trades_df, metrics_full):
  if yearly_df is None or len(yearly_df) == 0:
    return 0, "REJECTED", ["sin resultados walk-forward"]

  score = 0
  notes = []
  pct_beats_spy = yearly_df["beats_spy"].mean()
  pct_positive = (yearly_df["return"] > 0).mean()
  n_trades = len(trades_df)
  pf = metrics_full.get("profit_factor", 0)
  avg_trade = metrics_full.get("avg_trade_return", 0)
  mdd = metrics_full.get("max_drawdown", -100)
  cost_thresh = (TRANSACTION_COST + SLIPPAGE) * 100 * 3

  if pct_beats_spy >= 0.60:
    score += 20
  elif pct_beats_spy >= 0.50:
    score += 10
  else:
    notes.append("pocos años vs SPY")

  if pct_positive >= 0.60:
    score += 15
  elif pct_positive >= 0.50:
    score += 8

  if n_trades >= 100:
    score += 15
  elif n_trades >= 50:
    score += 8
  else:
    notes.append("pocos trades")

  if pf > 1.2:
    score += 15
  elif pf > 1.0:
    score += 8
  else:
    notes.append("profit_factor bajo")

  if avg_trade > cost_thresh:
    score += 10
  else:
    notes.append("avg trade no cubre costes x3")

  if mdd > -30:
    score += 15
  elif mdd > -40:
    score += 8
  else:
    notes.append("drawdown alto")
    score -= 10

  pos_ret = yearly_df["return"].clip(lower=0)
  if len(yearly_df) >= 3 and pos_ret.sum() > 0 and pos_ret.max() / pos_ret.sum() > 0.55:
    score -= 15
    notes.append("depende de pocos años")

  recent = yearly_df[yearly_df["year"].isin([2025, 2026])]
  if len(recent) > 0 and (recent["return"] <= 0).any():
    score -= 10
    notes.append("falla 2025/2026")

  if yearly_df["sharpe"].mean() > 0.5:
    score += 10

  score = int(np.clip(score, 0, 100))
  if score >= 75:
    status = "APPROVED_FOR_WEB_PAPER"
  elif score >= 60:
    status = "CANDIDATE"
  else:
    status = "REJECTED"
  return score, status, notes


full_metrics, _, full_eq = backtest_signal_engine(signals_oos, data_dict) if len(signals_oos) else ({}, pd.DataFrame(), pd.Series())
signal_engine_score, signal_status, score_notes = compute_signal_engine_score(yearly_df, trades_df, full_metrics)
APPROVED_FOR_WEB_PAPER = signal_status == "APPROVED_FOR_WEB_PAPER"
APPROVED_FOR_REAL_MONEY = False
print(f"Signal engine score: {signal_engine_score}/100 | {signal_status}")
if score_notes:
  print("Notas:", "; ".join(score_notes))

# %% [markdown]
# ## 12. Señales actuales

# %%
last_date = close_prices.index[-1]
ml_latest = predict_ml_snapshot(panel_df, horizon=ML_HORIZON, as_of_date=last_date)
if len(ml_latest) == 0 and len(ml_predictions_5d):
  ml_latest = ml_predictions_5d[ml_predictions_5d["date"] == ml_predictions_5d["date"].max()]
current_signals = generate_daily_signals(last_date, features_by_ticker, ml_latest)

current_signals["entry_plan"] = np.where(
  current_signals["signal"] == "BUY", "Entrada proxima apertura", "No entrar"
)
current_signals["exit_plan"] = np.where(
  current_signals["signal"] == "BUY",
  "Salir por stop / take profit / max dias",
  "-",
)

display_cols = [
  "ticker", "signal", "confidence_score", "entry_plan", "exit_plan",
  "stop_loss", "take_profit", "max_holding_days", "reason",
]
print("Señales actuales |", last_date.date())
print(current_signals[display_cols].sort_values(["signal", "confidence_score"], ascending=[True, False]).to_string(index=False))

buy_now = current_signals[current_signals["signal"] == "BUY"].sort_values("confidence_score", ascending=False)
print(f"\nBUY ahora: {', '.join(buy_now['ticker'].head(TOP_N_SIGNALS).tolist()) if len(buy_now) else 'ninguno'}")

current_signals.to_csv("current_signals_v7.csv", index=False)

# %% [markdown]
# ## 13. Comparacion benchmarks

# %%
def load_v6_equity():
  paths = [
    Path("research_outputs/v6/research_v6_equity_curves.csv"),
    Path("research_v6_equity_curves.csv"),
  ]
  for p in paths:
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
ew_cols = [c for c in close_prices.columns if c not in ("SPY", "QQQ")]
ew = close_prices[ew_cols].mean(axis=1)
ew_total = (ew.iloc[-1] / ew.iloc[0] - 1) * 100 if len(ew) >= 2 else 0

# %% [markdown]
# ## 14. Exportar resultados

# %%
def export_csv(df, name):
  if df is None or len(df) == 0:
    pd.DataFrame().to_csv(name, index=False)
  else:
    df.to_csv(name, index=False)
  print("Exportado", name)


summary_row = {
  "strategy_name": "V7 Signal Alpha Engine",
  "signal_engine_score": signal_engine_score,
  "status": signal_status,
  "approved_for_web_paper": APPROVED_FOR_WEB_PAPER,
  "approved_for_real_money": False,
  **full_metrics,
  "pct_years_beats_spy": round(yearly_df["beats_spy"].mean() * 100, 1) if len(yearly_df) else 0,
  "spy_buyhold_return": round(spy_total, 2),
  "qqq_buyhold_return": round(qqq_total, 2),
  "ew_buyhold_return": round(ew_total, 2),
  "v6_blend_return": round(v6_total, 2) if pd.notna(v6_total) else np.nan,
}
summary_df = pd.DataFrame([summary_row])

export_csv(summary_df, "research_v7_signal_summary.csv")
export_csv(trades_df, "research_v7_trades.csv")
export_csv(yearly_df, "research_v7_yearly.csv")
export_csv(signals_oos, "research_v7_daily_signals.csv")
export_csv(current_signals, "research_v7_current_signals.csv")
if len(equity_oos):
  equity_oos.to_frame("equity").to_csv("research_v7_equity_curve.csv")
  print("Exportado research_v7_equity_curve.csv")
else:
  pd.DataFrame().to_csv("research_v7_equity_curve.csv", index=False)

config_out = {
  "version": "v7_signal_alpha_lab",
  "strategy_name": "V7 Signal Alpha Engine",
  "approved_for_web_paper": APPROVED_FOR_WEB_PAPER,
  "approved_for_real_money": False,
  "signal_engine_score": signal_engine_score,
  "status": signal_status,
  "rules": {
    "combined_score_buy": 70,
    "combined_score_hold": "45-70",
    "combined_score_avoid": "<45",
    "ml_weight": 0.40,
    "momentum_weight": 0.25,
    "trend_weight": 0.20,
    "risk_weight": 0.15,
    "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
    "position_size_pct": POSITION_SIZE_PCT,
    "top_n_signals": TOP_N_SIGNALS,
    "stop_loss": "Close - 1.5 * ATR_14",
    "take_profit": "Close + 2.5 * ATR_14",
    "entry": "proxima apertura tras señal al cierre",
  },
  "warnings": [
    "Backtest no garantiza resultados futuros.",
    "No es asesoramiento financiero.",
    "No aprobado para dinero real.",
    "No conecta broker ni ejecuta ordenes.",
    "ML solo out-of-sample en walk-forward.",
  ],
  "metrics": full_metrics,
  "how_to_use": (
    "Revisar research_v7_current_signals.csv para señales del día. "
    "BUY = buscar entrada próxima apertura. SELL = salir/no comprar. "
    "Respetar stop_loss, take_profit y max_holding_days."
  ),
}
Path("research_v7_selected_signal_engine_config.json").write_text(
  json.dumps(config_out, indent=2, default=str), encoding="utf-8"
)
print("Exportado research_v7_selected_signal_engine_config.json")

# %% [markdown]
# ## 15. Reporte final

# %%
print("=" * 80)
print("REPORTE FINAL V7 SIGNAL ALPHA LAB")
print("=" * 80)
print("Disclaimer: Backtest no garantiza resultados futuros.")
print("")
print(f"1. ¿Supera a SPY? {'SI' if full_metrics.get('total_return', 0) > spy_total else 'NO'} "
      f"({full_metrics.get('total_return', 0):.2f}% vs SPY {spy_total:.2f}%)")
print(f"2. ¿Supera a QQQ? {'SI' if full_metrics.get('total_return', 0) > qqq_total else 'NO'} "
      f"({full_metrics.get('total_return', 0):.2f}% vs QQQ {qqq_total:.2f}%)")
if pd.notna(v6_total):
  print(f"3. ¿Supera a V6 blend? {'SI' if full_metrics.get('total_return', 0) > v6_total else 'NO'} "
        f"({full_metrics.get('total_return', 0):.2f}% vs V6 {v6_total:.2f}%)")
else:
  print("3. ¿Supera a V6? No disponible (falta research_v6_equity_curves.csv)")
print(f"4. Operaciones: {full_metrics.get('num_trades', 0)}")
print(f"5. Win rate: {full_metrics.get('win_rate', 0)}%")
print(f"6. Rentabilidad media por operacion: {full_metrics.get('avg_trade_return', 0)}%")
print(f"7. Peor drawdown: {full_metrics.get('max_drawdown', 0)}%")
print(f"8. Aprobado web paper: {APPROVED_FOR_WEB_PAPER} ({signal_status}, score {signal_engine_score}/100)")
print(f"9. BUY ahora: {', '.join(buy_now['ticker'].head(TOP_N_SIGNALS).tolist()) if len(buy_now) else 'ninguno'}")
print("10. Como vender: senal SELL, stop_loss, take_profit, max_holding_days o score < 45")
print("")
if APPROVED_FOR_WEB_PAPER:
  print("Integrar como V7 Signal Engine Paper Trading.")
else:
  print("No integrar todavía. Seguir investigando.")
print(f"APPROVED_FOR_REAL_MONEY={APPROVED_FOR_REAL_MONEY} (siempre False)")
