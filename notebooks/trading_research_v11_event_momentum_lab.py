# %% [markdown]
# # Trading Research V11 Event Momentum / Earnings Drift Lab
#
# Motor event-driven: earnings proxy gap, gap-and-go, breakout, relative strength.
#
# **Disclaimer:** Backtest no garantiza resultados futuros. No es asesoramiento financiero.
# **No conecta broker. No ejecuta ordenes. APPROVED_FOR_REAL_MONEY siempre False.**

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
  from sklearn.pipeline import make_pipeline
  from sklearn.impute import SimpleImputer
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
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
MAX_CONCURRENT_POSITIONS = 5
POSITION_SIZE_PCT = 0.10
HOLDING_DAYS = [5, 10, 20, 40]
STOP_ATR_MULTIPLIERS = [1.5, 2.0]
TAKE_ATR_MULTIPLIERS = [2.0, 3.0, 4.0]
WF_START_YEAR = 2019
EMBARGO_DAYS = 20
BUY_THRESHOLD = 0.60
WATCH_THRESHOLD = 0.50
THRESHOLD_GRID = [0.55, 0.60, 0.65, 0.70]
DEFAULT_HOLD = 20
DEFAULT_STOP_ATR = 2.0
DEFAULT_TAKE_ATR = 3.0
MIN_DOLLAR_VOL = 5_000_000
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
  "PANW", "NOW", "SHOP", "UBER", "C",
]

SECTOR_MAP = {
  "AAPL": "mega_tech", "MSFT": "mega_tech", "NVDA": "semis", "AMD": "semis",
  "AVGO": "semis", "INTC": "semis", "MU": "semis", "GOOGL": "internet",
  "META": "internet", "AMZN": "internet", "NFLX": "internet", "ADBE": "software",
  "CRM": "software", "ORCL": "software", "JPM": "banks", "BAC": "banks",
  "GS": "banks", "MS": "banks", "C": "banks", "XOM": "energy", "CVX": "energy",
  "UNH": "healthcare", "LLY": "healthcare", "JNJ": "healthcare",
  "WMT": "consumer", "COST": "consumer", "HD": "consumer", "MCD": "consumer",
  "PANW": "software", "NOW": "software", "SHOP": "internet", "UBER": "internet",
}

EVENT_TYPES = [
  "earnings_proxy_gap", "gap_and_go", "breakout_volume",
  "relative_strength_leader", "pullback_after_momentum",
]

if QUICK_TEST:
  UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META", "AMZN", "JPM"]
  START_DATE = "2020-01-01"
  WF_START_YEAR = 2021
  HOLDING_DAYS = [10, 20]
  STOP_ATR_MULTIPLIERS = [2.0]
  TAKE_ATR_MULTIPLIERS = [3.0]
  print("QUICK_TEST activo")

print("V11 Event Momentum Lab | desde", START_DATE)
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
    print("Fallidos:", failed[:12])
  close = pd.DataFrame({k: v["Close"] for k, v in data.items()}).sort_index().ffill()
  close.index = pd.DatetimeIndex(close.index)
  if close.index.tz:
    close.index = close.index.tz_localize(None)
  return data, close


data_dict, close_prices = download_data(UNIVERSE, START_DATE, END_DATE)
print("Tickers:", len(data_dict), "| dias:", len(close_prices))

# %% [markdown]
# ## 3. Features de eventos

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


def _single_event_features(df):
  d = df.copy()
  c, o, h, l, v = d["Close"], d["Open"], d["High"], d["Low"], d.get("Volume", pd.Series(0, index=d.index))
  prev_c = c.shift(1)
  d["GAP_OPEN"] = o / prev_c - 1
  d["GAP_CLOSE"] = c / prev_c - 1
  d["INTRADAY_RETURN"] = c / o.replace(0, np.nan) - 1
  d["GAP_AND_GO"] = ((d["GAP_OPEN"] > 0.02) & (d["INTRADAY_RETURN"] > 0)).astype(float)
  for n in [1, 3, 5, 10, 20, 60, 120]:
    d[f"RET_{n}D"] = c.pct_change(n)
  for n in [20, 60, 120]:
    d[f"MOM_{n}"] = c.pct_change(n)
  d["MOM_SKIP_5_60"] = c.shift(5) / c.shift(65) - 1
  for n in [20, 50, 100, 200]:
    d[f"SMA_{n}"] = c.rolling(n).mean()
  for span in [21, 50]:
    d[f"EMA_{span}"] = c.ewm(span=span, adjust=False).mean()
  d["DIST_SMA50"] = c / d["SMA_50"] - 1
  d["DIST_SMA200"] = c / d["SMA_200"] - 1
  d["TREND_OK"] = ((c > d["SMA_50"]) & (d["SMA_50"] > d["SMA_200"])).astype(float)
  for n in [10, 20, 60]:
    d[f"VOL_{n}"] = c.pct_change().rolling(n).std() * np.sqrt(252)
  d["ATR_14"] = _atr(d, 14)
  d["ATR_PCT"] = d["ATR_14"] / c.replace(0, np.nan)
  d["VOLUME_AVG_20"] = v.rolling(20).mean()
  d["VOLUME_RATIO"] = v / d["VOLUME_AVG_20"].replace(0, np.nan)
  d["DOLLAR_VOLUME"] = c * v
  d["LIQUID"] = (d["DOLLAR_VOLUME"].rolling(20).mean() > MIN_DOLLAR_VOL).astype(float)
  for n in [20, 60]:
    d[f"HIGH_{n}_PREV"] = h.rolling(n).max().shift(1)
  d["BREAKOUT_20"] = (c > d["HIGH_20_PREV"]).astype(float)
  d["BREAKOUT_60"] = (c > d["HIGH_60_PREV"]).astype(float)
  d["RSI_2"] = _rsi(c, 2)
  d["RSI_14"] = _rsi(c, 14)
  return d


def _market_overlay(data_dict):
  out = {}
  for mkt, prefix in [(MARKET, "SPY"), ("QQQ", "QQQ")]:
    df = data_dict.get(mkt)
    if df is None:
      continue
    f = _single_event_features(df.copy())
    out[f"{prefix}_ABOVE_SMA200"] = (f["Close"] > f["SMA_200"]).astype(float)
    out[f"{prefix}_RET_20"] = f["RET_20D"]
    out[f"{prefix}_RET_60"] = f["RET_60D"]
  if out:
    mdf = pd.DataFrame(out)
    mdf["MARKET_RISK_ON"] = (
      (mdf.get("SPY_ABOVE_SMA200", 0) >= 0.5) & (mdf.get("QQQ_ABOVE_SMA200", 0) >= 0.5)
    ).astype(float)
    return mdf
  return pd.DataFrame()


def _add_cross_sectional(features_by_ticker, sector_map):
  tickers = [t for t in features_by_ticker if t not in ("SPY", "QQQ", "IWM")]
  if not tickers:
    return features_by_ticker
  dates = features_by_ticker[tickers[0]].index
  ret20 = pd.DataFrame({t: features_by_ticker[t]["RET_20D"] for t in tickers}, index=dates)
  ret60 = pd.DataFrame({t: features_by_ticker[t]["RET_60D"] for t in tickers}, index=dates)
  mom_vol = pd.DataFrame({
    t: features_by_ticker[t]["MOM_60"] / features_by_ticker[t]["VOL_20"].replace(0, np.nan)
    for t in tickers
  }, index=dates)
  rank20 = ret20.rank(axis=1, pct=True)
  rank60 = ret60.rank(axis=1, pct=True)
  rank_mv = mom_vol.rank(axis=1, pct=True)
  for t in tickers:
    idx = features_by_ticker[t].index
    features_by_ticker[t]["rank_ret_20"] = rank20[t].reindex(idx)
    features_by_ticker[t]["rank_ret_60"] = rank60[t].reindex(idx)
    features_by_ticker[t]["rank_momentum_vol"] = rank_mv[t].reindex(idx)
    sec = sector_map.get(t, "other")
    peers = [p for p in tickers if sector_map.get(p) == sec]
    s20 = ret20[peers].mean(axis=1)
    s60 = ret60[peers].mean(axis=1)
    features_by_ticker[t]["sector_relative_ret_20"] = ret20[t] - s20
    features_by_ticker[t]["sector_relative_ret_60"] = ret60[t] - s60
    features_by_ticker[t]["rank_within_sector"] = ret60[peers].rank(axis=1, pct=True)[t]
  return features_by_ticker


def add_event_features(data_dict, close_prices, sector_map=None):
  sector_map = sector_map or SECTOR_MAP
  market = _market_overlay(data_dict)
  features = {}
  for ticker, df in data_dict.items():
    d = _single_event_features(df.copy())
    if len(market):
      for col in market.columns:
        d[col] = market[col].reindex(d.index).ffill()
    features[ticker] = d
  return _add_cross_sectional(features, sector_map)


features_by_ticker = add_event_features(data_dict, close_prices, SECTOR_MAP)
print("Features eventos:", len(features_by_ticker), "tickers")

# %% [markdown]
# ## 4. Deteccion de eventos

# %%
def _event_score(parts):
  return float(np.clip(sum(parts), 0, 100))


def detect_events(features_by_ticker, tickers=None):
  tickers = tickers or [t for t in features_by_ticker if t not in ("SPY", "QQQ", "IWM")]
  rows = []
  for ticker in tickers:
    df = features_by_ticker.get(ticker)
    if df is None:
      continue
    for dt, row in df.iterrows():
      if _sf(row.get("LIQUID"), 0) < 0.5:
        continue
      events_today = []
      gc = _sf(row.get("GAP_CLOSE"), 0)
      go = _sf(row.get("GAP_OPEN"), 0)
      ir = _sf(row.get("INTRADAY_RETURN"), 0)
      vr = _sf(row.get("VOLUME_RATIO"), 0)
      c = _sf(row.get("Close"))
      sma50 = _sf(row.get("SMA_50"))
      sma100 = _sf(row.get("SMA_100"))
      sma200 = _sf(row.get("SMA_200"))
      ema21 = _sf(row.get("EMA_21"), c)
      risk_on = _sf(row.get("MARKET_RISK_ON"), 0) >= 0.5
      r60 = _sf(row.get("rank_ret_60"), 0)
      rsi14 = _sf(row.get("RSI_14"), 50)
      rsi2 = _sf(row.get("RSI_2"), 50)
      mom20 = _sf(row.get("MOM_20"), 0)
      mom_skip = _sf(row.get("MOM_SKIP_5_60"), 0)

      if gc > 0.04 and vr > 1.5 and c > sma50 and risk_on:
        sc = _event_score([25, min(gc * 200, 25), min(vr * 10, 20), 20 if risk_on else 0])
        events_today.append(("earnings_proxy_gap", sc))
      if go > 0.02 and ir > 0.01 and vr > 1.2 and c > sma50:
        sc = _event_score([20, min(go * 150, 20), min(ir * 200, 20), min(vr * 8, 15)])
        events_today.append(("gap_and_go", sc))
      if _sf(row.get("BREAKOUT_20"), 0) >= 0.5 and vr > 1.2 and mom20 > 0 and 50 <= rsi14 <= 80:
        sc = _event_score([25, min(vr * 10, 20), min(mom20 * 100, 20), 15])
        events_today.append(("breakout_volume", sc))
      if r60 >= 0.80 and c > sma100 and mom_skip > 0 and risk_on:
        sc = _event_score([25, r60 * 25, 20 if mom_skip > 0 else 0, 15 if risk_on else 0])
        events_today.append(("relative_strength_leader", sc))
      if r60 >= 0.70 and c > sma200 and rsi2 < 20 and abs(c - ema21) / max(c, 1) < 0.04 and risk_on:
        sc = _event_score([20, 20, 15, 15 if rsi2 < 15 else 8])
        events_today.append(("pullback_after_momentum", sc))

      if not events_today:
        continue
      event_type, score = max(events_today, key=lambda x: x[1])
      rec = {k: _sf(v) for k, v in row.items() if isinstance(v, (int, float, np.integer, np.floating))}
      rec.update({
        "date": dt, "ticker": ticker, "event_type": event_type,
        "raw_event_score": round(score, 1), "sector": SECTOR_MAP.get(ticker, "other"),
      })
      rows.append(rec)
  events_df = pd.DataFrame(rows)
  if len(events_df):
    events_df["date"] = pd.to_datetime(events_df["date"])
  return events_df


events_df = detect_events(features_by_ticker)
print("Eventos detectados:", len(events_df))
if len(events_df):
  print(events_df["event_type"].value_counts().head())

# %% [markdown]
# ## 5. Labels de eventos (solo train/eval)

# %%
def _spy_risk_on(dt):
  spy = features_by_ticker.get(MARKET)
  if spy is None or dt not in spy.index:
    return True
  return _sf(spy.loc[dt].get("SPY_ABOVE_SMA200"), 1) >= 0.5


def simulate_event_trade(data_dict, ticker, event_date, hold_days, stop_atr, take_atr, check_market=True):
  df = data_dict.get(ticker)
  if df is None or event_date not in df.index:
    return None
  future = df.index[df.index > event_date]
  if len(future) == 0:
    return None
  entry_dt = future[0]
  entry_row = df.loc[entry_dt]
  entry_px = _sf(entry_row.get("Open"), _sf(entry_row.get("Close")))
  if not np.isfinite(entry_px) or entry_px <= 0:
    return None
  atr_row = features_by_ticker[ticker].loc[event_date] if ticker in features_by_ticker and event_date in features_by_ticker[ticker].index else None
  atr = _sf(atr_row.get("ATR_14") if atr_row is not None else entry_px * 0.02, entry_px * 0.02)
  stop_px = entry_px - stop_atr * atr
  take_px = entry_px + take_atr * atr
  cost = TRANSACTION_COST + SLIPPAGE
  label_win, label_loss, exit_reason, realized, hold_real = 0, 0, "max_hold", 0.0, hold_days
  path = df.loc[entry_dt:].head(hold_days + 1)
  for i, (dt, row) in enumerate(path.iterrows()):
    if i == 0:
      continue
    if check_market and not _spy_risk_on(dt):
      cl = _sf(row.get("Close"), entry_px)
      realized = cl / entry_px - 1 - 2 * cost
      exit_reason, hold_real = "market_regime", i
      break
    lo, hi, cl = _sf(row.get("Low")), _sf(row.get("High")), _sf(row.get("Close"))
    if lo <= stop_px:
      label_loss, exit_reason = 1, "stop_loss"
      realized = stop_px / entry_px - 1 - 2 * cost
      hold_real = i
      break
    if hi >= take_px:
      label_win, exit_reason = 1, "take_profit"
      realized = take_px / entry_px - 1 - 2 * cost
      hold_real = i
      break
  else:
    if len(path) > 1:
      cl = _sf(path["Close"].iloc[-1], entry_px)
      realized = cl / entry_px - 1 - 2 * cost
      exit_reason = "max_hold"
      hold_real = len(path) - 1
  return {
    "label_win": label_win, "label_loss": label_loss,
    "realized_return": round(realized * 100, 3),
    "exit_reason": exit_reason, "holding_days_real": hold_real,
    "entry_date": entry_dt, "entry_price": entry_px,
    "hold_param": hold_days, "stop_atr": stop_atr, "take_atr": take_atr,
  }


def select_best_params_on_events(events_sub, data_dict):
  """Elige hold/stop/take solo con eventos de train (sin mirar test)."""
  best_score, best = -999, (DEFAULT_HOLD, DEFAULT_STOP_ATR, DEFAULT_TAKE_ATR)
  if events_sub is None or len(events_sub) < 10:
    return best
  for hold, stop, take in product(HOLDING_DAYS, STOP_ATR_MULTIPLIERS, TAKE_ATR_MULTIPLIERS):
    sims = []
    for _, ev in events_sub.iterrows():
      s = simulate_event_trade(data_dict, ev["ticker"], ev["date"], hold, stop, take)
      if s:
        sims.append(s)
    if len(sims) < 8:
      continue
    rets = pd.Series([s["realized_return"] for s in sims])
    wins = rets[rets > 0].sum()
    losses = abs(rets[rets <= 0].sum())
    pf = wins / losses if losses > 0 else 0
    wr = (rets > 0).mean()
    score = pf + wr * 0.5 - (0.05 if len(sims) < 20 else 0)
    if score > best_score:
      best_score, best = score, (hold, stop, take)
  return best


def label_events(events_df, data_dict, hold=None, stop_atr=None, take_atr=None, train_end=None):
  if events_df is None or len(events_df) == 0:
    return pd.DataFrame()
  train_events = events_df[events_df["date"] <= train_end] if train_end is not None else events_df
  if hold is None:
    hold, stop_atr, take_atr = select_best_params_on_events(train_events, data_dict)
  labeled = []
  for _, ev in events_df.iterrows():
    sim = simulate_event_trade(data_dict, ev["ticker"], ev["date"], hold, stop_atr, take_atr)
    if sim is None:
      continue
    labeled.append({**ev.to_dict(), **sim})
  out = pd.DataFrame(labeled)
  if len(out):
    out["target_win"] = out["label_win"].astype(int)
  return out, (hold, stop_atr, take_atr)


train_cutoff = pd.Timestamp(f"{WF_START_YEAR}-01-01") - pd.Timedelta(days=EMBARGO_DAYS + 1)
labeled_events, selected_params = label_events(events_df, data_dict, train_end=train_cutoff)
DEFAULT_HOLD, DEFAULT_STOP_ATR, DEFAULT_TAKE_ATR = selected_params
print("Params elegidos en train:", selected_params)
print("Eventos etiquetados:", len(labeled_events),
      "| win rate:", round(labeled_events["label_win"].mean() * 100, 1) if len(labeled_events) else 0, "%")

# %% [markdown]
# ## 6. Baseline por tipo de evento

# %%
def backtest_event_strategy(events_sub, data_dict, score_col="raw_event_score", min_score=55, hold=DEFAULT_HOLD,
                            stop_atr=DEFAULT_STOP_ATR, take_atr=DEFAULT_TAKE_ATR):
  if events_sub is None or len(events_sub) == 0:
    return pd.DataFrame()
  cost = TRANSACTION_COST + SLIPPAGE
  trades = []
  events_sub = events_sub.sort_values("date")
  open_pos = {}
  for _, ev in events_sub.iterrows():
    dt = ev["date"]
    if _sf(ev.get(score_col), 0) < min_score:
      continue
    ticker = ev["ticker"]
    if ticker in open_pos:
      continue
    if len(open_pos) >= MAX_CONCURRENT_POSITIONS:
      continue
    sim = simulate_event_trade(data_dict, ticker, dt, hold, stop_atr, take_atr)
    if sim is None:
      continue
    alloc = INITIAL_CAPITAL * POSITION_SIZE_PCT
    ret = sim["realized_return"]
    trades.append({
      "ticker": ticker, "event_type": ev["event_type"], "event_date": dt,
      "entry_date": sim["entry_date"], "entry_price": sim["entry_price"],
      "exit_date": sim["entry_date"] + pd.Timedelta(days=sim["holding_days_real"]),
      "exit_reason": sim["exit_reason"], "holding_days": sim["holding_days_real"],
      "return_pct": ret, "pnl": round(alloc * ret / 100, 2),
      "score_entry": _sf(ev.get(score_col)), "setup_type": ev["event_type"],
    })
    open_pos[ticker] = True
  return pd.DataFrame(trades)


baseline_by_type = {}
for et in EVENT_TYPES:
  sub = labeled_events[labeled_events["event_type"] == et] if len(labeled_events) else pd.DataFrame()
  baseline_by_type[et] = backtest_event_strategy(sub, data_dict)
  n = len(baseline_by_type[et])
  wr = (baseline_by_type[et]["return_pct"] > 0).mean() * 100 if n else 0
  print(f"  {et}: {n} trades | win {wr:.1f}%")

# %% [markdown]
# ## 7. ML meta-labeling sobre eventos

# %%
LABEL_COLS = {
  "label_win", "label_loss", "realized_return", "exit_reason", "holding_days_real",
  "entry_date", "entry_price", "target_win", "hold_param", "stop_atr", "take_atr",
  "date", "ticker", "event_type", "raw_event_score", "sector",
}


def get_event_feature_cols(df):
  exclude = LABEL_COLS | {"Open", "High", "Low", "Close", "Volume"}
  return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


EVENT_FEATURE_COLS = get_event_feature_cols(labeled_events) if len(labeled_events) else get_event_feature_cols(
  pd.DataFrame([{k: _sf(v) for k, v in features_by_ticker["AAPL"].iloc[-1].items() if isinstance(v, (int, float, np.integer, np.floating))}])
  if "AAPL" in features_by_ticker else []
)


def clean_matrix(df, cols):
  X = df.copy()
  for c in cols:
    if c not in X.columns:
      X[c] = 0.0
  X = X[cols].apply(pd.to_numeric, errors="coerce")
  return X.replace([np.inf, -np.inf], np.nan)


def fit_event_models(train_df, test_df, feature_cols):
  if not HAS_SKLEARN or len(train_df) < 40 or len(test_df) == 0:
    test_out = test_df.copy()
    test_out["probability_win"] = test_df["raw_event_score"] / 100.0
    return test_out
  X = clean_matrix(train_df, feature_cols)
  y = train_df["target_win"].astype(int)
  Xt = clean_matrix(test_df, feature_cols)
  models = [
    make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                  LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_SEED)),
    make_pipeline(SimpleImputer(strategy="median"),
                  RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced_subsample", random_state=RANDOM_SEED, n_jobs=-1)),
    make_pipeline(SimpleImputer(strategy="median"),
                  HistGradientBoostingClassifier(max_depth=4, max_iter=120, random_state=RANDOM_SEED)),
  ]
  probs = []
  for m in models:
    try:
      m.fit(X, y)
      probs.append(m.predict_proba(Xt)[:, 1])
    except Exception:
      pass
  out = test_df.copy()
  out["probability_win"] = np.mean(probs, axis=0) if probs else test_df["raw_event_score"].values / 100.0
  if "target_win" in out.columns:
    out["true_label"] = out["target_win"]
  if "realized_return" in out.columns:
    out["forward_return"] = out["realized_return"]
  return out


def select_threshold_train(val_df, col="probability_win"):
  best_th, best = BUY_THRESHOLD, -999
  for th in THRESHOLD_GRID:
    sub = val_df[val_df[col] >= th]
    if len(sub) < 8:
      continue
    wins = sub[sub.get("target_win", sub.get("label_win", 0)) == 1]
    losses = sub[sub.get("target_win", sub.get("label_win", 0)) != 1]
    gw = wins["realized_return"].clip(lower=0).sum() if len(wins) and "realized_return" in wins else 0
    gl = abs(losses["realized_return"].clip(upper=0).sum()) if len(losses) and "realized_return" in losses else 0
    pf = gw / gl if gl > 0 else 0
    score = pf + len(sub) * 0.01
    if score > best:
      best, best_th = score, th
  return best_th


def run_ml_walk_forward_events(labeled_df, feature_cols, start_year=WF_START_YEAR):
  if not RUN_ML or len(labeled_df) == 0:
    labeled_df = labeled_df.copy()
    labeled_df["probability_win"] = labeled_df["raw_event_score"] / 100.0
    labeled_df["year"] = labeled_df["date"].dt.year
    return labeled_df, []
  preds, th_log = [], []
  current_year = pd.Timestamp.today().year
  for year in range(start_year, current_year + 1):
    test_start = pd.Timestamp(f"{year}-01-01")
    train_end = test_start - pd.Timedelta(days=EMBARGO_DAYS + 1)
    train = labeled_df[labeled_df["date"] <= train_end].copy()
    test = labeled_df[(labeled_df["date"] >= test_start) & (labeled_df["date"] <= pd.Timestamp(f"{year}-12-31"))].copy()
    if len(train) < 50 or len(test) < 5:
      continue
    split = train["date"].quantile(0.8)
    tr_fit = train[train["date"] < split]
    tr_val = train[train["date"] >= split]
    val_pred = fit_event_models(tr_fit, tr_val, feature_cols)
    th = select_threshold_train(val_pred) if len(val_pred) else BUY_THRESHOLD
    th_log.append({"year": year, "threshold": th})
    test_pred = fit_event_models(train, test, feature_cols)
    test_pred["year"] = year
    test_pred["threshold_used"] = th
    preds.append(test_pred)
  if not preds:
    out = labeled_df.copy()
    out["probability_win"] = out["raw_event_score"] / 100.0
    return out, th_log
  return pd.concat(preds, ignore_index=True), th_log


# ML walk-forward integrado en run_v11_walk_forward (seccion 9)
threshold_log = []

# %% [markdown]
# ## 8. Backtest event engine

# %%
def compute_metrics(trades_df, equity, initial_capital=INITIAL_CAPITAL):
  if trades_df is None or len(trades_df) == 0:
    return {"num_trades": 0, "total_return": 0, "sharpe": 0, "profit_factor": 0, "win_rate": 0}
  if equity is None or len(equity) < 2:
    equity = initial_capital + trades_df.sort_values("exit_date")["pnl"].cumsum()
    equity = pd.Series(equity.values, index=trades_df.sort_values("exit_date")["exit_date"])
  rets = equity.pct_change().fillna(0)
  total_ret = equity.iloc[-1] / initial_capital - 1
  years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
  cagr = (equity.iloc[-1] / initial_capital) ** (1 / years) - 1
  peak = equity.cummax()
  mdd = ((equity - peak) / peak.replace(0, np.nan)).min()
  sh = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
  ds = rets[rets < 0].std()
  so = rets.mean() / ds * np.sqrt(252) if ds and ds > 0 else 0
  wins = trades_df[trades_df["return_pct"] > 0]
  losses = trades_df[trades_df["return_pct"] <= 0]
  pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) and losses["pnl"].sum() != 0 else 0
  return {
    "total_return": round(total_ret * 100, 2),
    "CAGR": round(cagr * 100, 2),
    "sharpe": round(sh, 3),
    "sortino": round(so, 3),
    "max_drawdown": round(mdd * 100, 2),
    "calmar": round(cagr / abs(mdd), 3) if mdd != 0 else 0,
    "profit_factor": round(pf, 3),
    "expectancy": round(trades_df["pnl"].mean(), 2),
    "num_trades": int(len(trades_df)),
    "win_rate": round(len(wins) / len(trades_df) * 100, 2),
    "avg_trade_return": round(trades_df["return_pct"].mean(), 3),
    "median_trade_return": round(trades_df["return_pct"].median(), 3),
    "avg_holding_days": round(trades_df["holding_days"].mean(), 1),
    "trades_per_year": round(len(trades_df) / years, 1),
    "exposure_pct": round(POSITION_SIZE_PCT * MAX_CONCURRENT_POSITIONS * 100, 1),
  }


def backtest_event_engine(events_pred, data_dict, score_col="probability_win", min_score=None):
  if events_pred is None or len(events_pred) == 0:
    return pd.DataFrame(), pd.Series(dtype=float), {}
  events_pred = events_pred.sort_values("date")
  trades = []
  active = {}
  cost = TRANSACTION_COST + SLIPPAGE
  for dt, day in events_pred.groupby("date"):
    th = _sf(day["threshold_used"].iloc[0], BUY_THRESHOLD) if "threshold_used" in day.columns else (min_score or BUY_THRESHOLD)
    for _, ev in day.iterrows():
      sc = _sf(ev.get(score_col), _sf(ev.get("raw_event_score"), 0) / 100.0)
      if score_col == "probability_win" and sc < th:
        continue
      if score_col == "raw_event_score" and sc < (min_score or 55):
        continue
      ticker = ev["ticker"]
      if ticker in active or len(active) >= MAX_CONCURRENT_POSITIONS:
        continue
      sim = simulate_event_trade(data_dict, ticker, ev["date"], DEFAULT_HOLD, DEFAULT_STOP_ATR, DEFAULT_TAKE_ATR)
      if sim is None:
        continue
      alloc = INITIAL_CAPITAL * POSITION_SIZE_PCT
      trades.append({
        "ticker": ticker, "event_type": ev["event_type"], "event_date": ev["date"],
        "entry_date": sim["entry_date"], "entry_price": sim["entry_price"],
        "exit_date": sim["entry_date"] + pd.Timedelta(days=sim["holding_days_real"]),
        "exit_reason": sim["exit_reason"], "holding_days": sim["holding_days_real"],
        "return_pct": sim["realized_return"], "pnl": round(alloc * sim["realized_return"] / 100, 2),
        "probability_win_entry": sc, "raw_event_score": _sf(ev.get("raw_event_score")),
        "setup_type": ev["event_type"],
      })
      active[ticker] = sim["entry_date"] + pd.Timedelta(days=sim["holding_days_real"])
    active = {t: d for t, d in active.items() if d > dt}
  trades_df = pd.DataFrame(trades)
  if len(trades_df):
    eq = INITIAL_CAPITAL + trades_df.sort_values("exit_date")["pnl"].cumsum()
    equity = pd.Series(eq.values, index=trades_df.sort_values("exit_date")["exit_date"])
  else:
    equity = pd.Series(dtype=float)
  return trades_df, equity, compute_metrics(trades_df, equity)

# %% [markdown]
# ## 9. Walk-forward completo

# %%
def benchmark_year(close_prices, year, col):
  if col not in close_prices.columns:
    return 0.0
  s = close_prices[col].loc[f"{year}-01-01":f"{year}-12-31"]
  return (s.iloc[-1] / s.iloc[0] - 1) * 100 if len(s) >= 2 else 0.0


def run_v11_walk_forward():
  global DEFAULT_HOLD, DEFAULT_STOP_ATR, DEFAULT_TAKE_ATR, threshold_log
  yearly_rows, all_trades, wf_oos_events = [], [], []
  threshold_log = []
  if len(events_df) == 0:
    return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), {}, pd.DataFrame()
  current_year = pd.Timestamp.today().year
  for year in range(WF_START_YEAR, current_year + 1):
    test_start = pd.Timestamp(f"{year}-01-01")
    train_end = test_start - pd.Timedelta(days=EMBARGO_DAYS + 1)
    train_ev = events_df[events_df["date"] <= train_end]
    test_ev = events_df[(events_df["date"] >= test_start) & (events_df["date"] <= pd.Timestamp(f"{year}-12-31"))]
    if len(test_ev) < 3:
      continue
    hold, stop, take = select_best_params_on_events(train_ev, data_dict)
    yr_labeled, _ = label_events(test_ev, data_dict, hold=hold, stop_atr=stop, take_atr=take)
    if RUN_ML and HAS_SKLEARN and len(train_ev) >= 50:
      train_labeled, _ = label_events(train_ev, data_dict, hold=hold, stop_atr=stop, take_atr=take)
      split = train_labeled["date"].quantile(0.8) if len(train_labeled) > 20 else train_end
      tr_fit = train_labeled[train_labeled["date"] < split]
      tr_val = train_labeled[train_labeled["date"] >= split]
      val_pred = fit_event_models(tr_fit, tr_val, EVENT_FEATURE_COLS)
      th = select_threshold_train(val_pred) if len(val_pred) else BUY_THRESHOLD
      threshold_log.append({"year": year, "threshold": th, "hold": hold, "stop_atr": stop, "take_atr": take})
      test_pred = fit_event_models(train_labeled, yr_labeled, EVENT_FEATURE_COLS)
      test_pred["year"] = year
      test_pred["threshold_used"] = th
      wf_oos_events.append(test_pred)
      yr_for_bt = test_pred
    else:
      yr_labeled = yr_labeled.copy()
      yr_labeled["probability_win"] = yr_labeled["raw_event_score"] / 100.0
      yr_labeled["year"] = year
      yr_labeled["threshold_used"] = BUY_THRESHOLD
      threshold_log.append({"year": year, "threshold": BUY_THRESHOLD, "hold": hold, "stop_atr": stop, "take_atr": take})
      wf_oos_events.append(yr_labeled)
      yr_for_bt = yr_labeled
    DEFAULT_HOLD, DEFAULT_STOP_ATR, DEFAULT_TAKE_ATR = hold, stop, take
    tr, eq, m = backtest_event_engine(yr_for_bt, data_dict)
    if len(tr):
      all_trades.append(tr.assign(year=year))
    spy_ret = benchmark_year(close_prices, year, MARKET)
    yearly_rows.append({
      "year": year, "return": m.get("total_return", 0), "SPY_return": round(spy_ret, 2),
      "num_trades": m.get("num_trades", 0), "win_rate": m.get("win_rate", 0),
      "profit_factor": m.get("profit_factor", 0), "sharpe": m.get("sharpe", 0),
      "max_drawdown": m.get("max_drawdown", 0), "beats_spy": m.get("total_return", 0) > spy_ret,
      "hold_days": hold, "stop_atr": stop, "take_atr": take,
    })
  oos_combined = pd.concat(wf_oos_events, ignore_index=True) if wf_oos_events else pd.DataFrame()
  yearly_df = pd.DataFrame(yearly_rows)
  trades_df, equity, full_metrics = backtest_event_engine(oos_combined, data_dict) if len(oos_combined) else (pd.DataFrame(), pd.Series(dtype=float), {})
  return yearly_df, trades_df, equity, full_metrics, oos_combined


yearly_df, trades_df, equity_oos, full_metrics, oos_events = run_v11_walk_forward()
print("WF años:", len(yearly_df), "| trades OOS:", len(trades_df))
if full_metrics:
  print("Return:", full_metrics.get("total_return"), "% | PF:", full_metrics.get("profit_factor"))

# %% [markdown]
# ## 10. Diagnostico

# %%
def perf_group(df, col):
  if df is None or len(df) == 0:
    return pd.DataFrame()
  return df.groupby(col).agg(
    n=("return_pct", "count"),
    win_rate=("return_pct", lambda x: (x > 0).mean() * 100),
    avg_return=("return_pct", "mean"),
    total_pnl=("pnl", "sum"),
  ).reset_index()


event_type_perf = perf_group(trades_df, "event_type")
sector_perf = perf_group(trades_df.assign(sector=trades_df["ticker"].map(SECTOR_MAP)), "sector") if len(trades_df) else pd.DataFrame()
ticker_perf = perf_group(trades_df, "ticker")

def signal_quality_bucket(events_pred):
  if events_pred is None or len(events_pred) == 0 or "probability_win" not in events_pred.columns:
    return pd.DataFrame()
  p = events_pred.copy()
  p["bucket"] = pd.cut(p["probability_win"], bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0], include_lowest=True)
  agg = {"n": ("probability_win", "count")}
  if "target_win" in p.columns:
    agg["win_rate"] = ("target_win", "mean")
  if "realized_return" in p.columns:
    agg["avg_return"] = ("realized_return", "mean")
  return p.groupby("bucket", observed=True).agg(**{k: v for k, v in agg.items()}).reset_index()


signal_quality = signal_quality_bucket(oos_events)
best_event_type = event_type_perf.sort_values("avg_return", ascending=False).iloc[0]["event_type"] if len(event_type_perf) else "none"

# %% [markdown]
# ## 11. Score final

# %%
def compute_event_engine_score(yearly_df, trades_df, metrics, ticker_perf_df):
  score, notes = 0, []
  pf = metrics.get("profit_factor", 0)
  sh = metrics.get("sharpe", 0)
  mdd = metrics.get("max_drawdown", -100)
  n = metrics.get("num_trades", 0)
  wr = metrics.get("win_rate", 0)
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
  if wr > 52:
    score += 10
  if avg > cost_thresh:
    score += 10
  if beats_spy >= 0.5:
    score += 10
  if len(ticker_perf_df) >= 3:
    top = ticker_perf_df["total_pnl"].clip(lower=0).max()
    tot = ticker_perf_df["total_pnl"].clip(lower=0).sum()
    if tot > 0 and top / tot < 0.45:
      score += 10
    else:
      score -= 20
      notes.append("concentracion en un ticker")
  if pf < 1:
    score -= 20
  if mdd < -30:
    score -= 20
  recent = yearly_df[yearly_df["year"].isin([2025, 2026])] if len(yearly_df) else pd.DataFrame()
  if len(recent) and (recent["return"] < -8).any():
    score -= 20
    notes.append("2025/2026 debiles")
  if len(trades_df) and (trades_df["ticker"].isin(["NVDA", "TSLA"]).mean() > 0.35):
    score -= 20
    notes.append("concentrado NVDA/TSLA")
  score = int(np.clip(score, 0, 100))
  status = "APPROVED_FOR_WEB_PAPER" if score >= 75 else ("CANDIDATE" if score >= 60 else "REJECTED")
  return score, status, notes


event_engine_score, event_status, score_notes = compute_event_engine_score(
  yearly_df, trades_df, full_metrics, ticker_perf
)
APPROVED_FOR_WEB_PAPER = event_status == "APPROVED_FOR_WEB_PAPER"
print(f"V11 score: {event_engine_score}/100 | {event_status}")

# %% [markdown]
# ## 12. Señales actuales (sin columnas futuras)

# %%
def build_current_signals_v11(features_by_ticker, data_dict, labeled_history, feature_cols):
  last_date = close_prices.index[-1]
  recent_events = detect_events(features_by_ticker)
  if len(recent_events) == 0:
    return pd.DataFrame()
  recent_events = recent_events[recent_events["date"] >= last_date - pd.Timedelta(days=5)]
  if len(recent_events) == 0:
    return pd.DataFrame()

  train_end = last_date - pd.Timedelta(days=EMBARGO_DAYS)
  train = labeled_history[labeled_history["date"] <= train_end] if len(labeled_history) else pd.DataFrame()
  if RUN_ML and HAS_SKLEARN and len(train) >= 40:
    preds = fit_event_models(train, recent_events, feature_cols)
    prob_col = "probability_win"
  else:
    preds = recent_events.copy()
    preds["probability_win"] = preds["raw_event_score"] / 100.0
    prob_col = "probability_win"

  rows = []
  for _, ev in preds.iterrows():
    ticker = ev["ticker"]
    prob = _sf(ev.get(prob_col), _sf(ev.get("raw_event_score"), 50) / 100.0)
    conf = int(np.clip(prob * 100, 0, 100))
    df = data_dict.get(ticker)
    if df is None or ev["date"] not in df.index:
      continue
    row = features_by_ticker[ticker].loc[ev["date"]]
    close = _sf(row.get("Close"))
    atr = _sf(row.get("ATR_14"), close * 0.02)
    vol20 = _sf(row.get("VOL_20"), 0.2)
    stop = round(close - DEFAULT_STOP_ATR * atr, 2)
    take = round(close + DEFAULT_TAKE_ATR * atr, 2)
    et = ev.get("event_type", "unknown")

    if vol20 > 0.55:
      signal, reason = "AVOID", "demasiado ruido / alta volatilidad"
    elif prob >= BUY_THRESHOLD:
      signal = "BUY"
      reason = f"{et}: gap/volumen/tendencia + mercado positivo"
    elif prob >= WATCH_THRESHOLD:
      signal = "WATCH"
      reason = f"evento {et} sin confianza suficiente"
    else:
      signal, reason = "AVOID", "baja probabilidad o riesgo alto"

    rows.append({
      "ticker": ticker, "signal": signal, "event_type": et, "confidence": conf,
      "entry_plan": "proxima apertura" if signal == "BUY" else ("esperar confirmacion" if signal == "WATCH" else "no entrar"),
      "exit_plan": "vender por stop/take/20 dias" if signal in ("BUY", "WATCH") else "-",
      "stop_loss": stop if signal != "AVOID" else np.nan,
      "take_profit": take if signal != "AVOID" else np.nan,
      "max_holding_days": DEFAULT_HOLD if signal in ("BUY", "WATCH") else np.nan,
      "reason": reason,
    })
  return pd.DataFrame(rows)


current_signals = build_current_signals_v11(features_by_ticker, data_dict, labeled_events, EVENT_FEATURE_COLS)
print("Señales actuales:", len(current_signals))
if len(current_signals):
  print(current_signals[["ticker", "signal", "event_type", "confidence", "reason"]].to_string(index=False))
else:
  print("No hay eventos actuales con ventaja.")

# %% [markdown]
# ## 13. Exportar y reporte

# %%
def load_v6_return():
  for p in [Path("research_outputs/v6/research_v6_equity_curves.csv"), Path("research_v6_equity_curves.csv")]:
    if p.exists():
      try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        col = "blended_champion_weights_alpha_0.5"
        if col in df.columns:
          s = df[col].dropna()
          if len(s) >= 2:
            return (s.iloc[-1] / s.iloc[0] - 1) * 100
      except Exception:
        pass
  return np.nan


spy_total = (close_prices[MARKET].iloc[-1] / close_prices[MARKET].iloc[0] - 1) * 100 if MARKET in close_prices else 0
qqq_total = (close_prices["QQQ"].iloc[-1] / close_prices["QQQ"].iloc[0] - 1) * 100 if "QQQ" in close_prices else 0
v6_total = load_v6_return()
ew_cols = [c for c in close_prices.columns if c not in ("SPY", "QQQ", "IWM")]
ew_total = (close_prices[ew_cols].mean(axis=1).iloc[-1] / close_prices[ew_cols].mean(axis=1).iloc[0] - 1) * 100 if ew_cols else 0


def export_csv(df, name):
  if df is None or len(df) == 0:
    pd.DataFrame().to_csv(name, index=False)
  else:
    df.to_csv(name, index=False)
  print("Exportado", name)


summary_df = pd.DataFrame([{
  "lab": "v11_event_momentum",
  "event_engine_score": event_engine_score,
  "status": event_status,
  "approved_for_web_paper": APPROVED_FOR_WEB_PAPER,
  "approved_for_real_money": False,
  "best_event_type": best_event_type,
  **full_metrics,
  "pct_years_beats_spy": round(yearly_df["beats_spy"].mean() * 100, 1) if len(yearly_df) else 0,
  "spy_return": round(spy_total, 2),
  "qqq_return": round(qqq_total, 2),
  "v6_blend_return": round(v6_total, 2) if pd.notna(v6_total) else np.nan,
  "ew_universe_return": round(ew_total, 2),
  "events_detected": len(events_df),
  "events_labeled": len(labeled_events),
}])

export_csv(summary_df, "research_v11_summary.csv")
export_csv(events_df, "research_v11_events.csv")
export_csv(trades_df, "research_v11_trades.csv")
export_csv(yearly_df, "research_v11_yearly.csv")
export_csv(current_signals, "research_v11_current_signals.csv")
if len(current_signals):
  current_signals.to_csv("current_signals_v11.csv", index=False)
  print("Exportado current_signals_v11.csv")
export_csv(event_type_perf, "research_v11_event_type_performance.csv")
export_csv(signal_quality, "research_v11_signal_quality.csv")
if len(equity_oos):
  equity_oos.to_frame("equity").to_csv("research_v11_equity_curve.csv")
  print("Exportado research_v11_equity_curve.csv")
else:
  pd.DataFrame().to_csv("research_v11_equity_curve.csv", index=False)

config_out = {
  "version": "v11_event_momentum_lab",
  "approved_for_web_paper": APPROVED_FOR_WEB_PAPER,
  "approved_for_real_money": False,
  "event_engine_score": event_engine_score,
  "status": event_status,
  "event_types": EVENT_TYPES,
  "trade_params": {"hold_days": DEFAULT_HOLD, "stop_atr": DEFAULT_STOP_ATR, "take_atr": DEFAULT_TAKE_ATR, "selected_on_train": list(selected_params)},
  "thresholds": {"buy": BUY_THRESHOLD, "watch": WATCH_THRESHOLD, "by_year": threshold_log},
  "warnings": [
    "Backtest no garantiza resultados futuros.",
    "earnings_proxy_gap usa proxy sin calendario earnings real.",
    "No conecta broker ni ejecuta ordenes.",
    "No aprobado para dinero real.",
  ],
  "metrics": full_metrics,
}
Path("research_v11_selected_config.json").write_text(json.dumps(config_out, indent=2, default=str), encoding="utf-8")
print("Exportado research_v11_selected_config.json")

print("=" * 80)
print("REPORTE FINAL V11 EVENT MOMENTUM / EARNINGS DRIFT LAB")
print("=" * 80)
print(f"Estado: {event_status} | Score: {event_engine_score}/100")
print(f"Mejor event_type: {best_event_type}")
print(f"PF: {full_metrics.get('profit_factor', 0)} | Sharpe: {full_metrics.get('sharpe', 0)} | "
      f"DD: {full_metrics.get('max_drawdown', 0)}% | Trades: {full_metrics.get('num_trades', 0)} | "
      f"Win: {full_metrics.get('win_rate', 0)}%")
print(f"Años vs SPY: {yearly_df['beats_spy'].mean()*100:.0f}%" if len(yearly_df) else "N/A")
print(f"¿Supera SPY? {'SI' if full_metrics.get('total_return', 0) > spy_total else 'NO'}")
print(f"¿Supera V6? {'SI' if pd.notna(v6_total) and full_metrics.get('total_return', 0) > v6_total else 'NO'}")
print(f"BUY actuales: {len(current_signals[current_signals['signal']=='BUY']) if len(current_signals) else 0}")
if score_notes:
  print("Notas:", "; ".join(score_notes))
print("")
if APPROVED_FOR_WEB_PAPER:
  print("Integrar V11 Event Signal Engine Paper Trading.")
else:
  print("V11 rejected. Mantener V6 como champion.")
print("APPROVED_FOR_REAL_MONEY=False (siempre)")
