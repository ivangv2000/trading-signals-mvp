# %% [markdown]
# # Trading Research V15 Meta Champion Regime Lab
#
# Meta-modelo semanal: V14 + V6 + benchmark core + defensive según régimen.
#
# **Disclaimer:** Backtest no garantiza resultados futuros. No es asesoramiento financiero.
# **APPROVED_FOR_REAL_MONEY siempre False.**

# %%
try:
  get_ipython().run_line_magic("pip", "install yfinance pandas numpy matplotlib plotly tqdm scipy scikit-learn -q")
except NameError:
  import subprocess
  import sys
  subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "yfinance", "pandas", "numpy", "matplotlib", "plotly", "tqdm", "scipy", "scikit-learn",
  ])

# %% [markdown]
# ## 1. Configuracion

# %%
import warnings
warnings.filterwarnings("ignore")

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tqdm.auto import tqdm

QUICK_TEST = True
START_DATE = "2007-01-01"
END_DATE = None
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001
SLIPPAGE = 0.001
REBALANCE_FREQ = "W-FRI"
WF_START_YEAR = 2015
CASH_ASSET = "SHY"
MAX_WEIGHT = 0.30
VOL_TARGET = 0.15
MIN_HISTORY_DAYS = 200
MARKET = "SPY"
COST_RATE = TRANSACTION_COST + SLIPPAGE
TOP_N_V14 = 3
LOOKBACK_V14 = 63
EMBARGO_DAYS = 20
ML_HORIZON_WEEKS = 4

UNIVERSE = [
  "SPY", "QQQ", "IWM", "DIA",
  "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLC", "XLB", "XLRE",
  "MTUM", "QUAL", "USMV", "VLUE", "SPLV", "SPHB", "SCHD",
  "SHY", "IEF", "TLT", "LQD", "HYG",
  "GLD", "SLV", "DBC", "VNQ",
  "EFA", "EEM",
  "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "GOOGL", "META", "AMZN",
  "JPM", "BAC", "XOM", "CVX", "UNH", "LLY", "WMT", "COST",
]
DEFENSIVE_POOL = ["SHY", "IEF", "TLT", "GLD", "USMV", "QUAL", "SCHD", "XLV", "XLP"]
V6_UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN"]
START_DATES_ROBUST = ["2010-01-01", "2015-01-01", "2018-01-01"] if QUICK_TEST else [
  "2010-01-01", "2012-01-01", "2015-01-01", "2018-01-01", "2020-01-01",
]
COST_SENSITIVITY = [0.0005, 0.001, 0.002] if not QUICK_TEST else [0.0005, 0.001]

if QUICK_TEST:
  UNIVERSE = ["SPY", "QQQ", "IWM", "XLK", "XLV", "XLF", "TLT", "IEF", "SHY", "GLD", "MTUM", "USMV", "AAPL", "MSFT", "NVDA"]
  START_DATE = "2015-01-01"
  print("QUICK_TEST activo")

print("V15 Meta Champion Regime Lab | desde", START_DATE)

# %% [markdown]
# ## 2. Descarga de datos

# %%
def download_data(tickers, start, end=None, min_days=MIN_HISTORY_DAYS):
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
      if len(df) < min_days:
        failed.append(f"{ticker}(short)")
        continue
      df.index = pd.DatetimeIndex(df.index)
      if df.index.tz:
        df.index = df.index.tz_localize(None)
      data[ticker.upper()] = df.sort_index()
    except Exception:
      failed.append(ticker)
  close = pd.DataFrame({k: v["Close"] for k, v in data.items()}).sort_index().ffill()
  close.index = pd.DatetimeIndex(close.index)
  return data, close


data_dict, close_prices = download_data(UNIVERSE, START_DATE, END_DATE)
print("Tickers:", len(close_prices.columns), "| Filas:", len(close_prices))

# %% [markdown]
# ## 3. Funciones comunes

# %%
def _sf(x, d=np.nan):
  try:
    v = float(x)
    return d if not np.isfinite(v) else v
  except Exception:
    return d


def calculate_returns(close):
  return close.pct_change().fillna(0)


def calculate_max_drawdown(equity):
  peak = equity.cummax()
  return float(((equity - peak) / peak.replace(0, np.nan)).min())


def calculate_sharpe(rets, ann=252):
  r = rets.dropna()
  return float(r.mean() / r.std() * math.sqrt(ann)) if r.std() > 0 else 0.0


def calculate_sortino(rets, ann=252):
  r = rets.dropna()
  ds = r[r < 0].std()
  return float(r.mean() / ds * math.sqrt(ann)) if ds and ds > 0 else 0.0


def calculate_cagr(equity):
  y = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
  return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / y) - 1)


def calculate_calmar(equity):
  mdd = calculate_max_drawdown(equity)
  return calculate_cagr(equity) / abs(mdd) if mdd != 0 else 0.0


def safe_normalize_weights(w):
  w = pd.Series(w).astype(float).clip(lower=0).fillna(0)
  return w / w.sum() if w.sum() > 0 else w


def cap_and_redistribute_weights(w, cap=MAX_WEIGHT):
  w = safe_normalize_weights(w)
  for _ in range(10):
    over = w[w > cap]
    if over.empty:
      break
    ex = (over - cap).sum()
    w.loc[over.index] = cap
    under = w[w < cap]
    if under.sum() > 0:
      w.loc[under.index] += ex * (under / under.sum())
  return safe_normalize_weights(w.clip(lower=0))


def inverse_vol_weights(tickers, vols, cap=MAX_WEIGHT):
  inv = pd.Series({t: 1.0 / max(_sf(vols.get(t), 0.15), 0.05) for t in tickers})
  return cap_and_redistribute_weights(inv, cap)


def calculate_turnover(nw, ow):
  idx = nw.index.union(ow.index)
  return float((nw.reindex(idx, fill_value=0) - ow.reindex(idx, fill_value=0)).abs().sum() / 2)


def get_rebalance_dates(close):
  return close.resample(REBALANCE_FREQ).last().dropna(how="all").index


def _align_idx(close, dt):
  dt = pd.Timestamp(dt)
  if dt in close.index:
    return dt
  loc = close.index.get_indexer([dt], method="pad")[0]
  return close.index[loc] if loc >= 0 else None


def build_equity_curve_from_weights(close, weight_schedule, data=None, initial=INITIAL_CAPITAL, cost_rate=COST_RATE):
  dates = close.index
  rebal_exec = {}
  for d, w in weight_schedule.items():
    future = dates[dates > d]
    if len(future) == 0:
      continue
    exec_dt = future[0]
    if data and exec_dt in dates:
      for t in w.index:
        if t in data and exec_dt in data[t].index and "Open" in data[t].columns:
          pass
    rebal_exec[exec_dt] = w

  equity, curve, current_w, turnover_log = initial, {}, pd.Series(dtype=float), []
  for i, dt in enumerate(dates):
    if dt in rebal_exec:
      nw = rebal_exec[dt]
      to = calculate_turnover(nw, current_w) if len(current_w) else nw.abs().sum()
      equity *= (1 - to * cost_rate)
      turnover_log.append({"date": dt, "turnover": to, "cost": equity * to * cost_rate})
      current_w = nw.copy()
    if i > 0 and len(current_w):
      prev = dates[i - 1]
      dr = sum(
        current_w.get(t, 0) * (close[t].loc[dt] / close[t].loc[prev] - 1)
        for t in current_w.index if t in close.columns and np.isfinite(close[t].loc[dt])
      )
      equity *= (1 + dr)
    curve[dt] = equity
  return pd.Series(curve), pd.DataFrame(turnover_log)


def calculate_yearly_results(equity, close):
  rows = []
  for year in sorted(set(equity.index.year)):
    e = equity.loc[f"{year}-01-01":f"{year}-12-31"]
    if len(e) < 2:
      continue
    sr = e.iloc[-1] / e.iloc[0] - 1
    spy = close[MARKET].loc[f"{year}-01-01":f"{year}-12-31"] if MARKET in close else pd.Series()
    qqq = close["QQQ"].loc[f"{year}-01-01":f"{year}-12-31"] if "QQQ" in close else pd.Series()
    spy_r = spy.iloc[-1] / spy.iloc[0] - 1 if len(spy) >= 2 else 0
    qqq_r = qqq.iloc[-1] / qqq.iloc[0] - 1 if len(qqq) >= 2 else 0
    tlt = close["TLT"].loc[f"{year}-01-01":f"{year}-12-31"] if "TLT" in close else pd.Series()
    tlt_r = tlt.iloc[-1] / tlt.iloc[0] - 1 if len(tlt) >= 2 else 0
    mix = 0.6 * spy_r + 0.4 * tlt_r
    rows.append({
      "year": year, "return": round(sr * 100, 2),
      "SPY": round(spy_r * 100, 2), "QQQ": round(qqq_r * 100, 2), "6040": round(mix * 100, 2),
      "beats_spy": sr > spy_r, "beats_qqq": sr > qqq_r, "beats_6040": sr > mix,
    })
  return pd.DataFrame(rows)


def compare_to_benchmarks(equity, close):
  rets = equity.pct_change().fillna(0)
  yearly = calculate_yearly_results(equity, close)
  return {
    "total_return": round((equity.iloc[-1] / equity.iloc[0] - 1) * 100, 2),
    "CAGR": round(calculate_cagr(equity) * 100, 2),
    "sharpe": round(calculate_sharpe(rets), 3),
    "sortino": round(calculate_sortino(rets), 3),
    "max_drawdown": round(calculate_max_drawdown(equity) * 100, 2),
    "calmar": round(calculate_calmar(equity), 3),
    "annual_volatility": round(rets.std() * math.sqrt(252) * 100, 2),
    "win_years_vs_spy": yearly["beats_spy"].mean() if len(yearly) else 0,
    "win_years_vs_qqq": yearly["beats_qqq"].mean() if len(yearly) else 0,
    "win_years_vs_6040": yearly["beats_6040"].mean() if len(yearly) else 0,
  }, yearly


def calculate_contribution_by_asset(close, schedule):
  dates = close.index
  rebal_exec = {dates[dates > d][0]: w for d, w in schedule.items() if len(dates[dates > d])}
  contrib = pd.Series(dtype=float)
  current_w = pd.Series(dtype=float)
  for i, dt in enumerate(dates):
    if dt in rebal_exec:
      current_w = rebal_exec[dt]
    if i == 0 or not len(current_w):
      continue
    prev = dates[i - 1]
    for t in current_w.index:
      if t not in close.columns:
        continue
      ar = close[t].loc[dt] / close[t].loc[prev] - 1
      if np.isfinite(ar):
        contrib[t] = contrib.get(t, 0) + current_w.get(t, 0) * ar
  if contrib.empty:
    return pd.DataFrame()
  total = contrib.abs().sum()
  out = contrib.sort_values(ascending=False).reset_index()
  out.columns = ["ticker", "contribution"]
  out["pct_of_total"] = (out["contribution"].abs() / total * 100).round(2) if total > 0 else 0
  return out


def _cash_fill(w, close_cols, frac=1.0):
  w = w.copy()
  safe = CASH_ASSET if CASH_ASSET in close_cols else "CASH"
  w[safe] = w.get(safe, 0) + max(0, frac - w.sum())
  return cap_and_redistribute_weights(w)


def _defensive_only(cols):
  if CASH_ASSET in cols:
    return pd.Series({CASH_ASSET: 1.0})
  return pd.Series({"SHY": 1.0})

# %% [markdown]
# ## 4. Features de regimen

# %%
def _pair_mom(close, a, b, dt, window=63):
  if a not in close.columns or b not in close.columns:
    return np.nan
  dt = _align_idx(close, dt)
  ca, cb = close[a].loc[:dt], close[b].loc[:dt]
  if len(ca) < window + 1:
    return np.nan
  ra = ca.iloc[-1] / ca.iloc[-window] - 1
  rb = cb.iloc[-1] / cb.iloc[-window] - 1
  return ra - rb


def calculate_regime_features(close):
  rets = calculate_returns(close)
  feats = pd.DataFrame(index=close.index)

  for t in [MARKET, "QQQ"]:
    if t not in close.columns:
      continue
    c = close[t]
    feats[f"{t}_MOM_21"] = c / c.shift(21) - 1
    feats[f"{t}_MOM_63"] = c / c.shift(63) - 1
    if t == MARKET:
      feats[f"{t}_MOM_126"] = c / c.shift(126) - 1
    sma200 = c.rolling(200).mean()
    feats[f"{t}_ABOVE_SMA200"] = (c > sma200).astype(float)
    feats[f"{t}_DD_63"] = c / c.rolling(63).max() - 1
    if t == MARKET:
      feats[f"{t}_DD_126"] = c / c.rolling(126).max() - 1
    feats[f"{t}_VOL_21"] = rets[t].rolling(21).std() * math.sqrt(252)
    feats[f"{t}_VOL_63"] = rets[t].rolling(63).std() * math.sqrt(252)

  feats["QQQ_SPY_MOM"] = feats.get("QQQ_MOM_63", 0) - feats.get("SPY_MOM_63", 0)
  if "XLK" in close.columns:
    feats["XLK_SPY_MOM"] = close["XLK"] / close["XLK"].shift(63) - close[MARKET] / close[MARKET].shift(63)
  if "MTUM" in close.columns and "USMV" in close.columns:
    feats["MTUM_USMV_MOM"] = close["MTUM"] / close["MTUM"].shift(63) - close["USMV"] / close["USMV"].shift(63)
  if "HYG" in close.columns and "LQD" in close.columns:
    feats["HYG_LQD_MOM"] = close["HYG"] / close["HYG"].shift(63) - close["LQD"] / close["LQD"].shift(63)
  if "GLD" in close.columns:
    g = close["GLD"]
    feats["GLD_TREND"] = (g > g.rolling(100).mean()).astype(float)
  if "TLT" in close.columns:
    tlt = close["TLT"]
    feats["TLT_TREND"] = (tlt > tlt.rolling(100).mean()).astype(float)

  spy_above = feats.get(f"{MARKET}_ABOVE_SMA200", pd.Series(0, index=close.index)).fillna(0)
  qqq_above = feats.get("QQQ_ABOVE_SMA200", spy_above).fillna(0)
  spy_mom = feats.get(f"{MARKET}_MOM_63", pd.Series(0, index=close.index)).fillna(0)
  qqq_mom = feats.get("QQQ_MOM_63", pd.Series(0, index=close.index)).fillna(0)
  spy_dd = feats.get(f"{MARKET}_DD_63", pd.Series(0, index=close.index)).fillna(0)
  spy_vol = feats.get(f"{MARKET}_VOL_21", pd.Series(0.15, index=close.index)).fillna(0.15)

  feats["risk_on_score"] = (
    spy_above * 25 + qqq_above * 25
    + (spy_mom > 0).astype(float) * 20 + (qqq_mom > 0).astype(float) * 20
    + (spy_mom > 0.05).astype(float) * 10
  ).clip(0, 100)

  feats["defensive_score"] = (
    (spy_above < 0.5).astype(float) * 30
    + (qqq_above < 0.5).astype(float) * 25
    + (spy_mom < 0).astype(float) * 25
    + (spy_dd < -0.08).astype(float) * 20
  ).clip(0, 100)

  feats["crash_risk_score"] = (
    (spy_dd < -0.10).astype(float) * 40
    + (spy_vol > 0.25).astype(float) * 30
    + (spy_mom < -0.05).astype(float) * 30
  ).clip(0, 100)

  return feats.fillna(0)


def _row_regime(regime_feats, dt):
  if regime_feats is None or len(regime_feats) == 0:
    return pd.Series(dtype=float)
  dt = _align_idx(regime_feats, dt)
  if dt is None:
    return pd.Series(dtype=float)
  return regime_feats.loc[dt]


REGIME_FEATS = None


def classify_regime(dt, rf=None):
  rf = rf if rf is not None else REGIME_FEATS
  r = _row_regime(rf, dt)
  if r.empty:
    return "mixed", 50.0
  crash = _sf(r.get("crash_risk_score", 0))
  defensive = _sf(r.get("defensive_score", 0))
  risk_on = _sf(r.get("risk_on_score", 0))
  spy_above = _sf(r.get(f"{MARKET}_ABOVE_SMA200", 0))
  qqq_above = _sf(r.get("QQQ_ABOVE_SMA200", 0))

  if crash >= 60:
    return "crash_risk", crash
  if defensive >= 65 and risk_on < 40:
    return "defensive", defensive
  if risk_on >= 75 and spy_above >= 0.5 and qqq_above >= 0.5:
    return "risk_on_strong", risk_on
  if risk_on >= 50 and spy_above >= 0.5:
    return "risk_on_normal", risk_on
  if defensive >= 45 and risk_on < 55:
    return "mixed", (risk_on + defensive) / 2
  return "mixed", risk_on


regime_feats = calculate_regime_features(close_prices)
REGIME_FEATS = regime_feats
rdates = [d for d in get_rebalance_dates(close_prices) if d.year >= WF_START_YEAR]
print("Regimen features | rebalance dates:", len(rdates))

# %% [markdown]
# ## 5-8. Motores V14, V6, Benchmark, Defensive

# %%
def _asset_feats_at(close, dt, ticker):
  if ticker not in close.columns:
    return None
  dt = _align_idx(close, dt)
  c = close[ticker].loc[:dt]
  if len(c) < 200:
    return None
  mom63 = c.iloc[-1] / c.iloc[-min(63, len(c) - 1)] - 1 if len(c) > 63 else 0
  sma200 = c.rolling(200).mean().iloc[-1]
  vol63 = c.pct_change().rolling(63).std().iloc[-1] * math.sqrt(252)
  vol20 = c.pct_change().rolling(20).std().iloc[-1] * math.sqrt(252)
  return {
    "mom63": _sf(mom63), "above_sma200": float(c.iloc[-1] > sma200) if np.isfinite(sma200) else 0,
    "vol63": _sf(vol63, 0.15), "vol20": _sf(vol20, 0.15),
    "trend": float(c.iloc[-1] > sma200) * 0.6 + float(mom63 > 0) * 0.4,
  }


def v14_return_engine_weights(close, dt):
  regime, _ = classify_regime(dt)
  risk_on = regime in ("risk_on_strong", "risk_on_normal")
  scores, vols = {}, {}
  universe = [c for c in close.columns if c != CASH_ASSET]
  if risk_on:
    for t in universe:
      if t in DEFENSIVE_POOL:
        continue
      af = _asset_feats_at(close, dt, t)
      if af is None or af["mom63"] <= 0 or af["above_sma200"] < 0.5:
        continue
      if af["vol63"] > 0.45:
        continue
      scores[t] = af["mom63"] * 0.5 + af["trend"] * 0.5
      vols[t] = af["vol63"]
  else:
    for t in DEFENSIVE_POOL:
      if t not in close.columns:
        continue
      af = _asset_feats_at(close, dt, t)
      scores[t] = af["trend"] if af else 0
      vols[t] = 0.12
  top = pd.Series(scores).sort_values(ascending=False).head(TOP_N_V14)
  if top.empty:
    return _defensive_only(close.columns)
  w = inverse_vol_weights(top.index, vols) * min(1, VOL_TARGET / 0.15) * (0.9 if risk_on else 0.55)
  return _cash_fill(w, close.columns)


def _v6_trend_weights(close, dt):
  universe = [t for t in V6_UNIVERSE if t in close.columns]
  elig, vols = [], {}
  for t in universe:
    af = _asset_feats_at(close, dt, t)
    if af is None:
      continue
    c = close[t].loc[:_align_idx(close, dt)]
    sma200 = c.rolling(200).mean().iloc[-1]
    ema50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
    if c.iloc[-1] > sma200 and ema50 > sma200:
      elig.append(t)
      vols[t] = af["vol20"]
  if not elig:
    return None
  return inverse_vol_weights(elig, vols)


def _v6_adaptive_weights(close, dt):
  regime, _ = classify_regime(dt)
  risk_on = regime in ("risk_on_strong", "risk_on_normal", "mixed")
  universe = [t for t in V6_UNIVERSE if t in close.columns]
  if risk_on:
    scores = {}
    for t in universe:
      af = _asset_feats_at(close, dt, t)
      if af and af["mom63"] > 0 and af["above_sma200"] >= 0.5:
        scores[t] = af["mom63"] / max(af["vol63"], 0.05)
    top = pd.Series(scores).sort_values(ascending=False).head(4)
    if top.empty:
      return None
    vols = {t: _asset_feats_at(close, dt, t)["vol63"] for t in top.index}
    w = inverse_vol_weights(top.index, vols) * 0.85
    return _cash_fill(w, close.columns)
  pool = [t for t in DEFENSIVE_POOL if t in close.columns]
  scores = {t: (_asset_feats_at(close, dt, t) or {}).get("trend", 0) for t in pool}
  top = pd.Series(scores).sort_values(ascending=False).head(3)
  if top.empty:
    return None
  w = pd.Series({t: 0.85 / len(top) for t in top.index})
  return _cash_fill(w, close.columns)


def v6_aggressive_engine_weights(close, dt):
  w1 = _v6_trend_weights(close, dt)
  w2 = _v6_adaptive_weights(close, dt)
  if w1 is None and w2 is None:
    return _defensive_only(close.columns)
  if w1 is None:
    return w2
  if w2 is None:
    return w1
  blend = w1 * 0.5 + w2.reindex(w1.index, fill_value=0) * 0.5
  for t in w2.index:
    if t not in blend.index:
      blend[t] = w2[t] * 0.5
  return cap_and_redistribute_weights(blend)


def benchmark_core_engine_weights(close, dt):
  regime, conf = classify_regime(dt)
  w = pd.Series(dtype=float)
  if regime == "risk_on_strong":
    for t, pct in [("QQQ", 0.40), ("SPY", 0.30)]:
      if t in close.columns:
        w[t] = pct
    for sat in ["XLK", "MTUM"]:
      if sat in close.columns:
        af = _asset_feats_at(close, dt, sat)
        if af and af["above_sma200"] >= 0.5:
          w[sat] = 0.15
          break
    safe = 0.15
    if CASH_ASSET in close.columns:
      w[CASH_ASSET] = w.get(CASH_ASSET, 0) + safe * 0.5
    if "GLD" in close.columns:
      w["GLD"] = w.get("GLD", 0) + safe * 0.5
  elif regime == "risk_on_normal":
    for t, pct in [("SPY", 0.35), ("QQQ", 0.25)]:
      if t in close.columns:
        w[t] = pct
    v14w = v14_return_engine_weights(close, dt)
    w = w.add(v14w * 0.20, fill_value=0)
    for t in [CASH_ASSET, "IEF"]:
      if t in close.columns:
        w[t] = w.get(t, 0) + 0.10
  else:
    for t, pct in [(CASH_ASSET, 0.50), ("IEF", 0.125), ("TLT", 0.125), ("GLD", 0.15)]:
      if t in close.columns:
        w[t] = pct
    if "SPY" in close.columns:
      af = _asset_feats_at(close, dt, "SPY")
      if af and af["above_sma200"] >= 0.5:
        w["SPY"] = 0.10
  return cap_and_redistribute_weights(w) if w.sum() > 0 else _defensive_only(close.columns)


def defensive_engine_weights(close, dt):
  scores, vols = {}, {}
  for t in DEFENSIVE_POOL:
    if t not in close.columns:
      continue
    af = _asset_feats_at(close, dt, t)
    if af is None:
      continue
    if af["mom63"] > 0 or af["trend"] > 0.3:
      scores[t] = af["trend"] + (1 / max(af["vol63"], 0.05)) * 0.1
      vols[t] = af["vol63"]
  top = pd.Series(scores).sort_values(ascending=False).head(5)
  if top.empty:
    return _defensive_only(close.columns)
  return _cash_fill(inverse_vol_weights(top.index, vols) * 0.9, close.columns)


ENGINE_FNS = {
  "v14_return": v14_return_engine_weights,
  "v6_aggressive": v6_aggressive_engine_weights,
  "benchmark_core": benchmark_core_engine_weights,
  "defensive": defensive_engine_weights,
}

# %% [markdown]
# ## 9. Meta engine rule-based

# %%
REGIME_BLEND = {
  "risk_on_strong": {"v14_return": 0.45, "benchmark_core": 0.30, "v6_aggressive": 0.20, "defensive": 0.05},
  "risk_on_normal": {"v14_return": 0.40, "v6_aggressive": 0.25, "benchmark_core": 0.20, "defensive": 0.15},
  "mixed": {"v14_return": 0.35, "v6_aggressive": 0.20, "benchmark_core": 0.15, "defensive": 0.30},
  "defensive": {"v14_return": 0.20, "v6_aggressive": 0.10, "benchmark_core": 0.10, "defensive": 0.60},
  "crash_risk": {"v14_return": 0.0, "v6_aggressive": 0.0, "benchmark_core": 0.0, "defensive": 1.0},
}


def blend_engine_weights(close, dt, blend_map):
  w = pd.Series(dtype=float)
  for eng, frac in blend_map.items():
    if frac <= 0 or eng not in ENGINE_FNS:
      continue
    ew = ENGINE_FNS[eng](close, dt)
    w = w.add(ew * frac, fill_value=0)
  return cap_and_redistribute_weights(w) if w.sum() > 0 else _defensive_only(close.columns)


def meta_engine_rule_based_weights(close, dt, blend_override=None):
  regime, conf = classify_regime(dt)
  blend = blend_override or REGIME_BLEND.get(regime, REGIME_BLEND["mixed"])
  w = blend_engine_weights(close, dt, blend)
  return w, regime, conf


def build_schedule_from_fn(fn, close, rdates):
  sched, regimes = {}, {}
  for dt in rdates:
    out = fn(close, dt)
    if isinstance(out, tuple):
      w, regime, conf = out
      regimes[dt] = {"regime": regime, "confidence": conf}
    else:
      w = out
    sched[dt] = w
  return sched, regimes


def meta_rule_schedule(close, rdates, blend_override=None):
  def _fn(c, d):
    return meta_engine_rule_based_weights(c, d, blend_override)
  return build_schedule_from_fn(_fn, close, rdates)

# %% [markdown]
# ## 10. Meta engine ML opcional

# %%
def _engine_forward_scores(close, rdates, engine_key, horizon_weeks=ML_HORIZON_WEEKS):
  """Retorno forward risk-adjusted por motor (solo para entrenamiento ML)."""
  fn = ENGINE_FNS[engine_key]
  scores = {}
  for i, dt in enumerate(rdates):
    if i + horizon_weeks >= len(rdates):
      break
    w = fn(close, dt)
    dt0 = _align_idx(close, dt)
    dt1 = _align_idx(close, rdates[i + horizon_weeks])
    if dt0 is None or dt1 is None:
      continue
    seg = close.loc[dt0:dt1]
    if len(seg) < 2:
      continue
    port_rets = []
    for j in range(1, len(seg)):
      d_prev, d_curr = seg.index[j - 1], seg.index[j]
      dr = sum(w.get(t, 0) * (close[t].loc[d_curr] / close[t].loc[d_prev] - 1) for t in w.index if t in close.columns)
      port_rets.append(dr)
    if not port_rets:
      continue
    r = pd.Series(port_rets)
    sharpe = r.mean() / r.std() * math.sqrt(52) if r.std() > 0 else 0
    scores[dt] = sharpe - abs(min(0, (np.prod(1 + r) - 1)))
  return scores


def build_ml_dataset(close, regime_feats, rdates):
  rows = []
  engine_keys = list(ENGINE_FNS.keys())
  fwd = {k: _engine_forward_scores(close, rdates, k) for k in engine_keys}
  feat_cols = [c for c in regime_feats.columns if c not in ("regime",)]
  for i, dt in enumerate(rdates):
    if i + ML_HORIZON_WEEKS >= len(rdates):
      break
    r = _row_regime(regime_feats, dt)
    if r.empty:
      continue
    row = {c: _sf(r.get(c, 0)) for c in feat_cols}
    fwd_vals = {k: fwd[k].get(dt, np.nan) for k in engine_keys}
    if any(pd.isna(v) for v in fwd_vals.values()):
      continue
    best = max(fwd_vals, key=fwd_vals.get)
    row["label"] = best
    row["date"] = dt
    for k, v in fwd_vals.items():
      row[f"fwd_{k}"] = v
    rows.append(row)
  return pd.DataFrame(rows)


def meta_engine_ml_selector(close, regime_feats, rdates, min_confidence=0.45):
  from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
  from sklearn.linear_model import LogisticRegression
  from sklearn.preprocessing import LabelEncoder

  df = build_ml_dataset(close, regime_feats, rdates)
  if len(df) < 80:
    return None, pd.DataFrame(), "insufficient_data"

  feat_cols = [c for c in df.columns if c not in ("label", "date") and not c.startswith("fwd_")]
  years = sorted(df["date"].dt.year.unique())
  oos_preds, ml_rows = [], []

  for test_year in years:
    if test_year <= WF_START_YEAR:
      continue
    train = df[df["date"].dt.year < test_year]
    test = df[df["date"].dt.year == test_year]
    if len(train) < 50 or len(test) < 5:
      continue
    train = train.iloc[:-EMBARGO_DAYS] if len(train) > EMBARGO_DAYS else train
    X_tr, y_tr = train[feat_cols].fillna(0), train["label"]
    X_te = test[feat_cols].fillna(0)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_tr)
    models = {
      "logistic": LogisticRegression(max_iter=500, C=0.5),
      "rf": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
      "hgb": HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=42),
    }
    probs_sum = np.zeros((len(X_te), len(le.classes_)))
    for name, mdl in models.items():
      mdl.fit(X_tr, y_enc)
      pr = mdl.predict_proba(X_te)
      probs_sum += pr / len(models)
    for j, (_, row) in enumerate(test.iterrows()):
      best_idx = probs_sum[j].argmax()
      best_conf = probs_sum[j][best_idx]
      best_eng = le.classes_[best_idx]
      oos_preds.append({"date": row["date"], "engine": best_eng, "confidence": best_conf})
      ml_rows.append({"year": test_year, "engine": best_eng, "confidence": best_conf})

  if not oos_preds:
    return None, pd.DataFrame(), "no_oos"

  pred_df = pd.DataFrame(oos_preds)
  ml_results = pd.DataFrame(ml_rows)

  def ml_weights_fn(c, dt):
    regime, reg_conf = classify_regime(dt)
    match = pred_df[pred_df["date"] == dt]
    if len(match):
      eng = match.iloc[0]["engine"]
      conf = _sf(match.iloc[0]["confidence"], 0.33)
    else:
      return meta_engine_rule_based_weights(c, dt)
    if conf < min_confidence:
      return meta_engine_rule_based_weights(c, dt)
    blend = {k: 0.0 for k in ENGINE_FNS}
    blend[eng] = 1.0
    w = blend_engine_weights(c, dt, blend)
    return w, regime, conf * 100

  return ml_weights_fn, ml_results, "ok"

# %% [markdown]
# ## 11-12. Backtest y estrategias M1-M7

# %%
def run_v15_backtest(name, schedule, cost_rate=COST_RATE):
  if not schedule:
    return {}, pd.Series(dtype=float), pd.DataFrame(), schedule
  eq, to = build_equity_curve_from_weights(close_prices, schedule, data_dict, cost_rate=cost_rate)
  if len(eq) < 2:
    return {}, eq, to, schedule
  m, yearly = compare_to_benchmarks(eq, close_prices)
  m["strategy"] = name
  m["turnover"] = round(to["turnover"].mean(), 4) if len(to) else 0
  m["total_cost"] = round(to["cost"].sum(), 2) if len(to) else 0
  m["num_rebalances"] = len(schedule)
  m["exposure"] = round(1 - m.get("turnover", 0), 3)
  yearly["strategy"] = name
  return m, eq, yearly, schedule


def m2_aggressive_blend(regime):
  b = REGIME_BLEND.get(regime, REGIME_BLEND["mixed"]).copy()
  if regime == "risk_on_strong":
    b["v6_aggressive"] = min(0.35, b.get("v6_aggressive", 0) + 0.15)
    b["benchmark_core"] = min(0.40, b.get("benchmark_core", 0) + 0.10)
    b["defensive"] = max(0, b.get("defensive", 0) - 0.25)
  return b


def m3_defensive_blend(regime):
  b = REGIME_BLEND.get(regime, REGIME_BLEND["mixed"]).copy()
  if regime in ("mixed", "defensive"):
    b["defensive"] = min(0.70, b.get("defensive", 0) + 0.20)
    b["v6_aggressive"] = max(0, b.get("v6_aggressive", 0) - 0.10)
  return b


def m4_benchmark_capture(close, dt):
  regime, conf = classify_regime(dt)
  w, _, _ = meta_engine_rule_based_weights(close, dt, REGIME_BLEND.get(regime))
  if regime == "risk_on_strong":
    bench_tickers = ["SPY", "QQQ", "XLK", "MTUM"]
    bench_w = sum(w.get(t, 0) for t in bench_tickers if t in w.index)
    if bench_w < 0.40:
      boost = 0.40 - bench_w
      for t in ["QQQ", "SPY", "XLK", "MTUM"]:
        if t in close.columns:
          w[t] = w.get(t, 0) + boost / 2
          break
      w = cap_and_redistribute_weights(w)
  return w, regime, conf


def m6_blend_50_50(close, dt):
  w1 = v14_return_engine_weights(close, dt)
  w2 = v6_aggressive_engine_weights(close, dt)
  w = w1 * 0.5 + w2.reindex(w1.index, fill_value=0) * 0.5
  for t in w2.index:
    w[t] = w.get(t, 0) + w2.get(t, 0) * 0.5
  regime, conf = classify_regime(dt)
  return cap_and_redistribute_weights(w), regime, conf


def m7_adaptive(close, dt):
  regime, conf = classify_regime(dt)
  w14 = v14_return_engine_weights(close, dt)
  w6 = v6_aggressive_engine_weights(close, dt)
  wd = defensive_engine_weights(close, dt)
  if regime in ("risk_on_strong", "risk_on_normal"):
    w = w14 * 0.60 + w6.reindex(w14.index, fill_value=0) * 0.40
    for t in w6.index:
      w[t] = w.get(t, 0) + w6.get(t, 0) * 0.40
  elif regime == "mixed":
    w = w14 * 0.50 + w6.reindex(w14.index, fill_value=0) * 0.50
    for t in w6.index:
      w[t] = w.get(t, 0) + w6.get(t, 0) * 0.50
    w = w * 0.70 + wd.reindex(w.index, fill_value=0) * 0.30
  else:
    w = wd * 0.80 + w14.reindex(wd.index, fill_value=0) * 0.20
  return cap_and_redistribute_weights(w), regime, conf


# Build all strategy schedules
print("Construyendo schedules...")
sched_m1, reg_m1 = meta_rule_schedule(close_prices, rdates)
sched_m2, _ = build_schedule_from_fn(
  lambda c, d: meta_engine_rule_based_weights(c, d, m2_aggressive_blend(classify_regime(d)[0])),
  close_prices, rdates,
)
sched_m3, _ = build_schedule_from_fn(
  lambda c, d: meta_engine_rule_based_weights(c, d, m3_defensive_blend(classify_regime(d)[0])),
  close_prices, rdates,
)
sched_m4, _ = build_schedule_from_fn(m4_benchmark_capture, close_prices, rdates)
sched_m6, _ = build_schedule_from_fn(m6_blend_50_50, close_prices, rdates)
sched_m7, _ = build_schedule_from_fn(m7_adaptive, close_prices, rdates)

# Pure engines for comparison
sched_v14, _ = build_schedule_from_fn(v14_return_engine_weights, close_prices, rdates)
sched_v6, _ = build_schedule_from_fn(v6_aggressive_engine_weights, close_prices, rdates)

ml_fn, ml_results_df, ml_status = meta_engine_ml_selector(close_prices, regime_feats, rdates)
sched_m5, reg_m5 = ({}, {})
if ml_fn:
  sched_m5, reg_m5 = build_schedule_from_fn(ml_fn, close_prices, rdates)

STRATEGIES = {
  "M1_RULE_META_BASE": sched_m1,
  "M2_RULE_META_MORE_AGGRESSIVE": sched_m2,
  "M3_RULE_META_MORE_DEFENSIVE": sched_m3,
  "M4_RULE_META_BENCHMARK_CAPTURE": sched_m4,
  "M6_V6_V14_SIMPLE_50_50": sched_m6,
  "M7_V6_V14_ADAPTIVE": sched_m7,
  "V14_RETURN_ENGINE": sched_v14,
  "V6_AGGRESSIVE_ENGINE": sched_v6,
}
if ml_fn and sched_m5:
  STRATEGIES["M5_ML_META_SELECTOR"] = sched_m5

strategy_results, equities, yearly_all, schedules = [], {}, [], {}
for name, sched in STRATEGIES.items():
  m, eq, yr, sched = run_v15_backtest(name, sched)
  if m:
    strategy_results.append(m)
    equities[name] = eq
    yearly_all.append(yr)
    schedules[name] = sched

results_df = pd.DataFrame(strategy_results)
yearly_df = pd.concat(yearly_all, ignore_index=True) if yearly_all else pd.DataFrame()
print("Estrategias probadas:", len(results_df))
if len(results_df):
  print(results_df[["strategy", "CAGR", "sharpe", "max_drawdown", "win_years_vs_spy"]].to_string(index=False))

# %% [markdown]
# ## 13-14. Robustez y scoring V15

# %%
def compute_robustness_score(results_df):
  if len(results_df) < 2:
    return 80
  gap = results_df["sharpe"].max() - results_df["sharpe"].median()
  penalty = max(0, len(results_df) - 8) * 2
  return int(np.clip(100 - gap * 30 - penalty, 0, 100))


def overfitting_risk(results_df, ml_used=False):
  if len(results_df) < 2:
    return "LOW"
  gap = results_df["sharpe"].max() - results_df["sharpe"].median()
  risk = "HIGH" if gap > 0.55 else ("MEDIUM" if gap > 0.30 else "LOW")
  if ml_used and gap > 0.4:
    risk = "MEDIUM" if risk == "LOW" else risk
  return risk


def avg_max_concentration(schedule):
  if not schedule:
    return 0.0
  return float(np.mean([w.max() for w in schedule.values() if len(w)]))


def compute_v15_score(row, robustness, ovr_risk, spy_total, yearly_row=None, max_conc=0.0, ml_only=False):
  score = 0
  if row.get("sharpe", 0) > 1.1:
    score += 15
  if row.get("sortino", 0) > 1.5:
    score += 15
  if row.get("max_drawdown", -100) > -25:
    score += 15
  if row.get("CAGR", 0) > 15:
    score += 10
  if row.get("total_return", 0) > spy_total:
    score += 10
  if row.get("win_years_vs_spy", 0) >= 0.55:
    score += 10
  if row.get("win_years_vs_qqq", 0) >= 0.40:
    score += 5
  if row.get("win_years_vs_6040", 0) >= 0.75:
    score += 10
  if robustness >= 80:
    score += 10
  if ovr_risk == "HIGH":
    score -= 20
  if max_conc > MAX_WEIGHT + 0.08:
    score -= 20
  if row.get("max_drawdown", -100) < -30:
    score -= 20
  gross = row.get("total_return", 0)
  cost = row.get("total_cost", 0)
  if gross > 0 and cost > gross * 0.35:
    score -= 15
  if yearly_row is not None and len(yearly_row):
    recent = yearly_row[yearly_row["year"].isin([2025, 2026])]
    if len(recent) and (recent["return"] < 0).any():
      score -= 15
  if ml_only:
    score -= 20
  return int(np.clip(score, 0, 100))


robustness_score = compute_robustness_score(results_df)
ovr = overfitting_risk(results_df, ml_used="M5" in results_df["strategy"].values if len(results_df) else False)
spy_total = (close_prices[MARKET].iloc[-1] / close_prices[MARKET].iloc[0] - 1) * 100

if len(results_df):
  conc_map = {n: avg_max_concentration(schedules.get(n, {})) for n in results_df["strategy"]}
  yearly_map = {n: yearly_df[yearly_df["strategy"] == n] for n in results_df["strategy"]}
  results_df["v15_score"] = results_df.apply(
    lambda r: compute_v15_score(
      r, robustness_score, ovr, spy_total,
      yearly_map.get(r["strategy"]),
      conc_map.get(r["strategy"], 0),
      ml_only="ML" in r["strategy"] and r["strategy"] == "M5_ML_META_SELECTOR",
    ),
    axis=1,
  )
  tradable = results_df[~results_df["strategy"].str.contains("RESEARCH", na=False)]
  champion = tradable.sort_values("v15_score", ascending=False).iloc[0]
else:
  champion = pd.Series(dtype=float)

v15_score = int(champion.get("v15_score", 0)) if len(champion) else 0
approved = (
  v15_score >= 85
  and champion.get("sharpe", 0) > 1.1
  and champion.get("max_drawdown", -100) > -25
  and robustness_score >= 80
  and ovr in ("LOW", "MEDIUM")
)
v15_status = "APPROVED_FOR_WEB_PAPER" if approved else ("CANDIDATE" if v15_score >= 70 else "REJECTED")
APPROVED_FOR_WEB_PAPER = approved
print(f"V15: {v15_status} | score {v15_score}")

# ML vs rule comparison
ml_research_only = True
if "M5_ML_META_SELECTOR" in results_df["strategy"].values and "M1_RULE_META_BASE" in results_df["strategy"].values:
  ml_row = results_df[results_df["strategy"] == "M5_ML_META_SELECTOR"].iloc[0]
  rule_row = results_df[results_df["strategy"] == "M1_RULE_META_BASE"].iloc[0]
  if ml_row["sharpe"] > rule_row["sharpe"] and ml_row["v15_score"] >= rule_row["v15_score"] - 5:
    ml_research_only = False
    print("ML meta selector competitivo OOS")
  else:
    print("ML meta selector marcado research-only (no mejora rule-based)")

# Start-date robustness
rob_rows = []
best_name = champion.get("strategy", "M1_RULE_META_BASE") if len(champion) else "M1_RULE_META_BASE"
for sd in START_DATES_ROBUST:
  if sd >= START_DATE:
    continue
  try:
    if pd.Timestamp(sd) >= close_prices.index[0]:
      c2 = close_prices.loc[sd:].copy()
      rf2 = calculate_regime_features(c2)
    else:
      _, c2 = download_data(UNIVERSE, sd, END_DATE, min_days=150)
      rf2 = calculate_regime_features(c2)
    rd2 = [d for d in get_rebalance_dates(c2) if d.year >= WF_START_YEAR]
    old_rf = REGIME_FEATS
    REGIME_FEATS = rf2
    sch, _ = meta_rule_schedule(c2, rd2)
    REGIME_FEATS = old_rf
    old_close, old_data = close_prices, data_dict
    close_prices, data_dict = c2, data_dict
    m2, _, _, _ = run_v15_backtest(best_name, sch)
    close_prices, data_dict = old_close, old_data
    if m2:
      rob_rows.append({"start_date": sd, "strategy": best_name, **m2})
  except Exception:
    pass
robustness_df = pd.DataFrame(rob_rows)

cost_rows = []
champ_sched = schedules.get(best_name, sched_m1)
for cr in COST_SENSITIVITY:
  m3, _, _, _ = run_v15_backtest(best_name, champ_sched, cost_rate=cr)
  if m3:
    cost_rows.append({"cost_slippage": cr, **m3})
cost_sens_df = pd.DataFrame(cost_rows)
contrib_df = calculate_contribution_by_asset(close_prices, champ_sched)

# %% [markdown]
# ## 15. Current signals

# %%
def _engine_source_for_ticker(ticker, dt, close):
  """Atribuye motor dominante por activo."""
  regime, conf = classify_regime(dt)
  eng_w = {}
  for eng, fn in ENGINE_FNS.items():
    ew = fn(close, dt)
    eng_w[eng] = ew.get(ticker, 0)
  best_eng = max(eng_w, key=eng_w.get) if eng_w and max(eng_w.values()) > 0 else "meta_blend"
  return best_eng, regime, conf


def generate_v15_signals(sched, regimes_map, close, strategy_name):
  if not sched:
    return pd.DataFrame()
  last = max(sched.keys())
  prev_keys = sorted([d for d in sched.keys() if d < last])
  target, prev = sched[last], sched[prev_keys[-1]] if prev_keys else pd.Series(dtype=float)
  regime_info = regimes_map.get(last, {})
  regime = regime_info.get("regime", classify_regime(last)[0])
  conf = regime_info.get("confidence", 50)

  rows = []
  for t in sorted(set(target.index) | set(prev.index)):
    tw, pw = _sf(target.get(t, 0), 0), _sf(prev.get(t, 0), 0)
    chg = tw - pw
    eng, _, c = _engine_source_for_ticker(t, last, close)
    if tw > 0 and pw == 0:
      sig = "BUY"
    elif tw > pw + 0.03:
      sig = "INCREASE"
    elif tw > 0 and abs(chg) <= 0.03:
      sig = "HOLD"
    elif pw > 0 and tw == 0:
      sig = "SELL"
    elif tw < pw - 0.03:
      sig = "REDUCE"
    else:
      sig = "AVOID"

    if t in ("QQQ", "SPY", "XLK", "MTUM") and regime == "risk_on_strong":
      reason = f"mercado fuerte + benchmark core | {regime}"
    elif t in DEFENSIVE_POOL and tw > pw:
      reason = f"modo defensivo | {regime}"
    elif sig == "BUY":
      reason = f"top momentum / meta blend | {regime}"
    else:
      reason = f"{eng} | {regime}"

    rows.append({
      "ticker": t, "signal": sig, "target_weight": round(tw, 4), "previous_weight": round(pw, 4),
      "change": round(chg, 4), "engine_source": eng, "regime": regime,
      "confidence": round(c if tw > 0 else conf, 1), "reason": reason,
      "entry_plan": "proxima apertura post-viernes" if sig in ("BUY", "INCREASE") else "-",
      "exit_plan": "rebalance semanal o cambio de regimen" if sig in ("SELL", "REDUCE") else "mantener hasta viernes",
      "next_review": "proximo viernes", "cash_account_executable": True,
    })
  return pd.DataFrame(rows)


champ_name = champion.get("strategy", "M1_RULE_META_BASE") if len(champion) else "M1_RULE_META_BASE"
champ_sched = schedules.get(champ_name, sched_m1)
regimes_champ = reg_m1 if "M1" in champ_name else {}
current_signals = generate_v15_signals(champ_sched, regimes_champ, close_prices, champ_name)
current_regime, current_conf = classify_regime(max(champ_sched.keys()) if champ_sched else rdates[-1])
print(f"Regimen actual: {current_regime} | conf {current_conf:.0f} | Señales: {len(current_signals)}")

# %% [markdown]
# ## 16. Exportar

# %%
def export_csv(df, name):
  (df if df is not None and len(df) else pd.DataFrame()).to_csv(name, index=False)
  print("Exportado", name)


v6_total = np.nan
v14_total = np.nan
for p, col in [
  (Path("research_outputs/v6/research_v6_equity_curves.csv"), "blended_champion_weights_alpha_0.5"),
  (Path("research_v14_equity_curve.csv"), "equity"),
]:
  if p.exists():
    try:
      eq = pd.read_csv(p, index_col=0, parse_dates=True)
      if col in eq.columns and len(eq) >= 2:
        val = (eq[col].iloc[-1] / eq[col].iloc[0] - 1) * 100
        if "v6" in str(p):
          v6_total = val
        else:
          v14_total = val
    except Exception:
      pass
if "V14_RETURN_ENGINE" in equities and len(equities["V14_RETURN_ENGINE"]) >= 2:
  e = equities["V14_RETURN_ENGINE"]
  v14_total = (e.iloc[-1] / e.iloc[0] - 1) * 100
if "V6_AGGRESSIVE_ENGINE" in equities and len(equities["V6_AGGRESSIVE_ENGINE"]) >= 2:
  e = equities["V6_AGGRESSIVE_ENGINE"]
  v6_total = (e.iloc[-1] / e.iloc[0] - 1) * 100

champ_eq = equities.get(champ_name, pd.Series(dtype=float))
summary = {
  "lab": "v15_meta_champion_regime",
  "v15_score": v15_score,
  "status": v15_status,
  "approved_for_web_paper": APPROVED_FOR_WEB_PAPER,
  "approved_for_real_money": False,
  "champion_strategy": champ_name,
  "current_regime": current_regime,
  "robustness_score": robustness_score,
  "overfitting_risk": ovr,
  "ml_research_only": ml_research_only,
  **(champion.to_dict() if len(champion) else {}),
  "spy_return": round(spy_total, 2),
  "v6_return": round(v6_total, 2) if pd.notna(v6_total) else np.nan,
  "v14_return": round(v14_total, 2) if pd.notna(v14_total) else np.nan,
}

export_csv(pd.DataFrame([summary]), "research_v15_summary.csv")
export_csv(results_df, "research_v15_strategy_results.csv")
export_csv(yearly_df, "research_v15_yearly.csv")
export_csv(robustness_df, "research_v15_robustness.csv")
export_csv(cost_sens_df, "research_v15_cost_sensitivity.csv")
export_csv(contrib_df, "research_v15_contribution_by_asset.csv")
export_csv(current_signals, "research_v15_current_signals.csv")
if len(current_signals):
  current_signals.to_csv("current_signals_v15.csv", index=False)
if len(champ_eq):
  champ_eq.to_frame("equity").to_csv("research_v15_equity_curve.csv")
if len(ml_results_df):
  export_csv(ml_results_df, "research_v15_ml_meta_results.csv")

config = {
  "version": "v15_meta_champion_regime",
  "approved_for_web_paper": APPROVED_FOR_WEB_PAPER,
  "approved_for_real_money": False,
  "v15_score": v15_score,
  "status": v15_status,
  "champion": champ_name,
  "current_regime": current_regime,
  "engines": list(ENGINE_FNS.keys()),
  "regime_blend": REGIME_BLEND,
  "ml_research_only": ml_research_only,
  "warnings": [
    "Backtest no garantiza resultados futuros.",
    "Meta-modelo combina V14/V6/benchmark/defensive.",
    "No aprobado para dinero real.",
  ],
}
Path("research_v15_selected_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
print("Exportado research_v15_selected_config.json")

# %% [markdown]
# ## 17. Reporte final

# %%
print("=" * 80)
print("REPORTE FINAL V15 META CHAMPION REGIME LAB")
print("=" * 80)
print(f"Estado: {v15_status} | Score: {v15_score}/100")
print(f"Mejor estrategia: {champ_name}")
print(f"Regimen actual: {current_regime} (conf {current_conf:.0f})")
if len(champion):
  print(f"CAGR {champion.get('CAGR')}% | Sharpe {champion.get('sharpe')} | Sortino {champion.get('sortino')}")
  print(f"DD {champion.get('max_drawdown')}% | Win SPY {champion.get('win_years_vs_spy', 0)*100:.0f}% | Win QQQ {champion.get('win_years_vs_qqq', 0)*100:.0f}% | Win 60/40 {champion.get('win_years_vs_6040', 0)*100:.0f}%")
print(f"Robustez: {robustness_score} | Overfitting: {ovr} | ML research-only: {ml_research_only}")
print(f"SPY: {spy_total:.1f}% | V14: {v14_total if pd.notna(v14_total) else 'N/A'} | V6: {v6_total if pd.notna(v6_total) else 'N/A'}")
if len(current_signals):
  active = current_signals[current_signals["target_weight"] > 0.001]
  print(f"BUY: {len(active[active['signal']=='BUY'])} | HOLD: {len(active[active['signal']=='HOLD'])}")
  print(active[["ticker", "signal", "target_weight", "engine_source", "regime", "reason"]].head(10).to_string(index=False))
print("")
if APPROVED_FOR_WEB_PAPER:
  print("Integrar V15 Meta Champion Regime como motor principal.")
else:
  print("V15 no mejora V14. Mantener V14/V6.")
print("APPROVED_FOR_REAL_MONEY=False (siempre)")
