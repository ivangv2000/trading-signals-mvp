# %% [markdown]
# # Trading Research V16 Drawdown Controlled Meta Champion Lab
#
# V15 M7/M6 + controles de volatilidad, drawdown, estrés y concentración.
#
# **Disclaimer:** Backtest no garantiza resultados futuros. No es asesoramiento financiero.
# **APPROVED_FOR_REAL_MONEY siempre False.** ML no se usa como champion.

# %%
try:
  get_ipython().run_line_magic("pip", "install yfinance pandas numpy matplotlib plotly tqdm scipy scikit-learn -q")
except NameError:
  import subprocess, sys
  subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "yfinance", "pandas", "numpy", "matplotlib", "plotly", "tqdm", "scipy", "scikit-learn"])

# %% [markdown]
# ## 1. Configuracion

# %%
import warnings
warnings.filterwarnings("ignore")
import json, math
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
MAX_WEIGHT_ETF = 0.30
MAX_WEIGHT_STOCK = 0.20
MAX_CONTRIBUTION_PCT = 0.35
VOL_TARGET = 0.15
VOL_TARGET_DEFENSIVE = 0.10
DRAWDOWN_SOFT_LIMIT = -0.15
DRAWDOWN_HARD_LIMIT = -0.22
MIN_HISTORY_DAYS = 200
MARKET = "SPY"
COST_RATE = TRANSACTION_COST + SLIPPAGE
TOP_N_V14 = 3
MAX_WEIGHT = MAX_WEIGHT_ETF

UNIVERSE = [
  "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLC", "XLB", "XLRE",
  "MTUM", "QUAL", "USMV", "VLUE", "SPLV", "SPHB", "SCHD", "SHY", "IEF", "TLT", "LQD", "HYG",
  "GLD", "SLV", "DBC", "VNQ", "EFA", "EEM",
  "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "GOOGL", "META", "AMZN",
  "JPM", "BAC", "XOM", "CVX", "UNH", "LLY", "WMT", "COST",
]
DEFENSIVE_POOL = ["SHY", "IEF", "TLT", "GLD", "USMV", "QUAL", "SCHD", "XLV", "XLP"]
DEFENSIVE_TICKERS = set(DEFENSIVE_POOL + [CASH_ASSET])
V6_UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN"]
ETF_SET = {
  "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLC", "XLB", "XLRE",
  "MTUM", "QUAL", "USMV", "VLUE", "SPLV", "SPHB", "SCHD", "SHY", "IEF", "TLT", "LQD", "HYG",
  "GLD", "SLV", "DBC", "VNQ", "EFA", "EEM",
}
STOCK_SET = set(UNIVERSE) - ETF_SET
SPECIAL_CAPS = {"NVDA": 0.20, "AAPL": 0.20, "MSFT": 0.20, "QQQ": 0.25}
ASSET_TYPE_MAP = {t: ("etf" if t in ETF_SET else "stock") for t in UNIVERSE}
START_DATES_ROBUST = ["2010-01-01", "2015-01-01", "2018-01-01"] if QUICK_TEST else [
  "2010-01-01", "2012-01-01", "2015-01-01", "2018-01-01", "2020-01-01",
]
COST_SENSITIVITY = [0.0005, 0.001, 0.002] if not QUICK_TEST else [0.0005, 0.001]

if QUICK_TEST:
  UNIVERSE = ["SPY", "QQQ", "IWM", "XLK", "XLV", "XLF", "TLT", "IEF", "SHY", "GLD", "MTUM", "USMV", "AAPL", "MSFT", "NVDA"]
  START_DATE = "2015-01-01"
  ETF_SET = ETF_SET & set(UNIVERSE)
  STOCK_SET = set(UNIVERSE) - ETF_SET
  ASSET_TYPE_MAP = {t: ("etf" if t in ETF_SET else "stock") for t in UNIVERSE}
  print("QUICK_TEST activo")

print("V16 Drawdown Controlled Meta Lab | desde", START_DATE)

# %% [markdown]
# ## 2. Descarga y funciones comunes (base V15)

# %%
def download_data(tickers, start, end=None, min_days=MIN_HISTORY_DAYS):
  data, failed = {}, []
  for ticker in tqdm(sorted(set(tickers)), desc="Download"):
    try:
      raw = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
      if raw is None or raw.empty:
        failed.append(ticker); continue
      df = raw.copy()
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
      colmap = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
      df = df.rename(columns={c: colmap.get(str(c).lower(), c) for c in df.columns})
      keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
      df = df[keep].dropna(subset=["Close"])
      if len(df) < min_days:
        failed.append(f"{ticker}(short)"); continue
      df.index = pd.DatetimeIndex(df.index)
      if df.index.tz:
        df.index = df.index.tz_localize(None)
      data[ticker.upper()] = df.sort_index()
    except Exception:
      failed.append(ticker)
  close = pd.DataFrame({k: v["Close"] for k, v in data.items()}).sort_index().ffill()
  return data, close


def _sf(x, d=np.nan):
  try:
    v = float(x); return d if not np.isfinite(v) else v
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


def cap_and_redistribute_weights(w, cap=MAX_WEIGHT_ETF):
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


def inverse_vol_weights(tickers, vols, cap=MAX_WEIGHT_ETF):
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
  rebal_exec = {dates[dates > d][0]: w for d, w in weight_schedule.items() if len(dates[dates > d])}
  equity, curve, current_w, turnover_log = initial, {}, pd.Series(dtype=float), []
  port_rets = []
  for i, dt in enumerate(dates):
    if dt in rebal_exec:
      nw = rebal_exec[dt]
      to = calculate_turnover(nw, current_w) if len(current_w) else nw.abs().sum()
      equity *= (1 - to * cost_rate)
      turnover_log.append({"date": dt, "turnover": to, "cost": equity * to * cost_rate})
      current_w = nw.copy()
    if i > 0 and len(current_w):
      prev = dates[i - 1]
      dr = sum(current_w.get(t, 0) * (close[t].loc[dt] / close[t].loc[prev] - 1)
               for t in current_w.index if t in close.columns and np.isfinite(close[t].loc[dt]))
      equity *= (1 + dr)
      port_rets.append(dr)
    curve[dt] = equity
  return pd.Series(curve), pd.DataFrame(turnover_log), port_rets


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
    mix = 0.6 * spy_r + 0.4 * (tlt.iloc[-1] / tlt.iloc[0] - 1 if len(tlt) >= 2 else 0)
    rows.append({"year": year, "return": round(sr * 100, 2), "SPY": round(spy_r * 100, 2),
                 "QQQ": round(qqq_r * 100, 2), "6040": round(mix * 100, 2),
                 "beats_spy": sr > spy_r, "beats_qqq": sr > qqq_r, "beats_6040": sr > mix})
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
  safe = CASH_ASSET if CASH_ASSET in close_cols else "SHY"
  w[safe] = w.get(safe, 0) + max(0, frac - w.sum())
  return cap_and_redistribute_weights(w)


def _defensive_only(cols):
  if CASH_ASSET in cols:
    return pd.Series({CASH_ASSET: 1.0})
  return pd.Series({"SHY": 1.0})


data_dict, close_prices = download_data(UNIVERSE, START_DATE, END_DATE)
print("Tickers:", len(close_prices.columns))

# %% [markdown]
# ## 3. Regimen y motores V15 (sin ML champion)

# %%
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
  if "HYG" in close.columns and "LQD" in close.columns:
    feats["HYG_LQD_MOM"] = close["HYG"] / close["HYG"].shift(63) - close["LQD"] / close["LQD"].shift(63)
  if "GLD" in close.columns:
    feats["GLD_TREND"] = (close["GLD"] > close["GLD"].rolling(100).mean()).astype(float)
  if "TLT" in close.columns:
    feats["TLT_TREND"] = (close["TLT"] > close["TLT"].rolling(100).mean()).astype(float)
  spy_above = feats.get(f"{MARKET}_ABOVE_SMA200", pd.Series(0, index=close.index)).fillna(0)
  qqq_above = feats.get("QQQ_ABOVE_SMA200", spy_above).fillna(0)
  spy_mom = feats.get(f"{MARKET}_MOM_63", pd.Series(0, index=close.index)).fillna(0)
  qqq_mom = feats.get("QQQ_MOM_63", pd.Series(0, index=close.index)).fillna(0)
  spy_dd = feats.get(f"{MARKET}_DD_63", pd.Series(0, index=close.index)).fillna(0)
  feats["risk_on_score"] = (spy_above * 25 + qqq_above * 25 + (spy_mom > 0).astype(float) * 20
                            + (qqq_mom > 0).astype(float) * 20 + (spy_mom > 0.05).astype(float) * 10).clip(0, 100)
  feats["defensive_score"] = ((spy_above < 0.5).astype(float) * 30 + (qqq_above < 0.5).astype(float) * 25
                                + (spy_mom < 0).astype(float) * 25 + (spy_dd < -0.08).astype(float) * 20).clip(0, 100)
  feats["crash_risk_score"] = ((spy_dd < -0.10).astype(float) * 40
                               + (feats.get(f"{MARKET}_VOL_21", 0) > 0.25).astype(float) * 30
                               + (spy_mom < -0.05).astype(float) * 30).clip(0, 100)
  return feats.fillna(0)


REGIME_FEATS = None


def _row_regime(rf, dt):
  if rf is None:
    return pd.Series(dtype=float)
  dt = _align_idx(rf, dt)
  return rf.loc[dt] if dt in rf.index else pd.Series(dtype=float)


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
  return "mixed", risk_on


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
  return {"mom63": _sf(mom63), "above_sma200": float(c.iloc[-1] > sma200) if np.isfinite(sma200) else 0,
          "vol63": _sf(vol63, 0.15), "vol20": _sf(vol20, 0.15),
          "trend": float(c.iloc[-1] > sma200) * 0.6 + float(mom63 > 0) * 0.4}


def v14_return_engine_weights(close, dt):
  regime, _ = classify_regime(dt)
  risk_on = regime in ("risk_on_strong", "risk_on_normal")
  scores, vols = {}, {}
  for t in close.columns:
    if t in DEFENSIVE_POOL or t == CASH_ASSET:
      continue
    af = _asset_feats_at(close, dt, t)
    if af is None or af["mom63"] <= 0 or af["above_sma200"] < 0.5 or af["vol63"] > 0.45:
      continue
    scores[t] = af["mom63"] * 0.5 + af["trend"] * 0.5
    vols[t] = af["vol63"]
  if not risk_on:
    scores = {t: (_asset_feats_at(close, dt, t) or {}).get("trend", 0) for t in DEFENSIVE_POOL if t in close.columns}
    vols = {t: 0.12 for t in scores}
  top = pd.Series(scores).sort_values(ascending=False).head(TOP_N_V14)
  if top.empty:
    return _defensive_only(close.columns)
  w = inverse_vol_weights(top.index, vols) * min(1, VOL_TARGET / 0.15) * (0.9 if risk_on else 0.55)
  return _cash_fill(w, close.columns)


def _v6_trend_weights(close, dt):
  elig, vols = [], {}
  for t in V6_UNIVERSE:
    if t not in close.columns:
      continue
    af = _asset_feats_at(close, dt, t)
    if af is None:
      continue
    c = close[t].loc[:_align_idx(close, dt)]
    sma200 = c.rolling(200).mean().iloc[-1]
    ema50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
    if c.iloc[-1] > sma200 and ema50 > sma200:
      elig.append(t); vols[t] = af["vol20"]
  return inverse_vol_weights(elig, vols) if elig else None


def _v6_adaptive_weights(close, dt):
  regime, _ = classify_regime(dt)
  risk_on = regime in ("risk_on_strong", "risk_on_normal", "mixed")
  if risk_on:
    scores = {t: (_asset_feats_at(close, dt, t)["mom63"] / max(_asset_feats_at(close, dt, t)["vol63"], 0.05))
              for t in V6_UNIVERSE if t in close.columns and _asset_feats_at(close, dt, t)
              and _asset_feats_at(close, dt, t)["mom63"] > 0 and _asset_feats_at(close, dt, t)["above_sma200"] >= 0.5}
    top = pd.Series(scores).sort_values(ascending=False).head(4)
    if top.empty:
      return None
    vols = {t: _asset_feats_at(close, dt, t)["vol63"] for t in top.index}
    return _cash_fill(inverse_vol_weights(top.index, vols) * 0.85, close.columns)
  pool = [t for t in DEFENSIVE_POOL if t in close.columns]
  scores = {t: (_asset_feats_at(close, dt, t) or {}).get("trend", 0) for t in pool}
  top = pd.Series(scores).sort_values(ascending=False).head(3)
  return _cash_fill(pd.Series({t: 0.85 / len(top) for t in top.index}), close.columns) if len(top) else None


def v6_aggressive_engine_weights(close, dt):
  w1, w2 = _v6_trend_weights(close, dt), _v6_adaptive_weights(close, dt)
  if w1 is None and w2 is None:
    return _defensive_only(close.columns)
  if w1 is None:
    return w2
  if w2 is None:
    return w1
  blend = w1 * 0.5 + w2.reindex(w1.index, fill_value=0) * 0.5
  for t in w2.index:
    blend[t] = blend.get(t, 0) + w2.get(t, 0) * 0.5
  return cap_and_redistribute_weights(blend)


def defensive_engine_weights(close, dt):
  scores, vols = {}, {}
  for t in DEFENSIVE_POOL:
    if t not in close.columns:
      continue
    af = _asset_feats_at(close, dt, t)
    if af and (af["mom63"] > 0 or af["trend"] > 0.3):
      scores[t] = af["trend"] + 0.1 / max(af["vol63"], 0.05)
      vols[t] = af["vol63"]
  top = pd.Series(scores).sort_values(ascending=False).head(5)
  return _cash_fill(inverse_vol_weights(top.index, vols) * 0.9, close.columns) if len(top) else _defensive_only(close.columns)


def benchmark_core_engine_weights(close, dt):
  regime, _ = classify_regime(dt)
  w = pd.Series(dtype=float)
  if regime == "risk_on_strong":
    for t, pct in [("QQQ", 0.40), ("SPY", 0.30)]:
      if t in close.columns:
        w[t] = pct
    for sat in ["XLK", "MTUM"]:
      if sat in close.columns and (_asset_feats_at(close, dt, sat) or {}).get("above_sma200", 0) >= 0.5:
        w[sat] = 0.15; break
    if CASH_ASSET in close.columns:
      w[CASH_ASSET] = w.get(CASH_ASSET, 0) + 0.075
    if "GLD" in close.columns:
      w["GLD"] = w.get("GLD", 0) + 0.075
  elif regime == "risk_on_normal":
    for t, pct in [("SPY", 0.35), ("QQQ", 0.25)]:
      if t in close.columns:
        w[t] = pct
    w = w.add(v14_return_engine_weights(close, dt) * 0.20, fill_value=0)
    for t in [CASH_ASSET, "IEF"]:
      if t in close.columns:
        w[t] = w.get(t, 0) + 0.10
  else:
    for t, pct in [(CASH_ASSET, 0.50), ("IEF", 0.125), ("TLT", 0.125), ("GLD", 0.15)]:
      if t in close.columns:
        w[t] = pct
  return cap_and_redistribute_weights(w) if w.sum() > 0 else _defensive_only(close.columns)


def m6_blend_50_50(close, dt):
  w1, w2 = v14_return_engine_weights(close, dt), v6_aggressive_engine_weights(close, dt)
  w = w1 * 0.5
  for t in w2.index:
    w[t] = w.get(t, 0) + w2.get(t, 0) * 0.5
  regime, conf = classify_regime(dt)
  return cap_and_redistribute_weights(w), regime, conf


def m7_adaptive(close, dt):
  regime, conf = classify_regime(dt)
  w14, w6, wd = v14_return_engine_weights(close, dt), v6_aggressive_engine_weights(close, dt), defensive_engine_weights(close, dt)
  if regime in ("risk_on_strong", "risk_on_normal"):
    w = w14 * 0.60
    for t in w6.index:
      w[t] = w.get(t, 0) + w6.get(t, 0) * 0.40
  elif regime == "mixed":
    w = w14 * 0.50
    for t in w6.index:
      w[t] = w.get(t, 0) + w6.get(t, 0) * 0.50
    w = w * 0.70 + wd.reindex(w.index, fill_value=0) * 0.30
  else:
    w = wd * 0.80 + w14.reindex(wd.index, fill_value=0) * 0.20
  return cap_and_redistribute_weights(w), regime, conf


def m7_smooth_regime(close, dt):
  rf = _row_regime(REGIME_FEATS, dt)
  risk_on = _sf(rf.get("risk_on_score", 50), 50)
  stress = calculate_market_stress_score(close, dt)
  x = (risk_on - stress - 50) / 12.0
  risk_weight = 1.0 / (1.0 + math.exp(-x))
  w14 = v14_return_engine_weights(close, dt)
  w6 = v6_aggressive_engine_weights(close, dt)
  wd = defensive_engine_weights(close, dt)
  w = pd.Series(dtype=float)
  for t in set(w14.index) | set(w6.index) | set(wd.index):
    w[t] = w14.get(t, 0) * (0.35 + 0.35 * risk_weight) + w6.get(t, 0) * (0.25 + 0.20 * risk_weight) + wd.get(t, 0) * (0.40 * (1 - risk_weight))
  regime, conf = classify_regime(dt)
  return cap_and_redistribute_weights(w), regime, conf


def v16_g_benchmark_capture(close, dt):
  regime, conf = classify_regime(dt)
  w, _, _ = m7_adaptive(close, dt)
  if regime in ("risk_on_strong", "risk_on_normal"):
    bench = sum(w.get(t, 0) for t in ["SPY", "QQQ", "XLK", "MTUM"] if t in w.index)
    if bench < 0.45:
      boost = 0.45 - bench
      for t in ["QQQ", "SPY", "MTUM", "XLK"]:
        if t in close.columns:
          w[t] = w.get(t, 0) + boost / 2
          break
  return cap_and_redistribute_weights(w), regime, conf


REGIME_FEATS = calculate_regime_features(close_prices)
rdates = [d for d in get_rebalance_dates(close_prices) if d.year >= WF_START_YEAR]
print("Rebalance dates:", len(rdates))

# %% [markdown]
# ## 4-8. Controles V16

# %%
def calculate_market_stress_score(close, dt):
  rf = _row_regime(REGIME_FEATS, dt)
  if rf.empty:
    return 50.0
  score = 0.0
  if _sf(rf.get(f"{MARKET}_ABOVE_SMA200", 1)) < 0.5:
    score += 20
  if _sf(rf.get("QQQ_ABOVE_SMA200", 1)) < 0.5:
    score += 20
  score += min(20, abs(min(0, _sf(rf.get(f"{MARKET}_DD_63", 0)))) * 100)
  score += min(15, abs(min(0, _sf(rf.get("QQQ_DD_63", _sf(rf.get(f"{MARKET}_DD_63", 0)))))) * 80)
  vol21 = _sf(rf.get(f"{MARKET}_VOL_21", 0.15))
  vol63 = _sf(rf.get(f"{MARKET}_VOL_63", 0.15))
  score += min(15, max(0, (vol21 - 0.15) * 80))
  score += min(10, max(0, (vol63 - 0.18) * 60))
  if _sf(rf.get("HYG_LQD_MOM", 0)) < 0:
    score += 10
  if _sf(rf.get("TLT_TREND", 0)) >= 0.5:
    score += 5
  if _sf(rf.get("GLD_TREND", 0)) >= 0.5:
    score += 5
  return float(np.clip(score, 0, 100))


def classify_stress(score):
  if score >= 75:
    return "crash_stress"
  if score >= 55:
    return "high_stress"
  if score >= 35:
    return "medium_stress"
  return "low_stress"


def _defensive_blend(cols):
  parts = [t for t in [CASH_ASSET, "IEF", "GLD"] if t in cols]
  if not parts:
    return _defensive_only(cols)
  return safe_normalize_weights(pd.Series({t: 1.0 / len(parts) for t in parts}))


def apply_volatility_managed_overlay(base_weights, close, dt, port_rets, vol_ewma=1.0, target_vol=VOL_TARGET):
  if len(port_rets) < 21:
    return base_weights.copy(), vol_ewma
  r21 = pd.Series(port_rets[-21:])
  r63 = pd.Series(port_rets[-63:]) if len(port_rets) >= 63 else r21
  vol21 = r21.std() * math.sqrt(252)
  vol63 = r63.std() * math.sqrt(252)
  realized = max(_sf(vol21, 0.15), 0.75 * _sf(vol63, 0.15))
  scale = np.clip(target_vol / max(realized, 0.05), 0.35, 1.20)
  vol_ewma = 0.30 * scale + 0.70 * vol_ewma
  risky = base_weights.drop(list(DEFENSIVE_TICKERS), errors="ignore")
  safe = _defensive_blend(close.columns)
  w = risky * vol_ewma
  residual = max(0, 1.0 - w.sum())
  w = w.add(safe * residual, fill_value=0)
  return cap_and_redistribute_weights(w), vol_ewma


def apply_drawdown_circuit_breaker(base_weights, eq_hist, dt):
  if len(eq_hist) < 5:
    return base_weights.copy()
  eq = pd.Series(dict(eq_hist))
  peak = eq.cummax()
  dd = (eq.iloc[-1] - peak.iloc[-1]) / peak.iloc[-1] if peak.iloc[-1] > 0 else 0
  recovery = eq.iloc[-1] >= eq.iloc[-min(100, len(eq)):].mean() if len(eq) >= 20 else False
  cut = 0.0
  if dd < DRAWDOWN_HARD_LIMIT:
    cut = 0.60
  elif dd < DRAWDOWN_SOFT_LIMIT:
    cut = 0.30
  if recovery and cut > 0:
    cut *= 0.50
  if cut <= 0:
    return base_weights.copy()
  risky = base_weights.drop(list(DEFENSIVE_TICKERS), errors="ignore")
  safe = _defensive_blend(base_weights.index)
  return cap_and_redistribute_weights(risky * (1 - cut) + safe * cut)


def apply_market_stress_filter(base_weights, close, dt):
  stress_score = calculate_market_stress_score(close, dt)
  level = classify_stress(stress_score)
  cut_map = {"low_stress": 0.0, "medium_stress": 0.20, "high_stress": 0.50, "crash_stress": 0.90}
  cut = cut_map[level]
  if cut <= 0:
    return base_weights.copy(), level, stress_score
  risky = base_weights.drop(list(DEFENSIVE_TICKERS), errors="ignore")
  safe = _defensive_blend(close.columns)
  w = cap_and_redistribute_weights(risky * (1 - cut) + safe * cut)
  return w, level, stress_score


def apply_concentration_cap(weights, asset_type_map, close, dt):
  w = weights.copy()
  regime, _ = classify_regime(dt)
  redist_pool = [t for t in ["SPY", "MTUM", "USMV", CASH_ASSET, "GLD"] if t in close.columns]
  for t in list(w.index):
    cap = SPECIAL_CAPS.get(t)
    if cap is None:
      cap = MAX_WEIGHT_ETF if asset_type_map.get(t, "etf") == "etf" else MAX_WEIGHT_STOCK
    if w.get(t, 0) > cap:
      excess = w[t] - cap
      w[t] = cap
      for r in redist_pool:
        if r != t:
          w[r] = w.get(r, 0) + excess / max(1, len(redist_pool) - 1)
  return cap_and_redistribute_weights(w)


def smooth_weights_ewma(new_w, prev_w, alpha=0.25):
  if prev_w is None or not len(prev_w):
    return new_w
  idx = new_w.index.union(prev_w.index)
  return cap_and_redistribute_weights(prev_w.reindex(idx, fill_value=0) * (1 - alpha) + new_w.reindex(idx, fill_value=0) * alpha)


def recent_years_penalty(yearly_df):
  if yearly_df is None or len(yearly_df) == 0:
    return 0, "NEUTRAL"
  score = 0
  y2025 = yearly_df[yearly_df["year"] == 2025]
  y2026 = yearly_df[yearly_df["year"] == 2026]
  if len(y2025) and not y2025.iloc[0]["beats_spy"]:
    score -= 8
  if len(y2026) and not y2026.iloc[0]["beats_spy"]:
    score -= 7
  recent = yearly_df.sort_values("year").tail(2)
  if len(recent) >= 2 and not recent["beats_spy"].all():
    score -= 5
  label = "POSITIVE" if score >= 0 else ("NEGATIVE" if score <= -10 else "NEUTRAL")
  return score, label


def concentration_risk_label(contrib_df):
  if contrib_df is None or len(contrib_df) == 0:
    return "LOW", 0.0
  top_pct = contrib_df.iloc[0]["pct_of_total"] if "pct_of_total" in contrib_df.columns else 0
  if top_pct > MAX_CONTRIBUTION_PCT * 100:
    return "HIGH", top_pct
  if top_pct > 25:
    return "MEDIUM", top_pct
  return "LOW", top_pct

# %% [markdown]
# ## 9. Build schedule con controles (equity-aware)

# %%
def _extract_base(fn, close, dt):
  out = fn(close, dt)
  if isinstance(out, tuple):
    return out[0], out[1], out[2] if len(out) > 2 else 50
  return out, *classify_regime(dt)


def _simulate_segment_equity(equity, current_w, close, d0, d1, port_rets):
  seg = close.loc[d0:d1]
  if len(seg) < 2 or not len(current_w):
    return equity, port_rets
  for j in range(1, len(seg)):
    dt, prev = seg.index[j], seg.index[j - 1]
    dr = sum(current_w.get(t, 0) * (close[t].loc[dt] / close[t].loc[prev] - 1)
             for t in current_w.index if t in close.columns and np.isfinite(close[t].loc[dt]))
    equity *= (1 + dr)
    port_rets.append(dr)
  return equity, port_rets


def build_v16_schedule(base_fn, controls, close, rdates):
  schedule, weights_rows, meta = {}, [], {}
  dates = close.index
  equity = INITIAL_CAPITAL
  eq_hist = []
  current_w = pd.Series(dtype=float)
  port_rets = []
  vol_ewma = 1.0
  prev_target = pd.Series(dtype=float)
  stress_level = "low_stress"
  last_dt = dates[0]

  for i, sig_dt in enumerate(rdates):
    sig_dt = _align_idx(close, sig_dt)
    future = dates[dates > sig_dt]
    if len(future) == 0:
      continue
    exec_dt = future[0]
    equity, port_rets = _simulate_segment_equity(equity, current_w, close, last_dt, sig_dt, port_rets)
    eq_hist.append((sig_dt, equity))

    base_w, regime, conf = _extract_base(base_fn, close, sig_dt)
    w = base_w.copy()
    dd_active = stress_active = vol_active = False

    if controls.get("vol"):
      w, vol_ewma = apply_volatility_managed_overlay(w, close, sig_dt, port_rets, vol_ewma)
      vol_active = True
    if controls.get("stress"):
      w, stress_level, _ = apply_market_stress_filter(w, close, sig_dt)
      stress_active = stress_level not in ("low_stress",)
    if controls.get("concentration"):
      w = apply_concentration_cap(w, ASSET_TYPE_MAP, close, sig_dt)
    if controls.get("dd"):
      w = apply_drawdown_circuit_breaker(w, eq_hist, sig_dt)
      eq_s = pd.Series(dict(eq_hist))
      peak = eq_s.cummax()
      dd_active = (eq_s.iloc[-1] - peak.iloc[-1]) / peak.iloc[-1] < DRAWDOWN_SOFT_LIMIT if len(eq_s) else False
    if controls.get("smooth"):
      w = smooth_weights_ewma(w, prev_target, alpha=0.25)

    w = cap_and_redistribute_weights(w)
    schedule[sig_dt] = w
    prev_target = w.copy()
    meta[sig_dt] = {"regime": regime, "confidence": conf, "stress_level": stress_level,
                    "dd_active": dd_active, "vol_active": vol_active, "stress_active": stress_active}

    to = calculate_turnover(w, current_w) if len(current_w) else w.sum()
    equity *= (1 - to * COST_RATE)
    current_w = w.copy()
    eq_hist.append((exec_dt, equity))
    weights_rows.append({"signal_date": sig_dt, "exec_date": exec_dt, "regime": regime,
                         "stress_level": stress_level, "equity": equity,
                         **{f"w_{k}": round(v, 4) for k, v in w.items()}})
    last_dt = exec_dt

  return schedule, pd.DataFrame(weights_rows), meta


# %% [markdown]
# ## 10. Backtest V16 y estrategias

# %%
def run_v16_backtest(name, schedule, cost_rate=COST_RATE):
  if not schedule:
    return {}, pd.Series(dtype=float), pd.DataFrame(), schedule, pd.DataFrame()
  eq, to, _ = build_equity_curve_from_weights(close_prices, schedule, data_dict, cost_rate=cost_rate)
  if len(eq) < 2:
    return {}, eq, to, schedule, pd.DataFrame()
  m, yearly = compare_to_benchmarks(eq, close_prices)
  contrib = calculate_contribution_by_asset(close_prices, schedule)
  conc_label, top_pct = concentration_risk_label(contrib)
  ry_score, ry_label = recent_years_penalty(yearly)
  m.update({"strategy": name, "turnover": round(to["turnover"].mean(), 4) if len(to) else 0,
            "total_cost": round(to["cost"].sum(), 2) if len(to) else 0,
            "num_rebalances": len(schedule), "exposure": round(1 - m.get("turnover", 0), 3),
            "concentration_risk": conc_label, "top_contributor_pct": round(top_pct, 2),
            "recent_years_score": ry_score, "recent_years_label": ry_label})
  yearly["strategy"] = name
  return m, eq, yearly, schedule, contrib


V16_STRATEGIES = {
  "V16_A_M7_VOL_MANAGED": (m7_adaptive, {"vol": True}),
  "V16_B_M7_DD_BREAKER": (m7_adaptive, {"dd": True}),
  "V16_C_M7_STRESS_FILTER": (m7_adaptive, {"stress": True}),
  "V16_D_M7_ALL_CONTROLS": (m7_adaptive, {"vol": True, "dd": True, "stress": True, "concentration": True}),
  "V16_E_M6_ALL_CONTROLS": (m6_blend_50_50, {"vol": True, "dd": True, "stress": True, "concentration": True}),
  "V16_F_V14_AGGRESSIVE_CAPPED": (v14_return_engine_weights, {"vol": True, "concentration": True}),
  "V16_G_BENCHMARK_CAPTURE_CAPPED": (v16_g_benchmark_capture, {"vol": True, "dd": True, "concentration": True}),
  "V16_H_SMOOTH_REGIME_ALLOC": (m7_smooth_regime, {"vol": True, "dd": True, "stress": True, "concentration": True, "smooth": True}),
}
BASELINES = {
  "V15_M7_BASELINE": (m7_adaptive, {}),
  "V14_RETURN_ENGINE": (v14_return_engine_weights, {}),
  "V6_AGGRESSIVE_ENGINE": (v6_aggressive_engine_weights, {}),
}

print("Construyendo V16 schedules...")
strategy_results, equities, yearly_all, schedules, contribs, weights_hist, metas = [], {}, [], {}, {}, {}, {}
all_strats = {**V16_STRATEGIES, **BASELINES}
for name, (base_fn, ctrl) in all_strats.items():
  sched, wh, meta = build_v16_schedule(base_fn, ctrl, close_prices, rdates)
  m, eq, yr, sched, contrib = run_v16_backtest(name, sched)
  if m:
    strategy_results.append(m)
    equities[name] = eq
    yearly_all.append(yr)
    schedules[name] = sched
    contribs[name] = contrib
    weights_hist[name] = wh
    metas[name] = meta

results_df = pd.DataFrame(strategy_results)
yearly_df = pd.concat(yearly_all, ignore_index=True) if yearly_all else pd.DataFrame()
print("Estrategias:", len(results_df))
if len(results_df):
  print(results_df[["strategy", "CAGR", "sharpe", "max_drawdown", "win_years_vs_spy", "concentration_risk"]].to_string(index=False))

# %% [markdown]
# ## 11. Robustez y scoring V16

# %%
def compute_robustness_score(results_df):
  if len(results_df) < 2:
    return 85
  gap = results_df["sharpe"].max() - results_df["sharpe"].median()
  return int(np.clip(100 - gap * 35 - max(0, len(results_df) - 10) * 2, 0, 100))


def overfitting_risk(results_df):
  if len(results_df) < 2:
    return "LOW"
  gap = results_df["sharpe"].max() - results_df["sharpe"].median()
  return "HIGH" if gap > 0.50 else ("MEDIUM" if gap > 0.28 else "LOW")


def compute_v16_score(row, robustness, ovr, spy_total, conc_label, ry_score):
  score = 0
  if row.get("sharpe", 0) > 1.15:
    score += 15
  if row.get("sortino", 0) > 1.5:
    score += 15
  if row.get("max_drawdown", -100) > -25:
    score += 20
  if row.get("CAGR", 0) > 15:
    score += 10
  if row.get("total_return", 0) > spy_total:
    score += 10
  if row.get("win_years_vs_spy", 0) >= 0.60:
    score += 10
  if row.get("win_years_vs_qqq", 0) >= 0.40:
    score += 5
  if row.get("win_years_vs_6040", 0) >= 0.75:
    score += 10
  if conc_label == "LOW":
    score += 5
  if ry_score >= 0:
    score += 5
  if row.get("max_drawdown", -100) < -30:
    score -= 25
  if conc_label == "HIGH":
    score -= 20
  if ovr == "HIGH":
    score -= 20
  if ry_score <= -10:
    score -= 15
  gross, cost = row.get("total_return", 0), row.get("total_cost", 0)
  if gross > 0 and cost > gross * 0.35:
    score -= 15
  if row.get("turnover", 0) > 0.35:
    score -= 15
  top_pct = row.get("top_contributor_pct", 0)
  if top_pct > MAX_CONTRIBUTION_PCT * 100:
    score -= 15
  return int(np.clip(score, 0, 100))


robustness_score = compute_robustness_score(results_df[results_df["strategy"].str.startswith("V16")])
ovr = overfitting_risk(results_df)
spy_total = (close_prices[MARKET].iloc[-1] / close_prices[MARKET].iloc[0] - 1) * 100

if len(results_df):
  v16_only = results_df[results_df["strategy"].str.startswith("V16")].copy()
  v16_only["v16_score"] = v16_only.apply(
    lambda r: compute_v16_score(r, robustness_score, ovr, spy_total, r.get("concentration_risk", "MEDIUM"), r.get("recent_years_score", 0)),
    axis=1,
  )
  results_df = results_df.merge(v16_only[["strategy", "v16_score"]], on="strategy", how="left")
  results_df["v16_score"] = results_df["v16_score"].fillna(0)
  champion = v16_only.sort_values("v16_score", ascending=False).iloc[0]
else:
  champion = pd.Series(dtype=float)

v16_score = int(champion.get("v16_score", 0)) if len(champion) else 0
approved = (
  v16_score >= 85 and champion.get("sharpe", 0) > 1.15
  and champion.get("max_drawdown", -100) > -25
  and champion.get("concentration_risk", "HIGH") != "HIGH"
  and ovr in ("LOW", "MEDIUM")
)
v16_status = "APPROVED_FOR_WEB_PAPER" if approved else ("CANDIDATE" if v16_score >= 70 else "REJECTED")
print(f"V16: {v16_status} | score {v16_score}")

rob_rows = []
best_name = champion.get("strategy", "V16_D_M7_ALL_CONTROLS") if len(champion) else "V16_D_M7_ALL_CONTROLS"
best_ctrl = V16_STRATEGIES.get(best_name, (m7_adaptive, {}))[1]
for sd in START_DATES_ROBUST:
  if sd >= START_DATE:
    continue
  old_rf, old_close = REGIME_FEATS, close_prices
  try:
    if pd.Timestamp(sd) >= close_prices.index[0]:
      c2 = close_prices.loc[sd:].copy()
    else:
      _, c2 = download_data(UNIVERSE, sd, END_DATE, min_days=150)
    REGIME_FEATS = calculate_regime_features(c2)
    close_prices = c2
    rd2 = [d for d in get_rebalance_dates(c2) if d.year >= WF_START_YEAR]
    bf = V16_STRATEGIES.get(best_name, (m7_adaptive, {}))[0]
    sch, _, _ = build_v16_schedule(bf, best_ctrl, c2, rd2)
    m2, _, _, _, _ = run_v16_backtest(best_name, sch)
    if m2:
      rob_rows.append({"start_date": sd, "strategy": best_name, **m2})
  except Exception:
    pass
  finally:
    REGIME_FEATS, close_prices = old_rf, old_close
robustness_df = pd.DataFrame(rob_rows)

cost_rows = []
champ_sched = schedules.get(best_name, {})
for cr in COST_SENSITIVITY:
  m3, _, _, _, _ = run_v16_backtest(best_name, champ_sched, cost_rate=cr)
  if m3:
    cost_rows.append({"cost_slippage": cr, **m3})
cost_sens_df = pd.DataFrame(cost_rows)
contrib_df = contribs.get(best_name, pd.DataFrame())

# %% [markdown]
# ## 12. Current signals V16

# %%
def generate_v16_signals(sched, meta, close, strategy_name):
  if not sched:
    return pd.DataFrame()
  last = max(sched.keys())
  prev_keys = sorted([d for d in sched.keys() if d < last])
  target, prev = sched[last], sched[prev_keys[-1]] if prev_keys else pd.Series(dtype=float)
  info = meta.get(last, {})
  regime = info.get("regime", classify_regime(last)[0])
  stress = info.get("stress_level", classify_stress(calculate_market_stress_score(close, last)))
  dd_active = info.get("dd_active", False)
  rows = []
  for t in sorted(set(target.index) | set(prev.index)):
    tw, pw = _sf(target.get(t, 0), 0), _sf(prev.get(t, 0), 0)
    chg = tw - pw
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
    eng = "v14_return" if t in STOCK_SET else ("benchmark_core" if t in ("SPY", "QQQ", "XLK", "MTUM") else "defensive")
    reason_parts = [eng, regime]
    if dd_active:
      reason_parts.append("drawdown_breaker_active")
    if stress not in ("low_stress",):
      reason_parts.append(f"stress_{stress}")
    rows.append({
      "ticker": t, "signal": sig, "target_weight": round(tw, 4), "previous_weight": round(pw, 4),
      "change": round(chg, 4), "engine_source": eng, "regime": regime, "stress_level": stress,
      "confidence": round(info.get("confidence", 70), 1),
      "reason": " | ".join(reason_parts),
      "entry_plan": "proxima apertura post-viernes" if sig in ("BUY", "INCREASE") else "-",
      "exit_plan": "rebalance semanal o breaker/stress" if sig in ("SELL", "REDUCE") else "mantener hasta viernes",
      "next_review": "proximo viernes", "cash_account_executable": True,
    })
  return pd.DataFrame(rows)


champ_name = champion.get("strategy", "V16_D_M7_ALL_CONTROLS") if len(champion) else "V16_D_M7_ALL_CONTROLS"
champ_sched = schedules.get(champ_name, {})
meta_champ = metas.get(champ_name, {})
wh_champ = weights_hist.get(champ_name, pd.DataFrame())
current_signals = generate_v16_signals(champ_sched, meta_champ, close_prices, champ_name)
current_regime, _ = classify_regime(max(champ_sched.keys()) if champ_sched else rdates[-1])
current_stress = classify_stress(calculate_market_stress_score(close_prices, rdates[-1]))

# %% [markdown]
# ## 13. Exportar y reporte

# %%
def export_csv(df, name):
  (df if df is not None and len(df) else pd.DataFrame()).to_csv(name, index=False)
  print("Exportado", name)


def _baseline_return(name):
  if name in equities and len(equities[name]) >= 2:
    e = equities[name]
    return round((e.iloc[-1] / e.iloc[0] - 1) * 100, 2)
  return np.nan


champ_eq = equities.get(champ_name, pd.Series(dtype=float))
summary = {
  "lab": "v16_drawdown_controlled_meta",
  "v16_score": v16_score, "status": v16_status,
  "approved_for_web_paper": approved, "approved_for_real_money": False,
  "champion_strategy": champ_name, "current_regime": current_regime,
  "current_stress": current_stress, "robustness_score": robustness_score,
  "overfitting_risk": ovr, **(champion.to_dict() if len(champion) else {}),
  "spy_return": round(spy_total, 2),
  "v15_m7_return": _baseline_return("V15_M7_BASELINE"),
  "v14_return": _baseline_return("V14_RETURN_ENGINE"),
  "v6_return": _baseline_return("V6_AGGRESSIVE_ENGINE"),
}

export_csv(pd.DataFrame([summary]), "research_v16_summary.csv")
export_csv(results_df, "research_v16_strategy_results.csv")
export_csv(yearly_df, "research_v16_yearly.csv")
export_csv(robustness_df, "research_v16_robustness.csv")
export_csv(cost_sens_df, "research_v16_cost_sensitivity.csv")
export_csv(contrib_df, "research_v16_contribution_by_asset.csv")
export_csv(current_signals, "research_v16_current_signals.csv")
if len(current_signals):
  current_signals.to_csv("current_signals_v16.csv", index=False)
if len(champ_eq):
  champ_eq.to_frame("equity").to_csv("research_v16_equity_curve.csv")
if len(wh_champ):
  wh_champ.to_csv("research_v16_weights_history.csv", index=False)

config = {
  "version": "v16_drawdown_controlled_meta",
  "approved_for_web_paper": approved, "approved_for_real_money": False,
  "v16_score": v16_score, "status": v16_status, "champion": champ_name,
  "controls": ["volatility_managed", "drawdown_breaker", "stress_filter", "concentration_cap"],
  "drawdown_limits": {"soft": DRAWDOWN_SOFT_LIMIT, "hard": DRAWDOWN_HARD_LIMIT},
  "weight_caps": {"etf": MAX_WEIGHT_ETF, "stock": MAX_WEIGHT_STOCK, "special": SPECIAL_CAPS},
}
Path("research_v16_selected_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
print("Exportado research_v16_selected_config.json")

print("=" * 80)
print("REPORTE FINAL V16 DRAWDOWN CONTROLLED META LAB")
print("=" * 80)
print(f"Estado: {v16_status} | Score: {v16_score}/100")
print(f"Mejor estrategia: {champ_name}")
print(f"Regimen: {current_regime} | Stress: {current_stress}")
if len(champion):
  print(f"CAGR {champion.get('CAGR')}% | Sharpe {champion.get('sharpe')} | Sortino {champion.get('sortino')}")
  print(f"DD {champion.get('max_drawdown')}% | Win SPY {champion.get('win_years_vs_spy', 0)*100:.0f}% | Win QQQ {champion.get('win_years_vs_qqq', 0)*100:.0f}%")
  print(f"Concentration: {champion.get('concentration_risk')} | Top contrib {champion.get('top_contributor_pct')}%")
  print(f"Recent years: {champion.get('recent_years_label')} ({champion.get('recent_years_score')})")
print(f"SPY: {spy_total:.1f}% | V15 M7: {_baseline_return('V15_M7_BASELINE')} | V14: {_baseline_return('V14_RETURN_ENGINE')}")
if len(current_signals):
  act = current_signals[current_signals["target_weight"] > 0.001]
  print(f"BUY: {len(act[act['signal']=='BUY'])} | REDUCE: {len(act[act['signal']=='REDUCE'])}")
  print(act[["ticker", "signal", "target_weight", "engine_source", "stress_level", "reason"]].head(8).to_string(index=False))
print("")
if approved:
  print("Integrar V16 Drawdown Controlled Meta Champion.")
else:
  print("V16 no mejora suficientemente. Mantener V14 como approved y V15 como candidate.")
print("APPROVED_FOR_REAL_MONEY=False (siempre)")
