# %% [markdown]
# # Trading Research V13 Institutional Premia Ensemble Lab
#
# Sleeves institucionales: TSMOM, XSMOM, defensive, carry proxy + ensemble robusto.
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
import math
from itertools import product
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
VOL_TARGETS = [0.10, 0.15, 0.20]
WF_START_YEAR = 2015
MAX_WEIGHT = 0.30
MIN_WEIGHT = 0.05
CASH_ASSET = "SHY"
MIN_HISTORY_DAYS = 252
MARKET = "SPY"
COST_RATE = TRANSACTION_COST + SLIPPAGE

ETF_UNIVERSE = [
  "SPY", "QQQ", "IWM", "DIA",
  "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLC", "XLB", "XLRE",
  "MTUM", "QUAL", "USMV", "VLUE", "SPLV", "SPHB", "SCHD",
  "SHY", "IEF", "TLT", "LQD", "HYG",
  "GLD", "SLV", "DBC", "VNQ",
  "EFA", "EEM",
]

STOCK_UNIVERSE = [
  "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "GOOGL", "META", "AMZN",
  "JPM", "BAC", "XOM", "CVX", "UNH", "LLY", "WMT", "COST",
]

UNIVERSE = ETF_UNIVERSE + STOCK_UNIVERSE
DEFENSIVE_POOL = ["SHY", "IEF", "TLT", "GLD", "USMV", "QUAL", "SCHD", "XLV", "XLP"]
CARRY_POOL = ["HYG", "LQD", "IEF", "TLT", "SCHD", "VNQ", "DBC", "GLD"]

if QUICK_TEST:
  UNIVERSE = ["SPY", "QQQ", "IWM", "XLK", "XLV", "XLF", "TLT", "IEF", "SHY", "GLD", "MTUM", "USMV", "AAPL", "MSFT", "NVDA"]
  START_DATE = "2015-01-01"
  VOL_TARGETS = [0.15]
  print("QUICK_TEST activo")

print("V13 Institutional Premia Ensemble Lab | desde", START_DATE)

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
    except Exception as exc:
      failed.append(f"{ticker}({exc})")
  if failed:
    print("Fallidos/cortos:", failed[:20])
  close = pd.DataFrame({k: v["Close"] for k, v in data.items()}).sort_index().ffill()
  close.index = pd.DatetimeIndex(close.index)
  if close.index.tz:
    close.index = close.index.tz_localize(None)
  return data, close


data_dict, close_prices = download_data(UNIVERSE, START_DATE, END_DATE)
print("Tickers OK:", len(close_prices.columns), "| dias:", len(close_prices))

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


def calculate_volatility(returns, window=63, ann=252):
  return returns.rolling(window).std() * math.sqrt(ann)


def calculate_drawdown(equity):
  peak = equity.cummax()
  return (equity - peak) / peak.replace(0, np.nan)


def calculate_max_drawdown(equity):
  return float(calculate_drawdown(equity).min())


def calculate_sharpe(returns, ann=252):
  r = returns.dropna()
  return float(r.mean() / r.std() * math.sqrt(ann)) if r.std() > 0 else 0.0


def calculate_sortino(returns, ann=252):
  r = returns.dropna()
  ds = r[r < 0].std()
  return float(r.mean() / ds * math.sqrt(ann)) if ds and ds > 0 else 0.0


def calculate_cagr(equity):
  years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
  return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def calculate_calmar(equity):
  mdd = calculate_max_drawdown(equity)
  return calculate_cagr(equity) / abs(mdd) if mdd != 0 else 0.0


def safe_normalize_weights(w):
  w = pd.Series(w).astype(float).clip(lower=0).fillna(0)
  s = w.sum()
  return w / s if s > 0 else w


def cap_and_redistribute_weights(w, cap=MAX_WEIGHT, min_w=MIN_WEIGHT):
  w = safe_normalize_weights(w)
  if w.empty:
    return w
  for _ in range(10):
    over = w[w > cap]
    if over.empty:
      break
    excess = (over - cap).sum()
    w.loc[over.index] = cap
    under = w[w < cap]
    if under.sum() > 0:
      w.loc[under.index] += excess * (under / under.sum())
  w = w.clip(lower=0)
  w[w < min_w] = 0
  return safe_normalize_weights(w)


def inverse_vol_weights(tickers, vols, cap=MAX_WEIGHT):
  inv = pd.Series({t: 1.0 / max(_sf(vols.get(t), 0.15), 0.05) for t in tickers})
  return cap_and_redistribute_weights(inv, cap)


def calculate_turnover(w_new, w_old):
  idx = w_new.index.union(w_old.index)
  return float((w_new.reindex(idx, fill_value=0) - w_old.reindex(idx, fill_value=0)).abs().sum() / 2)


def apply_transaction_costs(equity, turnover):
  return equity * (1 - turnover * COST_RATE)


def get_rebalance_dates(close):
  return close.resample(REBALANCE_FREQ).last().dropna(how="all").index


def build_equity_curve_from_weights(close, data_dict, weight_schedule, initial=INITIAL_CAPITAL):
  """weight_schedule: dict signal_date -> Series weights (Friday close)."""
  dates = close.index
  rebal_exec = {}
  for sig_dt, w in weight_schedule.items():
    future = dates[dates > sig_dt]
    if len(future):
      exec_dt = future[0]
      if "Open" in data_dict.get(w.index[0] if len(w) else MARKET, pd.DataFrame()).columns:
        rebal_exec[exec_dt] = w
      else:
        rebal_exec[exec_dt] = w

  equity = initial
  curve = {}
  current_w = pd.Series(dtype=float)
  turnover_log = []
  for i, dt in enumerate(dates):
    if dt in rebal_exec:
      new_w = rebal_exec[dt]
      to = calculate_turnover(new_w, current_w) if len(current_w) else new_w.abs().sum()
      equity *= (1 - to * COST_RATE)
      turnover_log.append({"date": dt, "turnover": to, "cost": equity * to * COST_RATE})
      current_w = new_w.copy()
    if i > 0 and len(current_w):
      prev = dates[i - 1]
      day_ret = 0.0
      for t, w in current_w.items():
        if t in close.columns:
          r = close[t].loc[dt] / close[t].loc[prev] - 1
          if np.isfinite(r):
            day_ret += w * r
      equity *= (1 + day_ret)
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
    mix6040 = 0.6 * spy_r + 0.4 * (0 if "TLT" not in close.columns else (
      close["TLT"].loc[f"{year}-01-01":f"{year}-12-31"].iloc[-1] /
      close["TLT"].loc[f"{year}-01-01":f"{year}-12-31"].iloc[0] - 1
    ) if len(close["TLT"].loc[f"{year}-01-01":f"{year}-12-31"]) >= 2 else 0)
    rows.append({
      "year": year, "return": round(sr * 100, 2),
      "SPY": round(spy_r * 100, 2), "QQQ": round(qqq_r * 100, 2),
      "6040": round(mix6040 * 100, 2),
      "beats_spy": sr > spy_r, "beats_qqq": sr > qqq_r, "beats_6040": sr > mix6040,
    })
  return pd.DataFrame(rows)


def compare_to_benchmarks(equity, close):
  rets = equity.pct_change().fillna(0)
  m = {
    "total_return": round((equity.iloc[-1] / equity.iloc[0] - 1) * 100, 2),
    "CAGR": round(calculate_cagr(equity) * 100, 2),
    "sharpe": round(calculate_sharpe(rets), 3),
    "sortino": round(calculate_sortino(rets), 3),
    "max_drawdown": round(calculate_max_drawdown(equity) * 100, 2),
    "calmar": round(calculate_calmar(equity), 3),
    "annual_volatility": round(rets.std() * math.sqrt(252) * 100, 2),
  }
  yearly = calculate_yearly_results(equity, close)
  m["win_years_vs_spy"] = yearly["beats_spy"].mean() if len(yearly) else 0
  m["win_years_vs_qqq"] = yearly["beats_qqq"].mean() if len(yearly) else 0
  m["win_years_vs_6040"] = yearly["beats_6040"].mean() if len(yearly) else 0
  return m, yearly


def load_v6_equity():
  for p in [Path("research_outputs/v6/research_v6_equity_curves.csv"), Path("research_v6_equity_curves.csv")]:
    if p.exists():
      try:
        return pd.read_csv(p, index_col=0, parse_dates=True)
      except Exception:
        pass
  return None

# %% [markdown]
# ## 4. Features premia

# %%
def calculate_premia_features(close):
  rets = calculate_returns(close)
  feats = {}
  for t in close.columns:
    c = close[t]
    f = pd.DataFrame(index=c.index)
    for w in [21, 63, 126, 252]:
      f[f"MOM_{w}"] = c / c.shift(w) - 1
    f["MOM_252_SKIP_21"] = c.shift(21) / c.shift(252) - 1
    for w in [50, 100, 200]:
      f[f"SMA_{w}"] = c.rolling(w).mean()
    f["ABOVE_SMA_50"] = (c > f["SMA_50"]).astype(float)
    f["ABOVE_SMA_100"] = (c > f["SMA_100"]).astype(float)
    f["ABOVE_SMA_200"] = (c > f["SMA_200"]).astype(float)
    f["TREND_SCORE"] = f["ABOVE_SMA_50"] * 0.3 + f["ABOVE_SMA_100"] * 0.3 + f["ABOVE_SMA_200"] * 0.4
    for w in [21, 63, 126]:
      f[f"VOL_{w}"] = rets[t].rolling(w).std() * math.sqrt(252)
    f["LOW_VOL_SCORE"] = 1.0 / f["VOL_63"].replace(0, np.nan)
    for w in [63, 126, 252]:
      f[f"DD_{w}"] = c / c.rolling(w).max() - 1
    feats[t] = f

  dates = close.index
  for col in ["MOM_63", "MOM_126", "MOM_252_SKIP_21", "VOL_63", "TREND_SCORE"]:
    wide = pd.DataFrame({t: feats[t][col] for t in feats}, index=dates)
    asc = col == "VOL_63"
    rk = wide.rank(axis=1, pct=True, ascending=asc)
    suffix = col.lower().replace("mom_", "mom_").replace("vol_63", "low_vol").replace("trend_score", "trend")
    name = f"rank_{suffix}" if "rank" not in suffix else suffix
    map_name = {
      "mom_63": "rank_mom_63", "mom_126": "rank_mom_126",
      "mom_252_skip_21": "rank_mom_252_skip_21", "low_vol": "rank_low_vol", "trend": "rank_trend",
    }
    cname = map_name.get(suffix, f"rank_{col}")
    for t in feats:
      feats[t][cname] = rk[t]

  mkt = pd.DataFrame(index=dates)
  if MARKET in feats:
    mkt["SPY_ABOVE_SMA200"] = feats[MARKET]["ABOVE_SMA_200"]
  if "QQQ" in feats:
    mkt["QQQ_ABOVE_SMA200"] = feats["QQQ"]["ABOVE_SMA_200"]
  else:
    mkt["QQQ_ABOVE_SMA200"] = mkt.get("SPY_ABOVE_SMA200", 0)
  if "HYG" in feats and "LQD" in feats:
    mkt["CREDIT_RISK"] = feats["HYG"]["MOM_63"] - feats["LQD"]["MOM_63"]
  else:
    mkt["CREDIT_RISK"] = 0
  if "TLT" in feats:
    mkt["BOND_TREND"] = feats["TLT"]["ABOVE_SMA_200"]
  if "GLD" in feats:
    mkt["GOLD_TREND"] = feats["GLD"]["ABOVE_SMA_200"]
  mkt["MARKET_RISK_ON"] = ((mkt["SPY_ABOVE_SMA200"] >= 0.5) & (mkt["QQQ_ABOVE_SMA200"] >= 0.5)).astype(float)
  mkt["MARKET_REGIME_SCORE"] = mkt["SPY_ABOVE_SMA200"] * 0.5 + mkt["QQQ_ABOVE_SMA200"] * 0.3
  if "BOND_TREND" in mkt:
    mkt["MARKET_REGIME_SCORE"] += mkt["BOND_TREND"] * 0.1
  if "GOLD_TREND" in mkt:
    mkt["MARKET_REGIME_SCORE"] += mkt["GOLD_TREND"] * 0.1
  for t in feats:
    for c in mkt.columns:
      feats[t][c] = mkt[c]
  return feats


features = calculate_premia_features(close_prices)
print("Premia features OK")

# %% [markdown]
# ## 5-8. Sleeves A-D

# %%
def _row(feats, t, dt):
  if t not in feats or dt not in feats[t].index:
    return None
  return feats[t].loc[dt]


def _cash_fill(w, feats, dt, frac=1.0):
  cash = CASH_ASSET if CASH_ASSET in close_prices.columns else None
  if cash:
    w = w.copy()
    rem = max(0, frac - w.sum())
    w[cash] = w.get(cash, 0) + rem
  return cap_and_redistribute_weights(w)


def strategy_time_series_momentum(close, feats, rebalance_dates, params):
  top_n = params.get("top_n", 5)
  vol_target = params.get("vol_target", 0.15)
  lookback = params.get("lookback", "126")
  schedule = {}
  for dt in rebalance_dates:
    if dt not in close.index:
      continue
    risk_on = _sf(_row(feats, MARKET, dt).get("MARKET_RISK_ON"), 1) >= 0.5 if _row(feats, MARKET, dt) is not None else True
    scores = {}
    vols = {}
    for t in close.columns:
      r = _row(feats, t, dt)
      if r is None:
        continue
      mom_col = {"63": "MOM_63", "126": "MOM_126", "252_SKIP_21": "MOM_252_SKIP_21"}.get(lookback, "MOM_126")
      mom = _sf(r.get(mom_col), -1)
      if mom <= 0 or _sf(r.get("ABOVE_SMA_200"), 0) < 0.5:
        continue
      if _sf(r.get("VOL_63"), 0.5) > 0.45:
        continue
      scores[t] = mom * 0.4 + _sf(r.get("TREND_SCORE"), 0) * 0.6
      vols[t] = _sf(r.get("VOL_63"), 0.15)
    if not risk_on:
      pool = [t for t in DEFENSIVE_POOL if t in close.columns]
      scores = {t: _sf(_row(feats, t, dt).get("TREND_SCORE"), 0) for t in pool if _row(feats, t, dt) is not None}
      vols = {t: _sf(_row(feats, t, dt).get("VOL_63"), 0.1) for t in scores}
    top = pd.Series(scores).sort_values(ascending=False).head(top_n)
    if top.empty:
      schedule[dt] = _cash_fill(pd.Series({CASH_ASSET: 1.0}), feats, dt)
      continue
    w = inverse_vol_weights(top.index, vols)
    scale = min(1.0, vol_target / np.mean([vols[t] for t in top.index]))
    w = w * (0.9 * scale if risk_on else 0.55)
    schedule[dt] = _cash_fill(w, feats, dt)
  return schedule


def strategy_cross_sectional_momentum(close, feats, rebalance_dates, params):
  top_n = params.get("top_n", 5)
  vol_target = params.get("vol_target", 0.15)
  combo = params.get("mom_combo", "126_252")
  schedule = {}
  for dt in rebalance_dates:
    scores, vols = {}, {}
    for t in close.columns:
      r = _row(feats, t, dt)
      if r is None or _sf(r.get("ABOVE_SMA_200"), 0) < 0.5:
        continue
      if _sf(r.get("VOL_63"), 0.5) > 0.50:
        continue
      parts = []
      if "63" in combo:
        parts.append(_sf(r.get("rank_mom_63"), 0.5))
      if "126" in combo:
        parts.append(_sf(r.get("rank_mom_126"), 0.5))
      if "252" in combo:
        parts.append(_sf(r.get("rank_mom_252_skip_21"), 0.5))
      scores[t] = np.mean(parts) if parts else 0
      vols[t] = _sf(r.get("VOL_63"), 0.15)
    top = pd.Series(scores).sort_values(ascending=False).head(top_n)
    if top.empty:
      schedule[dt] = _cash_fill(pd.Series({CASH_ASSET: 1.0}), feats, dt)
      continue
    w = inverse_vol_weights(top.index, vols, cap=min(MAX_WEIGHT, 0.25))
    scale = min(1.0, vol_target / np.mean([vols[t] for t in top.index]))
    schedule[dt] = _cash_fill(w * scale, feats, dt)
  return schedule


def strategy_defensive_low_vol(close, feats, rebalance_dates, params):
  vol_target = params.get("vol_target", 0.15)
  schedule = {}
  for dt in rebalance_dates:
    risk_on = _sf(_row(feats, MARKET, dt).get("MARKET_RISK_ON"), 1) >= 0.5 if _row(feats, MARKET, dt) is not None else True
    pool = DEFENSIVE_POOL if not risk_on else [t for t in close.columns if t in DEFENSIVE_POOL + ["MTUM", "QUAL", "USMV", "SCHD", "XLV", "XLP", "XLK", "QQQ"]]
    pool = [t for t in pool if t in close.columns]
    scores, vols = {}, {}
    for t in pool:
      r = _row(feats, t, dt)
      if r is None:
        continue
      scores[t] = _sf(r.get("rank_low_vol"), 0.5) * 0.4 + _sf(r.get("TREND_SCORE"), 0) * 0.4 + (1 + _sf(r.get("DD_63"), 0)) * 0.2
      vols[t] = _sf(r.get("VOL_63"), 0.12)
    top = pd.Series(scores).sort_values(ascending=False).head(params.get("top_n", 5))
    if top.empty:
      schedule[dt] = _cash_fill(pd.Series({CASH_ASSET: 1.0}), feats, dt)
      continue
    w = inverse_vol_weights(top.index, vols)
    exp = 0.85 if risk_on else 0.60
    schedule[dt] = _cash_fill(w * exp * min(1, vol_target / 0.15), feats, dt)
  return schedule


def strategy_carry_proxy(close, feats, rebalance_dates, params):
  schedule = {}
  cap = min(0.20, MAX_WEIGHT)
  for dt in rebalance_dates:
    scores, vols = {}, {}
    pool = [t for t in CARRY_POOL if t in close.columns]
    if "HYG" in close.columns and "LQD" in close.columns:
      cr = _sf(_row(feats, "HYG", dt).get("MOM_63"), 0) - _sf(_row(feats, "LQD", dt).get("MOM_63"), 0)
    else:
      cr = 0
    for t in pool:
      r = _row(feats, t, dt)
      if r is None:
        continue
      mom = _sf(r.get("MOM_63"), 0)
      trend = _sf(r.get("TREND_SCORE"), 0)
      if mom <= 0 and trend < 0.4:
        continue
      bonus = 0.1 if t in ("HYG", "LQD", "SCHD", "VNQ") and cr > 0 else 0
      scores[t] = mom * 0.5 + trend * 0.5 + bonus
      vols[t] = _sf(r.get("VOL_63"), 0.12)
    top = pd.Series(scores).sort_values(ascending=False).head(4)
    if top.empty:
      schedule[dt] = _cash_fill(pd.Series({CASH_ASSET: 1.0}), feats, dt)
      continue
    w = inverse_vol_weights(top.index, vols, cap=cap)
    schedule[dt] = _cash_fill(w * 0.75, feats, dt)
  return schedule

# %% [markdown]
# ## 9-10. Backtest por sleeve

# %%
SLEEVE_FUNCS = {
  "A_tsmom": strategy_time_series_momentum,
  "B_xsmom": strategy_cross_sectional_momentum,
  "C_defensive": strategy_defensive_low_vol,
  "D_carry": strategy_carry_proxy,
}


def run_sleeve_backtest(name, func, close, feats, data_dict, params):
  rdates = [d for d in get_rebalance_dates(close) if d.year >= WF_START_YEAR]
  sched = func(close, feats, rdates, params)
  eq, to = build_equity_curve_from_weights(close, data_dict, sched)
  if len(eq) < 2:
    return {}, pd.Series(dtype=float), pd.DataFrame(), sched
  metrics, yearly = compare_to_benchmarks(eq, close)
  metrics["turnover"] = round(to["turnover"].mean(), 4) if len(to) else 0
  metrics["total_cost"] = round(to["cost"].sum(), 2) if len(to) else 0
  metrics["num_rebalances"] = len(sched)
  metrics["sleeve"] = name
  metrics["params"] = json.dumps(params)
  yearly["sleeve"] = name
  return metrics, eq, yearly, sched


def run_sleeve_param_grid(quick=QUICK_TEST):
  top_ns = [3, 5] if quick else [3, 5, 8]
  vols = VOL_TARGETS
  rows, yearly_all, schedules = [], [], {}
  grids = {
    "A_tsmom": [{"top_n": n, "vol_target": v, "lookback": lb} for n, v, lb in product(top_ns, vols, ["63", "126", "252_SKIP_21"][:2 if quick else 3])],
    "B_xsmom": [{"top_n": n, "vol_target": v, "mom_combo": c} for n, v, c in product(top_ns, vols, ["63_126", "126_252", "63_126_252"][:2 if quick else 3])],
    "C_defensive": [{"top_n": n, "vol_target": v} for n, v in product(top_ns, vols)],
    "D_carry": [{"vol_target": v} for v in vols],
  }
  for sleeve, grid in grids.items():
    func = SLEEVE_FUNCS[sleeve]
    for params in tqdm(grid, desc=sleeve):
      m, eq, yr, sched = run_sleeve_backtest(sleeve, func, close_prices, features, data_dict, params)
      if m:
        rows.append(m)
        yr = yr.copy()
        yr["config"] = json.dumps(params)
        yearly_all.append(yr)
        schedules[f"{sleeve}|{json.dumps(params)}"] = sched
  return pd.DataFrame(rows), pd.concat(yearly_all, ignore_index=True) if yearly_all else pd.DataFrame(), schedules

# %% [markdown]
# ## 11. Grid y robustez

# %%
sleeve_results, yearly_by_sleeve, sleeve_schedules = run_sleeve_param_grid()
print("Sleeve configs:", len(sleeve_results))
if len(sleeve_results):
  print(sleeve_results.groupby("sleeve")["sharpe"].agg(["mean", "max", "median"]))

robustness_rows = []
if len(sleeve_results):
  for sleeve, grp in sleeve_results.groupby("sleeve"):
    robustness_rows.append({
      "sleeve": sleeve, "n_configs": len(grp),
      "median_sharpe": grp["sharpe"].median(),
      "best_sharpe": grp["sharpe"].max(),
      "sharpe_gap": grp["sharpe"].max() - grp["sharpe"].median(),
      "median_cagr": grp["CAGR"].median(),
      "pct_positive_cagr": (grp["CAGR"] > 0).mean(),
      "neighbor_robust": (grp["sharpe"] > grp["sharpe"].median() * 0.85).mean(),
    })
robustness_df = pd.DataFrame(robustness_rows)

# %% [markdown]
# ## 12. Ensemble

# %%
def combine_weight_schedules(schedules, weights_map, close):
  """weights_map: sleeve_name -> weight fraction."""
  all_dates = sorted(set().union(*[set(s.keys()) for s in schedules.values()]))
  combined = {}
  for dt in all_dates:
    w = pd.Series(dtype=float)
    for sleeve, frac in weights_map.items():
      if sleeve not in schedules or dt not in schedules[sleeve]:
        continue
      sw = schedules[sleeve][dt] * frac
      w = w.add(sw, fill_value=0)
    combined[dt] = cap_and_redistribute_weights(w)
  return combined


def get_best_schedule_per_sleeve(sleeve_results, sleeve_schedules):
  best = {}
  for sleeve in SLEEVE_FUNCS:
    sub = sleeve_results[sleeve_results["sleeve"] == sleeve]
    if len(sub) == 0:
      continue
    row = sub.sort_values(["sharpe", "CAGR"], ascending=False).iloc[0]
    key = f"{sleeve}|{row['params']}"
    if key in sleeve_schedules:
      best[sleeve] = sleeve_schedules[key]
  return best


best_schedules = get_best_schedule_per_sleeve(sleeve_results, sleeve_schedules) if len(sleeve_results) else {}

ENSEMBLE_SPECS = {
  "E1_balanced": {"A_tsmom": 0.40, "B_xsmom": 0.30, "C_defensive": 0.20, "D_carry": 0.10},
  "E2_tsmom_def": {"A_tsmom": 0.50, "C_defensive": 0.25, "B_xsmom": 0.25},
  "E3_equal_mom": {"A_tsmom": 0.33, "B_xsmom": 0.33, "C_defensive": 0.34},
}


def strategy_ensemble_adaptive(close, feats, rebalance_dates, base_schedules):
  schedule = {}
  for dt in rebalance_dates:
    risk_on = _sf(_row(feats, MARKET, dt).get("MARKET_RISK_ON"), 1) >= 0.5 if _row(feats, MARKET, dt) is not None else True
    if risk_on:
      wmap = {"A_tsmom": 0.45, "B_xsmom": 0.35, "C_defensive": 0.15, "D_carry": 0.05}
    else:
      wmap = {"A_tsmom": 0.10, "B_xsmom": 0.10, "C_defensive": 0.55, "D_carry": 0.25}
    w = pd.Series(dtype=float)
    for sleeve, frac in wmap.items():
      if sleeve in base_schedules and dt in base_schedules[sleeve]:
        w = w.add(base_schedules[sleeve][dt] * frac, fill_value=0)
    schedule[dt] = cap_and_redistribute_weights(w)
  return schedule


ensemble_results = []
ensemble_equities = {}
ensemble_schedules = {}
if best_schedules:
  for ename, wmap in ENSEMBLE_SPECS.items():
    sched = combine_weight_schedules({k: best_schedules[k] for k in wmap if k in best_schedules}, wmap, close_prices)
    eq, to = build_equity_curve_from_weights(close_prices, data_dict, sched)
    if len(eq) >= 2:
      m, yr = compare_to_benchmarks(eq, close_prices)
      m["ensemble"] = ename
      m["turnover"] = round(to["turnover"].mean(), 4) if len(to) else 0
      m["total_cost"] = round(to["cost"].sum(), 2) if len(to) else 0
      ensemble_results.append(m)
      ensemble_equities[ename] = eq
      ensemble_schedules[ename] = sched
  e4_dates = list(next(iter(best_schedules.values())).keys()) if best_schedules else []
  e4_sched = strategy_ensemble_adaptive(close_prices, features, e4_dates, best_schedules)
  if e4_sched:
    eq4, to4 = build_equity_curve_from_weights(close_prices, data_dict, e4_sched)
    if len(eq4) >= 2:
      m4, _ = compare_to_benchmarks(eq4, close_prices)
      m4["ensemble"] = "E4_adaptive"
      m4["turnover"] = round(to4["turnover"].mean(), 4) if len(to4) else 0
      m4["total_cost"] = round(to4["cost"].sum(), 2) if len(to4) else 0
      ensemble_results.append(m4)
      ensemble_equities["E4_adaptive"] = eq4
      ensemble_schedules["E4_adaptive"] = e4_sched

ensemble_df = pd.DataFrame(ensemble_results)
print("Ensembles probados:", len(ensemble_df))

# %% [markdown]
# ## 13. Overfitting control

# %%
def overfitting_report(sleeve_df, ensemble_df):
  n_trials = len(sleeve_df) + len(ensemble_df)
  all_sharpe = pd.concat([sleeve_df["sharpe"], ensemble_df["sharpe"]]) if len(sleeve_df) and len(ensemble_df) else (sleeve_df["sharpe"] if len(sleeve_df) else ensemble_df["sharpe"])
  best_sh = all_sharpe.max() if len(all_sharpe) else 0
  med_sh = all_sharpe.median() if len(all_sharpe) else 0
  gap = best_sh - med_sh
  n_years = max(len(calculate_yearly_results(close_prices[MARKET].dropna(), close_prices)), 5)
  deflated_proxy = best_sh - math.sqrt(2 * math.log(max(n_trials, 2)) / max(n_years, 1))
  robustness_score = int(np.clip(100 - gap * 30 - max(0, n_trials - 30) * 0.5, 0, 100))
  risk = "HIGH" if gap > 0.8 or deflated_proxy < 0.3 else ("MEDIUM" if gap > 0.4 else "LOW")
  param_sens = 100 - min(100, gap * 40)
  return pd.DataFrame([{
    "number_of_trials": n_trials,
    "best_sharpe": round(best_sh, 3),
    "median_sharpe": round(med_sh, 3),
    "sharpe_gap_vs_median": round(gap, 3),
    "deflated_sharpe_proxy": round(deflated_proxy, 3),
    "robustness_score": robustness_score,
    "parameter_sensitivity_score": round(param_sens, 1),
    "overfitting_risk": risk,
  }])


overfit_df = overfitting_report(sleeve_results, ensemble_df)
print(overfit_df.to_string(index=False))

# %% [markdown]
# ## 14. Seleccion champion V13

# %%
def select_v13_champion(sleeve_df, ensemble_df, overfit_df):
  cands = []
  if len(sleeve_df):
    for _, r in sleeve_df.iterrows():
      cands.append({**r.to_dict(), "type": "sleeve"})
  if len(ensemble_df):
    for _, r in ensemble_df.iterrows():
      cands.append({**r.to_dict(), "type": "ensemble"})
  if not cands:
    return None, "REJECTED", 0, []
  cdf = pd.DataFrame(cands)
  spy_cagr = calculate_cagr(close_prices[MARKET].dropna()) * 100 if MARKET in close_prices else 0
  rob = overfit_df["robustness_score"].iloc[0] if len(overfit_df) else 50
  ovr = overfit_df["overfitting_risk"].iloc[0] if len(overfit_df) else "MEDIUM"
  cdf["v13_score"] = 0
  cdf.loc[cdf["sharpe"] > 1, "v13_score"] += 20
  cdf.loc[cdf["CAGR"] > spy_cagr, "v13_score"] += 20
  cdf.loc[cdf["max_drawdown"] > -25, "v13_score"] += 15
  cdf.loc[cdf["win_years_vs_spy"] >= 0.6, "v13_score"] += 15
  cdf.loc[cdf["win_years_vs_qqq"] >= 0.4, "v13_score"] += 10
  cdf.loc[cdf["total_cost"] < INITIAL_CAPITAL * 0.08, "v13_score"] += 10
  cdf.loc[cdf["type"] == "ensemble", "v13_score"] += 5
  if ovr == "HIGH":
    cdf["v13_score"] -= 25
  elif ovr == "MEDIUM":
    cdf["v13_score"] -= 10
  score = int(np.clip(cdf.loc[cdf["v13_score"].idxmax(), "v13_score"] + rob * 0.1, 0, 100))
  best = cdf.sort_values("v13_score", ascending=False).iloc[0]
  notes = []
  approved = (
    best.get("sharpe", 0) > 1 and best.get("max_drawdown", -100) > -25
    and best.get("win_years_vs_spy", 0) >= 0.6 and ovr != "HIGH" and rob >= 70
  )
  candidate = best.get("sharpe", 0) > 0.8 and best.get("max_drawdown", -100) > -30
  if best.get("win_years_vs_spy", 0) < 0.6:
    notes.append("pocos años vs SPY")
  status = "APPROVED_FOR_WEB_PAPER" if approved and score >= 75 else ("CANDIDATE" if candidate and score >= 60 else "REJECTED")
  return best.to_dict(), status, score, notes


champion, v13_status, v13_score, score_notes = select_v13_champion(sleeve_results, ensemble_df, overfit_df)
APPROVED_FOR_WEB_PAPER = v13_status == "APPROVED_FOR_WEB_PAPER"
print(f"V13: {v13_status} | score {v13_score}")

# Equity champion
champion_eq = pd.Series(dtype=float)
champion_sched = {}
if champion:
  if champion.get("type") == "ensemble" and champion.get("ensemble") in ensemble_equities:
    champion_eq = ensemble_equities[champion["ensemble"]]
    champion_sched = ensemble_schedules.get(champion["ensemble"], {})
  elif champion.get("type") == "sleeve":
    key = f"{champion['sleeve']}|{champion['params']}"
    if key in sleeve_schedules:
      champion_sched = sleeve_schedules[key]
      champion_eq, _ = build_equity_curve_from_weights(close_prices, data_dict, champion_sched)
yearly_df = calculate_yearly_results(champion_eq, close_prices) if len(champion_eq) > 1 else pd.DataFrame()

# V6 compare
v6_eq = load_v6_equity()
v6_total = np.nan
if v6_eq is not None and "blended_champion_weights_alpha_0.5" in v6_eq.columns:
  s = v6_eq["blended_champion_weights_alpha_0.5"].dropna()
  if len(s) >= 2:
    v6_total = (s.iloc[-1] / s.iloc[0] - 1) * 100

# %% [markdown]
# ## 15. Señales actuales

# %%
def generate_v13_signals(sched, feats, champion_info):
  if not sched:
    return pd.DataFrame()
  last_sig = max(sched.keys())
  target = sched[last_sig]
  prev_keys = sorted([d for d in sched.keys() if d < last_sig])
  prev = sched[prev_keys[-1]] if prev_keys else pd.Series(dtype=float)
  rows = []
  tickers = sorted(set(target.index) | set(prev.index))
  for t in tickers:
    tw, pw = _sf(target.get(t), 0), _sf(prev.get(t), 0)
    chg = tw - pw
    sc = 50.0
    r = _row(feats, t, last_sig)
    if r is not None:
      sc = (_sf(r.get("rank_mom_126"), 0.5) + _sf(r.get("TREND_SCORE"), 0.5)) * 50
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
    src = champion_info.get("ensemble") or champion_info.get("sleeve", "V13")
    rows.append({
      "ticker": t, "signal": sig, "target_weight": round(tw, 4), "previous_weight": round(pw, 4),
      "change": round(chg, 4), "score": round(sc, 1), "sleeve_source": src,
      "reason": f"premia ensemble ranking | {sig.lower()}",
      "entry_plan": "proxima apertura tras cierre viernes" if sig in ("BUY", "INCREASE") else "-",
      "exit_plan": "rebalance semanal o reducir si pierde ranking" if sig in ("SELL", "REDUCE") else "mantener hasta proximo viernes",
      "next_review": "proximo viernes",
      "cash_account_executable": True,
    })
  return pd.DataFrame(rows)


signal_sched = champion_sched
if not signal_sched and best_schedules:
  signal_sched = next(iter(best_schedules.values()))
current_signals = generate_v13_signals(signal_sched, features, champion or {})
print("Señales:", len(current_signals))

# %% [markdown]
# ## 16. Exportar

# %%
def export_csv(df, name):
  (df if len(df) else pd.DataFrame()).to_csv(name, index=False)
  print("Exportado", name)


spy_total = (close_prices[MARKET].iloc[-1] / close_prices[MARKET].iloc[0] - 1) * 100
champ_ret = (champion_eq.iloc[-1] / champion_eq.iloc[0] - 1) * 100 if len(champion_eq) > 1 else 0

summary = {
  "lab": "v13_institutional_premia",
  "v13_score": v13_score,
  "status": v13_status,
  "approved_for_web_paper": APPROVED_FOR_WEB_PAPER,
  "approved_for_real_money": False,
  "champion_type": champion.get("type") if champion else "",
  "champion_name": champion.get("ensemble") or champion.get("sleeve") if champion else "",
  "champion_total_return": round(champ_ret, 2),
  "champion_sharpe": champion.get("sharpe") if champion else 0,
  "champion_cagr": champion.get("CAGR") if champion else 0,
  "champion_max_drawdown": champion.get("max_drawdown") if champion else 0,
  "spy_return": round(spy_total, 2),
  "v6_return": round(v6_total, 2) if pd.notna(v6_total) else np.nan,
  "overfitting_risk": overfit_df["overfitting_risk"].iloc[0] if len(overfit_df) else "",
  "robustness_score": overfit_df["robustness_score"].iloc[0] if len(overfit_df) else 0,
}

export_csv(pd.DataFrame([summary]), "research_v13_summary.csv")
export_csv(sleeve_results, "research_v13_sleeve_results.csv")
export_csv(ensemble_df, "research_v13_ensemble_results.csv")
export_csv(yearly_df, "research_v13_yearly.csv")
export_csv(yearly_by_sleeve, "research_v13_yearly_by_sleeve.csv")
export_csv(robustness_df, "research_v13_robustness.csv")
export_csv(overfit_df, "research_v13_overfitting_report.csv")
export_csv(current_signals, "research_v13_current_signals.csv")
if len(champion_eq):
  champion_eq.to_frame("equity").to_csv("research_v13_equity_curve.csv")
else:
  pd.DataFrame().to_csv("research_v13_equity_curve.csv", index=False)

config = {
  "version": "v13_institutional_premia",
  "approved_for_web_paper": APPROVED_FOR_WEB_PAPER,
  "approved_for_real_money": False,
  "v13_score": v13_score,
  "status": v13_status,
  "champion": champion,
  "ensemble_specs": ENSEMBLE_SPECS,
  "warnings": ["Backtest no garantiza resultados futuros.", "Carry sleeve es proxy, no carry puro.", "No conecta broker."],
}
Path("research_v13_selected_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
print("Exportado research_v13_selected_config.json")

# %% [markdown]
# ## 17. Reporte final

# %%
print("=" * 80)
print("REPORTE FINAL V13 INSTITUTIONAL PREMIA ENSEMBLE LAB")
print("=" * 80)
if champion:
  print(f"Champion: {champion.get('type')} | {champion.get('ensemble') or champion.get('sleeve')}")
  print(f"CAGR {champion.get('CAGR')}% | Sharpe {champion.get('sharpe')} | DD {champion.get('max_drawdown')}%")
if len(sleeve_results):
  best_sleeve = sleeve_results.sort_values("sharpe", ascending=False).iloc[0]
  print(f"Mejor sleeve: {best_sleeve['sleeve']} Sharpe {best_sleeve['sharpe']}")
if len(ensemble_df):
  best_ens = ensemble_df.sort_values("sharpe", ascending=False).iloc[0]
  print(f"Mejor ensemble: {best_ens['ensemble']} Sharpe {best_ens['sharpe']}")
print(f"Score V13: {v13_score}/100 | {v13_status}")
print(f"Win years SPY: {yearly_df['beats_spy'].mean()*100:.0f}%" if len(yearly_df) else "N/A")
print(f"Overfitting: {overfit_df['overfitting_risk'].iloc[0] if len(overfit_df) else 'N/A'} | Robustez: {overfit_df['robustness_score'].iloc[0] if len(overfit_df) else 0}")
if pd.notna(v6_total):
  print(f"¿Supera V6 ({v6_total:.1f}%)? {'SI' if champ_ret > v6_total else 'NO'}")
print(f"BUY: {len(current_signals[current_signals['signal']=='BUY']) if len(current_signals) else 0}")
if score_notes:
  print("Notas:", "; ".join(score_notes))
print("")
if APPROVED_FOR_WEB_PAPER:
  print("Integrar V13 Institutional Premia Ensemble como motor principal de señales semanales.")
else:
  print("V13 rejected. Mantener V6 como champion.")
print("APPROVED_FOR_REAL_MONEY=False (siempre)")
