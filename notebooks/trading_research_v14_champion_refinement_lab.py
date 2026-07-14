# %% [markdown]
# # Trading Research V14 Champion Refinement Lab
#
# Refinamiento de V13: A_tsmom + E4_adaptive + benchmark-aware overlay.
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
MIN_WEIGHT = 0.05
VOL_TARGET = 0.15
MIN_HISTORY_DAYS = 200
MARKET = "SPY"
COST_RATE = TRANSACTION_COST + SLIPPAGE

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
START_DATES_ROBUST = ["2010-01-01", "2015-01-01", "2018-01-01"] if QUICK_TEST else [
  "2010-01-01", "2012-01-01", "2015-01-01", "2018-01-01", "2020-01-01",
]
COST_SENSITIVITY = [0.0005, 0.001, 0.002] if not QUICK_TEST else [0.0005, 0.001]

if QUICK_TEST:
  UNIVERSE = ["SPY", "QQQ", "IWM", "XLK", "XLV", "XLF", "TLT", "IEF", "SHY", "GLD", "MTUM", "USMV", "AAPL", "MSFT", "NVDA"]
  START_DATE = "2015-01-01"
  print("QUICK_TEST activo")

print("V14 Champion Refinement Lab | desde", START_DATE)

# %% [markdown]
# ## 2. Descarga y funciones (V13)

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
  for _ in range(8):
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


def build_equity_curve_from_weights(close, weight_schedule, initial=INITIAL_CAPITAL, cost_rate=COST_RATE):
  dates = close.index
  rebal_exec = {dates[dates > d][0]: w for d, w in weight_schedule.items() if len(dates[dates > d])}
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
      dr = sum(current_w.get(t, 0) * (close[t].loc[dt] / close[t].loc[prev] - 1)
               for t in current_w.index if t in close.columns and np.isfinite(close[t].loc[dt]))
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


def load_v6_equity():
  for p in [Path("research_outputs/v6/research_v6_equity_curves.csv"), Path("research_v6_equity_curves.csv")]:
    if p.exists():
      try:
        return pd.read_csv(p, index_col=0, parse_dates=True)
      except Exception:
        pass
  return None


data_dict, close_prices = download_data(UNIVERSE, START_DATE, END_DATE)
print("Tickers:", len(close_prices.columns))

# %% [markdown]
# ## 3. Features premia (V13)

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
    f["ABOVE_SMA_100"] = (c > f["SMA_100"]).astype(float)
    f["ABOVE_SMA_200"] = (c > f["SMA_200"]).astype(float)
    f["TREND_SCORE"] = f["ABOVE_SMA_100"] * 0.4 + f["ABOVE_SMA_200"] * 0.6
    f["VOL_63"] = rets[t].rolling(63).std() * math.sqrt(252)
    f["DD_63"] = c / c.rolling(63).max() - 1
    feats[t] = f
  dates = close.index
  for col, name in [("MOM_63", "rank_mom_63"), ("MOM_126", "rank_mom_126"), ("VOL_63", "rank_low_vol")]:
    wide = pd.DataFrame({t: feats[t][col] for t in feats}, index=dates)
    rk = wide.rank(axis=1, pct=True, ascending=(col == "VOL_63"))
    for t in feats:
      feats[t][name] = rk[t]
  mkt = pd.DataFrame(index=dates)
  if MARKET in feats:
    mkt["SPY_ABOVE_SMA200"] = feats[MARKET]["ABOVE_SMA_200"]
  mkt["QQQ_ABOVE_SMA200"] = feats["QQQ"]["ABOVE_SMA_200"] if "QQQ" in feats else mkt.get("SPY_ABOVE_SMA200", 0)
  mkt["MARKET_RISK_ON"] = ((mkt["SPY_ABOVE_SMA200"] >= 0.5) & (mkt["QQQ_ABOVE_SMA200"] >= 0.5)).astype(float)
  mkt["MARKET_REGIME_SCORE"] = mkt["SPY_ABOVE_SMA200"] * 0.55 + mkt["QQQ_ABOVE_SMA200"] * 0.45
  for t in feats:
    for c in mkt.columns:
      feats[t][c] = mkt[c]
  return feats


def _row(feats, t, dt):
  return feats[t].loc[dt] if t in feats and dt in feats[t].index else None


def _market_flag(feats, dt, col, default=1.0):
  r = _row(feats, MARKET, dt)
  return _sf(r.get(col), default) if r is not None else default


def _ticker_flag(feats, ticker, dt, col, default=0.0):
  r = _row(feats, ticker, dt)
  return _sf(r.get(col), default) if r is not None else default


def _cash_fill(w, frac=1.0):
  w = w.copy()
  if CASH_ASSET in close_prices.columns:
    w[CASH_ASSET] = w.get(CASH_ASSET, 0) + max(0, frac - w.sum())
  return cap_and_redistribute_weights(w)


features = calculate_premia_features(close_prices)

# %% [markdown]
# ## 4. Bases V13

# %%
def strategy_tsmom(close, feats, rdates, top_n=3, vol_target=0.15, lookback="63"):
  sched = {}
  mom_col = {"63": "MOM_63", "126": "MOM_126"}.get(lookback, "MOM_63")
  for dt in rdates:
    risk_on = _market_flag(feats, dt, "MARKET_RISK_ON", 1) >= 0.5
    scores, vols = {}, {}
    for t in close.columns:
      r = _row(feats, t, dt)
      if r is None:
        continue
      if _sf(r.get(mom_col), -1) <= 0 or _sf(r.get("ABOVE_SMA_200"), 0) < 0.5:
        continue
      if _sf(r.get("VOL_63"), 0.5) > 0.45:
        continue
      scores[t] = _sf(r.get(mom_col)) * 0.5 + _sf(r.get("TREND_SCORE"), 0) * 0.5
      vols[t] = _sf(r.get("VOL_63"), 0.15)
    if not risk_on:
      pool = [t for t in DEFENSIVE_POOL if t in close.columns]
      scores = {t: _sf(_row(feats, t, dt).get("TREND_SCORE"), 0) for t in pool}
      vols = {t: 0.12 for t in scores}
    top = pd.Series(scores).sort_values(ascending=False).head(top_n)
    if top.empty:
      sched[dt] = _cash_fill(pd.Series({CASH_ASSET: 1.0}))
      continue
    w = inverse_vol_weights(top.index, vols) * min(1, vol_target / 0.15) * (0.9 if risk_on else 0.55)
    sched[dt] = _cash_fill(w)
  return sched


def strategy_xsmom(close, feats, rdates, top_n=5):
  sched = {}
  for dt in rdates:
    scores, vols = {}, {}
    for t in close.columns:
      r = _row(feats, t, dt)
      if r is None or _sf(r.get("ABOVE_SMA_200"), 0) < 0.5:
        continue
      scores[t] = np.mean([_sf(r.get("rank_mom_63"), 0.5), _sf(r.get("rank_mom_126"), 0.5)])
      vols[t] = _sf(r.get("VOL_63"), 0.15)
    top = pd.Series(scores).sort_values(ascending=False).head(top_n)
    sched[dt] = _cash_fill(inverse_vol_weights(top.index, vols) * 0.85) if len(top) else _cash_fill(pd.Series({CASH_ASSET: 1.0}))
  return sched


def strategy_defensive(close, feats, rdates, top_n=5):
  sched = {}
  for dt in rdates:
    risk_on = _market_flag(feats, dt, "MARKET_RISK_ON", 1) >= 0.5
    pool = DEFENSIVE_POOL if not risk_on else [t for t in close.columns if t in DEFENSIVE_POOL + ["USMV", "MTUM", "XLV", "XLP"]]
    scores = {t: _sf(_row(feats, t, dt).get("rank_low_vol"), 0.5) + _sf(_row(feats, t, dt).get("TREND_SCORE"), 0)
              for t in pool if _row(feats, t, dt) is not None}
    top = pd.Series(scores).sort_values(ascending=False).head(top_n)
    vols = {t: 0.12 for t in top.index}
    sched[dt] = _cash_fill(inverse_vol_weights(top.index, vols) * (0.85 if risk_on else 0.65)) if len(top) else _cash_fill(pd.Series({CASH_ASSET: 1.0}))
  return sched


def build_e4_adaptive(base_schedules, feats, rdates):
  sched = {}
  for dt in rdates:
    risk_on = _market_flag(feats, dt, "MARKET_RISK_ON", 1) >= 0.5
    wmap = {"A": 0.45, "B": 0.35, "C": 0.15, "D": 0.05} if risk_on else {"A": 0.10, "B": 0.10, "C": 0.55, "D": 0.25}
    w = pd.Series(dtype=float)
    for k, frac in wmap.items():
      if k in base_schedules and dt in base_schedules[k]:
        w = w.add(base_schedules[k][dt] * frac, fill_value=0)
    sched[dt] = cap_and_redistribute_weights(w)
  return sched


rdates = [d for d in get_rebalance_dates(close_prices) if d.year >= WF_START_YEAR]
base_a = strategy_tsmom(close_prices, features, rdates, top_n=3, vol_target=0.15, lookback="63")
base_b = strategy_xsmom(close_prices, features, rdates)
base_c = strategy_defensive(close_prices, features, rdates)
base_d = base_c.copy()
base_schedules = {"A": base_a, "B": base_b, "C": base_c, "D": base_d}
sched_e4 = build_e4_adaptive(base_schedules, features, rdates)
print("Bases V13 reconstruidas | A:", len(base_a), "| E4:", len(sched_e4))

# %% [markdown]
# ## 5. Estrategias refinadas R1-R7

# %%
def blend_schedules(sa, sb, wa, wb=None):
  wb = 1.0 - wa if wb is None else wb
  dates = sorted(set(sa.keys()) | set(sb.keys()))
  out = {}
  for dt in dates:
    w = pd.Series(dtype=float)
    if dt in sa:
      w = w.add(sa[dt] * wa, fill_value=0)
    if dt in sb:
      w = w.add(sb[dt] * wb, fill_value=0)
    out[dt] = cap_and_redistribute_weights(w)
  return out


def fixed_alloc(alloc):
  out = {}
  for dt in rdates:
    w = pd.Series({k: v for k, v in alloc.items() if k in close_prices.columns})
    out[dt] = cap_and_redistribute_weights(w)
  return out


def strategy_r3_core_satellite(feats):
  out = {}
  for dt in rdates:
    risk_on = _market_flag(feats, dt, "MARKET_RISK_ON", 1) >= 0.5
    if risk_on:
      w = pd.Series(dtype=float)
      if dt in base_a:
        w = w.add(base_a[dt] * 0.40, fill_value=0)
      for t, pct in [("QQQ", 0.30), ("SPY", 0.20), (CASH_ASSET, 0.10)]:
        if t in close_prices.columns:
          w[t] = w.get(t, 0) + pct
    else:
      w = pd.Series(dtype=float)
      if dt in sched_e4:
        w = w.add(sched_e4[dt] * 0.40, fill_value=0)
      for t, pct in [(CASH_ASSET, 0.30), ("IEF", 0.10), ("TLT", 0.10), ("GLD", 0.10)]:
        if t in close_prices.columns:
          w[t] = w.get(t, 0) + pct
    out[dt] = cap_and_redistribute_weights(w)
  return out


def rebuild_bases(close, feats):
  rd = [d for d in get_rebalance_dates(close) if d.year >= WF_START_YEAR]
  ba = strategy_tsmom(close, feats, rd, 3, 0.15, "63")
  bb = strategy_xsmom(close, feats, rd)
  bc = strategy_defensive(close, feats, rd)
  be4 = build_e4_adaptive({"A": ba, "B": bb, "C": bc, "D": bc}, feats, rd)
  return ba, be4, rd


def strategy_r4_on(close, feats, rdates, ba, be4):
  out = {}
  for dt in rdates:
    spy_ok = _market_flag(feats, dt, "SPY_ABOVE_SMA200", 0) >= 0.5
    qqq_ok = _ticker_flag(feats, "QQQ", dt, "QQQ_ABOVE_SMA200", 0) >= 0.5 if "QQQ" in feats else spy_ok
    w = pd.Series(dtype=float)
    if spy_ok and qqq_ok:
      if dt in ba:
        w = w.add(ba[dt] * 0.50, fill_value=0)
      for t, pct in [("QQQ", 0.25), ("SPY", 0.15)]:
        if t in close.columns:
          w[t] = w.get(t, 0) + pct
      for sat in ["MTUM", "XLK"]:
        if sat in close.columns and _ticker_flag(feats, sat, dt, "ABOVE_SMA_200", 0) >= 0.5:
          w[sat] = w.get(sat, 0) + 0.05
          break
      w = _cash_fill(w, 1.0)
    elif spy_ok:
      if dt in ba:
        w = w.add(ba[dt] * 0.40, fill_value=0)
      if "SPY" in close.columns:
        w["SPY"] = w.get("SPY", 0) + 0.30
      if dt in be4:
        w = w.add(be4[dt] * 0.20, fill_value=0)
      if CASH_ASSET in close.columns:
        w[CASH_ASSET] = w.get(CASH_ASSET, 0) + 0.10
      w = cap_and_redistribute_weights(w)
    else:
      if dt in be4:
        w = w.add(be4[dt] * 0.50, fill_value=0)
      for t, pct in [(CASH_ASSET, 0.20), ("IEF", 0.15), ("GLD", 0.08), ("TLT", 0.07)]:
        if t in close.columns:
          w[t] = w.get(t, 0) + pct
      w = cap_and_redistribute_weights(w)
    out[dt] = w
  return out


def strategy_r4_benchmark_aware(feats):
  return strategy_r4_on(close_prices, feats, rdates, base_a, sched_e4)


def strategy_r5_drawdown_controlled():
  """Overlay equity SMA100 — solo datos pasados."""
  dates = close_prices.index
  rebal_exec = {dates[dates > d][0]: d for d in rdates if len(dates[dates > d])}
  sig_map = blend_schedules(base_a, sched_e4, 0.70, 0.30)
  equity, curve, current_w, turnover_log = INITIAL_CAPITAL, {}, pd.Series(dtype=float), []
  eq_hist = []
  for i, dt in enumerate(dates):
    if dt in rebal_exec:
      sig_dt = rebal_exec[dt]
      nw = sig_map.get(sig_dt, pd.Series({CASH_ASSET: 1.0})).copy()
      if len(eq_hist) >= 100:
        eq_s = pd.Series(eq_hist[-100:])
        if eq_hist[-1] < eq_s.mean():
          risky = nw.drop([CASH_ASSET, "IEF", "TLT", "GLD"], errors="ignore")
          def_tickers = [t for t in ["SHY", "IEF", "GLD", "TLT"] if t in close_prices.columns]
          def_part = (
            pd.Series({t: 1.0 / len(def_tickers) for t in def_tickers})
            if def_tickers else pd.Series({CASH_ASSET: 1.0})
          )
          nw = cap_and_redistribute_weights(risky * 0.5 + def_part * 0.5)
      to = calculate_turnover(nw, current_w) if len(current_w) else nw.abs().sum()
      equity *= (1 - to * COST_RATE)
      turnover_log.append({"date": dt, "turnover": to, "cost": equity * to * COST_RATE})
      current_w = nw
    if i > 0 and len(current_w):
      prev = dates[i - 1]
      dr = sum(current_w.get(t, 0) * (close_prices[t].loc[dt] / close_prices[t].loc[prev] - 1)
               for t in current_w.index if t in close_prices.columns)
      equity *= (1 + dr)
    curve[dt] = equity
    eq_hist.append(equity)
  return pd.Series(curve), pd.DataFrame(turnover_log)


REFINED_STRATEGIES = {
  "R1_RETURN_ENGINE": lambda: base_a,
  "R2_DEFENSIVE_ENGINE": lambda: sched_e4,
  "R3_CORE_SATELLITE": lambda: strategy_r3_core_satellite(features),
  "R4_BENCHMARK_AWARE": lambda: strategy_r4_benchmark_aware(features),
  "R6_SIMPLE_BLEND": lambda: blend_schedules(base_a, sched_e4, 0.50, 0.50),
}

# %% [markdown]
# ## 6. Backtest

# %%
def run_strategy_backtest(name, schedule, cost_rate=COST_RATE):
  if schedule is None or len(schedule) == 0:
    return {}, pd.Series(dtype=float), pd.DataFrame(), schedule
  eq, to = build_equity_curve_from_weights(close_prices, schedule, cost_rate=cost_rate)
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


strategy_results, equities, yearly_all, schedules = [], {}, [], {}
for name, fn in REFINED_STRATEGIES.items():
  sched = fn()
  m, eq, yr, sched = run_strategy_backtest(name, sched)
  if m:
    strategy_results.append(m)
    equities[name] = eq
    yearly_all.append(yr)
    schedules[name] = sched

m5, eq5, yr5, sched5 = {}, pd.Series(dtype=float), pd.DataFrame(), {}
eq5, to5 = strategy_r5_drawdown_controlled()
if len(eq5) >= 2:
  m5, yr5 = compare_to_benchmarks(eq5, close_prices)
  m5["strategy"] = "R5_DRAWDOWN_CONTROLLED"
  m5["turnover"] = round(to5["turnover"].mean(), 4) if len(to5) else 0
  m5["total_cost"] = round(to5["cost"].sum(), 2) if len(to5) else 0
  m5["num_rebalances"] = len(rdates)
  strategy_results.append(m5)
  equities["R5_DRAWDOWN_CONTROLLED"] = eq5
  yr5["strategy"] = "R5_DRAWDOWN_CONTROLLED"
  yearly_all.append(yr5)
  schedules["R5_DRAWDOWN_CONTROLLED"] = blend_schedules(base_a, sched_e4, 0.7, 0.3)

# R7 V6 + R4 research-only
v6_eq = load_v6_equity()
r4_eq = equities.get("R4_BENCHMARK_AWARE")
if v6_eq is not None and r4_eq is not None and len(r4_eq) > 1:
  v6_col = "blended_champion_weights_alpha_0.5"
  if v6_col in v6_eq.columns:
    v6s = v6_eq[v6_col].dropna()
    blended = pd.concat([
      r4_eq.pct_change().fillna(0).rename("r4"),
      v6s.reindex(r4_eq.index).ffill().pct_change().fillna(0).rename("v6"),
    ], axis=1).dropna()
    if len(blended):
      combo_rets = 0.5 * blended["r4"] + 0.5 * blended["v6"]
      eq7 = INITIAL_CAPITAL * (1 + combo_rets).cumprod()
      m7, yr7 = compare_to_benchmarks(eq7, close_prices)
      m7["strategy"] = "R7_V6_PLUS_R13_RESEARCH_ONLY"
      m7["research_only"] = True
      strategy_results.append(m7)
      equities["R7_V6_PLUS_R13_RESEARCH_ONLY"] = eq7
      yr7["strategy"] = "R7_V6_PLUS_R13_RESEARCH_ONLY"
      yearly_all.append(yr7)

results_df = pd.DataFrame(strategy_results)
yearly_df = pd.concat(yearly_all, ignore_index=True) if yearly_all else pd.DataFrame()
print("Estrategias probadas:", len(results_df))
if len(results_df):
  print(results_df[["strategy", "CAGR", "sharpe", "max_drawdown", "win_years_vs_spy"]].to_string(index=False))

# %% [markdown]
# ## 7. Robustez y scoring V14

# %%
def compute_robustness_score(results_df):
  if len(results_df) < 2:
    return 70
  gap = results_df["sharpe"].max() - results_df["sharpe"].median()
  return int(np.clip(100 - gap * 25 - max(0, len(results_df) - 8) * 2, 0, 100))


def overfitting_risk(results_df):
  if len(results_df) < 2:
    return "LOW"
  gap = results_df["sharpe"].max() - results_df["sharpe"].median()
  return "HIGH" if gap > 0.6 else ("MEDIUM" if gap > 0.35 else "LOW")


def compute_v14_score(row, robustness, ovr_risk, spy_total, yearly_row=None, max_conc=0.0):
  score = 0
  if row.get("sharpe", 0) > 1:
    score += 15
  if row.get("sortino", 0) > 1.3:
    score += 15
  if row.get("max_drawdown", -100) > -25:
    score += 15
  if row.get("CAGR", 0) > 10:
    score += 10
  if row.get("total_return", 0) > spy_total:
    score += 10
  if row.get("win_years_vs_spy", 0) >= 0.5:
    score += 10
  if row.get("win_years_vs_qqq", 0) >= 0.35:
    score += 5
  if row.get("win_years_vs_6040", 0) >= 0.7:
    score += 10
  if robustness >= 70:
    score += 10
  if ovr_risk == "HIGH":
    score -= 20
  if row.get("max_drawdown", -100) < -30:
    score -= 20
  gross = row.get("total_return", 0)
  cost = row.get("total_cost", 0)
  if gross > 0 and cost > gross * 0.3:
    score -= 15
  if max_conc > MAX_WEIGHT + 0.05:
    score -= 15
  if yearly_row is not None and len(yearly_row):
    recent = yearly_row[yearly_row["year"].isin([2025, 2026])]
    if len(recent) and (recent["return"] < 0).any():
      score -= 15
  return int(np.clip(score, 0, 100))


def contribution_by_asset(close, schedule):
  if not schedule:
    return pd.DataFrame()
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


def avg_max_concentration(schedule):
  if not schedule:
    return 0.0
  return float(np.mean([w.max() for w in schedule.values() if len(w)]))


def leave_one_out_rows(name, schedule, top_n=3):
  contrib = contribution_by_asset(close_prices, schedule)
  if contrib.empty:
    return []
  rows = []
  for t in contrib.head(top_n)["ticker"]:
    loo = {}
    for dt, w in schedule.items():
      w2 = w.drop(t, errors="ignore")
      loo[dt] = cap_and_redistribute_weights(w2) if w2.sum() > 0 else pd.Series({CASH_ASSET: 1.0})
    m, _, _, _ = run_strategy_backtest(f"{name}_LOO_{t}", loo)
    if m:
      rows.append({"test": "leave_one_out", "removed": t, "strategy": name, **m})
  return rows


robustness_score = compute_robustness_score(results_df)
ovr = overfitting_risk(results_df)
spy_total = (close_prices[MARKET].iloc[-1] / close_prices[MARKET].iloc[0] - 1) * 100

if len(results_df):
  conc_map = {n: avg_max_concentration(schedules.get(n, {})) for n in results_df["strategy"]}
  yearly_map = {n: yearly_df[yearly_df["strategy"] == n] for n in results_df["strategy"]}
  results_df["v14_score"] = results_df.apply(
    lambda r: compute_v14_score(
      r, robustness_score, ovr, spy_total,
      yearly_row=yearly_map.get(r["strategy"]),
      max_conc=conc_map.get(r["strategy"], 0),
    ),
    axis=1,
  )
  tradable = results_df[~results_df["strategy"].str.contains("RESEARCH_ONLY", na=False)]
  champion = tradable.sort_values("v14_score", ascending=False).iloc[0]
else:
  champion = pd.Series(dtype=float)

v14_score = int(champion.get("v14_score", 0)) if len(champion) else 0
approved = (
  v14_score >= 75 and champion.get("sharpe", 0) > 1
  and champion.get("max_drawdown", -100) > -25 and ovr != "HIGH"
)
v14_status = "APPROVED_FOR_WEB_PAPER" if approved else ("CANDIDATE" if v14_score >= 60 else "REJECTED")
APPROVED_FOR_WEB_PAPER = approved
print(f"V14: {v14_status} | score {v14_score}")

# Start-date robustness
rob_rows = []
best_name = champion.get("strategy") if len(champion) else "R4_BENCHMARK_AWARE"
for sd in START_DATES_ROBUST:
  if sd >= START_DATE:
    continue
  try:
    _, c2 = download_data(UNIVERSE, sd, END_DATE, min_days=150)
    f2 = calculate_premia_features(c2)
    ba2, be4_2, rd2 = rebuild_bases(c2, f2)
    if "R4" in best_name:
      sch = strategy_r4_on(c2, f2, rd2, ba2, be4_2)
    elif "R6" in best_name:
      sch = blend_schedules(ba2, be4_2, 0.5)
    else:
      sch = ba2
    m2, _, _, _ = run_strategy_backtest(best_name, sch)
    if m2:
      rob_rows.append({"start_date": sd, "strategy": best_name, **m2})
  except Exception:
    pass
robustness_df = pd.DataFrame(rob_rows)

# Cost sensitivity
cost_rows = []
champ_sched = schedules.get(best_name, schedules.get("R4_BENCHMARK_AWARE", {}))
for cr in COST_SENSITIVITY:
  m3, _, _, _ = run_strategy_backtest(best_name, champ_sched, cost_rate=cr)
  if m3:
    cost_rows.append({"cost_slippage": cr, **m3})
cost_sens_df = pd.DataFrame(cost_rows)

# %% [markdown]
# ## 8. Señales actuales

# %%
def generate_v14_signals(sched, feats, strategy_name):
  if not sched:
    return pd.DataFrame()
  last = max(sched.keys())
  prev_keys = sorted([d for d in sched.keys() if d < last])
  target, prev = sched[last], sched[prev_keys[-1]] if prev_keys else pd.Series(dtype=float)
  rows = []
  for t in sorted(set(target.index) | set(prev.index)):
    tw, pw = _sf(target.get(t), 0), _sf(prev.get(t), 0)
    chg = tw - pw
    sc = 50.0
    r = _row(feats, t, last)
    if r is not None:
      sc = (_sf(r.get("rank_mom_63"), 0.5) + _sf(r.get("TREND_SCORE"), 0.5)) * 50
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
    rows.append({
      "ticker": t, "signal": sig, "target_weight": round(tw, 4), "previous_weight": round(pw, 4),
      "change": round(chg, 4), "score": round(sc, 1), "strategy_source": strategy_name,
      "reason": f"V14 refinement | {sig.lower()}",
      "entry_plan": "proxima apertura post-viernes" if sig in ("BUY", "INCREASE") else "-",
      "exit_plan": "rebalance semanal" if sig in ("SELL", "REDUCE") else "mantener hasta viernes",
      "next_review": "proximo viernes", "cash_account_executable": True,
    })
  return pd.DataFrame(rows)


champ_name = champion.get("strategy", "R4_BENCHMARK_AWARE") if len(champion) else "R4_BENCHMARK_AWARE"
champ_sched = schedules.get(champ_name, schedules.get("R4_BENCHMARK_AWARE", {}))
contrib_df = contribution_by_asset(close_prices, champ_sched)
if not QUICK_TEST:
  rob_rows.extend(leave_one_out_rows(champ_name, champ_sched))
  robustness_df = pd.DataFrame(rob_rows)
current_signals = generate_v14_signals(champ_sched, features, champ_name)
print("Señales:", len(current_signals))

# %% [markdown]
# ## 9. Exportar y reporte

# %%
def export_csv(df, name):
  (df if df is not None and len(df) else pd.DataFrame()).to_csv(name, index=False)
  print("Exportado", name)


v6_total = np.nan
if v6_eq is not None and "blended_champion_weights_alpha_0.5" in v6_eq.columns:
  s = v6_eq["blended_champion_weights_alpha_0.5"].dropna()
  if len(s) >= 2:
    v6_total = (s.iloc[-1] / s.iloc[0] - 1) * 100

champ_eq = equities.get(champ_name, pd.Series(dtype=float))
summary = {
  "lab": "v14_champion_refinement",
  "v14_score": v14_score,
  "status": v14_status,
  "approved_for_web_paper": APPROVED_FOR_WEB_PAPER,
  "approved_for_real_money": False,
  "champion_strategy": champ_name,
  "robustness_score": robustness_score,
  "overfitting_risk": ovr,
  **(champion.to_dict() if len(champion) else {}),
  "spy_return": round(spy_total, 2),
  "v6_return": round(v6_total, 2) if pd.notna(v6_total) else np.nan,
  "v13_reference": "A_tsmom + E4_adaptive refinement",
}

export_csv(pd.DataFrame([summary]), "research_v14_summary.csv")
export_csv(results_df, "research_v14_strategy_results.csv")
export_csv(yearly_df, "research_v14_yearly.csv")
export_csv(robustness_df, "research_v14_robustness.csv")
export_csv(cost_sens_df, "research_v14_cost_sensitivity.csv")
export_csv(current_signals, "research_v14_current_signals.csv")
if len(current_signals):
  current_signals.to_csv("current_signals_v14.csv", index=False)
if len(contrib_df):
  contrib_df.to_csv("research_v14_contribution_by_asset.csv", index=False)
if len(champ_eq):
  champ_eq.to_frame("equity").to_csv("research_v14_equity_curve.csv")
else:
  pd.DataFrame().to_csv("research_v14_equity_curve.csv", index=False)

config = {
  "version": "v14_champion_refinement",
  "approved_for_web_paper": APPROVED_FOR_WEB_PAPER,
  "approved_for_real_money": False,
  "v14_score": v14_score,
  "status": v14_status,
  "champion": champ_name,
  "bases": {"A_tsmom_63": "top_n=3 vol=0.15 lookback=63", "E4_adaptive": "risk-on/off sleeve mix"},
  "warnings": ["Backtest no garantiza resultados futuros.", "R7 es research-only si existe V6."],
}
Path("research_v14_selected_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
print("Exportado research_v14_selected_config.json")

print("=" * 80)
print("REPORTE FINAL V14 CHAMPION REFINEMENT LAB")
print("=" * 80)
print(f"Estado: {v14_status} | Score: {v14_score}/100")
print(f"Mejor estrategia: {champ_name}")
if len(champion):
  print(f"CAGR {champion.get('CAGR')}% | Sharpe {champion.get('sharpe')} | Sortino {champion.get('sortino')}")
  print(f"DD {champion.get('max_drawdown')}% | Win SPY {champion.get('win_years_vs_spy', 0)*100:.0f}% | Win QQQ {champion.get('win_years_vs_qqq', 0)*100:.0f}%")
print(f"Robustez: {robustness_score} | Overfitting: {ovr}")
if pd.notna(v6_total):
  print(f"V6: {v6_total:.1f}% | V14 champ: {champion.get('total_return', 0)}%")
print(f"BUY actuales: {len(current_signals[current_signals['signal']=='BUY']) if len(current_signals) else 0}")
print("")
if APPROVED_FOR_WEB_PAPER:
  print("Integrar V14 como Candidate/Approved Paper Trading.")
else:
  print("V14 no mejora suficiente. Mantener V13 candidate y V6 champion.")
print("APPROVED_FOR_REAL_MONEY=False (siempre)")
