# %% [markdown]
# # Trading Research V9 Clean Signal Research Lab
#
# Tres motores separados: True Pairs L/S, Pairs Long-Only, Cross-Sectional Reversion.
#
# **Disclaimer:** Backtest no garantiza resultados futuros. No es asesoramiento financiero.
# **APPROVED_FOR_REAL_MONEY siempre False.**

# %%
!pip install yfinance pandas numpy matplotlib plotly tqdm scipy statsmodels scikit-learn -q

# %% [markdown]
# ## 1. Configuracion

# %%
import warnings
warnings.filterwarnings("ignore")

import json
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tqdm.auto import tqdm

try:
  import statsmodels.api as sm
  from statsmodels.tsa.stattools import adfuller, coint
  HAS_STATSMODELS = True
except Exception:
  sm = None
  HAS_STATSMODELS = False

QUICK_TEST = True
START_DATE = "2015-01-01"
END_DATE = None
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001
SLIPPAGE = 0.001
SHORT_BORROW_COST_ANNUAL = 0.03
MAX_CONCURRENT_TRADES = 5
POSITION_SIZE_PCT = 0.10
ENTRY_Z_VALUES = [1.5, 2.0, 2.5]
EXIT_Z_VALUES = [0.25, 0.5, 0.75]
STOP_Z_VALUES = [3.0, 3.5, 4.0]
MAX_HOLD_DAYS_VALUES = [5, 10, 20]
TRAIN_YEARS = 3
WF_START_YEAR = 2019
ROLLING_Z = 60
TOP_PAIRS_PER_YEAR = 20
MIN_TRAIN_DAYS = 500
MARKET = "SPY"
DEBUG = False

UNIVERSE = [
  "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "INTC", "MU",
  "GOOGL", "META", "AMZN", "NFLX", "ADBE", "CRM", "ORCL",
  "JPM", "BAC", "GS", "MS",
  "XOM", "CVX",
  "UNH", "LLY", "JNJ",
  "WMT", "COST", "HD", "MCD",
  "SPY", "QQQ", "IWM",
]

SECTOR_GROUPS = {
  "mega_tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN"],
  "semis": ["NVDA", "AMD", "AVGO", "INTC", "MU"],
  "software": ["MSFT", "ADBE", "CRM", "ORCL"],
  "banks": ["JPM", "BAC", "GS", "MS"],
  "energy": ["XOM", "CVX"],
  "healthcare": ["UNH", "LLY", "JNJ"],
  "consumer": ["WMT", "COST", "HD", "MCD"],
}

if QUICK_TEST:
  UNIVERSE = sorted(set(SECTOR_GROUPS["mega_tech"] + SECTOR_GROUPS["semis"] + ["SPY", "QQQ"]))
  SECTOR_GROUPS = {
    "mega_tech": [t for t in SECTOR_GROUPS["mega_tech"] if t in UNIVERSE],
    "semis": [t for t in SECTOR_GROUPS["semis"] if t in UNIVERSE],
  }
  START_DATE = "2020-01-01"
  WF_START_YEAR = 2021
  ENTRY_Z_VALUES = [2.0]
  EXIT_Z_VALUES = [0.5]
  STOP_Z_VALUES = [3.5]
  MAX_HOLD_DAYS_VALUES = [10]
  TOP_PAIRS_PER_YEAR = 8
  print("QUICK_TEST activo")

print("V9 Clean Signal Research Lab | desde", START_DATE)
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
# ## 3. Funciones robustas

# %%
def safe_float(x, default=np.nan):
  try:
    if x is None or (isinstance(x, float) and np.isnan(x)):
      return default
    if isinstance(x, pd.Series):
      if len(x) == 0:
        return default
      x = x.iloc[0]
    v = float(x)
    return default if not np.isfinite(v) else v
  except Exception:
    return default


def safe_ticker(x, default=""):
  try:
    if x is None or (isinstance(x, float) and np.isnan(x)):
      return default
    s = str(x).strip().upper()
    return default if s in ("", "NONE", "NAN", "NULL") else s
  except Exception:
    return default


def safe_pair_name(long_ticker=None, short_ticker=None, fallback=None):
  if fallback:
    fp = str(fallback).strip()
    if fp and fp.upper() not in ("NONE", "NAN", "NULL"):
      return fp
  return f"{safe_ticker(long_ticker, 'LONG')}-{safe_ticker(short_ticker, 'SHORT')}"


def calculate_log_prices(close_prices):
  return np.log(close_prices.replace(0, np.nan)).ffill()


def calculate_returns(close_prices):
  return close_prices.pct_change().fillna(0)


def calculate_zscore(series, window=ROLLING_Z):
  s = pd.Series(series)
  mu = s.rolling(window).mean().shift(1)
  sd = s.rolling(window).std().shift(1).replace(0, np.nan)
  return (s - mu) / sd


def ols_hedge_ratio(y, x):
  if not HAS_STATSMODELS:
    return np.nan
  df = pd.concat([y, x], axis=1).dropna()
  if len(df) < 60:
    return np.nan
  yy = pd.to_numeric(df.iloc[:, 0], errors="coerce")
  xx = pd.to_numeric(df.iloc[:, 1], errors="coerce")
  clean = pd.concat([yy, xx], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
  if len(clean) < 60:
    return np.nan
  X = sm.add_constant(clean.iloc[:, 1], has_constant="add")
  try:
    model = sm.OLS(clean.iloc[:, 0], X).fit()
    return float(model.params.iloc[1]) if len(model.params) >= 2 else np.nan
  except Exception:
    return np.nan


def adf_pvalue(series):
  s = pd.Series(series).dropna()
  if len(s) < 30 or not HAS_STATSMODELS:
    return 1.0
  try:
    return float(adfuller(s, maxlag=1, regression="c", autolag=None)[1])
  except Exception:
    return 1.0


def cointegration_pvalue(y, x):
  y, x = pd.Series(y).dropna(), pd.Series(x).dropna()
  idx = y.index.intersection(x.index)
  if len(idx) < 60 or not HAS_STATSMODELS:
    return 1.0
  try:
    return float(coint(y.loc[idx], x.loc[idx])[1])
  except Exception:
    return 1.0


def estimate_half_life(spread):
  s = pd.Series(spread).dropna()
  if len(s) < 30 or not HAS_STATSMODELS:
    return np.nan
  lag = s.shift(1)
  delta = s - lag
  df = pd.DataFrame({"lag": lag, "delta": delta}).dropna()
  if len(df) < 20:
    return np.nan
  try:
    model = sm.OLS(df["delta"], sm.add_constant(df["lag"], has_constant="add")).fit()
    beta = float(model.params.iloc[1])
    if beta >= 0:
      return np.nan
    return float(-np.log(2) / beta)
  except Exception:
    return np.nan


def max_drawdown(equity):
  eq = pd.Series(equity).dropna().astype(float)
  if eq.empty:
    return 0.0
  peak = eq.cummax()
  dd = (eq - peak) / peak.replace(0, np.nan)
  return float(dd.min())


def sharpe(returns, ann=252):
  r = pd.Series(returns).dropna().astype(float)
  if r.std() <= 0 or len(r) < 2:
    return 0.0
  return float(r.mean() / r.std() * np.sqrt(ann))


def sortino(returns, ann=252):
  r = pd.Series(returns).dropna().astype(float)
  ds = r[r < 0].std()
  if not ds or ds <= 0:
    return 0.0
  return float(r.mean() / ds * np.sqrt(ann))


def profit_factor(trades_df):
  if trades_df is None or len(trades_df) == 0:
    return 0.0
  wins = trades_df[trades_df["net_return_pct"] > 0]["pnl"].sum()
  losses = abs(trades_df[trades_df["net_return_pct"] <= 0]["pnl"].sum())
  return float(wins / losses) if losses > 0 else 0.0


def exec_price(data_dict, ticker, dt, use_open=True):
  df = data_dict.get(ticker)
  if df is None or dt not in df.index:
    return np.nan
  row = df.loc[dt]
  if use_open and "Open" in row.index and pd.notna(row["Open"]):
    return float(row["Open"])
  return float(row["Close"])


def param_grid():
  return [
    {"entry_z": e, "exit_z": x, "stop_z": s, "max_hold_days": m}
    for e, x, s, m in product(ENTRY_Z_VALUES, EXIT_Z_VALUES, STOP_Z_VALUES, MAX_HOLD_DAYS_VALUES)
  ]

# %% [markdown]
# ## 4. Seleccion de pares por train (solo grupos sectoriales)

# %%
def score_pair_in_group(log_px, rets, y_t, x_t, train_start, train_end):
  try:
    sub = log_px.loc[train_start:train_end, [y_t, x_t]].dropna()
    if len(sub) < MIN_TRAIN_DAYS:
      return None
    corr = rets[y_t].loc[sub.index].corr(rets[x_t].loc[sub.index])
    if pd.isna(corr) or corr < 0.55:
      return None
    beta = ols_hedge_ratio(sub[y_t], sub[x_t])
    if pd.isna(beta):
      return None
    spread = sub[y_t] - beta * sub[x_t]
    hl = estimate_half_life(spread)
    if pd.isna(hl) or hl < 2 or hl > 40:
      return None
    coint_p = cointegration_pvalue(sub[y_t], sub[x_t])
    if coint_p >= 0.15:
      return None
    spread_vol = spread.diff().std()
    if pd.isna(spread_vol) or spread_vol < 1e-4:
      return None
    hl_pref = 1.0 if 5 <= hl <= 20 else 0.7
    rank = (1 - coint_p) * corr * hl_pref / max(spread_vol, 1e-4)
    return {
      "pair": f"{y_t}-{x_t}",
      "y_ticker": y_t,
      "x_ticker": x_t,
      "sector_group": "",
      "hedge_ratio": beta,
      "correlation": round(corr, 3),
      "coint_pvalue": round(coint_p, 4),
      "half_life": round(hl, 2),
      "spread_vol": round(spread_vol, 5),
      "rank_score": round(rank, 4),
    }
  except Exception:
    return None


def select_pairs_for_train(close_prices, train_start, train_end, universe, sector_groups, top_n=TOP_PAIRS_PER_YEAR):
  log_px = calculate_log_prices(close_prices)
  rets = calculate_returns(close_prices)
  uni = [c for c in universe if c in close_prices.columns]
  rows = []
  for group_name, members in sector_groups.items():
    group = [m for m in members if m in uni]
    if len(group) < 2:
      continue
    for y_t, x_t in combinations(group, 2):
      for a, b in [(y_t, x_t), (x_t, y_t)]:
        rec = score_pair_in_group(log_px, rets, a, b, train_start, train_end)
        if rec:
          rec["sector_group"] = group_name
          rows.append(rec)
  if not rows:
    return pd.DataFrame()
  df = pd.DataFrame(rows).drop_duplicates(subset=["pair"]).sort_values("rank_score", ascending=False)
  return df.head(top_n).reset_index(drop=True)

# %% [markdown]
# ## 5. Motor A: True Pairs Long/Short

# %%
def fit_pair_spread(close_prices, y_t, x_t, train_start, train_end):
  log_px = calculate_log_prices(close_prices)
  sub = log_px.loc[train_start:train_end, [y_t, x_t]].dropna()
  if len(sub) < 120:
    return np.nan, pd.Series(dtype=float)
  beta = ols_hedge_ratio(sub[y_t], sub[x_t])
  if pd.isna(beta):
    return np.nan, pd.Series(dtype=float)
  spread = log_px[y_t] - beta * log_px[x_t]
  z = calculate_zscore(spread, ROLLING_Z)
  return beta, z


def simulate_pair_trades(pair_row, close_prices, data_dict, train_start, train_end, test_start, test_end,
                       params, long_short=True, strategy_name="true_pairs_long_short"):
  y_t = pair_row["y_ticker"]
  x_t = pair_row["x_ticker"]
  beta, z_full = fit_pair_spread(close_prices, y_t, x_t, train_start, train_end)
  if pd.isna(beta) or z_full.empty:
    return pd.DataFrame()
  z = z_full.loc[test_start:test_end].dropna()
  if z.empty:
    return pd.DataFrame()
  entry_z = params["entry_z"]
  exit_z = params["exit_z"]
  stop_z = params["stop_z"]
  max_hold = params["max_hold_days"]
  cost_side = TRANSACTION_COST + SLIPPAGE
  borrow_daily = SHORT_BORROW_COST_ANNUAL / 252
  size_pct = POSITION_SIZE_PCT * (0.6 if not long_short else 1.0)
  trades = []
  position = None
  dates = list(z.index)
  for i, dt in enumerate(dates):
    zval = z.loc[dt]
    if pd.isna(zval):
      continue
    if position is None:
      future = close_prices.index[close_prices.index > dt]
      if len(future) == 0:
        continue
      entry_exec = future[0]
      if zval > entry_z:
        position = {
          "signal_dt": dt, "entry_exec": entry_exec,
          "long_t": x_t, "short_t": y_t, "entry_z": zval, "dir": "short_spread",
        }
      elif zval < -entry_z:
        position = {
          "signal_dt": dt, "entry_exec": entry_exec,
          "long_t": y_t, "short_t": x_t, "entry_z": zval, "dir": "long_spread",
        }
    else:
      hold = (dt - position["entry_exec"]).days
      reason = None
      if abs(zval) < exit_z:
        reason = "z_exit"
      elif abs(zval) > stop_z:
        reason = "z_stop"
      elif hold >= max_hold:
        reason = "max_hold"
      elif position["dir"] == "long_spread" and zval > 0:
        reason = "z_cross"
      elif position["dir"] == "short_spread" and zval < 0:
        reason = "z_cross"
      if reason:
        exit_future = close_prices.index[close_prices.index > dt]
        if len(exit_future) == 0:
          position = None
          continue
        entry_exec = position["entry_exec"]
        exit_exec = exit_future[0]
        long_t = position["long_t"]
        short_t = position["short_t"] if long_short else ""
        l0 = exec_price(data_dict, long_t, entry_exec)
        l1 = exec_price(data_dict, long_t, exit_exec, use_open=False)
        gross_parts = []
        if pd.notna(l0) and pd.notna(l1) and l0 > 0:
          gross_parts.append(l1 / l0 - 1)
        net_parts = []
        if gross_parts:
          net_parts.append(gross_parts[0] - 2 * cost_side)
        if long_short and short_t:
          s0 = exec_price(data_dict, short_t, entry_exec)
          s1 = exec_price(data_dict, short_t, exit_exec, use_open=False)
          if pd.notna(s0) and pd.notna(s1) and s0 > 0:
            short_ret = s0 / s1 - 1 - 2 * cost_side - borrow_daily * max((exit_exec - entry_exec).days, 1)
            gross_parts.append(s0 / s1 - 1)
            net_parts.append(short_ret)
        if not net_parts:
          position = None
          continue
        gross = np.mean(gross_parts) * 100
        net = np.mean(net_parts) * 100
        alloc = INITIAL_CAPITAL * size_pct
        trades.append({
          "strategy": strategy_name,
          "engine": "A" if long_short else "B",
          "pair": safe_pair_name(long_t, short_t, pair_row.get("pair")),
          "long_ticker": long_t,
          "short_ticker": short_t if long_short else "",
          "entry_date": entry_exec,
          "exit_date": exit_exec,
          "entry_z": round(position["entry_z"], 2),
          "exit_z": round(zval, 2),
          "exit_reason": reason,
          "gross_return_pct": round(gross, 3),
          "net_return_pct": round(net, 3),
          "pnl": round(alloc * net / 100, 2),
          "holding_days": (exit_exec - entry_exec).days,
          "requires_short": long_short,
          "executable_in_cash_account": not long_short,
        })
        position = None
  return pd.DataFrame(trades)


def generate_true_pair_trades(pair_row, close_prices, data_dict, train_start, train_end, test_start, test_end, params):
  return simulate_pair_trades(
    pair_row, close_prices, data_dict, train_start, train_end, test_start, test_end,
    params, long_short=True, strategy_name="true_pairs_long_short",
  )

# %% [markdown]
# ## 6. Motor B: Pairs Long-Only Fallback

# %%
def generate_pair_long_only_trades(pair_row, close_prices, data_dict, train_start, train_end, test_start, test_end, params):
  return simulate_pair_trades(
    pair_row, close_prices, data_dict, train_start, train_end, test_start, test_end,
    params, long_short=False, strategy_name="pair_long_only_fallback",
  )

# %% [markdown]
# ## 7. Motor C: Cross-Sectional Mean Reversion Long-Only

# %%
def add_cs_features(df, spy_close=None):
  d = df.copy()
  c = d["Close"]
  r = c.pct_change().fillna(0)
  d["RET_1D"] = r
  d["RET_3D"] = c.pct_change(3)
  d["RET_5D"] = c.pct_change(5)
  d["SMA_50"] = c.rolling(50).mean()
  d["SMA_200"] = c.rolling(200).mean()
  d["DIST_SMA50"] = c / d["SMA_50"] - 1
  d["DIST_SMA200"] = c / d["SMA_200"] - 1
  delta = c.diff()
  gain = delta.clip(lower=0).rolling(14).mean()
  loss = (-delta.clip(upper=0)).rolling(14).mean()
  d["RSI_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
  gain2 = delta.clip(lower=0).rolling(2).mean()
  loss2 = (-delta.clip(upper=0)).rolling(2).mean()
  d["RSI_2"] = 100 - 100 / (1 + gain2 / loss2.replace(0, np.nan))
  tr = (d["High"] - d["Low"]).replace(0, np.nan) if "High" in d.columns else c * 0.02
  d["ATR_14"] = tr.rolling(14).mean()
  d["ATR_PCT"] = d["ATR_14"] / c.replace(0, np.nan)
  if "Volume" in d.columns:
    d["VOL_RATIO"] = d["Volume"] / d["Volume"].rolling(20).mean().replace(0, np.nan)
  else:
    d["VOL_RATIO"] = 1.0
  if spy_close is not None:
    spy = spy_close.reindex(d.index).ffill()
    d["SPY_SMA200"] = spy.rolling(200).mean()
    d["SPY_ABOVE_200"] = (spy > d["SPY_SMA200"]).astype(float)
  return d


def generate_cross_sectional_reversion_trades(data_dict, close_prices, sector_groups, test_start, test_end, max_hold=5):
  spy = close_prices[MARKET] if MARKET in close_prices.columns else None
  feats = {t: add_cs_features(data_dict[t], spy) for t in data_dict if t not in ("SPY", "QQQ", "IWM")}
  test_dates = close_prices.loc[test_start:test_end].index
  cost_side = TRANSACTION_COST + SLIPPAGE
  trades = []
  open_pos = {}
  for dt in test_dates:
    for group_name, members in sector_groups.items():
      group = [m for m in members if m in feats]
      if len(group) < 2:
        continue
      rel_rets = {}
      for t in group:
        f = feats[t]
        if dt not in f.index:
          continue
        row = f.loc[dt]
        rel_rets[t] = safe_float(row.get("RET_3D"), 0)
      if len(rel_rets) < 2:
        continue
      s = pd.Series(rel_rets)
      p20 = s.quantile(0.20)
      for t, ret3 in rel_rets.items():
        row = feats[t].loc[dt]
        if t in open_pos:
          pos = open_pos[t]
          hold = (dt - pos["entry_dt"]).days
          rsi2 = safe_float(row.get("RSI_2"), 50)
          rel_now = ret3 - s.mean()
          exit_reason = None
          if rsi2 > 50:
            exit_reason = "rsi2_exit"
          elif abs(rel_now) < 0.005:
            exit_reason = "rel_mean"
          elif hold >= max_hold:
            exit_reason = "max_hold"
          low = safe_float(row.get("Low"), safe_float(row.get("Close")))
          high = safe_float(row.get("High"), safe_float(row.get("Close")))
          if low <= pos["stop"]:
            exit_reason = "stop_loss"
          elif high >= pos["take_profit"]:
            exit_reason = "take_profit"
          if exit_reason:
            future_exit = close_prices.index[close_prices.index > dt]
            if len(future_exit) == 0:
              del open_pos[t]
              continue
            exit_exec = future_exit[0]
            e0 = exec_price(data_dict, t, pos["entry_exec"])
            e1 = exec_price(data_dict, t, exit_exec, use_open=False)
            if pd.notna(e0) and pd.notna(e1) and e0 > 0:
              net = (e1 / e0 - 1 - 2 * cost_side) * 100
              trades.append({
                "strategy": "cross_sectional_reversion_long_only",
                "engine": "C",
                "pair": t,
                "long_ticker": t,
                "short_ticker": "",
                "entry_date": pos["entry_exec"],
                "exit_date": exit_exec,
                "entry_z": np.nan,
                "exit_z": np.nan,
                "exit_reason": exit_reason,
                "gross_return_pct": round(net, 3),
                "net_return_pct": round(net, 3),
                "pnl": round(INITIAL_CAPITAL * POSITION_SIZE_PCT * 0.6 * net / 100, 2),
                "holding_days": (exit_exec - pos["entry_exec"]).days,
                "requires_short": False,
                "executable_in_cash_account": True,
                "sector_group": group_name,
              })
            del open_pos[t]
        if t not in open_pos and len(open_pos) < MAX_CONCURRENT_TRADES:
          c = safe_float(row.get("Close"))
          sma200 = safe_float(row.get("SMA_200"))
          spy_ok = safe_float(row.get("SPY_ABOVE_200"), 1) >= 0.5
          rsi2 = safe_float(row.get("RSI_2"), 50)
          atr_pct = safe_float(row.get("ATR_PCT"), 0.02)
          vol_ratio = safe_float(row.get("VOL_RATIO"), 1)
          if (
            c > sma200 and spy_ok and ret3 <= p20 and rsi2 < 15
            and atr_pct < 0.06 and vol_ratio > 0.3
          ):
            future = close_prices.index[close_prices.index > dt]
            if len(future) == 0:
              continue
            entry_exec = future[0]
            atr = safe_float(row.get("ATR_14"), c * 0.02)
            open_pos[t] = {
              "entry_dt": dt,
              "entry_exec": entry_exec,
              "stop": c - 1.5 * atr,
              "take_profit": c + 2.0 * atr,
            }
  return pd.DataFrame(trades)

# %% [markdown]
# ## 8. Metricas y scoring

# %%
def compute_engine_metrics(trades_df, close_prices, label="engine"):
  if trades_df is None or len(trades_df) == 0:
    return {"engine": label, "num_trades": 0, "total_return": 0, "sharpe": 0, "profit_factor": 0}
  pnl = trades_df.sort_values("exit_date")["pnl"]
  equity = INITIAL_CAPITAL + pnl.cumsum()
  rets = equity.pct_change().fillna(0)
  total_ret = equity.iloc[-1] / INITIAL_CAPITAL - 1
  years = max((trades_df["exit_date"].max() - trades_df["entry_date"].min()).days / 365.25, 1 / 365.25)
  cagr = (equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1
  mdd = max_drawdown(equity)
  pf = profit_factor(trades_df)
  win_rate = (trades_df["net_return_pct"] > 0).mean() * 100
  beta = 0.0
  if MARKET in close_prices.columns and len(rets) > 10:
    spy_r = calculate_returns(close_prices[MARKET]).reindex(rets.index).fillna(0)
    if spy_r.std() > 0:
      beta = float(rets.cov(spy_r) / spy_r.var())
  return {
    "engine": label,
    "strategy": trades_df["strategy"].iloc[0] if len(trades_df) else label,
    "total_return": round(total_ret * 100, 2),
    "CAGR": round(cagr * 100, 2),
    "sharpe": round(sharpe(rets), 3),
    "sortino": round(sortino(rets), 3),
    "max_drawdown": round(mdd * 100, 2),
    "calmar": round(cagr / abs(mdd), 3) if mdd != 0 else 0,
    "num_trades": int(len(trades_df)),
    "win_rate": round(win_rate, 2),
    "avg_trade_return": round(trades_df["net_return_pct"].mean(), 3),
    "median_trade_return": round(trades_df["net_return_pct"].median(), 3),
    "best_trade": round(trades_df["net_return_pct"].max(), 3),
    "worst_trade": round(trades_df["net_return_pct"].min(), 3),
    "profit_factor": round(pf, 3),
    "expectancy": round(trades_df["pnl"].mean(), 2),
    "avg_holding_days": round(trades_df["holding_days"].mean(), 1),
    "exposure_pct": round(POSITION_SIZE_PCT * MAX_CONCURRENT_TRADES * 100, 1),
    "trades_per_year": round(len(trades_df) / years, 1),
    "beta_to_SPY": round(beta, 3),
  }


def score_params_on_train(pairs_df, close_prices, data_dict, train_start, train_end, params, long_short):
  all_trades = []
  for _, pr in pairs_df.iterrows():
    fn = generate_true_pair_trades if long_short else generate_pair_long_only_trades
    t = fn(pr, close_prices, data_dict, train_start, train_end, train_start, train_end, params)
    if len(t):
      all_trades.append(t)
  if not all_trades:
    return -999
  trades = pd.concat(all_trades, ignore_index=True)
  m = compute_engine_metrics(trades, close_prices)
  penalty = 0 if m["num_trades"] >= 20 else (20 - m["num_trades"]) * 0.05
  return m["sharpe"] + m["profit_factor"] * 0.5 - penalty


def select_best_params(pairs_df, close_prices, data_dict, train_start, train_end, long_short):
  best_score, best_params = -999, param_grid()[0]
  for params in param_grid():
    sc = score_params_on_train(pairs_df, close_prices, data_dict, train_start, train_end, params, long_short)
    if sc > best_score:
      best_score, best_params = sc, params.copy()
  return best_params, best_score


def approve_engine(metrics, yearly_df=None):
  pf = metrics.get("profit_factor", 0)
  sh = metrics.get("sharpe", 0)
  mdd = metrics.get("max_drawdown", -100)
  n = metrics.get("num_trades", 0)
  avg = metrics.get("avg_trade_return", 0)
  cost_thresh = (TRANSACTION_COST + SLIPPAGE) * 100 * 3
  beats_spy = 0.5
  if yearly_df is not None and len(yearly_df) and "beats_spy" in yearly_df.columns:
    beats_spy = yearly_df["beats_spy"].mean()
  approved = pf > 1.2 and sh > 0.8 and mdd > -25 and n >= 100 and beats_spy >= 0.5 and avg > cost_thresh
  candidate = pf > 1.1 and sh > 0.5 and n >= 50
  if approved:
    return "APPROVED_FOR_WEB_PAPER", 85
  if candidate:
    return "CANDIDATE", 65
  return "REJECTED", 35

# %% [markdown]
# ## 9. Walk-forward por motor

# %%
def run_engine_walk_forward(close_prices, data_dict, engine_key, long_short=None):
  yearly_rows, all_trades, pairs_log, param_log = [], [], [], []
  current_year = pd.Timestamp.today().year
  for year in tqdm(range(WF_START_YEAR, current_year + 1), desc=f"WF {engine_key}"):
    test_start = f"{year}-01-01"
    test_end = f"{year}-12-31"
    train_end = f"{year - 1}-12-31"
    train_start = (pd.Timestamp(train_end) - pd.DateOffset(years=TRAIN_YEARS)).strftime("%Y-%m-%d")
    if pd.Timestamp(train_end) <= pd.Timestamp(train_start):
      continue
    year_trades = []
    best_params = {"entry_z": 2.0, "exit_z": 0.5, "stop_z": 3.5, "max_hold_days": 10}
    if engine_key in ("A", "B"):
      pairs = select_pairs_for_train(close_prices, train_start, train_end, UNIVERSE, SECTOR_GROUPS)
      if len(pairs) == 0:
        continue
      pairs = pairs.assign(year=year)
      pairs_log.append(pairs)
      best_params, train_score = select_best_params(
        pairs, close_prices, data_dict, train_start, train_end, long_short=(engine_key == "A")
      )
      param_log.append({"year": year, "engine": engine_key, **best_params, "train_score": round(train_score, 3)})
      gen_fn = generate_true_pair_trades if engine_key == "A" else generate_pair_long_only_trades
      for _, pr in pairs.iterrows():
        t = gen_fn(pr, close_prices, data_dict, train_start, train_end, test_start, test_end, best_params)
        if len(t):
          year_trades.append(t)
    elif engine_key == "C":
      t = generate_cross_sectional_reversion_trades(data_dict, close_prices, SECTOR_GROUPS, test_start, test_end)
      if len(t):
        year_trades.append(t)
    if not year_trades:
      continue
    trades_year = pd.concat(year_trades, ignore_index=True)
    all_trades.append(trades_year.assign(year=year))
    m = compute_engine_metrics(trades_year, close_prices, engine_key)
    spy_slice = close_prices[MARKET].loc[test_start:test_end] if MARKET in close_prices.columns else pd.Series()
    spy_ret = (spy_slice.iloc[-1] / spy_slice.iloc[0] - 1) * 100 if len(spy_slice) >= 2 else 0
    yearly_rows.append({
      "year": year,
      "engine": engine_key,
      "return": m["total_return"],
      "SPY_return": round(spy_ret, 2),
      "num_trades": m["num_trades"],
      "win_rate": m["win_rate"],
      "profit_factor": m["profit_factor"],
      "sharpe": m["sharpe"],
      "max_drawdown": m["max_drawdown"],
      "beats_spy": m["total_return"] > spy_ret,
    })
  return (
    pd.DataFrame(yearly_rows),
    pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(),
    pd.concat(pairs_log, ignore_index=True) if pairs_log else pd.DataFrame(),
    pd.DataFrame(param_log),
  )


yearly_a, trades_a, pairs_a, params_a = run_engine_walk_forward(close_prices, data_dict, "A", long_short=True)
yearly_b, trades_b, pairs_b, params_b = run_engine_walk_forward(close_prices, data_dict, "B", long_short=False)
yearly_c, trades_c, pairs_c, params_c = run_engine_walk_forward(close_prices, data_dict, "C")
print("Trades A:", len(trades_a), "| B:", len(trades_b), "| C:", len(trades_c))

# %% [markdown]
# ## 10. Resultados por motor

# %%
metrics_a = compute_engine_metrics(trades_a, close_prices, "A_true_pairs_long_short")
metrics_b = compute_engine_metrics(trades_b, close_prices, "B_pair_long_only")
metrics_c = compute_engine_metrics(trades_c, close_prices, "C_cross_sectional")
engine_results = pd.DataFrame([metrics_a, metrics_b, metrics_c])

status_a, score_a = approve_engine(metrics_a, yearly_a)
status_b, score_b = approve_engine(metrics_b, yearly_b)
status_c, score_c = approve_engine(metrics_c, yearly_c)
engine_results["status"] = [status_a, status_b, status_c]
engine_results["engine_score"] = [score_a, score_b, score_c]
engine_results["approved_for_web_paper"] = [s == "APPROVED_FOR_WEB_PAPER" for s in [status_a, status_b, status_c]]
engine_results["approved_for_real_money"] = False

print(engine_results.to_string(index=False))

results_by_engine = engine_results.copy()
yearly_by_engine = pd.concat([yearly_a, yearly_b, yearly_c], ignore_index=True) if len(yearly_a) + len(yearly_b) + len(yearly_c) else pd.DataFrame()
trades_by_engine = pd.concat([trades_a, trades_b, trades_c], ignore_index=True) if len(trades_a) + len(trades_b) + len(trades_c) else pd.DataFrame()
pairs_selected = pd.concat([pairs_a, pairs_b], ignore_index=True).drop_duplicates() if len(pairs_a) + len(pairs_b) else pd.DataFrame()
param_selection = pd.concat([params_a, params_b], ignore_index=True) if len(params_a) + len(params_b) else pd.DataFrame()

pair_quality = pairs_selected.groupby("year").agg(
  n_pairs=("pair", "count"),
  avg_corr=("correlation", "mean"),
  avg_half_life=("half_life", "mean"),
  avg_coint_p=("coint_pvalue", "mean"),
).reset_index() if len(pairs_selected) else pd.DataFrame()

# %% [markdown]
# ## 11. Benchmarks V6 / V8

# %%
def load_equity(path, col):
  p = Path(path)
  if not p.exists():
    return np.nan
  try:
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    if col in df.columns:
      s = df[col].dropna()
      if len(s) >= 2:
        return (s.iloc[-1] / s.iloc[0] - 1) * 100
  except Exception:
    pass
  return np.nan


spy_total = (close_prices[MARKET].iloc[-1] / close_prices[MARKET].iloc[0] - 1) * 100 if MARKET in close_prices else 0
qqq_total = (close_prices["QQQ"].iloc[-1] / close_prices["QQQ"].iloc[0] - 1) * 100 if "QQQ" in close_prices else 0
v6_total = load_equity("research_outputs/v6/research_v6_equity_curves.csv", "blended_champion_weights_alpha_0.5")
if pd.isna(v6_total):
  v6_total = load_equity("research_v6_equity_curves.csv", "blended_champion_weights_alpha_0.5")
v8_total = load_equity("research_v8_equity_curve.csv", "equity")

# %% [markdown]
# ## 12. Señales actuales

# %%
def build_current_signals(close_prices, data_dict, pairs_df):
  rows = []
  last = close_prices.index[-1]
  train_end = (last - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
  train_start = (last - pd.DateOffset(years=TRAIN_YEARS)).strftime("%Y-%m-%d")
  if pairs_df is None or len(pairs_df) == 0:
    pairs_df = select_pairs_for_train(close_prices, train_start, train_end, UNIVERSE, SECTOR_GROUPS, top_n=5)
  params = {"entry_z": 2.0, "exit_z": 0.5, "stop_z": 3.5, "max_hold_days": 10}
  for _, pr in pairs_df.head(5).iterrows():
    beta, z = fit_pair_spread(close_prices, pr["y_ticker"], pr["x_ticker"], train_start, train_end)
    if pd.isna(beta) or z.empty:
      continue
    zval = safe_float(z.iloc[-1])
    if abs(zval) < params["entry_z"]:
      continue
    if zval > params["entry_z"]:
      long_t, short_t = pr["x_ticker"], pr["y_ticker"]
    else:
      long_t, short_t = pr["y_ticker"], pr["x_ticker"]
    conf = int(np.clip(50 + abs(zval) * 12, 0, 100))
    rows.append({
      "engine": "A",
      "strategy": "true_pairs_long_short",
      "ticker_or_pair": pr["pair"],
      "long_ticker": long_t,
      "short_ticker": short_t,
      "signal": f"LONG {long_t} / SHORT {short_t}",
      "entry_plan": "entrada proxima apertura",
      "exit_plan": f"salida z < {params['exit_z']}",
      "stop_plan": f"stop z {params['stop_z']}",
      "take_profit_plan": "-",
      "max_hold_days": params["max_hold_days"],
      "confidence_score": conf,
      "executable_in_cash_account": False,
      "requires_short": True,
      "reason": f"z={zval:.1f} mean-reversion pair",
    })
    rows.append({
      "engine": "B",
      "strategy": "pair_long_only_fallback",
      "ticker_or_pair": pr["pair"],
      "long_ticker": long_t,
      "short_ticker": "",
      "signal": f"BUY {long_t}",
      "entry_plan": "entrada proxima apertura",
      "exit_plan": f"salida z < {params['exit_z']}",
      "stop_plan": f"stop z {params['stop_z']}",
      "take_profit_plan": "-",
      "max_hold_days": params["max_hold_days"],
      "confidence_score": max(40, conf - 10),
      "executable_in_cash_account": True,
      "requires_short": False,
      "reason": f"long-only fallback z={zval:.1f}",
    })
  cs = generate_cross_sectional_reversion_trades(
    data_dict, close_prices, SECTOR_GROUPS,
    (last - pd.Timedelta(days=5)).strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d"),
  )
  for _, tr in cs.tail(5).iterrows():
    rows.append({
      "engine": "C",
      "strategy": "cross_sectional_reversion_long_only",
      "ticker_or_pair": tr["long_ticker"],
      "long_ticker": tr["long_ticker"],
      "short_ticker": "",
      "signal": f"BUY {tr['long_ticker']}",
      "entry_plan": "entrada proxima apertura",
      "exit_plan": "salir RSI2>50 o 5 dias",
      "stop_plan": "stop 1.5 ATR",
      "take_profit_plan": "take 2 ATR",
      "max_hold_days": 5,
      "confidence_score": 70,
      "executable_in_cash_account": True,
      "requires_short": False,
      "reason": "cross-sectional reversion",
    })
  return pd.DataFrame(rows)


latest_pairs = select_pairs_for_train(
  close_prices,
  (close_prices.index[-1] - pd.DateOffset(years=TRAIN_YEARS)).strftime("%Y-%m-%d"),
  (close_prices.index[-1] - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
  UNIVERSE, SECTOR_GROUPS,
)
current_signals = build_current_signals(close_prices, data_dict, latest_pairs)
print("Current signals:", len(current_signals))
if len(current_signals):
  print(current_signals[["engine", "strategy", "signal", "confidence_score", "executable_in_cash_account"]].to_string(index=False))
current_signals.to_csv("research_v9_current_signals.csv", index=False)

# %% [markdown]
# ## 13. Exportar

# %%
def export_csv(df, name):
  if df is None or len(df) == 0:
    pd.DataFrame().to_csv(name, index=False)
  else:
    df.to_csv(name, index=False)
  print("Exportado", name)


best_idx = engine_results["engine_score"].idxmax() if len(engine_results) else 0
best_engine = engine_results.iloc[best_idx] if len(engine_results) else pd.Series()
best_approved = bool(best_engine.get("approved_for_web_paper", False)) if len(best_engine) else False

summary_df = pd.DataFrame([{
  "lab": "v9_clean_signal",
  "best_engine": best_engine.get("engine", "none"),
  "best_strategy": best_engine.get("strategy", ""),
  "best_score": best_engine.get("engine_score", 0),
  "best_status": best_engine.get("status", "REJECTED"),
  "approved_for_web_paper": best_approved,
  "approved_for_real_money": False,
  "spy_return": round(spy_total, 2),
  "qqq_return": round(qqq_total, 2),
  "v6_blend_return": round(v6_total, 2) if pd.notna(v6_total) else np.nan,
  "v8_return": round(v8_total, 2) if pd.notna(v8_total) else np.nan,
  "trades_A": len(trades_a),
  "trades_B": len(trades_b),
  "trades_C": len(trades_c),
}])

export_csv(summary_df, "research_v9_summary.csv")
export_csv(engine_results, "research_v9_engine_results.csv")
export_csv(yearly_by_engine, "research_v9_yearly.csv")
export_csv(trades_by_engine, "research_v9_trades.csv")
export_csv(pairs_selected, "research_v9_pairs_selected.csv")
export_csv(pair_quality, "research_v9_pair_quality_by_year.csv")
export_csv(param_selection, "research_v9_parameter_selection_by_year.csv")
export_csv(current_signals, "research_v9_current_signals.csv")

equity_curves = {}
for label, trades in [("A", trades_a), ("B", trades_b), ("C", trades_c)]:
  if len(trades):
    eq = INITIAL_CAPITAL + trades.sort_values("exit_date")["pnl"].cumsum()
    equity_curves[label] = eq
if equity_curves:
  pd.DataFrame(equity_curves).to_csv("research_v9_equity_curves.csv")
  print("Exportado research_v9_equity_curves.csv")

config_out = {
  "version": "v9_clean_signal_lab",
  "engines": {
    "A": "true_pairs_long_short",
    "B": "pair_long_only_fallback",
    "C": "cross_sectional_reversion_long_only",
  },
  "best_engine": best_engine.to_dict() if len(best_engine) else {},
  "approved_for_web_paper": best_approved,
  "approved_for_real_money": False,
  "warnings": [
    "Backtest no garantiza resultados futuros.",
    "Motor A requiere cuenta con cortos.",
    "Motores B y C aplicables a cuenta cash.",
    "No mezclar resultados entre motores.",
  ],
}
Path("research_v9_selected_config.json").write_text(json.dumps(config_out, indent=2, default=str), encoding="utf-8")
print("Exportado research_v9_selected_config.json")

# %% [markdown]
# ## 14. Reporte final

# %%
print("=" * 80)
print("REPORTE FINAL V9 CLEAN SIGNAL RESEARCH LAB")
print("=" * 80)
print("Disclaimer: Backtest no garantiza resultados futuros.")
print("")
print("--- Por motor ---")
for _, row in engine_results.iterrows():
  print(f"  {row['engine']} | {row['strategy']} | return {row['total_return']}% | PF {row['profit_factor']} | "
        f"Sharpe {row['sharpe']} | trades {row['num_trades']} | {row['status']}")
print("")
print(f"Mejor motor V9: {best_engine.get('engine', 'none')} ({best_engine.get('strategy', '')}) score {best_engine.get('engine_score', 0)}")
print(f"¿Supera V8 ({v8_total if pd.notna(v8_total) else 'N/A'})? "
      f"{'SI' if safe_float(best_engine.get('total_return'), 0) > safe_float(v8_total, -999) else 'NO'}")
print(f"¿Supera V6 ({v6_total if pd.notna(v6_total) else 'N/A'})? "
      f"{'SI' if safe_float(best_engine.get('total_return'), 0) > safe_float(v6_total, -999) else 'NO'}")
cash_engines = engine_results[engine_results["engine"].isin(["B", "C"])]
best_cash = cash_engines.loc[cash_engines["engine_score"].idxmax()] if len(cash_engines) else None
print(f"¿Aplicable sin cortos? {'SI' if best_cash is not None and best_cash['engine_score'] >= 60 else 'LIMITADO'} "
      f"(mejor cash: {best_cash['engine'] if best_cash is not None else 'none'})")
print(f"Señales actuales: {len(current_signals)}")
print(f"¿Integrar web paper? {best_approved}")
print("")
if best_approved:
  print("Integrar V9 como Signal Engine Paper Trading.")
else:
  print("V9 rejected. Mantener V6 como champion y seguir research.")
print(f"APPROVED_FOR_REAL_MONEY=False (siempre)")
