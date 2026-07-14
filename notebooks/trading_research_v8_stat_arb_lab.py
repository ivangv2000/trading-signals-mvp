# %% [markdown]
# # Trading Research V8 Statistical Arbitrage Lab
#
# Pairs trading, cointegration, PCA residual y factor residual arbitrage.
#
# **Disclaimer:** Backtest no garantiza resultados futuros. No es asesoramiento financiero.
# **No broker. No HFT. APPROVED_FOR_REAL_MONEY siempre False.**

# %%
!pip install yfinance pandas numpy matplotlib plotly tqdm scipy statsmodels scikit-learn -q

# %% [markdown]
# ## 1. Configuracion

# %%
import warnings
warnings.filterwarnings("ignore")

import json
from itertools import combinations
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

try:
  from sklearn.decomposition import PCA
  from sklearn.preprocessing import StandardScaler
  HAS_SKLEARN = True
except Exception:
  HAS_SKLEARN = False

QUICK_TEST = True
START_DATE = "2015-01-01"
END_DATE = None
TRANSACTION_COST = 0.001
SLIPPAGE = 0.001
SHORT_BORROW_COST_ANNUAL = 0.03
INITIAL_CAPITAL = 10000
ALLOW_SHORT = True
LONG_ONLY_FALLBACK = True
MAX_CONCURRENT_TRADES = 5
POSITION_SIZE_PCT = 0.10
ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 3.5
MAX_HOLD_DAYS = 10
PAIR_LOOKBACK = 252
ROLLING_Z_LOOKBACK = 60
MIN_HALF_LIFE = 2
MAX_HALF_LIFE = 30
WF_START_YEAR = 2018
TOP_PAIRS = 8
PCA_N_COMPONENTS = 5
PCA_HOLD_DAYS = 5
FACTOR_HOLD_DAYS = 5
RANDOM_SEED = 42
DEBUG = False

DEFAULT_UNIVERSE = [
  "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "INTC", "MU",
  "GOOGL", "META", "AMZN", "NFLX", "ADBE", "CRM", "ORCL",
  "JPM", "BAC", "GS", "MS",
  "XOM", "CVX",
  "UNH", "LLY", "JNJ",
  "WMT", "COST", "HD", "MCD",
  "SPY", "QQQ", "IWM",
]

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU"]
FACTOR_ETFS = ["SPY", "QQQ", "IWM", "MTUM", "QUAL", "USMV", "VLUE"]

SECTOR_MAP = {
  "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK", "AVGO": "XLK", "INTC": "XLK", "MU": "XLK",
  "GOOGL": "XLK", "META": "XLK", "AMZN": "XLK", "NFLX": "XLK", "ADBE": "XLK", "CRM": "XLK", "ORCL": "XLK",
  "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "MS": "XLF",
  "XOM": "XLE", "CVX": "XLE",
  "UNH": "XLV", "LLY": "XLV", "JNJ": "XLV",
  "WMT": "XLP", "COST": "XLP", "HD": "XLI", "MCD": "XLP",
}

if QUICK_TEST:
  DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META", "AMZN", "JPM", "BAC", "SPY", "QQQ",
  ]
  START_DATE = "2020-01-01"
  WF_START_YEAR = 2021
  TOP_PAIRS = 4
  print("QUICK_TEST activo")

MARKET = "SPY"
print("V8 Statistical Arbitrage Lab | desde", START_DATE)
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


all_tickers = sorted(set(DEFAULT_UNIVERSE + SECTOR_ETFS + FACTOR_ETFS))
data_dict, close_prices = download_data(all_tickers, START_DATE, END_DATE)
print("Tickers:", len(data_dict), "| dias:", len(close_prices))

# %% [markdown]
# ## 3. Funciones estadisticas

# %%
def calculate_returns(close_prices):
  return close_prices.pct_change().fillna(0)


def calculate_log_prices(close_prices):
  return np.log(close_prices.replace(0, np.nan)).ffill()


def calculate_zscore(series, window):
  s = pd.Series(series)
  mu = s.rolling(window).mean().shift(1)
  sd = s.rolling(window).std().shift(1).replace(0, np.nan)
  return (s - mu) / sd


def prepare_ols_exog(df, factor_cols=None, expected_cols=None, add_const=True):
  """
  Prepara matriz X para statsmodels OLS/predict de forma segura.
  - Convierte a numérico
  - Reemplaza inf por NaN
  - Rellena NaN con 0
  - Añade constante siempre si add_const=True
  - Si expected_cols existe, reindexa exactamente a esas columnas
  """
  X = df.copy()

  if factor_cols is not None:
    for col in factor_cols:
      if col not in X.columns:
        X[col] = 0.0
    X = X[factor_cols]

  for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

  X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

  if expected_cols is not None:
    out = pd.DataFrame(0.0, index=X.index, columns=list(expected_cols))
    for col in X.columns:
      if col in out.columns:
        out[col] = X[col].values
    const_col = next(
      (c for c in expected_cols if str(c).lower() in ("const", "intercept")),
      None,
    )
    if const_col is not None:
      out[const_col] = 1.0
    return out.astype(float)

  if add_const:
    if HAS_STATSMODELS:
      X = sm.add_constant(X, has_constant="add")
    elif "const" not in X.columns:
      X.insert(0, "const", 1.0)

  return X.astype(float)


def safe_ols_predict(model, X_exog, expected_cols=None):
  """Predict OLS con alineacion estricta de columnas (evita shape mismatch)."""
  cols = expected_cols or list(getattr(model.model, "exog_names", []))
  if not cols and isinstance(X_exog, pd.DataFrame):
    cols = list(X_exog.columns)
  if isinstance(X_exog, pd.DataFrame):
    X_arr = X_exog.reindex(columns=cols, fill_value=0.0).to_numpy(dtype=float)
  else:
    X_arr = np.asarray(X_exog, dtype=float)
  if X_arr.ndim == 1:
    X_arr = X_arr.reshape(1, -1)
  n_params = len(model.params)
  if X_arr.shape[1] != n_params:
    raise ValueError(
      f"OLS shape mismatch: X has {X_arr.shape[1]} cols, model has {n_params} params. "
      f"expected_cols={cols}"
    )
  pred = model.predict(X_arr)
  return float(pred.iloc[0]) if hasattr(pred, "iloc") else float(np.asarray(pred).ravel()[0])


def ols_hedge_ratio(y, x):
  if not HAS_STATSMODELS:
    cov = np.cov(pd.Series(y).dropna(), pd.Series(x).dropna())
    return cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else np.nan
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
    params = model.params
    if len(params) >= 2:
      return float(params.iloc[1])
    return np.nan
  except Exception:
    return np.nan


def adf_test_pvalue(series):
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
  if len(s) < 30:
    return np.nan
  lag = s.shift(1)
  delta = s - lag
  df = pd.DataFrame({"lag": lag, "delta": delta}).dropna()
  if len(df) < 20 or not HAS_STATSMODELS:
    return np.nan
  try:
    model = sm.OLS(df["delta"], sm.add_constant(df["lag"], has_constant="add")).fit()
    beta = float(model.params.iloc[1])
    if beta >= 0:
      return np.nan
    return float(-np.log(2) / beta)
  except Exception:
    return np.nan


def calculate_hurst_exponent(series):
  s = pd.Series(series).dropna().values
  if len(s) < 100:
    return np.nan
  try:
    lags = range(2, 20)
    tau = [np.std(np.subtract(s[lag:], s[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return float(poly[0])
  except Exception:
    return np.nan


def pair_spread(log_prices, y_ticker, x_ticker, hedge_ratio):
  y = log_prices[y_ticker]
  x = log_prices[x_ticker]
  return y - hedge_ratio * x


def safe_ticker(x, default=""):
  """
  Convierte un ticker a string seguro.
  Si viene None, NaN o vacío, devuelve default.
  """
  try:
    if x is None:
      return default
    if pd.isna(x):
      return default
    s = str(x).strip().upper()
    if s in ["", "NONE", "NAN", "NULL"]:
      return default
    return s
  except Exception:
    return default


def safe_pair_name(long_ticker=None, short_ticker=None, fallback_pair=None):
  """
  Crea nombre de par robusto.
  """
  if fallback_pair is not None:
    fp = str(fallback_pair).strip()
    if fp and fp.upper() not in ["NONE", "NAN", "NULL"]:
      return fp

  long_ticker = safe_ticker(long_ticker, default="LONG")
  short_ticker = safe_ticker(short_ticker, default="SHORT")
  return f"{long_ticker}-{short_ticker}"


def validate_residual_tickers(long_ticker, short_ticker):
  """Valida tickers residual; devuelve dict normalizado o None si inválido."""
  long_t = safe_ticker(long_ticker)
  short_t = safe_ticker(short_ticker)
  if not long_t:
    return None
  if not short_t:
    if ALLOW_SHORT and not LONG_ONLY_FALLBACK:
      return None
    return {
      "long_ticker": long_t,
      "short_ticker": "",
      "trade_type": "long_only_fallback",
    }
  return {
    "long_ticker": long_t,
    "short_ticker": short_t,
    "trade_type": "residual_long_short",
  }


RESIDUAL_SIGNAL_COLS = [
  "strategy", "signal_type", "pair", "long_ticker", "short_ticker", "signal_date",
  "exit_signal_date", "entry_z", "exit_z", "exit_reason", "confidence_score",
  "entry_plan", "exit_plan", "stop_plan", "max_holding_days", "position_size_pct",
  "reason", "trade_type",
]


def normalize_residual_row(row, max_hold_days=PCA_HOLD_DAYS):
  """Asegura columnas estándar en señales residual."""
  long_t = safe_ticker(row.get("long_ticker"))
  short_t = safe_ticker(row.get("short_ticker"))
  trade_type = row.get("trade_type", "long_only_fallback")
  if not short_t:
    trade_type = "long_only_fallback"
  elif trade_type == "long_only_fallback":
    short_t = ""
  out = {
    "strategy": row.get("strategy", ""),
    "signal_type": row.get("signal_type", "RESIDUAL"),
    "pair": safe_pair_name(long_t, short_t, row.get("pair")),
    "long_ticker": long_t,
    "short_ticker": short_t,
    "signal_date": row.get("signal_date"),
    "exit_signal_date": row.get("exit_signal_date", pd.NaT),
    "entry_z": row.get("entry_z", np.nan),
    "exit_z": row.get("exit_z", np.nan),
    "exit_reason": row.get("exit_reason"),
    "confidence_score": row.get("confidence_score", 60),
    "entry_plan": row.get("entry_plan", "entrar proxima apertura"),
    "exit_plan": row.get("exit_plan", f"salir z < {EXIT_Z}"),
    "stop_plan": row.get("stop_plan", f"stop z > {STOP_Z}"),
    "max_holding_days": row.get("max_holding_days", max_hold_days),
    "position_size_pct": row.get("position_size_pct", POSITION_SIZE_PCT),
    "reason": row.get("reason", ""),
    "trade_type": trade_type,
  }
  return out

# %% [markdown]
# ## 4. Seleccion de pares (solo train)

# %%
def score_pair(log_prices, rets, y_ticker, x_ticker, train_start, train_end):
  try:
    sub = log_prices.loc[train_start:train_end, [y_ticker, x_ticker]].dropna()
    if len(sub) < PAIR_LOOKBACK // 2:
      return None
    ry = rets[y_ticker].loc[sub.index]
    rx = rets[x_ticker].loc[sub.index]
    corr = ry.corr(rx)
    if pd.isna(corr) or corr < 0.5:
      return None
    beta = ols_hedge_ratio(sub[y_ticker], sub[x_ticker])
    if pd.isna(beta):
      return None
    spread = sub[y_ticker] - beta * sub[x_ticker]
    hl = estimate_half_life(spread)
    if pd.isna(hl) or hl < MIN_HALF_LIFE or hl > MAX_HALF_LIFE:
      return None
    coint_p = cointegration_pvalue(sub[y_ticker], sub[x_ticker])
    if coint_p >= 0.10:
      return None
    spread_vol = spread.diff().std()
    if pd.isna(spread_vol) or spread_vol < 1e-4:
      return None
    rank = (1 - coint_p) * corr * min(hl, MAX_HALF_LIFE) / MAX_HALF_LIFE / max(spread_vol, 1e-4)
    return {
      "pair": f"{y_ticker}-{x_ticker}",
      "y_ticker": y_ticker,
      "x_ticker": x_ticker,
      "hedge_ratio": beta,
      "correlation": round(corr, 3),
      "coint_pvalue": round(coint_p, 4),
      "half_life": round(hl, 2),
      "spread_vol": round(spread_vol, 5),
      "rank_score": round(rank, 4),
    }
  except Exception:
    return None


def select_pairs_walk_forward(close_prices, train_start, train_end, universe=None, top_n=TOP_PAIRS):
  universe = universe or [c for c in close_prices.columns if c not in SECTOR_ETFS]
  universe = [c for c in universe if c in close_prices.columns]
  log_px = calculate_log_prices(close_prices)
  rets = calculate_returns(close_prices)
  rows = []
  for y_ticker, x_ticker in combinations(universe, 2):
    if y_ticker == x_ticker:
      continue
    for a, b in [(y_ticker, x_ticker), (x_ticker, y_ticker)]:
      rec = score_pair(log_px, rets, a, b, train_start, train_end)
      if rec:
        rows.append(rec)
  if not rows:
    return pd.DataFrame()
  df = pd.DataFrame(rows).drop_duplicates(subset=["pair"]).sort_values("rank_score", ascending=False)
  return df.head(top_n).reset_index(drop=True)

# %% [markdown]
# ## 5. Estrategia 1: Pairs trading clasico

# %%
def generate_pair_signals(pair_row, close_prices, train_end, test_start, test_end):
  y_ticker = pair_row["y_ticker"]
  x_ticker = pair_row["x_ticker"]
  beta = pair_row["hedge_ratio"]
  log_px = calculate_log_prices(close_prices)
  spread = pair_spread(log_px, y_ticker, x_ticker, beta)
  z = calculate_zscore(spread, ROLLING_Z_LOOKBACK)
  test = z.loc[test_start:test_end].dropna()
  rows = []
  position = None
  for dt, zval in test.items():
    if pd.isna(zval):
      continue
    if position is None:
      if zval > ENTRY_Z:
        position = {
          "entry_date": dt, "entry_z": zval, "long_ticker": x_ticker, "short_ticker": y_ticker,
          "direction": "short_spread",
        }
      elif zval < -ENTRY_Z:
        position = {
          "entry_date": dt, "entry_z": zval, "long_ticker": y_ticker, "short_ticker": x_ticker,
          "direction": "long_spread",
        }
    else:
      hold = (dt - position["entry_date"]).days
      exit_reason = None
      if abs(zval) < EXIT_Z:
        exit_reason = "z_exit"
      elif abs(zval) > STOP_Z:
        exit_reason = "z_stop"
      elif hold >= MAX_HOLD_DAYS:
        exit_reason = "max_hold"
      elif position["direction"] == "long_spread" and zval > ENTRY_Z:
        exit_reason = "z_cross"
      elif position["direction"] == "short_spread" and zval < -ENTRY_Z:
        exit_reason = "z_cross"
      if exit_reason:
        use_short = ALLOW_SHORT
        trade_type = "pairs_long_short" if use_short else "long_only_fallback"
        long_t = safe_ticker(position["long_ticker"])
        short_t = safe_ticker(position["short_ticker"]) if use_short else ""
        if not long_t:
          position = None
          continue
        conf = int(np.clip(50 + abs(position["entry_z"]) * 12, 0, 100))
        rows.append({
          "strategy": "pairs_trading",
          "signal_type": "PAIR",
          "pair": safe_pair_name(long_t, short_t, pair_row.get("pair")),
          "signal_date": position["entry_date"],
          "exit_signal_date": dt,
          "long_ticker": long_t,
          "short_ticker": short_t,
          "entry_z": round(position["entry_z"], 2),
          "exit_z": round(zval, 2),
          "exit_reason": exit_reason,
          "max_holding_days": MAX_HOLD_DAYS,
          "trade_type": trade_type,
          "confidence_score": conf,
          "hedge_ratio": beta,
          "entry_plan": "entrar proxima apertura",
          "exit_plan": f"salir z < {EXIT_Z}",
          "stop_plan": f"stop z > {STOP_Z}",
          "position_size_pct": POSITION_SIZE_PCT,
          "reason": f"z-entry {position['entry_z']:.1f} mean-reversion pair",
        })
        position = None
  return pd.DataFrame(rows)

# %% [markdown]
# ## 6. Modo LONG_ONLY

# %%
def apply_long_only_policy(signal_row):
  if isinstance(signal_row, pd.Series):
    row = signal_row.to_dict()
  else:
    row = dict(signal_row)
  if ALLOW_SHORT and not LONG_ONLY_FALLBACK:
    return pd.Series(row) if isinstance(signal_row, pd.Series) else row
  row["short_ticker"] = ""
  row["pair"] = safe_pair_name(row.get("long_ticker"), "", row.get("pair"))
  row["trade_type"] = "long_only_fallback"
  row["position_size_pct"] = POSITION_SIZE_PCT * 0.6
  row["reason"] = str(row.get("reason", "")) + " | long-only fallback sin short"
  return pd.Series(row) if isinstance(signal_row, pd.Series) else row

# %% [markdown]
# ## 7. Estrategia 2: PCA residual stat arb

# %%
def pca_residual_stat_arb(close_prices, universe, train_start, train_end, test_start, test_end):
  if not HAS_SKLEARN:
    return pd.DataFrame()
  uni = [c for c in universe if c in close_prices.columns and c not in ("SPY", "QQQ")]
  if len(uni) < 5:
    return pd.DataFrame()
  rets = calculate_returns(close_prices[uni])
  train = rets.loc[train_start:train_end].dropna(how="all")
  test = rets.loc[test_start:test_end].dropna(how="all")
  if len(train) < 120 or len(test) < 20:
    return pd.DataFrame()
  scaler = StandardScaler()
  X_train = scaler.fit_transform(train.fillna(0))
  n_comp = min(PCA_N_COMPONENTS, X_train.shape[1] - 1, X_train.shape[0] - 1)
  if n_comp < 2:
    return pd.DataFrame()
  pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
  pca.fit(X_train)
  rows = []
  position = {}
  for dt in test.index:
    x = scaler.transform(test.loc[[dt]].fillna(0))
    reconstructed = pca.inverse_transform(pca.transform(x))
    residual = test.loc[dt].values - reconstructed[0]
    resid_s = pd.Series(residual, index=uni)
    z = (resid_s - resid_s.mean()) / (resid_s.std() if resid_s.std() > 0 else 1)
    longs = z.nsmallest(2).index.tolist()
    shorts = z.nlargest(2).index.tolist() if ALLOW_SHORT else []
    for t in list(position.keys()):
      hold = (dt - position[t]["entry"]).days
      if hold >= PCA_HOLD_DAYS or abs(z.get(t, 0)) < EXIT_Z:
        rows.append(_residual_trade_row("pca_residual", dt, position[t], z.get(t, 0), "pca_exit", PCA_HOLD_DAYS))
        del position[t]
    slots = MAX_CONCURRENT_TRADES - len(position)
    for t in longs[:slots]:
      if t not in position and z[t] < -ENTRY_Z:
        validated = validate_residual_tickers(t, None)
        if validated is None:
          continue
        position[t] = {
          "entry": dt, "side": "long", "entry_z": z[t],
          "long_ticker": validated["long_ticker"], "short_ticker": validated["short_ticker"],
          "trade_type": validated["trade_type"],
        }
        rows.append(normalize_residual_row(_residual_signal_row(
          "pca_residual", dt, validated["long_ticker"], validated["short_ticker"] or None,
          z[t], "long residual extremo", validated["trade_type"], PCA_HOLD_DAYS,
        ), PCA_HOLD_DAYS))
  return pd.DataFrame([r for r in rows if r and (r.get("long_ticker") or r.get("exit_reason"))])


def _residual_signal_row(strategy, dt, long_t, short_t, zval, reason, trade_type=None, max_hold=PCA_HOLD_DAYS):
  long_t = safe_ticker(long_t)
  short_t = safe_ticker(short_t)
  if not long_t:
    return {}
  if not short_t:
    trade_type = trade_type or "long_only_fallback"
  else:
    trade_type = trade_type or ("residual_long_short" if ALLOW_SHORT else "long_only_fallback")
    if not ALLOW_SHORT:
      short_t = ""
      trade_type = "long_only_fallback"
  return {
    "strategy": strategy,
    "signal_type": "RESIDUAL",
    "pair": safe_pair_name(long_t, short_t),
    "signal_date": dt,
    "exit_signal_date": pd.NaT,
    "long_ticker": long_t,
    "short_ticker": short_t,
    "entry_z": round(zval, 2),
    "exit_z": np.nan,
    "exit_reason": None,
    "max_holding_days": max_hold,
    "trade_type": trade_type,
    "confidence_score": int(np.clip(55 + abs(zval) * 10, 0, 100)),
    "entry_plan": "entrar proxima apertura",
    "exit_plan": f"salir z < {EXIT_Z}",
    "stop_plan": f"stop z > {STOP_Z}",
    "position_size_pct": POSITION_SIZE_PCT,
    "reason": reason,
  }


def _residual_trade_row(strategy, exit_dt, pos, exit_z, reason, max_hold=PCA_HOLD_DAYS):
  long_ticker = safe_ticker(pos.get("long_ticker"), default="")
  short_ticker = safe_ticker(pos.get("short_ticker"), default="")
  pair_name = safe_pair_name(
    long_ticker=long_ticker,
    short_ticker=short_ticker,
    fallback_pair=pos.get("pair", None),
  )
  trade_type = pos.get("trade_type", "long_only_fallback")
  if not short_ticker:
    trade_type = "long_only_fallback"
  return normalize_residual_row({
    "strategy": strategy,
    "signal_type": "RESIDUAL",
    "pair": pair_name,
    "signal_date": pos["entry"],
    "exit_signal_date": exit_dt,
    "long_ticker": long_ticker,
    "short_ticker": short_ticker,
    "entry_z": round(pos.get("entry_z", 0), 2),
    "exit_z": round(exit_z, 2) if pd.notna(exit_z) else np.nan,
    "exit_reason": reason,
    "max_holding_days": pos.get("max_holding_days", max_hold),
    "trade_type": trade_type,
    "confidence_score": 60,
    "reason": reason,
  }, max_hold)

# %% [markdown]
# ## 8. Estrategia 3: Factor residual arbitrage

# %%
def factor_residual_stat_arb(close_prices, universe, train_start, train_end, test_start, test_end):
  if not HAS_STATSMODELS:
    return pd.DataFrame()
  factors = [f for f in FACTOR_ETFS if f in close_prices.columns]
  stocks = [s for s in universe if s in close_prices.columns and s not in factors and s not in SECTOR_ETFS]
  if len(factors) < 2 or len(stocks) < 3:
    return pd.DataFrame()
  factor_returns = calculate_returns(close_prices)
  stock_returns = factor_returns.copy()
  rows = []
  position = {}
  test_dates = factor_returns.loc[test_start:test_end].index
  for stock in stocks:
    try:
      fac_cols = list(factors)
      sec = SECTOR_MAP.get(stock)
      if sec and sec in close_prices.columns and sec not in fac_cols:
        fac_cols.append(sec)
      available_factors = [f for f in fac_cols if f in factor_returns.columns]
      if len(available_factors) < 2:
        continue

      X_train_raw = factor_returns.loc[train_start:train_end, available_factors]
      X_train = prepare_ols_exog(
        X_train_raw,
        factor_cols=available_factors,
        expected_cols=None,
        add_const=True,
      )
      y_train = stock_returns.loc[train_start:train_end, stock]
      common_idx = X_train.index.intersection(y_train.dropna().index)
      X_train = X_train.loc[common_idx]
      y_train = y_train.loc[common_idx]
      if len(X_train) < 60:
        continue

      model = sm.OLS(y_train, X_train).fit()
      expected_cols = list(X_train.columns)
      resid_hist = y_train - safe_ols_predict(model, X_train, expected_cols)
      resid_std = resid_hist.std()
      if resid_std <= 0:
        continue

      for dt in test_dates:
        try:
          if dt not in factor_returns.index:
            continue
          if stock not in factor_returns.columns:
            continue
          X_pred_raw = factor_returns.loc[[dt], available_factors]
          if X_pred_raw.isna().any().any():
            continue
          X_pred = prepare_ols_exog(
            X_pred_raw,
            factor_cols=available_factors,
            expected_cols=expected_cols,
            add_const=True,
          )
          pred = safe_ols_predict(model, X_pred, expected_cols)
          actual = float(factor_returns.loc[dt, stock])
          if not np.isfinite(pred) or not np.isfinite(actual):
            continue
          resid = actual - pred
          z = resid / resid_std
          if stock in position:
            hold = (dt - position[stock]["entry"]).days
            if hold >= FACTOR_HOLD_DAYS or abs(z) < EXIT_Z:
              rows.append(_residual_trade_row("factor_residual", dt, position[stock], z, "factor_exit", FACTOR_HOLD_DAYS))
              del position[stock]
          if stock not in position:
            if z < -ENTRY_Z:
              validated = validate_residual_tickers(stock, None)
              if validated is None:
                continue
              position[stock] = {
                "entry": dt, "entry_z": z,
                "long_ticker": validated["long_ticker"], "short_ticker": validated["short_ticker"],
                "trade_type": validated["trade_type"],
              }
              rows.append(normalize_residual_row(_residual_signal_row(
                "factor_residual", dt, validated["long_ticker"], validated["short_ticker"] or None,
                z, "residual factor negativo", validated["trade_type"], FACTOR_HOLD_DAYS,
              ), FACTOR_HOLD_DAYS))
        except Exception as e:
          if DEBUG:
            print(f"Factor residual predict failed for {stock} @ {dt}: {e}")
          continue
    except Exception as e:
      if DEBUG:
        print(f"Factor residual failed for {stock}: {e}")
      continue
  return pd.DataFrame([r for r in rows if r and (r.get("long_ticker") or r.get("exit_reason"))])

# %% [markdown]
# ## 9. Backtest event-driven

# %%
def _exec_price(data_dict, ticker, dt, prefer_open=True):
  df = data_dict.get(ticker)
  if df is None or dt not in df.index:
    return np.nan
  row = df.loc[dt]
  if prefer_open and "Open" in row.index and pd.notna(row["Open"]):
    return float(row["Open"])
  return float(row["Close"])


def backtest_trades_from_signals(signals_df, data_dict, close_prices, initial_capital=INITIAL_CAPITAL):
  if signals_df is None or len(signals_df) == 0:
    return {}, pd.DataFrame(), pd.Series(dtype=float)

  cost_side = TRANSACTION_COST + SLIPPAGE
  borrow_daily = SHORT_BORROW_COST_ANNUAL / 252
  cash = initial_capital
  open_trades = []
  closed = []
  equity = []

  sigs = signals_df.copy()
  sigs["signal_date"] = pd.to_datetime(sigs["signal_date"])
  if "exit_signal_date" in sigs.columns:
    sigs["exit_signal_date"] = pd.to_datetime(sigs["exit_signal_date"], errors="coerce")

  completed = sigs[sigs["exit_signal_date"].notna()].copy()
  if "exit_reason" in completed.columns:
    completed = completed[completed["exit_reason"].notna()]

  for _, sig in completed.iterrows():
    try:
      entry_dt = pd.Timestamp(sig["signal_date"])
      exit_dt = pd.Timestamp(sig["exit_signal_date"])
      long_t = safe_ticker(sig.get("long_ticker"))
      short_t = safe_ticker(sig.get("short_ticker"))
      trade_type = sig.get("trade_type", "pairs_long_short")

      if not long_t:
        continue
      if long_t not in data_dict:
        continue
      if short_t and short_t not in data_dict:
        if LONG_ONLY_FALLBACK:
          short_t = ""
          trade_type = "long_only_fallback"
        else:
          continue

      size_pct = float(sig.get("position_size_pct", POSITION_SIZE_PCT))
      if isinstance(sig.get("position_size_pct"), str):
        try:
          size_pct = float(str(sig.get("position_size_pct")).replace("%", "")) / 100
        except Exception:
          size_pct = POSITION_SIZE_PCT
      if trade_type == "long_only_fallback":
        size_pct *= 0.6
      alloc = initial_capital * size_pct

      future = close_prices.index[close_prices.index > entry_dt]
      if len(future) == 0:
        continue
      entry_exec = future[0]
      future_x = close_prices.index[close_prices.index > exit_dt]
      exit_exec = future_x[0] if len(future_x) else close_prices.index[-1]

      long_px0 = _exec_price(data_dict, long_t, entry_exec) if long_t else np.nan
      long_px1 = _exec_price(data_dict, long_t, exit_exec, prefer_open=False) if long_t else np.nan
      short_px0 = _exec_price(data_dict, short_t, entry_exec) if short_t else np.nan
      short_px1 = _exec_price(data_dict, short_t, exit_exec, prefer_open=False) if short_t else np.nan

      ret_parts = []
      if pd.notna(long_px0) and pd.notna(long_px1) and long_px0 > 0:
        ret_parts.append((long_px1 / long_px0 - 1) - 2 * cost_side)
      if pd.notna(short_px0) and pd.notna(short_px1) and short_px0 > 0 and short_t and ALLOW_SHORT:
        hold_days = max((exit_exec - entry_exec).days, 1)
        ret_parts.append((short_px0 / short_px1 - 1) - 2 * cost_side - borrow_daily * hold_days)

      if not ret_parts:
        continue
      trade_ret = np.mean(ret_parts)
      pnl = alloc * trade_ret
      cash += pnl
      closed.append({
        "strategy": sig.get("strategy"),
        "pair": safe_pair_name(long_t, short_t, sig.get("pair")),
        "long_ticker": long_t,
        "short_ticker": short_t,
        "entry_date": entry_exec,
        "exit_date": exit_exec,
        "entry_z": sig.get("entry_z"),
        "exit_z": sig.get("exit_z"),
        "exit_reason": sig.get("exit_reason"),
        "return_pct": round(trade_ret * 100, 3),
        "pnl": round(pnl, 2),
        "holding_days": (exit_exec - entry_exec).days,
        "trade_type": trade_type,
        "confidence_score": sig.get("confidence_score"),
      })
    except Exception:
      continue

  trades_df = pd.DataFrame(closed)
  if len(trades_df):
    cum = initial_capital + trades_df["pnl"].cumsum()
    eq_idx = trades_df["exit_date"].values
    equity = pd.Series(cum.values, index=pd.DatetimeIndex(eq_idx))
  else:
    equity = pd.Series([initial_capital], index=[close_prices.index[-1]])

  metrics = compute_stat_arb_metrics(trades_df, equity, close_prices, initial_capital)
  return metrics, trades_df, equity


def compute_stat_arb_metrics(trades_df, equity, close_prices, initial_capital):
  if equity is None or len(equity) < 2:
    return {"num_trades": 0, "total_return": 0, "sharpe": 0, "profit_factor": 0}
  eq = equity.sort_index().astype(float)
  rets = eq.pct_change().fillna(0)
  total_ret = eq.iloc[-1] / initial_capital - 1
  years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1 / 365.25)
  cagr = (eq.iloc[-1] / initial_capital) ** (1 / years) - 1
  peak = eq.cummax()
  dd = (eq - peak) / peak.replace(0, np.nan)
  mdd = dd.min() if len(dd) else 0
  sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
  downside = rets[rets < 0].std()
  sortino = rets.mean() / downside * np.sqrt(252) if downside and downside > 0 else 0
  wins = trades_df[trades_df["return_pct"] > 0] if len(trades_df) else pd.DataFrame()
  losses = trades_df[trades_df["return_pct"] <= 0] if len(trades_df) else pd.DataFrame()
  win_rate = len(wins) / len(trades_df) if len(trades_df) else 0
  gross_win = wins["pnl"].sum() if len(wins) else 0
  gross_loss = abs(losses["pnl"].sum()) if len(losses) else 0
  pf = gross_win / gross_loss if gross_loss > 0 else 0
  avg_ret = trades_df["return_pct"].mean() if len(trades_df) else 0
  beta_spy = 0.0
  if MARKET in close_prices.columns and len(rets) > 20:
    spy = calculate_returns(close_prices[MARKET]).reindex(rets.index).fillna(0)
    if spy.std() > 0:
      beta_spy = rets.cov(spy) / spy.var()
  neutrality = 1 - min(abs(beta_spy), 1)
  return {
    "total_return": round(total_ret * 100, 2),
    "CAGR": round(cagr * 100, 2),
    "sharpe": round(sharpe, 3),
    "sortino": round(sortino, 3),
    "max_drawdown": round(mdd * 100, 2),
    "calmar": round(cagr / abs(mdd), 3) if mdd != 0 else 0,
    "num_trades": int(len(trades_df)),
    "win_rate": round(win_rate * 100, 2),
    "avg_trade_return": round(avg_ret, 3),
    "median_trade_return": round(trades_df["return_pct"].median(), 3) if len(trades_df) else 0,
    "best_trade": round(trades_df["return_pct"].max(), 3) if len(trades_df) else 0,
    "worst_trade": round(trades_df["return_pct"].min(), 3) if len(trades_df) else 0,
    "profit_factor": round(pf, 3),
    "expectancy": round(trades_df["pnl"].mean(), 2) if len(trades_df) else 0,
    "avg_holding_days": round(trades_df["holding_days"].mean(), 1) if len(trades_df) else 0,
    "exposure_pct": round(POSITION_SIZE_PCT * MAX_CONCURRENT_TRADES * 100, 1),
    "beta_to_SPY": round(beta_spy, 3),
    "market_neutrality_score": round(neutrality * 100, 1),
  }

# %% [markdown]
# ## 10a. Test safe_ticker y safe_pair_name

# %%
test_pos = {
  "long_ticker": None,
  "short_ticker": "MSFT",
  "pair": None,
  "entry": pd.Timestamp("2024-01-01"),
}
print(safe_pair_name(test_pos.get("long_ticker"), test_pos.get("short_ticker"), test_pos.get("pair")))

test_pos_2 = {
  "long_ticker": "AAPL",
  "short_ticker": None,
  "pair": None,
  "entry": pd.Timestamp("2024-01-01"),
}
print(safe_pair_name(test_pos_2.get("long_ticker"), test_pos_2.get("short_ticker"), test_pos_2.get("pair")))

test_row = _residual_trade_row(
  "pca_residual",
  pd.Timestamp("2024-06-01"),
  {"long_ticker": None, "short_ticker": "MSFT", "entry": pd.Timestamp("2024-01-01"), "entry_z": -2.1},
  -0.4,
  "test_exit",
)
print("trade row pair:", test_row.get("pair"))
assert test_row.get("pair"), "safe_pair_name fallo en trade row"
print("OK: safe_ticker y safe_pair_name")

# %% [markdown]
# ## 10b. Test prepare_ols_exog

# %%
if HAS_STATSMODELS:
  test_train = pd.DataFrame({
    "SPY": [0.01, 0.02, -0.01],
    "QQQ": [0.02, 0.01, -0.02],
  })
  test_pred = pd.DataFrame({
    "SPY": [0.01],
  })
  X_train = prepare_ols_exog(test_train, factor_cols=["SPY", "QQQ"], add_const=True)
  model = sm.OLS(pd.Series([0.01, 0.02, 0.00]), X_train).fit()
  expected_cols = list(X_train.columns)
  X_pred = prepare_ols_exog(
    test_pred,
    factor_cols=["SPY", "QQQ"],
    expected_cols=expected_cols,
    add_const=True,
  )
  print("Train cols:", X_train.columns.tolist())
  print("Pred cols:", X_pred.columns.tolist())
  print("Predict:", safe_ols_predict(model, X_pred, expected_cols))
  assert X_train.shape[1] == X_pred.shape[1]
  assert X_pred.shape[1] == len(model.params)
  print("OK: prepare_ols_exog alinea train/predict")
else:
  print("SKIP OLS test: statsmodels no disponible")

# %% [markdown]
# ## 10. Walk-forward completo

# %%
def run_stat_arb_walk_forward(close_prices, data_dict, start_year=WF_START_YEAR):
  yearly_rows, all_trades, all_signals, pairs_log = [], [], [], []
  current_year = pd.Timestamp.today().year
  capital = INITIAL_CAPITAL

  for year in tqdm(range(start_year, current_year + 1), desc="WF Stat Arb"):
    test_start = f"{year}-01-01"
    test_end = f"{year}-12-31"
    train_end = f"{year - 1}-12-31"
    train_start = close_prices.index[0].strftime("%Y-%m-%d")
    if pd.Timestamp(train_end) <= pd.Timestamp(train_start):
      continue

    year_signals = []

    pairs = select_pairs_walk_forward(close_prices, train_start, train_end)
    if len(pairs):
      pairs = pairs.assign(year=year)
      pairs_log.append(pairs)
      for _, pr in pairs.iterrows():
        ps = generate_pair_signals(pr, close_prices, train_end, test_start, test_end)
        if len(ps):
          ps = ps.apply(apply_long_only_policy, axis=1)
          year_signals.append(ps)

    # pca
    pca_s = pca_residual_stat_arb(close_prices, DEFAULT_UNIVERSE, train_start, train_end, test_start, test_end)
    if len(pca_s):
      year_signals.append(pca_s)

    # factor
    fac_s = factor_residual_stat_arb(close_prices, DEFAULT_UNIVERSE, train_start, train_end, test_start, test_end)
    if len(fac_s):
      year_signals.append(fac_s)

    if not year_signals:
      continue
    signals_year = pd.concat(year_signals, ignore_index=True)
    all_signals.append(signals_year)

    metrics, trades, eq = backtest_trades_from_signals(signals_year, data_dict, close_prices, initial_capital=capital)
    if len(trades):
      all_trades.append(trades.assign(year=year))
      capital = INITIAL_CAPITAL + trades["pnl"].sum()

    spy_slice = close_prices[MARKET].loc[test_start:test_end]
    spy_ret = (spy_slice.iloc[-1] / spy_slice.iloc[0] - 1) * 100 if len(spy_slice) >= 2 else 0
    qqq_ret = 0
    if "QQQ" in close_prices.columns:
      q = close_prices["QQQ"].loc[test_start:test_end]
      qqq_ret = (q.iloc[-1] / q.iloc[0] - 1) * 100 if len(q) >= 2 else 0

    yearly_rows.append({
      "year": year,
      "return": metrics.get("total_return", 0),
      "SPY_return": round(spy_ret, 2),
      "QQQ_return": round(qqq_ret, 2),
      "num_trades": metrics.get("num_trades", 0),
      "win_rate": metrics.get("win_rate", 0),
      "profit_factor": metrics.get("profit_factor", 0),
      "max_drawdown": metrics.get("max_drawdown", 0),
      "sharpe": metrics.get("sharpe", 0),
      "beats_spy": metrics.get("total_return", 0) > spy_ret,
      "num_pairs": len(pairs) if len(pairs) else 0,
    })

  yearly_df = pd.DataFrame(yearly_rows)
  trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
  signals_df = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
  pairs_selected_df = pd.concat(pairs_log, ignore_index=True) if pairs_log else pd.DataFrame()

  if len(trades_df):
    equity_oos = INITIAL_CAPITAL + trades_df.sort_values("exit_date")["pnl"].cumsum()
    equity_oos.index = trades_df.sort_values("exit_date")["exit_date"].values
  else:
    equity_oos = pd.Series(dtype=float)

  return yearly_df, trades_df, signals_df, pairs_selected_df, equity_oos


yearly_df, trades_df, signals_df, pairs_selected_df, equity_oos = run_stat_arb_walk_forward(
  close_prices, data_dict, WF_START_YEAR
)
print("WF años:", len(yearly_df), "| trades:", len(trades_df), "| pares seleccionados:", len(pairs_selected_df))

# %% [markdown]
# ## 11. Aprobacion stat_arb_score

# %%
def compute_stat_arb_score(yearly_df, trades_df, metrics):
  if yearly_df is None or len(yearly_df) == 0:
    return 0, "REJECTED", ["sin walk-forward"]
  score = 0
  notes = []
  if metrics.get("sharpe", 0) > 1:
    score += 20
  elif metrics.get("sharpe", 0) > 0.5:
    score += 10
  if metrics.get("profit_factor", 0) > 1.2:
    score += 15
  elif metrics.get("profit_factor", 0) > 1.0:
    score += 8
  else:
    notes.append("profit_factor bajo")
  if metrics.get("max_drawdown", -100) > -20:
    score += 15
  elif metrics.get("max_drawdown", -100) > -30:
    score += 8
  else:
    score -= 20
    notes.append("drawdown alto")
  if metrics.get("num_trades", 0) >= 100:
    score += 10
  elif metrics.get("num_trades", 0) < 50:
    score -= 20
    notes.append("pocos trades")
  if metrics.get("win_rate", 0) > 52:
    score += 10
  cost_thresh = (TRANSACTION_COST + SLIPPAGE) * 100 * 3
  if metrics.get("avg_trade_return", 0) > cost_thresh:
    score += 10
  if abs(metrics.get("beta_to_SPY", 1)) < 0.3:
    score += 10
  if yearly_df["beats_spy"].mean() >= 0.60:
    score += 10
  recent = yearly_df[yearly_df["year"].isin([2025, 2026])]
  if len(recent) > 0 and (recent["return"] <= 0).any():
    score -= 20
    notes.append("pierde años recientes")
  if len(trades_df):
    pair_counts = trades_df["pair"].value_counts()
    if len(pair_counts) and pair_counts.iloc[0] / len(trades_df) > 0.5:
      score -= 20
      notes.append("depende de un solo par")
  score = int(np.clip(score, 0, 100))
  if score >= 75:
    status = "APPROVED_FOR_WEB_PAPER"
  elif score >= 60:
    status = "CANDIDATE"
  else:
    status = "REJECTED"
  return score, status, notes


full_metrics, _, _ = backtest_trades_from_signals(signals_df, data_dict, close_prices) if len(signals_df) else ({}, pd.DataFrame(), pd.Series())
stat_arb_score, stat_arb_status, score_notes = compute_stat_arb_score(yearly_df, trades_df, full_metrics)
APPROVED_FOR_WEB_PAPER = stat_arb_status == "APPROVED_FOR_WEB_PAPER"
APPROVED_FOR_REAL_MONEY = False
print(f"stat_arb_score: {stat_arb_score}/100 | {stat_arb_status}")
if score_notes:
  print("Notas:", "; ".join(score_notes))

# %% [markdown]
# ## 12. Señales actuales (Current Stat Arb Signals)

# %%
def current_pair_signals(close_prices, pairs_df):
  if pairs_df is None or len(pairs_df) == 0:
    return pd.DataFrame()
  last = close_prices.index[-1]
  train_end = last - pd.Timedelta(days=30)
  train_start = close_prices.index[0]
  test_start = (last - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
  test_end = last.strftime("%Y-%m-%d")
  rows = []
  for _, pr in pairs_df.head(TOP_PAIRS).iterrows():
    log_px = calculate_log_prices(close_prices)
    spread = pair_spread(log_px, pr["y_ticker"], pr["x_ticker"], pr["hedge_ratio"])
    z = calculate_zscore(spread, ROLLING_Z_LOOKBACK)
    zval = z.iloc[-1] if len(z) else np.nan
    if pd.isna(zval):
      continue
    if zval > ENTRY_Z:
      long_t, short_t = pr["x_ticker"], pr["y_ticker"]
      sig = "SHORT_SPREAD"
    elif zval < -ENTRY_Z:
      long_t, short_t = pr["y_ticker"], pr["x_ticker"]
      sig = "LONG_SPREAD"
    else:
      continue
    use_short = ALLOW_SHORT
    rows.append({
      "strategy": "pairs_trading",
      "signal_type": "PAIR",
      "pair": safe_pair_name(long_t, short_t if use_short else "", pr["pair"]),
      "long_ticker": long_t,
      "short_ticker": short_t if use_short else "",
      "entry_z": round(zval, 2),
      "confidence_score": int(np.clip(50 + abs(zval) * 12, 0, 100)),
      "entry_plan": "entrar proxima apertura",
      "exit_plan": f"salir z < {EXIT_Z}",
      "stop_plan": f"stop z > {STOP_Z}",
      "max_holding_days": MAX_HOLD_DAYS,
      "position_size_pct": f"{POSITION_SIZE_PCT*100:.0f}%",
      "trade_type": "pairs_long_short" if use_short else "long_only_fallback",
      "reason": f"{sig} z={zval:.1f} mean-reversion",
    })
  return pd.DataFrame(rows)


latest_pairs = select_pairs_walk_forward(
  close_prices,
  close_prices.index[0].strftime("%Y-%m-%d"),
  (close_prices.index[-1] - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
)
current_pair = current_pair_signals(close_prices, latest_pairs)
current_pca = pca_residual_stat_arb(
  close_prices, DEFAULT_UNIVERSE,
  close_prices.index[0].strftime("%Y-%m-%d"),
  (close_prices.index[-1] - pd.Timedelta(days=120)).strftime("%Y-%m-%d"),
  (close_prices.index[-1] - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
  close_prices.index[-1].strftime("%Y-%m-%d"),
)
current_factor = factor_residual_stat_arb(
  close_prices, DEFAULT_UNIVERSE,
  close_prices.index[0].strftime("%Y-%m-%d"),
  (close_prices.index[-1] - pd.Timedelta(days=120)).strftime("%Y-%m-%d"),
  (close_prices.index[-1] - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
  close_prices.index[-1].strftime("%Y-%m-%d"),
)

def format_current(df):
  if df is None or len(df) == 0:
    return pd.DataFrame()
  out = df.copy()
  for col in ["entry_plan", "exit_plan", "stop_plan", "position_size_pct"]:
    if col not in out.columns:
      out[col] = "-"
  return out


current_signals = pd.concat([
  format_current(current_pair),
  format_current(current_pca.tail(5) if len(current_pca) else current_pca),
  format_current(current_factor.tail(5) if len(current_factor) else current_factor),
], ignore_index=True)

print("Current Stat Arb Signals |", close_prices.index[-1].date())
if len(current_signals):
  cols = [c for c in [
    "strategy", "signal_type", "pair", "long_ticker", "short_ticker", "entry_z",
    "confidence_score", "entry_plan", "exit_plan", "stop_plan", "max_holding_days",
    "position_size_pct", "trade_type", "reason",
  ] if c in current_signals.columns]
  print(current_signals[cols].to_string(index=False))
else:
  print("Sin señales activas en z-threshold actual.")

current_signals.to_csv("research_v8_current_stat_arb_signals.csv", index=False)

# %% [markdown]
# ## 13. Comparacion benchmarks

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

spy_total = (close_prices[MARKET].iloc[-1] / close_prices[MARKET].iloc[0] - 1) * 100
qqq_total = (close_prices["QQQ"].iloc[-1] / close_prices["QQQ"].iloc[0] - 1) * 100 if "QQQ" in close_prices else 0

# %% [markdown]
# ## 14. Exportar

# %%
def export_csv(df, name):
  if df is None or len(df) == 0:
    pd.DataFrame().to_csv(name, index=False)
  else:
    df.to_csv(name, index=False)
  print("Exportado", name)


summary_df = pd.DataFrame([{
  "strategy_name": "V8 Statistical Arbitrage Lab",
  "stat_arb_score": stat_arb_score,
  "status": stat_arb_status,
  "approved_for_web_paper": APPROVED_FOR_WEB_PAPER,
  "approved_for_real_money": False,
  **full_metrics,
  "pct_years_beats_spy": round(yearly_df["beats_spy"].mean() * 100, 1) if len(yearly_df) else 0,
  "spy_buyhold_return": round(spy_total, 2),
  "qqq_buyhold_return": round(qqq_total, 2),
  "v6_blend_return": round(v6_total, 2) if pd.notna(v6_total) else np.nan,
  "allow_short": ALLOW_SHORT,
  "long_only_fallback": LONG_ONLY_FALLBACK,
}])

export_csv(summary_df, "research_v8_summary.csv")
export_csv(pairs_selected_df, "research_v8_pairs_selected.csv")
export_csv(trades_df, "research_v8_trades.csv")
export_csv(yearly_df, "research_v8_yearly.csv")
export_csv(current_signals, "research_v8_current_stat_arb_signals.csv")
if len(equity_oos):
  equity_oos.to_frame("equity").to_csv("research_v8_equity_curve.csv")
  print("Exportado research_v8_equity_curve.csv")
else:
  pd.DataFrame().to_csv("research_v8_equity_curve.csv", index=False)

config_out = {
  "version": "v8_stat_arb_lab",
  "strategy_name": "V8 Statistical Arbitrage",
  "approved_for_web_paper": APPROVED_FOR_WEB_PAPER,
  "approved_for_real_money": False,
  "stat_arb_score": stat_arb_score,
  "status": stat_arb_status,
  "rules": {
    "entry_z": ENTRY_Z,
    "exit_z": EXIT_Z,
    "stop_z": STOP_Z,
    "max_hold_days": MAX_HOLD_DAYS,
    "allow_short": ALLOW_SHORT,
    "long_only_fallback": LONG_ONLY_FALLBACK,
    "pair_selection": "cointegration + half-life on train only",
    "strategies": ["pairs_trading", "pca_residual", "factor_residual"],
  },
  "warnings": [
    "Backtest no garantiza resultados futuros.",
    "No es asesoramiento financiero.",
    "No aprobado para dinero real.",
    "No HFT ni market making.",
    "Shorts incluyen borrow cost estimado.",
  ],
  "metrics": full_metrics,
  "how_to_use": (
    "Revisar research_v8_current_stat_arb_signals.csv. "
    "PAIR: long una pata y short otra (o long-only fallback). "
    "Entrada proxima apertura. Salida por z-score, stop o max hold."
  ),
}
Path("research_v8_selected_strategy_config.json").write_text(json.dumps(config_out, indent=2, default=str), encoding="utf-8")
print("Exportado research_v8_selected_strategy_config.json")

# %% [markdown]
# ## 15. Reporte final

# %%
print("=" * 80)
print("REPORTE FINAL V8 STATISTICAL ARBITRAGE LAB")
print("=" * 80)
print("Disclaimer: Backtest no garantiza resultados futuros.")
print("")
print(f"1. ¿Supera a SPY? {'SI' if full_metrics.get('total_return', 0) > spy_total else 'NO'} "
      f"({full_metrics.get('total_return', 0):.2f}% vs SPY {spy_total:.2f}%)")
print(f"2. ¿Supera a QQQ? {'SI' if full_metrics.get('total_return', 0) > qqq_total else 'NO'} "
      f"({full_metrics.get('total_return', 0):.2f}% vs QQQ {qqq_total:.2f}%)")
if pd.notna(v6_total):
  print(f"3. ¿Supera a V6? {'SI' if full_metrics.get('total_return', 0) > v6_total else 'NO'} "
        f"({full_metrics.get('total_return', 0):.2f}% vs V6 {v6_total:.2f}%)")
else:
  print("3. ¿Supera a V6? No disponible (falta equity V6)")
print(f"4. Operaciones: {full_metrics.get('num_trades', 0)}")
print(f"5. Profit factor: {full_metrics.get('profit_factor', 0)}")
print(f"6. Drawdown: {full_metrics.get('max_drawdown', 0)}%")
if len(yearly_df):
  pos_years = (yearly_df['return'] > 0).mean() * 100
  print(f"7. Años positivos: {pos_years:.0f}% ({len(yearly_df)} años test)")
else:
  print("7. Años: sin datos walk-forward")
print(f"8. Web paper: {APPROVED_FOR_WEB_PAPER} ({stat_arb_status}, score {stat_arb_score}/100)")
print(f"9. Señales actuales: {len(current_signals)}")
viable_lo = "SI (long_only_fallback)" if LONG_ONLY_FALLBACK else ("NO" if ALLOW_SHORT else "SI")
print(f"10. ¿Viable sin short? {viable_lo}")
print("")
if APPROVED_FOR_WEB_PAPER:
  print("Integrar como V8 Stat Arb Paper Trading.")
else:
  print("No integrar todavía. Seguir investigando.")
print(f"APPROVED_FOR_REAL_MONEY={APPROVED_FOR_REAL_MONEY} (siempre False)")
