# %% [markdown]
# # Trading Research V6 Professional Audit Lab
#
# Professional audit notebook for V5 results and V6 candidate strategies.
#
# **Disclaimer:** Backtest no garantiza resultados futuros. No es asesoramiento financiero.
# **APPROVED_FOR_REAL_MONEY siempre es False.**

# %%
!pip install yfinance pandas numpy matplotlib plotly tqdm scikit-learn scipy -q

# %% [markdown]
# ## 1. Configuracion

# %%
import warnings
warnings.filterwarnings("ignore")

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tqdm.auto import tqdm

try:
  from scipy import stats
  HAS_SCIPY = True
except Exception:
  stats = None
  HAS_SCIPY = False

try:
  from sklearn.pipeline import make_pipeline
  from sklearn.impute import SimpleImputer
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import Ridge
  from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
  HAS_SKLEARN = True
except Exception:
  HAS_SKLEARN = False

QUICK_TEST = False
# ML experimental puede tardar y romperse si hay datos incompletos.
# Por defecto lo dejamos desactivado hasta que el resto del notebook esté validado.
RUN_ML_EXPERIMENTAL = False
START_DATE = "2010-01-01"
END_DATE = None
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001
SLIPPAGE = 0.001
N_BOOTSTRAP = 1000
WF_START_YEAR = 2017
RANDOM_SEED = 42

RISKY_ASSETS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN"]
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
FACTOR_ETFS = ["MTUM", "QUAL", "USMV", "VLUE", "SIZE"]
ETF_ONLY_UNIVERSE = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "MTUM", "QUAL", "USMV", "VLUE", "SHY", "IEF", "TLT", "GLD"]
GLOBAL_TACTICAL_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "SHY", "GLD", "DBC", "VNQ", "UUP"]
DEFENSIVE_ASSETS = ["SHY", "IEF", "TLT", "GLD", "CASH"]
MACRO_TICKERS = ["^VIX", "^TNX", "HYG", "LQD", "UUP"]
FULL_UNIVERSE = sorted(set(RISKY_ASSETS + SECTOR_ETFS + FACTOR_ETFS + ETF_ONLY_UNIVERSE + GLOBAL_TACTICAL_UNIVERSE + DEFENSIVE_ASSETS))

MARKET = "SPY"
QQQ = "QQQ"
IEF = "IEF"

CHAMPION_PARAMS = dict(
  fast_ma=50,
  slow_ma=200,
  vol_target=0.15,
  max_asset_weight=0.30,
  defensive_asset="SHY",
  rebalance_freq="W-FRI",
)

if QUICK_TEST:
  START_DATE = "2018-01-01"
  N_BOOTSTRAP = 100
  RISKY_ASSETS = ["SPY", "QQQ", "NVDA", "AMD", "AAPL"]
  SECTOR_ETFS = ["XLK", "XLF", "XLV"]
  FACTOR_ETFS = ["MTUM", "USMV"]
  ETF_ONLY_UNIVERSE = ["SPY", "QQQ", "IWM", "XLK", "XLV", "MTUM", "USMV", "SHY", "IEF", "GLD"]
  GLOBAL_TACTICAL_UNIVERSE = ["SPY", "QQQ", "EFA", "EEM", "IEF", "SHY", "GLD", "UUP"]
  FULL_UNIVERSE = sorted(set(RISKY_ASSETS + SECTOR_ETFS + FACTOR_ETFS + ETF_ONLY_UNIVERSE + GLOBAL_TACTICAL_UNIVERSE + DEFENSIVE_ASSETS))
  WF_START_YEAR = 2019
  print("QUICK_TEST activo")

print("V6 Professional Audit Lab | desde", START_DATE)
print("Backtest no garantiza resultados futuros.")

# %% [markdown]
# ## 2. Cargar artefactos V5 si existen

# %%
def load_v5_artifacts(base_path="."):
  base = Path(base_path)
  artifacts = {}
  for path in sorted(base.glob("research_v5_*.csv")):
    try:
      artifacts[path.name] = pd.read_csv(path)
    except Exception as exc:
      print(f"WARNING: no se pudo leer {path.name}: {exc}")
  for path in sorted(base.glob("research_v5_*.json")):
    try:
      artifacts[path.name] = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
      print(f"WARNING: no se pudo leer {path.name}: {exc}")
  print(f"Artefactos V5 encontrados: {len(artifacts)}")
  return artifacts


V5_ARTIFACTS = load_v5_artifacts(".")
if V5_ARTIFACTS:
  print("V5:", ", ".join(V5_ARTIFACTS.keys()))
else:
  print("No se encontraron research_v5_*.csv/json en el directorio actual.")

# %% [markdown]
# ## 3. Datos

# %%
def canonical_ticker(ticker):
  if ticker == "^VIX":
    return "VIX"
  if ticker == "^TNX":
    return "TNX"
  return ticker.upper()


def download_data(tickers, start, end=None):
  data, failed = {}, []
  for ticker in tqdm(sorted(set(tickers)), desc="Download"):
    if ticker == "CASH":
      continue
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
      data[canonical_ticker(ticker)] = df.sort_index()
    except Exception as exc:
      failed.append(f"{ticker}({exc})")
  if failed:
    print("Fallidos:", failed)
  close = pd.DataFrame({k: v["Close"] for k, v in data.items()}).sort_index().ffill()
  close.index = pd.DatetimeIndex(close.index)
  if close.index.tz:
    close.index = close.index.tz_localize(None)
  close["CASH"] = 1.0
  data["CASH"] = pd.DataFrame({"Close": 1.0}, index=close.index)
  return data, close


all_tickers = list(set(FULL_UNIVERSE + MACRO_TICKERS + [MARKET, QQQ, IEF]))
data, close_prices = download_data(all_tickers, START_DATE, END_DATE)
print("Columnas descargadas:", list(close_prices.columns))

# %% [markdown]
# ## 4. Features

# %%
def _rsi(close, window):
  delta = close.diff()
  gain = delta.clip(lower=0).rolling(window).mean()
  loss = (-delta.clip(upper=0)).rolling(window).mean()
  return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def add_features(df):
  d = df.copy()
  c = d["Close"]
  r1 = c.pct_change(1)
  for n in [1, 5, 20, 60, 120, 252]:
    d[f"RET_{n}D"] = r1 if n == 1 else c.pct_change(n)
  for n in [20, 50, 100, 200]:
    d[f"SMA_{n}"] = c.rolling(n).mean()
  for span in [21, 50, 100]:
    d[f"EMA_{span}"] = c.ewm(span=span, adjust=False).mean()
  for n in [20, 60, 120]:
    d[f"VOL_{n}"] = r1.rolling(n).std() * np.sqrt(252)
  eq = (1 + r1.fillna(0)).cumprod()
  d["ROLLING_DD_60"] = (eq - eq.rolling(60).max()) / eq.rolling(60).max()
  d["ROLLING_DD_120"] = (eq - eq.rolling(120).max()) / eq.rolling(120).max()
  for n in [20, 60, 120, 252]:
    d[f"MOM_{n}"] = c.pct_change(n)
  d["MOM_COMBO"] = 0.25 * d["MOM_20"] + 0.35 * d["MOM_60"] + 0.40 * d["MOM_120"]
  d["MOM_COMBO_VOL"] = d["MOM_COMBO"] / d["VOL_60"].replace(0, np.nan)
  d["RSI_2"] = _rsi(c, 2)
  d["RSI_14"] = _rsi(c, 14)
  if "High" in d.columns and "Low" in d.columns:
    hl = (d["High"] - d["Low"]).replace(0, np.nan)
    d["IBS"] = (c - d["Low"]) / hl
  d["DIST_SMA200"] = c / d["SMA_200"] - 1
  if "Volume" in d.columns:
    d["VOLUME_AVG_20"] = d["Volume"].rolling(20).mean()
    d["VOLUME_RATIO"] = d["Volume"] / d["VOLUME_AVG_20"].replace(0, np.nan)
  d["forward_return_20d"] = c.pct_change(20).shift(-20)
  return d


features_dict = {k: add_features(v) for k, v in data.items()}

# %% [markdown]
# ## 5. Utilidades de fechas y pesos

# %%
def norm_idx(ix):
  ix = pd.DatetimeIndex(ix)
  return ix.tz_localize(None) if ix.tz else ix


def align_date(ix, dt):
  ix, dt = norm_idx(ix), pd.Timestamp(dt)
  if dt.tz:
    dt = dt.tz_localize(None)
  if dt in ix:
    return dt
  loc = ix.get_indexer([dt], method="pad")[0]
  return ix[loc] if loc >= 0 else None


def idx_pos(ix, dt):
  aligned = align_date(ix, dt)
  return ix.get_loc(aligned) if aligned is not None else None


def safe_float(x, default=np.nan):
  try:
    if x is None:
      return default
    if isinstance(x, pd.Series):
      if len(x) == 0:
        return default
      x = x.iloc[0]
    elif isinstance(x, pd.DataFrame):
      if x.empty:
        return default
      x = x.iloc[-1, 0]
    value = float(x)
    if not np.isfinite(value):
      return default
    return value
  except Exception:
    return default


def row_asof(df, date):
  if df is None or getattr(df, "empty", True):
    return pd.Series(dtype=float)
  if not isinstance(df.index, pd.DatetimeIndex):
    try:
      df = df.copy()
      df.index = pd.DatetimeIndex(df.index)
    except Exception:
      return pd.Series(dtype=float)
  date = pd.Timestamp(date)
  if date.tzinfo is not None:
    date = date.tz_localize(None)
  idx = df.index[df.index <= date]
  if len(idx) == 0:
    return pd.Series(dtype=float)
  row = df.loc[idx[-1]]
  if isinstance(row, pd.DataFrame):
    row = row.iloc[-1]
  return row


def clean_score_series(scores):
  """Convierte dict de scores a Serie numerica segura para nlargest."""
  if not scores:
    return pd.Series(dtype=float)
  s = pd.Series(scores)
  s = pd.to_numeric(s.apply(safe_float), errors="coerce")
  s = s.replace([np.inf, -np.inf], np.nan).dropna()
  return s


def defensive_weight_row(wr):
  wr = pd.Series(0.0, index=wr.index) if not isinstance(wr, pd.Series) else wr.copy()
  if "SHY" in wr.index:
    wr["SHY"] = 1.0
  elif "CASH" in wr.index:
    wr["CASH"] = 1.0
  return wr


def reb_dates(ix, freq="W-FRI"):
  ix = norm_idx(ix)
  s = pd.Series(np.arange(len(ix)), index=ix)
  return pd.DatetimeIndex([g.index[-1] for _, g in s.groupby(pd.Grouper(freq=freq)) if len(g)])


def next_rebalance_pos(index, rlist, i):
  if i + 1 < len(rlist):
    pos = idx_pos(index, rlist[i + 1])
    return pos if pos is not None else len(index) - 1
  return len(index) - 1


def assign_w(wdf, i0, i1, wrow):
  i1 = min(int(i1), len(wdf) - 1)
  v = wrow.reindex(wdf.columns).fillna(0).values
  for j in range(int(i0) + 1, i1 + 1):
    wdf.iloc[j] = v


def normalize_weight_row(w, cap=None):
  w = pd.Series(w).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
  if cap is not None and cap > 0:
    for _ in range(10):
      if w.sum() <= 0:
        break
      w = w / w.sum()
      over = w > cap
      if not over.any():
        break
      excess = (w[over] - cap).sum()
      w[over] = cap
      under = ~over
      if under.any() and w[under].sum() > 0:
        w[under] = w[under] + excess * w[under] / w[under].sum()
  return w / w.sum() if w.sum() > 0 else w


def inv_vol_w(tickers, vols, cap=0.30):
  vols = pd.Series(vols).replace(0, np.nan).dropna()
  tickers = [t for t in tickers if t in vols.index and pd.notna(vols[t])]
  if not tickers:
    return pd.Series(dtype=float)
  raw = 1 / vols[tickers].clip(lower=0.01)
  return normalize_weight_row(raw, cap=cap)


def vol_scale(wdf, close, target=0.15, lb=20):
  cols = [c for c in wdf.columns if c in close.columns]
  r = close[cols].pct_change().fillna(0)
  pr = (wdf[cols].shift(1).fillna(0) * r).sum(axis=1)
  rv = pr.rolling(lb).std() * np.sqrt(252)
  scale = (target / rv.replace(0, np.nan)).clip(0, 1).shift(1).fillna(1)
  return wdf.mul(scale, axis=0)

# %% [markdown]
# ## 6. Metricas recalculadas desde curvas de equity

# %%
def calculate_returns_from_equity(equity):
  equity = pd.Series(equity).dropna().astype(float)
  returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
  return returns


def calculate_drawdown(equity):
  equity = pd.Series(equity).dropna().astype(float)
  if equity.empty:
    return pd.Series(dtype=float)
  peak = equity.cummax()
  return (equity - peak) / peak.replace(0, np.nan)


def calculate_cagr(equity):
  equity = pd.Series(equity).dropna().astype(float)
  if len(equity) < 2 or equity.iloc[0] <= 0:
    return 0.0
  years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25) if isinstance(equity.index, pd.DatetimeIndex) else len(equity) / 252
  return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def calculate_sharpe(returns, risk_free=0.0):
  returns = pd.Series(returns).dropna().astype(float)
  if returns.std() <= 0 or len(returns) < 2:
    return 0.0
  excess = returns - risk_free / 252
  return excess.mean() / returns.std() * np.sqrt(252)


def calculate_sortino(returns, risk_free=0.0):
  returns = pd.Series(returns).dropna().astype(float)
  downside = returns[returns < risk_free / 252]
  ds = downside.std()
  if ds <= 0 or len(returns) < 2:
    return 0.0
  excess = returns - risk_free / 252
  return excess.mean() / ds * np.sqrt(252)


def calculate_calmar(equity):
  cagr = calculate_cagr(equity)
  dd = calculate_drawdown(equity)
  mdd = abs(dd.min()) if len(dd) else 0
  return cagr / mdd if mdd > 0 else 0.0


def calculate_var_cvar(returns, alpha=0.05):
  returns = pd.Series(returns).dropna().astype(float)
  if returns.empty:
    return 0.0, 0.0
  var = returns.quantile(alpha)
  cvar = returns[returns <= var].mean() if (returns <= var).any() else var
  return var, cvar


def metrics_from_returns(returns, equity=None, initial_capital=INITIAL_CAPITAL):
  returns = pd.Series(returns).replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
  if equity is None:
    equity = (1 + returns).cumprod() * initial_capital
  else:
    equity = pd.Series(equity).dropna().astype(float)
  dd = calculate_drawdown(equity)
  cagr = calculate_cagr(equity)
  vol = returns.std() * np.sqrt(252)
  sharpe = calculate_sharpe(returns)
  sortino = calculate_sortino(returns)
  var95, cvar95 = calculate_var_cvar(returns, 0.05)
  total_return = equity.iloc[-1] / equity.iloc[0] - 1 if len(equity) >= 2 and equity.iloc[0] != 0 else 0
  max_dd = dd.min() if len(dd) else 0
  return {
    "total_return": round(total_return * 100, 2),
    "CAGR": round(cagr * 100, 2),
    "annual_volatility": round(vol * 100, 2),
    "sharpe": round(sharpe, 3),
    "sortino": round(sortino, 3),
    "max_drawdown": round(max_dd * 100, 2),
    "calmar": round(cagr / abs(max_dd), 3) if max_dd != 0 else 0,
    "best_day": round(returns.max() * 100, 3),
    "worst_day": round(returns.min() * 100, 3),
    "positive_days_pct": round((returns > 0).mean() * 100, 2),
    "skew": round(returns.skew(), 3),
    "var_95": round(var95 * 100, 3),
    "cvar_95": round(cvar95 * 100, 3),
  }


def audit_v5_metrics(v5_artifacts, tolerance=0.05):
  rows = []
  summary_candidates = []
  equity_candidates = []
  for name, obj in v5_artifacts.items():
    if not isinstance(obj, pd.DataFrame):
      continue
    cols = {str(c).lower(): c for c in obj.columns}
    if "equity" in name.lower() or any(k in cols for k in ["equity", "equity_curve", "portfolio_value"]):
      equity_candidates.append((name, obj))
    if any(k in cols for k in ["strategy", "cagr", "sharpe", "max_drawdown"]):
      summary_candidates.append((name, obj))
  for eq_name, eq_df in equity_candidates:
    date_col = next((c for c in eq_df.columns if str(c).lower() in ["date", "datetime", "index"]), None)
    work = eq_df.copy()
    if date_col is not None:
      work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
      work = work.dropna(subset=[date_col]).set_index(date_col)
    for col in work.columns:
      eq = pd.to_numeric(work[col], errors="coerce").dropna()
      if len(eq) < 20:
        continue
      recalculated = metrics_from_returns(calculate_returns_from_equity(eq), eq)
      strategy = str(col) if len(work.columns) > 1 else eq_name.replace("research_v5_", "").replace(".csv", "")
      matched = None
      for summary_name, summary_df in summary_candidates:
        if "strategy" not in summary_df.columns:
          continue
        hit = summary_df[summary_df["strategy"].astype(str).str.lower() == strategy.lower()]
        if len(hit):
          matched = (summary_name, hit.iloc[0])
          break
      status = "RECALCULATED_ONLY"
      mismatch_fields = []
      if matched is not None:
        for metric in ["total_return", "CAGR", "sharpe", "max_drawdown", "calmar"]:
          if metric in matched[1].index and metric in recalculated:
            old = pd.to_numeric(pd.Series([matched[1][metric]]), errors="coerce").iloc[0]
            new = recalculated[metric]
            if pd.notna(old) and abs(old - new) > tolerance:
              mismatch_fields.append(metric)
        status = "WARNING_MISMATCH" if mismatch_fields else "OK"
      rows.append({
        "equity_file": eq_name,
        "strategy": strategy,
        "status": status,
        "mismatch_fields": ",".join(mismatch_fields),
        **recalculated,
      })
      if mismatch_fields:
        print(f"WARNING: V5 metric mismatch {strategy}: {mismatch_fields}")
  return pd.DataFrame(rows)


v5_metric_audit_df = audit_v5_metrics(V5_ARTIFACTS)
if len(v5_metric_audit_df):
  print(v5_metric_audit_df[["strategy", "status", "mismatch_fields", "CAGR", "sharpe", "max_drawdown"]].to_string(index=False))
else:
  print("No se encontraron curvas de equity V5 auditables.")

# %% [markdown]
# ## 7. Benchmarks y backtest robusto

# %%
def benchmark_returns(close):
  r = close.pct_change().fillna(0)
  ew_cols = [c for c in FULL_UNIVERSE if c in close.columns and c != "CASH"]
  spy = r[MARKET] if MARKET in r else pd.Series(0, index=r.index)
  qqq = r[QQQ] if QQQ in r else spy
  ief = r[IEF] if IEF in r else pd.Series(0, index=r.index)
  gld = r["GLD"] if "GLD" in r else pd.Series(0, index=r.index)
  shy = r["SHY"] if "SHY" in r else pd.Series(0, index=r.index)
  return {
    "SPY": spy,
    "QQQ": qqq,
    "EW": r[ew_cols].mean(axis=1) if ew_cols else spy,
    "6040": 0.6 * spy + 0.4 * ief,
    "DEF": 0.5 * ief + 0.25 * gld + 0.25 * shy,
  }


BENCHMARKS = benchmark_returns(close_prices)


def backtest_portfolio_weights(close, weights, transaction_cost=TRANSACTION_COST, slippage=SLIPPAGE, initial_capital=INITIAL_CAPITAL, benchmarks=None):
  close = close.copy().sort_index()
  close.index = norm_idx(close.index)
  r = close.pct_change().fillna(0)
  cols = [c for c in weights.columns if c in r.columns]
  W = weights[cols].copy()
  W.index = norm_idx(W.index)
  W = W.reindex(close.index).ffill().fillna(0)
  row_sum = W.abs().sum(axis=1).replace(0, np.nan)
  W = W.div(row_sum.where(row_sum > 1, 1), axis=0).fillna(0)
  executed_w = W.shift(1).fillna(0)
  turnover = W.diff().abs().sum(axis=1).fillna(0)
  gross_returns = (executed_w * r[cols]).sum(axis=1)
  total_cost = turnover * (transaction_cost + slippage)
  portfolio_returns = gross_returns - total_cost
  equity = (1 + portfolio_returns).cumprod() * initial_capital
  drawdown = calculate_drawdown(equity)
  metrics = metrics_from_returns(portfolio_returns, equity, initial_capital)
  metrics.update({
    "turnover_avg": round(turnover.mean(), 4),
    "turnover_ann": round(turnover.mean() * 252, 3),
    "total_cost_pct": round(total_cost.sum() * 100, 2),
    "exposure_pct": round(executed_w.abs().sum(axis=1).mean() * 100, 2),
    "days": int(len(portfolio_returns)),
  })
  bms = benchmarks if benchmarks is not None else BENCHMARKS
  for bm_name, bm_ret in bms.items():
    br = bm_ret.reindex(portfolio_returns.index).fillna(0)
    bm_total = (1 + br).prod() - 1
    strat_total = equity.iloc[-1] / initial_capital - 1 if len(equity) else 0
    metrics[f"excess_vs_{bm_name.lower()}"] = round((strat_total - bm_total) * 100, 2)
  return metrics, portfolio_returns, equity, drawdown, W

# %% [markdown]
# ## 8. Estrategias V6

# %%
def champion_trend_following_v4(close, features_dict, universe=None, fast_ma=50, slow_ma=200, vol_target=0.15, max_asset_weight=0.30, defensive_asset="SHY", rebalance_freq="W-FRI"):
  universe = universe or RISKY_ASSETS
  cols = [c for c in universe if c in close.columns and c != "CASH"]
  wdf = pd.DataFrame(0.0, index=norm_idx(close.index), columns=close.columns)
  rlist = list(reb_dates(wdf.index, rebalance_freq))
  slow_col = f"SMA_{slow_ma}"
  fast_col = f"EMA_{fast_ma}" if fast_ma in [21, 50, 100] else "EMA_50"
  for i, rd in enumerate(rlist):
    i0 = idx_pos(wdf.index, rd)
    if i0 is None:
      continue
    rd = wdf.index[i0]
    i1 = next_rebalance_pos(wdf.index, rlist, i)
    elig, vols = [], {}
    for t in cols:
      row = row_asof(features_dict.get(t, pd.DataFrame()), rd)
      if row.empty:
        continue
      sma = safe_float(row.get(slow_col, np.nan))
      ema = safe_float(row.get(fast_col, np.nan))
      px = safe_float(row.get("Close", np.nan))
      if pd.notna(sma) and pd.notna(ema) and pd.notna(px) and px > sma and ema > sma:
        elig.append(t)
        vols[t] = safe_float(row.get("VOL_60", np.nan))
    wr = pd.Series(0.0, index=wdf.columns)
    if elig:
      for t, wt in inv_vol_w(elig, vols, max_asset_weight).items():
        wr[t] = wt
    else:
      wr[defensive_asset if defensive_asset in wr.index else "CASH"] = 1.0
    assign_w(wdf, i0, i1, wr)
  return vol_scale(wdf, close, vol_target)


def cs_momentum_weights(close, features_dict, universe, top_n=3, freq="W-FRI", cap=0.30):
  cols = [c for c in universe if c in close.columns and c != "CASH"]
  wdf = pd.DataFrame(0.0, index=norm_idx(close.index), columns=close.columns)
  rlist = list(reb_dates(wdf.index, freq))
  for i, rd in enumerate(rlist):
    i0 = idx_pos(wdf.index, rd)
    if i0 is None:
      continue
    rd = wdf.index[i0]
    i1 = next_rebalance_pos(wdf.index, rlist, i)
    scores = {}
    for t in cols:
      row = row_asof(features_dict.get(t, pd.DataFrame()), rd)
      if row.empty:
        continue
      val = safe_float(row.get("MOM_COMBO_VOL", np.nan))
      if pd.notna(val):
        scores[t] = val
    wr = pd.Series(0.0, index=wdf.columns)
    s = clean_score_series(scores)
    s = s[s > 0]
    if s.empty:
      wr = defensive_weight_row(wr)
    else:
      s = s.nlargest(top_n)
      vols = {}
      for t in s.index:
        r2 = row_asof(features_dict.get(t, pd.DataFrame()), rd)
        if not r2.empty:
          vols[t] = safe_float(r2.get("VOL_60", np.nan))
      for t, wt in inv_vol_w(list(s.index), vols, cap).items():
        wr[t] = wt
    assign_w(wdf, i0, i1, wr)
  return wdf


def defensive_weights(close, features_dict, cap=0.40):
  cols = [c for c in DEFENSIVE_ASSETS if c in close.columns and c != "CASH"]
  wdf = pd.DataFrame(0.0, index=norm_idx(close.index), columns=close.columns)
  rlist = list(reb_dates(wdf.index, "M"))
  for i, rd in enumerate(rlist):
    i0 = idx_pos(wdf.index, rd)
    if i0 is None:
      continue
    rd = wdf.index[i0]
    i1 = next_rebalance_pos(wdf.index, rlist, i)
    scores = {}
    for t in cols:
      row = row_asof(features_dict.get(t, pd.DataFrame()), rd)
      if row.empty:
        continue
      val = safe_float(row.get("MOM_60", np.nan))
      if pd.notna(val):
        scores[t] = val
    wr = pd.Series(0.0, index=wdf.columns)
    s = clean_score_series(scores)
    if s.empty:
      wr = defensive_weight_row(wr)
    else:
      s = s.nlargest(min(3, len(s)))
      s = s - min(safe_float(s.min(), 0), 0) + 0.01
      for t, wt in normalize_weight_row(s, cap=cap).items():
        wr[t] = wt
    assign_w(wdf, i0, i1, wr)
  return wdf


def risk_on_score(features_dict, close, dt):
  score = 0
  for ticker in [MARKET, QQQ]:
    row = row_asof(features_dict.get(ticker, pd.DataFrame()), dt)
    if not row.empty:
      if safe_float(row.get("Close", 0)) > safe_float(row.get("SMA_200", np.inf)):
        score += 1
      if safe_float(row.get("MOM_60", -1)) > 0:
        score += 1
  spy = row_asof(features_dict.get(MARKET, pd.DataFrame()), dt)
  if not spy.empty:
    if safe_float(spy.get("ROLLING_DD_60", 0)) < -0.10:
      score -= 1
    if safe_float(spy.get("VOL_20", 0)) > 0.25:
      score -= 1
  vix = row_asof(features_dict.get("VIX", pd.DataFrame()), dt)
  if not vix.empty and pd.notna(safe_float(vix.get("SMA_50", np.nan))):
    if safe_float(vix.get("Close", 999)) < safe_float(vix.get("SMA_50", 0)):
      score += 1
  hyg = row_asof(features_dict.get("HYG", pd.DataFrame()), dt)
  lqd = row_asof(features_dict.get("LQD", pd.DataFrame()), dt)
  if not hyg.empty and not lqd.empty:
    if safe_float(hyg.get("MOM_60", 0)) > safe_float(lqd.get("MOM_60", 0)):
      score += 1
  return score


def adaptive_ensemble(close, features_dict):
  w_tf = champion_trend_following_v4(close, features_dict, RISKY_ASSETS, **CHAMPION_PARAMS)
  w_cs = cs_momentum_weights(close, features_dict, RISKY_ASSETS)
  w_etf = cs_momentum_weights(close, features_dict, SECTOR_ETFS + FACTOR_ETFS)
  w_def = defensive_weights(close, features_dict)
  out = pd.DataFrame(0.0, index=norm_idx(close.index), columns=close.columns)
  rlist = list(reb_dates(out.index, "W-FRI"))
  prev = None
  for i, rd in enumerate(rlist):
    i0 = idx_pos(out.index, rd)
    if i0 is None:
      continue
    rd = out.index[i0]
    i1 = next_rebalance_pos(out.index, rlist, i)
    ros = risk_on_score(features_dict, close, rd)
    if ros >= 4:
      mix = 0.45 * w_tf.loc[rd] + 0.35 * w_cs.loc[rd] + 0.15 * w_etf.loc[rd] + 0.05 * w_def.loc[rd]
      cap_risky = 1.0
    elif ros >= 2:
      mix = 0.25 * w_tf.loc[rd] + 0.25 * w_cs.loc[rd] + 0.25 * w_etf.loc[rd] + 0.25 * w_def.loc[rd]
      cap_risky = 0.6
    else:
      mix = 0.10 * w_tf.loc[rd] + 0.10 * w_cs.loc[rd] + 0.10 * w_etf.loc[rd] + 0.70 * w_def.loc[rd]
      cap_risky = 0.3
    mix = mix.clip(lower=0)
    risky = [c for c in mix.index if c not in DEFENSIVE_ASSETS and c != "CASH"]
    risky_sum = mix[risky].sum()
    if risky_sum > cap_risky and risky_sum > 0:
      mix[risky] = mix[risky] * (cap_risky / risky_sum)
      mix["SHY" if "SHY" in mix.index else "CASH"] += 1 - mix.sum()
    mix = normalize_weight_row(mix, cap=0.30)
    if prev is not None and mix.sum() > 0:
      mix = normalize_weight_row(0.7 * mix + 0.3 * prev, cap=0.30)
    prev = mix
    assign_w(out, i0, i1, mix)
  return out


def defensive_growth(close, features_dict, alpha=0.35):
  growth = [c for c in ["QQQ", "XLK", "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN"] if c in close.columns]
  defense = [c for c in ["SHY", "IEF", "TLT", "GLD", "XLP", "XLU", "XLV", "USMV"] if c in close.columns]
  wdf = pd.DataFrame(0.0, index=norm_idx(close.index), columns=close.columns)
  rlist = list(reb_dates(wdf.index, "M"))
  prev = None
  for i, rd in enumerate(rlist):
    i0 = idx_pos(wdf.index, rd)
    if i0 is None:
      continue
    rd = wdf.index[i0]
    i1 = next_rebalance_pos(wdf.index, rlist, i)
    growth_score = 0
    defense_score = 0
    for t in ["QQQ", "XLK", "SPY"]:
      row = row_asof(features_dict.get(t, pd.DataFrame()), rd)
      if not row.empty:
        growth_score += safe_float(row.get("MOM_60", 0)) + safe_float(row.get("MOM_120", 0))
        if safe_float(row.get("Close", 0)) > safe_float(row.get("SMA_200", 0)):
          growth_score += 0.05
    spy = row_asof(features_dict.get(MARKET, pd.DataFrame()), rd)
    if not spy.empty:
      defense_score += max(0, -safe_float(spy.get("ROLLING_DD_120", 0))) * 2
      if safe_float(spy.get("VOL_20", 0)) > 0.22:
        defense_score += 0.2
    for t in ["SHY", "IEF", "GLD"]:
      row = row_asof(features_dict.get(t, pd.DataFrame()), rd)
      if not row.empty:
        defense_score += safe_float(row.get("MOM_60", 0))
    basket = growth if growth_score >= defense_score else defense
    pct = 0.85 if abs(growth_score - defense_score) > 0.1 else 0.55
    scores = {}
    for t in basket:
      row = row_asof(features_dict.get(t, pd.DataFrame()), rd)
      if row.empty:
        continue
      val = safe_float(row.get("MOM_COMBO_VOL", np.nan))
      if pd.isna(val):
        val = safe_float(row.get("MOM_60", np.nan))
      if pd.notna(val):
        scores[t] = val
    wr = pd.Series(0.0, index=wdf.columns)
    s = clean_score_series(scores)
    if s.empty:
      wr = defensive_weight_row(wr)
    else:
      s = s.nlargest(min(3, len(s)))
      for t in s.index:
        wr[t] = pct / len(s)
      safe_col = "SHY" if "SHY" in wr.index else "CASH"
      wr[safe_col] = wr.get(safe_col, 0) + (1 - pct)
    if prev is not None:
      wr = alpha * wr + (1 - alpha) * prev
    wr = normalize_weight_row(wr, cap=0.40)
    prev = wr
    assign_w(wdf, i0, i1, wr)
  return wdf


def factor_rotation(close, features_dict):
  universe = [c for c in FACTOR_ETFS + [MARKET, QQQ, "IWM", "SHY", "IEF", "GLD"] if c in close.columns]
  return cs_momentum_weights(close, features_dict, universe, top_n=3, freq="M", cap=0.40)


def risk_parity(close, features_dict, universe=None, cap=0.30):
  universe = universe or FULL_UNIVERSE
  cols = [c for c in universe if c in close.columns and c != "CASH"]
  wdf = pd.DataFrame(0.0, index=norm_idx(close.index), columns=close.columns)
  rlist = list(reb_dates(wdf.index, "M"))
  for i, rd in enumerate(rlist):
    i0 = idx_pos(wdf.index, rd)
    if i0 is None:
      continue
    rd = wdf.index[i0]
    i1 = next_rebalance_pos(wdf.index, rlist, i)
    elig, vols = [], {}
    for t in cols:
      row = row_asof(features_dict.get(t, pd.DataFrame()), rd)
      if row.empty:
        continue
      mom = safe_float(row.get("MOM_60", np.nan))
      if pd.notna(mom) and (mom > 0 or t in DEFENSIVE_ASSETS):
        elig.append(t)
        vols[t] = safe_float(row.get("VOL_60", np.nan))
    wr = pd.Series(0.0, index=wdf.columns)
    if elig:
      for t, wt in inv_vol_w(elig, vols, cap=cap).items():
        wr[t] = wt
    else:
      wr["CASH"] = 1.0
    assign_w(wdf, i0, i1, wr)
  return wdf


def blended_champion_weights(close, features_dict, alpha=0.8):
  w_v4 = champion_trend_following_v4(close, features_dict, RISKY_ASSETS, **CHAMPION_PARAMS)
  w_adaptive = adaptive_ensemble(close, features_dict)
  mix = alpha * w_v4.reindex(close.index).fillna(0) + (1 - alpha) * w_adaptive.reindex(close.index).fillna(0)
  return mix.apply(lambda row: normalize_weight_row(row, cap=0.35), axis=1)


def dynamic_champion_switcher_weights(close, features_dict, lookback=126):
  w_v4 = champion_trend_following_v4(close, features_dict, RISKY_ASSETS, **CHAMPION_PARAMS)
  w_ad = adaptive_ensemble(close, features_dict)
  r = close.pct_change().fillna(0)
  cols = [c for c in close.columns if c in w_v4.columns]
  ret_v4 = (w_v4[cols].shift(1).fillna(0) * r[cols]).sum(axis=1)
  ret_ad = (w_ad[cols].shift(1).fillna(0) * r[cols]).sum(axis=1)
  score_v4 = ret_v4.rolling(lookback).mean() / ret_v4.rolling(lookback).std().replace(0, np.nan)
  score_ad = ret_ad.rolling(lookback).mean() / ret_ad.rolling(lookback).std().replace(0, np.nan)
  out = pd.DataFrame(0.0, index=norm_idx(close.index), columns=close.columns)
  rlist = list(reb_dates(out.index, "M"))
  prev = None
  for i, rd in enumerate(rlist):
    i0 = idx_pos(out.index, rd)
    if i0 is None:
      continue
    rd = out.index[i0]
    i1 = next_rebalance_pos(out.index, rlist, i)
    if pd.notna(score_ad.loc[rd]) and score_ad.loc[rd] > score_v4.loc[rd]:
      wr = w_ad.loc[rd].copy()
    else:
      wr = w_v4.loc[rd].copy()
    if prev is not None:
      wr = 0.8 * wr + 0.2 * prev
    wr = normalize_weight_row(wr, cap=0.35)
    prev = wr
    assign_w(out, i0, i1, wr)
  return out


def etf_only_robust_portfolio(close, features_dict):
  universe = [c for c in ETF_ONLY_UNIVERSE if c in close.columns]
  w_mom = cs_momentum_weights(close, features_dict, universe, top_n=4, freq="W-FRI", cap=0.25)
  w_rp = risk_parity(close, features_dict, universe, cap=0.25)
  mix = 0.6 * w_mom.reindex(close.index).fillna(0) + 0.4 * w_rp.reindex(close.index).fillna(0)
  return mix.apply(lambda row: normalize_weight_row(row, cap=0.25), axis=1)


def gtaa_dual_momentum_weights(close, features_dict, top_n=3):
  universe = [c for c in GLOBAL_TACTICAL_UNIVERSE if c in close.columns and c != "CASH"]
  safe = "SHY" if "SHY" in close.columns else "CASH"
  wdf = pd.DataFrame(0.0, index=norm_idx(close.index), columns=close.columns)
  rlist = list(reb_dates(wdf.index, "M"))
  for i, rd in enumerate(rlist):
    i0 = idx_pos(wdf.index, rd)
    if i0 is None:
      continue
    rd = wdf.index[i0]
    i1 = next_rebalance_pos(wdf.index, rlist, i)
    scores = {}
    for t in universe:
      row = row_asof(features_dict.get(t, pd.DataFrame()), rd)
      if row.empty:
        continue
      abs_mom = safe_float(row.get("MOM_120", np.nan))
      if pd.notna(abs_mom) and abs_mom > 0:
        scores[t] = 0.5 * safe_float(row.get("MOM_60", 0)) + 0.5 * abs_mom
    wr = pd.Series(0.0, index=wdf.columns)
    s = clean_score_series(scores)
    if s.empty:
      wr[safe] = 1.0
    else:
      s = s.nlargest(top_n)
      vols = {}
      for t in s.index:
        r2 = row_asof(features_dict.get(t, pd.DataFrame()), rd)
        if not r2.empty:
          vols[t] = safe_float(r2.get("VOL_60", np.nan))
      for t, wt in inv_vol_w(list(s.index), vols, cap=0.34).items():
        wr[t] = wt
    assign_w(wdf, i0, i1, normalize_weight_row(wr, cap=0.34))
  return wdf


STRATEGIES = {
  "champion_trend_following_v4": lambda c, f: champion_trend_following_v4(c, f, RISKY_ASSETS, **CHAMPION_PARAMS),
  "adaptive_ensemble": adaptive_ensemble,
  "defensive_growth": defensive_growth,
  "factor_rotation": factor_rotation,
  "risk_parity": lambda c, f: risk_parity(c, f, FULL_UNIVERSE),
  "dynamic_champion_switcher_weights": dynamic_champion_switcher_weights,
  "etf_only_robust_portfolio": etf_only_robust_portfolio,
  "gtaa_dual_momentum_weights": gtaa_dual_momentum_weights,
}

for alpha in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
  STRATEGIES[f"blended_champion_weights_alpha_{alpha:.1f}"] = lambda c, f, a=alpha: blended_champion_weights(c, f, alpha=a)

# %% [markdown]
# ## 9. ML experimental - NO PARA WEB
#
# Este bloque es exploratorio, con alto riesgo de overfitting. No alimenta aprobaciones web.

# %%
ML_FEATURE_COLS = ["MOM_20", "MOM_60", "MOM_120", "VOL_20", "VOL_60", "DIST_SMA200", "RSI_14", "VOLUME_RATIO"]


def clean_ml_matrix(X, feature_cols):
  """
  Limpia matriz de features para modelos sklearn.
  - Asegura columnas correctas
  - Convierte a numérico
  - Reemplaza inf por NaN
  - Rellena NaN con 0
  - Devuelve DataFrame limpio
  """
  X = X.copy()
  for col in feature_cols:
    if col not in X.columns:
      X[col] = 0.0
  X = X[feature_cols]
  for col in feature_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")
  X = X.replace([np.inf, -np.inf], np.nan)
  X = X.fillna(0.0)
  return X.astype(float)


# %% [markdown]
# ### 9a. Test clean_ml_matrix

# %%
test_X = pd.DataFrame({
  "MOM_20": [1.0, np.nan, "bad"],
  "MOM_60": [0.5, np.inf, 0.2],
  "MOM_120": [0.1, 0.2, None],
})
for col in ML_FEATURE_COLS:
  if col not in test_X.columns:
    test_X[col] = np.nan
cleaned = clean_ml_matrix(test_X, ML_FEATURE_COLS)
print(cleaned)
assert not cleaned.isna().any().any()
print("OK: clean_ml_matrix sin NaN")


def evaluate_ml_experimental(ml_df):
  """
  ML experimental nunca gana automaticamente ni integra en web.
  Requiere pasar filtros estrictos anti-overfitting.
  """
  if ml_df is None or len(ml_df) == 0:
    return False, "ML experimental, no integrar en web."
  failures = []
  if ml_df["total_return"].mean() <= 0:
    failures.append("walk-forward no positivo")
  beats_spy = (ml_df.get("excess_vs_spy", pd.Series(dtype=float)) > 0).mean()
  if beats_spy < 0.60:
    failures.append(f"supera SPY solo {beats_spy:.0%} anos (<60%)")
  if ml_df["max_drawdown"].mean() <= -30:
    failures.append("max_drawdown peor que -30%")
  pos_ret = ml_df["total_return"].clip(lower=0)
  if len(ml_df) >= 3 and pos_ret.sum() > 0 and pos_ret.max() / pos_ret.sum() > 0.50:
    failures.append("depende de un solo ano")
  recent = ml_df[ml_df["year"].isin([2025, 2026])]
  if len(recent) > 0 and (recent["total_return"] <= 0).any():
    failures.append("falla en 2025/2026")
  if ml_df["sharpe"].mean() > 2.0:
    failures.append("senal de overfitting (sharpe alto)")
  if failures:
    return False, "ML experimental, no integrar en web. " + "; ".join(failures)
  return False, "ML experimental paso filtros pero sigue siendo NO PARA WEB sin validacion manual."


ML_APPROVED_FOR_WEB = False


def ml_experimental_weights(close, features_dict, train_end, top_n=3):
  wdf = pd.DataFrame(0.0, index=norm_idx(close.index), columns=close.columns)
  if not HAS_SKLEARN or not RUN_ML_EXPERIMENTAL:
    return wdf
  cols = [c for c in RISKY_ASSETS + SECTOR_ETFS if c in close.columns and c != "CASH"]
  rows = []
  for t in cols:
    df = features_dict.get(t, pd.DataFrame()).loc[:train_end].copy()
    for dt, row in df.iterrows():
      if pd.isna(row.get("forward_return_20d")):
        continue
      rec = {k: safe_float(row.get(k, np.nan)) for k in ML_FEATURE_COLS}
      rec.update({"date": dt, "ticker": t, "target": row["forward_return_20d"]})
      rows.append(rec)
  train = pd.DataFrame(rows).dropna(subset=["target"])
  if len(train) < 250:
    return wdf
  X = clean_ml_matrix(train, ML_FEATURE_COLS)
  y = pd.to_numeric(train["target"], errors="coerce")
  valid = y.replace([np.inf, -np.inf], np.nan).notna()
  X = X.loc[valid]
  y = y.loc[valid]
  if len(X) < 250:
    return wdf
  if X.isna().any().any():
    print("WARNING: NaN detected in ML train features after cleaning")
    X = X.fillna(0.0)
  models = [
    make_pipeline(
      SimpleImputer(strategy="median"),
      StandardScaler(),
      Ridge(alpha=1.0),
    ),
    make_pipeline(
      SimpleImputer(strategy="median"),
      RandomForestRegressor(n_estimators=75, max_depth=5, random_state=RANDOM_SEED),
    ),
    make_pipeline(
      SimpleImputer(strategy="median"),
      HistGradientBoostingRegressor(max_depth=4, random_state=RANDOM_SEED),
    ),
  ]
  fitted = []
  for model in models:
    try:
      model.fit(X, y)
      fitted.append(model)
    except Exception as exc:
      print(f"WARNING: ML model fit failed: {exc}")
  if not fitted:
    return wdf
  post = close.loc[train_end:]
  rlist = list(reb_dates(post.index, "M"))
  for i, rd in enumerate(rlist):
    i0 = idx_pos(wdf.index, rd)
    if i0 is None:
      continue
    rd = wdf.index[i0]
    i1 = idx_pos(wdf.index, rlist[i + 1]) if i + 1 < len(rlist) else len(wdf.index) - 1
    scores = {}
    for t in cols:
      row = row_asof(features_dict.get(t, pd.DataFrame()), rd)
      if row.empty:
        continue
      x = pd.DataFrame([{k: safe_float(row.get(k, 0.0), default=0.0) for k in ML_FEATURE_COLS}])
      x = clean_ml_matrix(x, ML_FEATURE_COLS)
      if x.isna().any().any():
        x = x.fillna(0.0)
      preds = []
      for model in fitted:
        try:
          pred = model.predict(x)[0]
          pred = safe_float(pred, default=np.nan)
          if np.isfinite(pred):
            preds.append(pred)
        except Exception:
          continue
      if not preds:
        continue
      mean_pred = safe_float(np.mean(preds), default=np.nan)
      if pd.notna(mean_pred):
        scores[t] = mean_pred
    wr = pd.Series(0.0, index=wdf.columns)
    s = clean_score_series(scores)
    s = s[s > 0]
    if s.empty:
      wr = defensive_weight_row(wr)
    else:
      s = s.nlargest(top_n)
      for t in s.index:
        wr[t] = 1 / len(s)
    assign_w(wdf, i0, i1, wr)
  return wdf


def run_ml_experimental_walk_forward(close, features_dict, start_year=WF_START_YEAR):
  rows = []
  if QUICK_TEST or not HAS_SKLEARN or not RUN_ML_EXPERIMENTAL:
    return pd.DataFrame()
  current_year = pd.Timestamp.today().year
  for year in range(start_year, current_year + 1):
    try:
      train_end = f"{year - 1}-12-31"
      cp = close.loc[f"{year}-01-01":f"{year}-12-31"]
      if len(cp) < 40:
        continue
      w = ml_experimental_weights(close, features_dict, train_end).reindex(cp.index).fillna(0)
      m, _, _, _, _ = backtest_portfolio_weights(cp, w)
      rows.append({"year": year, "model": "ml_experimental_not_for_web", **m})
    except Exception as e:
      print(f"ML experimental failed for year {year}: {e}")
      continue
  return pd.DataFrame(rows)


ml_results_df = run_ml_experimental_walk_forward(close_prices, features_dict)
ml_passed_filters, ml_status_msg = evaluate_ml_experimental(ml_results_df)
ML_APPROVED_FOR_WEB = False

if not RUN_ML_EXPERIMENTAL:
  print("ML experimental no ejecutado. Esto es intencional. Primero se valida el resto del notebook.")
elif len(ml_results_df):
  print("ML experimental (NO PARA WEB):")
  print(ml_results_df[["year", "CAGR", "sharpe", "max_drawdown", "excess_vs_spy"]].to_string(index=False))
  print(ml_status_msg)
  print(f"ML_APPROVED_FOR_WEB={ML_APPROVED_FOR_WEB}")
else:
  print("ML experimental no produjo resultados validos. No se integra en web.")

# %% [markdown]
# ## 10. Walk-forward anual desde 2017

# %%
def slice_features(features_dict, end_date=None):
  if end_date is None:
    return features_dict
  return {k: v.loc[:end_date] for k, v in features_dict.items()}


def run_strategy_metrics(close, features_dict, strategy_fn, transaction_cost=TRANSACTION_COST, slippage=SLIPPAGE):
  weights = strategy_fn(close, features_dict)
  return backtest_portfolio_weights(close, weights, transaction_cost=transaction_cost, slippage=slippage)


def run_walk_forward_annual(close, features_dict, strategies, start_year=WF_START_YEAR):
  rows = []
  current_year = pd.Timestamp.today().year
  champion_fn = strategies["champion_trend_following_v4"]
  for year in range(start_year, current_year + 1):
    start, end = f"{year}-01-01", f"{year}-12-31"
    cp = close.loc[start:end]
    if len(cp) < 40:
      continue
    fd = slice_features(features_dict, end)
    w_ch = champion_fn(cp, fd)
    m_ch, pr_ch, _, _, _ = backtest_portfolio_weights(cp, w_ch)
    champion_return = ((1 + pr_ch).prod() - 1) * 100
    for name, fn in tqdm(strategies.items(), desc=f"WF {year}", leave=False):
      w = fn(cp, fd).reindex(cp.index).fillna(0)
      m, pr, _, _, _ = backtest_portfolio_weights(cp, w)
      rows.append({
        "year": year,
        "strategy": name,
        "return": m["total_return"],
        "CAGR": m["CAGR"],
        "sharpe": m["sharpe"],
        "sortino": m["sortino"],
        "max_drawdown": m["max_drawdown"],
        "turnover_avg": m["turnover_avg"],
        "excess_vs_spy": m.get("excess_vs_spy", np.nan),
        "excess_vs_champion": round(m["total_return"] - champion_return, 2),
        "beats_spy": m.get("excess_vs_spy", 0) > 0,
        "beats_champion": m["total_return"] > champion_return,
      })
  return pd.DataFrame(rows)


# %% [markdown]
# ## 10a. Test safe_float y nlargest

# %%
test_scores = {"A": 1.2, "B": np.nan, "C": "bad", "D": pd.Series([0.5])}
s = pd.Series(test_scores)
s = pd.to_numeric(s.apply(lambda x: safe_float(x)), errors="coerce")
s = s.replace([np.inf, -np.inf], np.nan).dropna()
print("Test scores limpios:")
print(s)
print("nlargest(2):", s.nlargest(2).to_dict())
assert len(s.nlargest(2)) == 2, "Test nlargest fallo"
print("OK: safe_float + nlargest funcionan")

wf_df = run_walk_forward_annual(close_prices, features_dict, STRATEGIES, WF_START_YEAR)
if len(wf_df):
  print("Walk-forward OOS summary:")
  print(wf_df.groupby("strategy").agg(
    oos_return=("return", "mean"),
    oos_sharpe=("sharpe", "mean"),
    oos_dd=("max_drawdown", "mean"),
    pct_beats_spy=("beats_spy", "mean"),
    pct_beats_champion=("beats_champion", "mean"),
  ).round(3).sort_values("oos_sharpe", ascending=False).to_string())

# %% [markdown]
# ## 11. Robustez: start dates, leave-one-out, costes y stress

# %%
def run_full_sample_table(close, features_dict, strategies):
  rows, returns, equities, weights = [], {}, {}, {}
  for name, fn in tqdm(strategies.items(), desc="Full sample"):
    w = fn(close, features_dict)
    m, pr, eq, dd, W = backtest_portfolio_weights(close, w)
    rows.append({"strategy": name, **m})
    returns[name] = pr
    equities[name] = eq
    weights[name] = W
  return pd.DataFrame(rows), returns, equities, weights


summary_df, returns_dict, equity_curves, weights_dict = run_full_sample_table(close_prices, features_dict, STRATEGIES)


def run_start_date_robustness(close, features_dict, strategies, years=None):
  years = years or [2012, 2014, 2016, 2018, 2020, 2022]
  rows = []
  for year in years:
    cp = close.loc[f"{year}-01-01":]
    if len(cp) < 252:
      continue
    fd = {k: v.loc[cp.index[0]:] for k, v in features_dict.items()}
    for name, fn in strategies.items():
      m, _, _, _, _ = run_strategy_metrics(cp, fd, fn)
      rows.append({"start_date": f"{year}-01-01", "strategy": name, **m, "beats_spy": m.get("excess_vs_spy", 0) > 0})
  return pd.DataFrame(rows)


def run_leave_one_out(close, features_dict):
  rows = []
  base_assets = [t for t in RISKY_ASSETS if t in close.columns]
  tests = {"full": base_assets}
  for ticker in base_assets:
    tests[f"without_{ticker}"] = [t for t in base_assets if t != ticker]
  for name, universe in tests.items():
    fn = lambda c, f, u=universe: champion_trend_following_v4(c, f, u, **CHAMPION_PARAMS)
    m, _, _, _, _ = run_strategy_metrics(close, features_dict, fn)
    rows.append({"universe_test": name, "strategy": "champion_trend_following_v4", **m, "beats_spy": m.get("excess_vs_spy", 0) > 0})
  return pd.DataFrame(rows)


def run_cost_sensitivity(close, features_dict, strategies):
  rows = []
  grid = [0.0005, 0.001, 0.002, 0.005]
  for name, fn in strategies.items():
    w = fn(close, features_dict)
    for tc, sl in itertools.product(grid, grid):
      m, _, _, _, _ = backtest_portfolio_weights(close, w, transaction_cost=tc, slippage=sl)
      rows.append({"strategy": name, "transaction_cost": tc, "slippage": sl, **m, "alive": m.get("excess_vs_spy", 0) > 0 and m["sharpe"] > 0.3})
  return pd.DataFrame(rows)


STRESS_PERIODS = {
  "covid_crash": ("2020-02-01", "2020-04-30"),
  "inflation_bear_2022": ("2022-01-01", "2022-12-31"),
  "recovery_2023": ("2023-01-01", "2023-12-31"),
  "rate_cut_expectations_2024": ("2024-01-01", "2024-12-31"),
  "recent_2025": ("2025-01-01", "2025-12-31"),
  "current_ytd": (f"{pd.Timestamp.today().year}-01-01", None),
}


def run_stress_periods(returns_dict, benchmark_returns, periods):
  rows = []
  for strategy, returns in returns_dict.items():
    for period, (start, end) in periods.items():
      sl = returns.loc[start:end] if end is not None else returns.loc[start:]
      if len(sl) < 5:
        continue
      spy = benchmark_returns["SPY"].reindex(sl.index).fillna(0)
      strat_return = ((1 + sl).prod() - 1) * 100
      spy_return = ((1 + spy).prod() - 1) * 100
      eq = (1 + sl).cumprod()
      dd = calculate_drawdown(eq).min() * 100
      rows.append({
        "strategy": strategy,
        "period": period,
        "strategy_return": round(strat_return, 2),
        "spy_return": round(spy_return, 2),
        "max_drawdown": round(dd, 2),
        "beats_spy": strat_return > spy_return,
      })
  return pd.DataFrame(rows)


start_robustness_df = run_start_date_robustness(close_prices, features_dict, STRATEGIES)
leave_one_out_df = run_leave_one_out(close_prices, features_dict)
cost_sensitivity_df = run_cost_sensitivity(close_prices, features_dict, STRATEGIES)
stress_df = run_stress_periods(returns_dict, BENCHMARKS, STRESS_PERIODS)

# %% [markdown]
# ## 12. Anti-overfitting y scoring V6

# %%
def psr_approx(returns, benchmark_sr=0.0):
  r = pd.Series(returns).dropna()
  n = len(r)
  if n < 30 or r.std() <= 0:
    return 0.0
  sr = calculate_sharpe(r)
  skew = r.skew()
  kurt = r.kurtosis()
  denom = np.sqrt(max(1e-9, 1 - skew * sr + ((kurt - 1) / 4) * sr ** 2))
  z = (sr - benchmark_sr) * np.sqrt(n - 1) / denom
  if HAS_SCIPY:
    return float(stats.norm.cdf(z))
  return float(1 / (1 + np.exp(-1.702 * z)))


def dsr_approx(returns, n_trials=20):
  r = pd.Series(returns).dropna()
  if len(r) < 30 or r.std() <= 0:
    return 0.0
  trial_penalty = np.sqrt(2 * np.log(max(n_trials, 2))) / np.sqrt(len(r))
  return psr_approx(r, benchmark_sr=trial_penalty)


def overfit_penalty(strategy, wf, start_df, cost_df, stress_df, returns_dict, n_trials):
  penalty = 0
  wf_sub = wf[wf["strategy"] == strategy]
  if len(wf_sub):
    if wf_sub["return"].std() > 50:
      penalty += 8
    if wf_sub["beats_spy"].mean() < 0.50:
      penalty += 10
    recent = wf_sub[wf_sub["year"] >= max(wf_sub["year"].max() - 2, WF_START_YEAR)]
    if len(recent) and recent["beats_spy"].mean() < 0.34:
      penalty += 10
  st_sub = start_df[start_df["strategy"] == strategy]
  if len(st_sub) and st_sub["beats_spy"].mean() < 0.50:
    penalty += 8
  c_sub = cost_df[cost_df["strategy"] == strategy]
  if len(c_sub) and c_sub["alive"].mean() < 0.50:
    penalty += 8
  stress_sub = stress_df[stress_df["strategy"] == strategy]
  if len(stress_sub) and stress_sub["beats_spy"].mean() < 0.40:
    penalty += 6
  dsr = dsr_approx(returns_dict.get(strategy, pd.Series(dtype=float)), n_trials=n_trials)
  if dsr < 0.60:
    penalty += 12
  elif dsr < 0.75:
    penalty += 6
  if "ml" in strategy.lower():
    penalty += 30
  return int(penalty)


def robustness_score(strategy, wf, start_df, cost_df, stress_df, loo_df=None):
  components = []
  wf_sub = wf[wf["strategy"] == strategy]
  if len(wf_sub):
    components.append(100 * wf_sub["beats_spy"].mean())
    components.append(100 * wf_sub["beats_champion"].mean())
    components.append(float(np.clip((wf_sub["sharpe"].mean() + 0.5) / 1.5 * 100, 0, 100)))
  st_sub = start_df[start_df["strategy"] == strategy]
  if len(st_sub):
    components.append(100 * st_sub["beats_spy"].mean())
  c_sub = cost_df[cost_df["strategy"] == strategy]
  if len(c_sub):
    components.append(100 * c_sub["alive"].mean())
  stress_sub = stress_df[stress_df["strategy"] == strategy]
  if len(stress_sub):
    components.append(100 * stress_sub["beats_spy"].mean())
  if strategy == "champion_trend_following_v4" and loo_df is not None and len(loo_df):
    components.append(100 * loo_df["beats_spy"].mean())
  return round(float(np.mean(components)), 2) if components else 0.0


def final_score_v6(strategy, summary, wf, start_df, cost_df, stress_df, loo_df, returns_dict, n_trials):
  wf_sub = wf[wf["strategy"] == strategy]
  full = summary[summary["strategy"] == strategy]
  if wf_sub.empty or full.empty:
    return 0, "REJECTED", 0, 100
  full = full.iloc[0]
  score = 0
  if wf_sub["return"].mean() > 0:
    score += 10
  if wf_sub["beats_spy"].mean() >= 0.65:
    score += 18
  elif wf_sub["beats_spy"].mean() >= 0.55:
    score += 12
  elif wf_sub["beats_spy"].mean() >= 0.45:
    score += 6
  if wf_sub["beats_champion"].mean() >= 0.55:
    score += 12
  elif strategy == "champion_trend_following_v4":
    score += 8
  if wf_sub["sharpe"].mean() >= 0.75:
    score += 16
  elif wf_sub["sharpe"].mean() >= 0.45:
    score += 10
  elif wf_sub["sharpe"].mean() >= 0.25:
    score += 5
  if full["max_drawdown"] > -25:
    score += 12
  elif full["max_drawdown"] > -35:
    score += 7
  if full["CAGR"] > 8:
    score += 10
  elif full["CAGR"] > 4:
    score += 5
  robust = robustness_score(strategy, wf, start_df, cost_df, stress_df, loo_df)
  score += int(np.clip(robust / 100 * 22, 0, 22))
  psr = psr_approx(returns_dict.get(strategy, pd.Series(dtype=float)))
  dsr = dsr_approx(returns_dict.get(strategy, pd.Series(dtype=float)), n_trials=n_trials)
  if psr >= 0.80:
    score += 5
  if dsr >= 0.75:
    score += 5
  penalty = overfit_penalty(strategy, wf, start_df, cost_df, stress_df, returns_dict, n_trials)
  score = int(np.clip(score - penalty, 0, 100))
  if score >= 75:
    status = "APPROVED_FOR_WEB_PAPER"
  elif score >= 60:
    status = "CANDIDATE"
  else:
    status = "REJECTED"
  return score, status, robust, penalty


score_rows = []
for strategy in STRATEGIES:
  score, status, robust, penalty = final_score_v6(
    strategy,
    summary_df,
    wf_df,
    start_robustness_df,
    cost_sensitivity_df,
    stress_df,
    leave_one_out_df,
    returns_dict,
    n_trials=len(STRATEGIES),
  )
  score_rows.append({
    "strategy": strategy,
    "final_score_v6": score,
    "status": status,
    "robustness_score": robust,
    "overfit_penalty": penalty,
    "psr_approx": round(psr_approx(returns_dict.get(strategy, pd.Series(dtype=float))), 3),
    "dsr_approx": round(dsr_approx(returns_dict.get(strategy, pd.Series(dtype=float)), len(STRATEGIES)), 3),
    "APPROVED_FOR_REAL_MONEY": False,
  })

score_df = pd.DataFrame(score_rows).sort_values(["final_score_v6", "robustness_score"], ascending=False)
print(score_df.to_string(index=False))

# %% [markdown]
# ## 13. Visualizacion

# %%
best_strategy = score_df.iloc[0]["strategy"] if len(score_df) else "champion_trend_following_v4"
fig, ax = plt.subplots(figsize=(12, 5))
for name in ["champion_trend_following_v4", best_strategy]:
  if name in equity_curves:
    ax.plot(equity_curves[name].index, equity_curves[name], label=name, lw=2)
spy_eq = (1 + BENCHMARKS["SPY"]).cumprod() * INITIAL_CAPITAL
ax.plot(spy_eq.index, spy_eq, label="SPY", alpha=0.7)
ax.set_title("V6 Champion / Best Candidate vs SPY")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 14. Reporte final

# %%
def decide_v6(score_df):
  if score_df.empty:
    return "RESEARCH_MORE"
  champ = score_df[score_df["strategy"] == "champion_trend_following_v4"]
  champ_score = int(champ["final_score_v6"].iloc[0]) if len(champ) else 0
  best = score_df.iloc[0]
  best_name = best["strategy"]
  best_score = int(best["final_score_v6"])
  best_status = best["status"]
  if best_status != "APPROVED_FOR_WEB_PAPER":
    return "RESEARCH_MORE" if best_status == "CANDIDATE" else "KEEP_V5_CHAMPION"
  if best_name == "champion_trend_following_v4":
    return "KEEP_V5_CHAMPION"
  if "blended_champion_weights" in best_name and best_score >= champ_score + 3:
    return "PROMOTE_V6_BLEND"
  if best_score >= champ_score + 5:
    return "PROMOTE_V6_STRATEGY"
  return "KEEP_V5_CHAMPION"


FINAL_DECISION = decide_v6(score_df)
APPROVED_FOR_WEB_PAPER = bool(len(score_df) and score_df.iloc[0]["status"] == "APPROVED_FOR_WEB_PAPER")
APPROVED_FOR_REAL_MONEY = False

champ_row = score_df[score_df["strategy"] == "champion_trend_following_v4"]
champ_score = int(champ_row["final_score_v6"].iloc[0]) if len(champ_row) else 0
best_row = score_df.iloc[0] if len(score_df) else pd.Series({"strategy": "none", "final_score_v6": 0, "status": "REJECTED"})

print("=" * 80)
print("REPORTE FINAL V6 PROFESSIONAL AUDIT LAB")
print("=" * 80)
print("Disclaimer: Backtest no garantiza resultados futuros.")
print(f"Champion V5/V4: champion_trend_following_v4 | score={champ_score}")
print(f"Mejor V6: {best_row['strategy']} | score={best_row['final_score_v6']} | status={best_row['status']}")
print(f"Decision final: {FINAL_DECISION}")
print(f"APPROVED_FOR_WEB_PAPER={APPROVED_FOR_WEB_PAPER}")
print(f"APPROVED_FOR_REAL_MONEY={APPROVED_FOR_REAL_MONEY} (siempre False)")
print("")
print("Reglas de decision:")
print("- >=75 APPROVED_FOR_WEB_PAPER")
print("- 60-74 CANDIDATE")
print("- <60 REJECTED")
print("- ML experimental esta marcado NO PARA WEB")
if not RUN_ML_EXPERIMENTAL:
  print("ML experimental no ejecutado. Esto es intencional. Primero se valida el resto del notebook.")
elif len(ml_results_df) == 0:
  print("ML experimental no produjo resultados validos. No se integra en web.")
else:
  print(ml_status_msg)
  print(f"ML_APPROVED_FOR_WEB={ML_APPROVED_FOR_WEB} (nunca True automaticamente)")

if FINAL_DECISION == "KEEP_V5_CHAMPION":
  print("Accion: mantener el champion V5/V4 para paper trading web si esta aprobado.")
elif FINAL_DECISION == "PROMOTE_V6_STRATEGY":
  print(f"Accion: promover {best_row['strategy']} como nueva estrategia paper experimental.")
elif FINAL_DECISION == "PROMOTE_V6_BLEND":
  print(f"Accion: promover blend V6 {best_row['strategy']} como paper experimental.")
else:
  print("Accion: investigar mas antes de cambiar el champion.")

# %% [markdown]
# ## 15. Exportar resultados V6

# %%
def export_csv(df, filename):
  if df is None or len(df) == 0:
    pd.DataFrame().to_csv(filename, index=False)
  else:
    df.to_csv(filename, index=False)
  print("Exportado", filename)


export_csv(summary_df, "research_v6_summary.csv")
export_csv(score_df, "research_v6_strategy_scores.csv")
export_csv(wf_df, "research_v6_walk_forward.csv")

blended_df = summary_df[summary_df["strategy"].str.contains("blended_champion", na=False)] if len(summary_df) else pd.DataFrame()
export_csv(blended_df, "research_v6_blended_champion.csv")

dyn_df = summary_df[summary_df["strategy"].str.contains("dynamic_champion", na=False)] if len(summary_df) else pd.DataFrame()
export_csv(dyn_df, "research_v6_dynamic_switcher.csv")

etf_df = summary_df[summary_df["strategy"].str.contains("etf_only", na=False)] if len(summary_df) else pd.DataFrame()
export_csv(etf_df, "research_v6_etf_only.csv")

gtaa_df = summary_df[summary_df["strategy"].str.contains("gtaa", na=False)] if len(summary_df) else pd.DataFrame()
export_csv(gtaa_df, "research_v6_gtaa.csv")

robust_df = score_df[["strategy", "psr_approx", "dsr_approx", "overfit_penalty", "robustness_score", "final_score_v6", "status"]].copy() if len(score_df) else pd.DataFrame()
export_csv(robust_df, "research_v6_robustness.csv")
export_csv(cost_sensitivity_df, "research_v6_cost_sensitivity.csv")
export_csv(v5_metric_audit_df, "research_v6_v5_metric_audit.csv")
export_csv(ml_results_df, "research_v6_ml_results.csv")

equity_export = pd.DataFrame({name: eq for name, eq in equity_curves.items()})
equity_export.index.name = "date"
equity_export.to_csv("research_v6_equity_curves.csv")
print("Exportado research_v6_equity_curves.csv")

selected_strategy = str(best_row["strategy"])
selected_config = {
  "version": "v6_professional_audit_lab",
  "selected_strategy": selected_strategy,
  "decision": FINAL_DECISION,
  "final_score_v6": int(best_row["final_score_v6"]) if "final_score_v6" in best_row else 0,
  "approved_for_web_paper": APPROVED_FOR_WEB_PAPER,
  "approved_for_real_money": False,
  "parameters": CHAMPION_PARAMS if "champion" in selected_strategy else {},
  "warnings": [
    "Backtest no garantiza resultados futuros.",
    "Metricas PSR/DSR son aproximadas.",
    "ML experimental no integrar en web salvo validacion estricta.",
    ml_status_msg if RUN_ML_EXPERIMENTAL else "ML experimental no ejecutado (RUN_ML_EXPERIMENTAL=False).",
  ],
  "ml_experimental": {
    "run_ml_experimental": RUN_ML_EXPERIMENTAL,
    "approved_for_web": ML_APPROVED_FOR_WEB,
    "passed_filters": ml_passed_filters,
    "status": ml_status_msg,
    "n_years": int(len(ml_results_df)),
  },
  "backtest_summary": summary_df[summary_df["strategy"] == selected_strategy].iloc[0].to_dict() if len(summary_df) and selected_strategy in summary_df["strategy"].values else {},
  "disclaimer": "No es asesoramiento financiero.",
}

Path("research_v6_selected_strategy_config.json").write_text(json.dumps(selected_config, indent=2), encoding="utf-8")
print("Exportado research_v6_selected_strategy_config.json")
print("V6 completo.")
