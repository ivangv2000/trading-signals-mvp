# %% [markdown]
# # Trading Research V3 — Portfolio Lab
#
# Laboratorio cuantitativo de estrategias de **portfolio** con gestión de riesgo,
# walk-forward y benchmarks profesionales.
#
# **Aviso:** El backtest no garantiza resultados futuros. Esto no es asesoramiento financiero.

# %%
!pip install yfinance pandas numpy matplotlib plotly tqdm scikit-learn -q

# %% [markdown]
# ## 1. Configuración

# %%
import warnings
warnings.filterwarnings("ignore")

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tqdm.auto import tqdm

# --- Variables editables ---
QUICK_TEST = False

RISKY_ASSETS = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN",
]
ETF_UNIVERSE = [
    "SPY", "QQQ", "IWM",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
    "TLT", "IEF", "SHY", "GLD",
]
DEFENSIVE_ASSETS = ["SHY", "TLT", "GLD", "CASH"]

START_DATE = "2015-01-01"
END_DATE = None

TRANSACTION_COST = 0.001
SLIPPAGE = 0.001
COST_PER_SIDE = TRANSACTION_COST + SLIPPAGE
INITIAL_CAPITAL = 10000

MARKET_TICKER = "SPY"
BENCHMARK_QQQ = "QQQ"

# Walk-forward: primer año de test OOS
WF_FIRST_TEST_YEAR = 2019

if QUICK_TEST:
    RISKY_ASSETS = ["SPY", "QQQ", "NVDA", "AMD"]
    ETF_UNIVERSE = ["SPY", "QQQ", "XLK", "TLT", "GLD", "SHY"]
    START_DATE = "2021-01-01"
    WF_FIRST_TEST_YEAR = 2022
    print("⚡ QUICK_TEST activo: menos tickers, periodo corto.")

print(f"Periodo: {START_DATE} → hoy | Capital inicial: ${INITIAL_CAPITAL:,}")

# %% [markdown]
# ## 2. Descarga de datos

# %%
def download_data(tickers, start, end=None):
  """Descarga OHLCV y devuelve (dict, close_prices). Incluye CASH sintético."""
  data, failed = {}, []
  tickers = sorted(set(tickers))

  for ticker in tqdm(tickers, desc="Descargando"):
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
      col_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
      df = df.rename(columns={c: col_map.get(str(c).lower(), c) for c in df.columns})
      keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
      df = df[keep].dropna(subset=["Close"])
      if df.empty:
        failed.append(ticker)
      else:
        data[ticker.upper()] = df
    except Exception as e:
      failed.append(f"{ticker} ({e})")

  if failed:
    print("Tickers fallidos:", failed)

  if not data:
    raise ValueError("No se descargaron datos.")

  close_prices = pd.DataFrame({t: d["Close"] for t, d in data.items()}).sort_index()
  close_prices = close_prices.dropna(how="all").ffill()
  close_prices.index = pd.DatetimeIndex(close_prices.index)
  if close_prices.index.tz is not None:
    close_prices.index = close_prices.index.tz_localize(None)
  for t, df in list(data.items()):
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is not None:
      df.index = df.index.tz_localize(None)
    data[t] = df

  # CASH: precio constante = 1, retorno 0
  close_prices["CASH"] = 1.0
  cash_df = pd.DataFrame({"Close": 1.0}, index=close_prices.index)
  data["CASH"] = cash_df

  print(f"Activos: {len(data)} | Filas: {len(close_prices)}")
  return data, close_prices


all_tickers = sorted(set(RISKY_ASSETS + ETF_UNIVERSE + DEFENSIVE_ASSETS + [MARKET_TICKER, BENCHMARK_QQQ]))
data, close_prices = download_data(all_tickers, START_DATE, END_DATE)

# %% [markdown]
# ## 3. Features profesionales (sin lookahead)

# %%
def add_features(df):
  """Indicadores técnicos y de régimen. Usar .shift(1) al tomar decisiones."""
  d = df.copy()
  c = d["Close"]
  ret1 = c.pct_change(1)

  for n in [1, 5, 20, 60, 120, 252]:
    d[f"RET_{n}D"] = c.pct_change(n) if n > 1 else ret1

  for n in [20, 50, 100, 200]:
    d[f"SMA_{n}"] = c.rolling(n).mean()
  d["EMA_21"] = c.ewm(span=21, adjust=False).mean()
  d["EMA_50"] = c.ewm(span=50, adjust=False).mean()

  d["VOL_20"] = ret1.rolling(20).std() * np.sqrt(252)
  d["VOL_60"] = ret1.rolling(60).std() * np.sqrt(252)

  if "High" in d.columns and "Low" in d.columns:
    prev = c.shift(1)
    tr = pd.concat([
      (d["High"] - d["Low"]).abs(),
      (d["High"] - prev).abs(),
      (d["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    d["ATR_14"] = tr.rolling(14).mean()

  eq = (1 + ret1.fillna(0)).cumprod()
  d["ROLLING_DD"] = (eq - eq.cummax()) / eq.cummax()
  d["DIST_SMA_200"] = (c / d["SMA_200"] - 1).replace([np.inf, -np.inf], np.nan)
  d["REALIZED_VOL"] = ret1.rolling(20).std() * np.sqrt(252)

  vol60 = d["VOL_60"].replace(0, np.nan)
  d["MOM_60_VOL"] = d["RET_60D"] / vol60
  d["MOM_120_VOL"] = d["RET_120D"] / vol60
  d["MOM_COMBO"] = 0.4 * d["RET_20D"] + 0.3 * d["RET_60D"] + 0.3 * d["RET_120D"]
  d["MOM_COMBO_VOL"] = d["MOM_COMBO"] / vol60

  if MARKET_TICKER in str(d.columns):
    pass
  return d


def build_features_dict(data):
  return {t: add_features(df) for t, df in data.items()}


features_dict = build_features_dict(data)

if MARKET_TICKER in features_dict:
  spy_f = features_dict[MARKET_TICKER]
  spy_f["SPY_ABOVE_SMA200"] = (spy_f["Close"] > spy_f["SMA_200"]).astype(float)
  spy_f["SPY_ABOVE_SMA100"] = (spy_f["Close"] > spy_f["SMA_100"]).astype(float)
  spy_f["SPY_RET_60D"] = spy_f["RET_60D"]
  spy_f["SPY_VOL_20"] = spy_f["VOL_20"]
  features_dict[MARKET_TICKER] = spy_f

print("Features calculados para", len(features_dict), "activos")

# %% [markdown]
# ## 4. Utilidades de portfolio

# %%
def get_momentum_value(features_row, lookback, skip=0):
  """Lee momentum desde features; fallback seguro."""
  col = f"RET_{lookback}D"
  mom = features_row.get(col, np.nan)
  if pd.isna(mom):
    return np.nan
  if skip > 0:
    skip_col = f"RET_{skip}D"
    if skip_col in features_row.index:
      mom = mom - features_row.get(skip_col, 0)
  return mom


def extract_params(row, keys):
  out = {}
  for k in keys:
    if k in row.index and pd.notna(row.get(k)):
      v = row[k]
      if k in ("top_n", "lookback_days", "skip_recent_days", "fast_ma", "slow_ma", "param_id"):
        v = int(v)
      elif k == "market_filter":
        v = bool(v)
      elif k in ("vol_target",):
        v = float(v)
      out[k] = v
  return out


def normalize_dt_index(index):
  """Índice datetime sin timezone (evita errores en Colab)."""
  idx = pd.DatetimeIndex(index)
  if idx.tz is not None:
    idx = idx.tz_localize(None)
  return idx


def align_date_to_index(index, date):
  """Mapea fecha de calendario al último día de trading disponible (pad)."""
  index = normalize_dt_index(index)
  date = pd.Timestamp(date)
  if date.tz is not None:
    date = date.tz_localize(None)
  if date in index:
    return date
  loc = index.get_indexer([date], method="pad")[0]
  if loc < 0:
    loc = index.get_indexer([date], method="nearest")[0]
  if loc < 0:
    return None
  return index[loc]


def index_position(index, date):
  """Posición entera segura en el índice."""
  aligned = align_date_to_index(index, date)
  if aligned is None:
    return None
  return index.get_loc(aligned)


def get_row_asof(df, date):
  """Fila de features al cierre del día de trading más reciente <= date."""
  aligned = align_date_to_index(df.index, date)
  if aligned is None:
    return None
  if aligned in df.index:
    return df.loc[aligned]
  loc = df.index.get_indexer([aligned], method="pad")[0]
  if loc < 0:
    return None
  return df.iloc[loc]


def get_rebalance_dates(index, freq="W-FRI"):
  """
  Fechas de rebalanceo = último día de trading de cada semana/mes.
  Evita KeyError en festivos (ej. Viernes Santo 2015-04-03).
  """
  index = normalize_dt_index(index)
  s = pd.Series(np.arange(len(index)), index=index)
  reb = []
  for _, grp in s.groupby(pd.Grouper(freq=freq)):
    if len(grp) > 0:
      reb.append(grp.index[-1])
  return pd.DatetimeIndex(reb)


def assign_weights_window(weights, idx, end_idx, w_row):
  """Asigna pesos entre idx+1 y end_idx (inclusive)."""
  end_idx = min(int(end_idx), len(weights) - 1)
  idx = int(idx)
  vals = w_row.reindex(weights.columns).fillna(0).values
  for j in range(idx + 1, end_idx + 1):
    weights.iloc[j] = vals


def inverse_vol_weights(tickers, vols, max_weight=0.40):
  """Pesos inverse-volatility con tope por activo."""
  vols = pd.Series(vols).replace(0, np.nan).dropna()
  tickers = [t for t in tickers if t in vols.index]
  if not tickers:
    return pd.Series(dtype=float)
  inv = 1.0 / vols[tickers]
  w = inv / inv.sum()
  for _ in range(5):
    w = w.clip(upper=max_weight)
    if w.sum() == 0:
      break
    w = w / w.sum()
  return w


def cap_normalize_row(row, max_weight=0.40):
  w = row.copy().clip(lower=0)
  for _ in range(5):
    w = w.clip(upper=max_weight)
    s = w.sum()
    if s <= 0:
      return w
    w = w / s
  return w


def spy_bull_series(features_dict, index):
  if MARKET_TICKER not in features_dict:
    return pd.Series(1.0, index=index)
  s = features_dict[MARKET_TICKER]["SPY_ABOVE_SMA200"].reindex(index).ffill().fillna(0)
  return s.shift(1).fillna(0)  # decisión con info de ayer


def momentum_score(close, lookback, skip=0, vol=None, use_vol_adjust=True):
  """Momentum = retorno lookback menos retorno reciente (skip)."""
  mom = close.pct_change(lookback)
  if skip > 0:
    mom = mom - close.pct_change(skip)
  mom = mom.shift(1)
  if use_vol_adjust and vol is not None:
    vol_s = vol.reindex(close.index).shift(1).replace(0, np.nan)
    mom = mom / vol_s
  return mom


def apply_vol_target_scale(weights_df, close_prices, vol_target=0.15, lookback=20):
  """Escala exposición total para acercarse a vol objetivo."""
  rets = close_prices.pct_change().fillna(0)
  port_ret = (weights_df.shift(1).fillna(0) * rets).sum(axis=1)
  realized = port_ret.rolling(lookback).std() * np.sqrt(252)
  scale = (vol_target / realized.replace(0, np.nan)).clip(0, 1).shift(1).fillna(1)
  return weights_df.mul(scale, axis=0)


def apply_drawdown_brake(weights_df, close_prices, cost_per_side, max_dd=-0.25):
  """Reduce exposición si el portfolio supera drawdown máximo (sin lookahead en pesos futuros)."""
  rets = close_prices.pct_change().fillna(0)
  w = weights_df.fillna(0)
  w_exec = w.shift(1).fillna(0)
  turnover = w.diff().abs().sum(axis=1).fillna(0)
  port_ret = (w_exec * rets).sum(axis=1) - turnover * cost_per_side
  equity = (1 + port_ret).cumprod()
  dd = equity / equity.cummax() - 1
  brake = (dd < max_dd).astype(float).shift(1).fillna(0)
  scale = 1 - 0.5 * brake  # reducir 50% exposición tras DD excesivo
  return w.mul(scale, axis=0)

# %% [markdown]
# ## 5. Backtest común de portfolio

# %%
def backtest_portfolio_weights(close_prices, target_weights, transaction_cost=TRANSACTION_COST, slippage=SLIPPAGE):
  """
  Backtest diario. Pesos ejecutados con shift(1). Costes proporcionales al turnover.
  """
  rets = close_prices.pct_change().fillna(0)
  cols = [c for c in target_weights.columns if c in rets.columns]
  w = target_weights[cols].reindex(close_prices.index).fillna(0)
  w_exec = w.shift(1).fillna(0)
  turnover = w.diff().abs().sum(axis=1).fillna(0)
  cost = turnover * (transaction_cost + slippage)
  port_ret = (w_exec * rets[cols]).sum(axis=1) - cost

  equity = (1 + port_ret).cumprod() * INITIAL_CAPITAL
  roll_max = equity.cummax()
  dd = (equity - roll_max) / roll_max

  years = max((port_ret.index[-1] - port_ret.index[0]).days / 365.25, 1 / 365.25)
  total_return = (equity.iloc[-1] / INITIAL_CAPITAL - 1) * 100
  cagr = ((equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
  ann_vol = port_ret.std() * np.sqrt(252) * 100
  sharpe = (port_ret.mean() / port_ret.std() * np.sqrt(252)) if port_ret.std() > 0 else 0
  downside = port_ret[port_ret < 0].std() * np.sqrt(252)
  sortino = (port_ret.mean() / downside * np.sqrt(252)) if downside > 0 else 0
  max_dd = dd.min() * 100
  calmar = cagr / abs(max_dd) if max_dd != 0 else 0

  def bm_ret(ticker):
    if ticker not in rets.columns:
      return pd.Series(0.0, index=rets.index)
    return rets[ticker]

  spy_r = bm_ret(MARKET_TICKER)
  qqq_r = bm_ret(BENCHMARK_QQQ)
  eq_cols = [c for c in cols if c != "CASH"]
  ew_r = rets[eq_cols].mean(axis=1) if eq_cols else spy_r

  spy_total = ((1 + spy_r).prod() - 1) * 100
  qqq_total = ((1 + qqq_r).prod() - 1) * 100
  ew_total = ((1 + ew_r).prod() - 1) * 100

  metrics = {
    "total_return": round(total_return, 2),
    "CAGR": round(cagr, 2),
    "annual_volatility": round(ann_vol, 2),
    "sharpe": round(sharpe, 3),
    "sortino": round(sortino, 3),
    "max_drawdown": round(max_dd, 2),
    "calmar": round(calmar, 3),
    "turnover_avg": round(turnover.mean(), 4),
    "exposure_pct": round(w_exec.abs().sum(axis=1).mean() * 100, 2),
    "best_day": round(port_ret.max() * 100, 3),
    "worst_day": round(port_ret.min() * 100, 3),
    "positive_days_pct": round((port_ret > 0).mean() * 100, 2),
    "benchmark_spy_return": round(spy_total, 2),
    "benchmark_qqq_return": round(qqq_total, 2),
    "benchmark_equal_weight_return": round(ew_total, 2),
    "excess_vs_spy": round(total_return - spy_total, 2),
    "excess_vs_qqq": round(total_return - qqq_total, 2),
    "excess_vs_equal_weight": round(total_return - ew_total, 2),
  }
  bt = pd.DataFrame({
    "portfolio_return": port_ret,
    "equity": equity,
    "drawdown": dd,
    "turnover": turnover,
    "exposure": w_exec.abs().sum(axis=1),
  })
  return bt, metrics, w


def slice_close(close_prices, start=None, end=None):
  out = close_prices
  if start:
    out = out.loc[start:]
  if end:
    out = out.loc[:end]
  return out.dropna(how="all")

# %% [markdown]
# ## 6. Estrategia 1 — Cross-sectional Momentum V2

# %%
def cross_sectional_momentum_v2(
    close_prices,
    features_dict,
    universe,
    top_n=3,
    rebalance_freq="W-FRI",
    lookback_days=120,
    skip_recent_days=5,
    use_vol_adjust=True,
    vol_target=0.15,
    max_asset_weight=0.40,
    defensive_asset="SHY",
    market_filter=True,
    transaction_cost=TRANSACTION_COST,
    slippage=SLIPPAGE,
):
  cols = [c for c in universe if c in close_prices.columns and c != "CASH"]
  panel = close_prices[cols]
  panel.index = normalize_dt_index(panel.index)
  weights = pd.DataFrame(0.0, index=panel.index, columns=close_prices.columns)
  bull = spy_bull_series(features_dict, panel.index)
  reb_dates = list(get_rebalance_dates(panel.index, rebalance_freq))

  for i, reb in enumerate(reb_dates):
    idx = index_position(panel.index, reb)
    if idx is None:
      continue
    reb = panel.index[idx]
    end_idx = index_position(panel.index, reb_dates[i + 1]) if i + 1 < len(reb_dates) else len(panel.index) - 1
    if end_idx is None:
      end_idx = len(panel.index) - 1

    bull_val = float(bull.asof(reb)) if reb in bull.index or bull.index[0] <= reb else 0.0

    if market_filter and bull_val < 0.5:
      w = pd.Series(0.0, index=weights.columns)
      def_asset = defensive_asset if defensive_asset in weights.columns else "CASH"
      w[def_asset] = 1.0
    else:
      scores = {}
      for t in cols:
        if t not in features_dict:
          continue
        row = get_row_asof(features_dict[t], reb)
        if row is None:
          continue
        mom = get_momentum_value(row, lookback_days, skip_recent_days)
        if pd.isna(mom):
          continue
        if use_vol_adjust:
          vol = row.get("VOL_60", np.nan)
          if vol and vol > 0:
            mom = mom / vol
        scores[t] = mom

      if not scores:
        w = pd.Series(0.0, index=weights.columns)
        w["CASH"] = 1.0
      else:
        s = pd.Series(scores).dropna()
        s = s[s > 0].nlargest(top_n)
        if len(s) == 0:
          def_asset = defensive_asset if defensive_asset in weights.columns else "CASH"
          w = pd.Series(0.0, index=weights.columns)
          w[def_asset] = 1.0
        else:
          vols = {}
          for t in s.index:
            row_t = get_row_asof(features_dict[t], reb)
            if row_t is not None:
              vols[t] = row_t.get("VOL_60", np.nan)
          w = pd.Series(0.0, index=weights.columns)
          iw = inverse_vol_weights(list(s.index), vols, max_asset_weight)
          for t, wt in iw.items():
            w[t] = wt

    assign_weights_window(weights, idx, end_idx, w)

  weights = apply_vol_target_scale(weights, close_prices, vol_target)
  weights = apply_drawdown_brake(weights, close_prices, transaction_cost + slippage)
  return weights

# %% [markdown]
# ## 7. Estrategia 2 — ETF Rotation V2

# %%
def etf_rotation_v2(
    close_prices,
    features_dict,
    universe,
    top_n=3,
    rebalance_freq="W-FRI",
    score_method="momentum_vol",
    defensive_assets=None,
    market_filter=True,
    vol_target=0.12,
    max_asset_weight=0.40,
):
  if defensive_assets is None:
    defensive_assets = ["SHY", "TLT", "GLD"]
  cols = [c for c in universe if c in close_prices.columns]
  panel = close_prices[cols]
  panel.index = normalize_dt_index(panel.index)
  weights = pd.DataFrame(0.0, index=panel.index, columns=close_prices.columns)
  bull = spy_bull_series(features_dict, panel.index)
  reb_dates = list(get_rebalance_dates(panel.index, rebalance_freq))

  for i, reb in enumerate(reb_dates):
    idx = index_position(panel.index, reb)
    if idx is None:
      continue
    reb = panel.index[idx]
    end_idx = index_position(panel.index, reb_dates[i + 1]) if i + 1 < len(reb_dates) else len(panel.index) - 1
    if end_idx is None:
      end_idx = len(panel.index) - 1

    bull_val = float(bull.asof(reb)) if len(bull) else 1.0
    pool = cols if (not market_filter or bull_val >= 0.5) else [c for c in defensive_assets if c in cols]
    scores = {}
    for t in pool:
      if t not in features_dict:
        continue
      row = get_row_asof(features_dict[t], reb)
      if row is None:
        continue
      if score_method == "momentum_vol":
        sc = row.get("MOM_COMBO_VOL", np.nan)
      else:
        sc = row.get("MOM_60_VOL", np.nan)
      if pd.notna(sc):
        scores[t] = sc

    w = pd.Series(0.0, index=weights.columns)
    if not scores or max(scores.values()) <= 0:
      w["CASH"] = 1.0
    else:
      s = pd.Series(scores).nlargest(top_n)
      s = s[s > 0]
      if len(s) == 0:
        w["CASH"] = 1.0
      else:
        vols = {}
        for t in s.index:
          row_t = get_row_asof(features_dict[t], reb)
          if row_t is not None:
            vols[t] = row_t.get("VOL_60", np.nan)
        iw = inverse_vol_weights(list(s.index), vols, max_asset_weight)
        for t, wt in iw.items():
          w[t] = wt

    assign_weights_window(weights, idx, end_idx, w)

  weights = apply_vol_target_scale(weights, close_prices, vol_target)
  return weights

# %% [markdown]
# ## 8. Estrategia 3 — Dual Momentum

# %%
def dual_momentum_strategy(
    close_prices,
    features_dict,
    offensive_assets=None,
    defensive_assets=None,
    lookback_days=120,
    rebalance_freq="M",
    top_n=1,
):
  if offensive_assets is None:
    offensive_assets = ["SPY", "QQQ", "IWM"]
  if defensive_assets is None:
    defensive_assets = ["TLT", "IEF", "SHY", "GLD"]
  weights = pd.DataFrame(0.0, index=close_prices.index, columns=close_prices.columns)
  weights.index = normalize_dt_index(weights.index)
  reb_dates = list(get_rebalance_dates(weights.index, rebalance_freq))

  for i, reb in enumerate(reb_dates):
    idx = index_position(weights.index, reb)
    if idx is None:
      continue
    reb = weights.index[idx]
    end_idx = index_position(weights.index, reb_dates[i + 1]) if i + 1 < len(reb_dates) else len(weights.index) - 1
    if end_idx is None:
      end_idx = len(weights.index) - 1

    off_scores = {}
    for t in offensive_assets:
      if t in features_dict:
        row = get_row_asof(features_dict[t], reb)
        if row is not None:
          off_scores[t] = get_momentum_value(row, lookback_days)
    off_scores = {k: v for k, v in off_scores.items() if pd.notna(v)}

    w = pd.Series(0.0, index=weights.columns)
    if off_scores and max(off_scores.values()) > 0:
      best = pd.Series(off_scores).nlargest(top_n)
      for t in best.index:
        w[t] = 1.0 / len(best)
    else:
      def_scores = {}
      for t in defensive_assets:
        if t in features_dict:
          row = get_row_asof(features_dict[t], reb)
          if row is not None:
            def_scores[t] = get_momentum_value(row, lookback_days)
      def_scores = {k: v for k, v in def_scores.items() if pd.notna(v)}
      if def_scores:
        best = pd.Series(def_scores).nlargest(top_n)
        for t in best.index:
          w[t] = 1.0 / len(best)
      else:
        w["CASH"] = 1.0

    assign_weights_window(weights, idx, end_idx, w)

  return weights

# %% [markdown]
# ## 9. Estrategia 4 — Trend Following Portfolio

# %%
def trend_following_portfolio(
    close_prices,
    features_dict,
    universe,
    fast_ma=50,
    slow_ma=200,
    rebalance_freq="W-FRI",
    vol_target=0.15,
    max_asset_weight=0.30,
    defensive_asset="SHY",
):
  cols = [c for c in universe if c in close_prices.columns and c != "CASH"]
  weights = pd.DataFrame(0.0, index=close_prices.index, columns=close_prices.columns)
  weights.index = normalize_dt_index(weights.index)
  reb_dates = list(get_rebalance_dates(weights.index, rebalance_freq))

  for i, reb in enumerate(reb_dates):
    idx = index_position(weights.index, reb)
    if idx is None:
      continue
    reb = weights.index[idx]
    end_idx = index_position(weights.index, reb_dates[i + 1]) if i + 1 < len(reb_dates) else len(weights.index) - 1
    if end_idx is None:
      end_idx = len(weights.index) - 1

    eligible = []
    vols = {}
    for t in cols:
      if t not in features_dict:
        continue
      row = get_row_asof(features_dict[t], reb)
      if row is None:
        continue
      sma_slow = row.get(f"SMA_{slow_ma}", np.nan)
      ema_fast = row.get("EMA_50" if fast_ma == 50 else f"EMA_{fast_ma}", row.get("EMA_50", np.nan))
      close = row.get("Close", np.nan)
      if pd.notna(sma_slow) and pd.notna(ema_fast) and pd.notna(close):
        if close > sma_slow and ema_fast > sma_slow:
          eligible.append(t)
          vols[t] = row.get("VOL_60", np.nan)

    w = pd.Series(0.0, index=weights.columns)
    if eligible:
      iw = inverse_vol_weights(eligible, vols, max_asset_weight)
      for t, wt in iw.items():
        w[t] = wt
    else:
      da = defensive_asset if defensive_asset in weights.columns else "CASH"
      w[da] = 1.0

    assign_weights_window(weights, idx, end_idx, w)

  weights = apply_vol_target_scale(weights, close_prices, vol_target)
  return weights

# %% [markdown]
# ## 10. Estrategia 5 — Ensemble Portfolio

# %%
def ensemble_portfolio(close_prices, features_dict, risky_universe, etf_universe, cs_params=None, etf_params=None, tf_params=None, dm_params=None):
  """Combina pesos de 4 estrategias con filtro de régimen SPY."""
  cs_params = cs_params or {}
  etf_params = etf_params or {}
  tf_params = tf_params or {}
  dm_params = dm_params or {}

  w_cs = cross_sectional_momentum_v2(close_prices, features_dict, risky_universe, **cs_params)
  w_etf = etf_rotation_v2(close_prices, features_dict, etf_universe, **etf_params)
  w_tf = trend_following_portfolio(close_prices, features_dict, risky_universe, **tf_params)
  w_dm = dual_momentum_strategy(close_prices, features_dict, **dm_params)

  combined = (w_cs.fillna(0) + w_etf.fillna(0) + w_tf.fillna(0) + w_dm.fillna(0)) / 4.0
  max_w = 0.35
  combined = combined.apply(lambda r: cap_normalize_row(r, max_w), axis=1)

  bull = spy_bull_series(features_dict, combined.index)
  for i in range(len(combined)):
    if bull.iloc[i] < 0.5:
      row = combined.iloc[i].copy()
      risky = [c for c in row.index if c not in DEFENSIVE_ASSETS and c != "CASH"]
      row[risky] = row[risky] * 0.5
      def_part = 0.5 - row[risky].sum()
      if def_part > 0:
        for da in ["SHY", "GLD", "CASH"]:
          if da in row.index:
            row[da] += def_part
            break
      combined.iloc[i] = cap_normalize_row(row, max_w)

  return combined

# %% [markdown]
# ## 11. Grids de parámetros (limitados, sin data mining brutal)

# %%
def get_param_grids(quick=False):
  if quick:
    return {
      "cross_sectional": [
        {"top_n": 3, "lookback_days": 120, "skip_recent_days": 5, "rebalance_freq": "W-FRI", "vol_target": 0.15, "market_filter": True},
      ],
      "etf_rotation": [
        {"top_n": 3, "rebalance_freq": "W-FRI", "vol_target": 0.12},
      ],
      "dual_momentum": [
        {"lookback_days": 120, "rebalance_freq": "M"},
      ],
      "trend_following": [
        {"fast_ma": 50, "slow_ma": 200, "vol_target": 0.15},
      ],
    }
  return {
    "cross_sectional": [
      dict(zip(["top_n", "lookback_days", "skip_recent_days", "rebalance_freq", "vol_target", "market_filter"], p))
      for p in itertools.product(
        [2, 3, 5], [60, 120, 252], [0, 5, 20], ["W-FRI", "M"], [0.10, 0.15], [True]
      )
    ],
    "etf_rotation": [
      dict(zip(["top_n", "rebalance_freq", "vol_target"], p))
      for p in itertools.product([2, 3], ["W-FRI", "M"], [0.10, 0.15])
    ],
    "dual_momentum": [
      {"lookback_days": lb, "rebalance_freq": "M"} for lb in [60, 120, 252]
    ],
    "trend_following": [
      dict(zip(["fast_ma", "slow_ma", "vol_target"], p))
      for p in itertools.product([50], [200], [0.10, 0.15])
    ],
  }


def run_strategy_grid(strategy_name, close_prices, features_dict, grids):
  rows = []
  for i, params in enumerate(tqdm(grids, desc=strategy_name)):
    try:
      if strategy_name == "cross_sectional":
        w = cross_sectional_momentum_v2(close_prices, features_dict, RISKY_ASSETS, **params)
      elif strategy_name == "etf_rotation":
        w = etf_rotation_v2(close_prices, features_dict, ETF_UNIVERSE, **params)
      elif strategy_name == "dual_momentum":
        w = dual_momentum_strategy(close_prices, features_dict, **params)
      elif strategy_name == "trend_following":
        w = trend_following_portfolio(close_prices, features_dict, RISKY_ASSETS, **params)
      else:
        continue
      bt, m, _ = backtest_portfolio_weights(close_prices, w)
      row = {"strategy": strategy_name, "param_id": i, **params, **m}
      rows.append(row)
    except Exception as e:
      rows.append({"strategy": strategy_name, "param_id": i, **params, "error": str(e)})
  return pd.DataFrame(rows)


grids = get_param_grids(quick=QUICK_TEST)
print("Combinaciones a probar:")
for k, v in grids.items():
  print(f"  {k}: {len(v)}")

# %% [markdown]
# ## 12. Ejecutar grid completo (periodo total)

# %%
all_results = []
for strat, g in grids.items():
  all_results.append(run_strategy_grid(strat, close_prices, features_dict, g))

# Ensemble con params por defecto
w_ens = ensemble_portfolio(close_prices, features_dict, RISKY_ASSETS, ETF_UNIVERSE)
bt_ens, m_ens, w_ens_final = backtest_portfolio_weights(close_prices, w_ens)
ens_row = {"strategy": "ensemble", "param_id": 0, **m_ens}
all_results.append(pd.DataFrame([ens_row]))

portfolio_results = pd.concat(all_results, ignore_index=True)
portfolio_results = portfolio_results.sort_values("sharpe", ascending=False).reset_index(drop=True)
print(f"Resultados totales: {len(portfolio_results)}")
portfolio_results.head(10)

# %% [markdown]
# ## 13. Walk-forward anual (selección sin lookahead)
#
# Si probamos muchas combinaciones, alguna puede salir bien por casualidad.
# Por eso **no aceptamos una estrategia solo por ser la mejor del ranking**.
# Tiene que superar reglas mínimas en test, por años y contra benchmark.
#
# Proceso: para cada año de test, elegir parámetros solo con años anteriores.

# %%
def composite_train_score(m):
  """Score para elegir parámetros en train (no solo rentabilidad)."""
  dd_pen = 0 if m["max_drawdown"] > -25 else -20
  return (
    m["sharpe"] * 20
    + m["excess_vs_spy"] * 0.3
    + m["calmar"] * 5
    + dd_pen
  )


def walk_forward_year(close_prices, features_dict, grids, test_year):
  train_end = f"{test_year - 1}-12-31"
  test_start = f"{test_year}-01-01"
  test_end = f"{test_year}-12-31"

  cp_train = slice_close(close_prices, end=train_end)
  cp_test = slice_close(close_prices, test_start, test_end)
  if len(cp_test) < 20 or len(cp_train) < 100:
    return []

  rows = []
  for strat, g in grids.items():
    best_score, best_params, best_m = -np.inf, None, None
    for params in g:
      try:
        if strat == "cross_sectional":
          w = cross_sectional_momentum_v2(cp_train, features_dict, RISKY_ASSETS, **params)
        elif strat == "etf_rotation":
          w = etf_rotation_v2(cp_train, features_dict, ETF_UNIVERSE, **params)
        elif strat == "dual_momentum":
          w = dual_momentum_strategy(cp_train, features_dict, **params)
        elif strat == "trend_following":
          w = trend_following_portfolio(cp_train, features_dict, RISKY_ASSETS, **params)
        else:
          continue
        _, m, _ = backtest_portfolio_weights(cp_train, w)
        sc = composite_train_score(m)
        if sc > best_score:
          best_score, best_params, best_m = sc, params, m
      except Exception:
        continue

    if best_params is None:
      continue

    # Evaluar en test con params elegidos en train
    try:
      if strat == "cross_sectional":
        w_full = cross_sectional_momentum_v2(close_prices, features_dict, RISKY_ASSETS, **best_params)
      elif strat == "etf_rotation":
        w_full = etf_rotation_v2(close_prices, features_dict, ETF_UNIVERSE, **best_params)
      elif strat == "dual_momentum":
        w_full = dual_momentum_strategy(close_prices, features_dict, **best_params)
      elif strat == "trend_following":
        w_full = trend_following_portfolio(close_prices, features_dict, RISKY_ASSETS, **best_params)
      else:
        continue
      w_test = w_full.loc[test_start:test_end]
      cp_test_full = close_prices.loc[test_start:test_end]
      bt, m, _ = backtest_portfolio_weights(cp_test_full, w_test)
      spy_yr = ((1 + close_prices[MARKET_TICKER].pct_change().loc[test_start:test_end].fillna(0)).prod() - 1) * 100
      rows.append({
        "strategy": strat,
        "test_year": test_year,
        "train_score": round(best_score, 2),
        "best_params": str(best_params),
        "oos_return": m["total_return"],
        "oos_sharpe": m["sharpe"],
        "oos_max_drawdown": m["max_drawdown"],
        "oos_excess_vs_spy": m["excess_vs_spy"],
        "spy_return": round(spy_yr, 2),
        "beats_spy": m["excess_vs_spy"] > 0,
      })
    except Exception as e:
      rows.append({"strategy": strat, "test_year": test_year, "error": str(e)})

  return rows


test_years = list(range(WF_FIRST_TEST_YEAR, 2027))
wf_rows = []
for yr in tqdm(test_years, desc="Walk-forward"):
  wf_rows.extend(walk_forward_year(close_prices, features_dict, grids, yr))

walk_forward_yearly = pd.DataFrame(wf_rows)
if not walk_forward_yearly.empty:
  wf_summary = walk_forward_yearly.groupby("strategy").agg(
    oos_return_mean=("oos_return", "mean"),
    oos_sharpe_mean=("oos_sharpe", "mean"),
    oos_dd_mean=("oos_max_drawdown", "mean"),
    pct_years_beating_spy=("beats_spy", "mean"),
    years_tested=("test_year", "count"),
  ).round(3)
  print("=== Walk-forward resumen por estrategia ===")
  print(wf_summary.to_string())
else:
  wf_summary = pd.DataFrame()
  print("Walk-forward sin resultados (periodo muy corto).")

# %% [markdown]
# ## 14. Criterios de aprobación V3

# %%
def evaluate_approval(portfolio_results, wf_summary):
  r = portfolio_results.copy()
  r["approved_v3"] = False
  r["candidate_v3"] = False

  wf_map = wf_summary.to_dict("index") if not wf_summary.empty else {}

  for idx, row in r.iterrows():
    strat = row["strategy"]
    wf = wf_map.get(strat)

    if wf:
      approved = (
        wf.get("oos_return_mean", 0) > 0
        and wf.get("pct_years_beating_spy", 0) >= 0.6
        and wf.get("oos_sharpe_mean", 0) > 0.5
        and wf.get("oos_dd_mean", -100) > -30
        and row.get("excess_vs_spy", 0) > 0
        and row.get("turnover_avg", 1) < 0.6
      )
      candidate = (
        not approved
        and wf.get("oos_return_mean", 0) > 0
        and wf.get("oos_sharpe_mean", 0) > 0.5
        and wf.get("oos_dd_mean", -100) < -30
      )
    else:
      approved = (
        row.get("excess_vs_spy", 0) > 0
        and row.get("sharpe", 0) > 0.5
        and row.get("max_drawdown", -100) > -30
        and row.get("turnover_avg", 1) < 0.6
      )
      candidate = (
        not approved
        and row.get("sharpe", 0) > 0.8
        and row.get("max_drawdown", -100) < -30
      )

    r.at[idx, "approved_v3"] = approved
    r.at[idx, "candidate_v3"] = candidate

  return r

portfolio_results = evaluate_approval(portfolio_results, wf_summary)
approved_v3 = portfolio_results[portfolio_results["approved_v3"]]
candidates_v3 = portfolio_results[portfolio_results["candidate_v3"]]
discarded_v3 = portfolio_results[~portfolio_results["approved_v3"] & ~portfolio_results["candidate_v3"]]

print(f"Aprobadas V3: {len(approved_v3)} | Candidatas: {len(candidates_v3)} | Descartadas: {len(discarded_v3)}")

# %% [markdown]
# ## 15. Gráficos

# %%
# Mejor estrategia por Sharpe (o ensemble si no hay mejor)
CS_KEYS = ["top_n", "lookback_days", "skip_recent_days", "rebalance_freq", "vol_target", "market_filter"]
ETF_KEYS = ["top_n", "rebalance_freq", "vol_target"]
DM_KEYS = ["lookback_days", "rebalance_freq"]
TF_KEYS = ["fast_ma", "slow_ma", "vol_target"]

if len(portfolio_results) > 0:
  best = portfolio_results.iloc[0]
  best_strat = best["strategy"]

  if best_strat == "ensemble":
    w_plot = w_ens_final
    title = "Ensemble Portfolio"
  elif best_strat == "cross_sectional":
    p = extract_params(best, CS_KEYS)
    w_plot = cross_sectional_momentum_v2(close_prices, features_dict, RISKY_ASSETS, **p)
    title = f"Cross-sectional V2 (top_n={p.get('top_n')})"
  elif best_strat == "etf_rotation":
    p = extract_params(best, ETF_KEYS)
    w_plot = etf_rotation_v2(close_prices, features_dict, ETF_UNIVERSE, **p)
    title = f"ETF Rotation V2 (top_n={p.get('top_n')})"
  elif best_strat == "dual_momentum":
    p = extract_params(best, DM_KEYS)
    w_plot = dual_momentum_strategy(close_prices, features_dict, **p)
    title = "Dual Momentum"
  elif best_strat == "trend_following":
    p = extract_params(best, TF_KEYS)
    w_plot = trend_following_portfolio(close_prices, features_dict, RISKY_ASSETS, **p)
    title = "Trend Following Portfolio"
  else:
    w_plot = w_ens_final
    title = str(best_strat)

  bt, _, _ = backtest_portfolio_weights(close_prices, w_plot)
  spy_eq = (1 + close_prices[MARKET_TICKER].pct_change().fillna(0)).cumprod() * INITIAL_CAPITAL
  qqq_eq = (1 + close_prices[BENCHMARK_QQQ].pct_change().fillna(0)).cumprod() * INITIAL_CAPITAL
  eq_cols = [c for c in RISKY_ASSETS if c in close_prices.columns]
  ew_eq = (1 + close_prices[eq_cols].pct_change().mean(axis=1).fillna(0)).cumprod() * INITIAL_CAPITAL

  fig, axes = plt.subplots(2, 2, figsize=(14, 10))

  axes[0, 0].plot(bt.index, bt["equity"], label="Estrategia", linewidth=2)
  axes[0, 0].plot(spy_eq.index, spy_eq, label="SPY", alpha=0.8)
  axes[0, 0].plot(qqq_eq.index, qqq_eq, label="QQQ", alpha=0.8)
  axes[0, 0].plot(ew_eq.index, ew_eq, label="Equal Weight", alpha=0.7)
  axes[0, 0].set_title(f"Equity curve — {title}")
  axes[0, 0].legend()
  axes[0, 0].grid(alpha=0.3)

  axes[0, 1].fill_between(bt.index, bt["drawdown"] * 100, 0, color="red", alpha=0.4, label="Estrategia")
  spy_dd = (spy_eq - spy_eq.cummax()) / spy_eq.cummax() * 100
  axes[0, 1].plot(spy_dd.index, spy_dd, label="SPY", alpha=0.7)
  axes[0, 1].set_title("Drawdown %")
  axes[0, 1].legend()
  axes[0, 1].grid(alpha=0.3)

  if not walk_forward_yearly.empty:
    ann = walk_forward_yearly.pivot_table(index="strategy", columns="test_year", values="oos_return", aggfunc="mean")
    im = axes[1, 0].imshow(ann.values, aspect="auto", cmap="RdYlGn")
    axes[1, 0].set_xticks(range(len(ann.columns)))
    axes[1, 0].set_xticklabels(ann.columns.astype(int), fontsize=8)
    axes[1, 0].set_yticks(range(len(ann.index)))
    axes[1, 0].set_yticklabels(ann.index, fontsize=8)
    axes[1, 0].set_title("Retornos anuales OOS (walk-forward)")
    plt.colorbar(im, ax=axes[1, 0])

  top_w = w_plot.iloc[-1].sort_values(ascending=False).head(8)
  axes[1, 1].barh(top_w.index, top_w.values)
  axes[1, 1].set_title("Pesos actuales (top 8)")
  axes[1, 1].invert_yaxis()

  plt.tight_layout()
  plt.show()

  # Heatmap sharpe por estrategia
  if "param_id" in portfolio_results.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    for strat in portfolio_results["strategy"].unique():
      sub = portfolio_results[portfolio_results["strategy"] == strat]
      ax.scatter(sub.index, [strat] * len(sub), c=sub["sharpe"], cmap="RdYlGn", s=30, vmin=-1, vmax=2)
    ax.set_title("Sharpe por combinación de parámetros")
    ax.set_xlabel("Índice de combinación")
    plt.tight_layout()
    plt.show()

  fig, ax = plt.subplots(figsize=(12, 3))
  ax.plot(bt.index, bt["turnover"], alpha=0.7)
  ax.set_title("Turnover diario")
  ax.grid(alpha=0.3)
  plt.tight_layout()
  plt.show()

  # Pesos en el tiempo (exposición por activo — top 5, muestreo semanal)
  w_sample = w_plot.resample("W-FRI").last().dropna(how="all")
  top_assets = w_plot.iloc[-1].nlargest(5).index.tolist()
  fig, ax = plt.subplots(figsize=(12, 4))
  for t in top_assets:
    if t in w_sample.columns:
      ax.plot(w_sample.index, w_sample[t], label=t, alpha=0.8)
  ax.set_title("Pesos del portfolio en el tiempo (top 5, semanal)")
  ax.legend(fontsize=8)
  ax.grid(alpha=0.3)
  plt.tight_layout()
  plt.show()

# %% [markdown]
# ## 16. Report final
#
# **Backtest no garantiza resultados futuros.**

# %%
print("=" * 70)
print("REPORT FINAL — TRADING RESEARCH V3")
print("Backtest no garantiza resultados futuros.")
print("=" * 70)

print("\n1. ESTRATEGIAS APROBADAS V3:")
if len(approved_v3) == 0:
  print("   (ninguna)")
else:
  for _, r in approved_v3.head(10).iterrows():
    print(f"   • {r['strategy']} | Sharpe={r['sharpe']} | DD={r['max_drawdown']}% | excess SPY={r['excess_vs_spy']}%")

print("\n2. ESTRATEGIAS CANDIDATAS (buen perfil pero drawdown/riesgo alto):")
if len(candidates_v3) == 0:
  print("   (ninguna)")
else:
  for _, r in candidates_v3.head(5).iterrows():
    print(f"   • {r['strategy']} | Sharpe={r['sharpe']} | DD={r['max_drawdown']}% | excess SPY={r['excess_vs_spy']}%")

print("\n3. ESTRATEGIAS DESCARTADAS:", len(discarded_v3))

print("\n4. MEJOR POR MÉTRICA:")
if len(portfolio_results) > 0:
  print(f"   Retorno:  {portfolio_results.loc[portfolio_results['total_return'].idxmax(), 'strategy']}")
  print(f"   Sharpe:   {portfolio_results.loc[portfolio_results['sharpe'].idxmax(), 'strategy']}")
  print(f"   Drawdown: {portfolio_results.loc[portfolio_results['max_drawdown'].idxmax(), 'strategy']} (menos negativo)")
  if not walk_forward_yearly.empty:
    yr_wins = walk_forward_yearly.groupby("strategy")["beats_spy"].mean()
    print(f"   Consistencia anual: {yr_wins.idxmax()} ({yr_wins.max()*100:.0f}% años vs SPY)")

print("\n5. RECOMENDACIÓN:")
if len(approved_v3) > 0:
  print("   → Meter en la web SOLO tras revisión manual de las aprobadas.")
elif len(candidates_v3) > 0:
  print("   → No meter todavía. Hay candidatas con buen retorno/Sharpe pero riesgo alto.")
  print("   → Seguir investigando: reducir drawdown (vol targeting, ensemble, filtros régimen).")
else:
  print("   → No meter todavía. Seguir investigando.")

print("\n⚠️ Investigación educativa. No es asesoramiento financiero.")

# %% [markdown]
# ## 17. Exportar resultados

# %%
portfolio_results.to_csv("research_v3_portfolio_results.csv", index=False)
walk_forward_yearly.to_csv("research_v3_walk_forward.csv", index=False)
approved_v3.to_csv("research_v3_approved.csv", index=False)

# Equity curves de top estrategia + ensemble
bt_ens_export, _, _ = backtest_portfolio_weights(close_prices, w_ens_final)
spy_eq = (1 + close_prices[MARKET_TICKER].pct_change().fillna(0)).cumprod() * INITIAL_CAPITAL
eq_export = pd.DataFrame({
  "date": bt_ens_export.index,
  "ensemble_equity": bt_ens_export["equity"],
  "spy_equity": spy_eq.reindex(bt_ens_export.index).values,
})
eq_export.to_csv("research_v3_equity_curves.csv", index=False)

print("Exportado:")
print("  research_v3_portfolio_results.csv")
print("  research_v3_walk_forward.csv")
print("  research_v3_approved.csv")
print("  research_v3_equity_curves.csv")

# Colab: from google.colab import files; files.download("research_v3_portfolio_results.csv")
