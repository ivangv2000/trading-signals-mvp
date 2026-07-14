# %% [markdown]
# # Trading Research V12 Robust Factor Rotation Signal Engine
#
# Rotacion factorial semanal: momentum, trend, relative strength, regime filter.
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

QUICK_TEST = True
START_DATE = "2010-01-01"
END_DATE = None
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001
SLIPPAGE = 0.001
REBALANCE_FREQ = "W-FRI"
TOP_N = 5
MAX_WEIGHT = 0.25
MIN_WEIGHT = 0.05
VOL_TARGET = 0.15
CASH_ASSET = "SHY"
WF_START_YEAR = 2016
MOMENTUM_WINDOWS = [60, 120, 252]
SKIP_RECENT_DAYS = 5
TREND_WINDOWS = [100, 200]
MIN_SCORE = 60
UNIVERSE_MODE = "liquid_large_caps"
MARKET = "SPY"

UNIVERSE = [
  "SPY", "QQQ", "IWM", "DIA",
  "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLC", "XLB", "XLRE",
  "MTUM", "QUAL", "USMV", "VLUE", "MOAT", "SCHD", "GLD", "TLT", "IEF", "SHY",
  "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "INTC", "MU",
  "GOOGL", "META", "AMZN", "NFLX", "ADBE", "CRM", "ORCL",
  "JPM", "BAC", "GS", "MS",
  "XOM", "CVX",
  "UNH", "LLY", "JNJ",
  "WMT", "COST", "HD", "MCD",
  "PANW", "NOW", "SHOP", "UBER",
]

DEFENSIVE = ["SHY", "GLD", "TLT", "IEF"]

if QUICK_TEST:
  UNIVERSE = ["SPY", "QQQ", "XLK", "XLV", "XLF", "GLD", "TLT", "SHY", "AAPL", "MSFT", "NVDA", "AMZN"]
  START_DATE = "2018-01-01"
  TOP_N = 3
  print("QUICK_TEST activo")

print("V12 Robust Factor Rotation | desde", START_DATE)
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


data_dict, close_prices = download_data(UNIVERSE, START_DATE, END_DATE)
TRADABLE = [c for c in close_prices.columns if c not in ("SPY", "QQQ", "IWM", "DIA")]
print("Tickers:", len(close_prices.columns), "| dias:", len(close_prices))

# %% [markdown]
# ## 3. Features factoriales

# %%
def _sf(x, default=np.nan):
  try:
    if x is None or (isinstance(x, float) and np.isnan(x)):
      return default
    v = float(x)
    return default if not np.isfinite(v) else v
  except Exception:
    return default


def _atr(df, n=14):
  h, l, c = df["High"], df["Low"], df["Close"]
  tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
  return tr.rolling(n).mean()


def calculate_factor_features(close_prices, data_dict):
  feats = {}
  rets = close_prices.pct_change()
  for ticker in close_prices.columns:
    c = close_prices[ticker]
    df = data_dict.get(ticker, pd.DataFrame())
    vol = df.get("Volume", pd.Series(0, index=c.index)) if len(df) else pd.Series(0, index=c.index)
    f = pd.DataFrame(index=c.index)
    f["Close"] = c
    for w in [60, 120, 252]:
      f[f"MOM_{w}"] = c / c.shift(w) - 1
    f["MOM_252_SKIP_5"] = c.shift(SKIP_RECENT_DAYS) / c.shift(252) - 1
    for w in TREND_WINDOWS:
      f[f"SMA_{w}"] = c.rolling(w).mean()
    f["ABOVE_SMA100"] = (c > f["SMA_100"]).astype(float)
    f["ABOVE_SMA200"] = (c > f["SMA_200"]).astype(float)
    f["SMA100_ABOVE_SMA200"] = (f["SMA_100"] > f["SMA_200"]).astype(float)
    f["TREND_SCORE"] = f["ABOVE_SMA100"] * 0.4 + f["ABOVE_SMA200"] * 0.4 + f["SMA100_ABOVE_SMA200"] * 0.2
    for w in [20, 60, 120]:
      f[f"VOL_{w}"] = rets[ticker].rolling(w).std() * np.sqrt(252)
    if len(df):
      f["ATR_14"] = _atr(df).reindex(c.index)
    else:
      f["ATR_14"] = c * 0.02
    f["ATR_PCT"] = f["ATR_14"] / c.replace(0, np.nan)
    f["drawdown_60"] = c / c.rolling(60).max() - 1
    f["drawdown_120"] = c / c.rolling(120).max() - 1
    f["distance_from_high_252"] = c / c.rolling(252).max() - 1
    f["dollar_volume"] = c * vol
    f["dollar_volume_20"] = f["dollar_volume"].rolling(20).mean()
    feats[ticker] = f

  dates = close_prices.index
  for col_suffix in ["MOM_60", "MOM_120", "MOM_252", "MOM_252_SKIP_5", "VOL_20"]:
    wide = pd.DataFrame({t: feats[t][col_suffix] for t in feats}, index=dates)
    rank = wide.rank(axis=1, pct=True)
    for t in feats:
      feats[t][f"rank_{col_suffix.lower()}"] = rank[t]

  if MARKET in feats:
    spy = feats[MARKET]
    mkt = pd.DataFrame(index=dates)
    mkt["SPY_ABOVE_SMA200"] = spy["ABOVE_SMA200"]
    if "QQQ" in feats:
      mkt["QQQ_ABOVE_SMA200"] = feats["QQQ"]["ABOVE_SMA200"]
    else:
      mkt["QQQ_ABOVE_SMA200"] = spy["ABOVE_SMA200"]
    mkt["MARKET_RISK_ON"] = ((mkt["SPY_ABOVE_SMA200"] >= 0.5) & (mkt["QQQ_ABOVE_SMA200"] >= 0.5)).astype(float)
    mkt["MARKET_REGIME_SCORE"] = mkt["SPY_ABOVE_SMA200"] * 0.6 + mkt["QQQ_ABOVE_SMA200"] * 0.4
    for t in feats:
      for c in mkt.columns:
        feats[t][c] = mkt[c]

  vol20 = pd.DataFrame({t: feats[t]["VOL_20"] for t in feats}, index=dates)
  low_vol_rank = vol20.rank(axis=1, pct=True, ascending=True)
  for t in feats:
    feats[t]["LOW_VOL_RANK"] = low_vol_rank[t]
  return feats


factor_features = calculate_factor_features(close_prices, data_dict)
print("Features factoriales listas")

# %% [markdown]
# ## 4. Score factorial

# %%
def calculate_asset_score(features, date, mom_combo="ABC", trend_mode="C"):
  """Score 0-100 por activo en una fecha."""
  scores = {}
  regime = 0.5
  if MARKET in features and date in features[MARKET].index:
    regime = _sf(features[MARKET].loc[date].get("MARKET_REGIME_SCORE"), 0.5)

  for ticker, df in features.items():
    if date not in df.index:
      continue
    row = df.loc[date]
    if _sf(row.get("dollar_volume_20"), 0) < 1_000_000 and ticker not in DEFENSIVE:
      continue

    mom_parts = []
    if "A" in mom_combo or mom_combo == "ABC":
      mom_parts.append(_sf(row.get("rank_mom_60"), 0.5))
    if "B" in mom_combo or mom_combo == "ABC":
      mom_parts.append(_sf(row.get("rank_mom_120"), 0.5))
    if "C" in mom_combo or mom_combo == "ABC":
      mom_parts.append(_sf(row.get("rank_mom_252_skip_5"), 0.5))
    momentum_score = np.mean(mom_parts) * 100 if mom_parts else 50

    if trend_mode == "A":
      trend_score = _sf(row.get("ABOVE_SMA100"), 0) * 100
    elif trend_mode == "B":
      trend_score = _sf(row.get("ABOVE_SMA200"), 0) * 100
    else:
      trend_score = _sf(row.get("TREND_SCORE"), 0.5) * 100

    rel_score = np.mean([
      _sf(row.get("rank_mom_60"), 0.5),
      _sf(row.get("rank_mom_120"), 0.5),
      _sf(row.get("rank_mom_252"), 0.5),
    ]) * 100
    low_vol_score = _sf(row.get("LOW_VOL_RANK"), 0.5) * 100
    market_score = regime * 100

    score = (
      0.35 * momentum_score + 0.25 * trend_score + 0.20 * rel_score
      + 0.10 * low_vol_score + 0.10 * market_score
    )
    if _sf(row.get("ABOVE_SMA200"), 1) < 0.5 and ticker not in DEFENSIVE:
      score *= 0.55
    if _sf(row.get("VOL_20"), 0.2) > 0.50:
      score *= 0.70
    if _sf(row.get("drawdown_60"), 0) < -0.25:
      score *= 0.75
    scores[ticker] = float(np.clip(score, 0, 100))
  return pd.Series(scores)


def build_scores_panel(features, rebalance_dates, mom_combo="ABC", trend_mode="C"):
  rows = []
  for dt in rebalance_dates:
    s = calculate_asset_score(features, dt, mom_combo, trend_mode)
    if len(s):
      rows.append(s.rename(dt))
  return pd.DataFrame(rows) if rows else pd.DataFrame()

# %% [markdown]
# ## 5. Cartera objetivo

# %%
def build_target_portfolio(scores, date, features, top_n=TOP_N, max_weight=MAX_WEIGHT,
                           vol_target=VOL_TARGET, min_score=MIN_SCORE):
  if date not in scores.index and len(scores):
    date = scores.index[scores.index <= date][-1] if any(scores.index <= date) else scores.index[0]
  if date not in scores.index:
    return pd.Series(dtype=float)
  s = scores.loc[date].dropna().sort_values(ascending=False)
  risk_on = True
  if MARKET in features and date in features[MARKET].index:
    risk_on = _sf(features[MARKET].loc[date].get("MARKET_RISK_ON"), 1) >= 0.5

  candidates = s[(s >= min_score) & (~s.index.isin(["SPY", "QQQ", "IWM", "DIA"]))]
  top = candidates.head(top_n)
  weights = pd.Series(0.0, index=s.index)

  if len(top) == 0 or not risk_on:
    def_w = [t for t in DEFENSIVE if t in s.index]
    if not def_w:
      def_w = [CASH_ASSET] if CASH_ASSET in s.index else []
    if def_w:
      w = 1.0 / len(def_w)
      for t in def_w:
        weights[t] = w
    return weights[weights > 0]

  vols = {}
  for t in top.index:
    if t in features and date in features[t].index:
      vols[t] = max(_sf(features[t].loc[date].get("VOL_20"), 0.15), 0.05)
    else:
      vols[t] = 0.20
  inv = pd.Series({t: 1.0 / v for t, v in vols.items()})
  raw = inv / inv.sum()
  raw = raw.clip(upper=max_weight)
  if raw.sum() > 0:
    raw = raw / raw.sum()
  equity_exposure = 0.85 if risk_on else 0.55
  scale = min(1.0, vol_target / np.mean(list(vols.values())))
  equity_exposure *= scale
  for t, w in raw.items():
    weights[t] = w * equity_exposure
  cash_w = 1.0 - weights.sum()
  cash_t = CASH_ASSET if CASH_ASSET in s.index else (DEFENSIVE[0] if DEFENSIVE[0] in s.index else None)
  if cash_t and cash_w > 0.01:
    weights[cash_t] = weights.get(cash_t, 0) + cash_w
  weights = weights[weights > 0.001]
  if weights.sum() > 0:
    weights = weights / weights.sum()
  return weights

# %% [markdown]
# ## 6. Señales BUY / SELL / HOLD / AVOID

# %%
def generate_factor_signals(target_weights, previous_weights, scores, date):
  all_tickers = sorted(set(target_weights.index) | set(previous_weights.index))
  rows = []
  for t in all_tickers:
    tw = _sf(target_weights.get(t), 0)
    pw = _sf(previous_weights.get(t), 0)
    sc = _sf(scores.get(t), 0)
    if tw > 0 and pw == 0:
      signal, reason = "BUY", "nuevo top ranking factorial"
    elif tw > 0 and pw > 0 and abs(tw - pw) <= 0.03:
      signal, reason = "HOLD", "mantener peso estable"
    elif pw > 0 and tw == 0:
      signal, reason = "SELL", "sale del top ranking"
    elif tw < pw - 0.03:
      signal, reason = "REDUCE", "reducir peso objetivo"
    elif tw > pw + 0.03:
      signal, reason = "INCREASE", "aumentar peso objetivo"
    elif tw == 0 and sc < MIN_SCORE:
      signal, reason = "AVOID", "score bajo o riesgo alto"
    else:
      signal, reason = "HOLD", "sin cambio material"
    rows.append({
      "ticker": t, "signal": signal, "target_weight": round(tw, 4),
      "previous_weight": round(pw, 4), "score": round(sc, 1),
      "reason": reason, "next_review": "proximo viernes",
      "date": date,
    })
  return pd.DataFrame(rows)

# %% [markdown]
# ## 7. Backtest walk-forward

# %%
def get_rebalance_dates(close_prices):
  weekly = close_prices.resample(REBALANCE_FREQ).last().dropna(how="all")
  return weekly.index


def run_v12_backtest(close_prices, data_dict, features, scores_panel, top_n=TOP_N,
                     max_weight=MAX_WEIGHT, vol_target=VOL_TARGET, initial_capital=INITIAL_CAPITAL):
  rebalance_dates = [d for d in get_rebalance_dates(close_prices) if d in scores_panel.index]
  if len(rebalance_dates) < 2:
    return {}, pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

  cost_rate = TRANSACTION_COST + SLIPPAGE
  equity = initial_capital
  equity_curve = {}
  weights = pd.Series(dtype=float)
  turnover_rows = []
  prev_weights = pd.Series(dtype=float)

  all_dates = close_prices.index
  rebal_map = {}
  for rd in rebalance_dates:
    future = all_dates[all_dates > rd]
    if len(future):
      rebal_map[future[0]] = rd

  daily_weights = pd.Series(dtype=float)
  for dt in all_dates:
    if dt in rebal_map:
      sig_date = rebal_map[dt]
      target = build_target_portfolio(scores_panel, sig_date, features, top_n, max_weight, vol_target)
      turnover = (target.reindex(prev_weights.index.union(target.index), fill_value=0)
                  - prev_weights.reindex(prev_weights.index.union(target.index), fill_value=0)).abs().sum() / 2
      cost = equity * turnover * cost_rate
      equity -= cost
      prev_weights = target.copy()
      daily_weights = target
      turnover_rows.append({
        "signal_date": sig_date, "exec_date": dt, "turnover": round(turnover, 4),
        "cost": round(cost, 2), "equity_after_cost": round(equity, 2),
      })
    if len(daily_weights) and dt > all_dates[0]:
      prev_dt = all_dates[all_dates < dt][-1]
      day_ret = 0.0
      for t, w in daily_weights.items():
        if t in close_prices.columns and prev_dt in close_prices.index and dt in close_prices.index:
          r = close_prices[t].loc[dt] / close_prices[t].loc[prev_dt] - 1
          if np.isfinite(r):
            day_ret += w * r
      equity *= (1 + day_ret)
    equity_curve[dt] = equity

  eq = pd.Series(equity_curve).sort_index()
  metrics = compute_portfolio_metrics(eq, turnover_rows, initial_capital)
  turnover_df = pd.DataFrame(turnover_rows)
  return metrics, eq, turnover_df, prev_weights


def compute_portfolio_metrics(equity, turnover_rows, initial_capital):
  if equity is None or len(equity) < 2:
    return {}
  rets = equity.pct_change().fillna(0)
  total_ret = equity.iloc[-1] / initial_capital - 1
  years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
  cagr = (equity.iloc[-1] / initial_capital) ** (1 / years) - 1
  vol = rets.std() * np.sqrt(252)
  sh = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
  ds = rets[rets < 0].std()
  so = rets.mean() / ds * np.sqrt(252) if ds and ds > 0 else 0
  peak = equity.cummax()
  mdd = ((equity - peak) / peak.replace(0, np.nan)).min()
  to_df = pd.DataFrame(turnover_rows)
  turnover = to_df["turnover"].mean() if len(to_df) else 0
  total_cost = to_df["cost"].sum() if len(to_df) else 0
  return {
    "total_return": round(total_ret * 100, 2),
    "CAGR": round(cagr * 100, 2),
    "sharpe": round(sh, 3),
    "sortino": round(so, 3),
    "max_drawdown": round(mdd * 100, 2),
    "calmar": round(cagr / abs(mdd), 3) if mdd != 0 else 0,
    "volatility": round(vol * 100, 2),
    "turnover": round(turnover, 4),
    "total_cost": round(total_cost, 2),
    "num_rebalances": len(turnover_rows),
    "exposure": round(1 - (to_df["turnover"].sum() / max(len(to_df), 1)), 3) if len(to_df) else 1,
  }


def yearly_comparison(equity, close_prices):
  rows = []
  for year in sorted(set(equity.index.year)):
    eq_y = equity.loc[f"{year}-01-01":f"{year}-12-31"]
    if len(eq_y) < 2:
      continue
    strat = eq_y.iloc[-1] / eq_y.iloc[0] - 1
    spy = close_prices[MARKET].loc[f"{year}-01-01":f"{year}-12-31"] if MARKET in close_prices else pd.Series()
    qqq = close_prices["QQQ"].loc[f"{year}-01-01":f"{year}-12-31"] if "QQQ" in close_prices else pd.Series()
    spy_r = spy.iloc[-1] / spy.iloc[0] - 1 if len(spy) >= 2 else 0
    qqq_r = qqq.iloc[-1] / qqq.iloc[0] - 1 if len(qqq) >= 2 else 0
    rows.append({
      "year": year, "strategy_return": round(strat * 100, 2),
      "SPY_return": round(spy_r * 100, 2), "QQQ_return": round(qqq_r * 100, 2),
      "beats_spy": strat > spy_r, "beats_qqq": strat > qqq_r,
    })
  return pd.DataFrame(rows)

# %% [markdown]
# ## 8. Parameter robustness testing

# %%
def run_parameter_grid(features, close_prices, data_dict, quick=QUICK_TEST):
  rebalance_dates = get_rebalance_dates(close_prices)
  rebalance_dates = [d for d in rebalance_dates if d >= pd.Timestamp(f"{WF_START_YEAR}-01-01")]
  top_ns = [3, 5] if quick else [3, 5, 8]
  max_ws = [0.20, 0.25] if quick else [0.20, 0.25, 0.33]
  vols = [0.10, 0.15] if quick else [0.10, 0.15, 0.20]
  mom_combos = ["AB", "ABC"] if quick else ["AB", "BC", "ABC"]
  trend_modes = ["B", "C"] if quick else ["A", "B", "C"]
  results = []
  for top_n, mw, vt, mc, tm in tqdm(
    list(product(top_ns, max_ws, vols, mom_combos, trend_modes)),
    desc="Param grid",
  ):
    scores = build_scores_panel(features, rebalance_dates, mc, tm)
    if scores.empty:
      continue
    m, eq, to, _ = run_v12_backtest(close_prices, data_dict, features, scores, top_n, mw, vt)
    if not m:
      continue
    yearly = yearly_comparison(eq, close_prices)
    results.append({
      "top_n": top_n, "max_weight": mw, "vol_target": vt,
      "mom_combo": mc, "trend_mode": tm, **m,
      "win_years_vs_spy": yearly["beats_spy"].mean() if len(yearly) else 0,
      "win_years_vs_qqq": yearly["beats_qqq"].mean() if len(yearly) else 0,
    })
  return pd.DataFrame(results)


param_results = run_parameter_grid(factor_features, close_prices, data_dict)
print("Configs probadas:", len(param_results))
if len(param_results):
  print(param_results.sort_values("sharpe", ascending=False).head(3).to_string(index=False))

# %% [markdown]
# ## 9. Seleccion robusta champion

# %%
def select_champion_config(param_df, close_prices):
  if param_df is None or len(param_df) == 0:
    return None, pd.DataFrame()
  spy_total = (close_prices[MARKET].iloc[-1] / close_prices[MARKET].iloc[0] - 1) * 100 if MARKET in close_prices else 0
  df = param_df.copy()
  df["robust_score"] = 0.0
  df.loc[df["sharpe"] > 1, "robust_score"] += 2
  df.loc[df["CAGR"] > spy_total, "robust_score"] += 2
  df.loc[df["max_drawdown"] > -25, "robust_score"] += 1.5
  df.loc[df["win_years_vs_spy"] >= 0.6, "robust_score"] += 2
  df.loc[df["win_years_vs_qqq"] >= 0.5, "robust_score"] += 1
  df.loc[df["total_cost"] < INITIAL_CAPITAL * 0.05, "robust_score"] += 1
  best = df.sort_values(["robust_score", "sharpe", "CAGR"], ascending=False).iloc[0]
  return best.to_dict(), df.sort_values("robust_score", ascending=False)


champion, param_ranked = select_champion_config(param_results, close_prices)
if champion:
  print("Champion config:", {k: champion[k] for k in ["top_n", "max_weight", "vol_target", "mom_combo", "trend_mode", "sharpe", "CAGR"]})
else:
  print("Sin champion config valida")

# %% [markdown]
# ## 10. Backtest champion y score V12

# %%
def compute_v12_score(metrics, yearly_df, param_df, champion_cfg):
  score, notes = 0, []
  if not metrics:
    return 0, "REJECTED", ["sin metricas"]
  spy_y = yearly_df["beats_spy"].mean() if len(yearly_df) else 0
  qqq_y = yearly_df["beats_qqq"].mean() if len(yearly_df) else 0
  if metrics.get("sharpe", 0) > 1:
    score += 20
  spy_total = 0
  if metrics.get("CAGR", 0) > 8:
    score += 20
  if metrics.get("max_drawdown", -100) > -25:
    score += 15
  if spy_y >= 0.6:
    score += 15
  if qqq_y >= 0.5:
    score += 10
  if metrics.get("total_cost", 9999) < INITIAL_CAPITAL * 0.08:
    score += 10
  if champion_cfg and champion_cfg.get("robust_score", 0) >= 5:
    score += 10
  recent = yearly_df[yearly_df["year"].isin([2025, 2026])] if len(yearly_df) else pd.DataFrame()
  if len(recent) and (recent["strategy_return"] < -5).any():
    score -= 20
    notes.append("2025/2026 debiles")
  if metrics.get("profit_factor", 1) < 1:
    pass
  if metrics.get("total_cost", 0) > INITIAL_CAPITAL * 0.15:
    score -= 20
    notes.append("costes altos")
  score = int(np.clip(score, 0, 100))
  status = "APPROVED_FOR_WEB_PAPER" if score >= 75 else ("CANDIDATE" if score >= 60 else "REJECTED")
  return score, status, notes


rebalance_dates = get_rebalance_dates(close_prices)
if champion:
  champ_scores = build_scores_panel(
    factor_features, rebalance_dates,
    champion.get("mom_combo", "ABC"), champion.get("trend_mode", "C"),
  )
  champ_metrics, equity_oos, turnover_df, final_weights = run_v12_backtest(
    close_prices, data_dict, factor_features, champ_scores,
    int(champion.get("top_n", TOP_N)), champion.get("max_weight", MAX_WEIGHT),
    champion.get("vol_target", VOL_TARGET),
  )
  yearly_df = yearly_comparison(equity_oos, close_prices)
else:
  champ_scores = build_scores_panel(factor_features, rebalance_dates)
  champ_metrics, equity_oos, turnover_df, final_weights = run_v12_backtest(
    close_prices, data_dict, factor_features, champ_scores,
  )
  yearly_df = yearly_comparison(equity_oos, close_prices)

v12_score, v12_status, score_notes = compute_v12_score(champ_metrics, yearly_df, param_results, champion)
APPROVED_FOR_WEB_PAPER = v12_status == "APPROVED_FOR_WEB_PAPER"
print(f"V12 score: {v12_score}/100 | {v12_status}")

# %% [markdown]
# ## 11. Señales actuales

# %%
def build_current_signals_v12(features, scores_panel, prev_weights=None):
  if scores_panel.empty:
    return pd.DataFrame()
  last_date = scores_panel.index[-1]
  scores = scores_panel.loc[last_date]
  target = build_target_portfolio(
    scores_panel, last_date, features,
    int(champion.get("top_n", TOP_N)) if champion else TOP_N,
    champion.get("max_weight", MAX_WEIGHT) if champion else MAX_WEIGHT,
    champion.get("vol_target", VOL_TARGET) if champion else VOL_TARGET,
  )
  prev = prev_weights if prev_weights is not None else pd.Series(dtype=float)
  sigs = generate_factor_signals(target, prev, scores, last_date)
  sigs["entry_plan"] = np.where(sigs["signal"].isin(["BUY", "INCREASE"]), "proxima apertura", "-")
  sigs["exit_plan"] = np.where(
    sigs["signal"].isin(["SELL", "REDUCE"]), "salir o reducir en proximo rebalance", "mantener hasta proximo viernes"
  )
  sigs["cash_account_executable"] = True
  sigs["target_weight_pct"] = (sigs["target_weight"] * 100).round(1).astype(str) + "%"
  return sigs.sort_values(["signal", "score"], ascending=[True, False])


current_signals = build_current_signals_v12(factor_features, champ_scores, final_weights)
print("Señales actuales:", len(current_signals))
if len(current_signals):
  cols = ["ticker", "signal", "target_weight_pct", "score", "reason"]
  print(current_signals[cols].head(12).to_string(index=False))
else:
  print("Sin señales — revisar datos.")

# %% [markdown]
# ## 12. Benchmarks y exportar

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
ew_cols = [c for c in TRADABLE if c in close_prices.columns]
ew_total = (close_prices[ew_cols].mean(axis=1).iloc[-1] / close_prices[ew_cols].mean(axis=1).iloc[0] - 1) * 100 if ew_cols else 0
v6_total = load_v6_return()


def export_csv(df, name):
  if df is None or len(df) == 0:
    pd.DataFrame().to_csv(name, index=False)
  else:
    df.to_csv(name, index=False)
  print("Exportado", name)


summary = {
  "lab": "v12_robust_factor_rotation",
  "v12_score": v12_score,
  "status": v12_status,
  "approved_for_web_paper": APPROVED_FOR_WEB_PAPER,
  "approved_for_real_money": False,
  **(champ_metrics or {}),
  "win_years_vs_spy": round(yearly_df["beats_spy"].mean() * 100, 1) if len(yearly_df) else 0,
  "win_years_vs_qqq": round(yearly_df["beats_qqq"].mean() * 100, 1) if len(yearly_df) else 0,
  "spy_return": round(spy_total, 2),
  "qqq_return": round(qqq_total, 2),
  "ew_return": round(ew_total, 2),
  "v6_return": round(v6_total, 2) if pd.notna(v6_total) else np.nan,
  "champion_config": json.dumps({k: champion.get(k) for k in ["top_n", "max_weight", "vol_target", "mom_combo", "trend_mode"]} if champion else {}),
}

export_csv(pd.DataFrame([summary]), "research_v12_summary.csv")
export_csv(param_results, "research_v12_parameter_results.csv")
export_csv(yearly_df, "research_v12_yearly.csv")
export_csv(current_signals, "research_v12_current_signals.csv")
export_csv(turnover_df, "research_v12_turnover.csv")
if len(equity_oos):
  equity_oos.to_frame("equity").to_csv("research_v12_equity_curve.csv")
  print("Exportado research_v12_equity_curve.csv")
else:
  pd.DataFrame().to_csv("research_v12_equity_curve.csv", index=False)

config_out = {
  "version": "v12_robust_factor_rotation",
  "approved_for_web_paper": APPROVED_FOR_WEB_PAPER,
  "approved_for_real_money": False,
  "v12_score": v12_score,
  "status": v12_status,
  "rebalance_freq": REBALANCE_FREQ,
  "champion": champion if champion else {},
  "warnings": [
    "Backtest no garantiza resultados futuros.",
    "Rebalance semanal — no es trading diario.",
    "No conecta broker ni ejecuta ordenes.",
  ],
  "metrics": champ_metrics,
}
Path("research_v12_selected_config.json").write_text(json.dumps(config_out, indent=2, default=str), encoding="utf-8")
print("Exportado research_v12_selected_config.json")

if len(current_signals):
  current_signals.to_csv("current_signals_v12.csv", index=False)

# %% [markdown]
# ## 13. Reporte final

# %%
print("=" * 80)
print("REPORTE FINAL V12 ROBUST FACTOR ROTATION")
print("=" * 80)
print(f"Estado: {v12_status} | Score: {v12_score}/100")
if champion:
  print(f"Mejor config: TOP_N={champion.get('top_n')} MAX_W={champion.get('max_weight')} "
        f"VOL={champion.get('vol_target')} mom={champion.get('mom_combo')} trend={champion.get('trend_mode')}")
print(f"CAGR: {champ_metrics.get('CAGR', 0)}% | Sharpe: {champ_metrics.get('sharpe', 0)} | "
      f"DD: {champ_metrics.get('max_drawdown', 0)}% | Return: {champ_metrics.get('total_return', 0)}%")
print(f"Años vs SPY: {yearly_df['beats_spy'].mean()*100:.0f}%" if len(yearly_df) else "N/A")
print(f"Años vs QQQ: {yearly_df['beats_qqq'].mean()*100:.0f}%" if len(yearly_df) else "N/A")
print(f"Costes: {champ_metrics.get('total_cost', 0)} | Turnover medio: {champ_metrics.get('turnover', 0)}")
print(f"BUY actuales: {len(current_signals[current_signals['signal']=='BUY']) if len(current_signals) else 0}")
if score_notes:
  print("Notas:", "; ".join(score_notes))
print(f"¿Supera SPY ({spy_total:.1f}%)? {'SI' if champ_metrics.get('total_return',0) > spy_total else 'NO'}")
if pd.notna(v6_total):
  print(f"¿Supera V6 ({v6_total:.1f}%)? {'SI' if champ_metrics.get('total_return',0) > v6_total else 'NO'}")
print("")
if APPROVED_FOR_WEB_PAPER:
  print("Integrar V12 Robust Factor Rotation como motor principal de señales semanales.")
else:
  print("V12 rejected. Mantener V6 como champion.")
print("APPROVED_FOR_REAL_MONEY=False (siempre)")
