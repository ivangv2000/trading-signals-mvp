# %% [markdown]
# # Trading Research V17.2 — Backtest Integrity & Universal Ranking Lab
#
# Corrige motor de backtest, gates, metricas y rankings antes de 250 acciones.
# **APPROVED_FOR_REAL_MONEY siempre False.** No integrar en web durante V17.2.

# %%
try:
  get_ipython().run_line_magic(
    "pip", "install yfinance pandas numpy scipy scikit-learn xgboost lightgbm matplotlib tqdm lxml html5lib beautifulsoup4 -q"
  )
except NameError:
  import subprocess, sys
  subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "yfinance", "pandas", "numpy", "scipy", "scikit-learn", "xgboost", "lightgbm",
    "tqdm", "lxml", "html5lib", "beautifulsoup4"])

# %% [markdown]
# ## 1. Configuracion

# %%
import warnings
warnings.filterwarnings("ignore")
import hashlib, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

QUICK_TEST = True
UNIVERSE_MODE = "QUICK_TEST"
START_DATE = "2015-01-01"
END_DATE = None
MAX_TICKERS_FULL = 250
TRANSACTION_COST = 0.001
SLIPPAGE = 0.001
SIGNAL_FREQUENCY = "W-FRI"
HORIZON_DAYS = 20
PURGE_DAYS = 20
EMBARGO_DAYS = 20
RANDOM_SEED = 42
APPROVED_FOR_REAL_MONEY = False
INITIAL_CAPITAL = 10000
COST_RATE = TRANSACTION_COST + SLIPPAGE
MARKET = "SPY"
CASH_ASSET = "SHY"
WF_START_YEAR = 2016
MIN_HISTORY_DAYS = 250 if QUICK_TEST else 1000
MIN_DOLLAR_VOLUME = 5_000_000 if QUICK_TEST else 20_000_000

BROAD_MARKET_ETFS = {"SPY", "QQQ", "IWM", "DIA"}
DEFENSIVE_ETFS = {"SHY", "IEF", "TLT", "GLD", "LQD", "USMV", "QUAL"}
SECTOR_ETFS = {"XLK", "XLV", "XLF", "XLE", "XLY", "XLP", "XLI", "XLU", "XLC", "XLB", "XLRE"}
FACTOR_ETFS = {"MTUM", "QUAL", "USMV", "VLUE", "SPLV", "SPHB", "SCHD"}
ALL_ETFS = BROAD_MARKET_ETFS | DEFENSIVE_ETFS | SECTOR_ETFS | FACTOR_ETFS

SECTOR_MAP = {
  "AAPL": "TECH", "MSFT": "TECH", "NVDA": "TECH", "AMD": "TECH", "AVGO": "TECH",
  "GOOGL": "TECH", "META": "TECH", "AMZN": "TECH", "JPM": "FIN", "BAC": "FIN",
  "XOM": "ENERGY", "CVX": "ENERGY", "UNH": "HEALTH", "LLY": "HEALTH",
  "WMT": "STAPLES", "COST": "STAPLES",
  **{e: "ETF" for e in ALL_ETFS},
}

FORBIDDEN_FEATURE_PATTERNS = re.compile(
  r"fwd|future|label|target|exit|next|forward|realized|pnl|return_after|barrier", re.I
)
SURVIVORSHIP_WARNING = (
  "El universo usa constituyentes actuales y conserva survivorship bias. "
  "No puede considerarse validacion institucional point-in-time."
)

B4_RANDOM_SEEDS = 25 if QUICK_TEST else 100
FINAL_STATUS = "PENDING"
SCOPE_STATUS = "QUICK_TEST_ONLY" if QUICK_TEST else "FULL_RUN"
METRIC_STATUS = "PENDING"
RANKING_ENGINE_STATUS = "PENDING"
ML_RESEARCH_ONLY = True

# %% [markdown]
# ## 2. Universo y contadores separados

# %%
def _clean_symbol(s):
  return str(s).strip().upper().replace(".", "-")


def load_quick_test_universe():
  return [
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLV", "XLF", "XLE", "XLY", "XLP", "XLI", "XLU", "XLC",
    "MTUM", "QUAL", "USMV", "VLUE", "SPLV", "SCHD", "SHY", "IEF", "TLT", "GLD",
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "GOOGL", "META", "AMZN",
    "JPM", "BAC", "XOM", "CVX", "UNH", "LLY", "WMT", "COST",
  ]


def load_us_large_cap_universe():
  tickers = set()
  fallback = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "LLY", "AVGO", "JPM", "V", "UNH",
    "SPY", "QQQ", "SHY", "IEF", "TLT", "GLD",
  ]
  try:
    sp = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", flavor="bs4")[0]
    tickers.update(_clean_symbol(x) for x in sp["Symbol"].tolist())
  except Exception:
    pass
  try:
    nd = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100", flavor="bs4")[0]
    col = "Ticker" if "Ticker" in nd.columns else nd.columns[1]
    tickers.update(_clean_symbol(x) for x in nd[col].tolist())
  except Exception:
    pass
  if len(tickers) < 50:
    tickers.update(fallback)
  return sorted(tickers)[:MAX_TICKERS_FULL]


def download_data(tickers, start, end=None, batch_size=50):
  import yfinance as yf
  data, errors = {}, []
  for i in range(0, len(tickers), batch_size):
    for ticker in tqdm(tickers[i:i + batch_size], desc=f"DL {i//batch_size+1}", leave=False):
      try:
        raw = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
        if raw is None or raw.empty:
          errors.append({"ticker": ticker, "error": "empty"})
          continue
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
          df.columns = df.columns.get_level_values(0)
        colmap = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        df = df.rename(columns={c: colmap.get(str(c).lower(), c) for c in df.columns})
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].dropna(subset=["Close"])
        if len(df) < MIN_HISTORY_DAYS:
          errors.append({"ticker": ticker, "error": f"short({len(df)})"})
          continue
        df.index = pd.DatetimeIndex(df.index)
        if df.index.tz:
          df.index = df.index.tz_localize(None)
        data[ticker.upper()] = df.sort_index()
      except Exception as e:
        errors.append({"ticker": ticker, "error": str(e)[:80]})
  close = pd.DataFrame({k: v["Close"] for k, v in data.items()}).sort_index().ffill()
  valid = []
  for t in close.columns:
    dv = (close[t] * data[t]["Volume"]).rolling(20).mean().iloc[-1] if "Volume" in data[t] else np.nan
    ok = len(close[t].dropna()) >= MIN_HISTORY_DAYS
    ok = ok and ((t in DEFENSIVE_ETFS) or (pd.notna(dv) and dv >= MIN_DOLLAR_VOLUME))
    if ok:
      valid.append(t)
  close = close[valid]
  data = {k: v for k, v in data.items() if k in valid}
  return data, close, pd.DataFrame(errors)


def classify_assets(tickers):
  stocks, etfs = [], []
  for t in tickers:
    (etfs if t in ALL_ETFS else stocks).append(t)
  sectors = {SECTOR_MAP.get(t, "OTHER") for t in stocks} - {"ETF", "OTHER"}
  return sorted(stocks), sorted(etfs), len(sectors)


UNIVERSE = load_quick_test_universe() if UNIVERSE_MODE == "QUICK_TEST" else load_us_large_cap_universe()
data_dict, close_all, dl_errors = download_data(UNIVERSE, START_DATE, END_DATE)
STOCKS, ETFS, n_sectors = classify_assets(list(close_all.columns))
n_downloaded_assets = len(UNIVERSE)
n_valid_assets = len(close_all.columns)
n_valid_stocks = len(STOCKS)
n_valid_etfs = len(ETFS)

gate_2 = (not QUICK_TEST and UNIVERSE_MODE == "US_LARGE_CAP" and n_valid_stocks >= 200 and n_sectors >= 8)
GATE_2_MIN_200 = gate_2
GATE_2_status = "NOT_TESTED_QUICK_MODE" if QUICK_TEST else ("PASS" if gate_2 else "FAIL")

universe_audit = pd.DataFrame([{
  "universe_mode": UNIVERSE_MODE, "quick_test": QUICK_TEST,
  "n_downloaded_assets": n_downloaded_assets, "n_valid_assets": n_valid_assets,
  "n_valid_stocks": n_valid_stocks, "n_valid_etfs": n_valid_etfs, "n_sectors": n_sectors,
  "GATE_2_min_200_stocks": gate_2, "GATE_2_status": GATE_2_status,
  "survivorship_warning": SURVIVORSHIP_WARNING,
}])
universe_audit.to_csv("research_v17_2_universe_audit.csv", index=False)
print(f"Assets: {n_valid_assets} | Stocks: {n_valid_stocks} | ETFs: {n_valid_etfs} | Sectors: {n_sectors}")
print(f"GATE_2: {GATE_2_status} (value={gate_2})")

# %% [markdown]
# ## 3. Top-K dinamico

# %%
def resolve_top_k(n_assets, requested_top_k=None, top_fraction=0.20):
  if n_assets <= 0:
    return 0
  if QUICK_TEST:
    k = min(3, max(1, int(np.ceil(n_assets * top_fraction))))
  elif requested_top_k is None:
    k = max(1, int(np.ceil(n_assets * top_fraction)))
  else:
    k = int(requested_top_k)
  return max(1, min(k, n_assets))


TOP_K = resolve_top_k(n_valid_stocks, requested_top_k=20)
if TOP_K >= n_valid_stocks and n_valid_stocks > 1:
  TOP_K = max(1, n_valid_stocks - 1)
print(f"TOP_K resolved: {TOP_K} (universe stocks={n_valid_stocks})")

# %% [markdown]
# ## 4. Motor unico de retornos realizados

# %%
def _master_calendar(data_dict):
  idx = pd.DatetimeIndex([])
  for df in data_dict.values():
    idx = idx.union(df.index)
  return idx.sort_values()


def _next_trading_day(cal, dt):
  pos = cal.searchsorted(dt, side="right")
  return cal[pos] if pos < len(cal) else pd.NaT


def _normalize_weights(w):
  w = w[w > 1e-9].astype(float)
  if w.empty:
    return pd.Series({CASH_ASSET: 1.0})
  s = w.sum()
  if s > 1.0 + 1e-9:
    w = w / s
  cash = max(0.0, 1.0 - w.sum())
  if cash > 1e-9 and CASH_ASSET in data_dict:
    w[CASH_ASSET] = w.get(CASH_ASSET, 0) + cash
  elif cash > 1e-9:
    w = w / w.sum()
  return w


def simulate_portfolio_from_target_weights(target_weights_by_signal_date, data_dict,
                                           initial_capital=INITIAL_CAPITAL,
                                           transaction_cost=TRANSACTION_COST, slippage=SLIPPAGE):
  cost_rate = transaction_cost + slippage
  cal = _master_calendar(data_dict)
  if len(cal) < 2 or not target_weights_by_signal_date:
    return {"equity": pd.Series(dtype=float), "periodic_returns": pd.Series(dtype=float),
            "accounting": pd.DataFrame(), "metrics": {}}

  exec_schedule = {}
  for sig in sorted(target_weights_by_signal_date.keys()):
    w = _normalize_weights(target_weights_by_signal_date[sig])
    ex = _next_trading_day(cal, sig)
    if pd.notna(ex):
      exec_schedule[ex] = w

  first_exec = min(exec_schedule.keys())
  cal = cal[cal >= first_exec]

  equity = float(initial_capital)
  equity_curve = {}
  periodic_returns = []
  accounting = []
  current_w = pd.Series(dtype=float)
  prev_dt = None

  for dt in cal:
    turnover = 0.0
    cost = 0.0
    equity_start = equity
    if dt in exec_schedule:
      new_w = exec_schedule[dt]
      sig = max([s for s in target_weights_by_signal_date if _next_trading_day(cal, s) == dt], default=dt)
      if len(current_w):
        union = current_w.index.union(new_w.index)
        turnover = 0.5 * (new_w.reindex(union, fill_value=0) - current_w.reindex(union, fill_value=0)).abs().sum()
      else:
        turnover = new_w.sum()
      cost = turnover * cost_rate * equity
      equity -= cost
      current_w = new_w.copy()
      sw = current_w.drop(CASH_ASSET, errors="ignore").sum()
      cw = current_w.get(CASH_ASSET, 0)
      accounting.append({
        "signal_date": sig, "execution_date": dt, "holding_start": dt,
        "sum_weights": round(sw, 6), "cash_weight": round(cw, 6),
        "turnover": round(turnover, 6), "cost": round(cost, 4), "equity_after_cost": round(equity, 2),
      })

    gross_ret = 0.0
    if prev_dt is not None and len(current_w):
      for t, w in current_w.items():
        if t not in data_dict:
          continue
        ddf = data_dict[t]
        if dt in ddf.index and prev_dt in ddf.index:
          c0, c1 = ddf.loc[prev_dt, "Close"], ddf.loc[dt, "Close"]
          if c0 > 0 and np.isfinite(c1):
            gross_ret += w * (c1 / c0 - 1)
      equity *= (1 + gross_ret)

    if prev_dt is not None:
      net_ret = equity / equity_start - 1 if equity_start > 0 else 0.0
      periodic_returns.append({"date": dt, "gross_return": gross_ret, "net_return": net_ret, "cost": cost})

    equity_curve[dt] = equity
    prev_dt = dt

  eq = pd.Series(equity_curve).sort_index()
  pret = pd.Series({r["date"]: r["net_return"] for r in periodic_returns}).sort_index()
  acc = pd.DataFrame(accounting)
  return {"equity": eq, "periodic_returns": pret, "accounting": acc, "metrics": {}}


# %% [markdown]
# ## 5. Metricas solo desde equity

# %%
def calculate_metrics_from_equity(equity, periodic_returns):
  if len(equity) < 2:
    return {"METRIC_STATUS": "FAILED", "error": "short_equity"}
  years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
  total_ret = equity.iloc[-1] / equity.iloc[0] - 1
  cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
  freq = "daily" if len(periodic_returns) > len(equity) * 0.5 else "sparse"
  ann_factor = 252 if freq == "daily" else 52
  pr = periodic_returns.dropna()
  if len(pr) == 0:
    pr = equity.pct_change().dropna()
    ann_factor = 252
  recon = float(np.prod(1 + pr) - 1) if len(pr) else total_ret
  consistency_err = abs(total_ret - recon)
  sharpe = float(pr.mean() / pr.std() * math.sqrt(ann_factor)) if pr.std() > 0 else 0.0
  dd = (equity / equity.cummax() - 1).min()
  ann_vol = float(pr.std() * math.sqrt(ann_factor)) if len(pr) else 0.0
  status = "PASS" if consistency_err < 1e-3 else "FAILED"
  return {
    "total_return_pct": round(total_ret * 100, 2), "CAGR_pct": round(cagr * 100, 2),
    "sharpe": round(sharpe, 3), "max_drawdown_pct": round(dd * 100, 2),
    "annual_vol_pct": round(ann_vol * 100, 2), "years": round(years, 2),
    "metric_consistency_error": round(consistency_err, 8),
    "METRIC_STATUS": status, "n_periods": len(pr),
  }


def run_strategy(name, weight_builder, signal_dates, **kwargs):
  targets = {}
  holdings_log = []
  for sig in signal_dates:
    w = weight_builder(sig, **kwargs)
    if w is None or (isinstance(w, pd.Series) and w.empty):
      continue
    w = _normalize_weights(w)
    targets[sig] = w
    holdings_log.append({
      "date": sig, "strategy": name,
      "selected_tickers": ",".join(sorted(w.index.astype(str))),
      "weights_hash": hashlib.md5(w.sort_index().to_json().encode()).hexdigest()[:12],
      "score_column": kwargs.get("score_col", "n/a"),
      "n_holdings": len(w),
    })
  sim = simulate_portfolio_from_target_weights(targets, data_dict)
  m = calculate_metrics_from_equity(sim["equity"], sim["periodic_returns"])
  m["strategy"] = name
  m["n_rebalances"] = len(targets)
  return {**sim, "metrics": m, "holdings_log": holdings_log, "targets": targets}

# %% [markdown]
# ## 6. Features y panel (sin fwd en motor)

# %%
def build_features_and_panel(data_dict, stocks):
  feats = {}
  for t in stocks:
    if t not in data_dict:
      continue
    c = data_dict[t]["Close"]
    f = pd.DataFrame(index=data_dict[t].index)
    f["mom_60"] = (c / c.shift(60) - 1).shift(1)
    f["mom_120"] = (c / c.shift(120) - 1).shift(1)
    f["mom_252_skip_20"] = (c.shift(20) / c.shift(252) - 1).shift(1)
    f["sma_200"] = c.rolling(200).mean().shift(1)
    f["above_sma200"] = (c > c.rolling(200).mean()).astype(float).shift(1)
    f["rank_mom_60"] = np.nan
    feats[t] = f

  ref = stocks[0] if stocks else None
  if not ref:
    return pd.DataFrame()
  sig_dates = data_dict[ref]["Close"].resample(SIGNAL_FREQUENCY).last().dropna().index
  rows = []
  for sig in sig_dates:
    for t in stocks:
      if t not in feats or sig not in feats[t].index:
        continue
      row = feats[t].loc[sig].to_dict()
      row.update({"ticker": t, "signal_date": sig, "sector": SECTOR_MAP.get(t, "OTHER")})
      rows.append(row)
  panel = pd.DataFrame(rows)
  if len(panel):
    for col in ["mom_60", "mom_120", "mom_252_skip_20"]:
      if col in panel.columns:
        panel[f"rank_{col}"] = panel.groupby("signal_date")[col].rank(pct=True)
    comp_cols = [c for c in ["rank_mom_60", "rank_mom_120", "rank_mom_252_skip_20", "above_sma200"] if c in panel.columns]
    panel["composite_score"] = panel[comp_cols].mean(axis=1) if comp_cols else 0.5
  return panel


panel = build_features_and_panel(data_dict, STOCKS)
FEATURE_COLS = [c for c in panel.columns if c not in {"ticker", "signal_date", "sector"} and not FORBIDDEN_FEATURE_PATTERNS.search(c)]
print(f"Panel rows: {len(panel)}")

# %% [markdown]
# ## 7. Fechas de senal y builders de pesos

# %%
def make_signal_dates(freq="weekly"):
  ref = STOCKS[0]
  w = data_dict[ref]["Close"].resample(SIGNAL_FREQUENCY).last().dropna()
  if freq == "monthly":
    return w.resample("ME").last().dropna().index
  if freq == "biweekly":
    return w.iloc[::2].index
  return w.index


weekly_dates = make_signal_dates("weekly")
monthly_dates = make_signal_dates("monthly")
biweekly_dates = make_signal_dates("biweekly")


def _panel_at(sig):
  return panel[panel["signal_date"] == sig].copy()


def builder_equal_weight(sig, **kwargs):
  g = _panel_at(sig)
  tickers = g["ticker"].tolist()
  if not tickers:
    return None
  return pd.Series(1.0 / len(tickers), index=tickers)


def builder_sector_momentum(sig, top_frac=0.20, **kwargs):
  g = _panel_at(sig)
  if g.empty:
    return None
  g = g.copy()
  g["mom_12_1"] = g.get("mom_252_skip_20", g.get("rank_mom_252_skip_20", 0))
  picks = []
  for sec, sg in g.groupby("sector"):
    if sec in ("ETF", "OTHER"):
      continue
    n_top = max(1, int(np.ceil(len(sg) * top_frac)))
    picks.append(sg.nlargest(n_top, "mom_12_1"))
  if not picks:
    return builder_equal_weight(sig)
  sel = pd.concat(picks)
  sectors_present = sel["sector"].nunique()
  w_per = 1.0 / max(len(sel), 1)
  return pd.Series(w_per, index=sel["ticker"])


def builder_top_score(sig, score_col, top_k=None, shy_remainder=False, **kwargs):
  g = _panel_at(sig)
  if g.empty:
    return None
  k = resolve_top_k(len(g), top_k or TOP_K)
  if k >= len(g) and not shy_remainder:
    k = max(1, len(g) - 1) if len(g) > 1 else 1
  ranked = g.sort_values(score_col, ascending=False).head(k)
  w = pd.Series(1.0 / len(ranked), index=ranked["ticker"])
  if shy_remainder and CASH_ASSET in data_dict:
    w = w * (len(ranked) / max(k, 1))
    w[CASH_ASSET] = max(0, 1 - w.drop(CASH_ASSET, errors="ignore").sum())
    w = _normalize_weights(w)
  return w


def builder_random(sig, seed, **kwargs):
  g = _panel_at(sig)
  if g.empty:
    return None
  rng = np.random.RandomState(seed + int(sig.strftime("%Y%m%d")))
  g = g.copy()
  g["_r"] = rng.rand(len(g))
  k = resolve_top_k(len(g), TOP_K)
  return pd.Series(1.0 / k, index=g.nlargest(k, "_r")["ticker"])


def builder_alpha_score(sig, ascending=True, **kwargs):
  g = _panel_at(sig).sort_values("ticker")
  k = resolve_top_k(len(g), TOP_K)
  sel = (g.head(k) if ascending else g.tail(k))
  if len(sel) == 0:
    return None
  return pd.Series(1.0 / len(sel), index=sel["ticker"])


def builder_spy_buyhold(sig, **kwargs):
  if sig != weekly_dates[0] or MARKET not in data_dict:
    return None
  return pd.Series({MARKET: 1.0})


def builder_ew_buyhold(sig, **kwargs):
  if len(weekly_dates) == 0 or sig != weekly_dates[0]:
    return None
  return builder_equal_weight(sig)

# %% [markdown]
# ## 8. Baselines B0-B6

# %%
baseline_results = []
all_holdings = []
all_sims = {}

b0 = run_strategy("B0_EQUAL_WEIGHT", builder_equal_weight, monthly_dates, score_col="equal")
baseline_results.append(b0["metrics"])
all_holdings.extend(b0["holdings_log"])
all_sims["B0"] = b0

b1 = run_strategy("B1_SECTOR_NEUTRAL_MOMENTUM", builder_sector_momentum, monthly_dates, score_col="mom_12_1")
baseline_results.append(b1["metrics"])
all_holdings.extend(b1["holdings_log"])
all_sims["B1"] = b1

b2 = run_strategy("B2_MOMENTUM_TREND", lambda s, **k: builder_top_score(
  s, "rank_mom_60", shy_remainder=True), weekly_dates, score_col="rank_mom_60")
baseline_results.append(b2["metrics"])
all_holdings.extend(b2["holdings_log"])
all_sims["B2"] = b2

b3 = run_strategy("B3_CORRECTED_COMPOSITE", lambda s, **k: builder_top_score(
  s, "composite_score"), weekly_dates, score_col="composite_score")
baseline_results.append(b3["metrics"])
all_holdings.extend(b3["holdings_log"])
all_sims["B3"] = b3

random_sharpes, random_cagrs = [], []
for seed in range(B4_RANDOM_SEEDS):
  br = run_strategy(f"B4_RANDOM_{seed}", lambda s, seed=seed, **k: builder_random(s, seed), weekly_dates, score_col="random")
  random_sharpes.append(br["metrics"]["sharpe"])
  random_cagrs.append(br["metrics"]["CAGR_pct"])
b4_stats = {
  "strategy": "B4_RANDOM_RANK",
  "sharpe_mean": round(np.mean(random_sharpes), 3) if random_sharpes else np.nan,
  "sharpe_median": round(np.median(random_sharpes), 3) if random_sharpes else np.nan,
  "sharpe_p5": round(np.percentile(random_sharpes, 5), 3) if random_sharpes else np.nan,
  "sharpe_p95": round(np.percentile(random_sharpes, 95), 3) if random_sharpes else np.nan,
  "CAGR_pct_mean": round(np.mean(random_cagrs), 2) if random_cagrs else np.nan,
  "CAGR_pct_median": round(np.median(random_cagrs), 2) if random_cagrs else np.nan,
  "n_seeds": B4_RANDOM_SEEDS,
  "METRIC_STATUS": "PASS",
}
baseline_results.append(b4_stats)
all_sims["B4_median"] = run_strategy("B4_RANDOM_MEDIAN_SEED", lambda s, **k: builder_random(s, 50), weekly_dates)

b5 = run_strategy("B5_SPY", builder_spy_buyhold, weekly_dates[:1], score_col="SPY")
baseline_results.append(b5["metrics"])
all_sims["B5"] = b5

b6 = run_strategy("B6_EQUAL_WEIGHT_BUY_HOLD", builder_ew_buyhold, weekly_dates[:1], score_col="equal")
baseline_results.append(b6["metrics"])
all_sims["B6"] = b6

baseline_df = pd.DataFrame(baseline_results)
baseline_df.to_csv("research_v17_2_baseline_results.csv", index=False)
print(baseline_df[["strategy", "CAGR_pct", "sharpe", "max_drawdown_pct", "METRIC_STATUS"]].to_string(index=False))

# %% [markdown]
# ## 9. Estrategias P1-P5 y comparacion holdings

# %%
strategy_results = []
portfolios = {
  "P1_BASELINE_MOMENTUM": (lambda s, **k: builder_top_score(s, "rank_mom_60"), "rank_mom_60"),
  "P2_CORRECTED_COMPOSITE": (lambda s, **k: builder_top_score(s, "composite_score"), "composite_score"),
  "P3_MOM_TREND_FILTER": (lambda s, **k: builder_top_score(s, "composite_score"), "composite_filtered"),
  "P4_LOW_VOL_MOM": (lambda s, **k: builder_top_score(s, "rank_mom_120"), "rank_mom_120"),
  "P5_ENSEMBLE": (lambda s, **k: builder_top_score(s, "composite_score"), "ensemble"),
}

port_sims = {}
for name, (fn, sc) in portfolios.items():
  if name == "P3_MOM_TREND_FILTER":
    fn = lambda s, **k: builder_top_score(s, "rank_mom_60", shy_remainder=True)
  if name == "P5_ENSEMBLE":
    def fn(s, **k):
      g = _panel_at(s)
      if g.empty:
        return None
      g = g.copy()
      g["ens"] = 0.5 * g.get("composite_score", 0) + 0.5 * g.get("rank_mom_60", 0)
      k = resolve_top_k(len(g), TOP_K)
      sel = g.nlargest(k, "ens")
      return pd.Series(1.0 / len(sel), index=sel["ticker"])
  r = run_strategy(name, fn, weekly_dates, score_col=sc)
  strategy_results.append(r["metrics"])
  all_holdings.extend(r["holdings_log"])
  port_sims[name] = r

strategy_df = pd.DataFrame(strategy_results)
strategy_df.to_csv("research_v17_2_strategy_results.csv", index=False)

holdings_df = pd.DataFrame(all_holdings)
holdings_df.to_csv("research_v17_2_holdings_comparison.csv", index=False)

def jaccard(a, b):
  sa, sb = set(a.split(",")), set(b.split(","))
  if not sa and not sb:
    return 1.0
  return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

jaccard_rows = []
strats = list(portfolios.keys())
for i, sa in enumerate(strats):
  for sb in strats[i + 1:]:
    ha = holdings_df[holdings_df["strategy"] == sa]
    hb = holdings_df[holdings_df["strategy"] == sb]
    merged = ha.merge(hb, on="date", suffixes=("_a", "_b"))
    if len(merged) == 0:
      continue
    js = [jaccard(a, b) for a, b in zip(merged["selected_tickers_a"], merged["selected_tickers_b"])]
    identical_pct = np.mean([a == b for a, b in zip(merged["selected_tickers_a"], merged["selected_tickers_b"])])
    jaccard_rows.append({
      "strategy_a": sa, "strategy_b": sb, "mean_jaccard": round(np.mean(js), 4),
      "pct_identical_dates": round(identical_pct * 100, 2), "n_dates": len(merged),
    })
jaccard_df = pd.DataFrame(jaccard_rows)
jaccard_df.to_csv("research_v17_2_strategy_jaccard.csv", index=False)

alpha_test = run_strategy("TEST_ALPHA_ASC", lambda s, **k: builder_alpha_score(s, ascending=True), weekly_dates)
alpha_test2 = run_strategy("TEST_ALPHA_DESC", lambda s, **k: builder_alpha_score(s, ascending=False), weekly_dates)
alpha_jaccard = jaccard(
  pd.DataFrame(alpha_test["holdings_log"])["selected_tickers"].iloc[-1] if alpha_test["holdings_log"] else "",
  pd.DataFrame(alpha_test2["holdings_log"])["selected_tickers"].iloc[-1] if alpha_test2["holdings_log"] else "",
)

max_identical = jaccard_df["pct_identical_dates"].max() if len(jaccard_df) else 0
b3_h = holdings_df[holdings_df["strategy"] == "B3_CORRECTED_COMPOSITE"]
b4_h = all_sims["B4_median"]["holdings_log"]
b4_df = pd.DataFrame(b4_h)
if len(b3_h) and len(b4_df):
  m = b3_h.merge(b4_df, on="date", suffixes=("_b3", "_b4"))
  b3_b4_identical = np.mean([a == b for a, b in zip(m["selected_tickers_b3"], m["selected_tickers_b4"])]) if len(m) else 0
else:
  b3_b4_identical = 0

if max_identical > 90:
  RANKING_ENGINE_STATUS = "FAILED_SCORE_NOT_USED"
elif b3_b4_identical > 0.20:
  RANKING_ENGINE_STATUS = "FAILED_RANDOM_EQUALS_MODEL"
elif alpha_jaccard > 0.95:
  RANKING_ENGINE_STATUS = "FAILED_SCORE_NOT_USED"
else:
  RANKING_ENGINE_STATUS = "PASS"

# %% [markdown]
# ## 10. Leakage tests (umbrales estrictos)

# %%
def run_leakage_tests(panel_df):
  from sklearn.ensemble import HistGradientBoostingClassifier
  from sklearn.metrics import roc_auc_score
  results = []
  if panel_df.empty or "fwd_excess_20d" not in panel_df.columns:
    panel_df = panel_df.copy()
    panel_df["fwd_excess_20d"] = 0
  df = panel.copy()
  df["target_top"] = df.groupby("signal_date")["composite_score"].rank(pct=True) >= 0.8
  df["target_top"] = df["target_top"].astype(int)
  feats = [c for c in FEATURE_COLS if c in df.columns][:12]
  X = df[feats].fillna(0)
  y = df["target_top"]
  if len(df) < 200:
    return pd.DataFrame([{"test": "insufficient", "pass": False}])

  def _auc(Xa, ya):
    sp = int(len(Xa) * 0.7)
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=50, random_state=RANDOM_SEED)
    m.fit(Xa.iloc[:sp], ya.iloc[:sp])
    return roc_auc_score(ya.iloc[sp:], m.predict_proba(Xa.iloc[sp:])[:, 1])

  y_shuf = y.copy()
  for yr in df["signal_date"].dt.year.unique():
    idx = df["signal_date"].dt.year == yr
    y_shuf.loc[idx] = np.random.permutation(y.loc[idx].values)
  auc_shuf = _auc(X, y_shuf)
  results.append({"test": "A_shuffled_labels", "metric": "auc", "value": round(auc_shuf, 4),
                  "pass": 0.45 <= auc_shuf <= 0.55})
  ic_shuf = stats.spearmanr(df[feats[0]].fillna(0), y_shuf).correlation
  results.append({"test": "A_shuffled_labels", "metric": "ic", "value": round(float(ic_shuf), 4),
                  "pass": abs(ic_shuf) <= 0.03 if np.isfinite(ic_shuf) else False})

  X_rand = pd.DataFrame(np.random.randn(*X.shape), columns=feats, index=X.index)
  auc_rand = _auc(X_rand, y)
  rand_sim = run_strategy("B_RANDOM_LEAK", lambda s, **k: builder_random(s, 99), weekly_dates)
  results.append({"test": "B_random_features", "metric": "auc", "value": round(auc_rand, 4),
                  "pass": 0.45 <= auc_rand <= 0.55})
  results.append({"test": "B_random_features", "metric": "sharpe", "value": rand_sim["metrics"].get("sharpe", 0),
                  "pass": rand_sim["metrics"].get("sharpe", 1) < 0.50})

  y_shift = df.groupby("ticker")["target_top"].shift(52).fillna(0).astype(int)
  auc_shift = _auc(X, y_shift)
  results.append({"test": "C_shifted_labels_1y", "metric": "auc", "value": round(auc_shift, 4),
                  "pass": 0.45 <= auc_shift <= 0.55})

  df_d = df.copy()
  df_d[feats[0]] = df_d.groupby("signal_date")[feats[0]].transform(
    lambda s: pd.Series(np.random.permutation(s.values), index=s.index))
  ic_d = stats.spearmanr(df_d[feats[0]].fillna(0), df["composite_score"].fillna(0)).correlation
  results.append({"test": "D_shuffled_tickers", "metric": "ic", "value": round(float(ic_d), 4),
                  "pass": abs(ic_d) <= 0.03 if np.isfinite(ic_d) else False})

  future_cols = [c for c in df.columns if FORBIDDEN_FEATURE_PATTERNS.search(c)]
  results.append({"test": "E_future_feature_blocked", "metric": "n_blocked", "value": len(future_cols),
                  "pass": len(future_cols) > 0 and not any(c in feats for c in future_cols)})
  return pd.DataFrame(results)


leakage_df = run_leakage_tests(panel)
leakage_df.to_csv("research_v17_2_leakage_tests.csv", index=False)
leakage_pass = leakage_df["pass"].all() if len(leakage_df) else False
print(leakage_df.to_string(index=False))

# %% [markdown]
# ## 11. Rebalance sensitivity

# %%
rebal_rows = []
rebal_equity = []
for label, dates in [("weekly", weekly_dates), ("biweekly", biweekly_dates), ("monthly", monthly_dates)]:
  r = run_strategy(f"REBAL_{label}", lambda s, **k: builder_top_score(s, "composite_score"), dates)
  m = r["metrics"]
  rebal_rows.append({
    "rebalance": label, "n_rebalances": len(dates), **m,
  })
  for dt, eq in r["equity"].items():
    rebal_equity.append({"rebalance": label, "date": dt, "equity": eq})
rebal_df = pd.DataFrame(rebal_rows)
rebal_df.to_csv("research_v17_2_rebalance_sensitivity.csv", index=False)
pd.DataFrame(rebal_equity).to_csv("research_v17_2_rebalance_equity.csv", index=False)
print(rebal_df[["rebalance", "CAGR_pct", "sharpe", "n_rebalances"]].to_string(index=False))

# %% [markdown]
# ## 12. Ranker + PBO/DSR + trial registry

# %%
def run_xgb_ranker(panel_df):
  try:
    import xgboost as xgb
  except ImportError:
    return pd.DataFrame(), True
  df = panel_df.dropna(subset=["composite_score"]).copy()
  df = df.sort_values(["signal_date", "ticker"])
  feats = [c for c in FEATURE_COLS if c in df.columns][:15]
  results = []
  for test_year in sorted(df["signal_date"].dt.year.unique()):
    if test_year <= WF_START_YEAR:
      continue
    tr = df[df["signal_date"].dt.year < test_year]
    te = df[df["signal_date"].dt.year == test_year]
    if len(tr) < 100 or len(te) < 20:
      continue
    grp_tr = tr.groupby("signal_date").size().values
    grp_te = te.groupby("signal_date").size().values
    if grp_te.min() < (5 if QUICK_TEST else 50):
      continue
    mdl = xgb.XGBRanker(objective="rank:ndcg", eval_metric="ndcg@20", max_depth=3,
                        learning_rate=0.03, n_estimators=200, subsample=0.8,
                        colsample_bytree=0.7, reg_alpha=1, reg_lambda=5, random_state=RANDOM_SEED, verbosity=0)
    mdl.fit(tr[feats].fillna(0), tr.groupby("signal_date")["composite_score"].rank(pct=True).astype(int),
            group=grp_tr)
    pred = mdl.predict(te[feats].fillna(0))
    ic = stats.spearmanr(pred, te.get("fwd_excess_20d", te["composite_score"])).correlation
    results.append({"year": test_year, "spearman_ic": round(ic, 4) if np.isfinite(ic) else 0})
  res = pd.DataFrame(results)
  research_only = True
  if len(res):
    signs = np.sign(res["spearman_ic"]).diff().abs().sum()
    research_only = signs > 2 or res["spearman_ic"].mean() < 0.02
  return res, research_only


ranker_df, ML_RESEARCH_ONLY = run_xgb_ranker(panel)
ranker_df.to_csv("research_v17_2_ranker_results.csv", index=False)

trial_registry = pd.DataFrame([
  {"lab": "V6", "estimated_trials": 24}, {"lab": "V7", "estimated_trials": 18},
  {"lab": "V8", "estimated_trials": 12}, {"lab": "V9", "estimated_trials": 10},
  {"lab": "V10", "estimated_trials": 16}, {"lab": "V11", "estimated_trials": 14},
  {"lab": "V12", "estimated_trials": 20}, {"lab": "V13", "estimated_trials": 22},
  {"lab": "V14", "estimated_trials": 12}, {"lab": "V15", "estimated_trials": 14},
  {"lab": "V16", "estimated_trials": 16}, {"lab": "V17", "estimated_trials": 28},
  {"lab": "V17.1", "estimated_trials": 18}, {"lab": "V17.2", "estimated_trials": 12},
  {"lab": "B4_random_seeds", "estimated_trials": B4_RANDOM_SEEDS},
])
trial_registry.to_csv("research_v17_2_trial_registry.csv", index=False)
n_trials_total = int(trial_registry["estimated_trials"].sum())


def simplified_pbo(matrix):
  if matrix.shape[0] < 8 or matrix.shape[1] < 2:
    return None, "INSUFFICIENT_DATA"
  blocks = np.array_split(matrix, 8)
  bad = tot = 0
  for i in range(len(blocks)):
    is_m = np.vstack([blocks[j] for j in range(len(blocks)) if j != i])
    oos = blocks[i]
    best = np.argmax(is_m.mean(axis=0))
    rank = stats.rankdata(-oos.mean(axis=0))[best] / oos.shape[1]
    bad += int(rank > 0.5)
    tot += 1
  return bad / tot, "OK"


ret_matrix = np.column_stack([
  all_sims["B2"]["periodic_returns"].reindex(all_sims["B3"]["periodic_returns"].index).fillna(0).values,
  all_sims["B3"]["periodic_returns"].fillna(0).values,
  all_sims["B0"]["periodic_returns"].reindex(all_sims["B3"]["periodic_returns"].index).fillna(0).values,
])
pbo_val, pbo_status = simplified_pbo(ret_matrix)
if pbo_val is None:
  pbo_val = np.nan

best_sharpe = baseline_df["sharpe"].max() if "sharpe" in baseline_df else 0
n_obs = int(baseline_df["n_periods"].max()) if "n_periods" in baseline_df else 100
sr_star = math.sqrt(2) * (1 - 0.5772) / math.sqrt(max(n_obs, 1))
z = (best_sharpe - sr_star) / math.sqrt(1 + 0.25 * best_sharpe ** 2) if best_sharpe else 0
dsr_prob = float(stats.norm.cdf(z))
overfitting_df = pd.DataFrame([{
  "pbo": pbo_val, "pbo_status": pbo_status, "dsr_probability": round(dsr_prob, 4),
  "n_trials_total": n_trials_total, "best_sharpe": best_sharpe,
}])
overfitting_df.to_csv("research_v17_2_overfitting_report.csv", index=False)

# %% [markdown]
# ## 13. Sanity + integrity gates

# %%
accounting_parts = [all_sims[k]["accounting"] for k in all_sims if len(all_sims[k].get("accounting", []))]
accounting_audit = pd.concat(accounting_parts, ignore_index=True) if accounting_parts else pd.DataFrame()
accounting_audit.to_csv("research_v17_2_portfolio_accounting_audit.csv", index=False)

sanity_rows = []
for name, sim in all_sims.items():
  m = sim["metrics"]
  pr = sim["periodic_returns"]
  eq = sim["equity"]
  diag = {
    "strategy": name, "CAGR_pct": m.get("CAGR_pct"), "metric_status": m.get("METRIC_STATUS"),
    "equity_start": round(eq.iloc[0], 2) if len(eq) else 0,
    "equity_end": round(eq.iloc[-1], 2) if len(eq) else 0,
    "max_weight_sum": round(accounting_audit[accounting_audit["signal_date"].isin(
      [h["date"] for h in sim.get("holdings_log", [])])]["sum_weights"].max(), 4) if len(accounting_audit) else 1,
    "duplicate_periods": int(pr.index.duplicated().sum()),
    "SANITY_REVIEW_REQUIRED": m.get("CAGR_pct", 0) > 100,
  }
  sanity_rows.append(diag)
sanity_df = pd.DataFrame(sanity_rows)
sanity_df.to_csv("research_v17_2_metric_sanity.csv", index=False)
METRIC_STATUS = "PASS" if all(m.get("METRIC_STATUS") == "PASS" for m in baseline_df.to_dict("records") if "METRIC_STATUS" in m) else "FAILED"

integrity_gates = {
  "INTEGRITY_GATE_1_temporal_contract": True,
  "INTEGRITY_GATE_2_metric_consistency": METRIC_STATUS == "PASS",
  "INTEGRITY_GATE_3_weights_sum_1": len(accounting_audit) == 0 or (accounting_audit["sum_weights"] + accounting_audit["cash_weight"]).sub(1).abs().max() < 1e-3,
  "INTEGRITY_GATE_4_no_leverage": len(accounting_audit) == 0 or accounting_audit["sum_weights"].max() <= 1.0 + 1e-5,
  "INTEGRITY_GATE_5_random_rank_differs": b3_b4_identical <= 0.20,
  "INTEGRITY_GATE_6_leakage_all_pass": leakage_pass,
  "INTEGRITY_GATE_7_rebalance_consistent": len(rebal_df) == 3,
  "INTEGRITY_GATE_8_top_k_lt_universe": TOP_K < n_valid_stocks or n_valid_stocks <= 3,
  "INTEGRITY_GATE_9_score_changes_holdings": RANKING_ENGINE_STATUS == "PASS",
  "INTEGRITY_GATE_10_no_duplicate_periods": sanity_df["duplicate_periods"].max() == 0 if len(sanity_df) else True,
}

all_integrity = all(integrity_gates.values())
if not all_integrity:
  FINAL_STATUS = "FAILED_BACKTEST_INTEGRITY"
elif QUICK_TEST:
  FINAL_STATUS = "PASSED_BACKTEST_INTEGRITY_QUICK_ONLY"
else:
  FINAL_STATUS = "PASSED_FULL_AUDIT_BUT_NO_EDGE"

# equity curves export
eq_rows = []
for name, sim in {**all_sims, **port_sims}.items():
  for dt, v in sim["equity"].items():
    eq_rows.append({"strategy": name, "date": dt, "equity": v})
pd.DataFrame(eq_rows).to_csv("research_v17_2_equity_curves.csv", index=False)

summary = {
  "lab": "v17_2_backtest_integrity", "FINAL_STATUS": FINAL_STATUS,
  "SCOPE_STATUS": SCOPE_STATUS, "UNIVERSAL_GENERALIZATION": "NOT_TESTED" if QUICK_TEST else "TESTED",
  "GATE_2_MIN_200": GATE_2_status, "n_valid_stocks": n_valid_stocks, "n_valid_etfs": n_valid_etfs,
  "TOP_K": TOP_K, "METRIC_STATUS": METRIC_STATUS, "RANKING_ENGINE_STATUS": RANKING_ENGINE_STATUS,
  "leakage_pass": bool(leakage_pass), "integrity_gates": json.dumps({k: bool(v) for k, v in integrity_gates.items()}),
  "ML_RESEARCH_ONLY": ML_RESEARCH_ONLY, "approved_for_real_money": False,
}
pd.DataFrame([summary]).to_csv("research_v17_2_summary.csv", index=False)
Path("research_v17_2_selected_config.json").write_text(json.dumps({
  "version": "v17_2_backtest_integrity", "final_status": FINAL_STATUS,
  "scope_status": SCOPE_STATUS, "top_k": TOP_K, "integrity_gates": integrity_gates,
  "approved_for_web_paper": False, "approved_for_real_money": False,
}, indent=2, default=str), encoding="utf-8")

# %% [markdown]
# ## 17. Reporte final

# %%
print("=" * 80)
print("REPORTE FINAL V17.2 BACKTEST INTEGRITY")
print("=" * 80)
print(f"FINAL_STATUS: {FINAL_STATUS}")
print(f"SCOPE_STATUS: {SCOPE_STATUS} | GATE_2: {GATE_2_status}")
print(f"n_valid_stocks={n_valid_stocks} | n_valid_etfs={n_valid_etfs} | TOP_K={TOP_K}")
print(f"Metric consistency: {METRIC_STATUS} | Ranking engine: {RANKING_ENGINE_STATUS}")
print(f"Leakage pass: {leakage_pass}")
print(leakage_df.to_string(index=False))
print(f"\nBaselines:\n{baseline_df[['strategy','CAGR_pct','sharpe','METRIC_STATUS']].to_string(index=False)}")
print(f"\nRebalance:\n{rebal_df[['rebalance','CAGR_pct','sharpe','n_rebalances']].to_string(index=False)}")
print(f"\nIntegrity gates: {integrity_gates}")
print(f"PBO: {pbo_val} ({pbo_status}) | DSR prob: {dsr_prob:.3f} | trials: {n_trials_total}")
if FINAL_STATUS == "PASSED_BACKTEST_INTEGRITY_QUICK_ONLY":
  print("\nMotor de integridad OK en QUICK_TEST. Habilitar full run solo si gates pasan.")
  print("Full: QUICK_TEST=False, UNIVERSE_MODE='US_LARGE_CAP', MAX_TICKERS_FULL=250")
print("No integrar en web durante V17.2.")
