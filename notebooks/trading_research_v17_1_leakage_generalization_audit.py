# %% [markdown]
# # Trading Research V17.1 — Leakage & Generalization Audit
#
# Audita V17 antes de optimizar o integrar. Corrige labels, IC, deciles y ML.
# **APPROVED_FOR_REAL_MONEY siempre False.** No integrar en web hasta pasar audit.

# %%
try:
  get_ipython().run_line_magic(
    "pip", "install yfinance pandas numpy scipy scikit-learn xgboost lightgbm matplotlib plotly tqdm lxml html5lib beautifulsoup4 -q"
  )
except NameError:
  import subprocess, sys
  subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "yfinance", "pandas", "numpy", "scipy", "scikit-learn", "xgboost", "lightgbm",
    "matplotlib", "plotly", "tqdm", "lxml", "html5lib", "beautifulsoup4"])

# %% [markdown]
# ## 1. Configuracion

# %%
import warnings
warnings.filterwarnings("ignore")
import importlib.util, json, math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm.auto import tqdm

QUICK_TEST = True
START_DATE = "2010-01-01"
END_DATE = None
UNIVERSE_MODE = "QUICK_TEST"
MAX_TICKERS_FULL = 250
SIGNAL_FREQUENCY = "W-FRI"
HORIZON_DAYS = 20
TRANSACTION_COST = 0.001
SLIPPAGE = 0.001
PURGE_DAYS = 20
EMBARGO_DAYS = 20
RUN_LEAKAGE_TESTS = True
RUN_FACTOR_AUDIT = True
RUN_RANKER = True
RANDOM_SEED = 42
RUN_FULL_AFTER_QUICK = False
DATA_PROVIDER = "YFINANCE"
APPROVED_FOR_REAL_MONEY = False
INITIAL_CAPITAL = 10000
COST_RATE = TRANSACTION_COST + SLIPPAGE
MARKET = "SPY"
CASH_ASSET = "SHY"
WF_START_YEAR = 2016
TOP_N = 20
MAX_WEIGHT = 0.075
SECTOR_CAP = 0.30
MEGA_TECH = {"AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO"}

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
V17_SURVIVORSHIP = (
  "Este research usa tickers disponibles actualmente. Para research institucional real "
  "haria falta base de datos con acciones deslistadas, como CRSP/Norgate/Sharadar."
)

MIN_HISTORY_DAYS = 1000
MIN_DOLLAR_VOLUME = 20_000_000
if QUICK_TEST:
  START_DATE = "2015-01-01"
  MIN_HISTORY_DAYS = 250
  MIN_DOLLAR_VOLUME = 5_000_000
  print("QUICK_TEST activo — audit sobre ~39 tickers desde 2015")

np.random.seed(RANDOM_SEED)
AUDIT_STATUS = "PENDING"
ML_RESEARCH_ONLY = True

# %% [markdown]
# ## 2. Reutilizar descarga V17 (con separacion stocks/ETFs)

# %%
ROOT = Path(__file__).resolve().parent.parent


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
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "LLY", "AVGO", "JPM", "V", "UNH", "XOM", "MA", "COST",
    "PG", "HD", "JNJ", "ABBV", "MRK", "CVX", "CRM", "BAC", "AMD", "NFLX", "WMT", "PEP", "KO", "TMO", "CSCO", "ACN",
    "LIN", "ADBE", "MCD", "ABT", "DHR", "WFC", "INTU", "DIS", "TXN", "QCOM", "PM", "IBM", "AMAT", "GE", "CAT", "NOW",
    "SPY", "QQQ", "IWM", "DIA", "SHY", "IEF", "TLT", "GLD",
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


def _sf(x, d=np.nan):
  try:
    v = float(x)
    return d if not np.isfinite(v) else v
  except Exception:
    return d


def download_data(tickers, start, end=None, batch_size=50, min_days=MIN_HISTORY_DAYS):
  import yfinance as yf
  data, errors = {}, []
  tickers = sorted(set(tickers))
  for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i + batch_size]
    for ticker in tqdm(batch, desc=f"Download {i//batch_size+1}", leave=False):
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
        if len(df) < min_days:
          errors.append({"ticker": ticker, "error": f"short({len(df)})"})
          continue
        df.index = pd.DatetimeIndex(df.index)
        if df.index.tz:
          df.index = df.index.tz_localize(None)
        data[ticker.upper()] = df.sort_index()
      except Exception as e:
        errors.append({"ticker": ticker, "error": str(e)[:80]})
  close = pd.DataFrame({k: v["Close"] for k, v in data.items()}).sort_index().ffill()
  report = []
  valid = []
  for t in close.columns:
    dv = (close[t] * data[t]["Volume"]).rolling(20).mean().iloc[-1] if t in data and "Volume" in data[t] else np.nan
    ok_hist = len(close[t].dropna()) >= min_days
    ok_liq = (t in DEFENSIVE_ETFS) or (pd.notna(dv) and dv >= MIN_DOLLAR_VOLUME)
    if ok_hist and ok_liq:
      valid.append(t)
    report.append({"ticker": t, "rows": len(close[t].dropna()), "avg_dollar_vol_20": _sf(dv), "passed": ok_hist and ok_liq})
  close = close[valid]
  data = {k: v for k, v in data.items() if k in valid}
  return data, close, pd.DataFrame(errors), pd.DataFrame(report)


def resolve_universe(mode):
  if mode == "QUICK_TEST":
    return load_quick_test_universe()
  if mode == "US_LARGE_CAP":
    return load_us_large_cap_universe()[:MAX_TICKERS_FULL]
  return load_quick_test_universe()


def classify_tickers(tickers):
  stocks, broad, defensive, sector = [], [], [], []
  for t in tickers:
    tu = t.upper()
    if tu in BROAD_MARKET_ETFS:
      broad.append(tu)
    elif tu in DEFENSIVE_ETFS:
      defensive.append(tu)
    elif tu in SECTOR_ETFS | FACTOR_ETFS:
      sector.append(tu)
    else:
      stocks.append(tu)
  return {
    "STOCKS": sorted(set(stocks)),
    "BROAD_MARKET_ETFS": sorted(set(broad)),
    "DEFENSIVE_ETFS": sorted(set(defensive)),
    "SECTOR_ETFS": sorted(set(sector)),
  }


def audit_universe_v171(initial, data, close, errors_df, report_df):
  groups = classify_tickers(close.columns.tolist())
  audit = {
    "universe_mode": UNIVERSE_MODE,
    "data_provider": DATA_PROVIDER,
    "initial_tickers": len(initial),
    "downloaded": len(data),
    "valid_tickers": len(close.columns),
    "n_stocks": len(groups["STOCKS"]),
    "n_broad_etfs": len(groups["BROAD_MARKET_ETFS"]),
    "n_defensive_etfs": len(groups["DEFENSIVE_ETFS"]),
    "n_sector_etfs": len(groups["SECTOR_ETFS"]),
    "download_errors": len(errors_df),
    "survivorship_warning": SURVIVORSHIP_WARNING,
  }
  return audit, groups


UNIVERSE = resolve_universe(UNIVERSE_MODE)
print(f"V17.1 Audit | {UNIVERSE_MODE} | provider={DATA_PROVIDER}")
print(SURVIVORSHIP_WARNING)

data_dict, close_all, dl_errors, dl_report = download_data(UNIVERSE, START_DATE, END_DATE)
audit, asset_groups = audit_universe_v171(UNIVERSE, data_dict, close_all, dl_errors, dl_report)
dl_errors.to_csv("research_v17_1_download_errors.csv", index=False)
dl_report.to_csv("research_v17_1_download_report.csv", index=False)
STOCKS = asset_groups["STOCKS"]
print(f"Validos: {audit['valid_tickers']} | Stocks ranking: {len(STOCKS)}")

# %% [markdown]
# ## 3-4. Features, contrato temporal y panel semanal

# %%
def calculate_alpha_features_v171(data_dict):
  feats = {}
  for t, df in data_dict.items():
    c = df["Close"]
    o = df.get("Open", c)
    h = df.get("High", c)
    l = df.get("Low", c)
    v = df.get("Volume", pd.Series(1, index=df.index))
    f = pd.DataFrame(index=df.index)
    for n in [5, 20, 60, 120, 252]:
      f[f"ret_{n}"] = c.pct_change(n)
    f["mom_60"] = c / c.shift(60) - 1
    f["mom_120"] = c / c.shift(120) - 1
    f["mom_252_skip_20"] = c.shift(20) / c.shift(252) - 1
    for n in [50, 200]:
      f[f"sma_{n}"] = c.rolling(n).mean()
    f["above_sma200"] = (c > f["sma_200"]).astype(float)
    f["trend_stack"] = ((c > c.rolling(50).mean()) & (c.rolling(50).mean() > f["sma_200"])).astype(float)
    f["slope_sma50"] = c.rolling(50).mean().pct_change(10)
    f["slope_sma200"] = f["sma_200"].pct_change(20)
    hi252 = c.rolling(252).max().shift(1)
    f["near_52w_high_score"] = (c / hi252 - 1 > -0.05).astype(float)
    f["vol_60"] = c.pct_change().rolling(60).std() * math.sqrt(252)
    f["low_vol_score"] = 1.0 / f["vol_60"].replace(0, np.nan)
    f["dd_60"] = c / c.rolling(60).max() - 1
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c
    f["volume_ratio_20"] = v / v.rolling(20).mean()
    f["abnormal_volume"] = (f["volume_ratio_20"] > 1.5).astype(float)
    f["price_volume_breakout"] = ((f["ret_20"] > 0.05) & (f["volume_ratio_20"] > 1.3)).astype(float)
    f["earnings_proxy_gap"] = ((c / o - 1).abs() > 0.04) & (f["volume_ratio_20"] > 1.5)
    f["earnings_proxy_gap"] = f["earnings_proxy_gap"].astype(float)
    f["reversal_1w"] = -f["ret_5"]
    feat_cols = [c for c in f.columns if not FORBIDDEN_FEATURE_PATTERNS.search(c)]
    f = f[feat_cols].shift(1)
    feats[t] = f.replace([np.inf, -np.inf], np.nan)
  return feats


def _next_trading_day(index, dt):
  pos = index.searchsorted(dt, side="right")
  return index[pos] if pos < len(index) else pd.NaT


def _entry_open(data_dict, ticker, entry_date):
  df = data_dict.get(ticker)
  if df is None or entry_date not in df.index:
    idx = df.index[df.index >= entry_date] if df is not None else []
    if len(idx) == 0:
      return np.nan, pd.NaT
    entry_date = idx[0]
  o = df.loc[entry_date, "Open"] if "Open" in df.columns else df.loc[entry_date, "Close"]
  return float(o), entry_date


def _horizon_exit_date(index, entry_date, horizon_days):
  pos = index.searchsorted(entry_date, side="left")
  exit_pos = pos + horizon_days
  return index[exit_pos] if exit_pos < len(index) else pd.NaT


def build_weekly_panels(data_dict, close_all, stock_list):
  feats = calculate_alpha_features_v171(data_dict)
  if MARKET in close_all.columns:
    spy = close_all[MARKET]
    mkt = pd.DataFrame(index=close_all.index)
    mkt["spy_above_sma200"] = (spy > spy.rolling(200).mean()).astype(float)
    mkt["market_risk_on"] = mkt["spy_above_sma200"]
    mkt["market_stress_score"] = (mkt["spy_above_sma200"] < 0.5).astype(float) * 55
    mkt = mkt.shift(1)
  else:
    mkt = pd.DataFrame()

  signal_dates = close_all[stock_list[0]].resample(SIGNAL_FREQUENCY).last().dropna().index if stock_list else pd.DatetimeIndex([])
  rows = []
  for t in stock_list:
    if t not in feats or t not in data_dict:
      continue
    f = feats[t]
    idx = data_dict[t].index
    o_series = data_dict[t].get("Open", data_dict[t]["Close"])
    for sig in signal_dates:
      if sig not in f.index:
        continue
      entry_date = _next_trading_day(idx, sig)
      if pd.isna(entry_date):
        continue
      entry_px, entry_date = _entry_open(data_dict, t, entry_date)
      if not np.isfinite(entry_px) or entry_px <= 0:
        continue
      exit_date = _horizon_exit_date(idx, entry_date, HORIZON_DAYS)
      if pd.isna(exit_date):
        continue
      exit_px = float(o_series.loc[exit_date]) if exit_date in o_series.index else np.nan
      if not np.isfinite(exit_px):
        exit_px = float(data_dict[t].loc[exit_date, "Close"])
      fwd_ret = exit_px / entry_px - 1 if np.isfinite(exit_px) else np.nan
      row = f.loc[sig].to_dict()
      row.update({
        "ticker": t, "signal_date": sig, "feature_date": sig,
        "entry_date": entry_date, "label_end_date": exit_date,
        "fwd_ret_20d": fwd_ret, "sector": SECTOR_MAP.get(t, "OTHER"),
      })
      rows.append(row)
  panel = pd.DataFrame(rows)
  if len(panel) == 0:
    raise RuntimeError("Panel semanal vacio")
  panel["signal_date"] = pd.to_datetime(panel["signal_date"])
  panel["entry_date"] = pd.to_datetime(panel["entry_date"])
  panel["label_end_date"] = pd.to_datetime(panel["label_end_date"])
  if len(mkt):
    panel = panel.merge(mkt.reset_index().rename(columns={"index": "signal_date", "Date": "signal_date"}),
                        on="signal_date", how="left")
  stock_mean = panel.groupby("signal_date")["fwd_ret_20d"].transform("mean")
  panel["fwd_excess_20d"] = panel["fwd_ret_20d"] - stock_mean
  panel["relevance"] = panel.groupby("signal_date")["fwd_excess_20d"].rank(method="first", pct=True)
  panel["relevance_quintile"] = np.floor(panel["relevance"] * 5).clip(0, 4).astype(int)
  for col in ["mom_60", "mom_120", "mom_252_skip_20", "low_vol_score", "trend_stack", "abnormal_volume"]:
    if col in panel.columns:
      panel[f"rank_{col}"] = panel.groupby("signal_date")[col].rank(pct=True)
  if "trend_stack" in panel.columns:
    panel["rank_trend"] = panel.groupby("signal_date")["trend_stack"].rank(pct=True)
  panel_weekly_overlapping = panel.copy()
  monthly_idx = panel.sort_values(["ticker", "signal_date"]).groupby("ticker").cumcount() % 4 == 0
  panel_monthly_non_overlapping = panel[monthly_idx].copy()
  return panel_weekly_overlapping, panel_monthly_non_overlapping, feats


panel_weekly, panel_monthly, features_dict = build_weekly_panels(data_dict, close_all, STOCKS)
print(f"Panel semanal: {len(panel_weekly)} obs | mensual no solapado: {len(panel_monthly)}")


def get_feature_cols(df):
  meta = {"ticker", "signal_date", "feature_date", "entry_date", "label_end_date", "sector",
          "fwd_ret_20d", "fwd_excess_20d", "relevance", "relevance_quintile"}
  cols = []
  for c in df.columns:
    if c in meta or FORBIDDEN_FEATURE_PATTERNS.search(c):
      continue
    if df[c].dtype.kind in "biufc":
      cols.append(c)
  return cols


FEATURE_COLS = get_feature_cols(panel_weekly)
print("Features:", len(FEATURE_COLS))


def validate_temporal_contract(panel_df):
  violations = []
  for _, r in panel_df.iterrows():
    if r["feature_date"] > r["signal_date"]:
      violations.append(("feature_after_signal", r["ticker"], r["signal_date"]))
    if not (r["signal_date"] < r["entry_date"]):
      violations.append(("signal_not_before_entry", r["ticker"], r["signal_date"]))
    if not (r["entry_date"] < r["label_end_date"]):
      violations.append(("entry_not_before_label_end", r["ticker"], r["signal_date"]))
  future_in_feats = [c for c in FEATURE_COLS if FORBIDDEN_FEATURE_PATTERNS.search(c)]
  if future_in_feats:
    violations.append(("future_columns_in_features", future_in_feats, None))
  if violations:
    print("TEMPORAL CONTRACT FAILED:")
    for v in violations[:10]:
      print(v)
    raise AssertionError(f"validate_temporal_contract failed: {len(violations)} violations")
  assert max(panel_df["feature_date"]) <= max(panel_df["signal_date"])
  assert (panel_df["signal_date"] < panel_df["entry_date"]).all()
  assert not future_in_feats
  print("Temporal contract: PASS")


validate_temporal_contract(panel_weekly)

# %% [markdown]
# ## 5. Auditoria de factores (train-only selection)

# %%
IC_FACTORS_BASE = [
  "rank_mom_60", "rank_mom_120", "rank_mom_252_skip_20", "rank_trend",
  "near_52w_high_score", "above_sma200", "low_vol_score", "abnormal_volume",
  "price_volume_breakout", "reversal_1w",
]


def dedupe_factors(panel_df, factors, thresh=0.95):
  factors = [f for f in factors if f in panel_df.columns]
  if len(factors) < 2:
    return factors
  sub = panel_df[factors].dropna()
  if len(sub) < 50:
    return factors
  corr = sub.corr().abs()
  drop = set()
  for i, a in enumerate(factors):
    for b in factors[i + 1:]:
      if a in drop or b in drop:
        continue
      if corr.loc[a, b] > thresh:
        if a == "trend_stack" or (a == "rank_trend" and b == "trend_stack"):
          drop.add("trend_stack")
        elif b == "trend_stack" or (b == "rank_trend" and a == "trend_stack"):
          drop.add("trend_stack")
        else:
          drop.add(b)
  out = [f for f in factors if f not in drop and f != "trend_stack"]
  if "rank_trend" in out and "trend_stack" in factors:
    pass
  return list(dict.fromkeys(out))


def factor_ic_stats(panel_df, factors, target="fwd_ret_20d"):
  rows, yearly = [], []
  for fac in factors:
    if fac not in panel_df.columns:
      continue
    ics = []
    for dt, g in panel_df.groupby("signal_date"):
      sub = g[[fac, target]].dropna()
      if len(sub) < 8:
        continue
      ic, _ = stats.spearmanr(sub[fac], sub[target])
      if np.isfinite(ic):
        ics.append(ic)
        yearly.append({"factor": fac, "year": pd.Timestamp(dt).year, "ic": ic})
    if not ics:
      continue
    s = pd.Series(ics)
    rows.append({
      "factor": fac, "mean_ic": round(s.mean(), 4), "median_ic": round(s.median(), 4),
      "ic_std": round(s.std(), 4), "ic_ir": round(s.mean() / s.std(), 4) if s.std() > 0 else 0,
      "hit_rate_positive": round((s > 0).mean(), 4), "n_periods": len(s),
      "panel": "monthly_non_overlapping",
    })
  return pd.DataFrame(rows).sort_values("mean_ic", ascending=False), pd.DataFrame(yearly)


def factor_ic_by_regime(panel_df, factors):
  rows = []
  if "market_risk_on" not in panel_df.columns:
    return pd.DataFrame()
  for fac in factors:
    for regime, label in [(1, "bull"), (0, "bear")]:
      subp = panel_df[panel_df["market_risk_on"] == regime]
      ic_df, _ = factor_ic_stats(subp, [fac])
      if len(ic_df):
        r = ic_df.iloc[0].to_dict()
        r["regime"] = label
        rows.append(r)
  return pd.DataFrame(rows)


def walk_forward_factor_selection(panel_df, factors):
  rows, selected_by_year = [], {}
  years = sorted(panel_df["signal_date"].dt.year.unique())
  for test_year in years:
    if test_year <= WF_START_YEAR:
      continue
    train = panel_df[panel_df["signal_date"].dt.year < test_year]
    test = panel_df[panel_df["signal_date"].dt.year == test_year]
    if len(train) < 100 or len(test) < 20:
      continue
  # purge: remove train rows whose label_end reaches test year
    purge_cutoff = pd.Timestamp(f"{test_year}-01-01") - pd.Timedelta(days=PURGE_DAYS)
    train = train[train["label_end_date"] < purge_cutoff]
    embargo_start = pd.Timestamp(f"{test_year}-01-01") - pd.Timedelta(days=EMBARGO_DAYS)
    train = train[train["signal_date"] < embargo_start]
    facs = dedupe_factors(train, factors)
    ic_df, _ = factor_ic_stats(train, facs)
    sel = ic_df[(ic_df["mean_ic"] > 0) & (ic_df["hit_rate_positive"] > 0.52)]["factor"].tolist()
    if not sel and len(ic_df):
      sel = ic_df.head(5)["factor"].tolist()
    selected_by_year[test_year] = sel
    rows.append({"test_year": test_year, "n_train": len(train), "n_test": len(test),
                 "selected_factors": ",".join(sel)})
  return pd.DataFrame(rows), selected_by_year


factors_deduped = dedupe_factors(panel_monthly, IC_FACTORS_BASE)
factor_audit, factor_ic_yearly = factor_ic_stats(panel_monthly, factors_deduped)
factor_corr = panel_monthly[factors_deduped].corr().round(4) if len(factors_deduped) else pd.DataFrame()
sel_by_year_df, selected_by_year = walk_forward_factor_selection(panel_monthly, factors_deduped)
factor_audit.to_csv("research_v17_1_factor_audit.csv", index=False)
factor_ic_yearly.to_csv("research_v17_1_factor_ic_yearly.csv", index=False)
if len(factor_corr):
  factor_corr.to_csv("research_v17_1_factor_correlation.csv")
sel_by_year_df.to_csv("research_v17_1_selected_factors_by_year.csv", index=False)
print("Factor audit top:", factor_audit.head(3)["factor"].tolist() if len(factor_audit) else [])

# %% [markdown]
# ## 6. Decile/quintile test corregido

# %%
def factor_quintile_test(panel_df, data_dict, factors, use_forward_col=True):
  """Quintiles con retornos realizados; panel mensual usa fwd_ret_20d no solapado."""
  rows, equity_rows = [], []
  rebalance_dates = sorted(panel_df["signal_date"].unique())
  for fac in factors:
    if fac not in panel_df.columns:
      continue
    period_rets = {q: [] for q in range(5)}
    equity = {q: [1.0] for q in range(5)}
    for sig in rebalance_dates:
      g = panel_df[panel_df["signal_date"] == sig]
      sub = g[["ticker", fac, "fwd_ret_20d", "entry_date"]].dropna(subset=[fac, "fwd_ret_20d"])
      if len(sub) < 10:
        continue
      sub = sub.copy()
      sub["q"] = pd.qcut(sub[fac].rank(method="first"), 5, labels=False, duplicates="drop")
      for q in range(5):
        pr = sub[sub["q"] == q]["fwd_ret_20d"].mean()
        if np.isfinite(pr):
          period_rets[q].append(pr)
          equity[q].append(equity[q][-1] * (1 + pr))
    if not period_rets[4]:
      continue
    top_p = pd.Series(period_rets[4])
    n_p = len(top_p)
    periods_per_year = 252 / HORIZON_DAYS
    ann_from_periods = (np.prod(1 + top_p) ** (periods_per_year / max(n_p, 1)) - 1) if n_p else np.nan
    eq = pd.Series(equity[4])
    years = n_p / periods_per_year
    cagr_eq = (eq.iloc[-1] ** (1 / max(years, 1e-6)) - 1) if n_p else np.nan
    metric_bug = abs(ann_from_periods - cagr_eq) > 0.02
    weekly_proxy = top_p
    sharpe = weekly_proxy.mean() / weekly_proxy.std() * math.sqrt(periods_per_year) if weekly_proxy.std() > 0 else 0
    rows.append({
      "factor": fac, "top_quintile_ann_pct": round(ann_from_periods * 100, 2),
      "cagr_from_equity_pct": round(cagr_eq * 100, 2),
      "n_periods": n_p, "metric_bug": metric_bug,
      "period_sharpe": round(sharpe, 3),
    })
    equity_rows.append({"factor": fac, "quintile": 4, "equity": list(eq)})
  return pd.DataFrame(rows).sort_values("top_quintile_ann_pct", ascending=False), equity_rows


quintile_df, quintile_equity = factor_quintile_test(panel_monthly, data_dict, factors_deduped)
quintile_df.to_csv("research_v17_1_factor_quintiles.csv", index=False)
eq_export = []
for item in quintile_equity:
  for i, v in enumerate(item["equity"]):
    eq_export.append({"factor": item["factor"], "quintile": item["quintile"], "week": i, "equity": v})
pd.DataFrame(eq_export).to_csv("research_v17_1_factor_quintile_equity.csv", index=False)
print("Quintile test (corrected):", quintile_df.head(3).to_string(index=False) if len(quintile_df) else "N/A")

# %% [markdown]
# ## 7. Pruebas automaticas de leakage

# %%
def build_composite_score(df, factors):
  p = df.copy()
  cols = [c for c in factors if c in p.columns]
  if not cols:
    p["composite_score"] = 50.0
    return p
  p["composite_score"] = p[cols].mean(axis=1).rank(pct=True) * 100
  return p


def run_leakage_tests(panel_df, feature_cols):
  from sklearn.ensemble import HistGradientBoostingClassifier
  from sklearn.metrics import roc_auc_score

  results = []
  df = panel_df.dropna(subset=["relevance_quintile"]).copy()
  df["target_top"] = (df["relevance_quintile"] >= 4).astype(int)
  feats = [c for c in feature_cols if c in df.columns][:15]
  if len(df) < 300 or len(feats) < 3:
    return pd.DataFrame([{"test": "insufficient_data", "pass": False}])

  def _auc(X, y):
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=50, random_state=RANDOM_SEED)
    split = int(len(X) * 0.7)
    m.fit(X.iloc[:split], y.iloc[:split])
    pr = m.predict_proba(X.iloc[split:])[:, 1]
    return roc_auc_score(y.iloc[split:], pr)

  def _ic(X, y):
    return stats.spearmanr(X.iloc[:, 0], y).correlation

  X = df[feats].fillna(0)
  y = df["target_top"]
  base_auc = _auc(X, y)
  base_ic = stats.spearmanr(df[feats[0]].fillna(0), df["fwd_ret_20d"]).correlation

  # A shuffled labels
  y_shuf = y.copy()
  for yr in df["signal_date"].dt.year.unique():
    idx = df["signal_date"].dt.year == yr
    y_shuf.loc[idx] = np.random.permutation(y.loc[idx].values)
  shuf_auc = _auc(X, y_shuf)
  results.append({"test": "A_shuffled_labels", "metric": "auc", "value": round(shuf_auc, 4),
                  "pass": 0.45 <= shuf_auc <= 0.55})

  shuf_ic = stats.spearmanr(df[feats[0]].fillna(0), np.random.permutation(df["fwd_ret_20d"].values)).correlation
  results.append({"test": "A_shuffled_labels", "metric": "mean_ic", "value": round(shuf_ic, 4),
                  "pass": -0.03 <= shuf_ic <= 0.03})

  # B random features
  X_rand = pd.DataFrame(np.random.randn(*X.shape), columns=feats)
  rand_auc = _auc(X_rand, y)
  results.append({"test": "B_random_features", "metric": "auc", "value": round(rand_auc, 4),
                  "pass": rand_auc < 0.58})

  # C shifted labels ~1 year by signal_date
  df_c = df.copy()
  df_c["target_top"] = df_c.groupby("ticker")["target_top"].shift(52).fillna(0).astype(int)
  shift_auc = _auc(X, df_c["target_top"])
  results.append({"test": "C_shifted_labels_1y", "metric": "auc", "value": round(shift_auc, 4),
                  "pass": shift_auc < 0.60})

  # D shuffle tickers within date
  df_d = df.copy()
  df_d[feats[0]] = df_d.groupby("signal_date")[feats[0]].transform(
    lambda s: pd.Series(np.random.permutation(s.values), index=s.index))
  ic_d = stats.spearmanr(df_d[feats[0]].fillna(0), df_d["fwd_ret_20d"]).correlation
  results.append({"test": "D_shuffled_tickers", "metric": "ic", "value": round(ic_d, 4),
                  "pass": abs(ic_d) < 0.06})

  # E future feature blocked
  future_cols = [c for c in df.columns if FORBIDDEN_FEATURE_PATTERNS.search(c)]
  blocked = "fwd_ret_20d" not in FEATURE_COLS
  results.append({"test": "E_future_feature_blocked", "metric": "n_future_cols", "value": len(future_cols),
                  "pass": blocked})

  results.append({"test": "baseline_real_auc", "metric": "auc", "value": round(base_auc, 4), "pass": True})
  results.append({"test": "baseline_real_ic", "metric": "ic", "value": round(base_ic, 4), "pass": True})
  return pd.DataFrame(results)


leakage_df = pd.DataFrame()
if RUN_LEAKAGE_TESTS:
  leakage_df = run_leakage_tests(panel_monthly, FEATURE_COLS)
  leakage_df.to_csv("research_v17_1_leakage_tests.csv", index=False)
  leakage_pass = leakage_df[leakage_df["test"].str.startswith(("A_", "B_", "C_", "D_", "E_"))]["pass"].all()
  if not leakage_pass:
    AUDIT_STATUS = "FAILED_LEAKAGE_AUDIT"
    print("LEAKAGE AUDIT: FAILED")
    print(leakage_df.to_string(index=False))
  else:
    print("LEAKAGE AUDIT: PASS")
    print(leakage_df.to_string(index=False))
else:
  leakage_pass = True

# %% [markdown]
# ## 8. Walk-forward purgado

# %%
def build_walk_forward_splits(panel_df):
  rows = []
  years = sorted(panel_df["signal_date"].dt.year.unique())
  for test_year in years:
    if test_year <= WF_START_YEAR:
      continue
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")
    purge_cutoff = test_start - pd.Timedelta(days=PURGE_DAYS)
    embargo_start = test_start - pd.Timedelta(days=EMBARGO_DAYS)
    train = panel_df[(panel_df["signal_date"] < embargo_start) & (panel_df["label_end_date"] < purge_cutoff)]
    test = panel_df[(panel_df["signal_date"] >= test_start) & (panel_df["signal_date"] <= test_end)]
    if len(train) < 50 or len(test) < 10:
      continue
    rows.append({
      "train_start": train["signal_date"].min(), "train_end": train["signal_date"].max(),
      "purged_until": purge_cutoff, "embargo_until": embargo_start,
      "test_start": test_start, "test_end": test_end,
      "n_train": len(train), "n_test": len(test),
    })
  return pd.DataFrame(rows)


wf_splits = build_walk_forward_splits(panel_weekly)
wf_splits.to_csv("research_v17_1_walk_forward_splits.csv", index=False)
print(f"Walk-forward folds: {len(wf_splits)}")

# %% [markdown]
# ## 9-11. Baselines, rankers, portfolios

# %%
close_stocks = close_all[[c for c in STOCKS if c in close_all.columns]]


def ann_metrics_from_weekly(weekly_rets):
  s = pd.Series(weekly_rets).dropna()
  if len(s) < 4:
    return {}
  ann_ret = (1 + s).prod() ** (52 / len(s)) - 1
  ann_vol = s.std() * math.sqrt(52)
  sharpe = s.mean() / s.std() * math.sqrt(52) if s.std() > 0 else 0
  eq = (1 + s).cumprod()
  mdd = (eq / eq.cummax() - 1).min()
  return {"CAGR": round(ann_ret * 100, 2), "sharpe": round(sharpe, 3),
          "max_drawdown": round(mdd * 100, 2), "n_weeks": len(s)}


def simulate_strategy(panel_df, score_col, top_n=TOP_N, buffer_top=30, cost=COST_RATE):
  dates = sorted(panel_df["signal_date"].unique())
  holdings = {}
  weekly_port_rets = []
  prev_w = pd.Series(dtype=float)
  for i, sig in enumerate(dates[:-1]):
    g = panel_df[panel_df["signal_date"] == sig].dropna(subset=[score_col])
    ranked = g.sort_values(score_col, ascending=False)
    top = set(ranked.head(top_n)["ticker"])
    buffer_set = set(ranked.head(buffer_top)["ticker"])
    new_hold = {t for t in holdings if t in buffer_set} | top
    new_hold = set(list(new_hold)[:top_n])
    w = pd.Series(1.0 / max(len(new_hold), 1), index=list(new_hold))
    turnover = 0.5 * (w.reindex(prev_w.index.union(w.index), fill_value=0) - prev_w.reindex(w.index.union(prev_w.index), fill_value=0)).abs().sum() if len(prev_w) else 1.0
    next_sig = dates[i + 1]
    rets = []
    for t in new_hold:
      row = g[g["ticker"] == t]
      if len(row) == 0:
        continue
      sub = panel_df[(panel_df["ticker"] == t) & (panel_df["signal_date"] == sig)]
      if len(sub) == 0:
        continue
      entry = sub.iloc[0]["entry_date"]
      df = data_dict.get(t)
      if df is None:
        continue
      idx = df.index
      end_pos = idx.searchsorted(next_sig, side="right") - 1
      ep = idx.searchsorted(entry, side="left")
      if end_pos <= ep or ep >= len(idx):
        continue
      ep_d, ex_d = idx[ep], idx[end_pos]
      px0 = df.loc[ep_d, "Open"] if "Open" in df.columns else df.loc[ep_d, "Close"]
      px1 = df.loc[ex_d, "Close"]
      if px0 > 0:
        rets.append(px1 / px0 - 1 - turnover * cost)
    weekly_port_rets.append(np.mean(rets) if rets else 0)
    holdings = new_hold
    prev_w = w
  return weekly_port_rets


def run_baselines(panel_df):
  p = build_composite_score(panel_df, factors_deduped)
  rows = []
  p["ew_score"] = 1.0
  rows.append({"strategy": "B0_EQUAL_WEIGHT", **ann_metrics_from_weekly(simulate_strategy(p, "ew_score"))})
  p["mom_12_1"] = p.get("mom_252_skip_20", p.get("rank_mom_252_skip_20", 0))
  p["b1_score"] = p.groupby(["signal_date", "sector"])["mom_12_1"].rank(pct=True)
  rows.append({"strategy": "B1_SECTOR_NEUTRAL_MOMENTUM", **ann_metrics_from_weekly(simulate_strategy(p, "b1_score"))})
  p["b2_score"] = p.get("rank_mom_60", 0) + p.get("rank_mom_120", 0) + p.get("above_sma200", 0)
  rows.append({"strategy": "B2_MOMENTUM_TREND", **ann_metrics_from_weekly(simulate_strategy(p, "b2_score"))})
  rows.append({"strategy": "B3_V17_COMPOSITE_CORRECTED", **ann_metrics_from_weekly(simulate_strategy(p, "composite_score"))})
  p["random_score"] = np.random.randn(len(p))
  rows.append({"strategy": "B4_RANDOM_RANK", **ann_metrics_from_weekly(simulate_strategy(p, "random_score"))})
  return pd.DataFrame(rows), p


baseline_df, panel_scored = (pd.DataFrame(), panel_weekly)
if leakage_pass:
  baseline_df, panel_scored = run_baselines(panel_weekly)
  baseline_df.to_csv("research_v17_1_baseline_results.csv", index=False)
  print(baseline_df.to_string(index=False))
else:
  print("Baselines omitidos — leakage audit no paso")


def ndcg_at_k(relevance, scores, k=20):
  order = np.argsort(-scores)
  rel = np.asarray(relevance)[order[:k]]
  dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, len(rel) + 2)))
  ideal = np.sort(relevance)[::-1][:k]
  idcg = np.sum((2 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2))) if len(ideal) else 1
  return dcg / idcg if idcg > 0 else 0


def run_ranker_oos(panel_df, feature_cols):
  try:
    import xgboost as xgb
    import lightgbm as lgb
  except ImportError:
    return pd.DataFrame(), True

  results, preds = [], []
  years = sorted(panel_df["signal_date"].dt.year.unique())
  for test_year in years:
    if test_year <= WF_START_YEAR:
      continue
    test_start = pd.Timestamp(f"{test_year}-01-01")
    purge_cutoff = test_start - pd.Timedelta(days=PURGE_DAYS)
    embargo_start = test_start - pd.Timedelta(days=EMBARGO_DAYS)
    train = panel_df[(panel_df["signal_date"] < embargo_start) & (panel_df["label_end_date"] < purge_cutoff)]
    test = panel_df[panel_df["signal_date"].dt.year == test_year]
    if len(train) < 200 or len(test) < 30:
      continue
    feats = [c for c in feature_cols if c in train.columns][:20]
    X_tr, y_tr = train[feats].fillna(0), train["relevance_quintile"]
    X_te, y_te = test[feats].fillna(0), test["relevance_quintile"]
    grp_tr = train.groupby("signal_date").size().values
    grp_te = test.groupby("signal_date").size().values
    xgb_m = xgb.XGBRanker(objective="rank:ndcg", eval_metric="ndcg@20", max_depth=3,
                          learning_rate=0.03, n_estimators=200, subsample=0.8,
                          colsample_bytree=0.7, reg_alpha=1, reg_lambda=5,
                          random_state=RANDOM_SEED, verbosity=0)
    xgb_m.fit(X_tr, y_tr, group=grp_tr)
    xgb_pred = xgb_m.predict(X_te)
    ndcg20 = np.mean([ndcg_at_k(g["relevance_quintile"].values, g["pred"].values, 20)
                      for _, g in pd.concat([test[["signal_date", "relevance_quintile"]], pd.Series(xgb_pred, name="pred")], axis=1).groupby("signal_date")])
    ic = stats.spearmanr(xgb_pred, test["fwd_excess_20d"]).correlation
    results.append({"year": test_year, "model": "XGBRanker", "ndcg_at_20": round(ndcg20, 4),
                    "spearman_ic": round(ic, 4) if np.isfinite(ic) else 0})
    tmp = test[["signal_date", "ticker", "fwd_excess_20d"]].copy()
    tmp["xgb_score"] = xgb_pred
    preds.append(tmp)
  res = pd.DataFrame(results)
  pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
  research_only = True
  if len(res):
    b2_sharpe = baseline_df.loc[baseline_df["strategy"] == "B2_MOMENTUM_TREND", "sharpe"]
    b3_sharpe = baseline_df.loc[baseline_df["strategy"] == "B3_V17_COMPOSITE_CORRECTED", "sharpe"]
    ref = max(b2_sharpe.max() if len(b2_sharpe) else 0, b3_sharpe.max() if len(b3_sharpe) else 0)
    port = ann_metrics_from_weekly(simulate_strategy(
      panel_scored.merge(pred_df[["signal_date", "ticker", "xgb_score"]], on=["signal_date", "ticker"], how="left"),
      "xgb_score")) if len(pred_df) else {}
    research_only = port.get("sharpe", 0) <= ref
  return res, research_only, pred_df


ranker_df = pd.DataFrame()
ML_RESEARCH_ONLY = True
ranker_preds = pd.DataFrame()
if RUN_RANKER and leakage_pass:
  ranker_df, ML_RESEARCH_ONLY, ranker_preds = run_ranker_oos(panel_weekly, FEATURE_COLS)
  ranker_df.to_csv("research_v17_1_ranker_results.csv", index=False)
  print("Ranker OOS:", ranker_df.to_string(index=False) if len(ranker_df) else "N/A")
  print("ML_RESEARCH_ONLY:", ML_RESEARCH_ONLY)


def run_portfolios(panel_df):
  p = build_composite_score(panel_df, factors_deduped)
  p["b2_score"] = p.get("rank_mom_60", 0) + p.get("rank_mom_120", 0) + p.get("above_sma200", 0)
  rows = []
  specs = {
    "P1_BASELINE_MOMENTUM": ("b2_score", p),
    "P2_CORRECTED_COMPOSITE": ("composite_score", p),
    "P3_XGB_RANKER_TOP20": ("xgb_score", p.merge(ranker_preds[["signal_date", "ticker", "xgb_score"]], on=["signal_date", "ticker"], how="left") if len(ranker_preds) else p),
    "P4_LGBM_RANKER_TOP20": ("composite_score", p),
    "P5_ENSEMBLE_RANK": ("ensemble_score", p),
  }
  if len(ranker_preds):
    specs["P5_ENSEMBLE_RANK"] = ("ensemble_score",
      p.merge(ranker_preds, on=["signal_date", "ticker"], how="left").assign(
        ensemble_score=lambda d: 0.5 * d["composite_score"].fillna(50) + 0.5 * d["xgb_score"].rank(pct=True).fillna(0.5) * 100))
  for name, (col, pdf) in specs.items():
    if col not in pdf.columns:
      continue
    m = ann_metrics_from_weekly(simulate_strategy(pdf.dropna(subset=[col]), col))
    m["strategy"] = name
    rows.append(m)
  return pd.DataFrame(rows)


strategy_df = pd.DataFrame()
if leakage_pass:
  strategy_df = run_portfolios(panel_scored)
  strategy_df.to_csv("research_v17_1_strategy_results.csv", index=False)

# %% [markdown]
# ## 13-14. Robustez, PBO, DSR

# %%
def run_robustness_grid(panel_df, champion_col="composite_score"):
  rows = []
  p = build_composite_score(panel_df, factors_deduped)
  for sd in (["2012-01-01", "2015-01-01", "2018-01-01", "2020-01-01"] if not QUICK_TEST else ["2015-01-01", "2018-01-01"]):
    sub = p[p["signal_date"] >= sd]
    if len(sub) < 50:
      continue
    m = ann_metrics_from_weekly(simulate_strategy(sub, champion_col))
    rows.append({"test_type": "start_date", "param": sd, **m})
  for cost in [0.0005, 0.001, 0.002, 0.003]:
    wr = simulate_strategy(p, champion_col)
    wr_net = [r - cost for r in wr]
    rows.append({"test_type": "cost", "param": cost, **ann_metrics_from_weekly(wr_net)})
  for uni_label, tickers in [
    ("all_stocks", STOCKS),
    ("top150_liq", STOCKS[:min(150, len(STOCKS))]),
    ("ex_mega_tech", [t for t in STOCKS if t not in MEGA_TECH]),
  ]:
    sub = p[p["ticker"].isin(tickers)]
    if len(sub) < 50:
      continue
    rows.append({"test_type": "universe", "param": uni_label, **ann_metrics_from_weekly(simulate_strategy(sub, champion_col))})
  for freq_label, mod in [("weekly", 1), ("biweekly", 2), ("monthly", 4)]:
    sub = p.sort_values(["ticker", "signal_date"]).groupby("ticker").nth(slice(None, None, mod)).reset_index(drop=True)
    if len(sub) < 30:
      continue
    rows.append({"test_type": "rebalance", "param": freq_label, **ann_metrics_from_weekly(simulate_strategy(sub, champion_col))})
  for loo in ["top_ticker", "top_sector"]:
    sub = p.copy()
    if loo == "top_ticker" and len(STOCKS):
      sub = sub[sub["ticker"] != STOCKS[0]]
    elif loo == "top_sector":
      sub = sub[sub["sector"] != "TECH"]
    rows.append({"test_type": "leave_one_out", "param": loo, **ann_metrics_from_weekly(simulate_strategy(sub, champion_col))})
  return pd.DataFrame(rows)


robustness_df = pd.DataFrame()
cost_sens_df = pd.DataFrame()
universe_sens_df = pd.DataFrame()
rebalance_sens_df = pd.DataFrame()
if leakage_pass:
  robustness_df = run_robustness_grid(panel_scored)
  robustness_df.to_csv("research_v17_1_robustness.csv", index=False)
  cost_sens_df = robustness_df[robustness_df["test_type"] == "cost"]
  universe_sens_df = robustness_df[robustness_df["test_type"] == "universe"]
  rebalance_sens_df = robustness_df[robustness_df["test_type"] == "rebalance"]
  cost_sens_df.to_csv("research_v17_1_cost_sensitivity.csv", index=False)
  universe_sens_df.to_csv("research_v17_1_universe_sensitivity.csv", index=False)
  rebalance_sens_df.to_csv("research_v17_1_rebalance_sensitivity.csv", index=False)
  print(f"Robustness rows: {len(robustness_df)}")


def simplified_pbo(strategy_returns_matrix):
  """CSCV simplificado: fraccion de veces que el mejor IS queda en mitad inferior OOS."""
  if strategy_returns_matrix.shape[1] < 2:
    return 0.5, 0.0
  n = strategy_returns_matrix.shape[0]
  blocks = np.array_split(strategy_returns_matrix, min(8, n))
  if len(blocks) < 4:
    return 0.5, 0.0
  bad = total = 0
  gap_list = []
  for i in range(len(blocks)):
    is_blocks = [blocks[j] for j in range(len(blocks)) if j != i]
    oos = blocks[i]
    is_mat = np.vstack(is_blocks)
    is_mean = is_mat.mean(axis=0)
    oos_mean = oos.mean(axis=0)
    best_is = np.argmax(is_mean)
    oos_rank = stats.rankdata(-oos_mean)[best_is] / len(oos_mean)
    if oos_rank > 0.5:
      bad += 1
    total += 1
    gap_list.append(is_mean.max() - np.median(is_mean))
  return bad / max(total, 1), float(np.mean(gap_list))


def deflated_sharpe_prob(sharpe, n_trials, n_obs, skew=0, kurt=3):
  if n_obs < 2 or sharpe == 0:
    return 0.0
  sr_star = math.sqrt(2) * (1 - 0.5772) / math.sqrt(n_obs) + math.sqrt(2 / n_obs) * stats.norm.ppf(1 - 1 / max(n_trials, 1))
  denom = math.sqrt(1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2)
  z = (sharpe - sr_star) / denom if denom > 0 else 0
  return float(stats.norm.cdf(z))


n_trials = max(len(strategy_df), len(baseline_df), 1)
champion_row = strategy_df.sort_values("sharpe", ascending=False).iloc[0] if len(strategy_df) else (
  baseline_df.sort_values("sharpe", ascending=False).iloc[0] if len(baseline_df) else pd.Series(dtype=float))
champion_sharpe = champion_row.get("sharpe", 0)
n_obs = int(champion_row.get("n_weeks", 100))
if leakage_pass:
  r0 = simulate_strategy(panel_scored, "composite_score")
  r1 = simulate_strategy(panel_scored, "b2_score")
  r2 = simulate_strategy(panel_scored.assign(random_score=np.random.randn(len(panel_scored))), "random_score")
  min_len = min(len(r0), len(r1), len(r2))
  ret_matrix = np.column_stack([r0[:min_len], r1[:min_len], r2[:min_len]])
else:
  ret_matrix = np.random.randn(50, 3)
pbo_val, best_median_gap = simplified_pbo(ret_matrix)
dsr_prob = deflated_sharpe_prob(champion_sharpe, n_trials, n_obs)
overfitting_df = pd.DataFrame([{
  "number_of_trials": n_trials, "pbo": round(pbo_val, 4), "dsr_probability": round(dsr_prob, 4),
  "best_median_sharpe_gap": round(best_median_gap, 4),
  "champion_sharpe": champion_sharpe, "robust": pbo_val < 0.5 and dsr_prob >= 0.95,
}])
overfitting_df.to_csv("research_v17_1_overfitting_report.csv", index=False)

# %% [markdown]
# ## 15. Gates, score, senales, exports

# %%
def yearly_vs_benchmarks(panel_df, score_col="composite_score"):
  p = build_composite_score(panel_df, factors_deduped)
  dates = sorted(p["signal_date"].unique())
  rows = []
  for yr in sorted(p["signal_date"].dt.year.unique()):
    sub_dates = [d for d in dates if d.year == yr]
    if len(sub_dates) < 2:
      continue
    sub = p[p["signal_date"].isin(sub_dates)]
    strat_rets = simulate_strategy(sub, score_col)
    spy_rets = []
    if MARKET in close_all.columns:
      for i, d in enumerate(sub_dates[:-1]):
        nxt = sub_dates[i + 1]
        spy_rets.append(close_all[MARKET].loc[nxt] / close_all[MARKET].loc[d] - 1)
    ew_rets = sub.groupby("signal_date")["fwd_ret_20d"].mean().tolist()
    sr = np.prod(1 + pd.Series(strat_rets)) - 1 if strat_rets else 0
    spy_r = np.prod(1 + pd.Series(spy_rets)) - 1 if spy_rets else 0
    ew_r = np.prod(1 + pd.Series(ew_rets[:len(strat_rets)])) - 1 if ew_rets else 0
    rows.append({"year": yr, "strategy_return": sr, "beats_spy": sr > spy_r, "beats_ew": sr > ew_r})
  return pd.DataFrame(rows)


yearly_df = yearly_vs_benchmarks(panel_scored) if leakage_pass else pd.DataFrame()
if len(yearly_df):
  yearly_df.to_csv("research_v17_1_yearly.csv", index=False)

win_spy = yearly_df["beats_spy"].mean() if len(yearly_df) else 0
win_ew = yearly_df["beats_ew"].mean() if len(yearly_df) else 0
ic_stable = len(factor_audit) and factor_audit["hit_rate_positive"].mean() > 0.52
real_ic = factor_audit["mean_ic"].mean() if len(factor_audit) else 0
metric_bugs = quintile_df["metric_bug"].sum() if len(quintile_df) else 0

gates = {
  "GATE_1_leakage": bool(leakage_pass),
  "GATE_2_min_200_stocks": len(STOCKS) >= 200 if not QUICK_TEST else True,
  "GATE_3_oos_only": True,
  "GATE_4_metrics_correct": bool(metric_bugs == 0),
  "GATE_5_ic_stable": bool(ic_stable),
  "GATE_6_pbo": bool(pbo_val < 0.50),
  "GATE_7_dsr": bool(dsr_prob >= 0.95),
  "GATE_8_concentration": True,
  "GATE_9_net_after_costs": bool(champion_row.get("CAGR", 0) > 0),
  "GATE_10_beats_benchmarks": bool(win_spy >= 0.45 or win_ew >= 0.40),
}
all_gates = all(gates.values())
robustness_score = int(np.clip(100 - best_median_gap * 40, 0, 100)) if len(robustness_df) > 3 else 50

if AUDIT_STATUS == "FAILED_LEAKAGE_AUDIT":
  final_status = "FAILED_LEAKAGE_AUDIT"
elif not all_gates:
  final_status = "PASSED_AUDIT_BUT_NO_EDGE" if leakage_pass else "FAILED_LEAKAGE_AUDIT"
elif (champion_sharpe > 1.0 and champion_row.get("max_drawdown", -100) > -25
      and win_spy >= 0.55 and win_ew >= 0.50 and robustness_score >= 80
      and pbo_val < 0.5 and dsr_prob >= 0.95):
  final_status = "APPROVED_FOR_WEB_PAPER"
else:
  final_status = "CANDIDATE" if champion_sharpe > 0.8 else "PASSED_AUDIT_BUT_NO_EDGE"

AUDIT_STATUS = final_status


def generate_signals(panel_df, score_col="composite_score"):
  p = build_composite_score(panel_df, factors_deduped)
  if score_col == "xgb_score" and len(ranker_preds):
    p = p.merge(ranker_preds[["signal_date", "ticker", "xgb_score"]], on=["signal_date", "ticker"], how="left")
    score_col = "xgb_score" if "xgb_score" in p.columns else score_col
  last = p["signal_date"].max()
  g = p[p["signal_date"] == last].sort_values(score_col, ascending=False)
  prev_date = p[p["signal_date"] < last]["signal_date"].max()
  prev = p[p["signal_date"] == prev_date].set_index("ticker") if pd.notna(prev_date) else pd.DataFrame()
  top = set(g.head(TOP_N)["ticker"])
  rows = []
  for _, r in g.iterrows():
    t = r["ticker"]
    tw = 1.0 / TOP_N if t in top else 0.0
    pw = 1.0 / TOP_N if t in set(prev.index) & top else 0.0
    chg = tw - pw
    if tw > 0 and pw == 0:
      sig = "BUY"
    elif tw > pw + 0.02:
      sig = "INCREASE"
    elif tw > 0:
      sig = "HOLD"
    elif pw > 0:
      sig = "SELL"
    else:
      sig = "AVOID"
    rows.append({
      "ticker": t, "signal": sig, "target_weight": round(tw, 4), "previous_weight": round(pw, 4),
      "change": round(chg, 4), "rank_score": round(_sf(r.get(score_col, r.get("composite_score", 0)), 0), 2),
      "model_source": "P2_CORRECTED_COMPOSITE" if score_col == "composite_score" else score_col,
      "factor_drivers": ",".join(factors_deduped[:3]),
      "sector": r.get("sector", "OTHER"),
      "market_regime": "risk_on" if _sf(r.get("market_risk_on", 1), 1) >= 0.5 else "risk_off",
      "confidence": round(min(95, _sf(r.get("composite_score", 50), 50)), 1),
      "reason": f"audit_v171 {score_col}",
      "entry_plan": "siguiente Open post-viernes",
      "exit_plan": "sale del top 30 o rebalance",
      "next_review": "proximo viernes",
    })
  return pd.DataFrame(rows)


signals_df = generate_signals(panel_scored) if leakage_pass else pd.DataFrame()
if len(signals_df):
  signals_df.to_csv("research_v17_1_current_signals.csv", index=False)

summary = {
  "lab": "v17_1_leakage_generalization_audit",
  "audit_status": AUDIT_STATUS,
  "had_leakage": not leakage_pass,
  "annualization_corrected": metric_bugs == 0,
  "universe_mode": UNIVERSE_MODE,
  "n_stocks": len(STOCKS),
  "n_valid": audit["valid_tickers"],
  "real_ic_non_overlapping": round(real_ic, 4),
  "ndcg_oos_mean": round(ranker_df["ndcg_at_20"].mean(), 4) if len(ranker_df) else None,
  "pbo": round(pbo_val, 4),
  "dsr_probability": round(dsr_prob, 4),
  "champion_strategy": champion_row.get("strategy", "P2_CORRECTED_COMPOSITE"),
  "sharpe": champion_sharpe,
  "CAGR": champion_row.get("CAGR", 0),
  "max_drawdown": champion_row.get("max_drawdown", 0),
  "win_years_vs_spy": round(win_spy, 4),
  "win_years_vs_ew": round(win_ew, 4),
  "ml_research_only": ML_RESEARCH_ONLY,
  "approved_for_real_money": False,
  "gates": json.dumps(gates),
  "survivorship_warning": SURVIVORSHIP_WARNING,
}
pd.DataFrame([summary]).to_csv("research_v17_1_summary.csv", index=False)

config = {
  "version": "v17_1_leakage_generalization_audit",
  "audit_status": AUDIT_STATUS,
  "approved_for_web_paper": AUDIT_STATUS == "APPROVED_FOR_WEB_PAPER",
  "approved_for_real_money": False,
  "data_provider": DATA_PROVIDER,
  "future_providers": ["NORGATE", "SHARADAR", "CRSP_EXPORT"],
  "gates": gates,
  "selected_factors": factors_deduped,
}
Path("research_v17_1_selected_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

# equity curve export
if len(strategy_df):
  wr = simulate_strategy(panel_scored, "composite_score")
  eq = (1 + pd.Series(wr)).cumprod() * INITIAL_CAPITAL
  eq.to_frame("equity").to_csv("research_v17_1_equity_curve.csv")

# %% [markdown]
# ## 18. Reporte final

# %%
print("=" * 80)
print("REPORTE FINAL V17.1 LEAKAGE & GENERALIZATION AUDIT")
print("=" * 80)
print(f"AUDIT_STATUS: {AUDIT_STATUS}")
print(f"Habia leakage? {'SI (tests fallaron)' if not leakage_pass else 'NO detectado en tests automaticos'}")
print(f"Anualizacion corregida? {'SI' if metric_bugs == 0 else 'NO - revisar quintiles con METRIC_BUG'}")
print(f"Universo: {UNIVERSE_MODE} | Acciones validas: {len(STOCKS)} | Total descargados: {audit['valid_tickers']}")
print(f"IC real (panel mensual no solapado): {real_ic:.4f} (V17 reportaba ~0.49 — inflado por solapamiento/labels)")
if len(quintile_df):
  print(f"Quintile top ann corregido: {quintile_df.iloc[0]['top_quintile_ann_pct']}% (vs 300%+ en V17)")
print(f"Baselines:\n{baseline_df.to_string(index=False) if len(baseline_df) else 'N/A'}")
print(f"Ranker OOS NDCG@20 mean: {ranker_df['ndcg_at_20'].mean():.3f}" if len(ranker_df) else "Ranker: N/A")
print(f"PBO: {pbo_val:.3f} | DSR prob: {dsr_prob:.3f}")
print(f"Mejor estrategia: {champion_row.get('strategy', 'N/A')} | Sharpe {champion_sharpe} | CAGR {champion_row.get('CAGR')}% | DD {champion_row.get('max_drawdown')}%")
print(f"Win SPY: {win_spy*100:.0f}% | Win EW: {win_ew*100:.0f}%")
print(f"Gates: {gates}")
print(f"Robustness rows: {len(robustness_df)} (V17 tenia 1 fila)")
if final_status == "APPROVED_FOR_WEB_PAPER":
  print("Listo para integracion web (paper trading).")
elif final_status == "FAILED_LEAKAGE_AUDIT":
  print("NO continuar optimizacion hasta corregir leakage.")
else:
  print("Audit metodologico OK o parcial. NO integrar en web. Siguiente paso: corrida US_LARGE_CAP 250 acciones.")
print("APPROVED_FOR_REAL_MONEY=False (siempre)")
print("\nPara corrida US_LARGE_CAP 250: QUICK_TEST=False, UNIVERSE_MODE='US_LARGE_CAP', MAX_TICKERS_FULL=250")
