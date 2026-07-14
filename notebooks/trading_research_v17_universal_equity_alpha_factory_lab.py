# %% [markdown]
# # Trading Research V17 Universal Equity Alpha Factory Lab
#
# Alpha Factory: factores universales, IC, deciles, composite, ensemble robusto.
#
# **Survivorship bias warning:** Este research usa tickers disponibles actualmente.
# Para research institucional real haría falta base con deslistadas (CRSP/Norgate/Sharadar).
# **APPROVED_FOR_REAL_MONEY siempre False.**

# %%
try:
  get_ipython().run_line_magic(
    "pip", "install yfinance pandas numpy matplotlib plotly tqdm scipy scikit-learn lxml html5lib beautifulsoup4 -q"
  )
except NameError:
  import subprocess, sys
  subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "yfinance", "pandas", "numpy", "matplotlib", "plotly", "tqdm", "scipy", "scikit-learn",
    "lxml", "html5lib", "beautifulsoup4"])

# %% [markdown]
# ## 1. Configuracion

# %%
import warnings
warnings.filterwarnings("ignore")
import json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
from tqdm.auto import tqdm

QUICK_TEST = True
START_DATE = "2010-01-01"
END_DATE = None
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001
SLIPPAGE = 0.001
REBALANCE_FREQ = "W-FRI"
WF_START_YEAR = 2016
UNIVERSE_MODE = "QUICK_TEST"
MAX_TICKERS_FULL = 600
MIN_HISTORY_DAYS = 1000 if not QUICK_TEST else 400
MIN_DOLLAR_VOLUME = 20_000_000
TOP_N_VALUES = [10, 20, 30]
MAX_WEIGHT_VALUES = [0.05, 0.075, 0.10]
SECTOR_CAP = 0.35
VOL_TARGET_VALUES = [0.12, 0.15, 0.18]
CASH_ASSET = "SHY"
PREDICTION_HORIZONS = [5, 20, 60]
USE_ML = True
ML_RESEARCH_ONLY_BY_DEFAULT = True
MARKET = "SPY"
COST_RATE = TRANSACTION_COST + SLIPPAGE
EMBARGO_DAYS = 20
DEFENSIVE_ETFS = {"SHY", "IEF", "TLT", "GLD", "LQD", "USMV", "QUAL"}
ETF_SET = {
  "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLC", "XLB", "XLRE",
  "MTUM", "QUAL", "USMV", "VLUE", "SPLV", "SPHB", "SCHD", "SHY", "IEF", "TLT", "LQD", "HYG",
  "GLD", "SLV", "DBC", "VNQ", "EFA", "EEM",
}
SECTOR_MAP = {
  "AAPL": "TECH", "MSFT": "TECH", "NVDA": "TECH", "AMD": "TECH", "AVGO": "TECH", "GOOGL": "TECH", "META": "TECH", "AMZN": "TECH",
  "JPM": "FIN", "BAC": "FIN", "XOM": "ENERGY", "CVX": "ENERGY", "UNH": "HEALTH", "LLY": "HEALTH", "WMT": "STAPLES", "COST": "STAPLES",
  "XLK": "TECH", "XLF": "FIN", "XLE": "ENERGY", "XLV": "HEALTH", "XLY": "DISC", "XLP": "STAPLES", "XLI": "IND", "XLU": "UTIL",
  "XLC": "COMM", "XLB": "MAT", "XLRE": "REAL", "SPY": "BROAD", "QQQ": "TECH", "IWM": "SMALL", "DIA": "BROAD",
}
START_DATES_ROBUST = ["2012-01-01", "2015-01-01", "2018-01-01"] if QUICK_TEST else ["2012-01-01", "2015-01-01", "2018-01-01", "2020-01-01"]
COST_SENSITIVITY = [0.0005, 0.001, 0.002] if not QUICK_TEST else [0.0005, 0.001]

if QUICK_TEST:
  START_DATE = "2015-01-01"
  MIN_HISTORY_DAYS = 250
  MIN_DOLLAR_VOLUME = 5_000_000
  print("QUICK_TEST activo")

SURVIVORSHIP_WARNING = (
  "Este research usa tickers disponibles actualmente. Para research institucional real "
  "haría falta base de datos con acciones deslistadas, como CRSP/Norgate/Sharadar."
)
print("V17 Universal Equity Alpha Factory |", UNIVERSE_MODE)
print(SURVIVORSHIP_WARNING)

# %% [markdown]
# ## 2. Universos

# %%
def load_quick_test_universe():
  return [
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLV", "XLF", "XLE", "XLY", "XLP", "XLI", "XLU", "XLC",
    "MTUM", "QUAL", "USMV", "VLUE", "SPLV", "SCHD", "SHY", "IEF", "TLT", "GLD",
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "GOOGL", "META", "AMZN",
    "JPM", "BAC", "XOM", "CVX", "UNH", "LLY", "WMT", "COST",
  ]


def _clean_symbol(s):
  return str(s).strip().upper().replace(".", "-")


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
  out = sorted(tickers)[:MAX_TICKERS_FULL]
  return out


def load_global_etf_universe():
  return [
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLC", "XLB", "XLRE",
    "MTUM", "QUAL", "USMV", "VLUE", "SPLV", "SPHB", "SCHD", "SHY", "IEF", "TLT", "LQD", "HYG",
    "GLD", "SLV", "DBC", "VNQ", "EFA", "EEM", "EWJ", "EWU", "EWG", "EWQ", "EWP", "EWI", "EWC", "EWA", "EWH", "EWS", "EWT", "EWY", "INDA", "FXI", "EWZ", "EWW",
  ]


def load_custom_csv_universe():
  p = Path("data/universe_custom.csv")
  if p.exists():
    df = pd.read_csv(p)
    col = "ticker" if "ticker" in df.columns else df.columns[0]
    return [_clean_symbol(x) for x in df[col].dropna().tolist()]
  return load_quick_test_universe()


def resolve_universe(mode):
  loaders = {
    "QUICK_TEST": load_quick_test_universe,
    "US_LARGE_CAP": load_us_large_cap_universe,
    "GLOBAL_ETF": load_global_etf_universe,
    "CUSTOM_CSV": load_custom_csv_universe,
  }
  return loaders.get(mode, load_quick_test_universe)()


UNIVERSE = resolve_universe(UNIVERSE_MODE)
print("Universo inicial:", len(UNIVERSE))

# %% [markdown]
# ## 3. Descarga de datos

# %%
def download_data(tickers, start, end=None, batch_size=50, min_days=MIN_HISTORY_DAYS):
  data, errors = {}, []
  tickers = sorted(set(tickers))
  for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i + batch_size]
    for ticker in tqdm(batch, desc=f"Download {i//batch_size+1}", leave=False):
      try:
        raw = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
        if raw is None or raw.empty:
          errors.append({"ticker": ticker, "error": "empty"}); continue
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
          df.columns = df.columns.get_level_values(0)
        colmap = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        df = df.rename(columns={c: colmap.get(str(c).lower(), c) for c in df.columns})
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].dropna(subset=["Close"])
        if len(df) < min_days:
          errors.append({"ticker": ticker, "error": f"short({len(df)})"}); continue
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


def _sf(x, d=np.nan):
  try:
    v = float(x); return d if not np.isfinite(v) else v
  except Exception:
    return d


def audit_universe(initial, data, close, report_df):
  audit = {
    "universe_mode": UNIVERSE_MODE,
    "initial_tickers": len(initial),
    "downloaded": len(data),
    "valid_history_liquidity": len(close.columns),
    "filtered_out": len(initial) - len(close.columns),
    "survivorship_warning": SURVIVORSHIP_WARNING,
  }
  pd.DataFrame([audit]).to_csv("research_v17_universe_audit.csv", index=False)
  return audit


data_dict, close_prices, dl_errors, dl_report = download_data(UNIVERSE, START_DATE, END_DATE)
audit = audit_universe(UNIVERSE, data_dict, close_prices, dl_report)
dl_report.to_csv("research_v17_download_report.csv", index=False)
print("Validos:", audit["valid_history_liquidity"], "| Filas:", len(close_prices))

# %% [markdown]
# ## 4. Features / factores universales

# %%
def calculate_alpha_features(data_dict, close_prices):
  feats = {}
  rets_wide = close_prices.pct_change().fillna(0)
  for t in close_prices.columns:
    if t not in data_dict:
      continue
    df = data_dict[t]
    c, o, h, l = df["Close"], df.get("Open", df["Close"]), df.get("High", df["Close"]), df.get("Low", df["Close"])
    v = df["Volume"] if "Volume" in df else pd.Series(1, index=df.index)
    f = pd.DataFrame(index=df.index)
    for n in [5, 20, 60, 120, 252]:
      f[f"ret_{n}"] = c.pct_change(n)
    f["mom_60"] = c / c.shift(60) - 1
    f["mom_120"] = c / c.shift(120) - 1
    f["mom_252"] = c / c.shift(252) - 1
    f["mom_252_skip_20"] = c.shift(20) / c.shift(252) - 1
    for n in [20, 50, 100, 200]:
      f[f"sma_{n}"] = c.rolling(n).mean()
    f["above_sma50"] = (c > f["sma_50"]).astype(float)
    f["above_sma100"] = (c > f["sma_100"]).astype(float)
    f["above_sma200"] = (c > f["sma_200"]).astype(float)
    f["trend_stack"] = ((c > f["sma_50"]) & (f["sma_50"] > f["sma_200"])).astype(float)
    f["slope_sma50"] = f["sma_50"].pct_change(10)
    f["slope_sma200"] = f["sma_200"].pct_change(20)
    f["reversal_1w"] = -f["ret_5"]
    f["reversal_1m"] = -f["ret_20"]
    hi252 = c.rolling(252).max().shift(1)
    f["dist_to_52w_high"] = c / hi252 - 1
    f["near_52w_high_score"] = (f["dist_to_52w_high"] > -0.05).astype(float)
    for n in [20, 60, 120]:
      f[f"vol_{n}"] = c.pct_change().rolling(n).std() * math.sqrt(252)
    f["low_vol_score"] = 1.0 / f["vol_60"].replace(0, np.nan)
    for n in [20, 60, 120, 252]:
      f[f"dd_{n}"] = c / c.rolling(n).max() - 1
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_14"] = tr.rolling(14).mean()
    f["atr_pct"] = f["atr_14"] / c
    f["volume_ratio_20"] = v / v.rolling(20).mean()
    f["dollar_volume"] = c * v
    f["abnormal_volume"] = (f["volume_ratio_20"] > 1.5).astype(float)
    f["gap_open"] = o / c.shift() - 1
    f["gap_close"] = c / o - 1
    f["price_volume_breakout"] = ((f["ret_20"] > 0.05) & (f["volume_ratio_20"] > 1.3)).astype(float)
    f["gap_and_go"] = ((f["gap_open"] > 0.01) & (c > o)).astype(float)
    f["earnings_proxy_gap"] = ((f["gap_close"].abs() > 0.04) & (f["volume_ratio_20"] > 1.5)).astype(float)
    f["sector"] = SECTOR_MAP.get(t, "OTHER")
    feats[t] = f.replace([np.inf, -np.inf], np.nan)
  return feats


def build_panel(feats, close_prices):
  parts = []
  for t, f in feats.items():
    tmp = f.copy()
    tmp["ticker"] = t
    tmp["sector"] = SECTOR_MAP.get(t, "OTHER")
    parts.append(tmp.reset_index().rename(columns={"index": "date", "Date": "date"}))
  panel = pd.concat(parts, ignore_index=True)
  panel["date"] = pd.to_datetime(panel["date"])
  dates = panel["date"].unique()
  rank_cols = ["mom_60", "mom_120", "mom_252_skip_20", "low_vol_score", "trend_stack", "abnormal_volume"]
  for dt in dates:
    mask = panel["date"] == dt
    for col in ["mom_60", "mom_120", "mom_252_skip_20"]:
      if col in panel.columns:
        panel.loc[mask, f"rank_{col}"] = panel.loc[mask, col].rank(pct=True)
    if "low_vol_score" in panel.columns:
      panel.loc[mask, "rank_low_vol"] = panel.loc[mask, "low_vol_score"].rank(pct=True, ascending=False)
    if "trend_stack" in panel.columns:
      panel.loc[mask, "rank_trend"] = panel.loc[mask, "trend_stack"].rank(pct=True)
    if "abnormal_volume" in panel.columns:
      panel.loc[mask, "rank_volume_attention"] = panel.loc[mask, "abnormal_volume"].rank(pct=True)
  if MARKET in close_prices.columns:
    spy_c = close_prices[MARKET]
    mkt = pd.DataFrame(index=close_prices.index)
    mkt["spy_above_sma200"] = (spy_c > spy_c.rolling(200).mean()).astype(float)
    mkt["spy_mom_60"] = spy_c / spy_c.shift(60) - 1
    mkt["spy_vol_20"] = spy_c.pct_change().rolling(20).std() * math.sqrt(252)
    if "QQQ" in close_prices.columns:
      qqq = close_prices["QQQ"]
      mkt["qqq_above_sma200"] = (qqq > qqq.rolling(200).mean()).astype(float)
      mkt["qqq_mom_60"] = qqq / qqq.shift(60) - 1
      mkt["qqq_vol_20"] = qqq.pct_change().rolling(20).std() * math.sqrt(252)
    else:
      mkt["qqq_above_sma200"] = mkt["spy_above_sma200"]
      mkt["qqq_mom_60"] = mkt["spy_mom_60"]
      mkt["qqq_vol_20"] = mkt["spy_vol_20"]
    if "HYG" in close_prices.columns and "LQD" in close_prices.columns:
      mkt["hyg_lqd_mom"] = close_prices["HYG"] / close_prices["HYG"].shift(60) - close_prices["LQD"] / close_prices["LQD"].shift(60)
    else:
      mkt["hyg_lqd_mom"] = 0
    mkt["market_risk_on"] = ((mkt["spy_above_sma200"] >= 0.5) & (mkt["qqq_above_sma200"] >= 0.5)).astype(float)
    mkt["market_stress_score"] = ((mkt["spy_above_sma200"] < 0.5).astype(float) * 30
                                    + (mkt["qqq_above_sma200"] < 0.5).astype(float) * 25
                                    + (mkt["spy_vol_20"] > 0.22).astype(float) * 25).clip(0, 100)
    panel = panel.merge(mkt.reset_index().rename(columns={"index": "date", "Date": "date"}), on="date", how="left")
  return panel


features = calculate_alpha_features(data_dict, close_prices)
panel = build_panel(features, close_prices)
print("Panel:", len(panel), "filas |", panel["ticker"].nunique(), "tickers")

# %% [markdown]
# ## 5. Labels forward (solo evaluacion/ML train)

# %%
def create_forward_labels(panel):
  p = panel.copy()
  for h in PREDICTION_HORIZONS:
    p[f"fwd_ret_{h}"] = p.groupby("ticker")["ret_5" if h == 5 else f"ret_{h}"].shift(-max(1, h // 5 if h == 5 else h // 20 if h == 20 else h // 60))
  p["fwd_ret_20"] = p.groupby("ticker")["ret_20"].shift(-1)
  p["fwd_ret_5"] = p.groupby("ticker")["ret_5"].shift(-1)
  p["fwd_ret_60"] = p.groupby("ticker")["ret_60"].shift(-1)
  uni_mean = p.groupby("date")["fwd_ret_20"].transform("mean")
  p["fwd_excess_20_vs_universe"] = p["fwd_ret_20"] - uni_mean
  if MARKET in close_prices.columns:
    spy_fwd = close_prices[MARKET].pct_change(20).shift(-20)
    p = p.merge(spy_fwd.rename("spy_fwd_20").reset_index().rename(columns={"index": "date", "Date": "date"}), on="date", how="left")
    p["fwd_excess_20_vs_spy"] = p["fwd_ret_20"] - p["spy_fwd_20"]
  else:
    p["fwd_excess_20_vs_spy"] = p["fwd_excess_20_vs_universe"]
  p["fwd_rank_20"] = p.groupby("date")["fwd_ret_20"].rank(pct=True)
  p["top_quintile_20"] = (p["fwd_rank_20"] >= 0.8).astype(int)
  p["positive_excess_20"] = (p["fwd_excess_20_vs_universe"] > 0).astype(int)
  p["bad_tail_20"] = (p["fwd_ret_20"] < -2 * p["vol_20"].fillna(0.15)).astype(int)
  return p


panel = create_forward_labels(panel)

# %% [markdown]
# ## 6-7. Factor IC y deciles

# %%
IC_FACTORS = [
  "rank_mom_60", "rank_mom_120", "rank_mom_252_skip_20", "rank_trend", "rank_low_vol",
  "near_52w_high_score", "trend_stack", "abnormal_volume", "price_volume_breakout",
  "reversal_1w", "earnings_proxy_gap", "rank_volume_attention",
]
rdates = [d for d in close_prices.resample(REBALANCE_FREQ).last().dropna(how="all").index if d.year >= WF_START_YEAR]
panel_w = panel[panel["date"].isin(rdates)].copy()


def factor_ic_analysis(panel_df, factors=None, target="fwd_ret_20"):
  factors = factors or IC_FACTORS
  rows, yearly_rows = [], []
  for fac in factors:
    if fac not in panel_df.columns or target not in panel_df.columns:
      continue
    ics = []
    for dt, g in panel_df.groupby("date"):
      sub = g[[fac, target]].dropna()
      if len(sub) < 8:
        continue
      ic, _ = stats.spearmanr(sub[fac], sub[target])
      if np.isfinite(ic):
        ics.append(ic)
        yearly_rows.append({"factor": fac, "year": pd.Timestamp(dt).year, "ic": ic})
    if not ics:
      continue
    s = pd.Series(ics)
    rows.append({
      "factor": fac, "mean_ic": round(s.mean(), 4), "median_ic": round(s.median(), 4),
      "ic_std": round(s.std(), 4), "ic_ir": round(s.mean() / s.std(), 4) if s.std() > 0 else 0,
      "hit_rate_ic_positive": round((s > 0).mean(), 4),
      "stability_score": round(1 - s.std(), 4),
    })
  ic_df = pd.DataFrame(rows).sort_values("mean_ic", ascending=False)
  ic_yearly = pd.DataFrame(yearly_rows)
  selected = ic_df[(ic_df["mean_ic"] > 0) & (ic_df["hit_rate_ic_positive"] > 0.52)]["factor"].tolist()
  return ic_df, ic_yearly, selected


def factor_decile_test(panel_df, factors=None):
  factors = factors or IC_FACTORS
  rows = []
  for fac in factors:
    if fac not in panel_df.columns:
      continue
    top_rets, bot_rets, ew_rets = [], [], []
    for dt, g in panel_df.groupby("date"):
      sub = g[[fac, "fwd_ret_20", "ticker"]].dropna()
      if len(sub) < 10:
        continue
      sub["q"] = pd.qcut(sub[fac].rank(method="first"), 5, labels=False, duplicates="drop")
      top_rets.append(sub[sub["q"] == sub["q"].max()]["fwd_ret_20"].mean())
      bot_rets.append(sub[sub["q"] == sub["q"].min()]["fwd_ret_20"].mean())
      ew_rets.append(sub["fwd_ret_20"].mean())
    if not top_rets:
      continue
    rows.append({
      "factor": fac,
      "top_quintile_ann": round(np.mean(top_rets) * 52 * 100, 2),
      "bottom_quintile_ann": round(np.mean(bot_rets) * 52 * 100, 2),
      "long_short_spread_ann": round((np.mean(top_rets) - np.mean(bot_rets)) * 52 * 100, 2),
      "top_vs_equal_weight_ann": round((np.mean(top_rets) - np.mean(ew_rets)) * 52 * 100, 2),
    })
  return pd.DataFrame(rows).sort_values("top_quintile_ann", ascending=False)


ic_df, ic_yearly, selected_factors = factor_ic_analysis(panel_w)
if selected_factors == [] and len(ic_df):
  selected_factors = ic_df.head(6)["factor"].tolist()
decile_df = factor_decile_test(panel_w)
ic_df.to_csv("research_v17_factor_ic.csv", index=False)
ic_yearly.to_csv("research_v17_factor_ic_yearly.csv", index=False)
decile_df.to_csv("research_v17_factor_deciles.csv", index=False)
print("Factores seleccionados:", selected_factors[:8])

# %% [markdown]
# ## 8. Alpha composite score

# %%
def build_alpha_composite_score(panel_df, selected):
  p = panel_df.copy()
  mom_cols = [c for c in ["rank_mom_60", "rank_mom_120", "rank_mom_252_skip_20", "near_52w_high_score"] if c in p.columns]
  trend_cols = [c for c in ["above_sma200", "trend_stack", "slope_sma50", "slope_sma200"] if c in p.columns]
  qual_cols = [c for c in ["rank_low_vol", "dd_60", "atr_pct"] if c in p.columns]
  event_cols = [c for c in ["abnormal_volume", "price_volume_breakout", "earnings_proxy_gap"] if c in p.columns]
  rev_cols = [c for c in ["reversal_1w"] if c in p.columns]

  def fam_mean(cols):
    cols = [c for c in cols if c in p.columns]
    if not cols:
      return pd.Series(0.5, index=p.index)
    return p[cols].mean(axis=1).rank(pct=True)

  momentum_family = fam_mean(mom_cols)
  trend_family = fam_mean(trend_cols)
  qual_df = p[[c for c in qual_cols if c in p.columns]].copy()
  if "dd_60" in qual_df:
    qual_df["dd_60"] = -qual_df["dd_60"]
  if "atr_pct" in qual_df:
    qual_df["atr_pct"] = -qual_df["atr_pct"]
  risk_quality_family = qual_df.mean(axis=1).rank(pct=True) if len(qual_df.columns) else pd.Series(0.5, index=p.index)
  event_family = fam_mean(event_cols)
  reversal_family = fam_mean(rev_cols)
  reversal_family = reversal_family.where((p.get("above_sma200", 0) >= 0.5) & (p.get("market_risk_on", 1) >= 0.5), 0)

  raw = (0.35 * momentum_family + 0.25 * trend_family + 0.20 * risk_quality_family
         + 0.10 * event_family + 0.10 * reversal_family)
  penalty = (
    (p.get("above_sma200", 1) < 0.5).astype(float) * 0.15
    + (p.get("market_stress_score", 0) > 55).astype(float) * 0.20
    + (p.get("vol_60", 0.15) > 0.45).astype(float) * 0.10
    + (p.get("dd_60", 0) < -0.20).astype(float) * 0.10
  )
  p["alpha_score"] = ((raw - penalty).clip(0, 1) * 100).round(2)
  p["main_factor_driver"] = np.where(momentum_family >= trend_family, "momentum", "trend")
  return p


panel = build_alpha_composite_score(panel, selected_factors)


def add_variant_scores(p):
  """Scores alternativos para estrategias U3-U5 (escala 0-100)."""
  p = p.copy()
  mom = p[[c for c in ["rank_mom_60", "rank_mom_120", "rank_mom_252_skip_20", "near_52w_high_score"] if c in p.columns]].mean(axis=1)
  trd = p[[c for c in ["above_sma200", "trend_stack", "slope_sma50", "slope_sma200"] if c in p.columns]].mean(axis=1)
  evt = p[[c for c in ["abnormal_volume", "price_volume_breakout", "earnings_proxy_gap"] if c in p.columns]].mean(axis=1)
  lvol = p["rank_low_vol"] if "rank_low_vol" in p.columns else p.get("low_vol_score", pd.Series(0.5, index=p.index))
  p["score_momentum_trend"] = ((0.6 * mom.rank(pct=True) + 0.4 * trd.rank(pct=True)) * 100).round(2)
  p["score_event_momentum"] = ((0.5 * mom.rank(pct=True) + 0.3 * trd.rank(pct=True) + 0.2 * evt.rank(pct=True)) * 100).round(2)
  p["score_low_vol_momentum"] = ((0.65 * mom.rank(pct=True) + 0.35 * lvol.rank(pct=True)) * 100).round(2)
  return p


panel = add_variant_scores(panel)

# %% [markdown]
# ## 9. ML ranker research-only

# %%
def ml_ranker_walk_forward(panel_df):
  from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import roc_auc_score, log_loss

  feat_cols = [c for c in selected_factors + ["rank_mom_60", "rank_trend", "rank_low_vol", "vol_60", "market_risk_on", "market_stress_score"]
               if c in panel_df.columns]
  feat_cols = list(dict.fromkeys(feat_cols))
  df = panel_df.dropna(subset=["top_quintile_20"] + feat_cols[:1]).copy()
  if len(df) < 500:
    return pd.DataFrame(), pd.DataFrame(), False

  results, preds = [], []
  years = sorted(df["date"].dt.year.unique())
  for test_year in years:
    if test_year <= WF_START_YEAR:
      continue
    train = df[df["date"].dt.year < test_year]
    test = df[df["date"].dt.year == test_year]
    if len(train) < 200 or len(test) < 50:
      continue
    train = train.iloc[:-EMBARGO_DAYS] if len(train) > EMBARGO_DAYS else train
    X_tr, y_tr = train[feat_cols].fillna(0), train["top_quintile_20"]
    X_te, y_te = test[feat_cols].fillna(0), test["top_quintile_20"]
    models = {
      "hgb": HistGradientBoostingClassifier(max_depth=4, max_iter=80, random_state=42),
      "rf": RandomForestClassifier(n_estimators=80, max_depth=5, random_state=42),
      "et": ExtraTreesClassifier(n_estimators=80, max_depth=5, random_state=42),
      "lr": LogisticRegression(max_iter=300, C=0.5),
    }
    probs = np.zeros(len(X_te))
    for name, mdl in models.items():
      mdl.fit(X_tr, y_tr)
      pr = mdl.predict_proba(X_te)[:, 1]
      probs += pr / len(models)
    try:
      auc = roc_auc_score(y_te, probs)
      ll = log_loss(y_te, np.clip(probs, 1e-6, 1 - 1e-6))
    except Exception:
      auc, ll = np.nan, np.nan
    results.append({"year": test_year, "auc": auc, "logloss": ll, "n_test": len(test)})
    tmp = test[["date", "ticker", "fwd_ret_20"]].copy()
    tmp["ml_prob"] = probs
    preds.append(tmp)
  res_df = pd.DataFrame(results)
  pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
  ml_ok = len(res_df) and res_df["auc"].mean() > 0.52 and res_df["auc"].std() < 0.08
  return res_df, pred_df, ml_ok


ml_results, ml_preds, ml_ok = (pd.DataFrame(), pd.DataFrame(), False)
if USE_ML:
  ml_results, ml_preds, ml_ok = ml_ranker_walk_forward(panel_w)
  if len(ml_results):
    ml_results.to_csv("research_v17_ml_results.csv", index=False)
  if len(ml_preds):
    ml_preds.to_csv("research_v17_ml_predictions.csv", index=False)
print("ML OOS OK:", ml_ok, "| research-only default:", ML_RESEARCH_ONLY_BY_DEFAULT or not ml_ok)

# %% [markdown]
# ## 10. Portfolio engine universal

# %%
def get_rebalance_dates(close):
  return close.resample(REBALANCE_FREQ).last().dropna(how="all").index


def calculate_turnover(nw, ow):
  idx = nw.index.union(ow.index)
  return float((nw.reindex(idx, fill_value=0) - ow.reindex(idx, fill_value=0)).abs().sum() / 2)


def cap_sector_weights(w, sector_map, sector_cap=SECTOR_CAP, max_w=0.075):
  w = w.copy()
  for _ in range(8):
    sec_sums = {}
    for t, wt in w.items():
      sec = sector_map.get(t, "OTHER")
      sec_sums[sec] = sec_sums.get(sec, 0) + wt
    over_sec = {s: v for s, v in sec_sums.items() if v > sector_cap}
    if not over_sec:
      break
    for sec, ex in over_sec.items():
      tickers = [t for t in w.index if sector_map.get(t, "OTHER") == sec]
      if not tickers:
        continue
      scale = sector_cap / sec_sums[sec]
      for t in tickers:
        w[t] *= scale
  w = w.clip(upper=max_w)
  return w / w.sum() if w.sum() > 0 else w


def build_weights_from_scores(panel_df, close, dt, top_n=20, max_w=0.075, min_score=60, vol_target=0.15, score_col="alpha_score"):
  sub = panel_df[panel_df["date"] == dt].copy()
  if score_col not in sub.columns:
    return pd.Series({CASH_ASSET: 1.0})
  thr = min_score if sub[score_col].max() > 1.5 else min_score / 100.0
  sub = sub[sub[score_col] >= thr].sort_values(score_col, ascending=False)
  if sub.empty:
    return pd.Series({CASH_ASSET: 1.0})
  top = sub.head(top_n)
  vols = top["vol_60"].fillna(0.15).clip(lower=0.05)
  inv = 1.0 / vols
  w = pd.Series({r["ticker"]: inv.loc[i] for i, r in top.iterrows()})
  w = w / w.sum()
  stress = _sf(top["market_stress_score"].iloc[0], 0)
  scale = 0.55 if stress > 55 else (0.75 if stress > 35 else min(1.0, vol_target / 0.15))
  w = w * scale
  sec_map = {r["ticker"]: r.get("sector", "OTHER") for _, r in top.iterrows()}
  w = cap_sector_weights(w, sec_map, SECTOR_CAP, max_w)
  if w.sum() < 1.0 and CASH_ASSET in close.columns:
    w[CASH_ASSET] = w.get(CASH_ASSET, 0) + (1.0 - w.sum())
  return w / w.sum() if w.sum() > 0 else pd.Series({CASH_ASSET: 1.0})


def build_schedule_from_scores(panel_df, close, score_col="alpha_score", **kwargs):
  sched = {}
  for dt in rdates:
    if dt not in panel_df["date"].values:
      continue
    sched[dt] = build_weights_from_scores(panel_df, close, dt, score_col=score_col, **kwargs)
  return sched


def build_equity_curve(close, schedule, cost_rate=COST_RATE):
  dates = close.index
  rebal_exec = {dates[dates > d][0]: w for d, w in schedule.items() if len(dates[dates > d])}
  equity, curve, current_w, to_log = INITIAL_CAPITAL, {}, pd.Series(dtype=float), []
  for i, dt in enumerate(dates):
    if dt in rebal_exec:
      nw = rebal_exec[dt]
      to = calculate_turnover(nw, current_w) if len(current_w) else nw.sum()
      equity *= (1 - to * cost_rate)
      to_log.append(to)
      current_w = nw
    if i > 0 and len(current_w):
      prev = dates[i - 1]
      dr = sum(current_w.get(t, 0) * (close[t].loc[dt] / close[t].loc[prev] - 1)
               for t in current_w.index if t in close.columns)
      equity *= (1 + dr)
    curve[dt] = equity
  return pd.Series(curve), np.mean(to_log) if to_log else 0


def compare_metrics(equity, close):
  rets = equity.pct_change().fillna(0)
  yearly = []
  for year in sorted(set(equity.index.year)):
    e = equity.loc[f"{year}-01-01":f"{year}-12-31"]
    if len(e) < 2:
      continue
    sr = e.iloc[-1] / e.iloc[0] - 1
    spy = close[MARKET].loc[f"{year}-01-01":f"{year}-12-31"] if MARKET in close else pd.Series()
    qqq = close["QQQ"].loc[f"{year}-01-01":f"{year}-12-31"] if "QQQ" in close else pd.Series()
    spy_r = spy.iloc[-1] / spy.iloc[0] - 1 if len(spy) >= 2 else 0
    qqq_r = qqq.iloc[-1] / qqq.iloc[0] - 1 if len(qqq) >= 2 else 0
    ew_cols = [c for c in close.columns if c not in DEFENSIVE_ETFS]
    ew = close[ew_cols].loc[f"{year}-01-01":f"{year}-12-31"].pct_change().mean(axis=1).fillna(0)
    ew_r = (1 + ew).prod() - 1 if len(ew) else 0
    tlt = close["TLT"].loc[f"{year}-01-01":f"{year}-12-31"] if "TLT" in close else pd.Series()
    mix = 0.6 * spy_r + 0.4 * (tlt.iloc[-1] / tlt.iloc[0] - 1 if len(tlt) >= 2 else 0)
    yearly.append({"year": year, "return": sr, "beats_spy": sr > spy_r, "beats_qqq": sr > qqq_r,
                   "beats_ew": sr > ew_r, "beats_6040": sr > mix})
  ydf = pd.DataFrame(yearly)
  mdd = calculate_max_drawdown(equity)
  return {
    "total_return": round((equity.iloc[-1] / equity.iloc[0] - 1) * 100, 2),
    "CAGR": round(calculate_cagr(equity) * 100, 2),
    "sharpe": round(calculate_sharpe(rets), 3),
    "sortino": round(calculate_sortino(rets), 3),
    "max_drawdown": round(mdd * 100, 2),
    "calmar": round(calculate_cagr(equity) / abs(mdd), 3) if mdd else 0,
    "annual_volatility": round(rets.std() * math.sqrt(252) * 100, 2),
    "win_years_vs_spy": ydf["beats_spy"].mean() if len(ydf) else 0,
    "win_years_vs_qqq": ydf["beats_qqq"].mean() if len(ydf) else 0,
    "win_years_vs_equal_weight": ydf["beats_ew"].mean() if len(ydf) else 0,
    "win_years_vs_6040": ydf["beats_6040"].mean() if len(ydf) else 0,
  }, ydf


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


def contribution_by_asset(close, schedule):
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
      if t in close.columns:
        ar = close[t].loc[dt] / close[t].loc[prev] - 1
        if np.isfinite(ar):
          contrib[t] = contrib.get(t, 0) + current_w.get(t, 0) * ar
  if contrib.empty:
    return pd.DataFrame()
  total = contrib.abs().sum()
  out = contrib.sort_values(ascending=False).reset_index()
  out.columns = ["ticker", "contribution"]
  out["pct_of_total"] = (out["contribution"].abs() / total * 100).round(2) if total > 0 else 0
  out["sector"] = out["ticker"].map(SECTOR_MAP).fillna("OTHER")
  return out


def run_universal_alpha_portfolio(panel_df, close, name, top_n=20, max_w=0.075, vol_target=0.15, score_col="alpha_score"):
  sched = build_schedule_from_scores(panel_df, close, score_col=score_col, top_n=top_n, max_w=max_w, vol_target=vol_target)
  eq, turnover = build_equity_curve(close, sched)
  if len(eq) < 2:
    return {}, eq, pd.DataFrame(), sched, pd.DataFrame()
  m, yearly = compare_metrics(eq, close)
  contrib = contribution_by_asset(close, sched)
  top_pct = contrib.iloc[0]["pct_of_total"] if len(contrib) else 0
  m.update({"strategy": name, "turnover": round(turnover, 4), "num_rebalances": len(sched),
            "concentration_risk": "HIGH" if top_pct > 35 else ("MEDIUM" if top_pct > 25 else "LOW"),
            "top_contributor_pct": top_pct})
  yearly["strategy"] = name
  return m, eq, yearly, sched, contrib

# %% [markdown]
# ## 11. Configuraciones U1-U8

# %%
def v14_weights_simple(close, dt):
  sub = panel[(panel["date"] == dt) & (panel["alpha_score"] >= 60)].sort_values("alpha_score", ascending=False).head(3)
  if sub.empty:
    return pd.Series({CASH_ASSET: 1.0})
  vols = sub["vol_60"].fillna(0.15)
  w = pd.Series({r["ticker"]: (1 / vols.loc[i]) for i, r in sub.iterrows()})
  return w / w.sum()


def blend_schedules(sa, sb, wa=0.5):
  dates = sorted(set(sa.keys()) | set(sb.keys()))
  out = {}
  for dt in dates:
    w = pd.Series(dtype=float)
    if dt in sa:
      w = w.add(sa[dt] * wa, fill_value=0)
    if dt in sb:
      w = w.add(sb[dt] * (1 - wa), fill_value=0)
    out[dt] = w / w.sum() if w.sum() > 0 else pd.Series({CASH_ASSET: 1.0})
  return out


STRATEGY_CONFIGS = {
  "U1_COMPOSITE_TOP20": {"top_n": 20, "max_w": 0.075, "vol_target": 0.15, "score_col": "alpha_score"},
  "U2_COMPOSITE_TOP30_DIVERSIFIED": {"top_n": 30, "max_w": 0.05, "vol_target": 0.15, "score_col": "alpha_score"},
  "U3_MOMENTUM_TREND_TOP20": {"top_n": 20, "max_w": 0.075, "vol_target": 0.15, "score_col": "score_momentum_trend"},
  "U4_EVENT_MOMENTUM_FILTER": {"top_n": 15, "max_w": 0.075, "vol_target": 0.15, "score_col": "score_event_momentum"},
  "U5_LOW_VOL_MOMENTUM": {"top_n": 20, "max_w": 0.075, "vol_target": 0.12, "score_col": "score_low_vol_momentum"},
}

strategy_results, equities, yearly_all, schedules, contribs = [], {}, [], {}, {}
for name, cfg in STRATEGY_CONFIGS.items():
  m, eq, yr, sch, cb = run_universal_alpha_portfolio(panel, close_prices, name, **cfg)
  if m:
    strategy_results.append(m); equities[name] = eq; yearly_all.append(yr); schedules[name] = sch; contribs[name] = cb

if ml_ok and not ML_RESEARCH_ONLY_BY_DEFAULT:
  panel_ml = panel.merge(ml_preds, on=["date", "ticker"], how="left")
  panel_ml["ml_score"] = panel_ml["ml_prob"].fillna(0) * 100
  m, eq, yr, sch, cb = run_universal_alpha_portfolio(panel_ml, close_prices, "U6_ML_RANKER_TOP20", top_n=20, max_w=0.075, score_col="ml_score")
  if m:
    m["research_only"] = False
    strategy_results.append(m); equities["U6_ML_RANKER_TOP20"] = eq; yearly_all.append(yr); schedules["U6_ML_RANKER_TOP20"] = sch
else:
  print("U6_ML_RANKER_TOP20: research-only (no champion)")

u2 = schedules.get("U2_COMPOSITE_TOP30_DIVERSIFIED", build_schedule_from_scores(panel, close_prices, top_n=30, max_w=0.05))
v14s = {dt: v14_weights_simple(close_prices, dt) for dt in rdates}
for blend_name, sa in [("U7_V14_PLUS_UNIVERSAL_ALPHA", v14s), ("U8_V15_PLUS_UNIVERSAL_ALPHA", v14s)]:
  sch = blend_schedules(sa, u2, 0.5)
  eq, turnover = build_equity_curve(close_prices, sch)
  if len(eq) >= 2:
    m, yr = compare_metrics(eq, close_prices)
    m.update({"strategy": blend_name, "turnover": turnover, "num_rebalances": len(sch)})
    strategy_results.append(m); equities[blend_name] = eq; yr["strategy"] = blend_name; yearly_all.append(yr); schedules[blend_name] = sch

results_df = pd.DataFrame(strategy_results)
yearly_df = pd.concat(yearly_all, ignore_index=True) if yearly_all else pd.DataFrame()
print(results_df[["strategy", "CAGR", "sharpe", "max_drawdown", "win_years_vs_spy"]].to_string(index=False) if len(results_df) else "Sin resultados")

# %% [markdown]
# ## 12-14. Walk-forward, robustez, scoring

# %%
def run_v17_walk_forward():
  rows = []
  for test_year in sorted(panel["date"].dt.year.unique()):
    if test_year <= WF_START_YEAR:
      continue
    train = panel[panel["date"].dt.year < test_year]
    test = panel[panel["date"].dt.year == test_year]
    if len(train) < 300 or len(test) < 50:
      continue
    tr_w = train[train["date"].isin(rdates)]
    _, _, sel = factor_ic_analysis(tr_w)
    if not sel:
      sel = selected_factors
    test_scored = build_alpha_composite_score(test, sel)
    test_dates = [d for d in rdates if d.year == test_year]
    sched = {dt: build_weights_from_scores(test_scored, close_prices, dt, top_n=20, max_w=0.075) for dt in test_dates}
    eq, _ = build_equity_curve(close_prices, sched)
    seg = eq.loc[f"{test_year}-01-01":f"{test_year}-12-31"]
    if len(seg) >= 2:
      rows.append({"year": test_year, "return": round(seg.iloc[-1] / seg.iloc[0] - 1, 4), "factors": ",".join(sel[:5])})
  return pd.DataFrame(rows)


wf_df = run_v17_walk_forward()


def compute_robustness_score(df):
  if len(df) < 2:
    return 80
  gap = df["sharpe"].max() - df["sharpe"].median()
  return int(np.clip(100 - gap * 30 - max(0, len(df) - 8) * 2, 0, 100))


def overfitting_risk(df):
  if len(df) < 2:
    return "LOW"
  gap = df["sharpe"].max() - df["sharpe"].median()
  return "HIGH" if gap > 0.5 else ("MEDIUM" if gap > 0.28 else "LOW")


def compute_v17_score(row, robustness, ovr, spy_total, ic_stable=True, quick_only=False):
  score = 0
  if row.get("sharpe", 0) > 1.20:
    score += 15
  if row.get("sortino", 0) > 1.70:
    score += 15
  if row.get("max_drawdown", -100) > -25:
    score += 15
  if row.get("CAGR", 0) > 18:
    score += 10
  if row.get("total_return", 0) > spy_total:
    score += 10
  if row.get("win_years_vs_spy", 0) >= 0.60:
    score += 10
  if row.get("win_years_vs_qqq", 0) >= 0.45:
    score += 5
  if row.get("win_years_vs_equal_weight", 0) >= 0.55:
    score += 10
  if row.get("win_years_vs_6040", 0) >= 0.75:
    score += 10
  if robustness >= 80:
    score += 10
  if row.get("max_drawdown", -100) < -30:
    score -= 25
  if row.get("concentration_risk") == "HIGH":
    score -= 20
  if ovr == "HIGH":
    score -= 20
  if quick_only and UNIVERSE_MODE == "QUICK_TEST":
    score -= 20
  if not ic_stable:
    score -= 20
  return int(np.clip(score, 0, 100))


robustness_score = compute_robustness_score(results_df)
ovr = overfitting_risk(results_df)
spy_total = (close_prices[MARKET].iloc[-1] / close_prices[MARKET].iloc[0] - 1) * 100
ic_stable = len(ic_df) and ic_df["hit_rate_ic_positive"].mean() > 0.52

if len(results_df):
  u_df = results_df[~results_df["strategy"].str.contains("RESEARCH", na=False)].copy()
  if "research_only" in u_df.columns:
    u_df = u_df[u_df["research_only"] != True]
  u_df["v17_score"] = u_df.apply(lambda r: compute_v17_score(r, robustness_score, ovr, spy_total, ic_stable, QUICK_TEST), axis=1)
  results_df = results_df.merge(u_df[["strategy", "v17_score"]], on="strategy", how="left")
  champion = u_df.sort_values("v17_score", ascending=False).iloc[0]
else:
  champion = pd.Series(dtype=float)

v17_score = int(champion.get("v17_score", 0)) if len(champion) else 0
approved = (v17_score >= 85 and champion.get("sharpe", 0) > 1.15 and champion.get("max_drawdown", -100) > -25
            and robustness_score >= 80 and champion.get("concentration_risk") != "HIGH" and ovr != "HIGH")
v17_status = "APPROVED_FOR_WEB_PAPER" if approved else ("CANDIDATE" if v17_score >= 70 else "REJECTED")

rob_rows = []
best_name = champion.get("strategy", "U2_COMPOSITE_TOP30_DIVERSIFIED") if len(champion) else "U2_COMPOSITE_TOP30_DIVERSIFIED"
for sd in START_DATES_ROBUST:
  if sd >= START_DATE:
    continue
  try:
    c2 = close_prices.loc[sd:].copy() if pd.Timestamp(sd) >= close_prices.index[0] else close_prices
    p2 = panel[panel["date"] >= sd]
    sch = build_schedule_from_scores(p2, c2, top_n=20, max_w=0.075)
    eq, _ = build_equity_curve(c2, sch)
    if len(eq) >= 2:
      m, _ = compare_metrics(eq, c2)
      rob_rows.append({"start_date": sd, "strategy": best_name, **m})
  except Exception:
    pass
robustness_df = pd.DataFrame(rob_rows)

cost_rows = []
for cr in COST_SENSITIVITY:
  eq, _ = build_equity_curve(close_prices, schedules.get(best_name, {}), cost_rate=cr)
  if len(eq) >= 2:
    m, _ = compare_metrics(eq, close_prices)
    cost_rows.append({"cost_slippage": cr, **m})
cost_sens_df = pd.DataFrame(cost_rows)
contrib_df = contribs.get(best_name, pd.DataFrame())
sector_df = contrib_df.groupby("sector")["pct_of_total"].sum().reset_index() if len(contrib_df) and "sector" in contrib_df else pd.DataFrame()

# %% [markdown]
# ## 15. Current signals

# %%
def generate_v17_signals(sched, panel_df, strategy_name):
  if not sched:
    return pd.DataFrame()
  last = max(sched.keys())
  prev = sorted([d for d in sched.keys() if d < last])[-1] if len(sched) > 1 else None
  target, previous = sched[last], sched.get(prev, pd.Series(dtype=float))
  last_panel = panel_df[panel_df["date"] == last].set_index("ticker")
  regime = _sf(last_panel["market_risk_on"].iloc[0], 1) if len(last_panel) else 1
  rows = []
  for t in sorted(set(target.index) | set(previous.index)):
    tw, pw = _sf(target.get(t, 0)), _sf(previous.get(t, 0))
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
    info = last_panel.loc[t] if t in last_panel.index else None
    asc = _sf(info["alpha_score"], 0) if info is not None else 0
    driver = info["main_factor_driver"] if info is not None else "composite"
    rows.append({
      "ticker": t, "signal": sig, "target_weight": round(tw, 4), "previous_weight": round(pw, 4),
      "change": round(chg, 4), "alpha_score": round(asc, 1), "main_factor_driver": driver,
      "market_regime": "risk_on" if regime >= 0.5 else "risk_off",
      "confidence": round(min(95, asc), 1),
      "reason": f"alpha_factory {driver} | score {asc:.0f}",
      "entry_plan": "proxima apertura post-viernes" if sig in ("BUY", "INCREASE") else "-",
      "exit_plan": "rebalance semanal o score < 60",
      "next_review": "proximo viernes", "cash_account_executable": True,
    })
  return pd.DataFrame(rows)


champ_name = champion.get("strategy", "U2_COMPOSITE_TOP30_DIVERSIFIED") if len(champion) else "U2_COMPOSITE_TOP30_DIVERSIFIED"
champ_sched = schedules.get(champ_name, {})
current_signals = generate_v17_signals(champ_sched, panel, champ_name)

# %% [markdown]
# ## 16-17. Exportar y reporte

# %%
def export_csv(df, name):
  (df if df is not None and len(df) else pd.DataFrame()).to_csv(name, index=False)
  print("Exportado", name)


summary = {
  "lab": "v17_universal_equity_alpha_factory",
  "v17_score": v17_score, "status": v17_status,
  "approved_for_web_paper": approved, "approved_for_real_money": False,
  "universe_mode": UNIVERSE_MODE, "tickers_initial": audit["initial_tickers"],
  "tickers_valid": audit["valid_history_liquidity"],
  "champion_strategy": champ_name, "selected_factors": ",".join(selected_factors[:8]),
  "robustness_score": robustness_score, "overfitting_risk": ovr,
  "ic_stable": ic_stable, "ml_ok": ml_ok,
  **(champion.to_dict() if len(champion) else {}),
  "survivorship_warning": SURVIVORSHIP_WARNING,
}

export_csv(pd.DataFrame([summary]), "research_v17_summary.csv")
export_csv(results_df, "research_v17_strategy_results.csv")
export_csv(yearly_df, "research_v17_yearly.csv")
export_csv(robustness_df, "research_v17_robustness.csv")
export_csv(cost_sens_df, "research_v17_cost_sensitivity.csv")
export_csv(contrib_df, "research_v17_contribution_by_asset.csv")
export_csv(sector_df, "research_v17_sector_contribution.csv")
export_csv(current_signals, "research_v17_current_signals.csv")
if champ_name in equities and len(equities[champ_name]):
  equities[champ_name].to_frame("equity").to_csv("research_v17_equity_curve.csv")
if len(current_signals):
  current_signals.to_csv("current_signals_v17.csv", index=False)

config = {
  "version": "v17_universal_equity_alpha_factory",
  "approved_for_web_paper": approved, "approved_for_real_money": False,
  "v17_score": v17_score, "status": v17_status, "champion": champ_name,
  "universe_mode": UNIVERSE_MODE, "selected_factors": selected_factors,
  "warnings": [SURVIVORSHIP_WARNING, "ML research-only unless OOS improvement proven"],
}
Path("research_v17_selected_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
print("Exportado research_v17_selected_config.json")

print("=" * 80)
print("REPORTE FINAL V17 UNIVERSAL EQUITY ALPHA FACTORY")
print("=" * 80)
print(f"Estado: {v17_status} | Score: {v17_score}/100")
print(f"Universo: {UNIVERSE_MODE} | Tickers validos: {audit['valid_history_liquidity']}")
print(f"Mejor estrategia: {champ_name}")
print(f"Factores: {selected_factors[:6]}")
if len(champion):
  print(f"CAGR {champion.get('CAGR')}% | Sharpe {champion.get('sharpe')} | DD {champion.get('max_drawdown')}%")
  print(f"Win SPY {champion.get('win_years_vs_spy', 0)*100:.0f}% | EW {champion.get('win_years_vs_equal_weight', 0)*100:.0f}%")
print(f"IC top: {ic_df.head(3)[['factor','mean_ic','hit_rate_ic_positive']].to_string(index=False) if len(ic_df) else 'N/A'}")
print(f"Robustez: {robustness_score} | Overfitting: {ovr} | Concentration: {champion.get('concentration_risk', 'N/A')}")
if len(current_signals):
  act = current_signals[current_signals["target_weight"] > 0.001]
  print(f"Posiciones: {len(act)} | BUY: {len(act[act['signal']=='BUY'])}")
  print(act.head(8)[["ticker", "signal", "target_weight", "alpha_score", "main_factor_driver"]].to_string(index=False))
print("")
if approved:
  print("Integrar V17 Universal Equity Alpha Factory.")
else:
  print("V17 no mejora suficiente. Mantener V14 como approved y revisar factores.")
print("APPROVED_FOR_REAL_MONEY=False (siempre)")
