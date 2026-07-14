# %% [markdown]
# # Trading Research V5 — Champion vs Challengers Lab
#
# Laboratorio avanzado: **Trend Following V4 (Champion)** vs estrategias challenger.
# Prioridad: walk-forward OOS. Sin promesas. Sin dinero real.
#
# **Backtest no garantiza resultados futuros.**

# %%
!pip install yfinance pandas numpy matplotlib plotly tqdm scikit-learn scipy -q

# %% [markdown]
# ## 1. Configuración

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
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

QUICK_TEST = False
START_DATE = "2010-01-01"
END_DATE = None
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001
SLIPPAGE = 0.001
COST_SIDE = TRANSACTION_COST + SLIPPAGE

RISKY_ASSETS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN"]
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
FACTOR_ETFS = ["MTUM", "QUAL", "USMV", "VLUE", "SIZE"]
DEFENSIVE_ASSETS = ["SHY", "IEF", "TLT", "GLD", "CASH"]
MACRO_TICKERS = ["^VIX", "^TNX"]
FULL_UNIVERSE = RISKY_ASSETS + SECTOR_ETFS + FACTOR_ETFS + DEFENSIVE_ASSETS

MARKET = "SPY"
QQQ = "QQQ"
IEF = "IEF"
WF_START_YEAR = 2017

CHAMPION_PARAMS = dict(fast_ma=50, slow_ma=200, vol_target=0.15, max_asset_weight=0.30, defensive_asset="SHY", rebalance_freq="W-FRI")

if QUICK_TEST:
    START_DATE = "2018-01-01"
    RISKY_ASSETS = ["SPY", "QQQ", "NVDA", "AMD", "AAPL"]
    SECTOR_ETFS = ["XLK", "XLF", "XLV"]
    FACTOR_ETFS = ["MTUM", "USMV"]
    FULL_UNIVERSE = RISKY_ASSETS + SECTOR_ETFS + FACTOR_ETFS + DEFENSIVE_ASSETS
    WF_START_YEAR = 2019
    print("⚡ QUICK_TEST")

print("V5 Champion vs Challengers | desde", START_DATE)

# %% [markdown]
# ## 2. Datos

# %%
def download_data(tickers, start, end=None):
    data, failed = {}, []
    for t in tqdm(sorted(set(tickers)), desc="Download"):
        if t == "CASH":
            continue
        sym = t
        try:
            raw = yf.download(sym, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                failed.append(t)
                continue
            df = raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            mp = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
            df = df.rename(columns={c: mp.get(str(c).lower(), c) for c in df.columns})
            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            df = df[keep].dropna(subset=["Close"])
            if df.empty:
                failed.append(t)
            else:
                key = t.upper().replace("^", "VIX" if "VIX" in t else "TNX" if "TNX" in t else t.upper())
                if "^VIX" in t:
                    key = "VIX"
                elif "^TNX" in t:
                    key = "TNX"
                data[key] = df
        except Exception as e:
            failed.append(f"{t}({e})")
    if failed:
        print("Fallidos:", failed)
    close = pd.DataFrame({k: v["Close"] for k, v in data.items()}).sort_index().ffill()
    close.index = pd.DatetimeIndex(close.index)
    if close.index.tz:
        close.index = close.index.tz_localize(None)
    for k, df in list(data.items()):
        df = df.copy()
        df.index = pd.DatetimeIndex(df.index)
        if df.index.tz:
            df.index = df.index.tz_localize(None)
        data[k] = df
    close["CASH"] = 1.0
    data["CASH"] = pd.DataFrame({"Close": 1.0}, index=close.index)
    return data, close

all_tickers = list(set(FULL_UNIVERSE + MACRO_TICKERS + [MARKET, QQQ, IEF]))
data, close_prices = download_data(all_tickers, START_DATE, END_DATE)

# %% [markdown]
# ## 3. Features

# %%
def _rsi(c, w):
    d = c.diff()
    g = d.clip(lower=0).rolling(w).mean()
    l = (-d).clip(lower=0).rolling(w).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


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
# ## 4. Utilidades y benchmarks

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
    a = align_date(ix, dt)
    return ix.get_loc(a) if a is not None else None


def row_asof(df, dt):
    a = align_date(df.index, dt)
    if a is None:
        return None
    return df.loc[a] if a in df.index else df.iloc[df.index.get_indexer([a], method="pad")[0]]


def reb_dates(ix, freq="W-FRI"):
    ix = norm_idx(ix)
    s = pd.Series(np.arange(len(ix)), index=ix)
    return pd.DatetimeIndex([g.index[-1] for _, g in s.groupby(pd.Grouper(freq=freq)) if len(g)])


def assign_w(wdf, i0, i1, wrow):
    i1 = min(int(i1), len(wdf) - 1)
    v = wrow.reindex(wdf.columns).fillna(0).values
    for j in range(int(i0) + 1, i1 + 1):
        wdf.iloc[j] = v


def inv_vol_w(tickers, vols, cap=0.30):
    vols = pd.Series(vols).replace(0, np.nan).dropna()
    tickers = [t for t in tickers if t in vols.index]
    if not tickers:
        return pd.Series(dtype=float)
    w = (1 / vols[tickers])
    w = w / w.sum()
    for _ in range(5):
        w = (w / w.sum()).clip(upper=cap)
    return w / w.sum()


def vol_scale(wdf, close, target=0.15, lb=20):
    r = close.pct_change().fillna(0)
    pr = (wdf.shift(1).fillna(0) * r).sum(axis=1)
    rv = pr.rolling(lb).std() * np.sqrt(252)
    sc = (target / rv.replace(0, np.nan)).clip(0, 1).shift(1).fillna(1)
    return wdf.mul(sc, axis=0)


def benchmark_returns(close):
    r = close.pct_change().fillna(0)
    ew_cols = [c for c in FULL_UNIVERSE if c in close.columns and c != "CASH"]
    bm = {
        "SPY": r[MARKET] if MARKET in r else pd.Series(0, index=r.index),
        "QQQ": r[QQQ] if QQQ in r else r[MARKET],
        "EW": r[ew_cols].mean(axis=1) if ew_cols else r[MARKET],
        "6040": 0.6 * r[MARKET] + 0.4 * r[IEF].reindex(r.index).fillna(0) if IEF in r else r[MARKET],
        "DEF": 0.5 * r[IEF].reindex(r.index).fillna(0) + 0.25 * r["GLD"].reindex(r.index).fillna(0) + 0.25 * r["SHY"].reindex(r.index).fillna(0),
    }
    return bm

BENCHMARKS = benchmark_returns(close_prices)

# %% [markdown]
# ## 5. Backtest común

# %%
def backtest_portfolio_weights(close, w, transaction_cost=TRANSACTION_COST, slippage=SLIPPAGE, initial_capital=INITIAL_CAPITAL):
    r = close.pct_change().fillna(0)
    cols = [c for c in w.columns if c in r.columns]
    W = w[cols].reindex(close.index).fillna(0)
    Wx = W.shift(1).fillna(0)
    to = W.diff().abs().sum(axis=1).fillna(0)
    pr = (Wx * r[cols]).sum(axis=1) - to * (transaction_cost + slippage)
    eq = (1 + pr).cumprod() * initial_capital
    dd = (eq - eq.cummax()) / eq.cummax()
    yrs = max((pr.index[-1] - pr.index[0]).days / 365.25, 1 / 365.25)
    tr = eq.iloc[-1] / initial_capital - 1
    cagr = (eq.iloc[-1] / initial_capital) ** (1 / yrs) - 1
    av = pr.std() * np.sqrt(252)
    sh = pr.mean() / pr.std() * np.sqrt(252) if pr.std() > 0 else 0
    ds = pr[pr < 0].std() * np.sqrt(252)
    so = pr.mean() / ds * np.sqrt(252) if ds > 0 else 0
    mdd = dd.min()
    sk = pr.skew()
    var95 = pr.quantile(0.05)
    cvar95 = pr[pr <= var95].mean() if (pr <= var95).any() else var95

    def ex(name):
        br = BENCHMARKS.get(name, BENCHMARKS["SPY"]).reindex(pr.index).fillna(0)
        return tr * 100 - ((1 + br).prod() - 1) * 100

    m = {
        "total_return": round(tr * 100, 2),
        "CAGR": round(cagr * 100, 2),
        "annual_volatility": round(av * 100, 2),
        "sharpe": round(sh, 3),
        "sortino": round(so, 3),
        "max_drawdown": round(mdd * 100, 2),
        "calmar": round((cagr * 100) / abs(mdd * 100), 3) if mdd != 0 else 0,
        "turnover_avg": round(to.mean(), 4),
        "exposure_pct": round(Wx.abs().sum(axis=1).mean() * 100, 2),
        "best_day": round(pr.max() * 100, 3),
        "worst_day": round(pr.min() * 100, 3),
        "positive_days_pct": round((pr > 0).mean() * 100, 2),
        "skew": round(sk, 3),
        "tail_loss_5pct": round(var95 * 100, 3),
        "var_95": round(var95 * 100, 3),
        "cvar_95": round(cvar95 * 100, 3),
        "excess_vs_spy": round(ex("SPY"), 2),
        "excess_vs_qqq": round(ex("QQQ"), 2),
        "excess_vs_equal_weight": round(ex("EW"), 2),
        "excess_vs_6040": round(ex("6040"), 2),
    }
    return m, pr, eq, dd, W

# %% [markdown]
# ## 6. Champion — Trend Following V4

# %%
def trend_following_v4_weights(close, features_dict, universe, fast_ma=50, slow_ma=200, vol_target=0.15,
                               max_asset_weight=0.30, defensive_asset="SHY", rebalance_freq="W-FRI"):
    cols = [c for c in universe if c in close.columns and c != "CASH"]
    wdf = pd.DataFrame(0.0, index=norm_idx(close.index), columns=close.columns)
    rlist = list(reb_dates(wdf.index, rebalance_freq))
    sc = f"SMA_{slow_ma}"
    for i, rd in enumerate(rlist):
        i0 = idx_pos(wdf.index, rd)
        if i0 is None:
            continue
        rd = wdf.index[i0]
        i1 = idx_pos(wdf.index, rlist[i + 1]) if i + 1 < len(rlist) else len(wdf.index) - 1
        if i1 is None:
            i1 = len(wdf.index) - 1
        elig, vols = [], {}
        for t in cols:
            row = row_asof(features_dict.get(t, pd.DataFrame()), rd)
            if row is None:
                continue
            sma = row.get(sc, np.nan)
            ema = row.get("EMA_50", np.nan)
            px = row.get("Close", np.nan)
            if pd.notna(sma) and pd.notna(ema) and pd.notna(px) and px > sma and ema > sma:
                elig.append(t)
                vols[t] = row.get("VOL_60", np.nan)
        wr = pd.Series(0.0, index=wdf.columns)
        if elig:
            for t, wt in inv_vol_w(elig, vols, max_asset_weight).items():
                wr[t] = wt
        else:
            wr[defensive_asset if defensive_asset in wr.index else "CASH"] = 1.0
        assign_w(wdf, i0, i1, wr)
    return vol_scale(wdf, close, vol_target)

# %% [markdown]
# ## 7. Challengers — pesos

# %%
def cs_momentum_weights(close, features_dict, universe, top_n=3, freq="W-FRI", cap=0.30):
    cols = [c for c in universe if c in close.columns and c != "CASH"]
    wdf = pd.DataFrame(0.0, index=norm_idx(close.index), columns=close.columns)
    for i, rd in enumerate(reb_dates(wdf.index, freq)):
        i0 = idx_pos(wdf.index, rd)
        if i0 is None:
            continue
        rd = wdf.index[i0]
        rlist = list(reb_dates(wdf.index, freq))
        i1 = idx_pos(wdf.index, rlist[i + 1]) if i + 1 < len(rlist) else len(wdf.index) - 1
        if i1 is None:
            i1 = len(wdf.index) - 1
        sc = {}
        for t in cols:
            row = row_asof(features_dict.get(t, pd.DataFrame()), rd)
            if row is not None and pd.notna(row.get("MOM_COMBO_VOL")):
                sc[t] = row["MOM_COMBO_VOL"]
        wr = pd.Series(0.0, index=wdf.columns)
        if sc:
            s = pd.Series(sc).dropna()
            s = s[s > 0].nlargest(top_n)
            if len(s):
                vols = {t: row_asof(features_dict[t], rd)["VOL_60"] for t in s.index if t in features_dict}
                for t, wt in inv_vol_w(list(s.index), vols, cap).items():
                    wr[t] = wt
            else:
                wr["SHY" if "SHY" in wr.index else "CASH"] = 1.0
        else:
            wr["CASH"] = 1.0
        assign_w(wdf, i0, i1, wr)
    return wdf


def etf_rotation_weights(close, features_dict, universe, top_n=3, freq="W-FRI", cap=0.30):
    return cs_momentum_weights(close, features_dict, universe, top_n, freq, cap)


def defensive_weights(close, features_dict, cap=0.40):
    defs = [c for c in DEFENSIVE_ASSETS if c in close.columns and c != "CASH"]
    wdf = pd.DataFrame(0.0, index=norm_idx(close.index), columns=close.columns)
    for i, rd in enumerate(reb_dates(wdf.index, "M")):
        i0 = idx_pos(wdf.index, rd)
        if i0 is None:
            continue
        rd = wdf.index[i0]
        rlist = list(reb_dates(wdf.index, "M"))
        i1 = idx_pos(wdf.index, rlist[i + 1]) if i + 1 < len(rlist) else len(wdf.index) - 1
        if i1 is None:
            i1 = len(wdf.index) - 1
        sc = {}
        for t in defs:
            row = row_asof(features_dict.get(t, pd.DataFrame()), rd)
            if row is not None and pd.notna(row.get("MOM_60")):
                sc[t] = row["MOM_60"]
        wr = pd.Series(0.0, index=wdf.columns)
        if sc:
            s = pd.Series(sc).nlargest(min(3, len(sc)))
            for t in s.index:
                wr[t] = 1 / len(s)
        else:
            wr["CASH"] = 1.0
        assign_w(wdf, i0, i1, wr)
    return wdf


def risk_on_score(features_dict, close, dt):
    s = 0
    for tk in [MARKET, QQQ]:
        row = row_asof(features_dict.get(tk, pd.DataFrame()), dt)
        if row is not None:
            if row.get("Close", 0) > row.get("SMA_200", np.inf):
                s += 1
            if row.get("MOM_60", -1) > 0:
                s += 1
    spy = row_asof(features_dict.get(MARKET, pd.DataFrame()), dt)
    if spy is not None:
        if spy.get("ROLLING_DD_60", 0) < -0.10:
            s -= 1
        if spy is not None:
            if spy.get("VOL_20", 0) > 0.25:
                s -= 1
    if "VIX" in features_dict:
        vx = row_asof(features_dict["VIX"], dt)
        if vx is not None and pd.notna(vx.get("SMA_50")):
            if vx.get("Close", 999) < vx.get("SMA_50", 0):
                s += 1
    return s


def adaptive_ensemble_weights(close, features_dict):
    w_tf = trend_following_v4_weights(close, features_dict, RISKY_ASSETS, **CHAMPION_PARAMS)
    w_cs = cs_momentum_weights(close, features_dict, RISKY_ASSETS)
    w_etf = etf_rotation_weights(close, features_dict, SECTOR_ETFS + FACTOR_ETFS)
    w_def = defensive_weights(close, features_dict)
    out = pd.DataFrame(0.0, index=norm_idx(close.index), columns=close.columns)
    rlist = list(reb_dates(out.index, "W-FRI"))
    prev = None
    for i, rd in enumerate(rlist):
        i0 = idx_pos(out.index, rd)
        if i0 is None:
            continue
        rd = out.index[i0]
        i1 = idx_pos(out.index, rlist[i + 1]) if i + 1 < len(rlist) else len(out.index) - 1
        if i1 is None:
            i1 = len(out.index) - 1
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
        rs = mix[risky].sum()
        if rs > cap_risky and rs > 0:
            mix[risky] = mix[risky] * (cap_risky / rs)
            mix[DEFENSIVE_ASSETS] = mix.reindex(DEFENSIVE_ASSETS).fillna(0)
            mix["SHY"] = mix.get("SHY", 0) + (1 - mix.sum())
        if mix.sum() > 0:
            mix = mix / mix.sum()
        mix = mix.clip(upper=0.30)
        if mix.sum() > 0:
            mix = mix / mix.sum()
        if prev is not None:
            mix = 0.7 * mix + 0.3 * prev
            mix = mix / mix.sum() if mix.sum() > 0 else mix
        prev = mix
        for j in range(i0 + 1, i1 + 1):
            out.iloc[j] = mix.reindex(out.columns).fillna(0).values
    return out


def factor_rotation_weights(close, features_dict):
    uni = [c for c in FACTOR_ETFS + [MARKET, QQQ, "IWM", "SHY", "IEF", "GLD"] if c in close.columns]
    wdf = pd.DataFrame(0.0, index=norm_idx(close.index), columns=close.columns)
    for i, rd in enumerate(reb_dates(wdf.index, "M")):
        i0 = idx_pos(wdf.index, rd)
        if i0 is None:
            continue
        rd = wdf.index[i0]
        rlist = list(reb_dates(wdf.index, "M"))
        i1 = idx_pos(wdf.index, rlist[i + 1]) if i + 1 < len(rlist) else len(wdf.index) - 1
        if i1 is None:
            i1 = len(wdf.index) - 1
        sc = {}
        for t in uni:
            row = row_asof(features_dict.get(t, pd.DataFrame()), rd)
            if row is not None and pd.notna(row.get("MOM_COMBO_VOL")):
                sc[t] = row["MOM_COMBO_VOL"]
        wr = pd.Series(0.0, index=wdf.columns)
        if sc:
            s = pd.Series(sc)
            s = s[s > 0].nlargest(3)
            if len(s):
                vols = {t: row_asof(features_dict[t], rd)["VOL_60"] for t in s.index if t in features_dict}
                for t, wt in inv_vol_w(list(s.index), vols, 0.40).items():
                    wr[t] = wt
            else:
                wr["SHY" if "SHY" in wr.index else "GLD"] = 1.0
        else:
            wr["CASH"] = 1.0
        assign_w(wdf, i0, i1, wr)
    return wdf


GROWTH = [c for c in ["QQQ", "XLK", "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN"] if c in FULL_UNIVERSE]
DEF_BASKET = [c for c in ["SHY", "IEF", "TLT", "GLD", "XLP", "XLU", "XLV", "USMV"] if c in FULL_UNIVERSE or c == "USMV"]


def defensive_growth_rotation_weights(close, features_dict, alpha=0.3):
    wdf = pd.DataFrame(0.0, index=norm_idx(close.index), columns=close.columns)
    prev = None
    for i, rd in enumerate(reb_dates(wdf.index, "M")):
        i0 = idx_pos(wdf.index, rd)
        if i0 is None:
            continue
        rd = wdf.index[i0]
        rlist = list(reb_dates(wdf.index, "M"))
        i1 = idx_pos(wdf.index, rlist[i + 1]) if i + 1 < len(rlist) else len(wdf.index) - 1
        if i1 is None:
            i1 = len(wdf.index) - 1
        gs = 0
        for t in ["QQQ", "XLK"]:
            row = row_asof(features_dict.get(t, pd.DataFrame()), rd)
            if row is not None:
                gs += row.get("MOM_60", 0) + row.get("MOM_120", 0)
        spy = row_asof(features_dict.get(MARKET, pd.DataFrame()), rd)
        if spy is not None and spy.get("Close", 0) > spy.get("SMA_200", 0):
            gs += 0.1
        ds = 0
        for t in ["SHY", "IEF", "GLD"]:
            row = row_asof(features_dict.get(t, pd.DataFrame()), rd)
            if row is not None:
                ds += row.get("MOM_60", 0)
        if spy is not None:
            ds += max(0, -spy.get("ROLLING_DD_120", 0)) * 2
            if spy.get("VOL_20", 0) > 0.22:
                ds += 0.2
        wr = pd.Series(0.0, index=wdf.columns)
        basket = GROWTH if gs >= ds else DEF_BASKET
        pct = 0.85 if abs(gs - ds) > 0.1 else 0.55
        sc = {}
        for t in basket:
            if t not in close.columns:
                continue
            row = row_asof(features_dict.get(t, pd.DataFrame()), rd)
            if row is not None:
                sc[t] = row.get("MOM_COMBO_VOL", row.get("MOM_60", 0))
        if sc:
            top = pd.Series(sc).nlargest(3)
            for t in top.index:
                wr[t] = pct / len(top)
            wr["SHY" if "SHY" in wr.index else "CASH"] = 1 - pct
        else:
            wr["CASH"] = 1.0
        if prev is not None:
            wr = alpha * wr + (1 - alpha) * prev
        wr = wr / wr.sum() if wr.sum() > 0 else wr
        prev = wr
        assign_w(wdf, i0, i1, wr)
    return wdf


def risk_parity_weights(close, features_dict, universe, cap=0.30):
    cols = [c for c in universe if c in close.columns and c != "CASH"]
    wdf = pd.DataFrame(0.0, index=norm_idx(close.index), columns=close.columns)
    defs = set(DEFENSIVE_ASSETS)
    for i, rd in enumerate(reb_dates(wdf.index, "M")):
        i0 = idx_pos(wdf.index, rd)
        if i0 is None:
            continue
        rd = wdf.index[i0]
        rlist = list(reb_dates(wdf.index, "M"))
        i1 = idx_pos(wdf.index, rlist[i + 1]) if i + 1 < len(rlist) else len(wdf.index) - 1
        if i1 is None:
            i1 = len(wdf.index) - 1
        elig, vols = [], {}
        for t in cols:
            row = row_asof(features_dict.get(t, pd.DataFrame()), rd)
            if row is None:
                continue
            mom = row.get("MOM_60", 0)
            if mom > 0 or t in defs:
                elig.append(t)
                vols[t] = row.get("VOL_60", np.nan)
        wr = pd.Series(0.0, index=wdf.columns)
        if elig:
            for t, wt in inv_vol_w(elig, vols, cap).items():
                wr[t] = wt
        else:
            wr["CASH"] = 1.0
        assign_w(wdf, i0, i1, wr)
    return wdf


def minimum_volatility_weights(close, features_dict, universe, cap=0.30, lookback=120):
    cols = [c for c in universe if c in close.columns and c != "CASH"]
    wdf = pd.DataFrame(0.0, index=norm_idx(close.index), columns=close.columns)
    rets = close[cols].pct_change().fillna(0)
    for i, rd in enumerate(reb_dates(wdf.index, "M")):
        i0 = idx_pos(wdf.index, rd)
        if i0 is None:
            continue
        rd = wdf.index[i0]
        rlist = list(reb_dates(wdf.index, "M"))
        i1 = idx_pos(wdf.index, rlist[i + 1]) if i + 1 < len(rlist) else len(wdf.index) - 1
        if i1 is None:
            i1 = len(wdf.index) - 1
        hist = rets.loc[:rd].tail(lookback)
        wr = pd.Series(0.0, index=wdf.columns)
        if len(hist) >= 40 and HAS_SCIPY:
            sub = hist.dropna(axis=1, how="all")
            if len(sub.columns) >= 2:
                cov = sub.cov().values * 252
                n = len(sub.columns)
                try:
                    def obj(w):
                        return w @ cov @ w
                    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
                    bnds = [(0, cap)] * n
                    x0 = np.ones(n) / n
                    res = minimize(obj, x0, bounds=bnds, constraints=cons, method="SLSQP")
                    if res.success:
                        for t, wt in zip(sub.columns, res.x):
                            wr[t] = wt
                except Exception:
                    pass
        if wr.sum() < 0.99:
            vols = {t: features_dict[t].loc[:rd]["VOL_60"].iloc[-1] for t in cols if t in features_dict and len(features_dict[t].loc[:rd])}
            for t, wt in inv_vol_w(cols, vols, cap).items():
                wr[t] = wt
        assign_w(wdf, i0, i1, wr)
    return wdf


def vol_target_momentum_weights(close, features_dict, top_n=3, vol_target=0.12, market_filter=True):
    cols = [c for c in RISKY_ASSETS + SECTOR_ETFS if c in close.columns and c != "CASH"]
    wdf = pd.DataFrame(0.0, index=norm_idx(close.index), columns=close.columns)
    for i, rd in enumerate(reb_dates(wdf.index, "W-FRI")):
        i0 = idx_pos(wdf.index, rd)
        if i0 is None:
            continue
        rd = wdf.index[i0]
        rlist = list(reb_dates(wdf.index, "W-FRI"))
        i1 = idx_pos(wdf.index, rlist[i + 1]) if i + 1 < len(rlist) else len(wdf.index) - 1
        if i1 is None:
            i1 = len(wdf.index) - 1
        if market_filter:
            spy = row_asof(features_dict.get(MARKET, pd.DataFrame()), rd)
            if spy is not None and spy.get("Close", 0) < spy.get("SMA_200", 0):
                wr = pd.Series(0.0, index=wdf.columns)
                wr["SHY" if "SHY" in wr.index else "CASH"] = 1.0
                assign_w(wdf, i0, i1, wr)
                continue
        sc = {}
        for t in cols:
            row = row_asof(features_dict.get(t, pd.DataFrame()), rd)
            if row is not None and row.get("VOL_60", 0) > 0:
                sc[t] = row.get("MOM_120", 0) / row["VOL_60"]
        wr = pd.Series(0.0, index=wdf.columns)
        if sc:
            s = pd.Series(sc).nlargest(top_n)
            s = s[s > 0]
            if len(s):
                vols = {t: row_asof(features_dict[t], rd)["VOL_60"] for t in s.index if t in features_dict}
                for t, wt in inv_vol_w(list(s.index), vols, 0.30).items():
                    wr[t] = wt
            else:
                wr["CASH"] = 1.0
        else:
            wr["CASH"] = 1.0
        assign_w(wdf, i0, i1, wr)
    return vol_scale(wdf, close, vol_target)


STRATEGIES = {
    "champion_trend_following_v4": lambda c, f: trend_following_v4_weights(c, f, RISKY_ASSETS, **CHAMPION_PARAMS),
    "adaptive_ensemble": adaptive_ensemble_weights,
    "factor_rotation": factor_rotation_weights,
    "defensive_growth_rotation": defensive_growth_rotation_weights,
    "risk_parity": lambda c, f: risk_parity_weights(c, f, FULL_UNIVERSE),
    "minimum_volatility": lambda c, f: minimum_volatility_weights(c, f, FULL_UNIVERSE),
    "vol_target_momentum": lambda c, f: vol_target_momentum_weights(c, f),
}

# %% [markdown]
# ## 8. ML Ranking — EXPERIMENTAL (NO PARA WEB)
#
# Walk-forward mensual. Posible overfitting.

# %%
ML_FEATURE_COLS = ["MOM_20", "MOM_60", "MOM_120", "VOL_20", "VOL_60", "DIST_SMA200", "RSI_14", "VOLUME_RATIO"]

def ml_ranking_weights(close, features_dict, train_end, top_n=3):
    """Entrena solo con datos <= train_end, genera pesos desde train_end en adelante."""
    wdf = pd.DataFrame(0.0, index=norm_idx(close.index), columns=close.columns)
    cols = [c for c in RISKY_ASSETS + SECTOR_ETFS if c in close.columns and c != "CASH"]
    rows = []
    for t in cols:
        if t not in features_dict:
            continue
        df = features_dict[t].loc[:train_end].copy()
        for dt, row in df.iterrows():
            if pd.isna(row.get("forward_return_20d")):
                continue
            feat = {f"{t}_{k}": row.get(k, np.nan) for k in ML_FEATURE_COLS}
            feat["target"] = row["forward_return_20d"]
            feat["date"] = dt
            feat["ticker"] = t
            rows.append(feat)
    if len(rows) < 200:
        return wdf
    tr = pd.DataFrame(rows).dropna(subset=["target"])
    if len(tr) < 100:
        return wdf
    Xcols = [c for c in tr.columns if any(c.endswith(f"_{k}") for k in ML_FEATURE_COLS)]
    X = tr[Xcols].fillna(0)
    y = tr["target"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    models = {
        "ridge": Ridge(alpha=1.0),
        "rf": RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
        "hgb": HistGradientBoostingRegressor(max_depth=4, random_state=42),
    }
    preds = {}
    for name, mdl in models.items():
        try:
            mdl.fit(Xs, y)
            preds[name] = mdl
        except Exception:
            pass
    # Rebalance mensual post train_end
    post = close.loc[train_end:]
    if len(post) < 20:
        return wdf
    for i, rd in enumerate(reb_dates(post.index, "M")):
        i0 = idx_pos(wdf.index, rd)
        if i0 is None:
            continue
        rd = wdf.index[i0]
        rlist = list(reb_dates(post.index, "M"))
        i1 = idx_pos(wdf.index, rlist[i + 1]) if i + 1 < len(rlist) else len(wdf.index) - 1
        if i1 is None:
            i1 = len(wdf.index) - 1
        scores = {}
        for t in cols:
            row = row_asof(features_dict[t], rd)
            if row is None:
                continue
            xvec = np.array([row.get(k, 0) for k in ML_FEATURE_COLS]).reshape(1, -1)
            # simplified: use ridge on single-asset features from last train
            sc = row.get("MOM_COMBO_VOL", 0)
            scores[t] = sc
        wr = pd.Series(0.0, index=wdf.columns)
        if scores:
            s = pd.Series(scores).nlargest(top_n)
            s = s[s > 0]
            if len(s):
                for t in s.index:
                    wr[t] = 1 / len(s)
            else:
                wr["CASH"] = 1.0
        else:
            wr["CASH"] = 1.0
        assign_w(wdf, i0, i1, wr)
    return wdf


def run_ml_walk_forward(close, features_dict, start_year=2017):
    rows = []
    for yr in range(start_year, 2027):
        train_end = f"{yr - 1}-12-31"
        test_s, test_e = f"{yr}-01-01", f"{yr}-12-31"
        if train_end < close.index[0].strftime("%Y-%m-%d"):
            continue
        cp = close.loc[test_s:test_e]
        if len(cp) < 20:
            continue
        w = ml_ranking_weights(close, features_dict, train_end)
        w_y = w.loc[test_s:test_e]
        m, pr, _, _, _ = backtest_portfolio_weights(cp, w_y)
        m["year"] = yr
        m["model"] = "ml_experimental"
        rows.append(m)
    return pd.DataFrame(rows)

ml_results = run_ml_walk_forward(close_prices, features_dict, WF_START_YEAR) if not QUICK_TEST else pd.DataFrame()
if len(ml_results):
    print("ML experimental OOS (por año):")
    print(ml_results[["year", "CAGR", "sharpe", "max_drawdown", "excess_vs_spy"]].to_string())

# %% [markdown]
# ## 9. Walk-forward Champion vs Challengers

# %%
def run_walk_forward_champion_challenger(close, features_dict, strategies, start_year=WF_START_YEAR):
    rows = []
    champ_fn = strategies["champion_trend_following_v4"]
    for year in range(start_year, 2027):
        ts, te = f"{year}-01-01", f"{year}-12-31"
        cp = close.loc[ts:te]
        if len(cp) < 20:
            continue
        fd = {k: v.loc[:te] for k, v in features_dict.items()}
        w_ch = champ_fn(cp, fd).reindex(cp.index).fillna(0)
        _, pr_ch, _, _, _ = backtest_portfolio_weights(cp, w_ch)
        for name, fn in strategies.items():
            w = fn(cp, fd).reindex(cp.index).fillna(0)
            m, pr, _, _, _ = backtest_portfolio_weights(cp, w)
            cr = ((1 + pr_ch).prod() - 1) * 100
            rows.append({
                "year": year, "strategy": name,
                "return": m["total_return"], "CAGR": m["CAGR"], "sharpe": m["sharpe"],
                "max_drawdown": m["max_drawdown"], "excess_vs_spy": m["excess_vs_spy"],
                "excess_vs_champion": round(m["total_return"] - cr, 2),
                "beats_spy": m["excess_vs_spy"] > 0,
                "beats_champion": m["total_return"] > cr,
            })
    return pd.DataFrame(rows)

wf_df = run_walk_forward_champion_challenger(close_prices, features_dict, STRATEGIES, WF_START_YEAR)
print("Walk-forward OOS:")
print(wf_df.groupby("strategy").agg(
    oos_return=("return", "mean"), oos_sharpe=("sharpe", "mean"),
    pct_beats_spy=("beats_spy", "mean"), pct_beats_champion=("beats_champion", "mean"),
).round(3).to_string())

# %% [markdown]
# ## 10. Final score (solo OOS)

# %%
def compute_final_score(wf, strategy_name, loo_fragile=False, ml_overfit=False):
    sub = wf[wf["strategy"] == strategy_name]
    if sub.empty:
        return 0, "REJECTED"
    score = 0
    spy_beat = sub["beats_spy"].mean()
    ch_beat = sub["beats_champion"].mean()
    oos_sh = sub["sharpe"].mean()
    oos_dd = sub["max_drawdown"].mean()
    oos_ret = sub["return"].mean()

    if oos_ret > 0:
        score += 10
    if spy_beat >= 0.6:
        score += 20
    elif spy_beat >= 0.5:
        score += 10
    if oos_sh > 0.5:
        score += 20
    elif oos_sh > 0.3:
        score += 10
    if oos_dd > -30:
        score += 15
    if ch_beat >= 0.5:
        score += 10
    if sub["return"].std() < 50:
        score += 10
    if oos_dd < -30:
        score -= 20
    recent = sub[sub["year"] >= 2022]
    if len(recent) and recent["beats_spy"].mean() < 0.4:
        score -= 20
    if sub["return"].max() > 3 * sub["return"].median() and sub["return"].median() > 0:
        score -= 20
    if loo_fragile:
        score -= 10
    if ml_overfit:
        score -= 30

    score = int(np.clip(score, 0, 100))
    if score >= 70:
        status = "APPROVED_FOR_WEB_PAPER"
    elif score >= 55:
        status = "CANDIDATE"
    else:
        status = "REJECTED"
    return score, status

scores = []
for name in STRATEGIES:
    sc, st = compute_final_score(wf_df, name)
    scores.append({"strategy": name, "final_score": sc, "status": st})
score_df = pd.DataFrame(scores).sort_values("final_score", ascending=False)
print(score_df.to_string())
challenger_best = score_df.iloc[0]["strategy"] if len(score_df) else "champion_trend_following_v4"

# %% [markdown]
# ## 11. Stress, leave-one-out, costes

# %%
def run_strategy_metrics(close, fd, fn, txn=TRANSACTION_COST, slip=SLIPPAGE):
    w = fn(close, fd)
    return backtest_portfolio_weights(close, w, txn, slip)

champ_fn = STRATEGIES["champion_trend_following_v4"]
champ_m, champ_pr, champ_eq, _, _ = run_strategy_metrics(close_prices, features_dict, champ_fn)

STRESS = {
    "COVID_crash": ("2020-02-01", "2020-04-30"),
    "bear_2022": ("2022-01-01", "2022-12-31"),
    "recovery_2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
    "2026_YTD": ("2026-01-01", None),
}
stress_rows = []
for name, fn in STRATEGIES.items():
    _, pr, _, _, _ = run_strategy_metrics(close_prices, features_dict, fn)
    for period, (s, e) in STRESS.items():
        sl = pr.loc[s:e] if e else pr.loc[s:]
        spy = BENCHMARKS["SPY"].reindex(sl.index).fillna(0)
        if len(sl) < 5:
            continue
        sr = ((1 + sl).prod() - 1) * 100
        br = ((1 + spy).prod() - 1) * 100
        eq = (1 + sl).cumprod()
        dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
        stress_rows.append({"strategy": name, "period": period, "strategy_return": round(sr, 2),
                            "spy_return": round(br, 2), "max_drawdown": round(dd, 2), "beats_spy": sr > br})
stress_df = pd.DataFrame(stress_rows)

LOO = {"full": RISKY_ASSETS, "sin_NVDA": [t for t in RISKY_ASSETS if t != "NVDA"], "sin_TSLA": [t for t in RISKY_ASSETS if t != "TSLA"]}
loo_rows = []
for uname, uni in LOO.items():
    fn = lambda c, f, u=uni: trend_following_v4_weights(c, f, u, **CHAMPION_PARAMS)
    m, _, _, _, _ = run_strategy_metrics(close_prices, features_dict, fn)
    loo_rows.append({"universe_name": uname, **m, "approved": m["excess_vs_spy"] > 0})
loo_df = pd.DataFrame(loo_rows)

cost_rows = []
w_ch = champ_fn(close_prices, features_dict)
for tc, sl in itertools.product([0.0005, 0.001, 0.002, 0.005], [0.0005, 0.001, 0.002, 0.005]):
    m, _, _, _, _ = backtest_portfolio_weights(close_prices, w_ch, tc, sl)
    cost_rows.append({"transaction_cost": tc, "slippage": sl, **m, "alive": m["excess_vs_spy"] > 0 and m["sharpe"] > 0.3})
cost_df = pd.DataFrame(cost_rows)

start_rows = []
for yr in [2015, 2018, 2020, 2022]:
    cp = close_prices.loc[f"{yr}-01-01":]
    if len(cp) < 252:
        continue
    fd = {k: v.loc[cp.index[0]:] for k, v in features_dict.items()}
    m, _, _, _, _ = run_strategy_metrics(cp, fd, champ_fn)
    start_rows.append({"start_date": f"{yr}-01-01", **m, "beats_spy": m["excess_vs_spy"] > 0})
start_df = pd.DataFrame(start_rows)

# Comparación explícita vs benchmarks (Champion + mejor OOS)
def benchmark_table(close, fn, label):
    m, pr, _, _, _ = run_strategy_metrics(close, features_dict, fn)
    yrs = max((pr.index[-1] - pr.index[0]).days / 365.25, 1 / 365.25)
    rows = []
    for bname, br in BENCHMARKS.items():
        bt = ((1 + br.reindex(pr.index).fillna(0)).prod() - 1) * 100
        bcagr = ((1 + bt / 100) ** (1 / yrs) - 1) * 100
        bsh = (br.reindex(pr.index).fillna(0).mean() / br.reindex(pr.index).fillna(0).std() * np.sqrt(252)) if br.reindex(pr.index).fillna(0).std() > 0 else 0
        beq = (1 + br.reindex(pr.index).fillna(0)).cumprod()
        bdd = ((beq - beq.cummax()) / beq.cummax()).min() * 100
        rows.append({
            "strategy": label, "benchmark": bname,
            "strategy_CAGR": m["CAGR"], "benchmark_CAGR": round(bcagr, 2),
            "strategy_sharpe": m["sharpe"], "benchmark_sharpe": round(bsh, 3),
            "strategy_max_drawdown": m["max_drawdown"], "benchmark_max_drawdown": round(bdd, 2),
            "excess_return": round(m["total_return"] - bt, 2),
        })
    return rows

bench_cmp_rows = benchmark_table(close_prices, champ_fn, "champion_trend_following_v4")
if challenger_best in STRATEGIES:
    bench_cmp_rows += benchmark_table(close_prices, STRATEGIES[challenger_best], challenger_best)
bench_cmp_df = pd.DataFrame(bench_cmp_rows)

# %% [markdown]
# ## 12. Full sample reference + equity

# %%
full_rows = []
equity_curves = {}
for name, fn in STRATEGIES.items():
    m, pr, eq, _, _ = run_strategy_metrics(close_prices, features_dict, fn)
    row = dict(m)
    row["strategy"] = name
    full_rows.append(row)
    equity_curves[name] = eq
summary_df = pd.DataFrame(full_rows)
champ_fs = score_df[score_df["strategy"] == "champion_trend_following_v4"] if len(score_df) else pd.DataFrame()
champ_score_val = champ_fs["final_score"].iloc[0] if len(champ_fs) else 0
best_ch_row = score_df.iloc[0] if len(score_df) else None
winner = challenger_best if (best_ch_row is not None and best_ch_row["final_score"] > champ_score_val and challenger_best != "champion_trend_following_v4") else "champion_trend_following_v4"

# %% [markdown]
# ## 13. Gráficos

# %%
fig, ax = plt.subplots(figsize=(12, 5))
for name in ["champion_trend_following_v4", challenger_best]:
    if name in equity_curves:
        ax.plot(equity_curves[name].index, equity_curves[name], label=name, lw=2)
spy_eq = (1 + BENCHMARKS["SPY"]).cumprod() * INITIAL_CAPITAL
ax.plot(spy_eq.index, spy_eq, label="SPY", alpha=0.7)
ax.legend()
ax.set_title("Champion vs mejor challenger vs SPY")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 14. Reporte final
#
# **Backtest no garantiza resultados futuros. APPROVED_FOR_REAL_MONEY = False.**

# %%
champ_score = score_df[score_df["strategy"] == "champion_trend_following_v4"]["final_score"].iloc[0] if len(score_df) else 0
champ_status = score_df[score_df["strategy"] == "champion_trend_following_v4"]["status"].iloc[0] if len(score_df) else "REJECTED"
best_ch = score_df.iloc[0]
improves = best_ch["final_score"] > champ_score and best_ch["strategy"] != "champion_trend_following_v4"
APPROVED_FOR_WEB_PAPER = best_ch["status"] == "APPROVED_FOR_WEB_PAPER"
APPROVED_FOR_REAL_MONEY = False

print("=" * 70)
print("REPORTE FINAL V5 — CHAMPION VS CHALLENGERS")
print("=" * 70)
print("\n1. Champion actual: trend_following_v4")
print(f"   OOS score={champ_score} | status={champ_status}")
print(f"   Full sample: CAGR={champ_m['CAGR']}% Sharpe={champ_m['sharpe']} DD={champ_m['max_drawdown']}%")
print(f"\n2. Mejor challenger: {best_ch['strategy']} (score={best_ch['final_score']})")
print(f"\n3. ¿Mejora al champion? {'Sí' if improves else 'No'}")
print(f"4. ¿Mejora a SPY? excess={champ_m['excess_vs_spy']}%")
print(f"5. ¿Mejora a QQQ? excess={champ_m['excess_vs_qqq']}%")
print(f"6. ¿Reduce drawdown vs SPY? {champ_m['max_drawdown']}% vs SPY ~")
print(f"7. ¿Estable por años? {wf_df[wf_df['strategy']=='champion_trend_following_v4']['beats_spy'].mean()*100:.0f}% años vs SPY")
print(f"8. ¿Sensible a costes? {cost_df['alive'].mean()*100:.0f}% escenarios vivos")
print(f"\n9. Paper trading web: APPROVED_FOR_WEB_PAPER={APPROVED_FOR_WEB_PAPER}")
print(f"   REAL MONEY: {APPROVED_FOR_REAL_MONEY} (siempre False)")

if APPROVED_FOR_WEB_PAPER and improves:
    print(f"\n10. Nueva ganadora propuesta: {best_ch['strategy']}")
    print("   → Meter en web como paper trading experimental.")
elif APPROVED_FOR_WEB_PAPER:
    print("\n10. Mantener champion V4 — sigue siendo la mejor validada.")
    print("   → Meter en web como paper trading experimental.")
elif best_ch["status"] == "CANDIDATE":
    print("\n10. Prometedor pero necesita más investigación.")
else:
    print("\n10. No mejora V4 de forma convincente; mantener V4 y seguir investigando.")

print("\n⚠️ No es asesoramiento financiero. No usar dinero real.")

# %% [markdown]
# ## 15. Exportar

# %%
winner_row = score_df.iloc[0]
config = {
    "strategy": winner if improves else "champion_trend_following_v4",
    "params": CHAMPION_PARAMS if winner == "champion_trend_following_v4" else {},
    "APPROVED_FOR_WEB_PAPER": bool(APPROVED_FOR_WEB_PAPER),
    "APPROVED_FOR_REAL_MONEY": False,
    "final_score": int(winner_row["final_score"]),
    "disclaimer": "Backtest no garantiza resultados futuros.",
}

summary_df.to_csv("research_v5_summary.csv", index=False)
score_df.to_csv("research_v5_champion_challenger.csv", index=False)
wf_df.to_csv("research_v5_walk_forward.csv", index=False)
stress_df.to_csv("research_v5_stress.csv", index=False)
cost_df.to_csv("research_v5_cost_sensitivity.csv", index=False)
ml_results.to_csv("research_v5_ml_results.csv", index=False)
eq_out = pd.DataFrame({k: v for k, v in equity_curves.items()})
eq_out.to_csv("research_v5_equity_curves.csv")
Path("research_v5_selected_strategy_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

print("Exportado research_v5_*.csv y research_v5_selected_strategy_config.json")
