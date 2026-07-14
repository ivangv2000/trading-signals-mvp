# %% [markdown]
# # Trading Research V4 — Robustness Lab
#
# Comprueba si las estrategias aprobadas en V3 son **robustas** o frágiles (suerte, un solo activo, un solo periodo).
#
# **Aviso:** El backtest no garantiza resultados futuros. No es asesoramiento financiero.
# **APPROVED_FOR_REAL_MONEY = False** siempre en esta fase.

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

QUICK_TEST = False

RISKY_ASSETS = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN",
]
ETF_ONLY = [
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
    "TLT", "IEF", "SHY", "GLD",
]
MEGA_TECH = ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN"]

START_DATE = "2015-01-01"
END_DATE = None
TRANSACTION_COST = 0.001
SLIPPAGE = 0.001
INITIAL_CAPITAL = 10000
MARKET_TICKER = "SPY"
BENCHMARK_QQQ = "QQQ"
BENCHMARK_IEF = "IEF"

# Parámetros base de trend_following aprobada en V3
BASE_TF = dict(fast_ma=50, slow_ma=200, vol_target=0.15, max_asset_weight=0.30, rebalance_freq="W-FRI")

N_BOOTSTRAP = 200 if QUICK_TEST else 1000

if QUICK_TEST:
    START_DATE = "2020-01-01"
    N_BOOTSTRAP = 200
    print("⚡ QUICK_TEST activo")

print("V4 Robustness Lab — periodo base:", START_DATE)

# %% [markdown]
# ## 2. Cargar resultados V3

# %%
V3_FILES = {
    "approved": "research_v3_approved.csv",
    "walk_forward": "research_v3_walk_forward.csv",
    "portfolio": "research_v3_portfolio_results.csv",
    "equity": "research_v3_equity_curves.csv",
}

def load_csv(name):
    for p in [Path(name), Path("notebooks") / name, Path.cwd() / name]:
        if p.exists():
            df = pd.read_csv(p)
            print(f"✓ {name} ({len(df)} filas)")
            return df
    print(f"⚠️ No encontrado: {name}")
    return None

v3_approved = load_csv(V3_FILES["approved"])
v3_wf = load_csv(V3_FILES["walk_forward"])
v3_portfolio = load_csv(V3_FILES["portfolio"])
v3_equity = load_csv(V3_FILES["equity"])

print("\n=== RESUMEN V3 ===")
if v3_approved is not None and len(v3_approved):
    print("Estrategias aprobadas V3:")
    for _, r in v3_approved.iterrows():
        print(f"  • {r.get('strategy', '?')} | Sharpe={r.get('sharpe', '?')} | DD={r.get('max_drawdown', '?')}%")

if v3_portfolio is not None and len(v3_portfolio):
    p = v3_portfolio
    print(f"\nMejor Sharpe:  {p.loc[p['sharpe'].idxmax(), 'strategy']} ({p['sharpe'].max()})")
    print(f"Mejor CAGR:    {p.loc[p['CAGR'].idxmax(), 'strategy']} ({p['CAGR'].max()}%)")
    print(f"Mejor DD:      {p.loc[p['max_drawdown'].idxmax(), 'strategy']} ({p['max_drawdown'].max()}%)")
    print(f"Superan SPY:   {(p['excess_vs_spy'] > 0).sum()} / {len(p)}")
    print(f"Superan QQQ:   {(p['excess_vs_qqq'] > 0).sum()} / {len(p)}")
    print(f"Superan EW:    {(p['excess_vs_equal_weight'] > 0).sum()} / {len(p)}")

if v3_wf is not None and len(v3_wf):
    wf_fail = v3_wf.groupby("strategy").agg(pct_beats=("beats_spy", "mean"), years=("test_year", "count"))
    weak = wf_fail[wf_fail["pct_beats"] < 0.6]
    if len(weak):
        print("\nFallan walk-forward (<60% años vs SPY):")
        for strat, row in weak.iterrows():
            print(f"  • {strat}: {row['pct_beats']*100:.0f}% en {int(row['years'])} años")

# %% [markdown]
# ## 3. Motor: datos, features y backtest

# %%
def download_data(tickers, start, end=None):
    data, failed = {}, []
    for ticker in tqdm(sorted(set(tickers)), desc="Descargando"):
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
            if not df.empty:
                data[ticker.upper()] = df
            else:
                failed.append(ticker)
        except Exception as e:
            failed.append(f"{ticker} ({e})")
    if failed:
        print("Fallidos:", failed[:10])
    close = pd.DataFrame({t: d["Close"] for t, d in data.items()}).sort_index().ffill()
    close.index = pd.DatetimeIndex(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    for t, df in list(data.items()):
        df = df.copy()
        df.index = pd.DatetimeIndex(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        data[t] = df
    close["CASH"] = 1.0
    data["CASH"] = pd.DataFrame({"Close": 1.0}, index=close.index)
    return data, close


def add_features(df):
    d = df.copy()
    c = d["Close"]
    ret1 = c.pct_change(1)
    for n in [5, 20, 60, 120, 252]:
        d[f"RET_{n}D"] = c.pct_change(n)
    for n in [20, 50, 100, 150, 200]:
        d[f"SMA_{n}"] = c.rolling(n).mean()
    for span in [20, 50, 100]:
        d[f"EMA_{span}"] = c.ewm(span=span, adjust=False).mean()
    d["VOL_60"] = ret1.rolling(60).std() * np.sqrt(252)
    return d


def build_features_dict(data):
    return {t: add_features(df) for t, df in data.items()}


def normalize_dt_index(index):
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize(None) if idx.tz else idx


def align_date_to_index(index, date):
    index = normalize_dt_index(index)
    date = pd.Timestamp(date)
    if date.tz:
        date = date.tz_localize(None)
    if date in index:
        return date
    loc = index.get_indexer([date], method="pad")[0]
    if loc < 0:
        loc = index.get_indexer([date], method="nearest")[0]
    return index[loc] if loc >= 0 else None


def index_position(index, date):
    a = align_date_to_index(index, date)
    return index.get_loc(a) if a is not None else None


def get_row_asof(df, date):
    a = align_date_to_index(df.index, date)
    if a is None:
        return None
    return df.loc[a] if a in df.index else df.iloc[df.index.get_indexer([a], method="pad")[0]]


def get_rebalance_dates(index, freq="W-FRI"):
    index = normalize_dt_index(index)
    s = pd.Series(np.arange(len(index)), index=index)
    return pd.DatetimeIndex([grp.index[-1] for _, grp in s.groupby(pd.Grouper(freq=freq)) if len(grp)])


def assign_weights_window(weights, idx, end_idx, w_row):
    end_idx = min(int(end_idx), len(weights) - 1)
    vals = w_row.reindex(weights.columns).fillna(0).values
    for j in range(int(idx) + 1, end_idx + 1):
        weights.iloc[j] = vals


def inverse_vol_weights(tickers, vols, max_weight=0.40):
    vols = pd.Series(vols).replace(0, np.nan).dropna()
    tickers = [t for t in tickers if t in vols.index]
    if not tickers:
        return pd.Series(dtype=float)
    w = (1.0 / vols[tickers])
    w = w / w.sum()
    for _ in range(5):
        w = (w / w.sum()).clip(upper=max_weight)
    return w / w.sum()


def apply_vol_target_scale(weights_df, close_prices, vol_target=0.15, lookback=20):
    rets = close_prices.pct_change().fillna(0)
    port_ret = (weights_df.shift(1).fillna(0) * rets).sum(axis=1)
    realized = port_ret.rolling(lookback).std() * np.sqrt(252)
    scale = (vol_target / realized.replace(0, np.nan)).clip(0, 1).shift(1).fillna(1)
    return weights_df.mul(scale, axis=0)


def get_fast_ma_value(row, fast_ma):
    if fast_ma == 50:
        return row.get("EMA_50", np.nan)
    if fast_ma == 20:
        return row.get("EMA_20", row.get("SMA_20", np.nan))
    if fast_ma == 100:
        return row.get("EMA_100", row.get("SMA_100", np.nan))
    return row.get(f"EMA_{fast_ma}", row.get(f"SMA_{fast_ma}", np.nan))


def trend_following_portfolio(close_prices, features_dict, universe, fast_ma=50, slow_ma=200,
                              rebalance_freq="W-FRI", vol_target=0.15, max_asset_weight=0.30, defensive_asset="SHY"):
    cols = [c for c in universe if c in close_prices.columns and c != "CASH"]
    weights = pd.DataFrame(0.0, index=normalize_dt_index(close_prices.index), columns=close_prices.columns)
    reb_dates = list(get_rebalance_dates(weights.index, rebalance_freq))
    slow_col = f"SMA_{slow_ma}"

    for i, reb in enumerate(reb_dates):
        idx = index_position(weights.index, reb)
        if idx is None:
            continue
        reb = weights.index[idx]
        end_idx = index_position(weights.index, reb_dates[i + 1]) if i + 1 < len(reb_dates) else len(weights.index) - 1
        if end_idx is None:
            end_idx = len(weights.index) - 1

        eligible, vols = [], {}
        for t in cols:
            if t not in features_dict:
                continue
            row = get_row_asof(features_dict[t], reb)
            if row is None:
                continue
            sma_slow = row.get(slow_col, np.nan)
            ema_fast = get_fast_ma_value(row, fast_ma)
            px = row.get("Close", np.nan)
            if pd.notna(sma_slow) and pd.notna(ema_fast) and pd.notna(px) and px > sma_slow and ema_fast > sma_slow:
                eligible.append(t)
                vols[t] = row.get("VOL_60", np.nan)

        w = pd.Series(0.0, index=weights.columns)
        if eligible:
            for t, wt in inverse_vol_weights(eligible, vols, max_asset_weight).items():
                w[t] = wt
        else:
            w[defensive_asset if defensive_asset in weights.columns else "CASH"] = 1.0
        assign_weights_window(weights, idx, end_idx, w)

    return apply_vol_target_scale(weights, close_prices, vol_target)


def backtest_portfolio_weights(close_prices, target_weights, transaction_cost=TRANSACTION_COST, slippage=SLIPPAGE):
    rets = close_prices.pct_change().fillna(0)
    cols = [c for c in target_weights.columns if c in rets.columns]
    w = target_weights[cols].reindex(close_prices.index).fillna(0)
    w_exec = w.shift(1).fillna(0)
    turnover = w.diff().abs().sum(axis=1).fillna(0)
    port_ret = (w_exec * rets[cols]).sum(axis=1) - turnover * (transaction_cost + slippage)
    equity = (1 + port_ret).cumprod() * INITIAL_CAPITAL
    dd = (equity - equity.cummax()) / equity.cummax()
    years = max((port_ret.index[-1] - port_ret.index[0]).days / 365.25, 1 / 365.25)
    total_return = (equity.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    cagr = ((equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    sharpe = (port_ret.mean() / port_ret.std() * np.sqrt(252)) if port_ret.std() > 0 else 0
    max_dd = dd.min() * 100
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    spy_r = rets[MARKET_TICKER] if MARKET_TICKER in rets.columns else pd.Series(0.0, index=rets.index)
    qqq_r = rets[BENCHMARK_QQQ] if BENCHMARK_QQQ in rets.columns else spy_r
    ew_cols = [c for c in cols if c != "CASH"]
    ew_r = rets[ew_cols].mean(axis=1) if ew_cols else spy_r

    spy_cagr = (((1 + spy_r).prod()) ** (1 / years) - 1) * 100
    spy_sharpe = (spy_r.mean() / spy_r.std() * np.sqrt(252)) if spy_r.std() > 0 else 0

    return {
        "total_return": round(total_return, 2),
        "CAGR": round(cagr, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
        "calmar": round(calmar, 3),
        "excess_vs_spy": round(total_return - ((1 + spy_r).prod() - 1) * 100, 2),
        "excess_vs_qqq": round(total_return - ((1 + qqq_r).prod() - 1) * 100, 2),
        "excess_vs_equal_weight": round(total_return - ((1 + ew_r).prod() - 1) * 100, 2),
        "spy_CAGR": round(spy_cagr, 2),
        "spy_sharpe": round(spy_sharpe, 3),
        "turnover_avg": round(turnover.mean(), 4),
    }, port_ret, equity, dd


def years_beating_spy(port_ret, close_prices):
    spy_r = close_prices[MARKET_TICKER].pct_change().fillna(0)
    years = sorted(set(port_ret.index.year))
    wins, total = 0, 0
    for y in years:
        m = port_ret.index.year == y
        if m.sum() < 20:
            continue
        sr = ((1 + port_ret.loc[m]).prod() - 1) * 100
        br = ((1 + spy_r.loc[m]).prod() - 1) * 100
        total += 1
        if sr > br:
            wins += 1
    return wins, total, (wins / total if total else 0)


def run_trend_following(close_prices, features_dict, universe, params, txn=TRANSACTION_COST, slip=SLIPPAGE):
    w = trend_following_portfolio(close_prices, features_dict, universe, **params)
    m, port_ret, _, _ = backtest_portfolio_weights(close_prices, w, txn, slip)
    yw, yt, ypct = years_beating_spy(port_ret, close_prices)
    m["years_beating_spy"] = round(ypct * 100, 1)
    m["years_beating_spy_n"] = f"{yw}/{yt}"
    return m, port_ret

# Descarga base
all_tickers = sorted(set(RISKY_ASSETS + ETF_ONLY + [MARKET_TICKER, BENCHMARK_QQQ, BENCHMARK_IEF]))
data, close_prices = download_data(all_tickers, START_DATE, END_DATE)
features_dict = build_features_dict(data)
print("Datos listos:", close_prices.shape)

# %% [markdown]
# ## 4. Sensibilidad de parámetros — Trend Following

# %%
if QUICK_TEST:
    FAST_MAS = [50]
    SLOW_MAS = [200]
    VOL_TARGETS = [0.15]
    MAX_WEIGHTS = [0.30]
    REB_FREQS = ["W-FRI"]
else:
    FAST_MAS = [20, 50, 100]
    SLOW_MAS = [100, 150, 200]
    VOL_TARGETS = [0.08, 0.10, 0.12, 0.15]
    MAX_WEIGHTS = [0.20, 0.30, 0.40]
    REB_FREQS = ["W-FRI", "M"]

param_rows = []
for fast_ma, slow_ma, vol_target, max_w, reb in tqdm(
    list(itertools.product(FAST_MAS, SLOW_MAS, VOL_TARGETS, MAX_WEIGHTS, REB_FREQS)),
    desc="Param grid",
):
    if fast_ma >= slow_ma:
        continue
    p = dict(fast_ma=fast_ma, slow_ma=slow_ma, vol_target=vol_target, max_asset_weight=max_w, rebalance_freq=reb)
    try:
        m, _ = run_trend_following(close_prices, features_dict, RISKY_ASSETS, p)
        param_rows.append({"strategy": "trend_following", **p, **m})
    except Exception as e:
        param_rows.append({"strategy": "trend_following", **p, "error": str(e)})

param_df = pd.DataFrame(param_rows)
if len(param_df) and "spy_CAGR" in param_df.columns:
    param_df["approved_robust"] = False
    for idx, row in param_df.iterrows():
        if pd.isna(row.get("CAGR")):
            param_df.at[idx, "approved_robust"] = False
            continue
        param_df.at[idx, "approved_robust"] = (
            row["CAGR"] > row["spy_CAGR"]
            and row["sharpe"] > row["spy_sharpe"]
            and row["max_drawdown"] > -30
            and row.get("years_beating_spy", 0) >= 60
            and row["excess_vs_spy"] > 0
        )
    n_pass = int(param_df["approved_robust"].sum())
    if n_pass > 0 and n_pass < 3:
        param_df["approved_robust"] = False
        print(f"⚠️ Solo {n_pass} combo(s) pasan — no es robusto (necesita >=3)")
    print(f"Combinaciones approved_robust: {n_pass} / {len(param_df)}")
    print(param_df.sort_values("sharpe", ascending=False).head(10)[
        ["fast_ma", "slow_ma", "vol_target", "CAGR", "sharpe", "max_drawdown", "approved_robust"]
    ].to_string())

# %% [markdown]
# ## 5. Leave-one-out (dependencia de activos)

# %%
LOO_UNIVERSES = {
    "full": RISKY_ASSETS,
    "sin_NVDA": [t for t in RISKY_ASSETS if t != "NVDA"],
    "sin_TSLA": [t for t in RISKY_ASSETS if t != "TSLA"],
    "sin_AMD": [t for t in RISKY_ASSETS if t != "AMD"],
    "sin_META": [t for t in RISKY_ASSETS if t != "META"],
    "sin_AAPL": [t for t in RISKY_ASSETS if t != "AAPL"],
    "solo_ETFs": ETF_ONLY,
    "solo_mega_tech": MEGA_TECH + ["SPY", "QQQ"],
    "sin_mega_tech": [t for t in RISKY_ASSETS if t not in MEGA_TECH],
}

loo_rows = []
base_m, _ = run_trend_following(close_prices, features_dict, RISKY_ASSETS, BASE_TF)
base_cagr = base_m["CAGR"]

for name, uni in LOO_UNIVERSES.items():
    uni = [t for t in uni if t in close_prices.columns]
    if len(uni) < 2:
        continue
    m, _ = run_trend_following(close_prices, features_dict, uni, BASE_TF)
    approved = m["excess_vs_spy"] > 0 and m["sharpe"] > 0.5 and m["max_drawdown"] > -30
    note = ""
    if name in ("sin_NVDA", "sin_TSLA") and m["CAGR"] < base_cagr * 0.5:
        note = "dependencia excesiva de un activo"
    loo_rows.append({"universe_name": name, **m, "approved": approved, "note": note})

loo_df = pd.DataFrame(loo_rows)
print(loo_df[["universe_name", "CAGR", "sharpe", "max_drawdown", "excess_vs_spy", "note"]].to_string())

# %% [markdown]
# ## 6. Robustez por fecha de inicio

# %%
START_YEARS = [2015, 2017, 2019, 2020, 2021, 2022] if not QUICK_TEST else [2019, 2021, 2022]
start_rows = []

for yr in START_YEARS:
    sd = f"{yr}-01-01"
    cp = close_prices.loc[sd:]
    if len(cp) < 252:
        continue
    sub_data = {t: data[t].loc[cp.index[0]:] for t in data if t in cp.columns or t == "CASH"}
    fd = build_features_dict(sub_data)
    m, _ = run_trend_following(cp, fd, RISKY_ASSETS, BASE_TF)
    start_rows.append({
        "start_date": sd,
        **m,
        "beats_spy": m["excess_vs_spy"] > 0,
    })

start_df = pd.DataFrame(start_rows)
danger = start_df[start_df["start_date"] != f"{START_YEARS[0]}-01-01"]
only_2015 = (start_df.loc[start_df["start_date"] == f"{START_YEARS[0]}-01-01", "beats_spy"].iloc[0]
             if len(start_df) and start_df.iloc[0]["start_date"].startswith(str(START_YEARS[0])) else False)
if len(danger) and only_2015 and danger["beats_spy"].sum() == 0:
    print("⚠️ Peligroso: solo funciona empezando en", START_YEARS[0])
print(start_df.to_string())

# %% [markdown]
# ## 7. Stress test por periodos

# %%
STRESS_PERIODS = {
    "COVID_crash_2020": ("2020-02-01", "2020-04-30"),
    "bull_2020_2021": ("2020-04-01", "2021-12-31"),
    "bear_2022": ("2022-01-01", "2022-12-31"),
    "recovery_2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
    "2026_YTD": ("2026-01-01", None),
}

_, port_ret_full = run_trend_following(close_prices, features_dict, RISKY_ASSETS, BASE_TF)
stress_rows = []

for period, (s, e) in STRESS_PERIODS.items():
    pr = port_ret_full.loc[s:e] if e else port_ret_full.loc[s:]
    spy = close_prices[MARKET_TICKER].pct_change().fillna(0).loc[pr.index]
    qqq = close_prices[BENCHMARK_QQQ].pct_change().fillna(0).reindex(pr.index).fillna(0)
    if len(pr) < 5:
        continue
    sr = ((1 + pr).prod() - 1) * 100
    spy_r = ((1 + spy).prod() - 1) * 100
    qqq_r = ((1 + qqq).prod() - 1) * 100
    eq = (1 + pr).cumprod()
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    stress_rows.append({
        "period": period,
        "strategy_return": round(sr, 2),
        "spy_return": round(spy_r, 2),
        "qqq_return": round(qqq_r, 2),
        "max_drawdown": round(dd, 2),
        "beats_spy": sr > spy_r,
    })

stress_df = pd.DataFrame(stress_rows)
print(stress_df.to_string())

# %% [markdown]
# ## 8. Bootstrap / Monte Carlo (bloques mensuales)

# %%
def block_bootstrap(port_ret, n_sim=N_BOOTSTRAP, block="M"):
    monthly = port_ret.resample(block).apply(lambda x: (1 + x).prod() - 1).dropna()
    if len(monthly) < 6:
        return None
    blocks = monthly.values
    n_blocks = len(blocks)
    results = []
    rng = np.random.default_rng(42)
    for _ in range(n_sim):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        sim = (1 + blocks[idx]).prod() - 1
        path = np.cumprod(1 + blocks[idx])
        dd = (path / np.maximum.accumulate(path) - 1).min()
        daily_equiv = blocks[idx].mean()
        vol = blocks[idx].std()
        sharpe = (daily_equiv / vol * np.sqrt(12)) if vol > 0 else 0
        years = n_blocks / 12
        cagr = ((1 + sim) ** (1 / years) - 1) * 100 if years > 0 else 0
        results.append({"total_return": sim * 100, "CAGR": cagr, "max_drawdown": dd * 100, "sharpe": sharpe})
    return pd.DataFrame(results)

boot_df = block_bootstrap(port_ret_full, N_BOOTSTRAP)
if boot_df is not None:
    pct = boot_df.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print("Percentiles bootstrap (bloques mensuales):")
    print(pct.round(2).to_string())

# %% [markdown]
# ## 9. Sensibilidad a costes y slippage

# %%
COST_GRID = [0.0005, 0.001, 0.002, 0.005] if not QUICK_TEST else [0.001, 0.002]
SLIP_GRID = [0.0005, 0.001, 0.002, 0.005] if not QUICK_TEST else [0.001, 0.002]

cost_rows = []
w_base = trend_following_portfolio(close_prices, features_dict, RISKY_ASSETS, **BASE_TF)
for tc, sl in itertools.product(COST_GRID, SLIP_GRID):
    m, _, _, _ = backtest_portfolio_weights(close_prices, w_base, tc, sl)
    alive = m["excess_vs_spy"] > 0 and m["sharpe"] > 0.3
    cost_rows.append({
        "transaction_cost": tc,
        "slippage": sl,
        "total_cost_per_turnover": tc + sl,
        **m,
        "strategy_alive": alive,
    })

cost_df = pd.DataFrame(cost_rows)
print(cost_df[["transaction_cost", "slippage", "CAGR", "sharpe", "excess_vs_spy", "strategy_alive"]].to_string())

# %% [markdown]
# ## 10. Comparación contra benchmarks

# %%
def benchmark_metrics(close_prices, label, ret_series=None):
    if ret_series is None:
        ret_series = close_prices[label].pct_change().fillna(0)
    years = max((ret_series.index[-1] - ret_series.index[0]).days / 365.25, 1 / 365.25)
    total = ((1 + ret_series).prod() - 1) * 100
    cagr = ((1 + total / 100) ** (1 / years) - 1) * 100
    sharpe = (ret_series.mean() / ret_series.std() * np.sqrt(252)) if ret_series.std() > 0 else 0
    eq = (1 + ret_series).cumprod()
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    return {"CAGR": round(cagr, 2), "sharpe": round(sharpe, 3), "max_drawdown": round(dd, 2), "total_return": round(total, 2)}


strat_m, strat_ret, _, _ = backtest_portfolio_weights(
    close_prices, trend_following_portfolio(close_prices, features_dict, RISKY_ASSETS, **BASE_TF)
)
spy_m = benchmark_metrics(close_prices, MARKET_TICKER)
qqq_m = benchmark_metrics(close_prices, BENCHMARK_QQQ)
ew_r = close_prices[[c for c in RISKY_ASSETS if c in close_prices.columns]].pct_change().mean(axis=1).fillna(0)
ew_m = benchmark_metrics(close_prices, "EW", ew_r)
mix_r = 0.6 * close_prices[MARKET_TICKER].pct_change().fillna(0) + 0.4 * close_prices[BENCHMARK_IEF].pct_change().reindex(close_prices.index).fillna(0)
mix_m = benchmark_metrics(close_prices, "60/40", mix_r)

bench_rows = []
for bname, bm in [("SPY", spy_m), ("QQQ", qqq_m), ("Equal_Weight", ew_m), ("60_40_SPY_IEF", mix_m)]:
    bench_rows.append({
        "strategy": "trend_following",
        "benchmark": bname,
        "strategy_CAGR": strat_m["CAGR"],
        "benchmark_CAGR": bm["CAGR"],
        "strategy_sharpe": strat_m["sharpe"],
        "benchmark_sharpe": bm["sharpe"],
        "strategy_max_drawdown": strat_m["max_drawdown"],
        "benchmark_max_drawdown": bm["max_drawdown"],
        "excess_return": round(strat_m["total_return"] - bm["total_return"], 2),
    })

bench_df = pd.DataFrame(bench_rows)
print(bench_df.to_string())

# %% [markdown]
# ## 11. Gráficos

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

eq = (1 + strat_ret).cumprod() * INITIAL_CAPITAL
spy_eq = (1 + close_prices[MARKET_TICKER].pct_change().fillna(0)).cumprod() * INITIAL_CAPITAL
axes[0, 0].plot(eq.index, eq, label="Trend Following", lw=2)
axes[0, 0].plot(spy_eq.index, spy_eq, label="SPY", alpha=0.8)
axes[0, 0].set_title("Equity vs SPY")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

if boot_df is not None:
    axes[0, 1].hist(boot_df["max_drawdown"], bins=40, color="salmon", alpha=0.8)
    axes[0, 1].axvline(boot_df["max_drawdown"].quantile(0.05), color="red", ls="--", label="p5")
    axes[0, 1].set_title("Bootstrap: distribución max drawdown")
    axes[0, 1].legend()

if len(param_df):
    axes[1, 0].scatter(param_df["max_drawdown"], param_df["sharpe"], c=param_df["CAGR"], cmap="RdYlGn", alpha=0.6)
    axes[1, 0].set_xlabel("max_drawdown %")
    axes[1, 0].set_ylabel("Sharpe")
    axes[1, 0].set_title("Sensibilidad parámetros")

if len(loo_df):
    axes[1, 1].barh(loo_df["universe_name"], loo_df["CAGR"], color="steelblue")
    axes[1, 1].set_title("Leave-one-out: CAGR")
    axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 12. Reporte final
#
# **Backtest no garantiza resultados futuros.**

# %%
def safe_pct(series, cond):
    return series[cond].mean() * 100 if len(series) else 0

n_param_robust = int(param_df["approved_robust"].sum()) if len(param_df) and "approved_robust" in param_df.columns else 0
loo_nvda_fragile = any(loo_df["note"].str.contains("dependencia", na=False)) if len(loo_df) else False
start_ok = start_df["beats_spy"].mean() >= 0.5 if len(start_df) else False
cost_ok = cost_df["strategy_alive"].mean() >= 0.5 if len(cost_df) else False
dd_vs_spy = strat_m["max_drawdown"] > spy_m["max_drawdown"]
stress_ok = stress_df["beats_spy"].mean() >= 0.5 if len(stress_df) else False
boot_dd_p5 = boot_df["max_drawdown"].quantile(0.05) if boot_df is not None else -50

checks = {
    "param_robust": n_param_robust >= 3,
    "not_nvda_dependent": not loo_nvda_fragile,
    "start_dates_ok": start_ok,
    "costs_ok": cost_ok,
    "dd_vs_spy": dd_vs_spy,
    "stress_ok": stress_ok,
    "excess_spy": strat_m["excess_vs_spy"] > 0,
    "sharpe_ok": strat_m["sharpe"] > spy_m["sharpe"],
}

APPROVED_FOR_WEB_PAPER = sum(checks.values()) >= 5 and checks["excess_spy"] and checks["sharpe_ok"]
APPROVED_FOR_REAL_MONEY = False

print("=" * 70)
print("REPORTE FINAL V4 — ROBUSTEZ")
print("Backtest no garantiza resultados futuros.")
print("=" * 70)

print("\nA) ¿Trend following sigue aprobada?")
print(f"   excess_vs_spy={strat_m['excess_vs_spy']}% | Sharpe={strat_m['sharpe']} | DD={strat_m['max_drawdown']}%")
print(f"   → {'Sí en muestra completa' if checks['excess_spy'] else 'No'}")

print("\nB) ¿Robusta o frágil?")
print(f"   Combinaciones approved_robust: {n_param_robust}")
print(f"   → {'Robusta' if checks['param_robust'] else 'Frágil / depende de params exactos'}")

print("\nC) ¿Depende de NVDA/TSLA?")
if loo_nvda_fragile:
    print("   ⚠️ Sí — dependencia excesiva detectada al excluir NVDA o TSLA")
else:
    print("   No detectada dependencia crítica en leave-one-out")

print("\nD) ¿Funciona con otros años de inicio?")
print(f"   Años inicio que baten SPY: {start_df['beats_spy'].sum()}/{len(start_df)}" if len(start_df) else "   N/A")

print("\nE) ¿Sobrevive a costes altos?")
print(f"   Escenarios vivos: {cost_df['strategy_alive'].sum()}/{len(cost_df)}" if len(cost_df) else "   N/A")

print("\nF) ¿Reduce drawdown vs SPY?")
print(f"   Estrategia DD={strat_m['max_drawdown']}% vs SPY DD={spy_m['max_drawdown']}%")
print(f"   → {'Sí' if dd_vs_spy else 'No'}")

print("\nG) ¿Pasar a web como PAPER TRADING?")
print(f"\n   APPROVED_FOR_WEB_PAPER = {APPROVED_FOR_WEB_PAPER}")
print(f"   APPROVED_FOR_REAL_MONEY = {APPROVED_FOR_REAL_MONEY}  (siempre False)")

if APPROVED_FOR_WEB_PAPER:
    print("\n   Podemos integrarla en la web como estrategia experimental de portfolio,")
    print("   NO como recomendación financiera ni dinero real.")
else:
    print("\n   No integrar todavía. Seguir investigando.")

print("\n⚠️ Investigación educativa. No usar para dinero real.")

# %% [markdown]
# ## 13. Exportar CSV

# %%
summary = pd.DataFrame([{
    "strategy": "trend_following",
    **strat_m,
    "APPROVED_FOR_WEB_PAPER": APPROVED_FOR_WEB_PAPER,
    "APPROVED_FOR_REAL_MONEY": APPROVED_FOR_REAL_MONEY,
    **{f"check_{k}": v for k, v in checks.items()},
    "boot_dd_p5": round(boot_dd_p5, 2) if boot_df is not None else None,
}])

summary.to_csv("research_v4_robustness_summary.csv", index=False)
param_df.to_csv("research_v4_parameter_sensitivity.csv", index=False)
loo_df.to_csv("research_v4_leave_one_out.csv", index=False)
start_df.to_csv("research_v4_start_date.csv", index=False)
stress_df.to_csv("research_v4_stress_periods.csv", index=False)
cost_df.to_csv("research_v4_cost_sensitivity.csv", index=False)

print("Exportado:")
for f in ["research_v4_robustness_summary.csv", "research_v4_parameter_sensitivity.csv",
          "research_v4_leave_one_out.csv", "research_v4_start_date.csv",
          "research_v4_stress_periods.csv", "research_v4_cost_sensitivity.csv"]:
    print(f"  {f}")
