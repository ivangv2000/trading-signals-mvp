# %% [markdown]
# # Trading Research V2 — Validación Profesional
#
# Fase 2 de investigación cuantitativa. Carga resultados de la fase 1, prueba estrategias
# más robustas con validación train/test y benchmarks múltiples.
#
# **Aviso:** El backtest no garantiza resultados futuros. Esto no es asesoramiento financiero.

# %%
# Instalación (ejecutar primero en Colab)
!pip install yfinance pandas numpy matplotlib plotly tqdm scikit-learn -q

# %% [markdown]
# ## 1. Imports y configuración

# %%
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tqdm.auto import tqdm

# --- Configuración editable ---
UNIVERSE_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN", "QQQ"]
ETF_TICKERS = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "IWM", "TLT", "GLD"]
MARKET_TICKER = "SPY"

START_DATE = "2018-01-01"
END_DATE = None

TRAIN_START = "2018-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"

TRANSACTION_COST = 0.001
SLIPPAGE = 0.001
COST_PER_SIDE = TRANSACTION_COST + SLIPPAGE

TOP_N_CS_MOM = 3          # Cross-sectional momentum: top N activos
CS_HOLD_DAYS = 5            # Mantener 5 días tras rebalanceo semanal
ETF_TOP_N = 3               # Sector rotation: top 3 ETFs
MR_MAX_HOLD = 3             # Mean reversion pro: máximo 3 días
TARGET_VOL_ANNUAL = 0.12    # Volatility target momentum

V1_RESULTS_FILE = "research_results.csv"  # Subir a Colab si no está

print("Configuración V2 cargada.")
print(f"Universo: {len(UNIVERSE_TICKERS)} tickers | ETFs: {len(ETF_TICKERS)}")
print(f"Train: {TRAIN_START} → {TRAIN_END} | Test: {TEST_START} → hoy")

# %% [markdown]
# ## 2. Cargar resultados de la investigación anterior (Fase 1)

# %%
def load_v1_results(filename=V1_RESULTS_FILE):
    """Carga research_results.csv desde varias rutas posibles."""
    candidates = [Path(filename), Path("notebooks") / filename, Path.cwd() / filename]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            print(f"Cargado: {p} ({len(df)} filas)")
            return df
    print(f"⚠️ No se encontró {filename}. Sube el archivo a Colab o ejecútalo en la misma carpeta.")
    return None

v1 = load_v1_results()

if v1 is not None:
    v1["approved_basic"] = (
        (v1["excess_return"] > 0)
        & (v1["profit_factor"] > 1.2)
        & (v1["sharpe"] > 0.5)
        & (v1["max_drawdown"] > -30)
        & (v1["num_trades"] >= 30)
    )

    n = len(v1)
    beats_bm = (v1["excess_return"] > 0).sum()
    pf_ok = (v1["profit_factor"] > 1).sum()
    sharpe_ok = (v1["sharpe"] > 0.5).sum()
    dd_ok = (v1["max_drawdown"] > -25).sum()
    approved = v1["approved_basic"].sum()

    print("\n=== RESUMEN FASE 1 ===")
    print(f"Combinaciones totales:        {n}")
    print(f"Superan comprar y mantener:   {beats_bm} ({beats_bm/n*100:.1f}%)")
    print(f"profit_factor > 1:            {pf_ok}")
    print(f"sharpe > 0.5:                 {sharpe_ok}")
    print(f"max_drawdown > -25%:          {dd_ok}")
    print(f"Aprobadas (approved_basic):   {approved}")

    if approved == 0:
        print("\n⛔ No hay estrategias aprobadas en la investigación actual.")
else:
    print("Continuando sin datos de Fase 1 (se puede ejecutar igualmente).")

# %% [markdown]
# ## 3. Problema detectado en Fase 1
#
# Las estrategias actuales hacen muchas operaciones cortas y pasan mucho tiempo fuera del mercado.
# En un periodo donde acciones como NVDA, AMD o TSLA subieron muchísimo, las estrategias no
# capturaron suficiente tendencia. Por eso pueden tener operaciones positivas pero aun así perder
# contra comprar y mantener.
#
# **Objetivo V2:** estrategias con más exposición a tendencia, rotación sistemática y validación
# out-of-sample estricta antes de considerar usarlas en la web.

# %% [markdown]
# ## 4. Utilidades: datos, features y métricas

# %%
def download_data(tickers, start, end=None):
    data, failed = {}, []
    for ticker in tqdm(sorted(set(tickers)), desc="Descargando"):
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
        print("Tickers fallidos:", failed)
    return data


def _rsi(close, window):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    rs = gain.rolling(window).mean() / loss.rolling(window).mean()
    return 100 - (100 / (1 + rs))


def add_features(df):
    d = df.copy()
    c = d["Close"]
    d["SMA_5"] = c.rolling(5).mean()
    d["SMA_50"] = c.rolling(50).mean()
    d["SMA_100"] = c.rolling(100).mean()
    d["SMA_200"] = c.rolling(200).mean()
    d["EMA_21"] = c.ewm(span=21, adjust=False).mean()
    d["RET_1D"] = c.pct_change(1)
    d["RET_20D"] = c.pct_change(20)
    d["RET_60D"] = c.pct_change(60)
    d["RET_120D"] = c.pct_change(120)
    d["VOL_20"] = d["RET_1D"].rolling(20).std()
    d["RSI_2"] = _rsi(c, 2)
    if "High" in d.columns and "Low" in d.columns:
        hl = (d["High"] - d["Low"]).replace(0, np.nan)
        d["IBS"] = (c - d["Low"]) / hl
    return d


def panel_from_dict(data_dict, tickers):
    cols = {}
    for t in tickers:
        if t in data_dict:
            cols[t] = data_dict[t]["Close"]
    panel = pd.DataFrame(cols).sort_index().dropna(how="all")
    return panel


def compute_metrics(strategy_ret, benchmarks, exposure=None, num_trades=0, trade_returns=None):
    """Métricas de una serie de retornos diarios vs varios benchmarks."""
    sr = strategy_ret.fillna(0)
    total_return = ((1 + sr).prod() - 1) * 100
    years = max((sr.index[-1] - sr.index[0]).days / 365.25, 1 / 365.25) if len(sr) > 1 else 1 / 365.25
    cagr = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if total_return > -100 else -100
    ann_vol = sr.std() * np.sqrt(252) * 100
    sharpe = (sr.mean() / sr.std() * np.sqrt(252)) if sr.std() > 0 else 0
    equity = (1 + sr).cumprod()
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    max_dd = dd.min() * 100
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    exp_pct = (exposure.mean() * 100) if exposure is not None else np.nan
    ret_per_exp = (total_return / exp_pct) if exposure is not None and exp_pct > 0 else np.nan

    tr = trade_returns if trade_returns else []
    wins = [r for r in tr if r > 0]
    losses = [r for r in tr if r <= 0]
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else (np.inf if wins else 0)
    win_rate = (len(wins) / len(tr) * 100) if tr else 0

    out = {
        "total_return": round(total_return, 2),
        "CAGR": round(cagr, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
        "calmar": round(calmar, 3),
        "num_trades": int(num_trades),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(pf, 3) if np.isfinite(pf) else 999,
        "exposure_pct": round(exp_pct, 2) if not np.isnan(exp_pct) else np.nan,
        "return_per_exposure": round(ret_per_exp, 3) if not np.isnan(ret_per_exp) else np.nan,
    }
    for bname, bret in benchmarks.items():
        bm_total = ((1 + bret.fillna(0)).prod() - 1) * 100
        out[f"benchmark_{bname}"] = round(bm_total, 2)
        out[f"excess_vs_{bname}"] = round(total_return - bm_total, 2)
    return out, equity, dd


def slice_period(series, start, end=None):
    s = series.loc[start:]
    if end:
        s = s.loc[:end]
    return s.dropna()

# %% [markdown]
# ## 5. Motor de backtest (sin lookahead, shift(1))

# %%
def portfolio_backtest(panel, weight_schedule, cost_per_side=COST_PER_SIDE):
    """
    Backtest de cartera. weight_schedule: pesos objetivo al cierre (se ejecutan con shift(1)).
    """
    rets = panel.pct_change().fillna(0)
    w = weight_schedule.reindex(panel.index).fillna(0)
    w_exec = w.shift(1).fillna(0)
    turnover = w.diff().abs().sum(axis=1).fillna(0)
    port_ret = (w_exec * rets).sum(axis=1) - turnover * cost_per_side
    exposure = w_exec.abs().sum(axis=1).clip(0, 1)
    num_trades = int((turnover > 0.01).sum())
    return port_ret, exposure, num_trades, turnover


def backtest_single_ticker(df, entry_signal, exit_signal, max_hold_days=None, cost_per_side=COST_PER_SIDE):
    """Backtest long-only por ticker. Señal al cierre, ejecución día siguiente."""
    d = df.copy()
    n = len(d)
    positions = np.zeros(n)
    current_pos, bars_held = 0, 0
    entry_arr = entry_signal.fillna(False).values
    exit_arr = exit_signal.fillna(False).values
    rets = d["Close"].pct_change().fillna(0).values
    trade_returns = []
    entry_price = None

    for i in range(n):
        if current_pos == 0:
            if entry_arr[i]:
                current_pos, bars_held = 1, 1
        else:
            bars_held += 1
            force_exit = max_hold_days is not None and bars_held >= max_hold_days
            if exit_arr[i] or force_exit:
                if entry_price is not None:
                    tr = (d["Close"].iloc[i] - entry_price) / entry_price - 2 * cost_per_side
                    trade_returns.append(tr)
                current_pos, bars_held, entry_price = 0, 0, None
        positions[i] = current_pos
        if current_pos == 1 and entry_price is None:
            entry_price = d["Close"].iloc[i]

    pos_series = pd.Series(positions, index=d.index)
    pos_exec = pos_series.shift(1).fillna(0)
    trades = pos_series.diff().abs().fillna(0)
    strategy_ret = pos_exec * rets - trades * cost_per_side
    return strategy_ret, pos_exec, len(trade_returns), trade_returns


def backtest_vol_target(df, cost_per_side=COST_PER_SIDE, target_vol=TARGET_VOL_ANNUAL):
    """Volatility Target Momentum: posición ajustada por volatilidad objetivo."""
    d = add_features(df)
    daily_target = target_vol / np.sqrt(252)
    vol = d["RET_1D"].rolling(20).std().replace(0, np.nan)
    vol_ok = vol < vol.quantile(0.85)  # volatilidad no extrema
    signal = (d["RET_60D"] > 0) & vol_ok
    raw_size = (daily_target / vol).clip(0, 1).fillna(0)
    position = (signal.astype(float) * raw_size).shift(1).fillna(0)
    rets = d["Close"].pct_change().fillna(0)
    turnover = position.diff().abs().fillna(position)
    strategy_ret = position * rets - turnover * cost_per_side
    num_trades = int((turnover > 0.05).sum())
    return strategy_ret, position, num_trades, []

# %% [markdown]
# ## 6. Estrategias profesionales V2

# %%
def strategy_cross_sectional_momentum(panel, top_n=TOP_N_CS_MOM, hold_days=CS_HOLD_DAYS):
    """A) Momentum cross-sectional semanal: top N del universo, hold 5 días."""
    ret20 = panel.pct_change(20).shift(1)
    ret60 = panel.pct_change(60).shift(1)
    ret120 = panel.pct_change(120).shift(1)
    score = 0.40 * ret20 + 0.35 * ret60 + 0.25 * ret120

    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    week_ends = panel.resample("W-FRI").last().index

    for reb in week_ends:
        if reb not in score.index:
            continue
        row = score.loc[reb].dropna()
        if len(row) < top_n:
            continue
        top = row.nlargest(top_n).index
        w = pd.Series(0.0, index=panel.columns)
        w[top] = 1.0 / top_n
        loc = panel.index.get_indexer([reb], method="nearest")[0]
        end_loc = min(loc + hold_days, len(panel.index) - 1)
        for j in range(loc + 1, end_loc + 1):
            weights.iloc[j] = w.values
    return weights


def strategy_etf_rotation(panel, top_n=ETF_TOP_N):
    """B) Rotación sectorial/ETF: top 3 por momentum ajustado por volatilidad, hold 1 semana."""
    ret60 = panel.pct_change(60).shift(1)
    vol20 = panel.pct_change().rolling(20).std().shift(1)
    score = ret60 / vol20.replace(0, np.nan)

    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    week_ends = panel.resample("W-FRI").last().index

    for i, reb in enumerate(week_ends):
        if reb not in score.index:
            continue
        row = score.loc[reb].dropna()
        if len(row) < top_n:
            continue
        top = row.nlargest(top_n).index
        w = pd.Series(0.0, index=panel.columns)
        w[top] = 1.0 / top_n
        loc = panel.index.get_indexer([reb], method="nearest")[0]
        next_reb = week_ends[i + 1] if i + 1 < len(week_ends) else panel.index[-1]
        end_loc = panel.index.get_indexer([next_reb], method="nearest")[0]
        for j in range(loc + 1, end_loc + 1):
            weights.iloc[j] = w.values
    return weights


def signals_trend_following(df, spy_df):
    """C) Trend Following con filtro de mercado SPY > SMA_200."""
    d = add_features(df)
    spy = add_features(spy_df)
    bull = (spy["Close"] > spy["SMA_200"]).reindex(d.index).ffill().fillna(0).astype(bool)
    entry = bull & (d["Close"] > d["SMA_100"]) & (d["EMA_21"] > d["SMA_50"])
    exit_sig = (d["Close"] < d["SMA_50"]) | (~bull)
    return entry.fillna(False), exit_sig.fillna(False)


def signals_mean_reversion_pro(df, spy_df):
    """D) Mean Reversion profesional con IBS y filtro de mercado."""
    d = add_features(df)
    spy = add_features(spy_df)
    bull = (spy["Close"] > spy["SMA_200"]).reindex(d.index).ffill().fillna(0).astype(bool)
    ibs_ok = d["IBS"] < 0.2 if "IBS" in d.columns else True
    entry = (d["RSI_2"] < 10) & ibs_ok & (d["Close"] > d["SMA_200"]) & bull
    exit_sig = (d["RSI_2"] > 50) | (d["Close"] > d["SMA_5"])
    return entry.fillna(False), exit_sig.fillna(False)

# %% [markdown]
# ## 7. Descargar datos y preparar benchmarks

# %%
all_tickers = sorted(set(UNIVERSE_TICKERS + ETF_TICKERS + [MARKET_TICKER]))
data = download_data(all_tickers, START_DATE, END_DATE)

universe_panel = panel_from_dict(data, UNIVERSE_TICKERS)
etf_panel = panel_from_dict(data, ETF_TICKERS)
spy_close = data[MARKET_TICKER]["Close"]

# Benchmarks diarios
def make_benchmarks(panel, spy_s, ticker_bh=None):
    """Benchmarks: ticker B&H, SPY, equal weight, cash."""
    bm = {}
    if ticker_bh is not None:
        bm["ticker"] = ticker_bh.reindex(panel.index).fillna(0)
    else:
        bm["ticker"] = panel.pct_change().mean(axis=1).fillna(0)
    bm["spy"] = spy_s.reindex(panel.index).pct_change().fillna(0)
    if len(panel.columns) > 1:
        bm["equal_weight"] = panel.pct_change().mean(axis=1).fillna(0)
    else:
        bm["equal_weight"] = bm["ticker"]
    bm["cash"] = pd.Series(0.0, index=panel.index)
    return bm

print(f"Panel universo: {universe_panel.shape} | Panel ETFs: {etf_panel.shape}")

# %% [markdown]
# ## 8. Ejecutar estrategias V2 con train / test

# %%
def evaluate_strategy(name, strategy_ret, exposure, num_trades, benchmarks_full, trade_returns=None, ticker="PORTFOLIO", turnover=None):
    """Evalúa una estrategia en train y test con todos los benchmarks."""
    train_ret = slice_period(strategy_ret, TRAIN_START, TRAIN_END)
    test_ret = slice_period(strategy_ret, TEST_START)
    train_exp = slice_period(exposure, TRAIN_START, TRAIN_END) if exposure is not None else None
    test_exp = slice_period(exposure, TEST_START) if exposure is not None else None

    train_bm = {k: slice_period(v, TRAIN_START, TRAIN_END) for k, v in benchmarks_full.items()}
    test_bm = {k: slice_period(v, TEST_START) for k, v in benchmarks_full.items()}

    if turnover is not None:
        num_trades_train = int((slice_period(turnover, TRAIN_START, TRAIN_END) > 0.01).sum())
        num_trades_test = int((slice_period(turnover, TEST_START) > 0.01).sum())
    else:
        num_trades_train = num_trades // 2
        num_trades_test = num_trades - num_trades_train

    train_m, _, _ = compute_metrics(train_ret, train_bm, train_exp, num_trades_train, trade_returns)
    test_m, _, _ = compute_metrics(test_ret, test_bm, test_exp, num_trades_test, trade_returns)
    full_m, equity, dd = compute_metrics(strategy_ret, benchmarks_full, exposure, num_trades, trade_returns)

    row = {
        "strategy": name,
        "ticker": ticker,
        "train_return": train_m["total_return"],
        "test_return": test_m["total_return"],
        "train_sharpe": train_m["sharpe"],
        "test_sharpe": test_m["sharpe"],
        "train_max_drawdown": train_m["max_drawdown"],
        "test_max_drawdown": test_m["max_drawdown"],
        "test_excess_vs_spy": test_m.get("excess_vs_spy", 0),
        "num_trades_train": num_trades_train,
        "num_trades_test": num_trades_test,
    }
    for k, v in full_m.items():
        row[k] = v

    row["approved_out_of_sample"] = (
        (row["test_return"] > 0)
        & (row["test_excess_vs_spy"] > 0)
        & (row["test_sharpe"] > 0.5)
        & (row["test_max_drawdown"] > -30)
        & (row["num_trades_test"] >= 10)
    )
    return row, equity, dd


def run_all_v2_strategies(data, universe_panel, etf_panel, spy_close):
    results = []
    spy_df = data[MARKET_TICKER]
    spy_bm = spy_close.pct_change().fillna(0)

    # A) Cross-sectional momentum
    w_cs = strategy_cross_sectional_momentum(universe_panel)
    ret_cs, exp_cs, n_cs, to_cs = portfolio_backtest(universe_panel, w_cs)
    bm_cs = make_benchmarks(universe_panel, spy_close)
    row, _, _ = evaluate_strategy("Cross-sectional Momentum", ret_cs, exp_cs, n_cs, bm_cs, turnover=to_cs)
    results.append(row)

    # B) ETF Rotation
    w_etf = strategy_etf_rotation(etf_panel)
    ret_etf, exp_etf, n_etf, to_etf = portfolio_backtest(etf_panel, w_etf)
    bm_etf = make_benchmarks(etf_panel, spy_close)
    row, _, _ = evaluate_strategy("ETF Rotation", ret_etf, exp_etf, n_etf, bm_etf, turnover=to_etf)
    results.append(row)

    # C, D, E) Por ticker del universo
    for ticker in UNIVERSE_TICKERS:
        if ticker not in data:
            continue
        df = add_features(data[ticker])
        ticker_bh = df["Close"].pct_change().fillna(0)
        bm_t = make_benchmarks(panel_from_dict(data, [ticker]), spy_close, ticker_bh=ticker_bh)

        # C) Trend Following
        entry, exit_sig = signals_trend_following(df, spy_df)
        ret_tf, exp_tf, n_tf, tr_tf = backtest_single_ticker(df, entry, exit_sig, max_hold_days=None)
        row, _, _ = evaluate_strategy("Trend Following", ret_tf, exp_tf, n_tf, bm_t, tr_tf, ticker)
        results.append(row)

        # D) Mean Reversion Pro
        entry, exit_sig = signals_mean_reversion_pro(df, spy_df)
        ret_mr, exp_mr, n_mr, tr_mr = backtest_single_ticker(df, entry, exit_sig, max_hold_days=MR_MAX_HOLD)
        row, _, _ = evaluate_strategy("Mean Reversion Pro", ret_mr, exp_mr, n_mr, bm_t, tr_mr, ticker)
        results.append(row)

        # E) Volatility Target Momentum
        ret_vt, exp_vt, n_vt, _ = backtest_vol_target(df)
        row, _, _ = evaluate_strategy("Vol Target Momentum", ret_vt, exp_vt, n_vt, bm_t, ticker=ticker)
        results.append(row)

    return pd.DataFrame(results)

v2_results = run_all_v2_strategies(data, universe_panel, etf_panel, spy_close)
print(f"Estrategias V2 evaluadas: {len(v2_results)}")
v2_results.head(10)

# %% [markdown]
# ## 9. Validación por años

# %%
def yearly_breakdown(strategy_ret, benchmark_ret, strategy_name, ticker="PORTFOLIO"):
    """Tabla anual: return, benchmark, excess, drawdown, trades aproximados."""
    rows = []
    years = range(2018, 2027)
    for year in years:
        mask = strategy_ret.index.year == year
        if mask.sum() < 20:
            continue
        sr = strategy_ret.loc[mask]
        br = benchmark_ret.loc[mask]
        strat_r = ((1 + sr).prod() - 1) * 100
        bench_r = ((1 + br).prod() - 1) * 100
        eq = (1 + sr).cumprod()
        dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
        trades = int((sr.abs() > 0.001).sum() / 5)  # aproximación
        rows.append({
            "strategy": strategy_name, "ticker": ticker, "year": year,
            "return": round(strat_r, 2), "benchmark_return": round(bench_r, 2),
            "excess_return": round(strat_r - bench_r, 2), "max_drawdown": round(dd, 2),
            "num_trades": trades,
        })
    return rows


def build_yearly_table(data, universe_panel, etf_panel, spy_close):
    all_rows = []
    spy_df = data[MARKET_TICKER]
    spy_bm = spy_close.pct_change().fillna(0)

    w_cs = strategy_cross_sectional_momentum(universe_panel)
    ret_cs, _, _, _ = portfolio_backtest(universe_panel, w_cs)
    all_rows += yearly_breakdown(ret_cs, spy_bm.reindex(ret_cs.index).fillna(0), "Cross-sectional Momentum")

    w_etf = strategy_etf_rotation(etf_panel)
    ret_etf, _, _, _ = portfolio_backtest(etf_panel, w_etf)
    all_rows += yearly_breakdown(ret_etf, spy_bm.reindex(ret_etf.index).fillna(0), "ETF Rotation")

    for ticker in UNIVERSE_TICKERS:
        if ticker not in data:
            continue
        df = add_features(data[ticker])
        bm = df["Close"].pct_change().fillna(0)
        entry, exit_sig = signals_trend_following(df, spy_df)
        ret, _, _, _ = backtest_single_ticker(df, entry, exit_sig)
        all_rows += yearly_breakdown(ret, bm, "Trend Following", ticker)
        entry, exit_sig = signals_mean_reversion_pro(df, spy_df)
        ret, _, _, _ = backtest_single_ticker(df, entry, exit_sig, max_hold_days=MR_MAX_HOLD)
        all_rows += yearly_breakdown(ret, bm, "Mean Reversion Pro", ticker)
        ret, _, _, _ = backtest_vol_target(df)
        all_rows += yearly_breakdown(ret, bm, "Vol Target Momentum", ticker)

    yearly_df = pd.DataFrame(all_rows)
    if not yearly_df.empty:
        years_pos = yearly_df.groupby(["strategy", "ticker"]).apply(
            lambda g: (g["excess_return"] > 0).sum()
        ).reset_index(name="years_positive")
        v2_yearly = yearly_df.merge(years_pos, on=["strategy", "ticker"], how="left")
    else:
        v2_yearly = yearly_df
    return v2_yearly

yearly_df = build_yearly_table(data, universe_panel, etf_panel, spy_close)
if not yearly_df.empty:
    print("Años positivos por estrategia (excess > 0):")
    print(yearly_df.groupby(["strategy", "ticker"])["years_positive"].max().sort_values(ascending=False).head(15).to_string())

# %% [markdown]
# ## 10. Evitar sobreoptimización
#
# Si probamos muchas combinaciones, alguna puede salir bien por casualidad. Por eso no aceptamos
# una estrategia solo por ser la mejor del ranking. Tiene que superar reglas mínimas en test,
# por años y contra benchmark.

# %% [markdown]
# ## 11. Ranking final (final_score 0-100)

# %%
def compute_final_score(df, yearly_df):
    """Score 0-100: no ordenar solo por rentabilidad."""
    r = df.copy()
    if "years_positive" not in r.columns and not yearly_df.empty:
        yp = yearly_df.groupby(["strategy", "ticker"])["years_positive"].max().reset_index()
        r = r.merge(yp, on=["strategy", "ticker"], how="left")
    r["years_positive"] = r.get("years_positive", pd.Series(0, index=r.index)).fillna(0)

    score = np.zeros(len(r))
    score += np.where(r["test_return"] > 0, 15, -10)
    score += np.where(r["test_excess_vs_spy"] > 0, 15, -15)
    score += (r["test_sharpe"].clip(-1, 2) + 1) / 3 * 15
    score += np.where(r["profit_factor"] > 1.2, 10, np.where(r["profit_factor"] > 1, 5, -10))
    score += np.where(r["test_max_drawdown"] > -20, 10, np.where(r["test_max_drawdown"] > -30, 5, -15))
    score += (r["years_positive"].clip(0, 8) / 8) * 15
    score += np.where(r["num_trades_test"] < 10, -20, 0)
    score += np.where(r["num_trades_test"] < 5, -10, 0)
    # Penalizar si train bueno pero test malo (overfitting)
    score += np.where((r["train_return"] > 20) & (r["test_return"] < 0), -20, 0)
    score += np.where((r["train_sharpe"] > 1) & (r["test_sharpe"] < 0), -15, 0)
    score += np.where(r["approved_out_of_sample"], 10, 0)

    r["final_score"] = np.clip(score, 0, 100).round(1)
    r["status"] = "descartada"
    r.loc[r["approved_out_of_sample"], "status"] = "aprobada"
    r.loc[(~r["approved_out_of_sample"]) & (r["final_score"] >= 50), "status"] = "prometedora"
    return r.sort_values("final_score", ascending=False).reset_index(drop=True)

v2_ranked = compute_final_score(v2_results, yearly_df)

print("=== TOP ESTRATEGIAS (final_score) ===")
cols_show = ["strategy", "ticker", "final_score", "status", "test_return", "test_excess_vs_spy",
             "test_sharpe", "test_max_drawdown", "years_positive", "approved_out_of_sample"]
print(v2_ranked.head(15)[cols_show].to_string())

approved_df = v2_ranked[v2_ranked["status"] == "aprobada"]
promising_df = v2_ranked[v2_ranked["status"] == "prometedora"]
discarded_df = v2_ranked[v2_ranked["status"] == "descartada"]

print(f"\nAprobadas: {len(approved_df)} | Prometedoras: {len(promising_df)} | Descartadas: {len(discarded_df)}")

# %% [markdown]
# ## 12. Visualizaciones rápidas

# %%
if len(v2_ranked) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Test return vs SPY excess
    top = v2_ranked.head(12)
    labels = [f"{r.strategy[:14]}\n{r.ticker}" for _, r in top.iterrows()]
    axes[0].barh(range(len(top)), top["test_return"], color="steelblue", alpha=0.8)
    axes[0].set_yticks(range(len(top)))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("Test return %")
    axes[0].set_title("Top 12: retorno en TEST (2023+)")
    axes[0].invert_yaxis()

    colors = top["status"].map({"aprobada": "green", "prometedora": "orange", "descartada": "gray"})
    axes[1].scatter(top["test_sharpe"], top["test_excess_vs_spy"], c=colors, s=80, alpha=0.8)
    for _, r in top.iterrows():
        axes[1].annotate(r.ticker[:4], (r.test_sharpe, r.test_excess_vs_spy), fontsize=7)
    axes[1].axhline(0, color="red", linestyle="--", alpha=0.5)
    axes[1].axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Test Sharpe")
    axes[1].set_ylabel("Excess vs SPY (test)")
    axes[1].set_title("Sharpe vs excess SPY en test")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 13. Conclusiones automáticas
#
# **Backtest no garantiza resultados futuros.**

# %%
print("=" * 70)
print("CONCLUSIONES AUTOMÁTICAS — FASE 2")
print("Backtest no garantiza resultados futuros.")
print("=" * 70)

print("\n📋 Estrategias aprobadas (out-of-sample):")
if len(approved_df) == 0:
    print("  (ninguna)")
else:
    for _, r in approved_df.iterrows():
        print(f"  • {r['strategy']} [{r['ticker']}] — test={r['test_return']}% | excess SPY={r['test_excess_vs_spy']}% | score={r['final_score']}")

print("\n🔶 Estrategias prometedoras (requieren más validación):")
if len(promising_df) == 0:
    print("  (ninguna)")
else:
    for _, r in promising_df.head(5).iterrows():
        print(f"  • {r['strategy']} [{r['ticker']}] — test={r['test_return']}% | score={r['final_score']}")

print("\n❌ Estrategias descartadas:")
print(f"  {len(discarded_df)} combinaciones no cumplen criterios mínimos.")

print("\n✅ Qué deberíamos meter en la web:")
if len(approved_df) > 0:
    for _, r in approved_df.iterrows():
        print(f"  • {r['strategy']} en {r['ticker']} (solo tras revisión manual)")
else:
    print("  Nada todavía. No actualizar la web con señales reales todavía. Seguir investigando.")

print("\n🚫 Qué NO deberíamos meter en la web:")
print("  • Todas las estrategias de la Fase 1 (ninguna superó comprar y mantener)")
print("  • Estrategias con test_return < 0 o test_excess_vs_spy < 0")
print("  • Estrategias con pocas operaciones en test (<10)")
print("  • Cualquier estrategia que solo funcionó en train y falló en test")

if len(approved_df) == 0:
    print("\n⛔ No actualizar la web con señales reales todavía. Seguir investigando.")

print("\n⚠️ Investigación educativa. No es asesoramiento financiero.")

# %% [markdown]
# ## 14. Exportar resultados

# %%
OUT_ALL = "research_v2_results.csv"
OUT_APPROVED = "research_v2_approved_strategies.csv"

v2_ranked.to_csv(OUT_ALL, index=False)
approved_df.to_csv(OUT_APPROVED, index=False)
yearly_df.to_csv("research_v2_yearly.csv", index=False)

print(f"Guardado: {OUT_ALL} ({len(v2_ranked)} filas)")
print(f"Guardado: {OUT_APPROVED} ({len(approved_df)} filas)")
print(f"Guardado: research_v2_yearly.csv ({len(yearly_df)} filas)")

# En Colab:
# from google.colab import files
# files.download("research_v2_results.csv")
# files.download("research_v2_approved_strategies.csv")
