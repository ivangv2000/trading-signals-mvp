# %% [markdown]
# # Trading Research Lab — Google Colab
#
# Notebook de investigación cuantitativa para validar estrategias de trading.
#
# **Aviso:** El backtest no garantiza resultados futuros. Esto no es asesoramiento financiero.

# %%
# Instalación (ejecutar primero en Colab)
!pip install yfinance pandas numpy matplotlib plotly tqdm scikit-learn -q

# %% [markdown]
# ## 1. Imports y configuración
#
# Edita las variables de abajo antes de ejecutar el resto del notebook.
#
# - `TRANSACTION_COST`: comisión aproximada por operación (entrada o salida).
# - `SLIPPAGE`: diferencia entre el precio teórico y el precio real de ejecución.

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
from tqdm.auto import tqdm

# Configuración editable
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN"]
START_DATE = "2018-01-01"
END_DATE = None  # None = hasta hoy

HOLDING_DAYS_LIST = [1, 2, 3, 5]
TRANSACTION_COST = 0.001
SLIPPAGE = 0.001

MARKET_TICKER = "SPY"  # Para filtro de régimen de mercado

print("Configuración cargada.")
print(f"Tickers: {len(TICKERS)} | Desde: {START_DATE} | Holding days: {HOLDING_DAYS_LIST}")

# %% [markdown]
# ## 2. Descarga de datos

# %%
def download_data(tickers, start, end=None):
    """Descarga datos diarios y devuelve {ticker: DataFrame}."""
    data = {}
    failed = []
    for ticker in tqdm(tickers, desc="Descargando"):
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
                continue
            data[ticker.upper()] = df
        except Exception as e:
            failed.append(f"{ticker} ({e})")
    if failed:
        print("Tickers fallidos:", failed)
    print(f"Descargados: {len(data)} tickers")
    return data

# %% [markdown]
# ## 3. Features / indicadores (sin lookahead)

# %%
def _rsi(close, window):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def add_features(df):
    """Añade indicadores técnicos al DataFrame."""
    d = df.copy()
    c = d["Close"]
    d["SMA_20"] = c.rolling(20).mean()
    d["SMA_50"] = c.rolling(50).mean()
    d["SMA_100"] = c.rolling(100).mean()
    d["SMA_200"] = c.rolling(200).mean()
    d["EMA_9"] = c.ewm(span=9, adjust=False).mean()
    d["EMA_21"] = c.ewm(span=21, adjust=False).mean()

    d["RET_1D"] = c.pct_change(1)
    d["RET_5D"] = c.pct_change(5)
    d["RET_10D"] = c.pct_change(10)
    d["RET_20D"] = c.pct_change(20)
    d["MOMENTUM_5"] = c - c.shift(5)
    d["MOMENTUM_20"] = c - c.shift(20)

    d["VOL_10"] = d["RET_1D"].rolling(10).std()
    d["VOL_20"] = d["RET_1D"].rolling(20).std()

    if "High" in d.columns and "Low" in d.columns:
        prev = c.shift(1)
        tr = pd.concat([(d["High"] - d["Low"]).abs(), (d["High"] - prev).abs(), (d["Low"] - prev).abs()], axis=1).max(axis=1)
        d["ATR_14"] = tr.rolling(14).mean()
        d["HIGH_10"] = d["High"].rolling(10).max().shift(1)
        d["HIGH_20"] = d["High"].rolling(20).max().shift(1)
        d["LOW_10"] = d["Low"].rolling(10).min().shift(1)
        d["LOW_20"] = d["Low"].rolling(20).min().shift(1)

    d["RSI_2"] = _rsi(c, 2)
    d["RSI_14"] = _rsi(c, 14)

    if "Volume" in d.columns:
        d["VOLUME_AVG_20"] = d["Volume"].rolling(20).mean()
        d["VOLUME_RATIO"] = d["Volume"] / d["VOLUME_AVG_20"].replace(0, np.nan)

    d["MOM_SCORE"] = d["RET_20D"] / d["VOL_20"].replace(0, np.nan)
    return d


def add_market_regime(df, spy_df):
    """Añade columna MARKET_BULL: SPY > SMA_200."""
    spy = add_features(spy_df.copy())
    regime = (spy["Close"] > spy["SMA_200"]).astype(float)
    regime = regime.reindex(df.index).ffill().fillna(0)
    out = df.copy()
    out["MARKET_BULL"] = regime
    return out

# %% [markdown]
# ## 4. Estrategias (entry / exit signals)

# %%
def ema_trend_signals(df):
    entry = (df["EMA_9"] > df["EMA_21"]) & (df["RSI_14"] >= 45) & (df["RSI_14"] <= 75) & (df["MOMENTUM_5"] > 0)
    exit_sig = (df["EMA_9"] < df["EMA_21"]) | (df["RSI_14"] < 40) | (df["MOMENTUM_5"] < 0)
    return entry.fillna(False), exit_sig.fillna(False)


def donchian_breakout_signals(df):
    entry = (df["Close"] > df["HIGH_20"]) & (df["MOMENTUM_5"] > 0) & (df["RSI_14"] >= 50) & (df["RSI_14"] <= 80)
    exit_sig = (df["Close"] < df["EMA_9"]) | (df["MOMENTUM_5"] < 0)
    return entry.fillna(False), exit_sig.fillna(False)


def pullback_trend_signals(df):
    near_ema = ((df["Close"] - df["EMA_21"]).abs() / df["Close"]) < 0.03
    entry = (df["Close"] > df["SMA_100"]) & (df["EMA_9"] > df["EMA_21"]) & (df["RSI_14"] >= 40) & (df["RSI_14"] <= 60) & near_ema
    exit_sig = (df["Close"] < df["EMA_21"]) | (df["RSI_14"] > 70)
    return entry.fillna(False), exit_sig.fillna(False)


def mean_reversion_rsi2_signals(df):
    if "SMA_200" in df.columns:
        sma_ok = df["Close"] > df["SMA_200"]
    else:
        sma_ok = pd.Series(True, index=df.index)
    entry = (df["RSI_2"] < 10) & sma_ok
    exit_sig = (df["RSI_2"] > 50) | (df["Close"] > df["EMA_9"])
    return entry.fillna(False), exit_sig.fillna(False)


def vol_adj_momentum_signals(df):
    entry = (df["MOM_SCORE"] > 1) & (df["Close"] > df["SMA_50"]) & (df["MOMENTUM_5"] > 0)
    exit_sig = (df["RET_5D"] < 0) | (df["Close"] < df["EMA_21"])
    return entry.fillna(False), exit_sig.fillna(False)


def market_regime_momentum_signals(df):
    bull = df.get("MARKET_BULL", pd.Series(1, index=df.index)) == 1
    entry = bull & (df["Close"] > df["HIGH_10"]) & (df["MOMENTUM_5"] > 0)
    exit_sig = (df["Close"] < df["EMA_21"]) | (~bull)
    return entry.fillna(False), exit_sig.fillna(False)


STRATEGIES = {
    "EMA Trend": ema_trend_signals,
    "Donchian Breakout": donchian_breakout_signals,
    "Pullback Trend": pullback_trend_signals,
    "Mean Reversion RSI2": mean_reversion_rsi2_signals,
    "Volatility Adjusted Momentum": vol_adj_momentum_signals,
    "Market Regime + Momentum": market_regime_momentum_signals,
}

# %% [markdown]
# ## 5. Backtest con costes y slippage

# %%
def backtest_strategy(df, entry_signal, exit_signal, holding_days, transaction_cost, slippage):
    """
    Backtest long-only. Señal al cierre, posición desde día siguiente (shift 1).
    """
    d = df.copy()
    n = len(d)
    positions = np.zeros(n)
    current_pos = 0
    bars_held = 0
    trade_returns = []
    entry_arr = entry_signal.fillna(False).values
    exit_arr = exit_signal.fillna(False).values
    rets = d["Close"].pct_change().fillna(0).values
    cost_per_side = transaction_cost + slippage

    for i in range(n):
        if current_pos == 0:
            if entry_arr[i]:
                current_pos = 1
                bars_held = 1
        else:
            bars_held += 1
            if exit_arr[i] or bars_held >= holding_days:
                current_pos = 0
                bars_held = 0
        positions[i] = current_pos

    pos_series = pd.Series(positions, index=d.index)
    pos_shifted = pos_series.shift(1).fillna(0)
    trades = pos_series.diff().abs().fillna(0)
    strategy_ret = pos_shifted * rets - trades * cost_per_side

    equity = (1 + strategy_ret).cumprod()
    benchmark = (1 + pd.Series(rets, index=d.index)).cumprod()

    # Operaciones individuales
    in_trade = False
    entry_price = None
    for i in range(1, n):
        if trades.iloc[i] == 1 and pos_series.iloc[i] == 1:
            entry_price = d["Close"].iloc[i]
            in_trade = True
        elif trades.iloc[i] == 1 and pos_series.iloc[i] == 0 and in_trade and entry_price:
            exit_price = d["Close"].iloc[i]
            tr = (exit_price - entry_price) / entry_price - 2 * cost_per_side
            trade_returns.append(tr)
            in_trade = False
            entry_price = None

    total_return = (equity.iloc[-1] - 1) * 100 if len(equity) else 0
    benchmark_return = (benchmark.iloc[-1] - 1) * 100 if len(benchmark) else 0
    excess_return = total_return - benchmark_return

    years = max((d.index[-1] - d.index[0]).days / 365.25, 1 / 365.25)
    cagr = ((equity.iloc[-1]) ** (1 / years) - 1) * 100 if equity.iloc[-1] > 0 else 0

    ann_vol = strategy_ret.std() * np.sqrt(252) * 100
    sharpe = (strategy_ret.mean() / strategy_ret.std() * np.sqrt(252)) if strategy_ret.std() > 0 else 0

    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    max_dd = dd.min() * 100

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    num_trades = len(trade_returns)
    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]
    win_rate = (len(wins) / num_trades * 100) if num_trades else 0
    avg_trade = (np.mean(trade_returns) * 100) if trade_returns else 0
    best = max(trade_returns) * 100 if trade_returns else 0
    worst = min(trade_returns) * 100 if trade_returns else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 0)
    exposure_pct = pos_shifted.mean() * 100

    metrics = {
        "total_return": round(total_return, 2),
        "benchmark_return": round(benchmark_return, 2),
        "excess_return": round(excess_return, 2),
        "CAGR": round(cagr, 2),
        "annual_volatility": round(ann_vol, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
        "calmar": round(calmar, 3),
        "num_trades": int(num_trades),
        "win_rate": round(win_rate, 2),
        "avg_trade_return": round(avg_trade, 3),
        "best_trade": round(best, 3),
        "worst_trade": round(worst, 3),
        "profit_factor": round(profit_factor, 3) if np.isfinite(profit_factor) else 999,
        "exposure_pct": round(exposure_pct, 2),
    }
    bt_df = d.copy()
    bt_df["Position"] = pos_series
    bt_df["Strategy_Return"] = strategy_ret
    bt_df["Equity"] = equity
    bt_df["Benchmark"] = benchmark
    bt_df["Drawdown"] = dd
    return bt_df, metrics

# %% [markdown]
# ## 6. Ejecutar investigación completa

# %%
def run_research(data_dict, holding_days_list, spy_df=None):
    rows = []
    for ticker, raw_df in tqdm(data_dict.items(), desc="Tickers"):
        df = add_features(raw_df)
        if spy_df is not None and ticker != MARKET_TICKER:
            df = add_market_regime(df, spy_df)
        elif "MARKET_BULL" not in df.columns:
            df["MARKET_BULL"] = 1

        for strat_name, strat_fn in STRATEGIES.items():
            entry, exit_sig = strat_fn(df)
            for hd in holding_days_list:
                try:
                    _, m = backtest_strategy(df, entry, exit_sig, hd, TRANSACTION_COST, SLIPPAGE)
                    rows.append({"ticker": ticker, "strategy": strat_name, "holding_days": hd, **m})
                except Exception as e:
                    rows.append({"ticker": ticker, "strategy": strat_name, "holding_days": hd, "error": str(e)})
    return pd.DataFrame(rows)


def compute_robust_score(df):
    """Score robusto: no ordenar solo por rentabilidad."""
    r = df.copy()
    sharpe_norm = (r["sharpe"].clip(-1, 3) + 1) / 4 * 25
    excess_pts = r["excess_return"].clip(-30, 30) / 30 * 20
    calmar_pts = r["calmar"].clip(0, 3) / 3 * 15
    win_pts = r["win_rate"].clip(0, 100) / 100 * 10

    penalty_trades = np.where(r["num_trades"] < 20, -25, 0)
    penalty_dd = np.where(r["max_drawdown"] < -30, -20, 0)
    penalty_pf = np.where(r["profit_factor"] < 1, -15, 0)
    penalty_excess = np.where(r["excess_return"] < 0, -15, 0)

    r["robust_score"] = (
        sharpe_norm + excess_pts + calmar_pts + win_pts
        + penalty_trades + penalty_dd + penalty_pf + penalty_excess
    ).round(2)
    return r

# %% [markdown]
# ## 7. Ejecutar (puede tardar varios minutos)

# %%
all_tickers = sorted(set(TICKERS + [MARKET_TICKER]))
data = download_data(all_tickers, START_DATE, END_DATE)

for t, df in data.items():
    data[t] = add_features(df)

spy_df = data.get(MARKET_TICKER)
research_tickers = {t: data[t] for t in TICKERS if t in data}
results = run_research(research_tickers, HOLDING_DAYS_LIST, spy_df)
results = compute_robust_score(results)
results = results.sort_values("robust_score", ascending=False).reset_index(drop=True)

print(f"Combinaciones probadas: {len(results)}")
results.head(10)

# %% [markdown]
# ## 8. Rankings

# %%
print("=== TOP 20 COMBINACIONES (robust_score) ===")
print(results.head(20)[["ticker", "strategy", "holding_days", "robust_score", "total_return", "benchmark_return", "excess_return", "sharpe", "max_drawdown", "num_trades"]].to_string())

print("\n=== TOP POR TICKER (mejor robust_score) ===")
top_ticker = results.loc[results.groupby("ticker")["robust_score"].idxmax().dropna()]
print(top_ticker.sort_values("robust_score", ascending=False)[["ticker", "strategy", "holding_days", "robust_score", "excess_return", "num_trades"]].to_string())

print("\n=== TOP POR ESTRATEGIA ===")
top_strat = results.loc[results.groupby("strategy")["robust_score"].idxmax().dropna()]
print(top_strat.sort_values("robust_score", ascending=False)[["strategy", "ticker", "holding_days", "robust_score", "excess_return"]].to_string())

print("\n=== PEORES 10 ===")
print(results.tail(10)[["ticker", "strategy", "holding_days", "robust_score", "excess_return", "max_drawdown"]].to_string())

# %% [markdown]
# ## 9. Visualizaciones

# %%
# Seleccionar mejor combinación para gráficos
best_row = results.iloc[0]
best_ticker = best_row["ticker"]
best_strat = best_row["strategy"]
best_hd = int(best_row["holding_days"])

df_best = data[best_ticker]
entry, exit_sig = STRATEGIES[best_strat](df_best)
bt_df, _ = backtest_strategy(df_best, entry, exit_sig, best_hd, TRANSACTION_COST, SLIPPAGE)

# A) Equity curve
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(bt_df.index, bt_df["Equity"], label="Estrategia", linewidth=2)
ax.plot(bt_df.index, bt_df["Benchmark"], label="Comprar y mantener", linewidth=1.5, alpha=0.8)
ax.set_title(f"Equity: {best_ticker} | {best_strat} | holding={best_hd}d")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# B) Barras total vs benchmark (top 15)
top15 = results.head(15)
x = np.arange(len(top15))
w = 0.35
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - w/2, top15["total_return"], w, label="Estrategia")
ax.bar(x + w/2, top15["benchmark_return"], w, label="Benchmark")
ax.set_xticks(x)
ax.set_xticklabels([f"{r.ticker}\n{r.strategy[:12]}" for _, r in top15.iterrows()], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Retorno %")
ax.set_title("Top 15: Estrategia vs Comprar y mantener")
ax.legend()
plt.tight_layout()
plt.show()

# C) Heatmap robust_score
pivot = results.pivot_table(index="strategy", columns="holding_days", values="robust_score", aggfunc="mean")
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_title("Heatmap robust_score (media por estrategia x holding)")
plt.colorbar(im)
plt.tight_layout()
plt.show()

# D) Drawdown
fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(bt_df.index, bt_df["Drawdown"] * 100, 0, color="red", alpha=0.4)
ax.set_title(f"Drawdown % — {best_ticker} | {best_strat}")
ax.set_ylabel("Drawdown %")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Validación por años (walk-forward simple)

# %%
def yearly_performance(data_dict, holding_days=3):
    """Retorno anual estrategia vs benchmark por año."""
    years = list(range(2018, 2027))
    rows = []
    for ticker, raw in data_dict.items():
        if ticker == MARKET_TICKER:
            continue
        df = add_features(raw)
        if spy_df is not None:
            df = add_market_regime(df, spy_df)
        for strat_name, strat_fn in STRATEGIES.items():
            entry, exit_sig = strat_fn(df)
            bt, m = backtest_strategy(df, entry, exit_sig, holding_days, TRANSACTION_COST, SLIPPAGE)
            for year in years:
                mask = bt.index.year == year
                if mask.sum() < 20:
                    continue
                yr_ret = ((1 + bt.loc[mask, "Strategy_Return"]).prod() - 1) * 100
                bm_ret = ((1 + bt.loc[mask, "Close"].pct_change().fillna(0)).prod() - 1) * 100
                rows.append({
                    "ticker": ticker, "strategy": strat_name, "year": year,
                    "strategy_return": round(yr_ret, 2), "benchmark_return": round(bm_ret, 2),
                    "beats_benchmark": yr_ret > bm_ret,
                })
    return pd.DataFrame(rows)

yearly = yearly_performance(data, holding_days=3)
yearly_summary = yearly.groupby(["strategy", "year"])["beats_benchmark"].mean().unstack(fill_value=np.nan)
print("=== % de tickers que superan benchmark por año (holding=3) ===")
print(yearly_summary.round(2).to_string())

# Detectar estrategias que solo funcionan un año
win_years = yearly.groupby("strategy").apply(lambda g: g.groupby("year")["beats_benchmark"].mean())
print("\nEstrategias con resultados inestables (pocas victorias en algunos años) — revisar manualmente.")

# %% [markdown]
# ## 11. Conclusiones automáticas
#
# **Backtest no garantiza resultados futuros.**

# %%
print("=" * 60)
print("CONCLUSIONES AUTOMÁTICAS")
print("Backtest no garantiza resultados futuros.")
print("=" * 60)

best_global = results.iloc[0]
print(f"\nMejor combinación global: {best_global['ticker']} | {best_global['strategy']} | holding={best_global['holding_days']}")
print(f"  robust_score={best_global['robust_score']} | excess_return={best_global['excess_return']}% | sharpe={best_global['sharpe']}")

beats_bm = results[results["excess_return"] > 0]
print(f"\nCombinaciones que superan comprar y mantener: {len(beats_bm)} / {len(results)}")

discarded = results[(results["num_trades"] < 20) | (results["max_drawdown"] < -30) | (results["profit_factor"] < 1) | (results["excess_return"] < 0)]
print(f"Combinaciones descartadas por reglas: {len(discarded)}")

print("\nAdvertencias:")
low_trades = results[results["num_trades"] < 20]
if len(low_trades):
    print(f"  - {len(low_trades)} combinaciones con pocas operaciones (<20)")
high_dd = results[results["max_drawdown"] < -30]
if len(high_dd):
    print(f"  - {len(high_dd)} combinaciones con drawdown > 30%")
no_bm = results[results["excess_return"] < 0]
if len(no_bm):
    print(f"  - {len(no_bm)} combinaciones NO superan benchmark")

best_tickers = top_ticker.sort_values("robust_score", ascending=False).head(5)
print("\nMejores tickers:")
for _, r in best_tickers.iterrows():
    print(f"  {r['ticker']}: {r['strategy']} (score={r['robust_score']})")

print("\n⚠️ Esto es investigación educativa. No es asesoramiento financiero.")

# %% [markdown]
# ## 12. Exportar resultados

# %%
OUTPUT_FILE = "research_results.csv"
results.to_csv(OUTPUT_FILE, index=False)
print(f"Guardado: {OUTPUT_FILE} ({len(results)} filas)")

# En Colab, descargar:
# from google.colab import files
# files.download("research_results.csv")
