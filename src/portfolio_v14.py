"""
Portfolio V14 — R1 Return Engine (A_tsmom_63, paper trading only).

Time-series momentum semanal: top 3 activos, vol target 15%, lookback 63 días.
No ejecuta órdenes ni conecta brokers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "approved_v14_strategy.json"
MIN_HISTORY_DAYS = 200
MAX_WEIGHT = 0.30
CASH_ASSET = "SHY"
MARKET = "SPY"

DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLC", "XLB", "XLRE",
    "MTUM", "QUAL", "USMV", "VLUE", "SPLV", "SPHB", "SCHD",
    "SHY", "IEF", "TLT", "LQD", "HYG",
    "GLD", "SLV", "DBC", "VNQ",
    "EFA", "EEM",
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "GOOGL", "META", "AMZN",
    "JPM", "BAC", "XOM", "CVX", "UNH", "LLY", "WMT", "COST",
]
DEFENSIVE_POOL = ["SHY", "IEF", "TLT", "GLD", "USMV", "QUAL", "SCHD", "XLV", "XLP"]


def load_v14_config(path: str | Path | None = None) -> dict:
    """Lee config/approved_v14_strategy.json."""
    cfg_path = Path(path) if path else CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"No se encontró configuración V14: {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _norm_idx(ix) -> pd.DatetimeIndex:
    ix = pd.DatetimeIndex(ix)
    return ix.tz_localize(None) if ix.tz else ix


def build_close_prices(data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Construye DataFrame de cierres desde datos por ticker.
    Añade SHY si existe en data_dict, CASH sintético, alinea fechas y limpia NaN.
    """
    if not data_dict:
        raise ValueError("data_dict vacío: no hay datos de precios.")

    series = {}
    for ticker, df in data_dict.items():
        if ticker == "CASH":
            continue
        if df is None or getattr(df, "empty", True):
            continue
        work = df.copy()
        if isinstance(work.columns, pd.MultiIndex):
            work.columns = work.columns.get_level_values(0)
        colmap = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        work = work.rename(columns={c: colmap.get(str(c).lower(), c) for c in work.columns})
        if "Close" not in work.columns:
            continue
        close = pd.to_numeric(work["Close"], errors="coerce").dropna()
        if close.empty:
            continue
        close.index = _norm_idx(close.index)
        series[ticker.upper()] = close.sort_index()

    if not series:
        raise ValueError("No se pudo extraer ningún precio de cierre válido.")

    close_prices = pd.DataFrame(series).sort_index().ffill()
    close_prices = close_prices.dropna(how="all")
    if close_prices.empty:
        raise ValueError("DataFrame de cierres vacío tras alinear fechas.")

    close_prices["CASH"] = 1.0
    return close_prices


def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, pd.Series):
            if len(x) == 0:
                return default
            x = x.iloc[0]
        value = float(x)
        return default if not np.isfinite(value) else value
    except Exception:
        return default


def calculate_v14_features(close_prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Features para A_tsmom_63: momentum, tendencia, volatilidad y régimen de mercado.
    """
    rets = close_prices.pct_change().fillna(0)
    feats: dict[str, pd.DataFrame] = {}

    for t in close_prices.columns:
        if t == "CASH":
            continue
        c = close_prices[t]
        f = pd.DataFrame(index=c.index)
        f["MOM_63"] = c / c.shift(63) - 1
        f["SMA_200"] = c.rolling(200).mean()
        f["ABOVE_SMA200"] = (c > f["SMA_200"]).astype(float)
        f["ABOVE_SMA_200"] = f["ABOVE_SMA200"]
        f["VOL_20"] = rets[t].rolling(20).std() * math.sqrt(252)
        f["VOL_63"] = rets[t].rolling(63).std() * math.sqrt(252)
        f["MOMENTUM_SCORE"] = f["MOM_63"]
        f["TREND_SCORE"] = f["ABOVE_SMA200"] * 0.4 + f["ABOVE_SMA200"] * 0.6
        f["COMBINED_SCORE"] = f["MOM_63"] * 0.5 + f["TREND_SCORE"] * 0.5
        feats[t] = f

    dates = close_prices.index
    mkt = pd.DataFrame(index=dates)
    if MARKET in feats:
        mkt["SPY_ABOVE_SMA200"] = feats[MARKET]["ABOVE_SMA200"]
    else:
        mkt["SPY_ABOVE_SMA200"] = 0.0
    if "QQQ" in feats:
        mkt["QQQ_ABOVE_SMA200"] = feats["QQQ"]["ABOVE_SMA200"]
    else:
        mkt["QQQ_ABOVE_SMA200"] = mkt["SPY_ABOVE_SMA200"]
    mkt["MARKET_RISK_ON"] = (
        (mkt["SPY_ABOVE_SMA200"] >= 0.5) & (mkt["QQQ_ABOVE_SMA200"] >= 0.5)
    ).astype(float)

    for t in feats:
        for col in mkt.columns:
            feats[t][col] = mkt[col]

    return feats


def _row_asof(feats: dict[str, pd.DataFrame], ticker: str, dt) -> pd.Series | None:
    if ticker not in feats:
        return None
    df = feats[ticker]
    dt = pd.Timestamp(dt)
    idx = df.index[df.index <= dt]
    if len(idx) == 0:
        return None
    return df.loc[idx[-1]]


def _normalize_weights(w: pd.Series, cap: float = MAX_WEIGHT) -> pd.Series:
    w = w.astype(float).clip(lower=0).fillna(0)
    if w.sum() <= 0:
        return w
    w = w / w.sum()
    for _ in range(8):
        over = w[w > cap]
        if over.empty:
            break
        excess = (over - cap).sum()
        w.loc[over.index] = cap
        under = w[w < cap]
        if under.sum() > 0:
            w.loc[under.index] += excess * (under / under.sum())
    return w / w.sum() if w.sum() > 0 else w


def _inverse_vol_weights(tickers, vols: dict, cap: float = MAX_WEIGHT) -> pd.Series:
    inv = pd.Series({t: 1.0 / max(_safe_float(vols.get(t), 0.15), 0.05) for t in tickers})
    return _normalize_weights(inv, cap)


def _cash_fill(w: pd.Series, close_cols, frac: float = 1.0) -> pd.Series:
    w = w.copy()
    safe = CASH_ASSET if CASH_ASSET in close_cols else "CASH"
    residual = max(0.0, frac - w.sum())
    if residual > 0:
        w[safe] = w.get(safe, 0) + residual
    return _normalize_weights(w)


def _reb_dates(ix, freq: str = "W-FRI") -> pd.DatetimeIndex:
    ix = _norm_idx(ix)
    s = pd.Series(np.arange(len(ix)), index=ix)
    return pd.DatetimeIndex([g.index[-1] for _, g in s.groupby(pd.Grouper(freq=freq)) if len(g)])


def calculate_v14_target_weights(
    close_prices: pd.DataFrame,
    config: dict | None = None,
    features: dict[str, pd.DataFrame] | None = None,
    as_of_date=None,
) -> pd.Series:
    """
    Calcula pesos objetivo A_tsmom_63 para la fecha más reciente (o as_of_date).
    Devuelve Series que suma 1.0; 100% SHY/CASH si no hay activos válidos.
    """
    cfg = config or load_v14_config()
    engine = cfg.get("base_engine", {})
    top_n = int(engine.get("top_n", 3))
    vol_target = float(engine.get("vol_target", 0.15))
    lookback = int(engine.get("lookback_days", 63))
    rebalance_freq = cfg.get("rebalance_freq", "W-FRI")
    mom_col = "MOM_63" if lookback == 63 else f"MOM_{lookback}"

    if features is None:
        features = calculate_v14_features(close_prices)

    rdates = list(_reb_dates(close_prices.index, rebalance_freq))
    if not rdates:
        return _defensive_only(close_prices.columns)

    if as_of_date is not None:
        as_of = pd.Timestamp(as_of_date)
        eligible = [d for d in rdates if d <= as_of]
        dt = eligible[-1] if eligible else rdates[-1]
    else:
        dt = rdates[-1]

    risk_on_row = _row_asof(features, MARKET, dt)
    risk_on = _safe_float(risk_on_row.get("MARKET_RISK_ON", 1) if risk_on_row is not None else 1) >= 0.5

    universe = [c for c in close_prices.columns if c not in ("CASH",)]
    scores: dict[str, float] = {}
    vols: dict[str, float] = {}

    if risk_on:
        for t in universe:
            if t in (CASH_ASSET,):
                continue
            row = _row_asof(features, t, dt)
            if row is None:
                continue
            mom = _safe_float(row.get(mom_col, row.get("MOM_63", np.nan)))
            above = _safe_float(row.get("ABOVE_SMA200", row.get("ABOVE_SMA_200", 0)))
            vol = _safe_float(row.get("VOL_63", 0.15))
            if mom <= 0 or above < 0.5:
                continue
            if vol > 0.45:
                continue
            scores[t] = _safe_float(row.get("MOM_63", mom)) * 0.5 + _safe_float(row.get("TREND_SCORE", 0)) * 0.5
            vols[t] = vol
    else:
        pool = [t for t in DEFENSIVE_POOL if t in close_prices.columns]
        for t in pool:
            row = _row_asof(features, t, dt)
            if row is None:
                continue
            scores[t] = _safe_float(row.get("TREND_SCORE", 0))
            vols[t] = 0.12

    top = pd.Series(scores).sort_values(ascending=False).head(top_n)
    if top.empty:
        return _defensive_only(close_prices.columns)

    w = _inverse_vol_weights(top.index, vols)
    scale = min(1.0, vol_target / 0.15) * (0.9 if risk_on else 0.55)
    w = w * scale
    return _cash_fill(w, close_prices.columns)


def _defensive_only(cols) -> pd.Series:
    w = pd.Series(dtype=float)
    if CASH_ASSET in cols:
        w[CASH_ASSET] = 1.0
    elif "SHY" in cols:
        w["SHY"] = 1.0
    else:
        w["CASH"] = 1.0
    return _normalize_weights(w)


def calculate_v14_weights_schedule(
    close_prices: pd.DataFrame,
    config: dict | None = None,
    features: dict[str, pd.DataFrame] | None = None,
) -> dict[pd.Timestamp, pd.Series]:
    """Pesos por fecha de rebalanceo (útil para previous_weights)."""
    cfg = config or load_v14_config()
    rebalance_freq = cfg.get("rebalance_freq", "W-FRI")
    if features is None:
        features = calculate_v14_features(close_prices)
    schedule = {}
    for dt in _reb_dates(close_prices.index, rebalance_freq):
        schedule[dt] = calculate_v14_target_weights(
            close_prices, cfg, features, as_of_date=dt
        )
    return schedule


def generate_v14_signals(
    target_weights: pd.Series | dict,
    previous_weights: pd.Series | dict | None = None,
    features: dict[str, pd.DataFrame] | None = None,
    as_of_date=None,
) -> pd.DataFrame:
    """Genera tabla de señales BUY/HOLD/SELL/etc."""
    target = pd.Series(target_weights).astype(float).fillna(0)
    prev = pd.Series(previous_weights).astype(float).fillna(0) if previous_weights is not None else pd.Series(dtype=float)
    tickers = sorted(set(target.index) | set(prev.index))

    rows = []
    for t in tickers:
        tw = _safe_float(target.get(t, 0), 0)
        pw = _safe_float(prev.get(t, 0), 0)
        chg = tw - pw
        score = 50.0
        if features and as_of_date is not None:
            row = _row_asof(features, t, as_of_date)
            if row is not None:
                score = (_safe_float(row.get("MOM_63", 0)) + _safe_float(row.get("TREND_SCORE", 0))) * 50

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

        reason = f"A_tsmom_63 | {sig.lower()}"
        if t in (CASH_ASSET, "SHY", "CASH") and tw > 0.5:
            reason = "Modo defensivo — sin momentum válido en universo"

        rows.append({
            "ticker": t,
            "signal": sig,
            "target_weight": round(tw, 4),
            "previous_weight": round(pw, 4),
            "change": round(chg, 4),
            "score": round(score, 1),
            "reason": reason,
            "entry_plan": "próxima apertura post-viernes" if sig in ("BUY", "INCREASE") else "-",
            "exit_plan": "rebalance semanal" if sig in ("SELL", "REDUCE") else "mantener hasta viernes",
            "next_review": "próximo viernes",
            "cash_account_executable": True,
        })

    return pd.DataFrame(rows)


def get_required_tickers(config: dict | None = None) -> list[str]:
    cfg = config or load_v14_config()
    tickers = set(DEFAULT_UNIVERSE)
    tickers.update(DEFENSIVE_POOL)
    tickers.update(["SPY", "QQQ"])
    tickers.discard("CASH")
    return sorted(tickers)


def get_current_v14_portfolio_signal(
    data_dict: dict[str, pd.DataFrame],
    capital: float = 10000,
    config: dict | None = None,
) -> dict:
    """
    Señal actual del portfolio V14 (paper trading).
    No ejecuta órdenes ni conecta broker.
    """
    cfg = config or load_v14_config()
    warnings = list(cfg.get("warnings", []))
    backtest = cfg.get("backtest_summary", {})

    try:
        close_prices = build_close_prices(data_dict)
    except Exception as exc:
        return {
            "error": f"No hay datos suficientes: {exc}",
            "strategy_name": cfg.get("strategy_name", "V14 R1 Return Engine"),
            "approved_for_web_paper": cfg.get("approved_for_web_paper", True),
            "approved_for_real_money": False,
            "warnings": warnings,
        }

    if len(close_prices) < MIN_HISTORY_DAYS:
        return {
            "error": f"Se necesitan al menos {MIN_HISTORY_DAYS} días de historia. Hay {len(close_prices)}.",
            "strategy_name": cfg.get("strategy_name", "V14 R1 Return Engine"),
            "approved_for_web_paper": cfg.get("approved_for_web_paper", True),
            "approved_for_real_money": False,
            "warnings": warnings,
        }

    features = calculate_v14_features(close_prices)
    schedule = calculate_v14_weights_schedule(close_prices, cfg, features)
    rdates = sorted(schedule.keys())
    last_dt = rdates[-1]
    prev_dt = rdates[-2] if len(rdates) >= 2 else None

    target = schedule[last_dt]
    previous = schedule[prev_dt] if prev_dt else pd.Series(dtype=float)

    if target.sum() <= 0:
        target = _defensive_only(close_prices.columns)
    elif abs(target.sum() - 1.0) > 0.01:
        target = target / target.sum()

    target_weights = {k: round(float(v), 4) for k, v in target.items() if v > 1e-6}
    capital_allocation = {k: round(capital * w, 2) for k, w in target_weights.items()}

    signals_df = generate_v14_signals(target, previous, features, last_dt)
    signals_df = signals_df[signals_df["target_weight"] > 0.001].sort_values(
        "target_weight", ascending=False
    )

    risk_row = _row_asof(features, MARKET, last_dt)
    risk_on = _safe_float(risk_row.get("MARKET_RISK_ON", 1) if risk_row is not None else 1) >= 0.5
    shy_w = target_weights.get(CASH_ASSET, 0) + target_weights.get("SHY", 0) + target_weights.get("CASH", 0)
    if shy_w >= 0.99:
        risk_mode = "defensive_100_shy"
    elif risk_on:
        risk_mode = "risk_on_momentum"
    else:
        risk_mode = "risk_off_defensive"

    last_date = str(close_prices.index[-1].date()) if len(close_prices) else "N/A"

    return {
        "strategy_name": cfg.get("strategy_name", "V14 R1 Return Engine"),
        "strategy_id": cfg.get("strategy_id", "v14_r1_return_engine"),
        "approved_for_web_paper": bool(cfg.get("approved_for_web_paper", True)),
        "approved_for_real_money": False,
        "score": cfg.get("score", 85),
        "status": cfg.get("status", "APPROVED_FOR_WEB_PAPER"),
        "last_date": last_date,
        "rebalance_date": str(last_dt.date()),
        "target_weights": target_weights,
        "capital_allocation": capital_allocation,
        "signals": signals_df,
        "risk_mode": risk_mode,
        "warnings": warnings,
        "backtest_summary": backtest,
        "config": cfg,
    }
