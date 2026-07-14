"""
Portfolio V6 — Blended Champion (paper trading only).

Combina trend following V4 con adaptive ensemble.
No ejecuta órdenes ni conecta brokers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "approved_v6_strategy.json"
MIN_HISTORY_DAYS = 252
RISKY_TOP_N = 4
DEFENSIVE_TOP_N = 3


def load_v6_config(path: str | Path | None = None) -> dict:
    """Lee config/approved_v6_strategy.json y devuelve dict."""
    cfg_path = Path(path) if path else CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"No se encontró configuración V6: {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _norm_idx(ix) -> pd.DatetimeIndex:
    ix = pd.DatetimeIndex(ix)
    return ix.tz_localize(None) if ix.tz else ix


def build_close_prices(data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Construye DataFrame de cierres desde datos por ticker.
    Añade CASH sintético, alinea fechas y limpia NaN razonablemente.
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
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = pd.to_numeric(d["Close"], errors="coerce")
    r1 = c.pct_change(1)
    for n in [20, 50, 100, 200]:
        d[f"SMA_{n}"] = c.rolling(n).mean()
    for span in [21, 50, 100]:
        d[f"EMA_{span}"] = c.ewm(span=span, adjust=False).mean()
    for n in [20, 60]:
        d[f"VOL_{n}"] = r1.rolling(n).std() * np.sqrt(252)
    for n in [20, 60, 120]:
        d[f"MOM_{n}"] = c.pct_change(n)
    d["MOM_COMBO"] = 0.25 * d["MOM_20"] + 0.35 * d["MOM_60"] + 0.40 * d["MOM_120"]
    d["MOM_COMBO_VOL"] = d["MOM_COMBO"] / d["VOL_60"].replace(0, np.nan)
    d["Close"] = c
    return d


def _build_features_dict(close_prices: pd.DataFrame, data_dict: dict) -> dict[str, pd.DataFrame]:
    features = {}
    for col in close_prices.columns:
        if col == "CASH":
            features["CASH"] = pd.DataFrame({"Close": 1.0}, index=close_prices.index)
            continue
        src = data_dict.get(col)
        if src is not None and not getattr(src, "empty", True):
            work = src.copy()
            if isinstance(work.columns, pd.MultiIndex):
                work.columns = work.columns.get_level_values(0)
            colmap = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
            work = work.rename(columns={c: colmap.get(str(c).lower(), c) for c in work.columns})
            if "Close" not in work.columns:
                work["Close"] = close_prices[col]
        else:
            work = pd.DataFrame({"Close": close_prices[col]}, index=close_prices.index)
        work.index = _norm_idx(work.index)
        features[col] = _add_features(work.reindex(close_prices.index).ffill())
    return features


def _reb_dates(ix, freq="W-FRI") -> pd.DatetimeIndex:
    ix = _norm_idx(ix)
    s = pd.Series(np.arange(len(ix)), index=ix)
    return pd.DatetimeIndex([g.index[-1] for _, g in s.groupby(pd.Grouper(freq=freq)) if len(g)])


def _align_date(ix, dt):
    ix, dt = _norm_idx(ix), pd.Timestamp(dt)
    if dt in ix:
        return dt
    loc = ix.get_indexer([dt], method="pad")[0]
    return ix[loc] if loc >= 0 else None


def _idx_pos(ix, dt):
    aligned = _align_date(ix, dt)
    return ix.get_loc(aligned) if aligned is not None else None


def _next_rebalance_pos(index, rlist, i):
    if i + 1 < len(rlist):
        pos = _idx_pos(index, rlist[i + 1])
        return pos if pos is not None else len(index) - 1
    return len(index) - 1


def _assign_w(wdf, i0, i1, wrow):
    i1 = min(int(i1), len(wdf) - 1)
    v = wrow.reindex(wdf.columns).fillna(0).values
    for j in range(int(i0) + 1, i1 + 1):
        wdf.iloc[j] = v


def normalize_weight_row(w, cap=None) -> pd.Series:
    w = pd.Series(w).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
    if cap is not None and cap > 0:
        for _ in range(10):
            if w.sum() <= 0:
                break
            w = w / w.sum()
            over = w > cap
            if not over.any():
                break
            excess = (w[over] - cap).sum()
            w[over] = cap
            under = ~over
            if under.any() and w[under].sum() > 0:
                w[under] = w[under] + excess * w[under] / w[under].sum()
    return w / w.sum() if w.sum() > 0 else w


def _inv_vol_w(tickers, vols, cap=0.30) -> pd.Series:
    vols = pd.Series(vols).replace(0, np.nan).dropna()
    tickers = [t for t in tickers if t in vols.index and pd.notna(vols[t])]
    if not tickers:
        return pd.Series(dtype=float)
    raw = 1 / vols[tickers].clip(lower=0.01)
    return normalize_weight_row(raw, cap=cap)


def _row_asof(df, date) -> pd.Series:
    if df is None or getattr(df, "empty", True):
        return pd.Series(dtype=float)
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.DatetimeIndex(df.index)
    date = pd.Timestamp(date)
    if date.tzinfo is not None:
        date = date.tz_localize(None)
    idx = df.index[df.index <= date]
    if len(idx) == 0:
        return pd.Series(dtype=float)
    row = df.loc[idx[-1]]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return row


def _defensive_weight_row(wr: pd.Series, defensive_asset: str = "SHY") -> pd.Series:
    wr = wr.copy()
    wr[:] = 0.0
    if defensive_asset in wr.index:
        wr[defensive_asset] = 1.0
    elif "SHY" in wr.index:
        wr["SHY"] = 1.0
    elif "CASH" in wr.index:
        wr["CASH"] = 1.0
    return wr


def _vol_scale(wdf: pd.DataFrame, close: pd.DataFrame, target=0.15, lb=20) -> pd.DataFrame:
    cols = [c for c in wdf.columns if c in close.columns]
    r = close[cols].pct_change().fillna(0)
    pr = (wdf[cols].shift(1).fillna(0) * r).sum(axis=1)
    rv = pr.rolling(lb).std() * np.sqrt(252)
    scale = (target / rv.replace(0, np.nan)).clip(0, 1).shift(1).fillna(1)
    return wdf.mul(scale, axis=0)


def calculate_trend_following_v4_weights(
    close_prices: pd.DataFrame,
    config: dict,
    features_dict: dict | None = None,
) -> pd.DataFrame:
    """
    Champion trend following V4:
    - Close > SMA slow y EMA fast > SMA slow
    - pesos inverse volatility
    - max_asset_weight y defensive_asset desde config
    """
    params = config.get("parameters", {})
    fast_ma = int(params.get("fast_ma", 50))
    slow_ma = int(params.get("slow_ma", 200))
    vol_target = float(params.get("vol_target", 0.15))
    max_w = float(params.get("max_asset_weight", 0.30))
    defensive_asset = params.get("defensive_asset", "SHY")
    rebalance_freq = params.get("rebalance_freq", "W-FRI")
    universe = [c for c in config.get("universe", []) if c in close_prices.columns and c != "CASH"]

    if features_dict is None:
        features_dict = _build_features_dict(close_prices, {})

    wdf = pd.DataFrame(0.0, index=_norm_idx(close_prices.index), columns=close_prices.columns)
    rlist = list(_reb_dates(wdf.index, rebalance_freq))
    slow_col = f"SMA_{slow_ma}"
    fast_col = f"EMA_{fast_ma}" if fast_ma in [21, 50, 100] else "EMA_50"

    for i, rd in enumerate(rlist):
        i0 = _idx_pos(wdf.index, rd)
        if i0 is None:
            continue
        rd = wdf.index[i0]
        i1 = _next_rebalance_pos(wdf.index, rlist, i)
        elig, vols = [], {}
        for t in universe:
            row = _row_asof(features_dict.get(t), rd)
            if row.empty:
                continue
            sma = _safe_float(row.get(slow_col, np.nan))
            ema = _safe_float(row.get(fast_col, np.nan))
            px = _safe_float(row.get("Close", np.nan))
            if pd.notna(sma) and pd.notna(ema) and pd.notna(px) and px > sma and ema > sma:
                elig.append(t)
                v20 = _safe_float(row.get("VOL_20", np.nan))
                v60 = _safe_float(row.get("VOL_60", np.nan))
                vols[t] = np.nanmean([v20, v60]) if pd.notna(v20) or pd.notna(v60) else np.nan
        wr = pd.Series(0.0, index=wdf.columns)
        if elig:
            for t, wt in _inv_vol_w(elig, vols, max_w).items():
                wr[t] = wt
        else:
            wr = _defensive_weight_row(wr, defensive_asset)
        _assign_w(wdf, i0, i1, wr)

    return _vol_scale(wdf, close_prices, vol_target)


def _risk_on_score(features_dict: dict, dt) -> int:
    score = 0
    for ticker in ["SPY", "QQQ"]:
        row = _row_asof(features_dict.get(ticker), dt)
        if row.empty:
            continue
        if _safe_float(row.get("Close", 0)) > _safe_float(row.get("SMA_200", np.inf)):
            score += 1
        if _safe_float(row.get("MOM_60", -1)) > 0:
            score += 1
    spy = _row_asof(features_dict.get("SPY"), dt)
    if not spy.empty and _safe_float(spy.get("Close", 0)) < _safe_float(spy.get("SMA_200", np.inf)):
        score -= 2
    return score


def calculate_adaptive_ensemble_weights(
    close_prices: pd.DataFrame,
    config: dict,
    features_dict: dict | None = None,
) -> pd.DataFrame:
    """
    Adaptive ensemble simplificado:
    - risk_on si SPY y QQQ > SMA_200
    - risk_off aumenta defensivos
    - momentum relativo + inverse volatility
    """
    params = config.get("parameters", {})
    max_w = float(params.get("max_asset_weight", 0.30))
    defensive_asset = params.get("defensive_asset", "SHY")
    universe = [c for c in config.get("universe", []) if c in close_prices.columns and c != "CASH"]
    defensive = [c for c in config.get("defensive_assets", ["SHY", "GLD", "CASH"]) if c in close_prices.columns]

    if features_dict is None:
        features_dict = _build_features_dict(close_prices, {})

    wdf = pd.DataFrame(0.0, index=_norm_idx(close_prices.index), columns=close_prices.columns)
    rlist = list(_reb_dates(wdf.index, "W-FRI"))

    for i, rd in enumerate(rlist):
        i0 = _idx_pos(wdf.index, rd)
        if i0 is None:
            continue
        rd = wdf.index[i0]
        i1 = _next_rebalance_pos(wdf.index, rlist, i)
        ros = _risk_on_score(features_dict, rd)
        wr = pd.Series(0.0, index=wdf.columns)

        if ros >= 2:
            scores = {}
            for t in universe:
                row = _row_asof(features_dict.get(t), rd)
                if row.empty:
                    continue
                val = _safe_float(row.get("MOM_COMBO_VOL", np.nan))
                if pd.isna(val):
                    val = _safe_float(row.get("MOM_60", np.nan))
                if pd.notna(val) and val > 0:
                    scores[t] = val
            if scores:
                s = pd.Series(scores).astype(float).nlargest(min(RISKY_TOP_N, len(scores)))
                vols = {}
                for t in s.index:
                    r2 = _row_asof(features_dict.get(t), rd)
                    if not r2.empty:
                        vols[t] = _safe_float(r2.get("VOL_60", np.nan))
                risky_w = _inv_vol_w(list(s.index), vols, max_w)
                risky_sum = min(0.85, risky_w.sum())
                if risky_w.sum() > 0:
                    risky_w = risky_w * (risky_sum / risky_w.sum())
                for t, wt in risky_w.items():
                    wr[t] = wt
                safe_col = defensive_asset if defensive_asset in wr.index else "CASH"
                wr[safe_col] = wr.get(safe_col, 0) + (1 - risky_sum)
            else:
                wr = _defensive_weight_row(wr, defensive_asset)
        else:
            scores = {}
            for t in defensive:
                if t == "CASH":
                    continue
                row = _row_asof(features_dict.get(t), rd)
                if row.empty:
                    continue
                val = _safe_float(row.get("MOM_60", np.nan))
                if pd.notna(val):
                    scores[t] = val
            if scores:
                s = pd.Series(scores).astype(float).nlargest(min(DEFENSIVE_TOP_N, len(scores)))
                for t in s.index:
                    wr[t] = 0.85 / len(s)
                wr["CASH"] = wr.get("CASH", 0) + 0.15
            else:
                wr = _defensive_weight_row(wr, defensive_asset)

        wr = normalize_weight_row(wr, cap=max_w)
        _assign_w(wdf, i0, i1, wr)

    return wdf


def calculate_blended_v6_weights(
    close_prices: pd.DataFrame,
    config: dict,
    features_dict: dict | None = None,
) -> pd.DataFrame:
    """Blend 50/50 (o según config) de V4 + adaptive, normalizado y capado."""
    blend = config.get("blend", {})
    alpha = float(blend.get("champion_trend_following_v4_weight", 0.5))
    beta = float(blend.get("adaptive_ensemble_weight", 1.0 - alpha))
    total = alpha + beta
    if total <= 0:
        alpha, beta = 0.5, 0.5
        total = 1.0
    alpha, beta = alpha / total, beta / total

    params = config.get("parameters", {})
    max_w = float(params.get("max_asset_weight", 0.30))
    defensive_asset = params.get("defensive_asset", "SHY")

    if features_dict is None:
        features_dict = _build_features_dict(close_prices, {})

    w_v4 = calculate_trend_following_v4_weights(close_prices, config, features_dict)
    w_ad = calculate_adaptive_ensemble_weights(close_prices, config, features_dict)
    mix = alpha * w_v4.reindex(close_prices.index).fillna(0) + beta * w_ad.reindex(close_prices.index).fillna(0)
    mix = mix.clip(lower=0)

    out = pd.DataFrame(index=mix.index, columns=mix.columns, dtype=float)
    for dt in mix.index:
        row = normalize_weight_row(mix.loc[dt], cap=max_w)
        residual = 1.0 - row.sum()
        if residual > 1e-6:
            safe = defensive_asset if defensive_asset in row.index else "CASH"
            row[safe] = row.get(safe, 0) + residual
        if row.sum() > 0:
            row = row / row.sum()
        out.loc[dt] = row

    return out


def get_required_tickers(config: dict | None = None) -> list[str]:
    cfg = config or load_v6_config()
    tickers = set(cfg.get("universe", []))
    tickers.update(cfg.get("defensive_assets", []))
    tickers.update(["SPY", "QQQ"])
    tickers.discard("CASH")
    return sorted(tickers)


def get_current_v6_portfolio_signal(
    data_dict: dict[str, pd.DataFrame],
    capital: float = 10000,
    config: dict | None = None,
) -> dict:
    """
    Calcula pesos actuales del portfolio V6 (paper trading).
    No ejecuta órdenes ni conecta broker.
    """
    cfg = config or load_v6_config()
    warnings = list(cfg.get("warnings", []))
    backtest = cfg.get("backtest_summary", {})

    try:
        close_prices = build_close_prices(data_dict)
    except Exception as exc:
        return {
            "error": f"No hay datos suficientes: {exc}",
            "strategy_name": cfg.get("strategy_name", "Blended Champion V6"),
            "approved_for_web_paper": cfg.get("approved_for_web_paper", True),
            "approved_for_real_money": False,
            "warnings": warnings,
        }

    if len(close_prices) < MIN_HISTORY_DAYS:
        return {
            "error": f"Se necesitan al menos {MIN_HISTORY_DAYS} días de historia. Hay {len(close_prices)}.",
            "strategy_name": cfg.get("strategy_name", "Blended Champion V6"),
            "approved_for_web_paper": cfg.get("approved_for_web_paper", True),
            "approved_for_real_money": False,
            "warnings": warnings,
        }

    features_dict = _build_features_dict(close_prices, data_dict)
    weights_df = calculate_blended_v6_weights(close_prices, cfg, features_dict)
    last_row = weights_df.iloc[-1].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
    if last_row.sum() <= 0:
        defensive = cfg.get("parameters", {}).get("defensive_asset", "SHY")
        last_row = _defensive_weight_row(last_row, defensive)

    if last_row.sum() > 0:
        last_row = last_row / last_row.sum()

    weights = {k: round(float(v), 4) for k, v in last_row.items() if v > 1e-6}
    if "CASH" not in weights:
        weights["CASH"] = 0.0
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        scale = 1.0 / weight_sum if weight_sum > 0 else 1.0
        weights = {k: round(v * scale, 4) for k, v in weights.items()}

    capital_allocation = {k: round(capital * w, 2) for k, w in weights.items() if w > 1e-6}

    mdd = backtest.get("max_drawdown", -34.91)
    risk_message = (
        f"Paper trading experimental. Peor caída histórica simulada: {mdd}%. "
        "No usar dinero real."
    )

    last_date = str(close_prices.index[-1].date()) if len(close_prices) else "N/A"

    return {
        "strategy_name": cfg.get("strategy_name", "Blended Champion V6"),
        "strategy_id": cfg.get("strategy_id", "blended_champion_v6_alpha_0_5"),
        "approved_for_web_paper": bool(cfg.get("approved_for_web_paper", True)),
        "approved_for_real_money": False,
        "final_score_v6": cfg.get("final_score_v6", 100),
        "last_date": last_date,
        "weights": weights,
        "capital_allocation": capital_allocation,
        "risk_message": risk_message,
        "warnings": warnings,
        "backtest_summary": backtest,
        "config": cfg,
    }
