"""Lógica de presentación para el panel '¿Qué hago hoy?' — sin modificar algoritmos."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

APPROVED_FOR_REAL_MONEY = False

DECISION_COLORS = {
  "BUY": "#22c55e",
  "SELL": "#ef4444",
  "HOLD": "#3b82f6",
  "AVOID": "#f97316",
  "NO DATA": "#9ca3af",
}

DECISION_ICONS = {
  "BUY": "↑",
  "SELL": "↓",
  "HOLD": "⏸",
  "AVOID": "🛡",
  "NO DATA": "⚠",
}

DECISION_MESSAGES = {
  "BUY": "Hay nuevas posiciones para comprar.",
  "SELL": "Hay posiciones que deben cerrarse.",
  "HOLD": "No hagas cambios hoy. Mantén las posiciones actuales.",
  "AVOID": "No abras nuevas posiciones. Mantén el capital disponible.",
  "NO DATA": "Falta cargar o actualizar el archivo de señales.",
}

ACTION_EXACT_TEXT = {
  "BUY": "Compra las posiciones marcadas como BUY usando el plan de órdenes inferior.",
  "SELL": "Vende las posiciones marcadas como SELL. No compres nuevas posiciones.",
  "HOLD": "No compres ni vendas. Espera al próximo rebalanceo.",
  "AVOID": "Conserva el dinero en efectivo o en el activo defensivo indicado.",
  "NO DATA": "Sube research_v17_6_current_signals.csv desde el panel lateral.",
}

SIGNAL_FILE_CANDIDATES = [
  "research_v17_6_current_signals.csv",
  "research_v17_5_2_1_current_signals.csv",
  "research_v17_5_1_current_signals.csv",
  "paper_research/tmp/current_signals.csv",
]

CASH_TICKERS = {"SHY", "CASH"}


@dataclass
class TodayDecision:
  decision: str
  headline: str
  what_to_do: str
  why: str
  signal_date: str | None
  next_rebalance: str | None
  strategy_name: str
  risk_level: str
  has_data: bool
  data_status: str = "SIN DATOS"
  hold_detail: str | None = None


@dataclass
class OrderPlanResult:
  rows: pd.DataFrame
  total_fees: float
  total_fx_cost: float
  total_estimated_cost: float
  cost_percentage_of_capital: float
  capital_total: float = 0.0
  amount_invested: float = 0.0
  cash_remaining: float = 0.0
  n_operations: int = 0
  warnings: list[str] = field(default_factory=list)
  executable: bool = True


def _find_signal_file(root: Path) -> Path | None:
  for rel in SIGNAL_FILE_CANDIDATES:
    path = root / rel
    if path.exists() and path.stat().st_size > 0:
      return path
  return None


def _signals_from_paper_snapshots(root: Path) -> pd.DataFrame | None:
  snap_path = root / "paper_research" / "data" / "signal_snapshots.csv"
  if not snap_path.exists() or snap_path.stat().st_size <= 30:
    return None
  snaps = pd.read_csv(snap_path)
  r0 = snaps[snaps["strategy"].astype(str) == "R0_BASE_S5"]
  if r0.empty:
    return None
  latest_date = r0["signal_date"].max()
  latest = r0[r0["signal_date"] == latest_date].copy()
  latest["signal"] = "BUY"
  latest["previous_weight"] = 0.0
  latest["overlay"] = "R0_BASE_S5"
  latest["strategy_source"] = "S5_XGBRANKER_LIVE"
  latest["paper_trading_start"] = latest_date
  return latest[
    [
      "ticker", "signal_date", "signal", "target_weight", "previous_weight",
      "overlay", "strategy_source", "paper_trading_start",
    ]
  ]


def load_latest_signals(root: Path | str) -> tuple[pd.DataFrame | None, str]:
  root = Path(root)
  path = _find_signal_file(root)
  if path is not None:
    return pd.read_csv(path), f"archivo: {path.name}"
  fallback = _signals_from_paper_snapshots(root)
  if fallback is not None:
    return fallback, "paper_research snapshot R0"
  return None, "sin fuente"


def compute_signal_freshness(
  signal_date: str | None,
  today: date | None = None,
) -> tuple[str, int | None]:
  """Devuelve estado de datos y días de antigüedad."""
  if not signal_date:
    return "SIN DATOS", None
  today = today or date.today()
  sig = pd.Timestamp(signal_date).date()
  age = (today - sig).days
  if age <= 7:
    return "ACTUALIZADA", age
  if age <= 14:
    return "REVISAR FECHA", age
  return "ANTIGUA", age


def next_weekly_rebalance(from_date: str | pd.Timestamp) -> str:
  d = pd.Timestamp(from_date).normalize()
  for offset in range(1, 15):
    candidate = d + pd.Timedelta(days=offset)
    if candidate.weekday() == 4:
      return f"Viernes {candidate.strftime('%d/%m')} 17:30"
  fallback = d + pd.Timedelta(days=7)
  return f"Viernes {fallback.strftime('%d/%m')} 17:30"


def derive_today_decision(signals_df: pd.DataFrame | None) -> TodayDecision:
  if signals_df is None or signals_df.empty:
    return TodayDecision(
      decision="NO DATA",
      headline=DECISION_MESSAGES["NO DATA"],
      what_to_do=ACTION_EXACT_TEXT["NO DATA"],
      why="Sin datos recientes no podemos recomendar una acción.",
      signal_date=None,
      next_rebalance=None,
      strategy_name="—",
      risk_level="MEDIUM",
      has_data=False,
      data_status="SIN DATOS",
    )

  df = signals_df.copy()
  df["signal"] = df["signal"].astype(str).str.upper()
  df["target_weight"] = pd.to_numeric(df.get("target_weight", 0), errors="coerce").fillna(0)
  df["previous_weight"] = pd.to_numeric(df.get("previous_weight", 0), errors="coerce").fillna(0)

  signal_date = str(pd.to_datetime(df["signal_date"].iloc[0]).date())
  strategy = str(df.get("overlay", df.get("strategy_source", ["S5"])).iloc[0])
  risk_exposure = pd.to_numeric(df.get("risk_exposure", 1.0), errors="coerce").fillna(1.0)
  next_reb = next_weekly_rebalance(signal_date)
  data_status, _ = compute_signal_freshness(signal_date)

  has_buy = (df["signal"] == "BUY").any()
  has_sell = df["signal"].isin(["SELL", "REDUCE"]).any()
  has_positions = (df["previous_weight"] > 0).any() | (df["target_weight"] > 0).any()

  if has_buy:
    decision = "BUY"
    why = "El modelo detectó nuevas posiciones que antes no tenías."
    risk = "MEDIUM" if float(risk_exposure.min()) >= 0.5 else "HIGH"
    hold_detail = None
  elif has_sell:
    decision = "SELL"
    why = "Algunas posiciones actuales deben cerrarse o reducirse."
    risk = "MEDIUM"
    hold_detail = None
  elif has_positions:
    decision = "HOLD"
    why = "La cartera sigue siendo válida y no hay compras nuevas urgentes."
    risk = "LOW" if float(risk_exposure.min()) >= 0.75 else "MEDIUM"
    hold_detail = (
      "DECISIÓN DE HOY: HOLD\n"
      "No hay nuevas compras.\n"
      "No vendas las posiciones existentes.\n"
      "Espera al próximo rebalanceo."
    )
  else:
    decision = "AVOID"
    why = "No hay posiciones recomendadas y el modelo prefiere esperar en efectivo."
    risk = "HIGH"
    hold_detail = (
      "DECISIÓN DE HOY: AVOID\n"
      "Mantén el capital en efectivo."
    )

  return TodayDecision(
    decision=decision,
    headline=DECISION_MESSAGES[decision],
    what_to_do=ACTION_EXACT_TEXT[decision],
    why=why,
    signal_date=signal_date,
    next_rebalance=next_reb,
    strategy_name=strategy,
    risk_level=risk,
    has_data=True,
    data_status=data_status,
    hold_detail=hold_detail,
  )


def build_human_explanations(signals_df: pd.DataFrame | None) -> list[str]:
  if signals_df is None or signals_df.empty:
    return []

  df = signals_df.copy()
  df["signal"] = df["signal"].astype(str).str.upper()
  df["target_weight"] = pd.to_numeric(df["target_weight"], errors="coerce").fillna(0)
  df["previous_weight"] = pd.to_numeric(df["previous_weight"], errors="coerce").fillna(0)
  lines: list[str] = []

  exposure = pd.to_numeric(df.get("risk_exposure", np.nan), errors="coerce").dropna()
  if len(exposure):
    exp = float(exposure.iloc[0])
    if exp >= 0.99:
      lines.append("La exposición actual al mercado es alta (cerca del 100%).")
    elif exp <= 0.40:
      lines.append(
        f"La exposición actual es defensiva ({exp:.0%}). "
        "Parte del capital se mantiene en activos más seguros."
      )
    else:
      lines.append(f"La exposición actual al mercado es del {exp:.0%}.")

  shy_rows = df[df["ticker"].astype(str).isin(CASH_TICKERS)]
  if not shy_rows.empty and float(shy_rows["target_weight"].sum()) > 0:
    shy_pct = float(shy_rows["target_weight"].sum()) * 100
    lines.append(
      f"SHY se usa como activo defensivo con un peso objetivo del {shy_pct:.1f}%."
    )

  buys = df[df["signal"] == "BUY"]
  for _, row in buys.iterrows():
    if str(row["ticker"]) in CASH_TICKERS:
      continue
    pct = float(row["target_weight"]) * 100
    lines.append(
      f"{row['ticker']} entra en la cartera con un peso objetivo del {pct:.1f}%, "
      "por eso la señal es BUY."
    )

  sells = df[df["signal"].isin(["SELL", "REDUCE"])]
  for _, row in sells.iterrows():
    lines.append(
      f"{row['ticker']} pasa de un peso positivo a cero, por eso la señal es SELL."
    )

  holds = df[df["signal"] == "HOLD"]
  for _, row in holds.iterrows():
    ticker = str(row["ticker"])
    if ticker in CASH_TICKERS:
      continue
    tw = float(row["target_weight"])
    pw = float(row["previous_weight"])
    if abs(tw - pw) < 1e-6 and tw > 0:
      lines.append(
        f"{ticker} continúa en la cartera con el mismo peso, por eso la señal es HOLD."
      )
    elif tw > pw:
      lines.append(f"{ticker} aumenta su peso en la cartera.")
    elif tw < pw:
      lines.append(f"{ticker} reduce su peso en la cartera.")

  if not buys.empty is False and not any("BUY" in l for l in lines):
    if not buys.empty:
      pass
    elif not sells.empty:
      pass
    elif (df["target_weight"] > 0).any():
      lines.append("No existen nuevas compras en esta revisión.")

  return lines[:12]


def _order_status(
  action: str,
  executable_amount: float,
  ideal_amount: float,
  whole_share_blocked: bool,
) -> str:
  action = str(action).upper()
  if action in ("AVOID",):
    return "Descartado"
  if whole_share_blocked and ideal_amount > 0:
    return "No ejecutable"
  if executable_amount <= 0 and action in ("BUY", "REDUCE"):
    return "Sin importe"
  if action == "HOLD":
    return "Mantener"
  if action == "SELL":
    return "Cerrar"
  return "Listo"


def build_order_plan(
  signals_df: pd.DataFrame | None,
  capital_eur: float = 100.0,
  fractional: bool = True,
  commission_eur: float = 0.0,
  fx_cost_pct: float = 0.0,
  max_cost_pct: float = 2.0,
  prices_eur: dict[str, float] | None = None,
) -> OrderPlanResult:
  columns = [
    "Ticker", "Acción", "Importe", "Peso objetivo", "Precio aproximado",
    "Unidades", "Coste estimado", "Estado",
  ]
  empty = OrderPlanResult(
    rows=pd.DataFrame(columns=columns),
    total_fees=0.0,
    total_fx_cost=0.0,
    total_estimated_cost=0.0,
    cost_percentage_of_capital=0.0,
    capital_total=capital_eur,
    warnings=["Sin señales para generar órdenes."],
    executable=False,
  )
  if signals_df is None or signals_df.empty or capital_eur <= 0:
    return empty

  prices_eur = prices_eur or {}
  rows = []
  total_fees = 0.0
  total_fx = 0.0
  whole_share_blocked = False
  n_ops = 0
  invested = 0.0
  cash_weight = 0.0

  df = signals_df.copy()
  df["signal"] = df["signal"].astype(str).str.upper()
  df["target_weight"] = pd.to_numeric(df["target_weight"], errors="coerce").fillna(0)

  for _, r in df.iterrows():
    ticker = str(r["ticker"])
    action = str(r["signal"])
    weight = float(r["target_weight"])
    ideal = round(capital_eur * weight, 2)
    price = prices_eur.get(ticker)

    if ticker in CASH_TICKERS:
      cash_weight += weight

    if ideal <= 0 and action not in ("SELL", "REDUCE", "AVOID"):
      continue

    row_blocked = False
    executable_amount = ideal
    units = "—"
    if price and price > 0 and not np.isnan(price):
      if fractional:
        units = f"{ideal / price:.4f}"
      else:
        share_count = math.floor(ideal / price)
        executable_amount = round(share_count * price, 2)
        units = str(share_count)
        if ideal > 0 and share_count < 1:
          whole_share_blocked = True
          row_blocked = True
    elif ideal > 0 and not fractional:
      whole_share_blocked = True
      row_blocked = True
      executable_amount = 0.0

    trade_commission = commission_eur if executable_amount > 0 and action in ("BUY", "SELL", "REDUCE") else 0.0
    fx_cost = executable_amount * (fx_cost_pct / 100.0) if executable_amount > 0 else 0.0
    row_cost = round(trade_commission + fx_cost, 2)
    total_fees += trade_commission
    total_fx += fx_cost

    if action in ("BUY", "HOLD", "REDUCE") and executable_amount > 0:
      invested += executable_amount
    if action in ("BUY", "SELL", "REDUCE") and executable_amount > 0:
      n_ops += 1

    price_txt = f"{price:.2f} €" if price and price > 0 and not np.isnan(price) else "—"
    rows.append({
      "Ticker": ticker,
      "Acción": action,
      "Importe": executable_amount if executable_amount > 0 else ideal,
      "Peso objetivo": f"{weight * 100:.1f}%",
      "Precio aproximado": price_txt,
      "Unidades": units,
      "Coste estimado": row_cost,
      "Estado": _order_status(action, executable_amount, ideal, row_blocked),
    })

  plan_df = pd.DataFrame(rows)
  total_cost = total_fees + total_fx
  cost_pct = (total_cost / capital_eur * 100.0) if capital_eur > 0 else 0.0
  cash_remaining = max(0.0, capital_eur - invested)

  warnings: list[str] = []
  executable = True
  if cost_pct > max_cost_pct:
    warnings.append(
      "NO EJECUTAR: los costes son demasiado altos para este capital."
    )
    executable = False
  if whole_share_blocked:
    warnings.append(
      "NO EJECUTABLE CON ESTE BRÓKER Y ESTE CAPITAL."
    )
    executable = False

  return OrderPlanResult(
    rows=plan_df,
    total_fees=round(total_fees, 2),
    total_fx_cost=round(total_fx, 2),
    total_estimated_cost=round(total_cost, 2),
    cost_percentage_of_capital=round(cost_pct, 2),
    capital_total=round(capital_eur, 2),
    amount_invested=round(invested, 2),
    cash_remaining=round(cash_remaining, 2),
    n_operations=n_ops,
    warnings=warnings,
    executable=executable,
  )


def filter_order_rows(df: pd.DataFrame, show_avoid: bool = False) -> pd.DataFrame:
  if df.empty:
    return df
  if show_avoid:
    return df
  return df[df["Acción"].astype(str).str.upper() != "AVOID"].copy()


def fetch_latest_prices_eur(tickers: list[str], eur_usd: float = 0.92) -> dict[str, float]:
  """Obtiene precios aproximados en EUR para la tabla de órdenes (solo UI)."""
  if not tickers:
    return {}
  try:
    import yfinance as yf

    data = yf.download(
      tickers,
      period="5d",
      progress=False,
      auto_adjust=False,
    )
    prices: dict[str, float] = {}
    if len(tickers) == 1:
      close = float(data["Close"].dropna().iloc[-1])
      prices[tickers[0]] = close * eur_usd
      return prices
    for ticker in tickers:
      try:
        close = float(data["Close"][ticker].dropna().iloc[-1])
        prices[ticker] = close * eur_usd
      except (KeyError, TypeError, IndexError):
        continue
    return prices
  except Exception:
    return {ticker: np.nan for ticker in tickers}


def strategy_status_cards() -> list[dict]:
  return [
    {
      "name": "V14 R1 Return Engine",
      "status": "PAPER CHAMPION",
      "approved_real_money": False,
    },
    {
      "name": "R0_BASE_S5",
      "status": "RESEARCH CONTROL",
      "approved_real_money": False,
    },
    {
      "name": "R2_SPY_TREND",
      "status": "PAPER RESEARCH CHALLENGER",
      "approved_real_money": False,
    },
  ]
