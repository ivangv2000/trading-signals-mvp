"""Import and validate weekly signal snapshots for V17.7 paper research."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from paper_research import CASH_ASSET, MARKET_BENCHMARK

MAX_STOCK_WEIGHT = 0.05
WEIGHT_SUM_TOL = 1e-6
LEVERAGE_TOL = 1e-6

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_GITHUB_CSV = (
  "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
)
SP500_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TradingResearchV177/1.0)"}


def _to_bool(value) -> bool:
  if isinstance(value, (bool, np.bool_)):
    return bool(value)
  if isinstance(value, str):
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes"):
      return True
    if normalized in ("false", "0", "no"):
      return False
  raise ValueError(f"No se pudo convertir a bool: {value!r}")


def validate_closure_config(config: dict) -> dict:
  """Valida research_v17_6_1_selected_config.json antes de importar."""
  closure_status = str(config.get("closure_status", ""))
  official_overlay = str(config.get("official_selected_overlay", "")).strip().upper()
  challenger = str(config.get("paper_research_challenger", "")).strip()
  approved = config.get("approved_for_real_money", True)

  if closure_status != "V17_6_CLOSED_VALIDATED":
    raise ValueError(
      f"closure_status invalido: {closure_status!r}. "
      "Se requiere V17_6_CLOSED_VALIDATED."
    )
  if official_overlay not in ("NONE", ""):
    raise ValueError(
      f"official_selected_overlay debe ser NONE, recibido: {official_overlay!r}"
    )
  if challenger != "R2_SPY_TREND":
    raise ValueError(
      f"paper_research_challenger debe ser R2_SPY_TREND, recibido: {challenger!r}"
    )
  if _to_bool(approved):
    raise ValueError("approved_for_real_money debe ser false")

  return {
    "closure_status": closure_status,
    "official_selected_overlay": "NONE",
    "paper_research_challenger": challenger,
    "approved_for_real_money": False,
  }


def file_sha256(path: Path | str, content: bytes | None = None) -> str:
  if content is not None:
    return hashlib.sha256(content).hexdigest()
  data = Path(path).read_bytes()
  return hashlib.sha256(data).hexdigest()


def make_snapshot_id(signal_date: str, strategy: str, source_hash: str) -> str:
  return f"{signal_date}|{strategy}|{source_hash[:16]}"


def _clean_symbol(symbol: str) -> str:
  return str(symbol).strip().upper().replace(".", "-")


def load_sp500_constituents() -> list[str]:
  try:
    tables = pd.read_html(SP500_WIKI_URL, storage_options=SP500_HTTP_HEADERS)
    df = tables[0]
    sym_col = next(c for c in ("Symbol", "Ticker symbol", "Ticker") if c in df.columns)
    tickers = [_clean_symbol(s) for s in df[sym_col].tolist()]
  except Exception:
    response = requests.get(
      SP500_GITHUB_CSV,
      headers=SP500_HTTP_HEADERS,
      timeout=30,
    )
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    sym_col = next(c for c in ("Symbol", "Ticker symbol", "Ticker") if c in df.columns)
    tickers = [_clean_symbol(s) for s in df[sym_col].tolist()]
  return sorted({t for t in tickers if t})


def build_r0_weights(signals_df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Series]:
  """Construye cartera R0 desde target_weight del CSV de señales."""
  if signals_df.empty:
    raise ValueError("CSV de señales vacio")

  signals_df = signals_df.copy()
  signals_df["signal_date"] = pd.to_datetime(signals_df["signal_date"])
  signal_dates = signals_df["signal_date"].dropna().unique()
  if len(signal_dates) != 1:
    raise ValueError(
      f"El snapshot debe tener una sola signal_date, encontradas: {len(signal_dates)}"
    )

  signal_date = pd.Timestamp(signal_dates[0])
  weights = (
    signals_df.groupby("ticker", as_index=True)["target_weight"]
    .sum()
    .astype(float)
    .sort_index()
  )
  weights = weights[weights > 0]

  if (weights < -WEIGHT_SUM_TOL).any():
    raise ValueError("Pesos R0 negativos detectados")

  total = float(weights.sum())
  if abs(total - 1.0) > WEIGHT_SUM_TOL:
    raise ValueError(f"Pesos R0 no suman 1: suma={total:.8f}")

  stocks = weights.drop(CASH_ASSET, errors="ignore")
  if (stocks > MAX_STOCK_WEIGHT + WEIGHT_SUM_TOL).any():
    offenders = stocks[stocks > MAX_STOCK_WEIGHT + WEIGHT_SUM_TOL]
    raise ValueError(
      f"Peso individual > {MAX_STOCK_WEIGHT:.0%}: {offenders.to_dict()}"
    )

  if float(weights.sum()) > 1.0 + LEVERAGE_TOL:
    raise ValueError("Leverage detectado en cartera R0")

  return signal_date, weights


def compute_r2_exposure(signal_date: pd.Timestamp, spy_close: pd.Series) -> float:
  """Regla R2 congelada: SMA200 desplazada 1 sesion."""
  hist = spy_close.loc[spy_close.index <= pd.Timestamp(signal_date)].copy()
  if len(hist) < 200 + 1 + 1:
    return 0.35

  sma = hist.rolling(200).mean().shift(1)
  close = float(hist.iloc[-1])
  sma_val = float(sma.iloc[-1])
  if not np.isfinite(sma_val):
    return 0.35
  return 1.0 if close > sma_val else 0.35


def build_r2_weights(r0_weights: pd.Series, exposure: float) -> pd.Series:
  """Deriva R2 exclusivamente desde pesos R0."""
  w = r0_weights.copy().astype(float)
  stocks = w.drop(CASH_ASSET, errors="ignore")
  stock_sum = float(stocks.sum())
  shy_orig = float(w.get(CASH_ASSET, 0.0))

  for ticker in stocks.index:
    w[ticker] = float(stocks[ticker]) * exposure
  w[CASH_ASSET] = shy_orig + (1.0 - exposure) * stock_sum

  total = float(w.sum())
  if abs(total - 1.0) > WEIGHT_SUM_TOL:
    raise ValueError(f"Pesos R2 no suman 1 tras overlay: suma={total:.8f}")
  return w


def build_b1_weights(members: list[str]) -> pd.Series:
  members = sorted({_clean_symbol(t) for t in members if t})
  if not members:
    raise ValueError("Lista B1 vacia")
  weight = 1.0 / len(members)
  return pd.Series({t: weight for t in members})


def snapshot_rows(
  snapshot_id: str,
  signal_date: pd.Timestamp,
  strategy: str,
  weights: pd.Series,
  source_hash: str,
  risk_exposure: float | None = None,
) -> list[dict]:
  imported_at = datetime.now(timezone.utc).isoformat()
  rows = []
  for ticker, weight in weights.items():
    if float(weight) <= 0:
      continue
    rows.append({
      "snapshot_id": snapshot_id,
      "signal_date": signal_date.strftime("%Y-%m-%d"),
      "strategy": strategy,
      "ticker": str(ticker),
      "target_weight": round(float(weight), 8),
      "source_file_hash": source_hash,
      "imported_at": imported_at,
      "risk_exposure": risk_exposure if risk_exposure is not None else np.nan,
    })
  return rows


def membership_rows(
  snapshot_id: str,
  signal_date: pd.Timestamp,
  members: list[str],
  source_hash: str,
) -> list[dict]:
  captured_at = datetime.now(timezone.utc).isoformat()
  return [
    {
      "snapshot_id": snapshot_id,
      "signal_date": signal_date.strftime("%Y-%m-%d"),
      "ticker": ticker,
      "source": "CURRENT_ASOF",
      "captured_at": captured_at,
      "source_file_hash": source_hash,
    }
    for ticker in sorted(members)
  ]


def load_closure_config(path: Path | str) -> dict:
  config = json.loads(Path(path).read_text(encoding="utf-8"))
  validate_closure_config(config)
  return config


def load_signals_csv(path: Path | str) -> pd.DataFrame:
  return pd.read_csv(path)


def prepare_import_payload(
  signals_path: Path | str,
  config_path: Path | str,
  spy_close: pd.Series | None = None,
) -> dict:
  """
  Valida inputs y prepara filas append-only para snapshots y membresia B1.
  No escribe en disco; el tracker decide si el snapshot ya existe.
  """
  config = load_closure_config(config_path)
  signals_path = Path(signals_path)
  signals_bytes = signals_path.read_bytes()
  source_hash = hashlib.sha256(signals_bytes).hexdigest()
  signals_df = pd.read_csv(StringIO(signals_bytes.decode("utf-8")))

  signal_date, r0_weights = build_r0_weights(signals_df)

  if spy_close is None:
    import yfinance as yf

    spy = yf.download(
      "SPY",
      start="2010-01-01",
      progress=False,
      auto_adjust=False,
    )
    if isinstance(spy.columns, pd.MultiIndex):
      spy.columns = spy.columns.get_level_values(0)
    spy_close = spy["Close"].dropna()

  exposure = compute_r2_exposure(signal_date, spy_close)
  r2_weights = build_r2_weights(r0_weights, exposure)

  members = load_sp500_constituents()
  b1_weights = build_b1_weights(members)
  spy_weights = pd.Series({MARKET_BENCHMARK: 1.0})

  snapshots = []
  snapshot_ids = {}
  for strategy, weights, risk_exp in (
    ("R0_BASE_S5", r0_weights, 1.0),
    ("R2_SPY_TREND", r2_weights, exposure),
    ("B1_EQUAL_WEIGHT_CURRENT_ASOF", b1_weights, np.nan),
    ("SPY", spy_weights, np.nan),
  ):
    sid = make_snapshot_id(signal_date.strftime("%Y-%m-%d"), strategy, source_hash)
    snapshot_ids[strategy] = sid
    snapshots.extend(
      snapshot_rows(sid, signal_date, strategy, weights, source_hash, risk_exp)
    )

  b1_sid = snapshot_ids["B1_EQUAL_WEIGHT_CURRENT_ASOF"]
  membership = membership_rows(b1_sid, signal_date, members, source_hash)

  return {
    "config": config,
    "source_file_hash": source_hash,
    "signal_date": signal_date,
    "r0_weights": r0_weights,
    "r2_weights": r2_weights,
    "r2_exposure": exposure,
    "snapshot_ids": snapshot_ids,
    "signal_snapshots": snapshots,
    "membership_snapshots": membership,
    "signals_path": str(signals_path),
  }
