"""Forward paper tracker for R0, R2, B1 and SPY — append-only, no backfill."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from paper_research import (
  APPROVED_FOR_REAL_MONEY,
  AUDIT_VERSION,
  CASH_ASSET,
  COST_RATE,
  DATA_DIR,
  INITIAL_CAPITAL,
  PAPER_TRADING_START,
  STATE_DIR,
  STATE_FILE,
  STRATEGY_B1,
  STRATEGY_R0,
  STRATEGY_R2,
  STRATEGY_SPY,
  TRACKED_STRATEGIES,
)
from paper_research.import_signal_snapshot import (
  make_snapshot_id,
  prepare_import_payload,
)
from paper_research.paper_metrics import compute_drawdown, metrics_table
from paper_research.paper_tracker_tests import run_paper_tracker_tests


class ForwardPaperTracker:
  SIGNAL_SNAPSHOTS = DATA_DIR / "signal_snapshots.csv"
  EXECUTIONS = DATA_DIR / "executions.csv"
  DAILY_EQUITY = DATA_DIR / "daily_equity.csv"
  CURRENT_POSITIONS = DATA_DIR / "current_positions.csv"
  MEMBERSHIP_SNAPSHOTS = DATA_DIR / "membership_snapshots.csv"

  CSV_SCHEMAS = {
    SIGNAL_SNAPSHOTS: [
      "snapshot_id", "signal_date", "strategy", "ticker", "target_weight",
      "source_file_hash", "imported_at", "risk_exposure",
    ],
    EXECUTIONS: [
      "execution_id", "snapshot_id", "strategy", "signal_date", "execution_date",
      "ticker", "target_weight", "shares", "price_open", "notional", "cost_usd",
    ],
    DAILY_EQUITY: [
      "date", "strategy", "equity", "cash_component", "invested_pct",
      "daily_return", "cumulative_return", "costs_cumulative",
    ],
    CURRENT_POSITIONS: [
      "as_of_date", "strategy", "ticker", "shares", "price_close",
      "market_value", "target_weight", "weight_actual",
    ],
    MEMBERSHIP_SNAPSHOTS: [
      "snapshot_id", "signal_date", "ticker", "source", "captured_at",
      "source_file_hash",
    ],
  }

  def __init__(self, root: Path | None = None):
    self.root = root or Path(__file__).resolve().parent.parent
    self._ensure_layout()

  def _ensure_layout(self) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    for path, columns in self.CSV_SCHEMAS.items():
      if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
    if not STATE_FILE.exists():
      self._write_state(self._default_state())

  def _default_state(self) -> dict:
    return {
      "version": AUDIT_VERSION,
      "approved_for_real_money": False,
      "paper_trading_start": PAPER_TRADING_START,
      "closure_validated": False,
      "imported_snapshot_ids": [],
      "last_equity_date": None,
      "last_signal_date": None,
      "first_execution_date": {},
      "last_updated": None,
      "n_observations": 0,
    }

  def _read_state(self) -> dict:
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))

  def _write_state(self, state: dict) -> None:
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

  def _read_csv(self, path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
      return pd.DataFrame(columns=self.CSV_SCHEMAS[path])
    return pd.read_csv(path)

  def _append_csv(self, path: Path, rows: list[dict]) -> int:
    if not rows:
      return 0
    df_new = pd.DataFrame(rows)
    for col in self.CSV_SCHEMAS[path]:
      if col not in df_new.columns:
        df_new[col] = np.nan
    df_new = df_new[self.CSV_SCHEMAS[path]]
    if path.exists() and path.stat().st_size > 0:
      df_new.to_csv(path, mode="a", header=False, index=False)
    else:
      df_new.to_csv(path, index=False)
    return len(df_new)

  @staticmethod
  def _next_trading_day(index: pd.DatetimeIndex, after: pd.Timestamp) -> pd.Timestamp | None:
    future = index[index > pd.Timestamp(after)]
    return pd.Timestamp(future[0]) if len(future) else None

  def _download_prices(
    self,
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
  ) -> dict[str, pd.DataFrame]:
    tickers = sorted({t for t in tickers if t})
    if not tickers:
      return {}
    end = end or pd.Timestamp.today().normalize()
    raw = yf.download(
      tickers,
      start=(pd.Timestamp(start) - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
      end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
      progress=False,
      auto_adjust=False,
      group_by="ticker",
    )
    out: dict[str, pd.DataFrame] = {}
    if len(tickers) == 1:
      ticker = tickers[0]
      df = raw.copy()
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
      out[ticker] = df
      return out

    for ticker in tickers:
      try:
        df = raw[ticker].dropna(how="all")
        if not df.empty:
          out[ticker] = df
      except (KeyError, TypeError):
        continue
    return out

  def import_signal_snapshot(
    self,
    signals_path: Path | str,
    config_path: Path | str,
  ) -> dict:
    """Importa snapshot semanal; no duplica snapshot_id existente."""
    test_report = run_paper_tracker_tests(self, config_path=config_path, pre_import=True)
    failed = test_report[~test_report["pass"]]
    if len(failed):
      raise AssertionError(
        "Tests pre-import fallidos: " + ", ".join(failed["test"].tolist())
      )

    payload = prepare_import_payload(signals_path, config_path)
    existing = self._read_csv(self.SIGNAL_SNAPSHOTS)
    existing_ids = set(existing["snapshot_id"].astype(str)) if not existing.empty else set()

    new_ids = []
    duplicate_ids = []
    for strategy, sid in payload["snapshot_ids"].items():
      if sid in existing_ids:
        duplicate_ids.append(sid)
      else:
        new_ids.append(sid)

    if duplicate_ids and not new_ids:
      return {
        "status": "duplicate",
        "message": "Snapshot ya importado",
        "duplicate_snapshot_ids": duplicate_ids,
        "signal_date": payload["signal_date"].strftime("%Y-%m-%d"),
      }

    if new_ids:
      new_snapshots = [
        row for row in payload["signal_snapshots"]
        if row["snapshot_id"] in new_ids
      ]
      self._append_csv(self.SIGNAL_SNAPSHOTS, new_snapshots)

      new_members = [
        row for row in payload["membership_snapshots"]
        if row["snapshot_id"] in new_ids
      ]
      if new_members:
        self._append_csv(self.MEMBERSHIP_SNAPSHOTS, new_members)

    state = self._read_state()
    state["closure_validated"] = True
    state["last_signal_date"] = payload["signal_date"].strftime("%Y-%m-%d")
    state["source_file_hash"] = payload["source_file_hash"]
    imported = set(state.get("imported_snapshot_ids", []))
    imported.update(new_ids)
    state["imported_snapshot_ids"] = sorted(imported)
    self._write_state(state)

    post_report = run_paper_tracker_tests(self, config_path=config_path)
    failed_post = post_report[~post_report["pass"]]
    if len(failed_post):
      raise AssertionError(
        "Tests post-import fallidos: " + ", ".join(failed_post["test"].tolist())
      )

    return {
      "status": "imported" if new_ids else "duplicate",
      "message": "Snapshot importado" if new_ids else "Snapshot ya importado",
      "signal_date": payload["signal_date"].strftime("%Y-%m-%d"),
      "r2_exposure": payload["r2_exposure"],
      "new_snapshot_ids": new_ids,
      "duplicate_snapshot_ids": duplicate_ids,
      "n_positions_r0": len(payload["r0_weights"]),
    }

  def _pending_snapshots(self) -> pd.DataFrame:
    snaps = self._read_csv(self.SIGNAL_SNAPSHOTS)
    execs = self._read_csv(self.EXECUTIONS)
    if snaps.empty:
      return pd.DataFrame()
    executed_ids = set()
    if not execs.empty:
      executed_ids = set(execs["snapshot_id"].astype(str).unique())
    pending_ids = sorted(set(snaps["snapshot_id"].astype(str)) - executed_ids)
    if not pending_ids:
      return pd.DataFrame()
    return snaps[snaps["snapshot_id"].astype(str).isin(pending_ids)].copy()

  def _execute_snapshot(
    self,
    snapshot_id: str,
    snaps: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    state: dict,
  ) -> tuple[list[dict], list[dict], dict]:
    rows = snaps[snaps["snapshot_id"].astype(str) == snapshot_id]
    if rows.empty:
      return [], [], state

    strategy = str(rows.iloc[0]["strategy"])
    signal_date = pd.Timestamp(rows.iloc[0]["signal_date"])
    weights = rows.set_index("ticker")["target_weight"].astype(float)

    ref_ticker = STRATEGY_SPY if strategy == STRATEGY_SPY else next(iter(weights.index))
    ref_df = price_data.get(ref_ticker)
    if ref_df is None or ref_df.empty:
      return [], [], state

    exec_date = self._next_trading_day(ref_df.index, signal_date)
    if exec_date is None:
      return [], [], state

    first_dates = state.get("first_execution_date", {})
    if strategy not in first_dates:
      first_dates[strategy] = exec_date.strftime("%Y-%m-%d")
    state["first_execution_date"] = first_dates

    equity_rows: list[dict] = []
    exec_rows: list[dict] = []
    capital = INITIAL_CAPITAL
    total_cost = 0.0

    shares_map: dict[str, float] = {}
    for ticker, weight in weights.items():
      if weight <= 0:
        continue
      df = price_data.get(ticker)
      if df is None or exec_date not in df.index:
        continue
      px = float(df.loc[exec_date, "Open"])
      if not np.isfinite(px) or px <= 0:
        continue
      notional = capital * float(weight)
      cost = notional * COST_RATE
      total_cost += cost
      eff = notional - cost
      shares = eff / px
      shares_map[ticker] = shares
      exec_rows.append({
        "execution_id": f"{snapshot_id}|{ticker}|{exec_date.strftime('%Y-%m-%d')}",
        "snapshot_id": snapshot_id,
        "strategy": strategy,
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "execution_date": exec_date.strftime("%Y-%m-%d"),
        "ticker": ticker,
        "target_weight": round(float(weight), 8),
        "shares": round(float(shares), 8),
        "price_open": round(px, 6),
        "notional": round(notional, 4),
        "cost_usd": round(cost, 4),
      })

    if not shares_map:
      return [], [], state

    close_date = exec_date
    equity = 0.0
    shy_value = 0.0
    for ticker, shares in shares_map.items():
      df = price_data[ticker]
      px_close = float(df.loc[close_date, "Close"])
      mv = shares * px_close
      equity += mv
      if ticker == CASH_ASSET:
        shy_value = mv

    invested_pct = (1.0 - shy_value / equity) * 100.0 if equity > 0 else 0.0
    equity_rows.append({
      "date": close_date.strftime("%Y-%m-%d"),
      "strategy": strategy,
      "equity": round(equity, 4),
      "cash_component": round(shy_value, 4),
      "invested_pct": round(invested_pct, 4),
      "daily_return": 0.0,
      "cumulative_return": round((equity / INITIAL_CAPITAL - 1.0) * 100.0, 4),
      "costs_cumulative": round(total_cost, 4),
    })

    return exec_rows, equity_rows, state

  def _positions_from_shares(
    self,
    strategy: str,
    as_of_date: pd.Timestamp,
    shares_map: dict[str, float],
    weights: pd.Series,
    price_data: dict[str, pd.DataFrame],
  ) -> list[dict]:
    rows = []
    equity = 0.0
    values = {}
    for ticker, shares in shares_map.items():
      df = price_data.get(ticker)
      if df is None or as_of_date not in df.index:
        continue
      px = float(df.loc[as_of_date, "Close"])
      mv = shares * px
      values[ticker] = mv
      equity += mv

    for ticker, mv in values.items():
      rows.append({
        "as_of_date": as_of_date.strftime("%Y-%m-%d"),
        "strategy": strategy,
        "ticker": ticker,
        "shares": round(float(shares_map[ticker]), 8),
        "price_close": round(float(price_data[ticker].loc[as_of_date, "Close"]), 6),
        "market_value": round(float(mv), 4),
        "target_weight": round(float(weights.get(ticker, 0.0)), 8),
        "weight_actual": round(float(mv / equity), 8) if equity > 0 else 0.0,
      })
    return rows

  def update_daily_prices(self, end_date: pd.Timestamp | None = None) -> dict:
    """Valoracion diaria forward; ejecuta snapshots pendientes en next open."""
    end_date = pd.Timestamp(end_date or pd.Timestamp.today().normalize())
    state = self._read_state()
    snaps = self._read_csv(self.SIGNAL_SNAPSHOTS)
    if snaps.empty:
      return {"status": "empty", "message": "No hay snapshots importados"}

    tickers = sorted(snaps["ticker"].astype(str).unique().tolist())
    if STRATEGY_SPY not in tickers:
      tickers.append(STRATEGY_SPY)

    first_signal = pd.Timestamp(snaps["signal_date"].min())
    prices = self._download_prices(tickers, start=first_signal, end=end_date)

    pending = self._pending_snapshots()
    exec_added = 0
    equity_added = 0

    if not pending.empty:
      for snapshot_id in sorted(pending["snapshot_id"].astype(str).unique()):
        exec_rows, eq_rows, state = self._execute_snapshot(
          snapshot_id, snaps, prices, state
        )
        exec_added += self._append_csv(self.EXECUTIONS, exec_rows)
        equity_added += self._append_csv(self.DAILY_EQUITY, eq_rows)

    daily = self._read_csv(self.DAILY_EQUITY)
    if daily.empty:
      self._write_state(state)
      return {
        "status": "waiting_execution",
        "message": "Sin ejecuciones aun; esperando apertura posterior a la senal",
        "executions_added": exec_added,
      }

    daily["date"] = pd.to_datetime(daily["date"])
    last_date = daily["date"].max()
    if last_date >= end_date:
      self._write_state(state)
      return {
        "status": "up_to_date",
        "message": f"Ya valorado hasta {last_date.date()}",
        "last_equity_date": last_date.strftime("%Y-%m-%d"),
      }

    execs = self._read_csv(self.EXECUTIONS)
    new_equity_rows = []
    new_position_rows = []

    for strategy in TRACKED_STRATEGIES:
      strat_execs = execs[execs["strategy"].astype(str) == strategy]
      if strat_execs.empty:
        continue

      latest_snapshot = (
        strat_execs.sort_values("execution_date")
        .iloc[-1]["snapshot_id"]
      )
      strat_snaps = snaps[snaps["snapshot_id"].astype(str) == str(latest_snapshot)]
      weights = strat_snaps.set_index("ticker")["target_weight"].astype(float)

      shares_map = (
        strat_execs.groupby("ticker")["shares"].sum().astype(float).to_dict()
      )
      strat_daily = daily[daily["strategy"].astype(str) == strategy].copy()
      strat_start = strat_daily["date"].max()
      costs_cum = float(strat_daily["costs_cumulative"].iloc[-1]) if not strat_daily.empty else 0.0
      prev_equity = float(strat_daily["equity"].iloc[-1]) if not strat_daily.empty else INITIAL_CAPITAL

      ref_ticker = next(iter(shares_map))
      ref_df = prices.get(ref_ticker)
      if ref_df is None:
        continue

      future_dates = ref_df.index[(ref_df.index > strat_start) & (ref_df.index <= end_date)]
      existing_dates = set(strat_daily["date"].astype(str)) if not strat_daily.empty else set()

      for dt in future_dates:
        if dt.strftime("%Y-%m-%d") in existing_dates:
          continue
        equity = 0.0
        shy_value = 0.0
        for ticker, shares in shares_map.items():
          df = prices.get(ticker)
          if df is None or dt not in df.index:
            continue
          mv = shares * float(df.loc[dt, "Close"])
          equity += mv
          if ticker == CASH_ASSET:
            shy_value = mv
        if equity <= 0:
          continue
        daily_ret = (equity / prev_equity - 1.0) * 100.0 if prev_equity > 0 else 0.0
        invested_pct = (1.0 - shy_value / equity) * 100.0 if equity > 0 else 0.0
        new_equity_rows.append({
          "date": dt.strftime("%Y-%m-%d"),
          "strategy": strategy,
          "equity": round(equity, 4),
          "cash_component": round(shy_value, 4),
          "invested_pct": round(invested_pct, 4),
          "daily_return": round(daily_ret, 6),
          "cumulative_return": round((equity / INITIAL_CAPITAL - 1.0) * 100.0, 4),
          "costs_cumulative": round(costs_cum, 4),
        })
        prev_equity = equity
        existing_dates.add(dt.strftime("%Y-%m-%d"))

      if future_dates.any():
        last_dt = future_dates[-1]
        new_position_rows.extend(
          self._positions_from_shares(
            strategy, pd.Timestamp(last_dt), shares_map, weights, prices
          )
        )

    equity_added += self._append_csv(self.DAILY_EQUITY, new_equity_rows)

    if new_position_rows:
      pos_df = self._read_csv(self.CURRENT_POSITIONS)
      if pos_df.empty:
        pd.DataFrame(new_position_rows).to_csv(self.CURRENT_POSITIONS, index=False)
      else:
        pos_df = pd.concat([pos_df, pd.DataFrame(new_position_rows)], ignore_index=True)
        pos_df = pos_df.drop_duplicates(
          subset=["as_of_date", "strategy", "ticker"], keep="last"
        )
        pos_df.to_csv(self.CURRENT_POSITIONS, index=False)

    daily = self._read_csv(self.DAILY_EQUITY)
    if not daily.empty:
      state["last_equity_date"] = str(daily["date"].max())
      state["n_observations"] = int(len(daily))

    self._write_state(state)

    test_report = run_paper_tracker_tests(self)
    print(test_report.to_string(index=False))

    return {
      "status": "updated",
      "executions_added": exec_added,
      "equity_rows_added": equity_added,
      "last_equity_date": state.get("last_equity_date"),
      "n_observations": state.get("n_observations", 0),
    }

  def get_positions(self, strategy: str | None = None) -> pd.DataFrame:
    pos = self._read_csv(self.CURRENT_POSITIONS)
    if pos.empty:
      return pos
    if strategy:
      pos = pos[pos["strategy"].astype(str) == strategy]
    return pos.sort_values(["strategy", "market_value"], ascending=[True, False])

  def get_daily_equity(self, strategy: str | None = None) -> pd.DataFrame:
    df = self._read_csv(self.DAILY_EQUITY)
    if strategy and not df.empty:
      df = df[df["strategy"].astype(str) == strategy]
    if not df.empty:
      df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["strategy", "date"])

  def get_forward_metrics(self) -> pd.DataFrame:
    daily = self.get_daily_equity()
    execs = self._read_csv(self.EXECUTIONS)
    return metrics_table(daily, execs, TRACKED_STRATEGIES)

  def get_r2_exposure_latest(self) -> float | None:
    snaps = self._read_csv(self.SIGNAL_SNAPSHOTS)
    if snaps.empty:
      return None
    r2 = snaps[snaps["strategy"].astype(str) == STRATEGY_R2]
    if r2.empty or "risk_exposure" not in r2.columns:
      return None
    latest = r2.sort_values("signal_date").iloc[-1]
    val = latest.get("risk_exposure")
    return float(val) if pd.notna(val) else None

  def export_results(self, out_dir: Path | str | None = None) -> Path:
    out_dir = Path(out_dir or self.root / "paper_research" / "exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "V17_7_PAPER_EXPORT.zip"

    metrics = self.get_forward_metrics()
    metrics.to_csv(out_dir / "paper_forward_metrics.csv", index=False)
    self.get_daily_equity().to_csv(out_dir / "paper_daily_equity.csv", index=False)
    self.get_positions().to_csv(out_dir / "paper_current_positions.csv", index=False)
    self._read_csv(self.EXECUTIONS).to_csv(out_dir / "paper_executions.csv", index=False)
    self._read_csv(self.SIGNAL_SNAPSHOTS).to_csv(
      out_dir / "paper_signal_snapshots.csv", index=False
    )

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
      for name in (
        "paper_forward_metrics.csv",
        "paper_daily_equity.csv",
        "paper_current_positions.csv",
        "paper_executions.csv",
        "paper_signal_snapshots.csv",
      ):
        p = out_dir / name
        if p.exists():
          zf.write(p, p.name)
    return zip_path

  def run_tests(self, config_path: Path | str | None = None) -> pd.DataFrame:
    return run_paper_tracker_tests(self, config_path=config_path)
