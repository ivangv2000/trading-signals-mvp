"""Forward-only metrics for V17.7 paper portfolios."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _annualize_factor(daily_returns: pd.Series) -> float:
  return 252.0


def compute_drawdown(equity: pd.Series) -> pd.Series:
  peak = equity.cummax()
  return (equity / peak - 1.0) * 100.0


def compute_forward_metrics(
  equity_df: pd.DataFrame,
  benchmark_equity: pd.Series | None = None,
  min_days_for_cagr: int = 30,
) -> dict:
  """
  Calcula metricas forward desde daily_equity.
  equity_df columns: date, strategy, equity, invested_pct, costs_cumulative
  """
  if equity_df.empty:
    return {
      "n_observations": 0,
      "cumulative_return_pct": np.nan,
      "CAGR_pct": np.nan,
      "annual_volatility_pct": np.nan,
      "sharpe": np.nan,
      "max_drawdown_pct": np.nan,
      "excess_return_vs_benchmark_pct": np.nan,
      "information_ratio_vs_benchmark": np.nan,
      "pct_mean_invested": np.nan,
      "costs_cumulative_usd": np.nan,
      "n_rebalances": 0,
    }

  equity_df = equity_df.copy()
  equity_df["date"] = pd.to_datetime(equity_df["date"])
  equity_df = equity_df.sort_values("date")
  equity = equity_df.set_index("date")["equity"].astype(float)
  initial = float(equity.iloc[0])
  final = float(equity.iloc[-1])
  n_obs = len(equity)

  daily_returns = equity.pct_change().dropna()
  ann_factor = _annualize_factor(daily_returns)

  cumulative_return_pct = (final / initial - 1.0) * 100.0 if initial > 0 else np.nan

  if n_obs >= min_days_for_cagr and initial > 0 and final > 0:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = ((final / initial) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else np.nan
  else:
    cagr = np.nan

  vol = float(daily_returns.std() * math.sqrt(ann_factor) * 100.0) if len(daily_returns) else np.nan
  sharpe = (
    float(daily_returns.mean() / daily_returns.std() * math.sqrt(ann_factor))
    if len(daily_returns) and daily_returns.std() > 0
    else np.nan
  )

  dd = compute_drawdown(equity)
  max_dd = float(dd.min()) if len(dd) else np.nan

  excess_return = np.nan
  information_ratio = np.nan
  if benchmark_equity is not None and not benchmark_equity.empty:
    bench = benchmark_equity.reindex(equity.index).ffill().dropna()
    aligned = equity.reindex(bench.index).dropna()
    bench = bench.reindex(aligned.index)
    if len(aligned) >= 2 and float(bench.iloc[0]) > 0:
      strat_ret = aligned.pct_change().dropna()
      bench_ret = bench.pct_change().dropna()
      common = strat_ret.index.intersection(bench_ret.index)
      strat_ret = strat_ret.reindex(common).dropna()
      bench_ret = bench_ret.reindex(common).dropna()
      excess_return = (
        (float(aligned.iloc[-1]) / float(aligned.iloc[0]))
        / (float(bench.iloc[-1]) / float(bench.iloc[0]))
        - 1.0
      ) * 100.0
      active = strat_ret - bench_ret.reindex(strat_ret.index)
      if active.std() > 0:
        information_ratio = float(active.mean() / active.std() * math.sqrt(ann_factor))

  invested = equity_df["invested_pct"].astype(float)
  costs = equity_df["costs_cumulative"].astype(float)

  return {
    "n_observations": int(n_obs),
    "cumulative_return_pct": round(cumulative_return_pct, 4),
    "CAGR_pct": round(cagr, 4) if np.isfinite(cagr) else np.nan,
    "annual_volatility_pct": round(vol, 4) if np.isfinite(vol) else np.nan,
    "sharpe": round(sharpe, 4) if np.isfinite(sharpe) else np.nan,
    "max_drawdown_pct": round(max_dd, 4) if np.isfinite(max_dd) else np.nan,
    "excess_return_vs_benchmark_pct": (
      round(excess_return, 4) if np.isfinite(excess_return) else np.nan
    ),
    "information_ratio_vs_benchmark": (
      round(information_ratio, 4) if np.isfinite(information_ratio) else np.nan
    ),
    "pct_mean_invested": round(float(invested.mean()), 4) if len(invested) else np.nan,
    "costs_cumulative_usd": round(float(costs.iloc[-1]), 4) if len(costs) else 0.0,
    "n_rebalances": int(equity_df.attrs.get("n_rebalances", 0)),
  }


def metrics_table(
  daily_equity: pd.DataFrame,
  executions: pd.DataFrame,
  strategies: list[str],
  benchmark_strategy: str = "B1_EQUAL_WEIGHT_CURRENT_ASOF",
) -> pd.DataFrame:
  rows = []
  bench_eq = None
  if not daily_equity.empty:
    bench_rows = daily_equity[
      daily_equity["strategy"].astype(str) == benchmark_strategy
    ]
    if not bench_rows.empty:
      bench_eq = (
        bench_rows.assign(date=pd.to_datetime(bench_rows["date"]))
        .sort_values("date")
        .set_index("date")["equity"]
        .astype(float)
      )

  for strategy in strategies:
    strat_rows = daily_equity[daily_equity["strategy"].astype(str) == strategy].copy()
    n_reb = 0
    if not executions.empty:
      n_reb = int(
        executions[executions["strategy"].astype(str) == strategy]["snapshot_id"]
        .nunique()
      )
    strat_rows.attrs = {"n_rebalances": n_reb}
    bench = bench_eq if strategy not in (benchmark_strategy, "SPY") else None
    metrics = compute_forward_metrics(strat_rows, benchmark_equity=bench)
    metrics["strategy"] = strategy
    rows.append(metrics)
  return pd.DataFrame(rows)
