"""Documentación técnica V14."""

from __future__ import annotations

import streamlit as st

from content.doc_utils import ROOT, fmt_metric, safe_read_csv, safe_read_json
from src.portfolio_v14 import DEFAULT_UNIVERSE, load_v14_config

V14_FORMULA = """
### Score principal (definido en código)

```
COMBINED_SCORE = 0.50 × MOM_63 + 0.50 × TREND_SCORE
```

- **MOM_63** = `close / close.shift(63) - 1` — fortaleza reciente del precio.
- **TREND_SCORE** = `ABOVE_SMA200 × 0.4 + ABOVE_SMA200 × 0.6` en `src/portfolio_v14.py`
  (equivalente a usar si el precio cierra por encima de la SMA200).
- El score **no es una probabilidad**; se usa para **ordenar** activos elegibles.
"""

V14_SECTIONS_MD = """
### A. Papel de V14
- Modelo público actual (**V14 R1 Return Engine**).
- Universo de producción: **49 activos** (`DEFAULT_UNIVERSE` en `src/portfolio_v14.py`).
- Revisión semanal (W-FRI), top 3 riesgosos + asignación defensiva SHY.
- Señal pública persistida en `data/v14_latest_signals.csv`.

### B. Datos de entrada
- OHLCV diario; frecuencia diaria; historial mínimo para features.
- Sin bfill de precios futuros; NaN excluye al activo de elegibilidad ese día.
- Universo fijo actual → **riesgo de survivorship bias** si se aplica retrospectivamente.

### C. Ranking y elegibilidad
- Comparación transversal el mismo `signal_date`.
- Filtros: MOM_63 > 0, precio ≥ SMA200, VOL_63 ≤ 45%, régimen risk-on.
- Un activo puede salir del top aunque siga subiendo si otros puntúan mejor.

### D. Construcción de cartera
- `top_n = 3`, `vol_target = 15%`, defensivo **SHY**.
- Pesos inversos a volatilidad, relleno de caja, rebalanceo semanal.
- Señales BUY/HOLD/REDUCE/SELL según cambio de peso vs semana anterior.

### E. Ejecución
- Señal **después del cierre** del viernes.
- Ejecución simulada en la **siguiente apertura** disponible.
- `cost_rate = 0.002` (costes + slippage agregados en investigación).
- Acciones fraccionadas teóricas; no es ejecución en broker real.
"""


def _render_live_signal() -> None:
    st.markdown("### G. Señal actual (lectura dinámica)")
    summary = safe_read_json("data/v14_latest_summary.json")
    signals = safe_read_csv("data/v14_latest_signals.csv")
    if not summary:
        st.info("Dato no disponible — sin snapshot V14 guardado.")
        return
    st.markdown(f"- **Fecha:** {summary.get('signal_date', '—')}")
    stats = summary.get("universe_stats") or {}
    st.markdown(f"- **Universo analizado:** {stats.get('analyzed', summary.get('number_of_positions', '—'))}")
    st.markdown(f"- **Acción global:** {summary.get('global_action', '—')}")
    st.markdown(f"- **Próxima revisión:** {summary.get('next_rebalance', 'Viernes después del cierre')}")
    if not signals.empty:
        active = signals[signals["target_weight"].fillna(0) > 0.001]
        st.dataframe(active[["ticker", "target_weight", "signal"]], hide_index=True, use_container_width=True)


def _render_historical_metrics() -> None:
    st.markdown("### H. Resultados históricos de producción (V18.3.4)")
    st.caption("BACKTEST · NO ES UNA PREVISIÓN · REUSED_TEST NO ES HOLDOUT VIRGEN · Survivorship bias: HIGH")
    metrics = safe_read_csv("research_v18_3_4_c0_vs_d2_metrics.csv")
    if metrics.empty:
        st.info("Dato no disponible — export de investigación no encontrado.")
        return
    row = metrics[metrics["strategy"] == "C0_PRODUCTION_V14"]
    if row.empty:
        st.info("Dato no disponible — fila C0 no encontrada.")
        return
    r = row.iloc[0]
    cols = st.columns(4)
    cols[0].metric("Total return", fmt_metric(r.get("total_return"), "%"))
    cols[1].metric("CAGR", fmt_metric(r.get("CAGR"), "%"))
    cols[2].metric("Sharpe", fmt_metric(r.get("sharpe")))
    cols[3].metric("Max DD", fmt_metric(r.get("max_drawdown"), "%"))
    st.warning("Estas cifras describen un backtest histórico. No implican que obtendrás la misma rentabilidad.")


def render_v14_documentation() -> None:
    cfg = load_v14_config()
    st.markdown(V14_SECTIONS_MD)
    st.markdown(V14_FORMULA)
    st.markdown(f"**Universo configurado en código:** {len(DEFAULT_UNIVERSE)} tickers de producción.")
    st.markdown(
        f"- Rebalanceo: `{cfg.get('rebalance_freq', 'W-FRI')}` · "
        f"Top N: `{cfg.get('base_engine', {}).get('top_n', 3)}` · "
        f"Vol target: `{cfg.get('base_engine', {}).get('vol_target', 0.15)}`"
    )
    _render_live_signal()
    _render_historical_metrics()
