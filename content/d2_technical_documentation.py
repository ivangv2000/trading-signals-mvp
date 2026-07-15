"""Documentación técnica D2."""

from __future__ import annotations

import streamlit as st

from content.doc_utils import fmt_metric, safe_read_csv, safe_read_json

D2_FORMULA = """
### Fórmula completa (preregistrada, sin cambios)

```
trend_quality_126 = slope × R²
```

Regresión OLS sobre **126 cierres** de `log(Close)` con `x = 0, 1, …, 125`.

```
D2_SCORE = 0.70 × baseline_rank + 0.30 × trend_quality_rank
```

- **baseline_rank:** ranking percentil del score V14 (COMBINED_SCORE).
- **trend_quality_rank:** ranking percentil de trend_quality_126 entre elegibles.
- No es una probabilidad; solo reordena activos ya elegibles por V14.
"""

D2_EXAMPLE = """
### Ejemplo sencillo

**Activo A:** subida progresiva durante seis meses → pendiente positiva y R² alto.

**Activo B:** lateral largo y salto brusco en pocos días → momentum similar pero
trayectoria menos estable → trend_quality_126 menor.

D2 intenta favorecer trayectorias más **estables**, no solo retornos recientes altos.
"""

D2_NOT_SELECTED = """
### Motivo de no selección (V18.3.4)

D2 mejoró varias métricas frente al baseline de producción, pero **no pasó todos los gates**:

| Gate | Resultado |
|------|-----------|
| **G7** Information ratio vs C0 | 0.24 < 0.30 requerido |
| **G13** PBO | 0.50 — no inferior a 0.50 |

Estado: **D2_NOT_SELECTED** · No es modelo oficial · **APPROVED_FOR_REAL_MONEY=False**
"""


def _render_comparison() -> None:
    st.markdown("### D. Comparación histórica")
    metrics = safe_read_csv("research_v18_3_4_c0_vs_d2_metrics.csv")
    if metrics.empty:
        st.info("Dato no disponible")
        return
    for sid, label in [("C0_PRODUCTION_V14", "V14"), ("D2_TREND_QUALITY", "D2")]:
        row = metrics[metrics["strategy"] == sid]
        if row.empty:
            continue
        r = row.iloc[0]
        st.markdown(
            f"**{label}:** CAGR {fmt_metric(r.get('CAGR'), '%')} · "
            f"Sharpe {fmt_metric(r.get('sharpe'))} · "
            f"Max DD {fmt_metric(r.get('max_drawdown'), '%')} · "
            f"Turnover {fmt_metric(r.get('turnover'))}"
        )
    periods = safe_read_csv("research_v18_3_4_period_and_predictive_results.csv")
    if not periods.empty:
        for pname in ("RESEARCH", "REUSED_TEST"):
            sub = periods[(periods["result_type"] == "period") & (periods["period"] == pname)]
            if not sub.empty:
                st.caption(f"Periodo {pname} — ver export completo para detalle por estrategia.")
    gates = safe_read_csv("research_v18_3_4_gate_and_reconciliation.csv")
    if not gates.empty and "G7_IR_vs_C0_at_least_0_30" in gates.columns:
        g7 = gates.iloc[0].get("G7_IR_vs_C0_at_least_0_30")
        g13 = gates.iloc[0].get("G13_PBO_below_0_50")
        st.markdown(f"- G7 pass: `{g7}` · G13 pass: `{g13}`")


def _render_shadow_tracker() -> None:
    st.markdown("### F. Shadow paper tracker (V18.3.5)")
    state = safe_read_json("paper_research/state/v18_3_5_d2_shadow_state.json")
    if not state:
        st.info("Dato no disponible")
        return
    st.markdown(f"- Paper start: `{state.get('paper_start_date')}`")
    st.markdown(f"- Primera señal permitida: `{state.get('first_permitted_signal_date')}`")
    st.markdown(f"- Estado paper: `{state.get('paper_status')}`")
    st.markdown(f"- Señales completadas: `{state.get('completed_signals', 0)}`")
    st.markdown("- Snapshots append-only con hashes e idempotencia.")
    st.markdown("- Checkpoints: 13 / 26 / 52 ejecuciones semanales (informe, no auto-reemplazo).")
    st.page_link("views/d2_shadow_research.py", label="Ver seguimiento D2", icon="🧪")


def render_d2_documentation() -> None:
    st.markdown("### A. Por qué se creó D2")
    st.markdown(
        "V14 prioriza momentum y tendencia. Dos activos pueden tener retornos similares con "
        "trayectorias distintas. D2 añade un 30% de **calidad de tendencia** al ranking."
    )
    st.markdown(D2_FORMULA)
    st.markdown(D2_EXAMPLE)
    _render_comparison()
    st.markdown(D2_NOT_SELECTED)
    _render_shadow_tracker()
