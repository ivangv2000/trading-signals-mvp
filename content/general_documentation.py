"""Visión general y flujo del sistema."""

from __future__ import annotations

OVERVIEW_SIMPLE = """
### Resumen sencillo

Este proyecto genera **señales cuantitativas semanales** para una cartera de ETFs y acciones.
No predice el precio exacto de mañana: **ordena activos** según reglas objetivas y construye
una cartera objetivo.

- **Modelo público:** V14 R1 Return Engine (49 activos de producción).
- **Investigación:** D2 Trend Quality (experimento en shadow paper, no oficial).
- **Paper trading:** simulación sin dinero real ni ejecución de órdenes.
- **Señal del modelo ≠ acción personal:** tú decides cómo aplicar la señal a tu cartera.

No se promete rentabilidad. Los resultados históricos son diagnósticos, no garantías.
"""

OVERVIEW_TECHNICAL = """
### Detalle técnico

| Concepto | Significado |
|----------|-------------|
| Señal cuantitativa | Regla reproducible que asigna pesos objetivo tras el cierre semanal |
| Rebalanceo W-FRI | La cartera se revisa los viernes tras el cierre |
| Ranking transversal | Los activos se comparan entre sí el mismo día |
| Modelo vs usuario | V14 emite pesos objetivo; la web traduce a BUY/HOLD/SELL según tu situación |
| Investigación | D2 comparte universo y ledger, cambia solo el ranking |

**Por qué paper trading:** permite validar ejecución, costes y señales prospectivas sin riesgo real.
"""

FLOW_STEPS = [
    ("Datos diarios", "OHLCV por ticker, sin rellenar huecos con información futura"),
    ("Features", "Momentum, tendencia, volatilidad, régimen de mercado"),
    ("Ranking", "Orden transversal por score o rank compuesto"),
    ("Cartera objetivo", "Top N + vol target + activo defensivo SHY"),
    ("Señal semanal", "Pesos y etiquetas BUY/HOLD/REDUCE/SELL"),
    ("Ejecución siguiente apertura", "Simulación en la apertura posterior al viernes"),
    ("Ledger", "Targets, ejecuciones, retornos diarios, equity"),
    ("Métricas", "CAGR, Sharpe, drawdown, costes, IC, gates"),
    ("Paper trading", "Seguimiento forward sin dinero real"),
]


def render_flow_html() -> str:
    cards = []
    for i, (title, desc) in enumerate(FLOW_STEPS):
        arrow = '<div class="doc-flow-arrow">→</div>' if i < len(FLOW_STEPS) - 1 else ""
        cards.append(
            f'<div class="doc-flow-step"><div class="doc-flow-step-title">{title}</div>'
            f'<div class="doc-flow-step-desc">{desc}</div></div>{arrow}'
        )
    return '<div class="doc-flow-track">' + "".join(cards) + "</div>"


def render_overview() -> str:
    return OVERVIEW_SIMPLE + OVERVIEW_TECHNICAL


def render_flow() -> str:
    return "### Flujo completo del sistema\n\n" + render_flow_html()


def render_data_universe() -> str:
    return """
### Datos y universo

- **Universo V14 producción:** 49 tickers (`config/v18_3_2_production_v14_contract.json`).
- **Recuentos distintos:**
  - `configured_count` — activos en configuración.
  - `analyzed_count` — activos con datos procesados.
  - `eligible_for_ranking_count` — activos que pasan filtros de elegibilidad.
  - `target_position_count` — posiciones con peso > 0 (p. ej. 4 = 3 + SHY).
- **Data bundle hash:** huella del histórico congelado; cambios → `DATA_REVISION_DETECTED`.
- **Sin bfill** ni membresía point-in-time inventada.

#### Lección de integridad (V18.3)
El C0 antiguo usó **15 activos** (QUICK_TEST) mientras candidatos usaban **49**.
La comparación quedó **invalidada** hasta V18.3.2/V18.3.3, que congelaron el universo real.
"""


def render_indicators() -> str:
    return """
### Indicadores y features

| Feature | Ventana | Uso |
|---------|---------|-----|
| MOM_63 | 63 días | Momentum de precio |
| ABOVE_SMA200 | 200 días | Filtro de tendencia |
| VOL_63 | 63 días | Volatilidad anualizada; filtro > 45% |
| MARKET_RISK_ON | SPY/QQQ vs SMA200 | Régimen risk-on / defensivo |
| trend_quality_126 | 126 cierres | Solo D2: slope × R² en log-precio |

Los NaN **excluyen** al activo; no se imputan valores futuros.
"""


def render_ranking_portfolio() -> str:
    return """
### Ranking y construcción de cartera

1. Calcular features por ticker y fecha.
2. Filtrar elegibles (momentum > 0, SMA200, vol, régimen).
3. Ordenar por COMBINED_SCORE (V14) o D2_SCORE (experimento).
4. Seleccionar top 3 con pesos inversos a volatilidad.
5. Escalar a vol target 15% y rellenar con SHY / caja.
6. Emitir señal semanal y pesos objetivo.
"""


def render_execution_costs() -> str:
    return """
### Ejecución y costes

- **signal_date:** viernes tras cierre.
- **execution_date:** siguiente sesión con apertura disponible.
- **cost_rate:** 0.002 en investigación de producción (costes + slippage).
- Sin apalancamiento; capital inicial 10 000 € teóricos.
- Diferencia simulación vs broker: spreads, impuestos, liquidez, horarios.
"""


def render_backtesting() -> str:
    return """
### Backtesting y ledger

**Principio:** *Una estrategia, un ledger, una equity curve.*

Componentes del ledger:
- `targets` — pesos objetivo por signal_date.
- `executions` — operaciones en siguiente apertura.
- `daily_returns` — retornos diarios de cartera.
- `equity_curve` — capital acumulado.

#### Error corregido en V18.3.1
V18.3 ejecutaba **dos simulaciones C0** distintas: un CSV mostraba CAGR 17.19% (15 tickers)
mientras la equity real en 49 tickers daba ~20.22%. Los tests iniciales no cruzaban equity.
V18.3.1 unificó `build_and_run_frozen_c0()` como única fuente.
"""


def render_metrics_doc() -> str:
    return """
### Métricas — qué miden y qué no

| Métrica | Mide | Mejor | No significa |
|---------|------|-------|--------------|
| Total Return | Ganancia acumulada | Mayor | Rentabilidad futura garantizada |
| CAGR | Tasa anual compuesta | Mayor | Previsión del próximo año |
| Volatilidad | Dispersión diaria | Menor (contexto) | Riesgo máximo |
| Sharpe | Retorno / vol total | Mayor | Éxito sin drawdowns |
| Sortino | Retorno / vol negativa | Mayor | Protección absoluta |
| Max Drawdown | Peor caída desde pico | Menor (menos negativo) | Pérdida máxima futura |
| Calmar | CAGR / |MDD| | Mayor | Robustez en crisis |
| Turnover | Rotación de cartera | Contextual | Coste directo |
| Information Ratio | Alpha / tracking error | Mayor | Selección automática |
| Spearman IC | Score vs retorno forward | > 0 | Correlación perfecta |
| PBO | Prob. sobreajuste backtest | < 0.5 | Certeza de alpha |
| DSR | Sharpe deflactado por pruebas | Mayor | Aprobación real |
"""


def render_validation() -> str:
    return """
### Validación estadística y periodos

| Periodo | Uso |
|---------|-----|
| **RESEARCH** | 2015–2023 — desarrollo y walk-forward histórico |
| **REUSED_TEST** | 2024+ — observado en iteraciones previas; **no es holdout virgen** |
| **FORWARD_PAPER** | Desde 2026-07-14 — seguimiento vivo; no para ajustar parámetros |

#### Gates G1–G19 (resumen)

**Integridad (G1–G5):** temporal, universo, costes, ejecución, pesos.

**Rendimiento (G6–G12):** CAGR, IR, drawdown, turnover, años ganadores, costes 2×.

**Sobreajuste (G13–G15):** PBO, DSR, IC predictivo.

**Concentración (G16–G17):** ticker y sector.

**Despliegue (G18–G19):** forward paper obligatorio; no dinero real.

No escoger una estrategia solo por su CAGR. Más candidatos probados → mayor riesgo de sobreajuste.
"""


def render_integrity() -> str:
    return """
### Controles de integridad

- Hash de contrato de universo (49 tickers).
- Hash de data bundle histórico.
- Elegibilidad idéntica antes de comparar rankings (V18.3.3).
- Cross-check equity → métricas (tolerancia 1e-8).
- Append-only en paper research; idempotencia de señales.
- `DATA_REVISION_DETECTED` detiene el tracker si cambian precios congelados.
"""


def render_forward_paper() -> str:
    return """
### Paper trading prospectivo

- **V17.7:** tracker forward R0/R2/B1.
- **V18.3.5:** shadow tracker D2 vs V14 producción.
- Inicio prospectivo 2026-07-15; primera señal W-FRI posterior.
- Checkpoints 13 / 26 / 52 semanas → informes, no reemplazo automático de V14.
- Métricas anualizadas con guardia de **63 sesiones** mínimas.
"""


def render_limitations() -> str:
    return """
### Limitaciones y riesgos (visibles)

- El backtest **no garantiza** rentabilidad futura.
- **Survivorship bias:** lista fija de 49 activos aplicada al pasado.
- Costes reales, spreads, impuestos y divisa pueden diferir.
- Acciones fraccionadas teóricas; liquidez no modelada al detalle.
- Concentración en 3 posiciones riesgosas + defensivo.
- Drawdowns prolongados son posibles.
- Cambios de régimen de mercado.
- Dependencia del proveedor de precios (yfinance en actualizaciones externas).
- Paper forward aún **corto** en duración.
- **APPROVED_FOR_REAL_MONEY=False** — sin aprobación para dinero real.
"""
