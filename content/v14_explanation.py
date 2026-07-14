"""Contenido educativo V14 — leído desde configuración y código existente."""

from __future__ import annotations

import json
from pathlib import Path

from src.portfolio_v14 import (
    CASH_ASSET,
    DEFAULT_UNIVERSE,
    DEFENSIVE_POOL,
    MAX_WEIGHT,
    MIN_HISTORY_DAYS,
    load_v14_config,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "approved_v14_strategy.json"

INTRO_TEXT = (
    "V14 es un sistema cuantitativo semanal. Analiza datos históricos, selecciona "
    "una cartera y genera señales objetivas. No intenta adivinar cada movimiento "
    "diario ni garantiza beneficios."
)

A_TSMOM_SIMPLE = (
    "V14 busca activos cuya evolución reciente ha sido más favorable según sus "
    "reglas. Los mejor clasificados tienen más posibilidades de entrar en la cartera."
)

A_TSMOM_MATH = """
**Indicador A_tsmom_63 (implementación real en `src/portfolio_v14.py`)**

- **Datos:** precios de cierre diarios (`Close`) por activo.
- **Ventana:** 63 días de lookback (`lookback_days` en config).
- **Fórmula principal:**
  ```
  MOM_63 = close / close.shift(63) - 1
  ```
- **Filtros adicionales en selección:**
  - `ABOVE_SMA200`: precio por encima de la media móvil de 200 días.
  - `VOL_63`: volatilidad anualizada a 63 días; se descartan activos con vol > 45%.
  - `MARKET_RISK_ON`: SPY y QQQ por encima de SMA200 para modo risk-on.
- **Score combinado:**
  ```
  COMBINED_SCORE = MOM_63 * 0.5 + TREND_SCORE * 0.5
  ```
  donde `TREND_SCORE` usa `ABOVE_SMA200`.
- **Interpretación:** momentum positivo y tendencia favorable aumentan la probabilidad de entrar en el top N.
"""

FLOW_DIAGRAM = """
Datos de mercado
→ Cálculo de señales
→ Ranking
→ Construcción de cartera
→ Control de riesgo
→ Señal semanal
"""

EXAMPLE_TEXT = """
**Ejemplo educativo (no es una señal real)**

- **Lunes:** AAPL, MSFT, UNH y VLO están en el universo.
- **Viernes:** V14 vuelve a calcular el ranking.
- **Resultado:**
  - AAPL entra → **BUY**
  - UNH continúa → **HOLD**
  - VLO reduce peso → **REDUCE**
  - MSFT sale → **SELL**

Este ejemplo ilustra cómo se leen las señales. No sustituye al snapshot semanal real.
"""

METRIC_EXPLANATIONS = {
    "CAGR": "Crecimiento anual compuesto del backtest. No es una previsión.",
    "Sharpe": "Relación entre rentabilidad y variabilidad.",
    "Sortino": "Similar al Sharpe, pero penaliza sobre todo las caídas.",
    "Max Drawdown": "Mayor caída histórica desde un máximo hasta un mínimo.",
    "Rebalances": "Número de rebalanceos simulados en el periodo de backtest.",
    "Overfitting risk": "Riesgo estimado de que el modelo esté demasiado ajustado al pasado.",
}

GLOSSARY = {
    "Ticker": "Símbolo bursátil de una acción o ETF (por ejemplo, AAPL o SPY).",
    "BUY": "Señal de entrada: el activo entra en la cartera objetivo esta semana.",
    "HOLD": "Mantener: el activo sigue en cartera sin cambio relevante de peso.",
    "REDUCE": "Reducir peso: el activo permanece pero con menor asignación.",
    "SELL": "Salida: el activo deja de formar parte de la cartera objetivo.",
    "Peso": "Porcentaje del capital asignado a cada activo en la cartera.",
    "Rebalanceo": "Ajuste periódico de pesos para alinear la cartera con la señal.",
    "Momentum": "Medida de la fuerza reciente del precio respecto al pasado.",
    "Backtest": "Simulación histórica con reglas fijas para evaluar la estrategia.",
    "Benchmark": "Referencia de comparación, como SPY o una cartera 60/40.",
    "Drawdown": "Caída desde un pico previo hasta un mínimo posterior.",
    "CAGR": "Tasa de crecimiento anual compuesta en un periodo.",
    "Sharpe": "Métrica que relaciona rentabilidad con volatilidad total.",
    "Paper trading": "Simulación sin dinero real ni ejecución de órdenes.",
    "Sobreajuste": "Cuando un modelo funciona muy bien en el pasado pero puede fallar fuera de muestra.",
}

PROJECT_HISTORY = """
- **Primeras versiones:** exploración de reglas simples y backtests por activo.
- **V6:** motor blended de investigación con validación paper.
- **V14:** campeón actual de paper trading con A_tsmom_63 y rebalanceo semanal.
- **V17:** línea de investigación avanzada con overlays y validación preregistrada.
- **Overlays:** capas de riesgo evaluadas sobre motores base.
- **Paper research:** seguimiento forward de snapshots importados.
"""

LIMITATIONS = [
    "Un backtest no garantiza resultados futuros.",
    "La estrategia puede atravesar periodos de pérdidas prolongadas.",
    "Depende de la calidad y disponibilidad de los datos de mercado.",
    "Existen costes, slippage y diferencias entre simulación y ejecución real.",
    "El mercado puede cambiar de régimen y el algoritmo puede dejar de funcionar.",
    "Una señal semanal no es una certeza ni una recomendación personalizada.",
    "No usa información del futuro (sin lookahead en el diseño del motor).",
    "No debe usarse con apalancamiento ni como sustituto de asesoramiento.",
]


def load_config() -> dict:
    return load_v14_config(CONFIG_PATH)


def portfolio_rules_from_config(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    engine = cfg.get("base_engine", {})
    bt = cfg.get("backtest_summary", {})

    universe_list = sorted(set(DEFAULT_UNIVERSE) | set(DEFENSIVE_POOL))
    universe_str = ", ".join(universe_list[:12]) + f" … ({len(universe_list)} activos en código)"

    return {
        "universo": universe_str,
        "frecuencia": cfg.get("rebalance_freq", "Dato no disponible en la configuración cargada."),
        "max_posiciones": engine.get("top_n", "Dato no disponible en la configuración cargada."),
        "lookback_dias": engine.get("lookback_days", "Dato no disponible en la configuración cargada."),
        "vol_objetivo": engine.get("vol_target", "Dato no disponible en la configuración cargada."),
        "seleccion": (
            f"Top {engine.get('top_n', 3)} por score A_tsmom_63 con filtros SMA200, volatilidad y régimen SPY/QQQ. "
            f"Peso máximo por activo: {MAX_WEIGHT * 100:.0f}%."
        ),
        "mantenimiento": "Posiciones se revisan en cada rebalanceo semanal (viernes). HOLD si el peso cambia ≤ 3 pp.",
        "pesos": "Asignación inversa a volatilidad entre seleccionados, escalada al vol target y relleno con activo defensivo.",
        "defensivo": f"Activo defensivo principal: {CASH_ASSET}. Pool defensivo: {', '.join(DEFENSIVE_POOL[:6])}…",
        "riesgo": (
            f"Mínimo {MIN_HISTORY_DAYS} días de historia. Modo risk-off rota a pool defensivo. "
            f"Overfitting risk backtest: {bt.get('overfitting_risk', 'Dato no disponible en la configuración cargada.')}"
        ),
        "ejecucion": "Señal semanal; entrada simulada próxima apertura post-viernes. Sin órdenes reales.",
        "costes": "Dato no disponible en la configuración cargada." if "cost" not in json.dumps(cfg).lower() else bt,
    }


def backtest_metrics_from_config(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    bt = cfg.get("backtest_summary", {})
    if not bt:
        return {}

    return {
        "CAGR": bt.get("CAGR"),
        "Sharpe": bt.get("sharpe"),
        "Sortino": bt.get("sortino"),
        "Max Drawdown": bt.get("max_drawdown"),
        "Rebalances": bt.get("num_rebalances"),
        "Overfitting risk": bt.get("overfitting_risk"),
        "win_years_vs_spy": bt.get("win_years_vs_spy"),
        "robustness_score": bt.get("robustness_score"),
        "turnover": bt.get("turnover"),
        "exposure": bt.get("exposure"),
    }


def champion_section() -> dict:
    cfg = load_config()
    return {
        "title": "CURRENT PAPER CHAMPION",
        "strategy": cfg.get("strategy_name", "V14 R1 Return Engine"),
        "approved_for_real_money": False,
        "status": cfg.get("status", "APPROVED_FOR_WEB_PAPER"),
        "score": cfg.get("score", 85),
    }
