"""Glosario de trading y cuant."""

from __future__ import annotations

GLOSSARY: dict[str, str] = {
    "acción": "Participación en una empresa que cotiza en bolsa.",
    "ETF": "Fondo cotizado que replica un índice, sector o activo.",
    "ticker": "Símbolo bursátil (ej. SPY, AAPL).",
    "OHLCV": "Open, High, Low, Close, Volume — datos diarios estándar.",
    "momentum": "Fuerza del movimiento reciente del precio respecto al pasado.",
    "tendencia": "Dirección persistente del precio en un horizonte (p. ej. SMA200).",
    "ranking": "Orden relativo de activos en una fecha según un score.",
    "feature": "Variable calculada a partir de precios (MOM_63, volatilidad, etc.).",
    "score": "Número para ordenar activos; no es probabilidad de subida.",
    "portfolio": "Conjunto de activos con pesos objetivo.",
    "peso": "Fracción del capital asignada a un activo (0–1 o %).",
    "rebalanceo": "Ajuste periódico de pesos para alinear con la señal.",
    "drawdown": "Caída desde un máximo histórico hasta un mínimo posterior.",
    "volatilidad": "Dispersión de los retornos; mayor vol = más variación.",
    "Sharpe": "Rentabilidad ajustada por volatilidad total.",
    "Sortino": "Similar al Sharpe pero penaliza solo retornos negativos.",
    "CAGR": "Tasa de crecimiento anual compuesta.",
    "turnover": "Rotación de cartera; cuánto cambian los pesos.",
    "slippage": "Diferencia entre precio esperado y precio ejecutado.",
    "benchmark": "Referencia de comparación (p. ej. SPY).",
    "backtest": "Simulación histórica con reglas fijas.",
    "paper trading": "Simulación sin dinero real.",
    "forward paper": "Seguimiento prospectivo tras congelar la fórmula.",
    "holdout": "Periodo no usado para ajustar parámetros.",
    "overfitting": "Sobreajuste al pasado; puede fallar fuera de muestra.",
    "look-ahead bias": "Usar información del futuro en el pasado.",
    "survivorship bias": "Usar solo activos que sobrevivieron hasta hoy.",
    "ledger": "Registro único de señales, ejecuciones y equity.",
    "PBO": "Probability of Backtest Overfitting — riesgo de sobreajuste.",
    "DSR": "Deflated Sharpe Ratio — Sharpe ajustado por múltiples pruebas.",
    "information ratio": "Rentabilidad activa vs tracking error respecto a benchmark.",
    "Spearman IC": "Correlación de rangos entre score y retorno forward.",
    "OLS": "Mínimos cuadrados ordinarios — ajuste de recta a datos.",
    "R²": "Coeficiente de determinación; calidad del ajuste lineal.",
}


def filter_glossary(query: str) -> dict[str, str]:
    q = (query or "").strip().lower()
    if not q:
        return GLOSSARY
    return {k: v for k, v in GLOSSARY.items() if q in k.lower() or q in v.lower()}
