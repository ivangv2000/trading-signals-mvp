# Trading Signals Lab

Herramienta educativa de **backtesting y señales** para acciones y ETFs. Compara varias estrategias simples contra comprar y mantener, y encuentra oportunidades entre varios tickers.

> **No es asesoramiento financiero.** El score no garantiza beneficio. No uses dinero real.

## Requisitos

- Python 3.10+
- Conexión a internet

## Ejecutar

```bash
cd trading-signals-mvp
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Abre `http://localhost:8501` en tu navegador.

## Portfolio V14 R1 Return Engine

Modo **nuevo y aprobado** para paper trading experimental en la app. Aparece como **Portfolio V14 Approved Paper Trading** en el panel izquierdo.

### Qué hace

**V14 R1 Return Engine** usa el motor **A_tsmom_63** del Champion Refinement Lab:

- Cada **viernes** evalúa momentum de 63 días en un universo amplio de ETFs y acciones.
- Selecciona los **top 3** activos con momentum positivo y precio por encima de la **SMA 200**.
- Asigna pesos **inverse volatility** con objetivo de volatilidad **15%** y máximo **30%** por activo.
- Si **SPY y QQQ** están débiles (por debajo de SMA 200), reduce riesgo y favorece **SHY** y defensivos.
- Si no hay activos válidos, la cartera va a **100% SHY/CASH**.

### Por qué mejoró frente a V7–V13

- V7–V11 usaban motores diarios o event-driven frágiles (rechazados).
- V12–V13 mejoraron con premia institucional, pero V13 seguía como CANDIDATE con pocos años ganando a SPY.
- V14 **refina** lo que ya funcionaba (A_tsmom + E4) sin reinventar: R1_RETURN_ENGINE simplifica al motor de retorno puro con mejor balance riesgo/retorno y **50% de años ganando a SPY** en backtest.

### Parámetros clave

| Parámetro | Significado |
|-----------|-------------|
| **top_n = 3** | Máximo 3 activos en cartera cada semana — diversificación controlada |
| **lookback 63** | Momentum calculado con ~3 meses de precios (63 días hábiles) |
| **vol_target 0.15** | Escala exposición para apuntar a ~15% de volatilidad anualizada |

### Métricas del backtest de investigación (histórico)

| Métrica | Valor |
|---------|-------|
| Score V14 | 85/100 |
| CAGR | 17.28% |
| Sharpe | 1.123 |
| Sortino | 1.515 |
| Peor caída (max drawdown) | -23.88% |
| Win años vs SPY | 50% |
| Win años vs QQQ | 42% |
| Robustez | 98 |
| Overfitting | LOW |

**Importante:** son resultados históricos simulados. **No garantizan rentabilidad futura.**

### Riesgos

- Drawdown histórico simulado de **-23.88%**.
- Puede permanecer semanas en **SHY** sin señales BUY.
- Rebalanceo **semanal** — no es trading intradía.
- Solo aprobado para **paper trading experimental**.
- **No aprobado para dinero real.**

### Configuración

La estrategia oficial está en `config/approved_v14_strategy.json`. Los resultados de investigación V14 van en `research_outputs/v14/`.

## Portfolio V6 Paper Trading

Modo **recomendado** de la app para simular una cartera diversificada basada en el audit V6.

### Qué es

**Blended Champion V6** combina dos estrategias con peso 50/50 (alpha 0.5):

- **Trend Following V4 (50%)** — sigue tendencias cuando el precio está por encima de medias largas y asigna más peso a activos con menor volatilidad.
- **Adaptive Ensemble (50%)** — ajusta la exposición según el régimen de mercado: más activos de riesgo si SPY/QQQ están fuertes, más defensivos (SHY, GLD, CASH) si el mercado se debilita.

### Qué significa alpha 0.5

El blend usa `0.5 × V4 + 0.5 × Adaptive`. Un alpha más alto daría más peso al trend following; más bajo, al modelo adaptativo.

### Métricas del backtest de investigación (histórico)

| Métrica | Valor |
|---------|-------|
| CAGR | 19.72% |
| Sharpe | 1.163 |
| Sortino | 1.523 |
| Peor caída (max drawdown) | -34.91% |
| Exceso vs SPY | +1065.97% |
| Exceso vs QQQ | +159.34% |

**Importante:** son resultados históricos simulados. **No garantizan rentabilidad futura.**

### Riesgos

- Drawdown histórico simulado de **-34.91%** — una caída alta.
- Costes de transacción estimados elevados en backtest (~36% acumulado).
- Solo aprobado para **paper trading experimental**.
- **No aprobado para dinero real.**

### Configuración

La estrategia oficial está en `config/approved_v6_strategy.json`. Los resultados de investigación V6 van en `research_outputs/v6/`.

## Cómo usar

### Portfolio V14 (nuevo — aprobado paper trading)

1. Abre la app y selecciona **Portfolio V14 Approved Paper Trading**.
2. Ajusta el capital simulado (por defecto 10.000).
3. Opcional: edita el universo de tickers.
4. Pulsa **Calcular cartera V14**.

### Portfolio V6 (champion histórico)

1. Abre la app y deja seleccionado **Portfolio V6 Paper Trading**.
2. Ajusta el capital simulado (por defecto 10.000).
3. Opcional: edita el universo de tickers.
4. Pulsa **Calcular cartera V6**.

### Swing / Intradía

1. Selecciona **Swing 1-5 días** o **Intradía experimental** en el panel izquierdo.
2. Escribe tickers (ej: `SPY, AAPL, NVDA`).
3. Configura **datos para probar** y **duración máxima de operación**.
4. Pulsa **Analizar**.

## Conceptos clave

### Datos para probar

Cuánto historial de precios usamos para **probar** las estrategias. No es cuánto dura la operación.

| Swing | Intradía |
|-------|----------|
| 6 meses, 1 año, 2 años, 5 años | 5 días, 1 mes, 60 días |

### Duración máxima de operación

Tiempo máximo que la estrategia mantiene una posición abierta antes de salir automáticamente.

| Swing | Intradía |
|-------|----------|
| 1, 2, 3 o 5 días | 3, 5, 8 o 13 velas |

### Score (0-100)

Puntuación de calidad del backtest. **No es garantía de beneficio.** Se basa en rentabilidad histórica, comparación vs mercado, aciertos y drawdown.

### Señales

| Señal | Significado |
|-------|-------------|
| BUY | Buscar compra (entrada activa + score ≥ 60) |
| SELL | Salir / no comprar (debilidad detectada) |
| HOLD | Esperar (sin entrada clara) |
| AVOID | Evitar (backtest débil o mucha incertidumbre) |

## Estrategias comparadas

1. **EMA Trend** — tendencia por medias exponenciales
2. **Momentum Breakout** — ruptura de máximos recientes
3. **Pullback Trend** — rebote dentro de tendencia
4. **Mean Reversion** — rebote tras sobreventa (RSI bajo)
5. **Weakness / Exit Signal** — detecta debilidad para salir

La app elige la mejor estrategia por ticker según el score y muestra si supera o no a **comprar y mantener**.

## Tickers recomendados

`SPY`, `QQQ`, `AAPL`, `MSFT`, `NVDA`, `TSLA`, `AMD`, `META`, `GOOGL`, `AMZN`

## Estructura

```
trading-signals-mvp/
├── app.py                  # Interfaz Trading Signals Lab
├── config/
│   ├── approved_v6_strategy.json  # Estrategia V6 oficial (paper)
│   └── approved_v14_strategy.json # Estrategia V14 oficial (paper)
├── research_outputs/
│   ├── v6/                 # Resultados exportados del notebook V6
│   └── v14/                # Resultados exportados del notebook V14
├── requirements.txt
└── src/
    ├── data.py             # Descarga de datos (yfinance)
    ├── portfolio_v6.py     # Algoritmo Blended Champion V6
    ├── portfolio_v14.py    # Algoritmo V14 R1 Return Engine
    ├── indicators.py       # Indicadores técnicos
    ├── strategies.py       # 5 estrategias
    ├── strategy_backtest.py # Backtest + scoring
    ├── backtest.py         # Backtest legacy
    └── signals.py          # Señales legacy
```

## Aviso legal

Esta herramienta es un experimento educativo. No ejecuta órdenes reales, no conecta brokers y no promete rentabilidad. Valida cualquier estrategia con más pruebas antes de arriesgar dinero real.
