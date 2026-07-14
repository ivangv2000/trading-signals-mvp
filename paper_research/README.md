# V17.7 — Dual Forward Paper Research

Seguimiento **prospectivo** (hacia adelante) de dos carteras en paper trading virtual:

| Código | Rol |
|--------|-----|
| **R0_BASE_S5** | Control (S5 congelado sin overlay) |
| **R2_SPY_TREND** | Challenger (overlay SPY SMA200) |
| **B1_EQUAL_WEIGHT_CURRENT_ASOF** | Benchmark equal-weight S&P 500 |
| **SPY** | Benchmark adicional buy & hold |

**No se usa dinero real.** `APPROVED_FOR_REAL_MONEY = False` siempre.

---

## 1. Cómo importar el CSV semanal

1. Ejecuta V17.6 y descarga `research_v17_6_current_signals.csv`.
2. Descarga también `research_v17_6_1_selected_config.json` (cierre V17.6.1 validado).
3. Abre la app Streamlit (ver sección 5).
4. En la barra lateral, sube ambos archivos.
5. Pulsa **Importar snapshot de señales**.

El importador comprueba que el cierre sea `V17_6_CLOSED_VALIDATED`, que no haya overlay oficial seleccionado y que el challenger sea `R2_SPY_TREND`.

Si el mismo snapshot ya existe, verás: **Snapshot ya importado** (no se duplica).

---

## 2. Cómo actualizar precios

1. Pulsa **Actualizar precios diarios** en la barra lateral.
2. El sistema:
   - ejecuta rebalanceos pendientes en la **siguiente apertura** tras cada señal;
   - valora las carteras al **cierre** de cada día nuevo;
   - aplica **costes y slippage** iguales a V17.6 (0.1% + 0.1%).

Repite este paso cada día laborable (o cuando quieras refrescar).

---

## 3. Cuándo cambia una cartera

| Evento | Qué ocurre |
|--------|------------|
| Nueva señal semanal importada | Se guardan pesos objetivo R0, R2, B1 y SPY |
| Siguiente día de mercado abierto | Ejecución virtual al **Open** |
| Días sin nueva señal | Posiciones constantes, valoración diaria al **Close** |

**R2** se deriva siempre de **R0** con la regla congelada:

- SPY > SMA200 (desplazada 1 sesión) → exposición 100%
- En caso contrario → exposición 35%, resto a SHY

---

## 4. Dónde se guardan los resultados

```
paper_research/
  data/
    signal_snapshots.csv      # pesos por snapshot (append-only)
    executions.csv            # ejecuciones en next open
    daily_equity.csv          # valoración diaria forward
    current_positions.csv     # últimas posiciones
    membership_snapshots.csv    # membresía B1 fechada
  state/
    paper_state.json            # estado del tracker
  exports/
    V17_7_PAPER_EXPORT.zip    # export manual
```

Los archivos originales de V17.6 **no se modifican**.

---

## 5. Cómo abrir la página Streamlit

Desde la raíz del proyecto:

```bash
streamlit run app.py
```

En el menú lateral de páginas, elige **Paper Research** (`2_Paper_Research`).

También puedes abrir directamente:

```bash
streamlit run pages/2_Paper_Research.py
```

---

## 6. Sin dinero real

- Capital virtual inicial: **10.000 USD** por cartera.
- No hay conexión con brokers ni APIs de ejecución real.
- Las métricas mostradas son **FORWARD OBSERVATIONS** — no backtest histórico.
- Los resultados prospectivos **no garantizan** rentabilidad futura.

---

## Tests integrados

Al importar se ejecutan 12 tests técnicos. La salida esperada es:

```
V17.7 PAPER TRACKER TESTS: 12/12 PASS
```

Desde Python:

```python
from paper_research.forward_paper_tracker import ForwardPaperTracker

tracker = ForwardPaperTracker()
tracker.run_tests(config_path="ruta/a/research_v17_6_1_selected_config.json")
```

---

## Punto de partida (V17.6.1)

- `CLOSURE_STATUS = V17_6_CLOSED_VALIDATED`
- `official_selected_overlay = NONE`
- `paper_research_challenger = R2_SPY_TREND`
- `paper_trading_start = 2026-07-10`
