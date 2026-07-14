"""
Generación de señales con explicación en lenguaje humano.
Incluye modo diario clásico y modo corto plazo (swing / intradía).
"""

import pandas as pd

# Textos de acción y decisión simple por señal
SIGNAL_ACTIONS = {
    "BUY": "Comprar / buscar entrada",
    "SELL": "Vender / salir",
    "HOLD": "Mantener / esperar",
    "AVOID": "Evitar / no entrar",
}

SIGNAL_SIMPLE_DECISIONS = {
    "BUY": "El algoritmo permite buscar una entrada, pero con control de riesgo.",
    "SELL": "El algoritmo detecta debilidad. Si estás dentro, conviene salir o reducir.",
    "HOLD": "No hay entrada clara. Mejor esperar a que el mercado confirme dirección.",
    "AVOID": "Hay demasiada incertidumbre o riesgo. Mejor no operar este ticker ahora.",
}


def _enrich_signal(
    signal: str,
    risk_level: str,
    explanation: str,
    reasons: list,
    suggested_horizon: str = "Próximos días",
) -> dict:
    """Añade campos de presentación amigable al diccionario de señal."""
    return {
        "signal": signal,
        "risk_level": risk_level,
        "suggested_horizon": suggested_horizon,
        "action_text": SIGNAL_ACTIONS.get(signal, signal),
        "simple_decision": SIGNAL_SIMPLE_DECISIONS.get(signal, ""),
        "explanation": explanation,
        "reasons": reasons,
    }


def generate_signal(df, metrics: dict) -> dict:
    """
    Genera una señal diaria clásica basada en la última fila válida.

    Señales posibles: BUY, SELL, HOLD, AVOID
    """
    required = ["Close", "SMA_50", "SMA_200", "RSI_14", "MOMENTUM_20", "VOLATILITY_20"]
    valid = df.dropna(subset=required)

    if valid.empty or len(valid) < 50:
        return _enrich_signal(
            "AVOID",
            "High",
            (
                "No hay suficientes datos históricos para generar una señal fiable. "
                "Prueba con un periodo más largo o verifica el ticker."
            ),
            ["Datos insuficientes para el análisis."],
            suggested_horizon="Próximos días",
        )

    last = valid.iloc[-1]
    close = last["Close"]
    sma_50 = last["SMA_50"]
    sma_200 = last["SMA_200"]
    rsi = last["RSI_14"]
    momentum = last["MOMENTUM_20"]
    volatility = last["VOLATILITY_20"]

    reasons = []

    price_above_sma50 = close > sma_50
    price_below_sma50 = close < sma_50
    sma50_above_sma200 = sma_50 > sma_200
    price_below_sma200 = close < sma_200
    rsi_in_buy_zone = 45 <= rsi <= 70
    rsi_oversold = rsi < 40
    momentum_positive = momentum > 0
    momentum_negative = momentum < 0

    vol_threshold = valid["VOLATILITY_20"].quantile(0.75)
    high_volatility = volatility > vol_threshold

    if price_above_sma50:
        reasons.append("El precio está por encima de la media de 50 días.")
    else:
        reasons.append("El precio está por debajo de la media de 50 días.")

    if sma50_above_sma200:
        reasons.append("La tendencia de medio plazo es alcista (SMA 50 > SMA 200).")
    else:
        reasons.append("La tendencia de medio plazo es bajista (SMA 50 < SMA 200).")

    if rsi_in_buy_zone:
        reasons.append(f"El RSI ({rsi:.1f}) está en zona neutral-alcista.")
    elif rsi_oversold:
        reasons.append(f"El RSI ({rsi:.1f}) indica sobreventa.")
    else:
        reasons.append(f"El RSI ({rsi:.1f}) está fuera de la zona ideal de compra.")

    if momentum_positive:
        reasons.append("El momentum de 20 días es positivo.")
    else:
        reasons.append("El momentum de 20 días es negativo.")

    if high_volatility:
        reasons.append("La volatilidad reciente es elevada.")

    contradictory = (
        (price_above_sma50 and momentum_negative and rsi_oversold)
        or (price_below_sma200 and momentum_negative)
        or high_volatility
    )

    if contradictory and (price_below_sma200 or high_volatility):
        signal, risk_level = "AVOID", "High"
        explanation = (
            "Las condiciones actuales presentan riesgo elevado o señales contradictorias. "
            "Es preferible esperar a que el mercado muestre una dirección más clara. "
            "Esto no es asesoramiento financiero."
        )
    elif (
        price_above_sma50
        and sma50_above_sma200
        and rsi_in_buy_zone
        and momentum_positive
        and not high_volatility
    ):
        signal, risk_level = "BUY", "Low"
        explanation = (
            "Los indicadores muestran una tendencia alcista con momentum positivo "
            "y el RSI en zona favorable. La estrategia sugiere considerar una posición larga. "
            "Recuerda: esto es solo un análisis técnico básico, no una recomendación de inversión."
        )
    elif (price_below_sma50 and momentum_negative) or rsi_oversold:
        signal, risk_level = "SELL", "Medium"
        explanation = (
            "Los indicadores muestran debilidad: el precio pierde soporte "
            "o el momentum es negativo. La estrategia sugiere reducir o cerrar posiciones. "
            "Esto no es asesoramiento financiero."
        )
    elif price_above_sma50 or sma50_above_sma200:
        signal, risk_level = "HOLD", "Medium"
        explanation = (
            "La tendencia general es positiva, pero no hay una señal clara de compra en este momento. "
            "Si ya tienes posición, podrías mantenerla. Si no, es mejor esperar confirmación. "
            "Esto no es asesoramiento financiero."
        )
    else:
        signal, risk_level = "AVOID", "High"
        explanation = (
            "Las condiciones no son favorables para operar. "
            "El precio está por debajo de las medias principales y el momentum es débil. "
            "Esto no es asesoramiento financiero."
        )

    return _enrich_signal(signal, risk_level, explanation, reasons, "Próximos días")


def generate_short_term_signal(df, metrics: dict, mode: str = "swing") -> dict:
    """
    Genera señal de corto plazo para swing o intradía.

    Args:
        df: DataFrame con precios e indicadores.
        metrics: Métricas del backtest corto plazo.
        mode: "swing" o "intraday".

    Returns:
        Diccionario con signal, risk_level, explanation, reasons, suggested_horizon,
        action_text y simple_decision.
    """
    min_rows = 30 if mode == "swing" else 50
    required = ["Close", "EMA_9", "EMA_21", "RSI_14", "MOMENTUM_5", "VOLATILITY_10"]
    if mode == "swing":
        required.append("SMA_50")

    valid = df.dropna(subset=[c for c in required if c in df.columns])

    horizon = "1-5 días" if mode == "swing" else "próximas velas intradía"

    if valid.empty or len(valid) < min_rows:
        return _enrich_signal(
            "AVOID",
            "High",
            (
                "No hay suficientes datos para generar una señal de corto plazo fiable. "
                "Prueba con un periodo más largo o un ticker más líquido. "
                "Esto no es asesoramiento financiero ni garantía de beneficio."
            ),
            ["Datos insuficientes para el análisis de corto plazo."],
            suggested_horizon=horizon,
        )

    last = valid.iloc[-1]
    close = last["Close"]
    ema_9 = last["EMA_9"]
    ema_21 = last["EMA_21"]
    rsi = last["RSI_14"]
    momentum = last["MOMENTUM_5"]
    volatility = last["VOLATILITY_10"]
    vwap = last.get("VWAP", float("nan"))
    sma_50 = last.get("SMA_50", float("nan"))

    reasons = []

    ema_bullish = ema_9 > ema_21
    ema_bearish = ema_9 < ema_21
    rsi_buy_zone = 45 <= rsi <= 70
    rsi_oversold = rsi < 40
    momentum_positive = momentum > 0
    momentum_negative = momentum < 0

    vol_threshold = valid["VOLATILITY_10"].quantile(0.75)
    high_volatility = pd.notna(volatility) and volatility > vol_threshold

    price_above_vwap = pd.notna(vwap) and close > vwap
    price_below_vwap = pd.notna(vwap) and close < vwap
    price_above_sma50 = pd.notna(sma_50) and close > sma_50

    if ema_bullish:
        reasons.append("La EMA 9 está por encima de la EMA 21 (momentum corto alcista).")
    else:
        reasons.append("La EMA 9 está por debajo de la EMA 21 (momentum corto bajista).")

    if mode == "swing" and pd.notna(sma_50):
        if price_above_sma50:
            reasons.append("El precio está por encima de la SMA 50.")
        else:
            reasons.append("El precio está por debajo de la SMA 50.")

    if rsi_buy_zone:
        reasons.append(f"El RSI ({rsi:.1f}) está en zona favorable para entrada.")
    elif rsi_oversold:
        reasons.append(f"El RSI ({rsi:.1f}) indica sobreventa.")
    else:
        reasons.append(f"El RSI ({rsi:.1f}) está fuera de la zona ideal.")

    if momentum_positive:
        reasons.append("El momentum de 5 periodos es positivo.")
    else:
        reasons.append("El momentum de 5 periodos es negativo.")

    if pd.notna(vwap):
        if price_above_vwap:
            reasons.append("El precio está por encima del VWAP aproximado.")
        else:
            reasons.append("El precio está por debajo del VWAP aproximado.")

    if high_volatility:
        reasons.append("La volatilidad reciente es elevada.")

    if mode == "swing":
        entry_ok = ema_bullish and price_above_sma50 and rsi_buy_zone and momentum_positive
    else:
        vwap_ok = price_above_vwap if pd.notna(vwap) else True
        entry_ok = ema_bullish and rsi_buy_zone and momentum_positive and vwap_ok

    exit_ok = ema_bearish or rsi_oversold or momentum_negative

    contradictory = (
        (ema_bullish and momentum_negative)
        or (rsi_buy_zone and ema_bearish)
        or (price_below_vwap and ema_bullish and mode == "intraday")
    )

    if high_volatility and contradictory:
        signal, risk_level = "AVOID", "High"
        explanation = (
            "Hay volatilidad alta y señales contradictorias. "
            "No es un buen momento para operar a corto plazo. "
            "Esto no es asesoramiento financiero ni garantía de beneficio."
        )
    elif high_volatility:
        signal, risk_level = "AVOID", "High"
        explanation = (
            "La volatilidad es demasiado alta para una operación de corto plazo segura. "
            "Es mejor esperar a que el mercado se calme. "
            "Esto no es asesoramiento financiero ni garantía de beneficio."
        )
    elif contradictory:
        signal, risk_level = "AVOID", "High"
        explanation = (
            "Los indicadores envían señales contradictorias. "
            "Es preferible no operar hasta que haya más claridad. "
            "Esto no es asesoramiento financiero ni garantía de beneficio."
        )
    elif entry_ok and not high_volatility:
        signal, risk_level = "BUY", "Low"
        explanation = (
            f"Los indicadores de corto plazo muestran condiciones favorables para una "
            f"posible entrada con horizonte de {horizon}. "
            f"Esto no es asesoramiento financiero ni garantía de beneficio."
        )
    elif exit_ok:
        signal, risk_level = "SELL", "Medium"
        explanation = (
            f"Los indicadores sugieren debilidad a corto plazo. "
            f"Considera cerrar o reducir posiciones con horizonte de {horizon}. "
            f"Esto no es asesoramiento financiero ni garantía de beneficio."
        )
    elif ema_bullish or (mode == "swing" and price_above_sma50):
        signal, risk_level = "HOLD", "Medium"
        explanation = (
            f"Hay tendencia positiva pero la entrada no es clara todavía. "
            f"Si ya tienes posición, podrías mantenerla. Si no, espera confirmación. "
            f"Esto no es asesoramiento financiero ni garantía de beneficio."
        )
    else:
        signal, risk_level = "AVOID", "High"
        explanation = (
            "Las condiciones no son favorables para operar a corto plazo. "
            "Esto no es asesoramiento financiero ni garantía de beneficio."
        )

    return _enrich_signal(signal, risk_level, explanation, reasons, horizon)
