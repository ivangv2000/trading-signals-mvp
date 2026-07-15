"""Línea temporal del proyecto."""

from __future__ import annotations

TIMELINE = [
    ("V14", "Baseline público de paper trading.", "VALIDADO"),
    ("V17.6", "Overlays de riesgo evaluados; ninguno aceptado como campeón.", "RECHAZADO"),
    ("V17.7", "Tracker prospectivo dual (R0/R2/B1).", "VALIDADO"),
    ("V18.0", "Baseline congelado y diagnóstico predictivo.", "VALIDADO"),
    ("V18.1", "Variantes momentum adicionales; no superaron V14.", "RECHAZADO"),
    ("V18.2", "D1 sector relative y D2 trend quality preregistrados.", "VALIDADO"),
    ("V18.2.1", "Reparación hash determinista del preregistro.", "CORREGIDO"),
    ("V18.3", "Primera evaluación candidatos; baseline C0 inconsistente (15 vs 49 tickers).", "INVALIDADO"),
    ("V18.3.1", "Ledger C0 reconciliado sobre motor V18.1 (15 tickers).", "CORREGIDO"),
    ("V18.3.2", "Descubierto universo real de producción: 49 activos.", "VALIDADO"),
    ("V18.3.3", "Baseline producción congelado; elegibilidad D2 idéntica a C0.", "VALIDADO"),
    ("V18.3.4", "D2 comparado limpiamente; no seleccionado (G7, G13).", "RECHAZADO"),
    ("V18.3.5", "D2 iniciado como shadow paper research prospectivo.", "EN PAPER RESEARCH"),
]

STATUS_COLORS = {
    "VALIDADO": "#22c55e",
    "RECHAZADO": "#ef4444",
    "INVALIDADO": "#f97316",
    "CORREGIDO": "#3b82f6",
    "EN PAPER RESEARCH": "#a855f7",
}


def render_timeline_html() -> str:
    rows = []
    for version, desc, status in TIMELINE:
        color = STATUS_COLORS.get(status, "#94a3b8")
        rows.append(
            f'<div class="doc-timeline-item">'
            f'<div class="doc-timeline-version">{version}</div>'
            f'<div class="doc-timeline-desc">{desc}</div>'
            f'<div class="doc-timeline-status" style="color:{color}">{status}</div>'
            f"</div>"
        )
    return '<div class="doc-timeline">' + "".join(rows) + "</div>"


def render_project_history() -> str:
    return (
        "### Evolución del proyecto\n\n"
        "Solo **V14 R1 Return Engine** es el modelo público activo. "
        "El resto son hitos de investigación, auditoría o paper tracking.\n\n"
        + render_timeline_html()
    )
