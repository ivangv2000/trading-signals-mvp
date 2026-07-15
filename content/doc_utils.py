"""Utilidades read-only para documentación — sin red ni escrituras."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def safe_read_csv(rel_path: str) -> pd.DataFrame:
    """Lee un CSV relativo al repo. Vacío o ausente → DataFrame vacío."""
    path = ROOT / rel_path
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, OSError):
        return pd.DataFrame()


def safe_read_json(rel_path: str) -> dict:
    """Lee un JSON relativo al repo. Ausente o mal formado → {}."""
    path = ROOT / rel_path
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def fmt_metric(value: Any, suffix: str = "") -> str:
    """Formatea una métrica; valores faltantes → 'Dato no disponible'."""
    if value is None:
        return "Dato no disponible"
    try:
        if pd.isna(value):
            return "Dato no disponible"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)):
        return f"{value}{suffix}"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return "Dato no disponible"
    return text


DOC_SECTIONS = [
    ("overview", "1. Visión general"),
    ("flow", "2. Flujo completo del sistema"),
    ("v14", "3. V14 R1 Return Engine"),
    ("d2", "4. D2 Trend Quality"),
    ("data", "5. Datos y universo"),
    ("indicators", "6. Indicadores y features"),
    ("ranking", "7. Ranking y construcción de cartera"),
    ("execution", "8. Ejecución y costes"),
    ("backtesting", "9. Backtesting"),
    ("metrics", "10. Métricas"),
    ("validation", "11. Validación estadística"),
    ("integrity", "12. Controles de integridad"),
    ("forward_paper", "13. Paper trading prospectivo"),
    ("limitations", "14. Limitaciones y riesgos"),
    ("history", "15. Evolución del proyecto"),
    ("glossary", "16. Glosario"),
]

DOC_SECTION_IDS = {s[0] for s in DOC_SECTIONS}
DOC_LABEL_TO_ID = {label: sid for sid, label in DOC_SECTIONS}
DOC_ID_TO_LABEL = {sid: label for sid, label in DOC_SECTIONS}

DOC_ALIASES = {
    "backtest": "backtesting",
    "gates": "validation",
    "risks": "limitations",
}


def resolve_doc_param(raw: str | None) -> str:
    """Resuelve ?doc=... a un id de capítulo válido; desconocidos → overview."""
    if not raw:
        return "overview"
    key = raw.strip().lower()
    if key in DOC_SECTION_IDS:
        return key
    if key in DOC_ALIASES:
        return DOC_ALIASES[key]
    return "overview"
