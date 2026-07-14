#!/usr/bin/env python3
"""Valida que el proyecto esté listo para Streamlit Cloud + Vercel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PUBLIC_PYTHON_DIRS = (
    ROOT,
    ROOT / "views",
    ROOT / "services",
    ROOT / "content",
    ROOT / "src",
    ROOT / "ui",
    ROOT / "scripts",
)

EXCLUDE_FILES = {
    "validate_cloud_deployment.py",
}

EXCLUDE_DIR_NAMES = {
    "venv",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "notebooks",
    "legacy_pages",
    "paper_research",
    "tests",
}

WINDOWS_USER_PATH_MARKERS = ("C:" + chr(92) + "Users", "C:/Users")

REQUIRED_REQUIREMENTS = (
    "streamlit>=1.36",
    "pandas",
    "numpy",
    "yfinance",
    "plotly",
    "pyarrow",
)


def _iter_public_python_files() -> list[Path]:
    files: list[Path] = []
    for base in PUBLIC_PYTHON_DIRS:
        if base == ROOT:
            app_py = base / "app.py"
            if app_py.exists():
                files.append(app_py)
            continue
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if path.name in EXCLUDE_FILES:
                continue
            if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
                continue
            files.append(path)
    return sorted(set(files))


def _fail(message: str) -> None:
    print(f"CLOUD DEPLOYMENT CHECK: FAIL\n{message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    errors: list[str] = []

    app_py = ROOT / "app.py"
    requirements = ROOT / "requirements.txt"
    signals_csv = ROOT / "data" / "v14_latest_signals.csv"
    summary_json = ROOT / "data" / "v14_latest_summary.json"
    secrets_toml = ROOT / ".streamlit" / "secrets.toml"
    vercel_index = ROOT / "vercel_site" / "index.html"
    vercel_config = ROOT / "vercel_site" / "vercel.json"
    v14_config = ROOT / "config" / "approved_v14_strategy.json"

    if not app_py.exists():
        errors.append("Falta app.py como punto de entrada.")
    if not requirements.exists():
        errors.append("Falta requirements.txt.")
    else:
        req_text = requirements.read_text(encoding="utf-8").lower()
        for dep in REQUIRED_REQUIREMENTS:
            name = dep.split(">=")[0].split("==")[0].strip()
            if name not in req_text:
                errors.append(f"requirements.txt no incluye: {dep}")

    for py_file in _iter_public_python_files():
        try:
            text = py_file.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"No se pudo leer {py_file.relative_to(ROOT)}: {exc}")
            continue
        if any(marker in text for marker in WINDOWS_USER_PATH_MARKERS):
            errors.append(f"Ruta Windows absoluta en {py_file.relative_to(ROOT)}")

    if not vercel_index.exists():
        errors.append("Falta vercel_site/index.html")
    if not vercel_config.exists():
        errors.append("Falta vercel_site/vercel.json")
    else:
        index_text = vercel_index.read_text(encoding="utf-8") if vercel_index.exists() else ""
        if "STREAMLIT_APP_URL" not in index_text:
            errors.append("index.html debe contener el placeholder STREAMLIT_APP_URL")
        if "embed=true" not in index_text:
            errors.append("index.html debe usar ?embed=true en el iframe")

    if secrets_toml.exists():
        errors.append("No debe incluirse .streamlit/secrets.toml en el repositorio.")

    if not signals_csv.exists():
        errors.append("Falta data/v14_latest_signals.csv")
    if not summary_json.exists():
        errors.append("Falta data/v14_latest_summary.json")

    approved_false = True
    if v14_config.exists():
        cfg = json.loads(v14_config.read_text(encoding="utf-8"))
        if cfg.get("approved_for_real_money") is not False:
            approved_false = False
            errors.append("config/approved_v14_strategy.json debe tener approved_for_real_money=false")
    else:
        errors.append("Falta config/approved_v14_strategy.json")

    if summary_json.exists() and summary_json.read_text(encoding="utf-8").strip() not in ("", "{}"):
        try:
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            if summary.get("approved_for_real_money") is not False:
                approved_false = False
                errors.append("v14_latest_summary.json debe tener approved_for_real_money=false")
        except json.JSONDecodeError:
            errors.append("v14_latest_summary.json no es JSON válido")

    if errors:
        _fail("\n".join(f"- {e}" for e in errors))

    print("CLOUD DEPLOYMENT CHECK: PASS")
    print("STREAMLIT ENTRYPOINT: app.py")
    print("VERCEL WRAPPER: READY")
    print("APPROVED_FOR_REAL_MONEY=False")


if __name__ == "__main__":
    main()
