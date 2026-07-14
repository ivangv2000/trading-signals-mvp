"""Genera .ipynb desde archivos .py con celdas # %%"""
import json
import sys
from pathlib import Path


def build_notebook(src_name: str, out_name: str | None = None):
    base = Path(__file__).parent
    src = base / src_name
    out = base / (out_name or src_name.replace(".py", ".ipynb"))

    text = src.read_text(encoding="utf-8")
    chunks = []
    current_type = "code"
    buffer = []

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped == "# %% [markdown]":
            if buffer:
                chunks.append((current_type, "".join(buffer)))
                buffer = []
            current_type = "markdown"
        elif stripped == "# %%":
            if buffer:
                chunks.append((current_type, "".join(buffer)))
                buffer = []
            current_type = "code"
        else:
            buffer.append(line)

    if buffer:
        chunks.append((current_type, "".join(buffer)))

    nb_cells = []
    for ctype, content in chunks:
        content = content.strip("\n")
        if not content.strip():
            continue
        if ctype == "markdown":
            lines = []
            for ln in content.splitlines():
                if ln.startswith("# "):
                    lines.append(ln[2:] + "\n")
                elif ln.startswith("#"):
                    lines.append(ln.lstrip("#") + "\n")
                else:
                    lines.append(ln + "\n")
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            nb_cells.append({
                "cell_type": "code",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
                "source": [ln + "\n" for ln in content.splitlines()],
            })

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "colab": {"provenance": []},
        },
        "cells": nb_cells,
    }

    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Created {out} with {len(nb_cells)} cells")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        build_notebook(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        build_notebook("trading_research_colab.py")
        build_notebook("trading_research_v2_professional.py")
        build_notebook("trading_research_v3_portfolio_lab.py")
        build_notebook("trading_research_v4_robustness_lab.py")
        build_notebook("trading_research_v5_champion_challenger_lab.py")
        build_notebook("trading_research_v6_professional_audit_lab.py")
        build_notebook("trading_research_v7_signal_alpha_lab.py")
        build_notebook("trading_research_v8_stat_arb_lab.py")
        build_notebook("trading_research_v9_clean_signal_lab.py")
        build_notebook("trading_research_v10_professional_signal_engine.py")
        build_notebook("trading_research_v11_event_momentum_lab.py")
        build_notebook("trading_research_v12_robust_factor_rotation_lab.py")
        build_notebook("trading_research_v13_institutional_premia_lab.py")
        build_notebook("trading_research_v14_champion_refinement_lab.py")
        build_notebook("trading_research_v15_meta_champion_regime_lab.py")
        build_notebook("trading_research_v16_drawdown_controlled_meta_lab.py")
        build_notebook("trading_research_v17_universal_equity_alpha_factory_lab.py")
        build_notebook("trading_research_v17_1_leakage_generalization_audit.py")
        build_notebook("trading_research_v17_2_backtest_integrity_lab.py")
        build_notebook("trading_research_v17_3_null_calibration_full_universe_lab.py")
        build_notebook("trading_research_v17_4_preregistered_full_universe_challenge.py")
        build_notebook("trading_research_v17_4_1_integrity_hotfix.py")
        build_notebook("trading_research_v17_5_constraint_universe_dsr_audit.py")
        build_notebook("trading_research_v17_5_1_benchmark_live_pnl_hotfix.py")
        build_notebook("trading_research_v17_5_2_execution_ledger_hotfix.py")
        build_notebook("trading_research_v17_5_2_1_missing_quote_hotfix.py")
        build_notebook("trading_research_v17_5_2_2_validation_closure.py")
        build_notebook("trading_research_v17_6_preregistered_risk_overlay.py")
        build_notebook("trading_research_v17_6_1_result_closure.py")
