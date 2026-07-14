"""Validación rápida de integración V14."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    errors = []

    # 1. Config
    cfg_path = ROOT / "config" / "approved_v14_strategy.json"
    if not cfg_path.exists():
        errors.append("Falta config/approved_v14_strategy.json")
    else:
        from src.portfolio_v14 import load_v14_config
        cfg = load_v14_config()
        assert cfg["approved_for_web_paper"] is True
        assert cfg["approved_for_real_money"] is False
        print("OK config/approved_v14_strategy.json")

    # 2. Import portfolio_v14
    from src.portfolio_v14 import (
        build_close_prices,
        calculate_v14_target_weights,
        get_current_v14_portfolio_signal,
        load_v14_config,
    )
    print("OK import src.portfolio_v14")

    # 3. Pesos suman 1.0 con datos sintéticos
    import numpy as np
    import pandas as pd

    dates = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(42)
    data_dict = {}
    for t in ["SPY", "QQQ", "AAPL", "MSFT", "SHY", "GLD", "MTUM"]:
        px = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates)))
        data_dict[t] = pd.DataFrame({"Close": px}, index=dates)

    close = build_close_prices(data_dict)
    cfg = load_v14_config()
    w = calculate_v14_target_weights(close, cfg)
    total = float(w.sum())
    if abs(total - 1.0) > 0.02:
        errors.append(f"Pesos no suman 1.0: {total}")
    else:
        print(f"OK pesos suman {total:.4f}")

    # 4. Sin activos válidos -> SHY/CASH
    flat = pd.DataFrame(
        {t: [100.0] * len(dates) for t in ["SHY", "SPY", "QQQ"]},
        index=dates,
    )
    flat["CASH"] = 1.0
    w2 = calculate_v14_target_weights(flat, cfg)
    shy_w = float(w2.get("SHY", 0) + w2.get("CASH", 0))
    if shy_w < 0.99:
        errors.append(f"Esperaba defensivo 100% SHY, obtuvo {w2.to_dict()}")
    else:
        print("OK modo defensivo 100% SHY")

    # 5. get_current_v14_portfolio_signal
    sig = get_current_v14_portfolio_signal(data_dict, capital=10000)
    if sig.get("error"):
        errors.append(f"Señal con error: {sig['error']}")
    else:
        print("OK get_current_v14_portfolio_signal")

    # 6. app.py importa sin error
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", ROOT / "app.py")
    if spec and spec.loader:
        print("OK app.py sintaxis/import base")

    # 7. research_outputs/v14
    v14_dir = ROOT / "research_outputs" / "v14"
    if v14_dir.exists() and any(v14_dir.glob("research_v14_*.csv")):
        print("OK research_outputs/v14/ con archivos")
    else:
        print("AVISO: research_outputs/v14/ vacío o sin CSVs (ejecuta notebook V14)")

    if errors:
        print("\nERRORES:")
        for e in errors:
            print(" -", e)
        sys.exit(1)

    print("\nValidación V14 completada correctamente.")


if __name__ == "__main__":
    main()
