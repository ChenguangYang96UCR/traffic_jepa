#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


METHODS = (
    ("stgnn_supervised", "ST-GNN supervised"),
    ("jepa_frozen_probe", "JEPA + frozen probe"),
    ("jepa_full_finetune", "JEPA + full fine-tune"),
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_comparison.py RUN_ROOT")
    root = Path(sys.argv[1])
    rows = []
    for directory, label in METHODS:
        path = root / directory / "test_metrics.json"
        if not path.exists():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result["test_masked_positions"]
        rows.append((label, metrics["mae"], metrics["mse"], metrics["rmse"]))

    header = f"{'Method':<28} {'MAE':>12} {'MSE':>12} {'RMSE':>12}"
    lines = [header, "-" * len(header)]
    lines.extend(
        f"{label:<28} {mae:>12.7f} {mse:>12.7f} {rmse:>12.7f}"
        for label, mae, mse, rmse in rows
    )
    summary = "\n".join(lines)
    print("\nMasked sensor-time comparison\n" + summary)
    (root / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(f"\nSaved {root / 'summary.txt'}")


if __name__ == "__main__":
    main()

