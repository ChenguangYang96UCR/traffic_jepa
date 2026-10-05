#!/usr/bin/env python3
"""Summarize the fresh-head latent-weight sweep without selecting on test data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = []
    for result_path in sorted(root.glob("lambda=*/abc_results.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        config = payload["fixed_configuration"]
        row = next(item for item in payload["rows"] if item["method"].startswith("C:"))
        rows.append({"latent_weight": config["C_latent_weight"], **row})
    if not rows:
        raise FileNotFoundError(f"No completed sweep results under {root}")

    # Validation MAE is the only selection signal. Test metrics are report-only.
    rows.sort(key=lambda row: (row["finetune_val_mae"], row["latent_weight"]))
    lines = [
        "Fresh-head task-aware SALT latent-weight sweep",
        "Selection rule: lowest Berkeley fine-tune validation MAE (test metrics are report-only)",
        "Fixed: Teacher d=128, blocks=8, future ratio=0.50, all-step, fresh head, full FT LR=1e-3",
        "",
        (
            f"{'Best':<5} {'Lambda':>9} {'Epoch':>7} {'Stu val latent':>15} "
            f"{'Stu val fcst':>13} {'FT val MAE':>11} {'Test MAE':>11} "
            f"{'Test MSE':>11} {'Test RMSE':>11}"
        ),
        "-" * 112,
    ]
    for index, row in enumerate(rows):
        lines.append(
            f"{('*' if index == 0 else ''):<5} "
            f"{row['latent_weight']:>9.4g} {row['student_epoch']:>7d} "
            f"{row['student_val_latent']:>15.7f} "
            f"{row['student_val_forecast_mae']:>13.7f} "
            f"{row['finetune_val_mae']:>11.7f} {row['test_mae']:>11.7f} "
            f"{row['test_mse']:>11.7f} {row['test_rmse']:>11.7f}"
        )
    summary = "\n".join(lines)
    print(summary)
    (root / "fresh_head_sweep_summary.txt").write_text(summary + "\n", encoding="utf-8")
    (root / "fresh_head_sweep_results.json").write_text(
        json.dumps({"selection_metric": "finetune_val_mae", "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
