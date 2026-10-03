#!/usr/bin/env python3
"""Summarize and select the Oakland-Teacher/Berkeley-Student SALT sweep."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CONFIG_RE = re.compile(r"^blocks=(\d+)_ratio=(.+)$")
LR_RE = re.compile(r"^lr=(.+)$")


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root

    rows = []
    for result_file in sorted(root.glob("results/*/lr=*/all_results.json")):
        config_match = CONFIG_RE.match(result_file.parents[1].name)
        lr_match = LR_RE.match(result_file.parent.name)
        if config_match is None or lr_match is None:
            continue
        blocks = int(config_match.group(1))
        ratio_text = config_match.group(2)
        ratio = float(ratio_text)
        lr = float(lr_match.group(1))
        tag = result_file.parents[1].name
        result = read_json(result_file)["salt_full"]
        validation = result.get("selection_validation")
        if validation is None:
            raise ValueError(
                f"{result_file} lacks selection_validation; rerun this downstream "
                "result with the current code so test data are not used for tuning."
            )
        teacher = read_json(
            root / "teachers" / tag / "teacher" / "teacher_metrics.json"
        )
        student = read_json(
            root / "students" / tag / "student" / "distillation_metrics.json"
        )
        rows.append({
            "mask_blocks": blocks,
            "teacher_future_block_ratio": ratio,
            "allocated_future_blocks": min(
                blocks, max(0, int(blocks * ratio + 0.5))
            ),
            "realized_future_mask_fraction": teacher["validation"].get(
                "future_mask_fraction"
            ),
            "finetune_lr": lr,
            "student_val_latent_l1": student["validation"]["scope"],
            "forecast_val_mae": validation["mae"],
            "test_mae": result["mae"],
            "test_mse": result["mse"],
            "test_rmse": result["rmse"],
            "result_file": str(result_file),
        })

    if not rows:
        raise FileNotFoundError(f"No completed sweep results found under {root}")
    # Hyperparameters and validation MAE are the only tie-breakers. Test metrics
    # remain strictly report-only.
    rows.sort(key=lambda row: (
        row["forecast_val_mae"],
        row["mask_blocks"],
        row["teacher_future_block_ratio"],
        row["finetune_lr"],
    ))
    best = rows[0]

    baseline_file = root / "baseline" / "all_results.json"
    baseline = None
    if baseline_file.exists():
        baseline = read_json(baseline_file).get("staeformer_scratch")

    lines = [
        "Oakland Teacher d=128 -> Berkeley Student | all-step SALT sweep",
        "Selection rule: lowest Berkeley validation MAE (test metrics are report-only)",
    ]
    if baseline is not None:
        lines.append(
            "Berkeley scratch test: "
            f"MAE={baseline['mae']:.7f} MSE={baseline['mse']:.7f} "
            f"RMSE={baseline['rmse']:.7f}"
        )
    lines.extend([
        "",
        (
            f"{'Best':<5} {'Blocks':>6} {'Ratio':>7} {'F blocks':>8} "
            f"{'Mask frac':>10} {'FT LR':>10} {'Val latent':>11} "
            f"{'Val MAE':>11} {'Test MAE':>11} {'Test MSE':>11} {'Test RMSE':>11}"
        ),
        "-" * 118,
    ])
    for index, row in enumerate(rows):
        mask_fraction = row["realized_future_mask_fraction"]
        mask_text = "n/a" if mask_fraction is None else f"{mask_fraction:.4f}"
        lines.append(
            f"{('*' if index == 0 else ''):<5} "
            f"{row['mask_blocks']:>6d} "
            f"{row['teacher_future_block_ratio']:>7.3f} "
            f"{row['allocated_future_blocks']:>8d} "
            f"{mask_text:>10} "
            f"{row['finetune_lr']:>10.2e} "
            f"{row['student_val_latent_l1']:>11.7f} "
            f"{row['forecast_val_mae']:>11.7f} "
            f"{row['test_mae']:>11.7f} "
            f"{row['test_mse']:>11.7f} "
            f"{row['test_rmse']:>11.7f}"
        )

    summary = "\n".join(lines)
    print(summary)
    root.mkdir(parents=True, exist_ok=True)
    (root / "sweep_summary.txt").write_text(summary + "\n", encoding="utf-8")
    (root / "best_config.json").write_text(
        json.dumps(
            {
                "selection_metric": "Berkeley validation MAE",
                "best": best,
                "scratch_test": baseline,
                "all_rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
