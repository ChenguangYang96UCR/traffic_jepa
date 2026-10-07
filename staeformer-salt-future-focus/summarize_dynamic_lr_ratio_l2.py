#!/usr/bin/env python3
"""Summarize scheduler, mask-ratio, and latent L1/L2 controlled ablations."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


RESULT_RE = re.compile(r"ratio=([^/]+)/loss=([^/]+)/scheduler=([^/]+)")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def student_validation(root: Path, base_root: Path | None, ratio: str, loss: str):
    candidates = [
        root / "students" / f"ratio={ratio}" / f"loss={loss}" / "student" / "distillation_metrics.json"
    ]
    if loss == "l1" and base_root is not None:
        candidates.append(
            base_root / "students" / f"blocks=8_ratio={ratio}" /
            "student" / "distillation_metrics.json"
        )
    for path in candidates:
        if path.exists():
            payload = read(path)
            validation = payload["validation"]
            selected = validation.get(f"scope_{loss}", validation.get("scope"))
            return {
                "student_val_latent": selected,
                "student_val_latent_l1": validation.get("scope_l1", validation.get("scope")),
                "student_val_latent_l2": validation.get("scope_l2"),
            }
    return {
        "student_val_latent": None,
        "student_val_latent_l1": None,
        "student_val_latent_l2": None,
    }


def collect(root: Path, base_root: Path | None):
    rows = []
    for path in sorted(root.glob("results/ratio=*/loss=*/scheduler=*/all_results.json")):
        match = RESULT_RE.search(str(path.parent.relative_to(root / "results")))
        if match is None:
            continue
        ratio, loss, scheduler = match.groups()
        result = read(path)["salt_full"]
        validation = result.get("selection_validation")
        if validation is None:
            raise ValueError(f"{path} lacks selection_validation")
        config = result.get("training_config", {})
        latent = student_validation(root, base_root, ratio, loss)
        rows.append({
            "ratio": float(ratio),
            "latent_loss": loss,
            "scheduler": scheduler,
            "best_epoch": config.get("best_epoch"),
            "val_mae": validation["mae"],
            "test_mae": result["mae"],
            "test_mse": result["mse"],
            "test_rmse": result["rmse"],
            "path": str(path),
            **latent,
        })
    return rows


def section(title: str, rows: list[dict]) -> list[str]:
    rows = sorted(rows, key=lambda x: (x["val_mae"], x["test_mae"]))
    lines = [
        "",
        title,
        "Selection: lowest Berkeley validation MAE; test metrics are report-only",
        (
            f"{'Best':<5} {'Ratio':>7} {'Latent':>8} {'Scheduler':>11} "
            f"{'Epoch':>7} {'Val latent':>11} {'Val MAE':>11} {'Test MAE':>11} "
            f"{'Test MSE':>11} {'Test RMSE':>11}"
        ),
        "-" * 104,
    ]
    for index, row in enumerate(rows):
        epoch = "n/a" if row["best_epoch"] is None else str(row["best_epoch"])
        latent = row["student_val_latent"]
        latent_text = "n/a" if latent is None else f"{latent:.7f}"
        lines.append(
            f"{('*' if index == 0 else ''):<5} {row['ratio']:>7.3f} "
            f"{row['latent_loss']:>8} {row['scheduler']:>11} {epoch:>7} "
            f"{latent_text:>11} {row['val_mae']:>11.7f} {row['test_mae']:>11.7f} "
            f"{row['test_mse']:>11.7f} {row['test_rmse']:>11.7f}"
        )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-sweep-root", type=Path)
    args = parser.parse_args()
    rows = collect(args.root, args.base_sweep_root)
    if not rows:
        raise FileNotFoundError(f"No completed results below {args.root}")

    scheduler_rows = [r for r in rows if r["ratio"] == 0.5 and r["latent_loss"] == "l1"]
    ratio_rows = [r for r in rows if r["latent_loss"] == "l1" and r["scheduler"] == "constant"]
    loss_rows = [r for r in rows if r["ratio"] == 0.5 and r["scheduler"] == "constant"]
    lines = [
        "Oakland Teacher d=128 -> Berkeley Student | controlled SALT extensions",
        "Fixed: 8 mask blocks, all-step latent matching, full fine-tuning, base LR=1e-3",
        "Reference reported best: val MAE=0.0537047 test MAE=0.0540987 test MSE=0.0122153 test RMSE=0.1105226",
    ]
    lines += section("A. Fine-tuning learning-rate scheduler", scheduler_rows)
    lines += section("B. Teacher future-block ratio", ratio_rows)
    lines += section("C. Student latent objective", loss_rows)
    summary = "\n".join(lines)
    print(summary)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "dynamic_lr_ratio_l2_summary.txt").write_text(summary + "\n", encoding="utf-8")

    def best(items):
        return min(items, key=lambda x: (x["val_mae"], x["test_mae"])) if items else None

    (args.root / "dynamic_lr_ratio_l2_results.json").write_text(
        json.dumps({
            "selection_metric": "Berkeley validation MAE",
            "test_policy": "report-only",
            "best_scheduler": best(scheduler_rows),
            "best_ratio": best(ratio_rows),
            "best_latent_loss": best(loss_rows),
            "all_unique_rows": rows,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
