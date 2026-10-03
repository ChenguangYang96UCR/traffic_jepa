#!/usr/bin/env python3
"""Summarize future-focused Teacher and Student-scope ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    experiments = [
        ("Berkeley T -> Berkeley S", "exp1_berkeley_berkeley"),
        ("Oakland T -> Berkeley S", "exp2_oakland_berkeley"),
        (
            "Oakland T -> Oakland S -> Berkeley",
            "exp3_oakland_oakland_berkeley",
        ),
    ]
    scratch = read_json(
        args.root / "exp1_berkeley_berkeley" / "all" / "all_results.json"
    )["staeformer_scratch"]

    lines = [
        "Future-focused SALT evaluation (Berkeley test set)",
        (
            f"{'Experiment':<39} {'Latent loss':<12} {'Method':<9} "
            f"{'Val latent':>11} {'MAE':>12} {'MSE':>12} {'RMSE':>12}"
        ),
        "-" * 113,
        (
            f"{'Berkeley downstream':<39} {'n/a':<12} {'Scratch':<9} "
            f"{'n/a':>11} {scratch['mae']:>12.7f} {scratch['mse']:>12.7f} "
            f"{scratch['rmse']:>12.7f}"
        ),
    ]
    for experiment, directory in experiments:
        for scope in ("all", "future"):
            base = args.root / directory / scope
            results = read_json(base / "all_results.json")
            distillation = read_json(base / "student" / "distillation_metrics.json")
            latent = distillation["validation"]["scope"]
            for method, key in (("Frozen", "salt_frozen"), ("Full FT", "salt_full")):
                item = results[key]
                lines.append(
                    f"{experiment:<39} {scope:<12} {method:<9} "
                    f"{latent:>11.7f} {item['mae']:>12.7f} {item['mse']:>12.7f} "
                    f"{item['rmse']:>12.7f}"
                )
    summary = "\n".join(lines)
    print(summary)
    (args.root / "future_focus_summary.txt").write_text(
        summary + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
