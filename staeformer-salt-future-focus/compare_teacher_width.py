#!/usr/bin/env python3
"""Compare d=32 and d=128 future-focused Oakland Teachers on Berkeley."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--teacher128-root", type=Path, required=True)
    args = parser.parse_args()

    baseline_dir = args.baseline_root / "exp2_oakland_berkeley" / "all"
    baseline = load(baseline_dir / "all_results.json")
    baseline_latent = load(
        baseline_dir / "student" / "distillation_metrics.json"
    )["validation"]["scope"]
    teacher128 = load(args.teacher128_root / "all_results.json")
    teacher128_latent = load(
        args.teacher128_root / "student" / "distillation_metrics.json"
    )["validation"]["scope"]

    rows = [
        ("Teacher d=32", "Frozen", baseline_latent, baseline["salt_frozen"]),
        ("Teacher d=32", "Full FT", baseline_latent, baseline["salt_full"]),
        ("Teacher d=128", "Frozen", teacher128_latent, teacher128["salt_frozen"]),
        ("Teacher d=128", "Full FT", teacher128_latent, teacher128["salt_full"]),
    ]
    lines = [
        "Future-focused SALT Teacher width comparison",
        "Oakland Teacher -> Berkeley Student | all-step latent loss",
        (
            f"{'Teacher':<16} {'Method':<10} {'Val latent':>12} "
            f"{'MAE':>12} {'MSE':>12} {'RMSE':>12}"
        ),
        "-" * 79,
    ]
    for teacher, method, latent, item in rows:
        lines.append(
            f"{teacher:<16} {method:<10} {latent:>12.7f} "
            f"{item['mae']:>12.7f} {item['mse']:>12.7f} {item['rmse']:>12.7f}"
        )
    summary = "\n".join(lines)
    print(summary)
    (args.teacher128_root / "teacher_width_comparison.txt").write_text(
        summary + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
