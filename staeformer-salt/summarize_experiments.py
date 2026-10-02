#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    experiments = [
        ("Fremont T -> Fremont S", "exp1_fremont_fremont"),
        ("Oakland T -> Fremont S", "exp2_oakland_fremont"),
        ("Oakland T -> Oakland S -> Fremont", "exp3_oakland_oakland_fremont"),
    ]
    methods = [
        ("Scratch", "staeformer_scratch"),
        ("Frozen", "salt_frozen"),
        ("Full FT", "salt_full"),
    ]
    lines = [
        f"{'Experiment':<39} {'Method':<10} {'MAE':>12} {'MSE':>12} {'RMSE':>12}",
        "-" * 91,
    ]
    for experiment, directory in experiments:
        path = args.root / directory / "all_results.json"
        if not path.exists():
            raise FileNotFoundError(path)
        results = json.loads(path.read_text(encoding="utf-8"))
        for method, key in methods:
            item = results[key]
            lines.append(
                f"{experiment:<39} {method:<10} "
                f"{item['mae']:>12.7f} {item['mse']:>12.7f} {item['rmse']:>12.7f}"
            )
    summary = "\n".join(lines)
    print(summary)
    (args.root / "three_experiment_summary.txt").write_text(summary + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
