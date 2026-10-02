#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import re


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target-city", default="Fremont")
    parser.add_argument("--source-city", default="Oakland")
    args = parser.parse_args()
    target = args.target_city
    source = args.source_city
    target_slug = slugify(target)
    source_slug = slugify(source)
    experiments = [
        (f"{target} T -> {target} S", f"exp1_{target_slug}_{target_slug}"),
        (f"{source} T -> {target} S", f"exp2_{source_slug}_{target_slug}"),
        (
            f"{source} T -> {source} S -> {target}",
            f"exp3_{source_slug}_{source_slug}_{target_slug}",
        ),
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
    (args.root / "three_experiment_summary.txt").write_text(
        summary + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
