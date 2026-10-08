#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("mae", "mse", "rmse")


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot aggregate an empty metric")
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.fmean(values), deviation


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate target-test metrics across independent random seeds"
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-seeds", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.glob("*/seed_*/summary.csv")):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise FileNotFoundError(f"No */seed_*/summary.csv under {root}")

    grouped = defaultdict(list)
    for row in rows:
        key = (row["model"], row.get("framework", "unknown"), row["method"])
        grouped[key].append(row)

    aggregate_rows = []
    for key, group in sorted(grouped.items()):
        seeds = sorted({int(row["seed"]) for row in group})
        if len(group) != len(seeds):
            raise ValueError(f"Duplicate seed rows for {key}: {len(group)} rows, seeds={seeds}")
        if len(seeds) != args.expected_seeds:
            raise ValueError(
                f"{key} has {len(seeds)} seeds {seeds}; expected {args.expected_seeds}"
            )
        output = {
            "model": key[0], "framework": key[1], "method": key[2],
            "n_seeds": len(seeds), "seeds": " ".join(map(str, seeds)),
        }
        for metric in METRICS:
            average, deviation = mean_std([float(row[metric]) for row in group])
            output[f"{metric}_mean"] = average
            output[f"{metric}_std"] = deviation
        latent = [float(row["val_latent"]) for row in group if row.get("val_latent")]
        if len(latent) == len(group):
            output["val_latent_mean"], output["val_latent_std"] = mean_std(latent)
        else:
            output["val_latent_mean"] = output["val_latent_std"] = ""
        aggregate_rows.append(output)

    output_csv = root / "summary_mean_std.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"{args.expected_seeds}-seed traffic forecasting comparison "
          "(target test set; sample std, ddof=1)")
    print(
        f"{'Backbone':<12} {'Framework':<10} {'Method':<38} "
        f"{'MAE mean +/- std':>24} {'MSE mean +/- std':>24} "
        f"{'RMSE mean +/- std':>24}"
    )
    print("-" * 138)
    for row in aggregate_rows:
        values = [
            f"{row[f'{metric}_mean']:.7f} +/- {row[f'{metric}_std']:.7f}"
            for metric in METRICS
        ]
        print(
            f"{row['model']:<12} {row['framework']:<10} {row['method']:<38} "
            f"{values[0]:>24} {values[1]:>24} {values[2]:>24}"
        )
    print(f"\nCSV: {output_csv}")


if __name__ == "__main__":
    main()
