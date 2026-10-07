#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def rows(root: str):
    result = []
    for path in sorted(Path(root).glob("*/summary.csv")):
        with path.open() as handle:
            result.extend(csv.DictReader(handle))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--salt-root", required=True)
    parser.add_argument("--jepa-root", required=True)
    args = parser.parse_args()
    values = rows(args.salt_root) + rows(args.jepa_root)
    if not values:
        raise FileNotFoundError("No per-model summary.csv files found")
    print("SALT vs JEPA encoder-pretraining comparison (Berkeley test set)")
    print(f"{'Backbone':<12} {'Framework':<10} {'Method':<34} {'MAE':>11} {'MSE':>11} {'RMSE':>11}")
    print("-" * 94)
    for row in values:
        print(
            f"{row['model']:<12} {row.get('framework', 'unknown'):<10} "
            f"{row['method']:<34} {float(row['mae']):>11.7f} "
            f"{float(row['mse']):>11.7f} {float(row['rmse']):>11.7f}"
        )


if __name__ == "__main__":
    main()
