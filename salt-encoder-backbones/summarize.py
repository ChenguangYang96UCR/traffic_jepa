#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.root).glob("*/summary.csv")):
        with path.open() as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise FileNotFoundError(f"No */summary.csv under {args.root}")
    print("SALT encoder-backbone comparison (target test set)")
    print(f"{'Backbone':<12} {'Method':<38} {'MAE':>11} {'MSE':>11} {'RMSE':>11} {'Val latent':>12}")
    print("-" * 101)
    for row in rows:
        latent = row.get("val_latent") or "n/a"
        print(
            f"{row['model']:<12} {row['method']:<38} "
            f"{float(row['mae']):>11.7f} {float(row['mse']):>11.7f} "
            f"{float(row['rmse']):>11.7f} {latent:>12}"
        )


if __name__ == "__main__":
    main()
