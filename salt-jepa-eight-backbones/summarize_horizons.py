#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


BACKBONES = (
    "pdformer", "flashst", "patchstg", "testam",
    "patchtst", "staeformer", "stgormer", "tsformer",
)
HORIZONS = (6, 9, 12)
METHODS = (
    ("SALT", "target scratch"),
    ("SALT", "supervised source -> target full FT"),
    ("SALT", "SALT -> full FT"),
    ("JEPA", "JEPA -> full FT"),
)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize the single-seed 6/9/12-step forecasting sweep"
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.glob(f"*/horizon_*/seed_{args.seed}/summary.csv")):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise FileNotFoundError(f"No horizon results found under {root}")

    indexed = {}
    for row in rows:
        horizon = int(row.get("horizon") or path_horizon(row))
        key = (row["model"], horizon, row["framework"], row["method"])
        if key in indexed:
            raise ValueError(f"Duplicate horizon result: {key}")
        if int(row["seed"]) != args.seed:
            raise ValueError(f"Unexpected seed in {key}: {row['seed']}")
        indexed[key] = row

    expected = {
        (model, horizon, framework, method)
        for model in BACKBONES
        for horizon in HORIZONS
        for framework, method in METHODS
    }
    missing = sorted(expected - set(indexed))
    extra = sorted(set(indexed) - expected)
    if missing or extra:
        raise ValueError(
            f"Incomplete horizon sweep: missing={missing[:8]}"
            f"{' ...' if len(missing) > 8 else ''}; extra={extra[:8]}"
        )

    output_rows = []
    for key in sorted(expected, key=lambda item: (item[0], item[2], item[3], item[1])):
        row = indexed[key]
        output_rows.append({
            "model": key[0], "horizon": key[1],
            "framework": key[2], "method": key[3],
            "seed": args.seed,
            "mae": float(row["mae"]), "mse": float(row["mse"]),
            "rmse": float(row["rmse"]),
            "val_latent": row.get("val_latent", ""),
        })

    output_csv = root / f"horizon_summary_seed{args.seed}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Traffic forecasting horizon comparison (seed={args.seed})")
    print(
        f"{'Backbone':<12} {'Framework':<10} {'Method':<38} {'H':>3} "
        f"{'MAE':>11} {'MSE':>11} {'RMSE':>11}"
    )
    print("-" * 104)
    for row in output_rows:
        print(
            f"{row['model']:<12} {row['framework']:<10} {row['method']:<38} "
            f"{row['horizon']:>3d} {row['mae']:>11.7f} "
            f"{row['mse']:>11.7f} {row['rmse']:>11.7f}"
        )
    print(f"\nCSV: {output_csv}")


def path_horizon(row: dict) -> int:
    raise ValueError(
        f"Result row has no horizon field; rerun this experiment with the horizon-aware pipeline: {row}"
    )


if __name__ == "__main__":
    main()
