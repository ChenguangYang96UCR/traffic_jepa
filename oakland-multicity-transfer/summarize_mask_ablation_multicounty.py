#!/usr/bin/env python3
"""Summarize the 3-county Traffic-SALT masking ablation."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ORDER = [
    (mode, ratio)
    for ratio in ("1:9", "3:7", "5:5", "8:2")
    for mode in ("random_block", "topology_block")
] + [("random_point", "n/a")]
LABELS = {
    "random_block": "Random block",
    "topology_block": "Topology block",
    "random_point": "Random point",
}


def average(values):
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    source = args.root / "mask_ablation_results.csv"
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {source}")

    by_city = defaultdict(list)
    overall = defaultdict(list)
    for row in rows:
        key = (row["mask_mode"], row["history_future_ratio"])
        metrics = {name: float(row[name]) for name in ("mae", "mse", "rmse")}
        by_city[(row["county"], row["city"], *key)].append(metrics)
        overall[key].append(metrics)

    city_rows = []
    city_pairs = sorted({(row["county"], row["city"]) for row in rows})
    for county, city in city_pairs:
        for mode, ratio in ORDER:
            values = by_city.get((county, city, mode, ratio), [])
            if not values:
                continue
            city_rows.append({
                "county": county, "city": city,
                "mask_mode": mode, "history_future_ratio": ratio,
                "horizons_completed": len(values),
                **{name: average([item[name] for item in values])
                   for name in ("mae", "mse", "rmse")},
            })
    city_path = args.root / "mask_ablation_city_horizon_average.csv"
    with city_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(city_rows[0]))
        writer.writeheader()
        writer.writerows(city_rows)

    overall_rows = []
    for mode, ratio in ORDER:
        values = overall.get((mode, ratio), [])
        if not values:
            continue
        overall_rows.append({
            "mask_mode": mode, "history_future_ratio": ratio,
            "completed_city_horizon_runs": len(values),
            **{name: average([item[name] for item in values])
               for name in ("mae", "mse", "rmse")},
        })
    overall_rows.sort(key=lambda row: row["mae"])
    for rank, row in enumerate(overall_rows, 1):
        row["mae_rank"] = rank
    overall_path = args.root / "mask_ablation_overall_ranking.csv"
    with overall_path.open("w", newline="") as handle:
        fields = (
            "mae_rank", "mask_mode", "history_future_ratio",
            "completed_city_horizon_runs", "mae", "mse", "rmse",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(overall_rows)

    latex = [
        "% Requires: \\usepackage{booktabs}", "\\begin{table}[t]", "\\centering",
        "\\small", "\\begin{tabular}{llrrr}", "\\toprule",
        "Mask & History:Future & MAE & MSE & RMSE \\\\", "\\midrule",
    ]
    best = overall_rows[0]["mae"]
    for row in overall_rows:
        mae = f"{row['mae']:.4f}"
        if row["mae"] == best:
            mae = f"\\textbf{{{mae}}}"
        latex.append(
            f"{LABELS[row['mask_mode']]} & {row['history_future_ratio']} & "
            f"{mae} & {row['mse']:.4f} & {row['rmse']:.4f} \\\\"
        )
    latex.extend([
        "\\bottomrule", "\\end{tabular}",
        "\\caption{Traffic-SALT masking ablation averaged over three counties, "
        "two target cities per county, and horizons 6/9/12. Random point uses "
        "no prescribed history/future allocation.}",
        "\\label{tab:mask-ablation-multicounty}", "\\end{table}", "",
    ])
    tex_path = args.root / "mask_ablation_overall_table.tex"
    tex_path.write_text("\n".join(latex))
    print(source)
    print(city_path)
    print(overall_path)
    print(tex_path)


if __name__ == "__main__":
    main()
