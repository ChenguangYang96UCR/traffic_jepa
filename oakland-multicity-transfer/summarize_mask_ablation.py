#!/usr/bin/env python3
"""Combine Traffic-SALT mask-ablation runs into CSV and LaTeX tables."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


MASKS = ("random_st", "topology_st")
LABELS = {"random_st": "Random ST", "topology_st": "Topology-aware ST"}
HORIZONS = (6, 9, 12)


def mean(values):
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for mask in MASKS:
        path = args.root / mask / "results.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["method"] != "ours":
                    continue
                row = dict(row)
                row["mask_mode"] = mask
                rows.append(row)

    cities = []
    values = {}
    for row in rows:
        city = row["city"]
        if city not in cities:
            cities.append(city)
        values[city, row["mask_mode"], int(row["horizon"])] = {
            metric: float(row[metric]) for metric in ("mae", "mse", "rmse")
        }

    output_csv = args.root / "mask_ablation_results.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("city", "horizon", "mask_mode", "mae", "mse", "rmse"))
        for city in cities:
            for mask in MASKS:
                for horizon in HORIZONS:
                    item = values[city, mask, horizon]
                    writer.writerow((city, horizon, mask, item["mae"],
                                     item["mse"], item["rmse"]))
                writer.writerow((
                    city, "Avg", mask,
                    mean([values[city, mask, h]["mae"] for h in HORIZONS]),
                    mean([values[city, mask, h]["mse"] for h in HORIZONS]),
                    mean([values[city, mask, h]["rmse"] for h in HORIZONS]),
                ))

    latex = [
        "% Requires: \\usepackage{booktabs,multirow,graphicx}",
        "\\begin{table}[t]", "\\centering", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{c|c|cc|cc}", "\\toprule",
        "\\multicolumn{2}{c|}{Mask} & "
        "\\multicolumn{2}{c|}{Random ST} & "
        "\\multicolumn{2}{c}{Topology-aware ST} \\\\",
        "\\midrule",
        "\\multicolumn{2}{c|}{Metric} & MSE & MAE & MSE & MAE \\\\",
        "\\midrule",
    ]
    for city_index, city in enumerate(cities):
        for row_index, horizon in enumerate((*HORIZONS, "Avg")):
            prefix = (
                f"\\multirow{{4}}{{*}}{{\\rotatebox{{90}}{{{city}}}}} & {horizon}"
                if row_index == 0 else f"& {horizon}"
            )
            items = {}
            for mask in MASKS:
                if horizon == "Avg":
                    items[mask] = {
                        metric: mean([values[city, mask, h][metric] for h in HORIZONS])
                        for metric in ("mse", "mae")
                    }
                else:
                    items[mask] = values[city, mask, horizon]
            minimum = {
                metric: min(items[mask][metric] for mask in MASKS)
                for metric in ("mse", "mae")
            }
            cells = []
            for mask in MASKS:
                for metric in ("mse", "mae"):
                    value = items[mask][metric]
                    text = f"{value:.4f}"
                    if value == minimum[metric]:
                        text = f"\\textbf{{{text}}}"
                    cells.append(text)
            latex.append(prefix + " & " + " & ".join(cells) + " \\\\")
        if city_index + 1 < len(cities):
            latex.append("\\midrule")
    latex += [
        "\\bottomrule", "\\end{tabular}",
        "\\caption{Traffic-SALT masking ablation. Both strategies use the same "
        "mask-size distribution, future-block ratio, source data, and target "
        "fine-tuning protocol.}",
        "\\label{tab:mask-ablation}", "\\end{table}", "",
    ]
    output_tex = args.root / "mask_ablation_table.tex"
    output_tex.write_text("\n".join(latex), encoding="utf-8")
    print(output_csv)
    print(output_tex)


if __name__ == "__main__":
    main()
