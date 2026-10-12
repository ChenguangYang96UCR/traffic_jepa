#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METHOD_ORDER = [
    "ours", "pdformer", "flashst", "testam", "patchstg", "staeformer",
    "scjepa", "ts_jepa", "staeformer_jepa", "dcrnn", "agcrn",
]
METHOD_LABEL = {
    "ours": "Ours (STAEformer+SALT)",
    "pdformer": "PDFormer", "flashst": "FlashST", "testam": "TESTAM",
    "patchstg": "PatchSTG", "staeformer": "STAEformer",
    "scjepa": "SC-JEPA", "ts_jepa": "TS-JEPA",
    "staeformer_jepa": "STAEformer-JEPA",
    "dcrnn": "DCRNN", "agcrn": "AGCRN",
}


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Experiment output directory")
    return parser.parse_args()


def mean(values):
    return sum(values) / len(values) if values else None


def fmt(value):
    return "--" if value is None else f"{value:.4f}"


def main():
    root = Path(args().root)
    with (root / "results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = {}
    cities = []
    for row in rows:
        city, method, horizon = row["city"], row["method"], int(row["horizon"])
        if city not in cities:
            cities.append(city)
        values[city, method, horizon] = {
            "mae": float(row["mae"]), "mse": float(row["mse"]),
            "rmse": float(row["rmse"]),
        }
    methods = [method for method in METHOD_ORDER if any(r["method"] == method for r in rows)]
    horizons = (6, 9, 12)

    csv_path = root / "paper_table_mae_mse.csv"
    header = ["City", "Horizon"]
    for method in methods:
        header += [f"{METHOD_LABEL[method]} MSE", f"{METHOD_LABEL[method]} MAE"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for city in cities:
            for horizon in horizons:
                line = [city, horizon]
                for method in methods:
                    item = values.get((city, method, horizon))
                    line += ["" if item is None else item["mse"],
                             "" if item is None else item["mae"]]
                writer.writerow(line)
            line = [city, "Avg"]
            for method in methods:
                available = [values.get((city, method, h)) for h in horizons]
                line += [
                    "" if any(v is None for v in available) else mean([v["mse"] for v in available]),
                    "" if any(v is None for v in available) else mean([v["mae"] for v in available]),
                ]
            writer.writerow(line)

    columns = "ll" + "cc" * len(methods)
    latex = [
        "\\begin{table*}[t]", "\\centering", "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.2pt}", f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        "City & Horizon & " + " & ".join(
            f"\\multicolumn{{2}}{{c}}{{{METHOD_LABEL[m]}}}" for m in methods
        ) + " \\\\",
        "\\cmidrule(lr){3-" + str(2 + 2 * len(methods)) + "}",
        " & & " + " & ".join(["MSE & MAE"] * len(methods)) + " \\\\",
        "\\midrule",
    ]
    for city_index, city in enumerate(cities):
        for row_index, horizon in enumerate((*horizons, "Avg")):
            cells = [f"\\multirow{{4}}{{*}}{{{city}}}" if row_index == 0 else "", str(horizon)]
            for method in methods:
                if horizon == "Avg":
                    available = [values.get((city, method, h)) for h in horizons]
                    item = None if any(v is None for v in available) else {
                        "mse": mean([v["mse"] for v in available]),
                        "mae": mean([v["mae"] for v in available]),
                    }
                else:
                    item = values.get((city, method, horizon))
                cells += [fmt(None if item is None else item["mse"]),
                          fmt(None if item is None else item["mae"])]
            latex.append(" & ".join(cells) + " \\\\")
        if city_index + 1 < len(cities):
            latex.append("\\midrule")
    latex += [
        "\\bottomrule", "\\end{tabular}",
        "\\caption{Oakland-to-city transfer forecasting. All transferred models use full fine-tuning; Avg is the arithmetic mean over horizons 6, 9, and 12.}",
        "\\label{tab:oakland-multicity-transfer}", "\\end{table*}", "",
    ]
    (root / "paper_table_mae_mse.tex").write_text("\n".join(latex), encoding="utf-8")
    print(csv_path)
    print(root / "paper_table_mae_mse.tex")


if __name__ == "__main__":
    main()
