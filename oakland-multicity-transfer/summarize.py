#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METHOD_ORDER = [
    "ours", "pdformer", "flashst", "testam", "patchstg",
    "scjepa", "ts_jepa", "dcrnn", "agcrn", "sdgformer", "dstfgcn",
]
METHOD_LABEL = {
    "ours": "Ours (STAEformer+SALT)",
    "pdformer": "PDFormer (AAAI'23)",
    "flashst": "FlashST (ICML'24)",
    "testam": "TESTAM (ICLR'24)",
    "patchstg": "PatchSTG (KDD'25)",
    "scjepa": "SC-JEPA (SDM'26)", "ts_jepa": "TS-JEPA",
    "dcrnn": "DCRNN", "agcrn": "AGCRN",
    "sdgformer": "SDGFormer (Neurocomputing'26)",
    "dstfgcn": "DSTFGCN (Neural Networks'26)",
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
    preferred_cities = [
        "Berkeley", "Dublin", "Newark", "San Leandro", "Pleasanton", "Livermore",
        "Laguna Niguel", "Newport Beach", "Westminster", "Fullerton",
        "Buena Park", "Mission Viejo",
    ]
    cities = [city for city in preferred_cities if city in cities] + [
        city for city in cities if city not in preferred_cities
    ]
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

    columns = "c|c|" + "|".join("cc" for _ in methods)
    display_label = {
        method: ("(\\textbf{Ours})" if method == "ours" else METHOD_LABEL[method])
        for method in methods
    }
    latex = [
        "% Requires: \\usepackage{booktabs,multirow,graphicx}",
        "\\begin{table*}[h!]", "\\centering",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\resizebox{0.98\\textwidth}{!}{",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        "\\multicolumn{2}{c|}{Methods} & " + " & ".join(
            f"\\multicolumn{{2}}{{{'c' if index + 1 == len(methods) else 'c|'}}}"
            f"{{{display_label[m]}}}"
            for index, m in enumerate(methods)
        ) + " \\\\",
        "\\midrule",
        "\\multicolumn{2}{c|}{Metric} & " + " & ".join(["MSE & MAE"] * len(methods)) + " \\\\",
        "\\midrule",
    ]
    for city_index, city in enumerate(cities):
        for row_index, horizon in enumerate((*horizons, "Avg")):
            cells = [
                f"\\multirow{{4}}{{*}}{{\\rotatebox{{90}}{{{city}}}}}"
                if row_index == 0 else "",
                str(horizon),
            ]
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
    source_cities = sorted({row["source_city"] for row in rows})
    source_label = source_cities[0] if len(source_cities) == 1 else "Cross-city"
    source_slug = source_label.casefold().replace(" ", "-")
    latex += [
        "\\bottomrule", "\\end{tabular}}",
        f"\\caption{{{source_label}-to-city transfer forecasting results. `Avg' denotes the arithmetic mean across prediction horizons $\\{{6,9,12\\}}$. All methods use full fine-tuning on the target city.}}",
        f"\\label{{tab:{source_slug}-multicity-transfer}}", "\\end{table*}", "",
    ]
    (root / "paper_table_mae_mse.tex").write_text("\n".join(latex), encoding="utf-8")
    print(csv_path)
    print(root / "paper_table_mae_mse.tex")


if __name__ == "__main__":
    main()
