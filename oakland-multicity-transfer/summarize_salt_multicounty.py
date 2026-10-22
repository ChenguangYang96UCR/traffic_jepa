#!/usr/bin/env python3
"""Merge new SALT-only runs with previously computed comparison results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


BACKBONES = (
    "pdformer", "flashst", "testam", "patchstg", "scjepa",
    "ts_jepa", "dcrnn", "agcrn", "sdgformer", "dstfgcn",
)
DISPLAY = {
    "pdformer": "PDFormer", "flashst": "FlashST", "testam": "TESTAM",
    "patchstg": "PatchSTG", "scjepa": "SC-JEPA", "ts_jepa": "TS-JEPA",
    "dcrnn": "DCRNN", "agcrn": "AGCRN", "sdgformer": "SDGFormer",
    "dstfgcn": "DSTFGCN",
}


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values):
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--baseline-csv", action="append", default=[],
        help="Earlier ordinary-transfer results.csv; repeat for each county run",
    )
    args = parser.parse_args()
    root = Path(args.root)
    salt_path = root / "salt_results.csv"
    if not salt_path.exists():
        raise FileNotFoundError(salt_path)
    salt_rows = read_csv(salt_path)
    baseline_rows = []
    for path in args.baseline_csv:
        baseline_rows.extend(read_csv(path))

    # Keep only the methods displayed in the paper table. Ours is retained as
    # the established reference but is never retrained by this pipeline.
    allowed = {"ours", *BACKBONES}
    baseline_rows = [row for row in baseline_rows if row.get("method") in allowed]
    baseline_index = {}
    for row in baseline_rows:
        key = (row.get("source_city"), row.get("city"), row.get("method"), row.get("horizon"))
        baseline_index[key] = row

    comparisons = []
    for salt in salt_rows:
        backbone = salt["backbone"]
        key = (salt["source_city"], salt["city"], backbone, salt["horizon"])
        baseline = baseline_index.get(key)
        baseline_mae = float(baseline["mae"]) if baseline else None
        salt_mae = float(salt["mae"])
        comparisons.append({
            "county": salt["county"], "source_city": salt["source_city"],
            "city": salt["city"], "horizon": salt["horizon"],
            "backbone": backbone,
            "baseline_mae": "" if baseline_mae is None else baseline_mae,
            "salt_mae": salt_mae,
            "mae_delta_salt_minus_baseline": (
                "" if baseline_mae is None else salt_mae - baseline_mae
            ),
            "salt_relative_gain_pct": (
                "" if baseline_mae is None else 100 * (baseline_mae - salt_mae) / baseline_mae
            ),
            "salt_mse": salt["mse"], "salt_rmse": salt["rmse"],
            "val_latent": salt.get("val_latent", ""),
        })

    comparison_path = root / "salt_vs_existing_detailed.csv"
    with comparison_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    grouped = defaultdict(lambda: {"baseline": [], "salt": []})
    for row in comparisons:
        grouped[row["backbone"]]["salt"].append(float(row["salt_mae"]))
        if row["baseline_mae"] != "":
            grouped[row["backbone"]]["baseline"].append(float(row["baseline_mae"]))
    summary = []
    for backbone in BACKBONES:
        baseline = mean(grouped[backbone]["baseline"])
        salt = mean(grouped[backbone]["salt"])
        summary.append({
            "backbone": backbone,
            "completed_salt_runs": len(grouped[backbone]["salt"]),
            "matched_baseline_runs": len(grouped[backbone]["baseline"]),
            "existing_transfer_mae": "" if baseline is None else baseline,
            "salt_transfer_mae": "" if salt is None else salt,
            "salt_relative_gain_pct": (
                "" if baseline is None or salt is None
                else 100 * (baseline - salt) / baseline
            ),
        })
    summary_path = root / "salt_vs_existing_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    # Long-form merged file supports rebuilding the paper table without
    # rerunning any comparison method.
    combined = []
    for row in baseline_rows:
        combined.append({
            "source_city": row.get("source_city", ""), "city": row.get("city", ""),
            "method": row.get("method", ""), "horizon": row.get("horizon", ""),
            "history": row.get("history", ""), "mae": row.get("mae", ""),
            "mse": row.get("mse", ""), "rmse": row.get("rmse", ""),
            "seed": row.get("seed", ""), "transfer": row.get("transfer", ""),
            "result_source": "reused_existing",
        })
    for row in salt_rows:
        combined.append({
            "source_city": row["source_city"], "city": row["city"],
            "method": row["method"], "horizon": row["horizon"],
            "history": row["history"], "mae": row["mae"], "mse": row["mse"],
            "rmse": row["rmse"], "seed": row["seed"],
            "transfer": row["transfer"], "result_source": "new_salt_run",
        })
    combined.sort(key=lambda row: (
        row["source_city"], row["city"], int(row["horizon"]), row["method"],
    ))
    combined_path = root / "combined_existing_and_salt_results.csv"
    with combined_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)

    latex = [
        "% Requires: \\usepackage{booktabs}", "\\begin{table}[t]", "\\centering",
        "\\small", "\\begin{tabular}{lrrr}", "\\toprule",
        "Backbone & Existing transfer & +SALT & Rel. gain \\\\", "\\midrule",
    ]
    for row in summary:
        old = row["existing_transfer_mae"]
        salt = row["salt_transfer_mae"]
        gain = row["salt_relative_gain_pct"]
        latex.append(
            f"{DISPLAY[row['backbone']]} & "
            f"{'--' if old == '' else f'{old:.4f}'} & "
            f"{'--' if salt == '' else f'{salt:.4f}'} & "
            f"{'--' if gain == '' else f'{gain:+.2f}\\%'} \\\\"
        )
    latex.extend([
        "\\bottomrule", "\\end{tabular}",
        "\\caption{Mean MAE over the matched county, city, and horizon runs. "
        "Existing comparison results are reused; only the SALT variants are newly trained.}",
        "\\label{tab:salt-backbone-ablation}", "\\end{table}", "",
    ])
    (root / "salt_vs_existing_summary.tex").write_text("\n".join(latex))
    print(comparison_path)
    print(summary_path)
    print(combined_path)


if __name__ == "__main__":
    main()
