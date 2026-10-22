#!/usr/bin/env python3
"""Plot MAE–efficiency bubble chart for traffic forecasting models.

Default encoding:
  x-axis      : batch-1 inference latency (ms)
  y-axis      : MAE (lower is better)
  bubble area : number of model parameters
  color       : model family

Usage:
  python plot_mae_efficiency_bubble.py
  python plot_mae_efficiency_bubble.py --output traffic_efficiency
  python plot_mae_efficiency_bubble.py --x latency_b32_ms
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


MODELS = [
    {"method": "Ours", "family": "Ours", "mae": 0.0421164582,
     "parameters": 796458, "latency_b1_ms": 0.9911403,
     "latency_b32_ms": 1.7378656, "model_size_mb": 3.0382462,
     "peak_memory_b32_mb": 29.3583984},
    {"method": "PDFormer", "family": "Transformer", "mae": 0.0869490294,
     "parameters": 292974, "latency_b1_ms": 2.9537625,
     "latency_b32_ms": 2.9690762, "model_size_mb": 1.1427002,
     "peak_memory_b32_mb": 33.0625},
    {"method": "FlashST", "family": "Transformer", "mae": 0.0663192182,
     "parameters": 360351, "latency_b1_ms": 3.6075577,
     "latency_b32_ms": 3.6492529, "model_size_mb": 1.4019356,
     "peak_memory_b32_mb": 34.9375},
    {"method": "TESTAM", "family": "Transformer", "mae": 0.0499820110,
     "parameters": 191252, "latency_b1_ms": 6.6951471,
     "latency_b32_ms": 6.6393909, "model_size_mb": 0.7295685,
     "peak_memory_b32_mb": 24.5996094},
    {"method": "PatchSTG", "family": "Transformer", "mae": 0.0452410941,
     "parameters": 122061, "latency_b1_ms": 1.1994171,
     "latency_b32_ms": 1.1867085, "model_size_mb": 0.4656258,
     "peak_memory_b32_mb": 8.7011719},
    {"method": "SC-JEPA", "family": "JEPA", "mae": 0.0524720468,
     "parameters": 2233228, "latency_b1_ms": 0.9490893,
     "latency_b32_ms": 0.9288551, "model_size_mb": 8.5190887,
     "peak_memory_b32_mb": 13.5322266},
    {"method": "TS-JEPA", "family": "JEPA", "mae": 0.0720448852,
     "parameters": 222220, "latency_b1_ms": 0.2873498,
     "latency_b32_ms": 0.2882766, "model_size_mb": 0.8477020,
     "peak_memory_b32_mb": 8.1787109},
    {"method": "STAEformer-JEPA", "family": "JEPA", "mae": 0.0654236706,
     "parameters": 635437, "latency_b1_ms": 1.2993997,
     "latency_b32_ms": 1.9051323, "model_size_mb": 2.4239998,
     "peak_memory_b32_mb": 22.3574219},
    {"method": "DCRNN", "family": "GNN", "mae": 0.0485580593,
     "parameters": 25932, "latency_b1_ms": 1.4271539,
     "latency_b32_ms": 1.6406471, "model_size_mb": 0.0993042,
     "peak_memory_b32_mb": 1.7094727},
    {"method": "AGCRN", "family": "GNN", "mae": 0.0493124183,
     "parameters": 26092, "latency_b1_ms": 1.4297499,
     "latency_b32_ms": 1.6456445, "model_size_mb": 0.0999146,
     "peak_memory_b32_mb": 1.7099609},
    {"method": "SDGFormer", "family": "Transformer", "mae": 0.0938068467,
     "parameters": 439053, "latency_b1_ms": 5.4009296,
     "latency_b32_ms": 5.4221515, "model_size_mb": 1.6995735,
     "peak_memory_b32_mb": 17.8100586},
    {"method": "DSTFGCN", "family": "GNN", "mae": 0.0441674006,
     "parameters": 334036, "latency_b1_ms": 6.2929303,
     "latency_b32_ms": 6.3890955, "model_size_mb": 1.2795410,
     "peak_memory_b32_mb": 24.9223633},
]


X_LABELS = {
    "latency_b1_ms": "Inference Latency (ms)",
    "latency_b32_ms": "Inference Latency (ms)",
    "parameters": "Number of Parameters",
    "model_size_mb": "Model Size (MB)",
    "peak_memory_b32_mb": "Peak GPU Memory, Batch Size 32 (MB)",
}

COLORS = {
    "Ours": "#F4B000",
    "Transformer": "#3CB7AE",
    "JEPA": "#4B83C3",
    "GNN": "#8A68B8",
}

# Label locations for the default batch-1 latency plot. Values are offsets
# measured in screen points. The crowded 0.9--1.5 ms region is deliberately
# fanned out with short leader lines.
LABEL_OFFSETS_B1 = {
    "Ours": (-48, -10),
    "PatchSTG": (-43, 25),
    "DCRNN": (30, -13),
    "AGCRN": (39, 14),
    "SC-JEPA": (-34, 23),
    "TS-JEPA": (30, 0),
    "STAEformer-JEPA": (0, -25),
    "DSTFGCN": (-22, 18),
    "TESTAM": (-25, 18),
    "PDFormer": (0, 17),
    "FlashST": (0, -20),
    "SDGFormer": (0, 18),
}


def bubble_areas(parameters: np.ndarray) -> np.ndarray:
    """Map parameter counts to readable marker areas without hiding small models."""
    log_p = np.log10(parameters)
    normalized = (log_p - log_p.min()) / (log_p.max() - log_p.min())
    return 75.0 + normalized * 390.0


def plot(x_key: str, output: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    params = np.asarray([m["parameters"] for m in MODELS], dtype=float)
    areas = bubble_areas(params)

    fig, ax = plt.subplots(figsize=(9.4, 5.4), dpi=180)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.28)

    for family in COLORS:
        indices = [i for i, m in enumerate(MODELS) if m["family"] == family]
        ax.scatter(
            [MODELS[i][x_key] for i in indices],
            [MODELS[i]["mae"] for i in indices],
            s=[areas[i] for i in indices],
            c=COLORS[family],
            edgecolors="white",
            linewidths=1.2,
            alpha=0.90,
            zorder=3 if family != "Ours" else 5,
        )

    for model in MODELS:
        if x_key == "latency_b1_ms":
            dx, dy = LABEL_OFFSETS_B1.get(model["method"], (0, 12))
        else:
            dx, dy = (0, 11)
        ax.annotate(
            model["method"],
            xy=(model[x_key], model["mae"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=7.7,
            fontweight="bold" if model["method"] == "Ours" else "normal",
            color="#202124",
            bbox=dict(
                boxstyle="round,pad=0.13",
                facecolor="white",
                edgecolor="none",
                alpha=0.78,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="#8A8A8A",
                linewidth=0.6,
                shrinkA=2,
                shrinkB=5,
            ),
            zorder=6,
        )

    ax.set_xlabel(X_LABELS[x_key], fontweight="bold")
    ax.set_ylabel("MAE", fontweight="bold")
    ax.margins(x=0.11, y=0.17)

    if x_key == "parameters":
        ax.set_xscale("log")

    ax.tick_params(axis="both", which="major", width=1.8, length=6)
    ax.tick_params(axis="both", which="minor", width=1.4, length=3)
    for tick_label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick_label.set_fontweight("bold")

    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", label=family,
            markerfacecolor=color, markeredgecolor="white",
            markeredgewidth=1.0, markersize=8,
        )
        for family, color in COLORS.items()
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        title="Model family",
        borderpad=0.65,
        labelspacing=0.55,
        handletextpad=0.5,
        borderaxespad=0,
    )
    legend.get_frame().set_alpha(0.92)
    legend.get_frame().set_linewidth(0.7)


    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)

    fig.subplots_adjust(left=0.10, right=0.80, top=0.96, bottom=0.20)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {output.with_suffix('.png')}")
    print(f"Saved {output.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--x",
        choices=tuple(X_LABELS),
        default="latency_b1_ms",
        help="Metric used on the x-axis (default: latency_b1_ms).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mae_efficiency_bubble"),
        help="Output path without extension.",
    )
    args = parser.parse_args()
    plot(args.x, args.output)


if __name__ == "__main__":
    main()
