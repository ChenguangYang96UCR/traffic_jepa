#!/usr/bin/env python3
"""Extract Fremont node features from Alameda incident samples."""

import argparse
import csv
import importlib
import sys
from pathlib import Path

import numpy as np


def enable_numpy_pickle_compatibility():
    """Allow NumPy 1.x to load object arrays saved by NumPy 2.x."""
    try:
        importlib.import_module("numpy._core")
        return
    except ModuleNotFoundError:
        pass

    numpy_core = importlib.import_module("numpy.core")
    sys.modules.setdefault("numpy._core", numpy_core)

    for module_name in (
        "multiarray",
        "numeric",
        "_multiarray_umath",
        "umath",
    ):
        old_module = importlib.import_module(
            f"numpy.core.{module_name}"
        )
        sys.modules.setdefault(
            f"numpy._core.{module_name}",
            old_module,
        )


enable_numpy_pickle_compatibility()


def slice_node_axis(array, indices, original_node_count):
    """Slice the unique axis whose size equals the original node count."""
    array = np.asarray(array)
    matching_axes = [
        axis
        for axis, size in enumerate(array.shape)
        if size == original_node_count
    ]

    if len(matching_axes) != 1:
        raise ValueError(
            f"Expected exactly one node axis of size {original_node_count}, "
            f"but found axes {matching_axes} in shape {array.shape}"
        )

    return np.take(array, indices, axis=matching_axes[0])


def process_split(source_path, output_path, indices, original_node_count):
    """Extract Fremont nodes while preserving sample order and metadata."""
    samples = np.load(source_path, allow_pickle=True)
    output_samples = []

    for sample_index, sample in enumerate(samples):
        item = dict(sample)

        x_data = np.asarray(item["x_data"])
        y_data = np.asarray(item["y_data"])

        if x_data.shape[1] != original_node_count:
            raise ValueError(
                f"Sample {sample_index} x_data has shape {x_data.shape}"
            )

        if y_data.shape[1] != original_node_count:
            raise ValueError(
                f"Sample {sample_index} y_data has shape {y_data.shape}"
            )

        item["x_data"] = x_data[:, indices, :]
        item["y_data"] = y_data[:, indices, :]

        if "incident_distances" in item:
            item["incident_distances"] = slice_node_axis(
                item["incident_distances"],
                indices,
                original_node_count,
            )

        output_samples.append(item)

    output = np.asarray(output_samples, dtype=object)
    np.save(output_path, output)

    first = output[0]
    print(f"{source_path.name}: {len(output)} samples")
    print(f"  x_data: {first['x_data'].shape}")
    print(f"  y_data: {first['y_data'].shape}")

    if "incident_distances" in first:
        print(
            "  incident_distances:",
            np.asarray(first["incident_distances"]).shape,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("alameda_dir", type=Path)
    parser.add_argument("fremont_dir", type=Path)
    args = parser.parse_args()

    alameda_dir = args.alameda_dir.resolve()
    fremont_dir = args.fremont_dir.resolve()

    indices = np.load(fremont_dir / "alameda_node_indices.npy")
    adjacency = np.load(alameda_dir / "adj_matrix.npy")

    original_node_count = adjacency.shape[0]

    for split in ("train", "val", "test"):
        process_split(
            alameda_dir / f"incident_{split}.npy",
            fremont_dir / f"incident_{split}.npy",
            indices,
            original_node_count,
        )


if __name__ == "__main__":
    main()