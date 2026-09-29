#!/usr/bin/env python3
"""Extract the Fremont-induced sensor subgraph from the Alameda IGSTGNN data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def normalized_column_map(columns: list[str]) -> dict[str, str]:
    return {
        "".join(ch for ch in str(column).lower() if ch.isalnum()): column
        for column in columns
    }


def find_column(columns: list[str], candidates: tuple[str, ...]) -> str:
    normalized = normalized_column_map(columns)
    for candidate in candidates:
        key = "".join(ch for ch in candidate.lower() if ch.isalnum())
        if key in normalized:
            return normalized[key]
    raise ValueError(
        f"Could not find any of {candidates}. Available columns: {columns}"
    )


def graph_stats(adjacency: np.ndarray) -> dict[str, object]:
    support = adjacency != 0
    n = adjacency.shape[0]
    return {
        "nodes": int(n),
        "nonzero_entries": int(np.count_nonzero(support)),
        "unordered_connected_pairs": int(
            np.count_nonzero(np.triu(support | support.T, k=1))
        ),
        "isolated_nodes": int(
            np.count_nonzero(~support.any(axis=0) & ~support.any(axis=1))
        ),
        "topology_symmetric": bool(np.array_equal(support, support.T)),
        "weights_symmetric": bool(np.allclose(adjacency, adjacency.T, atol=1e-6)),
        "density": float(
            np.count_nonzero(support) / (n * (n - 1)) if n > 1 else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing Alameda/adj_matrix.npy and sensors.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: a Fremont sibling of input_dir)",
    )
    parser.add_argument(
        "--city",
        default="Fremont",
        help="City value to select from sensors.csv (default: Fremont)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else input_dir.parent / "Fremont"
    )

    adjacency = np.load(input_dir / "adj_matrix.npy")
    with (input_dir / "sensors.csv").open(
        "r", newline="", encoding="utf-8-sig"
    ) as sensor_file:
        reader = csv.DictReader(sensor_file)
        sensor_columns = list(reader.fieldnames or [])
        sensors = list(reader)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Adjacency must be square; received {adjacency.shape}")
    if adjacency.shape[0] != len(sensors):
        raise ValueError(
            f"Adjacency has {adjacency.shape[0]} nodes but sensors.csv has {len(sensors)} rows"
        )

    city_column = find_column(sensor_columns, ("City",))
    target_city = args.city.strip().casefold()
    city_values = [str(row.get(city_column, "")).strip() for row in sensors]
    selected = np.asarray(
        [city.casefold() == target_city for city in city_values], dtype=bool
    )
    indices = np.flatnonzero(selected)
    if not len(indices):
        available_cities = sorted({city for city in city_values if city})
        raise RuntimeError(
            f"No sensors matched city {args.city!r}. Available cities: {available_cities}"
        )

    fremont_adjacency = adjacency[np.ix_(indices, indices)]
    fremont_sensors = []
    for index in indices:
        row = {"alameda_matrix_index": str(index)}
        row.update(sensors[index])
        fremont_sensors.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "adj_matrix.npy", fremont_adjacency)
    np.save(output_dir / "alameda_node_indices.npy", indices)
    with (output_dir / "sensors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as sensor_file:
        writer = csv.DictWriter(
            sensor_file,
            fieldnames=["alameda_matrix_index", *sensor_columns],
        )
        writer.writeheader()
        writer.writerows(fremont_sensors)
    metadata = {
        "source_directory": str(input_dir),
        "selection": f"sensors.csv city equals {args.city!r} after trimming and case folding",
        "city_column": city_column,
        "selected_source_indices": indices.tolist(),
        "original_graph": graph_stats(adjacency),
        "fremont_induced_graph": graph_stats(fremont_adjacency),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(json.dumps(metadata, indent=2))
    print(f"Saved Fremont files to: {output_dir}")


if __name__ == "__main__":
    main()
