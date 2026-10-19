#!/usr/bin/env python3
"""Create traffic-only city datasets from a county-level window dataset."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np


def numpy_pickle_compatibility():
    """Load NumPy-2 pickles in the project's NumPy-1 environment."""
    try:
        importlib.import_module("numpy._core")
    except ModuleNotFoundError:
        core = importlib.import_module("numpy.core")
        sys.modules.setdefault("numpy._core", core)
        for name in ("multiarray", "numeric", "_multiarray_umath", "umath"):
            sys.modules.setdefault(
                f"numpy._core.{name}", importlib.import_module(f"numpy.core.{name}")
            )


def normalized_column(columns, wanted):
    mapping = {
        "".join(character for character in str(column).casefold()
                if character.isalnum()): column
        for column in columns
    }
    key = "".join(character for character in wanted.casefold() if character.isalnum())
    if key not in mapping:
        raise ValueError(f"No {wanted!r} column; available={columns}")
    return mapping[key]


def slugify(value):
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_value.casefold()).strip("_")


def read_sensor_groups(path: Path, requested):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        city_column = normalized_column(columns, "city")
        rows = list(reader)
    names = [str(row.get(city_column, "")).strip() for row in rows]
    available = {}
    for index, name in enumerate(names):
        if name:
            available.setdefault(name.casefold(), {"name": name, "indices": []})
            available[name.casefold()]["indices"].append(index)
    groups = []
    for requested_name in requested:
        key = requested_name.casefold()
        if key not in available:
            raise ValueError(
                f"City {requested_name!r} is absent from {path}; "
                f"available={sorted(item['name'] for item in available.values())}"
            )
        item = available[key]
        groups.append((item["name"], np.asarray(item["indices"], dtype=np.int64),
                       columns, rows))
    return groups, len(rows)


def subset_sample(sample, indices, total_nodes, source, sample_index):
    if not isinstance(sample, dict) or not {"x_data", "y_data"} <= sample.keys():
        raise ValueError(f"{source}: sample {sample_index} has no x_data/y_data")
    result = dict(sample)
    for key in ("x_data", "y_data"):
        value = np.asarray(sample[key])
        if value.ndim != 3 or value.shape[1] != total_nodes:
            raise ValueError(
                f"{source}: sample {sample_index} {key} shape={value.shape}; "
                f"expected sensor axis 1 of length {total_nodes}"
            )
        result[key] = value[:, indices, :]
    # This field is node-indexed in the released incident container. Other
    # incident metadata is event-indexed and is preserved unchanged.
    if "incident_distances" in sample:
        distances = np.asarray(sample["incident_distances"])
        if distances.ndim and distances.shape[-1] == total_nodes:
            result["incident_distances"] = distances[..., indices]
    return result


def complete_city(root: Path):
    required = [root / "adj_matrix.npy", root / "sensors.csv"]
    required += [root / f"incident_{split}.npy" for split in ("train", "val", "test")]
    return all(path.exists() for path in required)


def prepare(county_root: Path, output_root: Path, requested, reuse_existing=False):
    numpy_pickle_compatibility()
    county_root, output_root = Path(county_root), Path(output_root)
    groups, total_nodes = read_sensor_groups(county_root / "sensors.csv", requested)
    adjacency = np.asarray(np.load(county_root / "adj_matrix.npy"), dtype=np.float32)
    if adjacency.shape != (total_nodes, total_nodes):
        raise ValueError(
            f"County adjacency={adjacency.shape}, sensors.csv rows={total_nodes}"
        )
    destinations = {city: output_root / city for city, *_ in groups}
    for city, destination in destinations.items():
        if destination.exists() and not (reuse_existing and complete_city(destination)):
            raise FileExistsError(f"Refusing to overwrite incomplete/existing {destination}")

    pending = [(city, indices, columns, rows) for city, indices, columns, rows in groups
               if not (reuse_existing and complete_city(destinations[city]))]
    for city, indices, columns, rows in pending:
        destination = destinations[city]
        destination.mkdir(parents=True, exist_ok=False)
        np.save(destination / "adj_matrix.npy", adjacency[np.ix_(indices, indices)])
        with (destination / "sensors.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows[int(index)] for index in indices)
        split_counts = {}
        for split in ("train", "val", "test"):
            source = county_root / f"incident_{split}.npy"
            samples = np.load(source, allow_pickle=True)
            selected = np.asarray([
                subset_sample(sample, indices, total_nodes, source, sample_index)
                for sample_index, sample in enumerate(samples)
            ], dtype=object)
            np.save(destination / f"incident_{split}.npy", selected, allow_pickle=True)
            split_counts[split] = len(selected)
        manifest = {
            "protocol": "county_city_induced_view_v1",
            "county_root": str(county_root.resolve()),
            "city": city,
            "city_slug": slugify(city),
            "sensor_count": len(indices),
            "sensor_indices": indices.tolist(),
            "split_counts": split_counts,
            "traffic_feature": 0,
        }
        (destination / "city_view_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"prepared {city}: sensors={len(indices)}, splits={split_counts}")
    for city, _, _, _ in groups:
        if city not in {item[0] for item in pending}:
            print(f"reusing {city}: {destinations[city]}")
    return destinations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--county-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--cities", required=True, help="comma-separated exact city names")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    requested = [value.strip() for value in args.cities.split(",") if value.strip()]
    if not requested:
        raise ValueError("--cities selected no cities")
    prepare(args.county_root, args.output_root, requested, args.reuse_existing)


if __name__ == "__main__":
    main()
