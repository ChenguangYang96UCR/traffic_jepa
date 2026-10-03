#!/usr/bin/env python3
"""Compare supervised Oakland transfer with future-focused SALT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervised-root", type=Path, required=True)
    parser.add_argument("--salt-root", type=Path, required=True)
    args = parser.parse_args()

    supervised = load(args.supervised_root / "all_results.json")
    salt_all = load(
        args.salt_root
        / "exp3_oakland_oakland_berkeley"
        / "all"
        / "all_results.json"
    )
    salt_future = load(
        args.salt_root
        / "exp3_oakland_oakland_berkeley"
        / "future"
        / "all_results.json"
    )
    rows = [
        ("Berkeley STAEformer scratch", supervised["berkeley_scratch"]),
        ("Oakland supervised -> frozen", supervised["supervised_frozen"]),
        ("Oakland supervised -> full FT", supervised["supervised_full"]),
        ("SALT all-step -> frozen", salt_all["salt_frozen"]),
        ("SALT all-step -> full FT", salt_all["salt_full"]),
        ("SALT future-only -> frozen", salt_future["salt_frozen"]),
        ("SALT future-only -> full FT", salt_future["salt_full"]),
    ]
    lines = [
        "Oakland -> Berkeley: supervised transfer vs future-focused SALT",
        f"{'Method':<38} {'MAE':>12} {'MSE':>12} {'RMSE':>12}",
        "-" * 77,
    ]
    for label, item in rows:
        lines.append(
            f"{label:<38} {item['mae']:>12.7f} "
            f"{item['mse']:>12.7f} {item['rmse']:>12.7f}"
        )
    summary = "\n".join(lines)
    print(summary)
    (args.supervised_root / "supervised_vs_salt.txt").write_text(
        summary + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
