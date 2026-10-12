#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EIGHT = REPO / "salt-jepa-eight-backbones"
sys.path.insert(0, str(EIGHT))

from salt_backbones.data import load_city  # noqa: E402
from salt_backbones.graph import load_adjacency  # noqa: E402
from salt_backbones.models import build_backbone, compatible_encoder_transfer  # noqa: E402
from salt_backbones.training import (  # noqa: E402
    SpatioTemporalBlockMasker,
    fit_distillation,
    fit_forecast,
    fit_jepa,
    fit_masked_teacher,
    make_loaders,
    seed_everything,
    write_json,
)
from transfer_benchmark.gnn import TrafficGNN, transferable_state  # noqa: E402
from transfer_benchmark.ours import STAEformerSALTBackbone  # noqa: E402
from transfer_benchmark.scjepa import SCJEPAForecast  # noqa: E402
from transfer_benchmark.tsjepa import TSJEPAForecast, fit_ts_jepa  # noqa: E402


METHODS = (
    "ours", "pdformer", "flashst", "testam", "patchstg", "staeformer",
    "scjepa", "ts_jepa", "staeformer_jepa", "dcrnn", "agcrn",
)
DEFAULT_TARGETS = "Berkeley,Dublin,Newark,San Leandro,Pleasanton,Livermore"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Oakland -> six-city full fine-tuning benchmark at horizons 6/9/12"
    )
    parser.add_argument("--data-root", required=True,
                        help="Directory containing Oakland and target city folders")
    parser.add_argument("--source-city", default="Oakland")
    parser.add_argument("--target-cities", default=DEFAULT_TARGETS)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--horizons", default="6,9,12")
    parser.add_argument("--output", default="runs/oakland_multicity")
    parser.add_argument("--file-pattern", default="incident_{split}.npy")
    parser.add_argument("--traffic-feature", type=int, default=0)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--source-epochs", type=int, default=100)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--student-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--source-lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--teacher-lr", type=float, default=1e-3)
    parser.add_argument("--student-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--mask-blocks", type=int, default=8)
    parser.add_argument("--future-block-ratio", type=float, default=0.5)
    parser.add_argument("--ts-jepa-patch-size", type=int, default=3)
    parser.add_argument("--ts-jepa-mask-ratio", type=float, default=0.7)
    parser.add_argument("--ts-jepa-ema-momentum", type=float, default=0.998)
    parser.add_argument("--force", action="store_true",
                        help="Retrain completed target runs; source caches remain reusable")
    return parser.parse_args()


def city_slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def city_root(data_root: Path, name: str) -> Path:
    candidates = [
        data_root / name,
        data_root / name.replace(" ", "_"),
        data_root / name.replace(" ", ""),
        data_root / name.title().replace(" ", "_"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Cannot locate city {name!r}. Tried: " + ", ".join(map(str, candidates))
    )


def adjacency(root: Path, nodes: int):
    path = root / "adj_matrix.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing graph for {root.name}: {path}")
    return load_adjacency(path, nodes)


def build(method, dataset, adj, args, horizon):
    if method in {"dcrnn", "agcrn"}:
        return TrafficGNN(dataset.num_nodes, args.history, horizon, adj, method)
    if method == "scjepa":
        return SCJEPAForecast(REPO, horizon)
    if method == "ts_jepa":
        return TSJEPAForecast(
            HERE / "upstream" / "TS_JEPA", horizon, args.history,
            args.ts_jepa_patch_size,
        )
    if method == "ours":
        return STAEformerSALTBackbone(
            dataset.num_nodes, args.history, horizon, dropout=0.1
        )
    backbone = "staeformer" if method == "staeformer_jepa" else method
    return build_backbone(
        backbone, dataset.num_nodes, args.history, horizon, 0.1,
        adj=adj, train_x=dataset.history.numpy(),
    )


def save_source(path: Path, method: str, horizon: int, state, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "method": method,
        "history": 12,
        "horizon": horizon,
        "state": state,
        **(extra or {}),
    }, path)


def train_supervised_source(method, args, horizon, source, source_loaders, source_adj,
                            source_dir, device):
    checkpoint = source_dir / "source_encoder.pt"
    if checkpoint.exists():
        return torch.load(checkpoint, map_location="cpu")
    seed_everything(args.seed + 100 + horizon)
    model = build(method, source["train"], source_adj, args, horizon)
    fit_forecast(
        model, source_loaders, device, source_dir / "forecast",
        args.source_epochs, args.source_lr, 1.0, args.patience,
        "supervised_source",
    )
    save_source(checkpoint, method, horizon,
                {k: v.detach().cpu() for k, v in model.state_dict().items()})
    return torch.load(checkpoint, map_location="cpu")


def train_salt_source(args, horizon, source, loaders, adj, source_dir, device):
    checkpoint = source_dir / "source_encoder.pt"
    if checkpoint.exists():
        return torch.load(checkpoint, map_location="cpu")
    masker = SpatioTemporalBlockMasker(
        num_blocks=args.mask_blocks, min_time=2,
        max_time=min(6, horizon), min_sensor_ratio=0.10,
        max_sensor_ratio=0.30, future_start=args.history,
        future_block_ratio=args.future_block_ratio,
    )
    seed_everything(args.seed + 200 + horizon)
    teacher = build("ours", source["train"], adj, args, horizon)
    teacher_result = fit_masked_teacher(
        teacher, loaders, device, source_dir / "teacher",
        args.teacher_epochs, args.teacher_lr, args.patience, masker,
        args.seed + 77_000 + horizon,
        checkpoint_metadata={
            "method": "staeformer_salt", "history": args.history,
            "horizon": horizon, "mask_blocks": args.mask_blocks,
            "future_block_ratio": args.future_block_ratio,
        },
    )
    seed_everything(args.seed + 201 + horizon)
    student = build("ours", source["train"], adj, args, horizon)
    distilled = fit_distillation(
        student, teacher, loaders, device, source_dir / "student",
        args.student_epochs, args.student_lr, args.patience, "l2", 0.1,
    )
    save_source(
        checkpoint, "ours", horizon, distilled["student"],
        {"val_latent": float(distilled["val_latent"]),
         "teacher_val_reconstruction": float(teacher_result["val_reconstruction"])},
    )
    return torch.load(checkpoint, map_location="cpu")


def train_jepa_source(args, horizon, source, loaders, adj, source_dir, device):
    checkpoint = source_dir / "source_encoder.pt"
    if checkpoint.exists():
        return torch.load(checkpoint, map_location="cpu")
    seed_everything(args.seed + 300 + horizon)
    online = build("staeformer_jepa", source["train"], adj, args, horizon)
    result = fit_jepa(
        online, loaders, device, source_dir / "jepa",
        args.pretrain_epochs, args.pretrain_lr, args.patience,
        "l2", 0.1, 0.996,
    )
    save_source(
        checkpoint, "staeformer_jepa", horizon, result["target"],
        {"val_latent": float(result["val_latent"])},
    )
    return torch.load(checkpoint, map_location="cpu")


def train_scjepa_source(args, source_root, source_dir):
    # Official SC-JEPA predicts four 3-step target patches. Its Oakland SSL
    # checkpoint is therefore shared by downstream horizons 6/9/12.
    checkpoint = Path(args.output) / "source" / "scjepa" / "official" / "checkpoint.pt"
    if checkpoint.exists():
        return checkpoint
    checkpoint.parent.parent.mkdir(parents=True, exist_ok=True)
    script = REPO / "sc-jepa-official-traffic" / "traffic_adapter" / "pretrain_official.py"
    command = [
        sys.executable, str(script), "--data-root", str(source_root),
        "--output", str(checkpoint.parent), "--seq-len", "12", "--pred-len", "12",
        "--epochs", str(args.pretrain_epochs), "--batch-size", str(max(32, args.batch_size)),
        "--lr", str(args.pretrain_lr), "--workers", str(args.workers),
        "--seed", str(args.seed), "--device", args.device,
        "--pattern", args.file_pattern, "--feature", str(args.traffic_feature),
    ]
    environment = os.environ.copy()
    # Avoid Intel/OpenMP shared-memory failures in restricted shells while
    # leaving CUDA selection controlled by --device.
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("KMP_USE_SHM", "0")
    subprocess.run(command, check=True, cwd=script.parent, env=environment)
    return checkpoint


def train_tsjepa_source(args, loaders, device):
    # Official TS-JEPA pretrains a node-independent temporal encoder. The same
    # Oakland representation checkpoint is shared across downstream horizons.
    source_dir = Path(args.output) / "source" / "ts_jepa" / "official"
    checkpoint = source_dir / "source_encoder.pt"
    if checkpoint.exists():
        return torch.load(checkpoint, map_location="cpu")
    if args.history % args.ts_jepa_patch_size:
        raise ValueError(
            f"history={args.history} must be divisible by "
            f"TS-JEPA patch size={args.ts_jepa_patch_size}"
        )
    seed_everything(args.seed + 400)
    saved = fit_ts_jepa(
        HERE / "upstream" / "TS_JEPA", loaders, device,
        source_dir / "pretrain", args.pretrain_epochs, args.pretrain_lr,
        args.patience, args.ts_jepa_mask_ratio,
        args.ts_jepa_ema_momentum, args.seed,
        input_steps=args.history, patch_size=args.ts_jepa_patch_size,
    )
    source = {
        "method": "ts_jepa", "history": args.history,
        "state": saved["encoder"], "val_latent": saved["val_latent"],
        "patch_size": args.ts_jepa_patch_size,
        "mask_ratio": args.ts_jepa_mask_ratio,
        "ema_momentum": args.ts_jepa_ema_momentum,
        "official_commit": saved["official_commit"],
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(source, checkpoint)
    return source


def source_checkpoint(method, args, horizon, source_root, source, loaders, adj, device):
    source_dir = Path(args.output) / "source" / f"h{horizon}" / method
    source_dir.mkdir(parents=True, exist_ok=True)
    if method == "ours":
        return train_salt_source(args, horizon, source, loaders, adj, source_dir, device)
    if method == "staeformer_jepa":
        return train_jepa_source(args, horizon, source, loaders, adj, source_dir, device)
    if method == "scjepa":
        path = train_scjepa_source(args, source_root, source_dir)
        return {"method": method, "horizon": horizon, "checkpoint": str(path)}
    if method == "ts_jepa":
        return train_tsjepa_source(args, loaders, device)
    return train_supervised_source(method, args, horizon, source, loaders, adj,
                                   source_dir, device)


def load_transfer(model, method, source):
    if method == "scjepa":
        saved = torch.load(source["checkpoint"], map_location="cpu")
        model.load_pretrained(saved)
        return {"copied": ["encoder", "quantizer", "predictor"],
                "skipped": ["forecast_head"]}
    if method == "ts_jepa":
        model.load_pretrained(source["state"])
        return {
            "copied": ["encoder"], "skipped": ["forecast_head"],
            "source_val_latent": source.get("val_latent"),
            "official_commit": source.get("official_commit"),
        }
    if method in {"dcrnn", "agcrn"}:
        return transferable_state(model, source["state"])
    if method == "ours":
        current = model.state_dict()
        copied, skipped = [], []
        for name, value in source["state"].items():
            if ("sensor_embedding" in name or name.startswith("output_projection.")
                    or name not in current or current[name].shape != value.shape):
                skipped.append(name)
            else:
                current[name] = value
                copied.append(name)
        model.load_state_dict(current)
        if not copied:
            raise ValueError("No transferable SALT Student tensors were found")
        return {"copied": copied, "skipped": skipped}
    return compatible_encoder_transfer(model, source["state"])


def completed_metrics(path: Path):
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    return value if {"mae", "mse", "rmse"} <= value.keys() else None


def append_result(path: Path, row):
    rows = []
    if path.exists():
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    identity = (row["city"], row["method"], str(row["horizon"]), str(row["seed"]))
    rows = [old for old in rows if (
        old["city"], old["method"], old["horizon"], old["seed"]
    ) != identity]
    rows.append({key: str(value) for key, value in row.items()})
    rows.sort(key=lambda x: (x["city"], int(x["horizon"]), x["method"]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    methods = [value.strip().lower() for value in args.methods.split(",") if value.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; valid={METHODS}")
    horizons = [int(value) for value in args.horizons.split(",")]
    if not horizons or any(value not in {6, 9, 12} for value in horizons):
        raise ValueError("--horizons must be a subset of 6,9,12")
    targets = [value.strip() for value in args.target_cities.split(",") if value.strip()]
    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    source_root = city_root(data_root, args.source_city)
    print(f"device={device}; source={source_root}; targets={targets}", flush=True)

    for horizon in horizons:
        source = load_city(source_root, args.history, horizon,
                           args.traffic_feature, args.file_pattern)
        source_loaders = make_loaders(source, args.batch_size, args.workers)
        source_adj = adjacency(source_root, source["train"].num_nodes)
        for method in methods:
            print(f"\n===== source={args.source_city} method={method} horizon={horizon} =====",
                  flush=True)
            pretrained = source_checkpoint(
                method, args, horizon, source_root, source, source_loaders,
                source_adj, device,
            )
            for city in targets:
                target_root = city_root(data_root, city)
                run_dir = output / "targets" / city_slug(city) / f"h{horizon}" / method
                metrics_path = run_dir / "full_finetune" / "metrics.json"
                metrics = None if args.force else completed_metrics(metrics_path)
                if metrics is None:
                    target = load_city(target_root, args.history, horizon,
                                       args.traffic_feature, args.file_pattern)
                    loaders = make_loaders(target, args.batch_size, args.workers)
                    target_adj = adjacency(target_root, target["train"].num_nodes)
                    seed_everything(args.seed + 1000 + horizon)
                    model = build(method, target["train"], target_adj, args, horizon)
                    report = load_transfer(model, method, pretrained)
                    write_json(run_dir / "transfer_report.json", report)
                    result = fit_forecast(
                        model, loaders, device, run_dir / "full_finetune",
                        args.finetune_epochs, args.finetune_lr,
                        args.encoder_lr_scale, args.patience,
                        "supervised_transfer",
                    )
                    metrics = asdict(result)
                row = {
                    "city": city, "source_city": args.source_city,
                    "method": method, "horizon": horizon, "history": args.history,
                    "mae": metrics["mae"], "mse": metrics["mse"],
                    "rmse": metrics["rmse"], "seed": args.seed,
                    "transfer": "full_finetune",
                }
                append_result(output / "results.csv", row)
                print(json.dumps(row), flush=True)

    write_json(output / "config.json", vars(args))
    print(f"\nCompleted. Results: {output / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
