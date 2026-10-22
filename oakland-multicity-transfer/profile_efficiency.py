#!/usr/bin/env python3
"""Profile deployment cost for one trained source-to-city transfer experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from run import adjacency, build, city_root, city_slug
from salt_backbones.data import load_city


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--target-city", default="Laguna Niguel")
    parser.add_argument("--methods", required=True)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--file-pattern", default="incident_{split}.npy")
    parser.add_argument("--traffic-feature", type=int, default=0)
    parser.add_argument("--batch-sizes", default="1,32")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--ts-jepa-patch-size", type=int, default=3)
    parser.add_argument("--worker-method", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--timing", choices=("cuda_event", "wall"), default="cuda_event",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def unique_parameters(parameters):
    result, seen = [], set()
    for parameter in parameters:
        if id(parameter) not in seen:
            result.append(parameter)
            seen.add(id(parameter))
    return result


def parameter_count(parameters):
    return sum(parameter.numel() for parameter in unique_parameters(parameters))


def model_breakdown(model):
    total = parameter_count(model.parameters())
    encoder = parameter_count(model.encoder_parameters()) if hasattr(
        model, "encoder_parameters"
    ) else None
    head = parameter_count(model.head_parameters()) if hasattr(
        model, "head_parameters"
    ) else None
    state_bytes = sum(
        value.numel() * value.element_size()
        for value in model.state_dict().values()
        if torch.is_tensor(value)
    )
    return total, encoder, head, state_bytes / (1024 ** 2)


def latency_profile(model, sample, batch_size, device, warmup, repeats, timing):
    if len(sample) < batch_size:
        copies = math.ceil(batch_size / len(sample))
        sample = sample.repeat(copies, 1, 1, 1)
    inputs = sample[:batch_size].to(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
            if timing == "cuda_event":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                stream = torch.cuda.current_stream(device)
                start.record(stream)
                for _ in range(repeats):
                    model(inputs)
                end.record(stream)
                end.synchronize()
                milliseconds = start.elapsed_time(end) / repeats
            else:
                start_time = time.perf_counter()
                for _ in range(repeats):
                    model(inputs)
                torch.cuda.synchronize(device)
                milliseconds = (time.perf_counter() - start_time) * 1000 / repeats
            peak_mb = (
                torch.cuda.max_memory_allocated(device) - baseline
            ) / (1024 ** 2)
        else:
            start_time = time.perf_counter()
            for _ in range(repeats):
                model(inputs)
            milliseconds = (time.perf_counter() - start_time) * 1000 / repeats
            peak_mb = None
    return milliseconds, batch_size * 1000 / milliseconds, peak_mb


def profile_one(args, method, target, graph, sample, device, batch_sizes):
    run_root = Path(args.run_root).resolve()
    build_args = SimpleNamespace(
        history=args.history, ts_jepa_patch_size=args.ts_jepa_patch_size,
    )
    run_dir = (
        run_root / "targets" / city_slug(args.target_city)
        / f"h{args.horizon}" / method / "full_finetune"
    )
    checkpoint = run_dir / "best_forecast.pt"
    metrics_path = run_dir / "metrics.json"
    if not checkpoint.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing completed run for {method}: {checkpoint} / {metrics_path}"
        )
    model = build(method, target["train"], graph, build_args, args.horizon)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    model.to(device).eval()
    total, encoder, head, model_size = model_breakdown(model)
    metrics = json.loads(metrics_path.read_text())
    row = {
        "method": method,
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
        "rmse": float(metrics["rmse"]),
        "parameters": total,
        "encoder_parameters": encoder,
        "head_parameters": head,
        "model_size_mb": model_size,
        "checkpoint_size_mb": checkpoint.stat().st_size / (1024 ** 2),
        "device": str(device),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing": args.timing,
    }
    for batch in batch_sizes:
        latency, throughput, peak_mb = latency_profile(
            model, sample, batch, device, args.warmup, args.repeats, args.timing,
        )
        row[f"latency_b{batch}_ms"] = latency
        row[f"throughput_b{batch}_samples_s"] = throughput
        row[f"peak_memory_b{batch}_mb"] = peak_mb
    return row


def worker_command(args, method, output, warmup, repeats, timing):
    return [
        sys.executable, str(Path(__file__).resolve()),
        "--data-root", args.data_root,
        "--run-root", args.run_root,
        "--target-city", args.target_city,
        "--methods", method,
        "--worker-method", method,
        "--worker-output", str(output),
        "--horizon", str(args.horizon),
        "--history", str(args.history),
        "--device", args.device,
        "--file-pattern", args.file_pattern,
        "--traffic-feature", str(args.traffic_feature),
        "--batch-sizes", args.batch_sizes,
        "--warmup", str(warmup),
        "--repeats", str(repeats),
        "--ts-jepa-patch-size", str(args.ts_jepa_patch_size),
        "--timing", timing,
    ]


def latex_table(rows, batch_sizes):
    columns = "lrrrrrr" + "rr" * len(batch_sizes)
    header = ["Method", "MAE", "MSE", "Params (M)", "Encoder (M)",
              "Head (M)", "Size (MB)"]
    for batch in batch_sizes:
        header.extend([f"Latency B{batch} (ms)", f"Throughput B{batch}"])
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        " & ".join(header) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        values = [
            row["method"], f'{row["mae"]:.4f}', f'{row["mse"]:.4f}',
            f'{row["parameters"] / 1e6:.3f}',
            "--" if row["encoder_parameters"] is None else
            f'{row["encoder_parameters"] / 1e6:.3f}',
            "--" if row["head_parameters"] is None else
            f'{row["head_parameters"] / 1e6:.3f}',
            f'{row["model_size_mb"]:.2f}',
        ]
        for batch in batch_sizes:
            values.extend([
                f'{row[f"latency_b{batch}_ms"]:.3f}',
                f'{row[f"throughput_b{batch}_samples_s"]:.1f}',
            ])
        lines.append(" & ".join(values) + r" \\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Irvine-to-Laguna Niguel transfer efficiency at horizon 12. "
        "Parameters and latency describe the deployed encoder and forecasting head; "
        "offline teachers and latent predictors are excluded.}",
        "\\label{tab:irvine-efficiency}",
        "\\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    data_root = Path(args.data_root).resolve()
    methods = [item.strip().lower() for item in args.methods.split(",") if item.strip()]
    batch_sizes = [int(item) for item in args.batch_sizes.split(",")]
    if not batch_sizes or any(item < 1 for item in batch_sizes):
        raise ValueError("--batch-sizes must contain positive integers")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    target_root = city_root(data_root, args.target_city)
    target = load_city(
        target_root, args.history, args.horizon,
        args.traffic_feature, args.file_pattern,
    )
    graph = adjacency(target_root, target["train"].num_nodes)
    sample = target["test"].history
    output = run_root / "efficiency"
    output.mkdir(parents=True, exist_ok=True)
    if args.worker_method:
        row = profile_one(
            args, args.worker_method, target, graph, sample, device, batch_sizes,
        )
        Path(args.worker_output).write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        return

    rows, failures = [], []
    workers = output / "workers"
    workers.mkdir(parents=True, exist_ok=True)
    for method in methods:
        worker_output = workers / f"{method}.json"
        attempts = [
            (args.warmup, args.repeats, "cuda_event"),
            (min(args.warmup, 10), max(20, args.repeats // 4), "wall"),
        ]
        completed = False
        for attempt, (warmup, repeats, timing) in enumerate(attempts, 1):
            worker_output.unlink(missing_ok=True)
            print(
                f"Profiling {method}: attempt={attempt}, warmup={warmup}, "
                f"repeats={repeats}, timing={timing}", flush=True,
            )
            result = subprocess.run(
                worker_command(
                    args, method, worker_output, warmup, repeats, timing,
                ),
                check=False,
            )
            if result.returncode == 0 and worker_output.exists():
                rows.append(json.loads(worker_output.read_text()))
                completed = True
                break
        if not completed:
            failure = {"method": method, "error": "profiling failed after two attempts"}
            failures.append(failure)
            print(json.dumps(failure), flush=True)

    if not rows:
        raise RuntimeError("No model could be profiled")
    fields = list(rows[0])
    with (output / "efficiency.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "efficiency.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    (output / "efficiency_table.tex").write_text(latex_table(rows, batch_sizes))
    print(f"Efficiency results: {output}", flush=True)


if __name__ == "__main__":
    main()
