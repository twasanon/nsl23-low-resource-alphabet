"""Synchronized batch-one model and end-to-end latency benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset import IMAGENET_MEAN, IMAGENET_STD, load_labels, read_rows
from .frame_cache import decode_targets, target_times
from .models import build_model, parameter_counts
from .train import environment_snapshot, file_hash, select_device, synchronize


def cached_tensor(path: Path) -> torch.Tensor:
    array = np.load(path, allow_pickle=False)
    frames = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float().div_(255.0)
    return ((frames - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)


def source_tensor(row: dict[str, str], num_frames: int, image_size: int) -> torch.Tensor:
    times = target_times(float(row["start_seconds"]), float(row["end_seconds"]), num_frames)
    requests = [(float(value), row["sample_id"], index) for index, value in enumerate(times)]
    decoded = decode_targets(Path(row["source_path"]), requests, image_size)[row["sample_id"]]
    array = np.stack([decoded[index] for index in range(num_frames)])
    frames = torch.from_numpy(array).permute(0, 3, 1, 2).float().div_(255.0)
    return ((frames - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)


def percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_ms_per_clip": float(np.median(array)),
        "p95_ms_per_clip": float(np.quantile(array, 0.95)),
        "mean_ms_per_clip": float(array.mean()),
        "clips_per_second_from_median": float(1000.0 / np.median(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_local_m4.json"))
    parser.add_argument("--classes", type=Path, default=Path("configs/nsl23_classes.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = str(checkpoint["model_name"])
    labels = load_labels(args.classes)
    num_frames = int(config["num_frames"])
    image_size = int(config["image_size"])
    device = select_device(args.device)
    model = build_model(model_name, config, len(labels), num_frames)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    rows = [row for row in read_rows(args.samples) if row["split"] == "test"]
    if not rows:
        raise RuntimeError("No test samples")
    runtime_config = config["runtime"]
    warmup = int(runtime_config["batch1_warmup"])
    iterations = int(runtime_config["batch1_iterations"])
    input_tensor = cached_tensor(Path(rows[0]["frame_cache_path"])).to(device)
    model_times: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            torch.softmax(model(input_tensor), dim=1)
        synchronize(device)
        for _ in range(iterations):
            start = time.perf_counter()
            torch.softmax(model(input_tensor), dim=1)
            synchronize(device)
            model_times.append((time.perf_counter() - start) * 1000.0)

        # A few untimed clips warm the decoder and allocation paths. Timed rows
        # still cover every held-out clip exactly once.
        for row in rows[: min(5, len(rows))]:
            tensor = source_tensor(row, num_frames, image_size).to(device)
            torch.softmax(model(tensor), dim=1)
            synchronize(device)

        end_to_end_times: list[float] = []
        per_clip: list[dict[str, Any]] = []
        for row in rows:
            start = time.perf_counter()
            tensor = source_tensor(row, num_frames, image_size).to(device)
            probabilities = torch.softmax(model(tensor), dim=1)
            synchronize(device)
            latency = (time.perf_counter() - start) * 1000.0
            end_to_end_times.append(latency)
            per_clip.append(
                {
                    "sample_id": row["sample_id"],
                    "relative_path": row["relative_path"],
                    "signer": row["signer"],
                    "lighting": row["lighting"],
                    "latency_ms": latency,
                    "predicted_index": int(probabilities.argmax(dim=1).item()),
                }
            )

    report = {
        "model": model_name,
        "fold": checkpoint.get("fold", "unspecified"),
        "seed": checkpoint["seed"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "sample_manifest": str(args.samples.resolve()),
        "sample_manifest_sha256": file_hash(args.samples),
        "batch_size": 1,
        "num_frames": num_frames,
        "image_size": image_size,
        "model_only_definition": "device-resident normalized tensor through model forward and softmax with explicit device synchronization",
        "end_to_end_definition": "open and decode original video, sample frames, letterbox, normalize, transfer, model forward, softmax, and synchronize",
        "model_only_iterations": iterations,
        "end_to_end_test_clips": len(rows),
        "model_only": percentiles(model_times),
        "end_to_end": percentiles(end_to_end_times),
        **parameter_counts(model),
        "environment": environment_snapshot(device),
        "machine": platform.machine(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runtime.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "end_to_end_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_clip[0]))
        writer.writeheader()
        writer.writerows(per_clip)
    with (args.output_dir / "model_only_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "latency_ms"])
        writer.writeheader()
        writer.writerows({"iteration": index, "latency_ms": value} for index, value in enumerate(model_times, start=1))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
