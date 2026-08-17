#!/usr/bin/env python3
"""Time landmark-model inference on CPU, including MediaPipe on cached frames."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from nsl23.landmarks import extract_clip_landmarks, make_detector
from nsl23.models import build_model


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty")
    index = min(len(ordered) - 1, max(0, int(round((q / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/folds/heldout_s2_cached.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/runtime/landmark_bilstm/runtime.json"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--model-iters", type=int, default=100)
    parser.add_argument("--e2e-clips", type=int, default=30)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["experiment_config"]
    labels = payload["labels"]
    model = build_model(payload["model_name"], config, len(labels), int(config["num_frames"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"][: args.e2e_clips]
    sample = np.load(rows[0]["landmark_cache_path"])
    tensor = torch.from_numpy(sample).unsqueeze(0)

    with torch.no_grad():
        for _ in range(args.warmup):
            model(tensor)
        model_times = []
        for _ in range(args.model_iters):
            start = time.perf_counter()
            model(tensor)
            model_times.append(1000.0 * (time.perf_counter() - start))

    detector = make_detector(static_image_mode=True, min_detection_confidence=0.35)
    e2e_times = []
    try:
        for row in rows:
            frames = np.load(row["frame_cache_path"])
            start = time.perf_counter()
            landmarks, _ = extract_clip_landmarks(frames, detector)
            clip = torch.from_numpy(landmarks).unsqueeze(0)
            with torch.no_grad():
                logits = model(clip)
                _ = torch.softmax(logits, dim=1)
            e2e_times.append(1000.0 * (time.perf_counter() - start))
    finally:
        detector.close()

    model_median = percentile(model_times, 50)
    e2e_median = percentile(e2e_times, 50)
    payload_out = {
        "model": payload["model_name"],
        "device": "cpu",
        "model_only": {
            "median_ms_per_clip": model_median,
            "p95_ms_per_clip": percentile(model_times, 95),
            "clips_per_second_from_median": 1000.0 / model_median,
        },
        "end_to_end": {
            "definition": "MediaPipe Hands on 16 cached frames plus landmark-model softmax",
            "median_ms_per_clip": e2e_median,
            "p95_ms_per_clip": percentile(e2e_times, 95),
            "clips_per_second_from_median": 1000.0 / e2e_median,
            "n_clips": len(e2e_times),
        },
        "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload_out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload_out, indent=2))


if __name__ == "__main__":
    main()
