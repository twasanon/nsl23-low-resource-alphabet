#!/usr/bin/env python3
"""Fail-closed checks for numbers claimed in the manuscript and README."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nsl23.models import build_model


ROOT = Path(__file__).resolve().parents[1]


def approx(value: float, target: float, tol: float = 5e-4) -> None:
    if abs(value - target) > tol:
        raise AssertionError(f"{value} != {target} ± {tol}")


def main() -> None:
    summary = json.loads((ROOT / "artifacts/results/primary_results_summary.json").read_text())
    if summary["status"] != "pass" or summary["completed_runs"] != 30:
        raise AssertionError(summary)
    model = pd.read_csv(ROOT / "artifacts/results/model_summary.csv").set_index("model")
    approx(float(model.loc["single_frame_cnn", "accuracy_mean"]), 0.05782312925170068)
    approx(float(model.loc["cnn_bilstm", "accuracy_mean"]), 0.06972789115646259)
    approx(float(model.loc["cnn_transformer", "accuracy_mean"]), 0.06292517006802721)
    approx(float(model.loc["landmark_mlp", "accuracy_mean"]), 0.18027210884353742)
    approx(float(model.loc["landmark_bilstm", "accuracy_mean"]), 0.24829931972789118)

    detection = json.loads((ROOT / "artifacts/audit/landmark_detection_summary.json").read_text())
    if detection["clips"] != 375 or detection["clips_with_no_detection"] != 0:
        raise AssertionError(detection)
    approx(float(detection["detection_rate"]), 0.978, tol=5e-4)

    cache_files = list((ROOT / "data/landmark_cache_primary").glob("*.npy"))
    if len(cache_files) != 375:
        raise AssertionError(f"expected 375 landmark files, found {len(cache_files)}")

    webcam = json.loads((ROOT / "artifacts/runtime/webcam_heldout_s1_A.json").read_text())
    if webcam["fps"] < 20 or webcam["last_prediction"]["label"] != "A":
        raise AssertionError(webcam)

    ckpt_path = ROOT / "artifacts/runs/primary/heldout_s1/landmark_bilstm/seed_42/best_checkpoint.pt"
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_nn = build_model(
        payload["model_name"],
        payload["experiment_config"],
        len(payload["labels"]),
        int(payload["experiment_config"]["num_frames"]),
    )
    model_nn.load_state_dict(payload["state_dict"])
    model_nn.eval()
    pred_path = ROOT / "artifacts/runs/primary/heldout_s1/landmark_bilstm/seed_42/test_predictions.csv"
    with pred_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = 0
    for row in rows[:25]:
        array = np.load(ROOT / "data/landmark_cache_primary" / f"{row['sample_id']}.npy")
        tensor = torch.from_numpy(array).unsqueeze(0)
        with torch.no_grad():
            predicted = int(model_nn(tensor).argmax(dim=1))
        if predicted != int(row["predicted_index"]):
            raise AssertionError((row["sample_id"], predicted, row["predicted_index"]))
        matches += 1

    paper = ROOT / "manuscript/nsl23_low_resource.pdf"
    if not paper.is_file() or paper.stat().st_size < 50_000:
        raise AssertionError(f"missing compiled paper at {paper}")

    print(
        json.dumps(
            {
                "status": "pass",
                "runs": 30,
                "landmark_bilstm_accuracy_mean": float(model.loc["landmark_bilstm", "accuracy_mean"]),
                "replayed_predictions": matches,
                "webcam_fps": webcam["fps"],
                "paper_bytes": paper.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
