#!/usr/bin/env python3
"""Average logits from a landmark model and a cropped-hand CNN."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from nsl23.dataset import load_labels, read_rows
from nsl23.geometry import transform_clip
from nsl23.metrics import classification_metrics
from nsl23.models import build_model


def load_net(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(payload["model_name"], payload["experiment_config"], len(payload["labels"]), int(payload["experiment_config"]["num_frames"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark-checkpoint", type=Path, required=True)
    parser.add_argument("--crop-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--crop-dir", type=Path, default=Path("data/hand_crop_cache_primary"))
    parser.add_argument("--classes", type=Path, default=Path("configs/nsl23_classes.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--landmark-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    landmark_model, landmark_payload = load_net(args.landmark_checkpoint, device)
    crop_model, crop_payload = load_net(args.crop_checkpoint, device)
    labels = load_labels(args.classes)
    rows = [row for row in read_rows(args.manifest) if row["split"] == "test"]
    mode = str(landmark_payload["experiment_config"].get("feature_mode", "xyz"))
    y_true = []
    y_land = []
    y_crop = []
    y_fuse = []
    with torch.no_grad():
        for row in rows:
            label = labels.index(row["canonical_label"])
            land = np.load(Path("data/landmark_cache_primary") / f"{row['sample_id']}.npy")
            land_tensor = torch.from_numpy(transform_clip(land, mode)).unsqueeze(0).to(device)
            crop = np.load(args.crop_dir / f"{row['sample_id']}.npy")
            crop_tensor = torch.from_numpy(crop.copy()).permute(0, 3, 1, 2).float().div_(255.0)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
            crop_tensor = ((crop_tensor.unsqueeze(0) - mean) / std).to(device)
            logit_l = landmark_model(land_tensor)[0]
            logit_c = crop_model(crop_tensor)[0]
            fused = args.landmark_weight * logit_l + (1.0 - args.landmark_weight) * logit_c
            y_true.append(label)
            y_land.append(int(logit_l.argmax()))
            y_crop.append(int(logit_c.argmax()))
            y_fuse.append(int(fused.argmax()))
    y_true_a = np.asarray(y_true)
    report = {
        "landmark": classification_metrics(y_true_a, np.asarray(y_land), 49)["accuracy"],
        "crop": classification_metrics(y_true_a, np.asarray(y_crop), 49)["accuracy"],
        "fusion": classification_metrics(y_true_a, np.asarray(y_fuse), 49)["accuracy"],
        "fusion_macro_f1": classification_metrics(y_true_a, np.asarray(y_fuse), 49)["macro_f1"],
        "n": len(rows),
        "landmark_checkpoint": str(args.landmark_checkpoint),
        "crop_checkpoint": str(args.crop_checkpoint),
        "landmark_weight": args.landmark_weight,
    }
    vowel = np.array([row["sign_type"] == "vowel" for row in rows])
    report["fusion_consonant_accuracy"] = float((y_true_a[~vowel] == np.asarray(y_fuse)[~vowel]).mean())
    report["fusion_vowel_accuracy"] = float((y_true_a[vowel] == np.asarray(y_fuse)[vowel]).mean())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
