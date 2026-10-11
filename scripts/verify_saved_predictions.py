#!/usr/bin/env python3
"""Replay saved checkpoints without training or modifying experiment outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nsl23.dataset import CachedClipDataset, HandCropDataset, load_labels, read_rows
from nsl23.geometry import transform_clip
from nsl23.models import build_model, parameter_counts
from nsl23.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-clips", type=int, default=3, help="Representative clips per RGB checkpoint.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rgb_clips < 1:
        raise SystemExit("--rgb-clips must be positive")
    torch.set_num_threads(1)
    labels = load_labels(ROOT / "configs/nsl23_classes.json")
    records = []
    enrollment_records = []
    for matrix in ("primary", "improvements"):
        for metrics_path in sorted((ROOT / "artifacts/runs" / matrix).glob("*/*/seed_*/metrics.json")):
            directory = metrics_path.parent
            fold, model_name = directory.parent.parent.name, directory.parent.name
            stored = read_rows(directory / "test_predictions.csv")
            samples = read_rows(ROOT / f"data/manifests/folds/{fold}_cached.csv")
            train_rows = [row for row in samples if row["split"] == "train"]
            test_by_id = {row["sample_id"]: row for row in samples if row["split"] == "test"}
            payload = torch.load(directory / "best_checkpoint.pt", map_location="cpu", weights_only=False)
            if payload["labels"] != labels:
                raise AssertionError(f"Checkpoint ontology mismatch: {directory}")
            config = payload["experiment_config"]
            is_proto = model_name.startswith("proto_bilstm")
            name = "landmark_bilstm" if is_proto else model_name
            model = build_model(name, config, len(labels), int(config["num_frames"]))
            model.load_state_dict(payload["state_dict"])
            model.eval()
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            counts = parameter_counts(model)
            if "parameters_trainable" in metrics:
                if counts["parameters_trainable"] != metrics["parameters_trainable"]:
                    raise AssertionError(f"Parameter-count mismatch: {directory}")
            landmark = is_proto or model_name.startswith("landmark_")
            selected = stored if landmark else [stored[i] for i in np.linspace(0, len(stored)-1, args.rgb_clips, dtype=int)]
            meta = [test_by_id[row["sample_id"]] for row in selected]
            with torch.no_grad():
                if landmark:
                    mode = str(payload.get("feature_mode") or config.get("feature_mode") or "xyz")
                    def tensor_for(rows):
                        return torch.from_numpy(np.stack([
                            transform_clip(np.load(ROOT / "data/landmark_cache_primary" / f"{row['sample_id']}.npy"), mode)
                            for row in rows
                        ]))
                    query = tensor_for(meta)
                    if is_proto:
                        embedded = model.forward_features(tensor_for(train_rows))
                        train_y = torch.tensor([labels.index(row["canonical_label"]) for row in train_rows])
                        prototypes = torch.stack([embedded[train_y == c].mean(dim=0) for c in range(len(labels))])
                        if "prototypes" in payload and not torch.allclose(prototypes, payload["prototypes"], atol=1e-5, rtol=1e-5):
                            raise AssertionError(f"Saved prototypes differ from training-class means: {directory}")
                        predicted = torch.cdist(model.forward_features(query), prototypes).argmin(dim=1)
                    else:
                        predicted = model(query).argmax(dim=1)
                    enrollment_prefix = None
                    if model_name == "landmark_bilstm":
                        enrollment_prefix = "enrollment" if matrix == "primary" else "enrollment_rawangle_bilstm"
                    elif model_name == "proto_bilstm":
                        enrollment_prefix = "enrollment_proto"
                    if enrollment_prefix:
                        embedding = model.forward_features(query)
                        truth = torch.tensor([labels.index(row["canonical_label"]) for row in meta])
                        enrolled = torch.empty(len(meta), dtype=torch.long)
                        for lighting in ("bright", "dark"):
                            queries = [i for i, row in enumerate(meta) if row["lighting"] == lighting]
                            support = [i for i, row in enumerate(meta) if row["lighting"] != lighting]
                            if len(queries) != 49 or sorted(truth[support].tolist()) != list(range(49)):
                                raise AssertionError("Closed enrollment requires exactly one other-lighting support per class")
                            order = sorted(support, key=lambda i: int(truth[i]))
                            enrolled[queries] = torch.cdist(embedding[queries], embedding[order]).argmin(dim=1)
                        receipt_path = ROOT / f"artifacts/results/improvements/{enrollment_prefix}_{fold}_seed{payload['seed']}.json"
                        expected_metrics = json.loads(receipt_path.read_text())["methods"]["enroll_closed_other_lighting"]
                        observed = classification_metrics(truth.numpy(), enrolled.numpy(), len(labels))
                        for metric in ("accuracy", "macro_f1"):
                            if not math.isclose(observed[metric], expected_metrics[metric], abs_tol=1e-12):
                                raise AssertionError(f"Closed enrollment replay differs: {receipt_path}/{metric}")
                        enrollment_records.append({"matrix": matrix, "fold": fold, "model": model_name,
                                                   "seed": int(payload["seed"]), "queries_checked": len(meta),
                                                   "accuracy": observed["accuracy"], "macro_f1": observed["macro_f1"]})
                else:
                    if model_name == "crop_cnn":
                        dataset = HandCropDataset(meta, labels, training=False, crop_dir=ROOT / "data/hand_crop_cache_primary")
                    else:
                        dataset = CachedClipDataset(meta, labels, training=False)
                    predicted = model(torch.stack([dataset[i]["frames"] for i in range(len(meta))])).argmax(dim=1)
            expected = torch.tensor([int(row["predicted_index"]) for row in selected])
            if not torch.equal(predicted.cpu(), expected):
                raise AssertionError(f"Checkpoint replay differs for {directory}: {int((predicted.cpu()!=expected).sum())} predictions")
            records.append({"matrix": matrix, "fold": fold, "model": model_name,
                            "seed": int(payload["seed"]), "predictions_checked": len(selected),
                            "scope": "all evaluation clips" if landmark else "representative RGB clips",
                            "parameters_trainable": counts["parameters_trainable"]})
            print(f"pass {matrix}/{fold}/{model_name}/seed_{payload['seed']}: {len(selected)} predictions", flush=True)
    if len(records) != 60:
        raise AssertionError(f"Expected 60 checkpoints, found {len(records)}")
    if len(enrollment_records) != 18:
        raise AssertionError(f"Expected 18 closed enrollment runs, found {len(enrollment_records)}")
    report = {"status": "pass", "retraining": False, "checkpoints": len(records),
              "predictions_checked": sum(row["predictions_checked"] for row in records), "runs": records,
              "closed_enrollment_queries": sum(row["queries_checked"] for row in enrollment_records),
              "closed_enrollment_runs": enrollment_records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
