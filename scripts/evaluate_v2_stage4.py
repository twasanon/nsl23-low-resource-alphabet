#!/usr/bin/env python3
"""Run the one evaluation-only Stage 4 batch from three frozen checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.models import LandmarkBiLSTM
from nsl23.train import select_device
from nsl23.v2_annotation import sha256_file, write_csv
from nsl23.v2_features import cache_set_sha256, feature_dim, load_feature_clip
from nsl23.v2_metrics import hierarchical_classification_metrics, mean_and_sample_sd
from nsl23.v2_protocol import (
    load_and_validate_frozen_protocol,
    load_and_validate_model_spec,
    validate_manifest_provenance,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True, help="stage4_eval manifest")
    parser.add_argument("--final-train-manifest", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs=3, type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/landmark_cache_v2"))
    parser.add_argument("--classes", type=Path, default=Path("configs/nsl23_classes.json"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/v2_protocol.json"))
    parser.add_argument("--model-spec", type=Path, default=Path("configs/v2_model_spec.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Stage 4 output already exists; evaluation is one batch: {args.output_dir}")
    try:
        protocol = load_and_validate_frozen_protocol(
            args.protocol,
            classes_path=args.classes,
            require_confirmatory=True,
        )
        model_spec = load_and_validate_model_spec(args.model_spec, protocol)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    test_rows = read_rows(args.samples)
    train_rows = read_rows(args.final_train_manifest)
    try:
        validate_manifest_provenance(test_rows, protocol, "stage4_eval")
        validate_manifest_provenance(train_rows, protocol, "final_retrain")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if any(row["split"] != "test" for row in test_rows):
        raise SystemExit("Stage 4 manifest must contain test rows only")
    if any(row["split"] != "train" for row in train_rows):
        raise SystemExit("Final retrain manifest must contain train rows only")
    testers = sorted(
        protocol["assigned_roles"]["frozen_test"], key=lambda value: int(value[1:])
    )
    if sorted({row["signer"] for row in test_rows}, key=lambda value: int(value[1:])) != testers:
        raise SystemExit("Stage 4 manifest testers do not match the frozen protocol")
    labels_config = json.loads(args.classes.read_text(encoding="utf-8"))
    labels = labels_config["consonants"] + labels_config["vowels"]
    for signer in testers:
        observed = {row["canonical_label"] for row in test_rows if row["signer"] == signer}
        if observed != set(labels):
            raise SystemExit(f"Tester {signer} does not have exactly the 49-class label set")
    missing_cache = [
        row["sample_id"]
        for row in test_rows
        if not (args.cache_dir / f"{row['sample_id']}.npz").is_file()
    ]
    if missing_cache:
        raise SystemExit(f"Missing Stage 4 caches; first sample: {missing_cache[0]}")

    expected_seeds = sorted(model_spec["seeds"])
    train_manifest_hash = sha256_file(args.final_train_manifest)
    train_cache_hash = cache_set_sha256(
        args.cache_dir,
        [row["sample_id"] for row in train_rows],
        require_npz=True,
    )
    test_cache_hash = cache_set_sha256(
        args.cache_dir,
        [row["sample_id"] for row in test_rows],
        require_npz=True,
    )
    checkpoint_records = []
    checkpoint_payloads = []
    for checkpoint_path in args.checkpoints:
        if not checkpoint_path.is_file():
            raise SystemExit(f"Missing checkpoint: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        seed = int(payload.get("seed", -1))
        if payload.get("stage") != "final_retrain" or payload.get("run_mode") != "train_only":
            raise SystemExit(f"Not a final train-only checkpoint: {checkpoint_path}")
        if payload.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise SystemExit(f"Checkpoint protocol mismatch: {checkpoint_path}")
        if payload.get("model_spec_sha256") != model_spec["model_spec_sha256"]:
            raise SystemExit(f"Checkpoint model-spec mismatch: {checkpoint_path}")
        if payload.get("manifest_sha256") != train_manifest_hash:
            raise SystemExit(f"Checkpoint training-manifest mismatch: {checkpoint_path}")
        if payload.get("train_cache_set_sha256") != train_cache_hash:
            raise SystemExit(f"Checkpoint training-cache mismatch: {checkpoint_path}")
        if payload.get("feature_mode") != model_spec["feature_mode"]:
            raise SystemExit(f"Checkpoint feature mismatch: {checkpoint_path}")
        if payload.get("labels") != labels:
            raise SystemExit(f"Checkpoint class order mismatch: {checkpoint_path}")
        prototypes = payload.get("prototypes")
        if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2:
            raise SystemExit(f"Checkpoint prototypes are missing or malformed: {checkpoint_path}")
        if prototypes.shape[0] != len(labels) or not torch.isfinite(prototypes).all():
            raise SystemExit(f"Checkpoint prototypes do not match the 49 classes: {checkpoint_path}")
        experiment = payload.get("experiment_config") or {}
        model_config = (experiment.get("models") or {}).get("landmark_bilstm") or {}
        expected_experiment = {
            "num_frames": model_spec["num_frames"],
            "epochs": model_spec["epochs"],
            "episodes": model_spec["episodes"],
            "support": model_spec["support"],
            "query": model_spec["query"],
            "learning_rate": model_spec["learning_rate"],
            "weight_decay": model_spec["weight_decay"],
            "hidden_dim": model_spec["hidden_dim"],
            "layers": model_spec["layers"],
            "dropout": model_spec["dropout"],
        }
        observed_experiment = {
            "num_frames": experiment.get("num_frames"),
            "epochs": experiment.get("epochs"),
            "episodes": experiment.get("episodes"),
            "support": experiment.get("support"),
            "query": experiment.get("query"),
            "learning_rate": experiment.get("learning_rate"),
            "weight_decay": experiment.get("weight_decay"),
            "hidden_dim": model_config.get("hidden_dim"),
            "layers": model_config.get("layers"),
            "dropout": model_config.get("dropout"),
        }
        if observed_experiment != expected_experiment:
            raise SystemExit(f"Checkpoint training specification mismatch: {checkpoint_path}")
        checkpoint_records.append(
            {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path), "seed": seed}
        )
        checkpoint_payloads.append(payload)
    if sorted(record["seed"] for record in checkpoint_records) != expected_seeds:
        raise SystemExit(f"Checkpoint seeds must be exactly {expected_seeds}")

    receipt = {
        "status": "started",
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol_sha256": protocol["protocol_sha256"],
        "model_spec_sha256": model_spec["model_spec_sha256"],
        "stage4_manifest_sha256": sha256_file(args.samples),
        "final_train_manifest_sha256": train_manifest_hash,
        "final_train_cache_set_sha256": train_cache_hash,
        "stage4_cache_set_sha256": test_cache_hash,
        "checkpoints": sorted(checkpoint_records, key=lambda row: row["seed"]),
        "no_retraining_or_adaptation": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output_dir / "stage4_evaluation_receipt.json"
    atomic_json(receipt_path, receipt)

    device = select_device(args.device)
    clips = np.stack(
        [
            load_feature_clip(
                args.cache_dir,
                row["sample_id"],
                model_spec["feature_mode"],
                expected_protocol_sha256=protocol["protocol_sha256"],
                expected_source_sha256=row.get("source_video_hash") or None,
                expected_num_frames=int(model_spec["num_frames"]),
            )
            for row in test_rows
        ]
    )
    test_x = torch.from_numpy(clips)
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_true = np.asarray([label_to_index[row["canonical_label"]] for row in test_rows])
    seed_results = []
    for payload in sorted(checkpoint_payloads, key=lambda item: int(item["seed"])):
        config = payload["experiment_config"]["models"]["landmark_bilstm"]
        encoder = LandmarkBiLSTM(
            num_classes=len(labels),
            input_dim=feature_dim(model_spec["feature_mode"]),
            hidden_dim=int(config["hidden_dim"]),
            layers=int(config["layers"]),
            dropout=float(config["dropout"]),
        ).to(device)
        encoder.load_state_dict(payload["state_dict"])
        encoder.eval()
        prototypes = payload["prototypes"].to(device)
        with torch.no_grad():
            embeddings = encoder.forward_features(test_x.to(device))
            pred = torch.cdist(embeddings, prototypes).argmin(dim=1).cpu().numpy()
        correct = y_true == pred
        metrics = hierarchical_classification_metrics(test_rows, correct, labels)
        metrics["seed"] = int(payload["seed"])
        seed_results.append(metrics)
        prediction_rows = []
        for row, true_index, pred_index in zip(test_rows, y_true, pred, strict=True):
            prediction_rows.append(
                {
                    "seed": int(payload["seed"]),
                    "sample_id": row["sample_id"],
                    "signer": row["signer"],
                    "sign_type": row["sign_type"],
                    "condition_group": row.get("condition_group", ""),
                    "true_label": labels[int(true_index)],
                    "predicted_label": labels[int(pred_index)],
                    "correct": int(true_index == pred_index),
                }
            )
        write_csv(
            args.output_dir / f"predictions_seed_{int(payload['seed'])}.csv",
            list(prediction_rows[0]),
            prediction_rows,
        )

    summary: dict[str, object] = {
        "primary_equal_signer_macro_recall": mean_and_sample_sd(
            [float(row["primary_equal_signer_macro_recall"]) for row in seed_results]
        ),
        "secondary_clip_pooled_top1": mean_and_sample_sd(
            [float(row["clip_pooled_top1"]) for row in seed_results]
        ),
        "per_tester": {},
        "per_seed": seed_results,
        "seeds": expected_seeds,
        "testers": testers,
        "estimand": protocol["estimand"],
        "protocol_sha256": protocol["protocol_sha256"],
        "model_spec_sha256": model_spec["model_spec_sha256"],
    }
    for signer in testers:
        summary["per_tester"][signer] = {
            "macro_recall": mean_and_sample_sd(
                [float(row["per_signer"][signer]["macro_recall"]) for row in seed_results]
            ),
            "top1": mean_and_sample_sd(
                [float(row["per_signer"][signer]["top1"]) for row in seed_results]
            ),
            "clips": int(seed_results[0]["per_signer"][signer]["clips"]),
        }
    atomic_json(args.output_dir / "stage4_summary.json", summary)
    receipt["status"] = "complete"
    receipt["completed_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    receipt["summary_sha256"] = sha256_file(args.output_dir / "stage4_summary.json")
    atomic_json(receipt_path, receipt)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
