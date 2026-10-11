#!/usr/bin/env python3
"""Train a prototypical encoder with explicit V2 train/evaluation modes."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nsl23.dataset import load_labels, read_rows
from nsl23.metrics import classification_metrics
from nsl23.models import LandmarkBiLSTM
from nsl23.train import environment_snapshot, seed_everything, select_device, write_csv
from nsl23.v2_annotation import sha256_file
from nsl23.v2_features import cache_set_sha256, feature_dim, load_feature_clip
from nsl23.v2_protocol import (
    load_and_validate_frozen_protocol,
    load_and_validate_model_spec,
    validate_manifest_provenance,
)


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_split(
    rows: list[dict[str, str]],
    labels: list[str],
    mode: str,
    cache_dir: Path,
    protocol_sha256: str | None = None,
    expected_num_frames: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, str]]]:
    if not rows:
        raise ValueError("Cannot load an empty split")
    label_to_index = {label: index for index, label in enumerate(labels)}
    clips = []
    y = []
    kept = []
    for row in rows:
        clips.append(
            load_feature_clip(
                cache_dir,
                row["sample_id"],
                mode,
                expected_protocol_sha256=protocol_sha256,
                expected_source_sha256=row.get("source_video_hash") or None,
                expected_num_frames=expected_num_frames,
            )
        )
        y.append(label_to_index[row["canonical_label"]])
        kept.append(row)
    return torch.from_numpy(np.stack(clips)), torch.tensor(y, dtype=torch.long), kept


def prototypical_loss(
    support: torch.Tensor,
    support_y: torch.Tensor,
    query: torch.Tensor,
    query_y: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = support.shape[1]
    prototypes = []
    for class_index in range(num_classes):
        members = support[support_y == class_index]
        prototypes.append(
            members.mean(dim=0) if len(members) else torch.zeros(dim, device=support.device)
        )
    prototypes = torch.stack(prototypes)
    logits = -torch.cdist(query, prototypes)
    loss = nn.functional.cross_entropy(logits, query_y)
    return loss, logits.argmax(dim=1)


def per_signer_metrics(
    meta: list[dict[str, str]], y_true: np.ndarray, pred: np.ndarray, n_classes: int
) -> dict[str, dict[str, float]]:
    result = {}
    for signer in sorted({row.get("signer", "") for row in meta}):
        mask = np.array([row.get("signer", "") == signer for row in meta])
        metrics = classification_metrics(y_true[mask], pred[mask], n_classes)
        result[signer] = {
            "top1": metrics["accuracy"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "clips": int(mask.sum()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--classes", type=Path, default=Path("configs/nsl23_classes.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--feature-mode", default="raw_angle")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--support", type=int, default=5)
    parser.add_argument("--query", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/landmark_cache_primary"))
    parser.add_argument("--run-mode", choices=["auto", "train_eval", "train_only"], default="auto")
    parser.add_argument("--v2-protocol", type=Path, default=None)
    parser.add_argument("--v2-model-spec", type=Path, default=None)
    args = parser.parse_args()

    for filename in ("best_checkpoint.pt", "metrics.json", "training_history.csv", "training_receipt.json"):
        existing = args.output_dir / filename
        if existing.exists():
            raise SystemExit(f"Run artifact already exists: {existing}. Choose a fresh --output-dir.")

    labels = load_labels(args.classes)
    rows = read_rows(args.samples)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]
    v2_stages = {row.get("v2_stage") or "" for row in rows if row.get("v2_stage")}
    is_v2 = bool(v2_stages)
    protocol = None
    model_spec = None
    stage = ""
    expected_run_mode = "train_eval"

    if is_v2:
        if len(v2_stages) != 1:
            raise SystemExit(f"V2 manifest mixes stages: {sorted(v2_stages)}")
        stage = next(iter(v2_stages))
        if args.v2_protocol is None:
            raise SystemExit("A V2 manifest requires --v2-protocol; the safety gate is not optional")
        try:
            protocol = load_and_validate_frozen_protocol(args.v2_protocol)
            validate_manifest_provenance(rows, protocol, stage)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if sha256_file(args.classes) != protocol["classes_sha256"]:
            raise SystemExit("--classes does not match the class ontology frozen in V2")
        if val_rows:
            raise SystemExit("V2 manifests use explicit LOSO test folds, not an ignored validation split")
        protected = set(protocol.get("test_hierarchy") or [])
        leaked = [row for row in train_rows if row["signer"] in protected]
        if leaked:
            raise SystemExit(
                "Sealed-test leak: "
                + ", ".join(sorted({f"{row['signer']}:{row['split']}" for row in leaked}))
            )
        allowed_seeds = set(protocol["estimand"]["seeds"])
        if args.seed not in allowed_seeds:
            raise SystemExit(f"V2 seed must be one of {sorted(allowed_seeds)}")
        training = protocol["model_training"]
        locked_arguments = {
            "epochs": int(training["epochs"]),
            "episodes": int(training["episodes_per_epoch"]),
            "support": int(training["support_per_class"]),
            "query": int(training["query_per_class"]),
            "hidden_dim": int(training["hidden_dim"]),
            "layers": int(training["layers"]),
            "dropout": float(training["dropout"]),
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
        }
        observed_arguments = {
            "epochs": args.epochs,
            "episodes": args.episodes,
            "support": args.support,
            "query": args.query,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        }
        if observed_arguments != locked_arguments:
            raise SystemExit(
                "V2 training arguments do not match the frozen protocol: "
                f"expected {locked_arguments}, observed {observed_arguments}"
            )
        if stage == "stage4_eval":
            raise SystemExit("Stage 4 is evaluation-only; use scripts/evaluate_v2_stage4.py")
        if stage == "stage2":
            expected_mode = protocol["stage2"]["feature_mode"]
            if args.feature_mode != expected_mode:
                raise SystemExit(f"Stage 2 feature mode is locked to {expected_mode}")
        elif stage == "stage3":
            allowed_modes = {trial["id"] for trial in protocol["stage3"]["trials"]}
            if args.feature_mode not in allowed_modes:
                raise SystemExit(f"Stage 3 mode must be one of {sorted(allowed_modes)}")
        elif stage == "final_retrain":
            expected_run_mode = "train_only"
            if args.v2_model_spec is None:
                raise SystemExit("Final retrain requires --v2-model-spec")
            try:
                model_spec = load_and_validate_model_spec(args.v2_model_spec, protocol)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            if args.feature_mode != model_spec["feature_mode"]:
                raise SystemExit(f"Final feature mode is locked to {model_spec['feature_mode']}")
            if args.epochs != int(model_spec["epochs"]):
                raise SystemExit(f"Final epochs are locked to {model_spec['epochs']}")
            if args.seed not in set(model_spec["seeds"]):
                raise SystemExit(f"Final seed must be one of {model_spec['seeds']}")
            for field, observed in (
                ("episodes", args.episodes),
                ("support", args.support),
                ("query", args.query),
                ("hidden_dim", args.hidden_dim),
                ("layers", args.layers),
                ("dropout", args.dropout),
                ("learning_rate", args.learning_rate),
                ("weight_decay", args.weight_decay),
            ):
                if observed != model_spec[field]:
                    raise SystemExit(f"Final {field} is locked to {model_spec[field]}")
        else:
            raise SystemExit(f"Unknown V2 stage {stage}")
    elif args.v2_protocol is not None:
        raise SystemExit("--v2-protocol was supplied, but the manifest has no V2 provenance fields")

    run_mode = expected_run_mode if args.run_mode == "auto" else args.run_mode
    if is_v2 and run_mode != expected_run_mode:
        raise SystemExit(f"{stage} requires --run-mode {expected_run_mode}")
    if not train_rows:
        raise SystemExit("Training split is empty")
    missing_train = set(labels) - {row["canonical_label"] for row in train_rows}
    if missing_train:
        raise SystemExit(f"Training split is missing classes: {sorted(missing_train)}")
    if run_mode == "train_eval" and not test_rows:
        raise SystemExit("train_eval requires a nonempty test split")
    if run_mode == "train_only" and test_rows:
        raise SystemExit("train_only refuses a manifest containing test rows")
    seed_everything(args.seed)
    device = select_device(args.device)
    train_x, train_y, _ = load_split(
        train_rows,
        labels,
        args.feature_mode,
        args.cache_dir,
        protocol.get("protocol_sha256") if protocol else None,
        int(protocol["feature_extraction"]["frames_per_clip"]) if protocol else None,
    )
    train_cache_sha256 = cache_set_sha256(
        args.cache_dir,
        [row["sample_id"] for row in train_rows],
        require_npz=is_v2,
    )
    encoder = LandmarkBiLSTM(
        num_classes=len(labels),
        input_dim=feature_dim(args.feature_mode),
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(train_y.tolist()):
        grouped[label].append(index)
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        epoch_loss = 0.0
        for _ in range(args.episodes):
            support_idx = []
            query_idx = []
            for class_index in range(len(labels)):
                pool = grouped[class_index]
                chosen = rng.choice(
                    pool, size=args.support + args.query, replace=len(pool) < args.support + args.query
                )
                support_idx.extend(chosen[: args.support].tolist())
                query_idx.extend(chosen[args.support :].tolist())
            support = encoder.forward_features(train_x[support_idx].to(device))
            query = encoder.forward_features(train_x[query_idx].to(device))
            loss, _ = prototypical_loss(
                support,
                train_y[support_idx].to(device),
                query,
                train_y[query_idx].to(device),
                len(labels),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        history.append({"epoch": epoch, "loss": epoch_loss / args.episodes})
        print(json.dumps(history[-1]), flush=True)

    encoder.eval()
    with torch.no_grad():
        train_embed = encoder.forward_features(train_x.to(device)).cpu()
    prototypes = []
    for class_index in range(len(labels)):
        members = train_embed[train_y == class_index]
        prototypes.append(members.mean(dim=0))
    prototypes = torch.stack(prototypes)
    if not torch.isfinite(prototypes).all():
        raise RuntimeError("Training produced non-finite prototypes")
    manifest_hash = sha256_file(args.samples)
    payload: dict[str, object] = {
        "model_name": "proto_bilstm",
        "feature_mode": args.feature_mode,
        "seed": args.seed,
        "stage": stage or "v1",
        "run_mode": run_mode,
        "labels": labels,
        "prototypes": prototypes,
        "manifest_sha256": manifest_hash,
        "protocol_sha256": protocol.get("protocol_sha256") if protocol else None,
        "segments_sha256": protocol.get("segments_sha256") if protocol else None,
        "model_spec_sha256": model_spec.get("model_spec_sha256") if model_spec else None,
        "train_cache_set_sha256": train_cache_sha256,
        "environment": environment_snapshot(device),
        "experiment_config": {
            "feature_mode": args.feature_mode,
            "num_frames": (
                int(protocol["feature_extraction"]["frames_per_clip"])
                if protocol
                else int(train_x.shape[1])
            ),
            "epochs": args.epochs,
            "episodes": args.episodes,
            "support": args.support,
            "query": args.query,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "models": {
                "landmark_bilstm": {
                    "hidden_dim": args.hidden_dim,
                    "layers": args.layers,
                    "dropout": args.dropout,
                }
            },
        },
        "state_dict": encoder.state_dict(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "training_history.csv", history)
    if run_mode == "train_eval":
        test_x, test_y, test_meta = load_split(
            test_rows,
            labels,
            args.feature_mode,
            args.cache_dir,
            protocol.get("protocol_sha256") if protocol else None,
            int(protocol["feature_extraction"]["frames_per_clip"]) if protocol else None,
        )
        test_cache_sha256 = cache_set_sha256(
            args.cache_dir,
            [row["sample_id"] for row in test_rows],
            require_npz=is_v2,
        )
        with torch.no_grad():
            test_embed = encoder.forward_features(test_x.to(device)).cpu()
        pred = torch.cdist(test_embed, prototypes).argmin(dim=1).numpy()
        y_true = test_y.numpy()
        metrics = classification_metrics(y_true, pred, len(labels))
        vowel = np.array([row["sign_type"] == "vowel" for row in test_meta])
        result = {
            "model_name": "proto_bilstm",
            "feature_mode": args.feature_mode,
            "seed": args.seed,
            "stage": stage or "v1",
            "fold": next((row.get("fold") for row in test_meta if row.get("fold")), None),
            "test_accuracy": metrics["accuracy"],
            "test_macro_recall": metrics["macro_recall"],
            "test_macro_f1": metrics["macro_f1"],
            "consonant_accuracy": float((y_true[~vowel] == pred[~vowel]).mean()),
            "vowel_accuracy": float((y_true[vowel] == pred[vowel]).mean()),
            "per_signer": per_signer_metrics(test_meta, y_true, pred, len(labels)),
            "manifest_sha256": manifest_hash,
            "protocol_sha256": payload["protocol_sha256"],
            "train_cache_set_sha256": train_cache_sha256,
            "test_cache_set_sha256": test_cache_sha256,
        }
        payload.update(result)
        pred_rows = []
        for row, true_index, pred_index in zip(test_meta, y_true, pred, strict=True):
            pred_rows.append(
                {
                    "fold": row.get("fold", ""),
                    "model": "proto_bilstm",
                    "seed": args.seed,
                    "sample_id": row["sample_id"],
                    "signer": row.get("signer", ""),
                    "sign_type": row.get("sign_type", ""),
                    "lighting": row.get("lighting", ""),
                    "true_index": int(true_index),
                    "true_label": labels[int(true_index)],
                    "predicted_index": int(pred_index),
                    "predicted_label": labels[int(pred_index)],
                    "correct": int(true_index == pred_index),
                }
            )
        write_csv(args.output_dir / "test_predictions.csv", pred_rows)
        (args.output_dir / "metrics.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
    else:
        result = {
            "model_name": "proto_bilstm",
            "feature_mode": args.feature_mode,
            "seed": args.seed,
            "stage": stage or "v1",
            "run_mode": run_mode,
            "manifest_sha256": manifest_hash,
            "protocol_sha256": payload["protocol_sha256"],
            "model_spec_sha256": payload["model_spec_sha256"],
            "test_evaluated": False,
            "train_cache_set_sha256": train_cache_sha256,
        }
        (args.output_dir / "training_receipt.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
    atomic_torch_save(payload, args.output_dir / "best_checkpoint.pt")


if __name__ == "__main__":
    main()
