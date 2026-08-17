"""Train one frozen-MobileNetV2 NSL23 model and export all test artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import torchvision
from torch import nn

from .dataset import make_crop_loaders, make_landmark_loaders, make_loaders
from .metrics import classification_metrics, stratified_bootstrap_intervals
from .models import build_model, parameter_counts


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def epoch_pass(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    num_classes: int,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    sample_ids: list[str] = []
    start = time.perf_counter()
    for batch in loader:
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(frames)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(labels)
        predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
        confidences.append(logits.softmax(dim=1).amax(dim=1).detach().cpu().numpy())
        truths.append(labels.detach().cpu().numpy())
        sample_ids.extend(batch["sample_id"])
    synchronize(device)
    y_true = np.concatenate(truths)
    y_pred = np.concatenate(predictions)
    metrics = classification_metrics(y_true, y_pred, num_classes)
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": np.concatenate(confidences),
        "sample_ids": sample_ids,
        "seconds": time.perf_counter() - start,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_matrix(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    rows = [
        {"true_label": label, **{predicted: matrix[row, column] for column, predicted in enumerate(labels)}}
        for row, label in enumerate(labels)
    ]
    write_csv(path, rows, ["true_label", *labels])


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_snapshot(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "scikit_learn": sklearn.__version__,
        "device": str(device),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "Apple M4" if device.type == "mps" else "CPU",
    }


def training_class_weights(rows: list[dict[str, str]], labels: list[str], device: torch.device) -> torch.Tensor:
    label_to_index = {label: index for index, label in enumerate(labels)}
    counts = np.zeros(len(labels), dtype=np.float64)
    for row in rows:
        counts[label_to_index[row["canonical_label"]]] += 1.0
    if np.any(counts == 0):
        missing = [labels[index] for index in np.flatnonzero(counts == 0)]
        raise ValueError(f"Training fold is missing labels: {missing}")
    weights = (1.0 / counts) / np.mean(1.0 / counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/experiment_local_m4.json"))
    parser.add_argument("--classes", type=Path, default=Path("configs/nsl23_classes.json"))
    parser.add_argument(
        "--model",
        choices=["single_frame_cnn", "cnn_bilstm", "cnn_transformer", "landmark_mlp", "landmark_bilstm", "crop_cnn"],
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    config = json.loads(args.experiment_config.read_text(encoding="utf-8"))
    seed_everything(args.seed)
    device = select_device(args.device)
    if args.model.startswith("landmark_"):
        loader_factory = make_landmark_loaders
    elif args.model.startswith("crop_"):
        loader_factory = make_crop_loaders
    else:
        loader_factory = make_loaders
    loaders, labels, rows_by_split = loader_factory(args.samples, args.classes, config, args.seed, args.num_workers)
    fold_values = {row.get("fold", "unspecified") for rows in rows_by_split.values() for row in rows}
    if len(fold_values) != 1:
        raise ValueError(f"Expected exactly one fold in a training manifest, found {sorted(fold_values)}")
    fold_name = next(iter(fold_values))
    train_signers = sorted({row["signer"] for row in rows_by_split["train"]}, key=lambda value: int(value[1:]))
    test_signers = sorted({row["signer"] for row in rows_by_split["test"]}, key=lambda value: int(value[1:]))
    sample_manifest_sha256 = file_hash(args.samples)
    num_classes = len(labels)
    model = build_model(args.model, config, num_classes, int(config["num_frames"])).to(device)
    counts = parameter_counts(model)
    training = config["training"]
    max_epochs = args.max_epochs or int(training["max_epochs"])
    selection_policy = str(training["selection_policy"])
    if selection_policy not in {"fixed_final_epoch", "validation_early_stopping"}:
        raise ValueError(f"Unknown selection policy: {selection_policy}")
    if selection_policy == "validation_early_stopping" and "validation" not in loaders:
        raise ValueError("validation_early_stopping requires validation rows")
    patience = min(int(training.get("early_stopping_patience", max_epochs)), max_epochs)
    if training["loss"] != "class_weighted_cross_entropy":
        raise ValueError(f"Unsupported loss: {training['loss']}")
    class_weights = training_class_weights(rows_by_split["train"], labels, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(training["label_smoothing"]))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_score: float | None = None
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        train_result = epoch_pass(model, loaders["train"], device, criterion, optimizer, num_classes)
        validation_result = (
            epoch_pass(model, loaders["validation"], device, criterion, None, num_classes)
            if selection_policy == "validation_early_stopping"
            else None
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_result["loss"],
            "train_accuracy": train_result["accuracy"],
            "train_macro_f1": train_result["macro_f1"],
            "validation_loss": validation_result["loss"] if validation_result else None,
            "validation_accuracy": validation_result["accuracy"] if validation_result else None,
            "validation_macro_f1": validation_result["macro_f1"] if validation_result else None,
            "epoch_seconds": train_result["seconds"] + (validation_result["seconds"] if validation_result else 0.0),
        }
        history.append(row)
        print(json.dumps({"model": args.model, "seed": args.seed, **row}), flush=True)
        if selection_policy == "fixed_final_epoch":
            if epoch == max_epochs:
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            score = float(validation_result["macro_f1"])
            if best_score is None or score > best_score + 1e-12:
                best_score = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    write_csv(args.output_dir / "training_history.csv", history)
    checkpoint_path = args.output_dir / "best_checkpoint.pt"
    torch.save(
        {
            "model_name": args.model,
            "fold": fold_name,
            "seed": args.seed,
            "epoch": best_epoch,
            "selection_policy": selection_policy,
            "validation_macro_f1": best_score,
            "state_dict": best_state,
            "labels": labels,
            "experiment_config": config,
            "experiment_config_sha256": config_hash(config),
            "sample_manifest": str(args.samples.resolve()),
            "sample_manifest_sha256": sample_manifest_sha256,
            "environment": environment_snapshot(device),
            "class_weights": class_weights.detach().cpu(),
        },
        checkpoint_path,
    )

    # The test set is first evaluated only after the validation-selected checkpoint is fixed.
    test_result = epoch_pass(model, loaders["test"], device, criterion, None, num_classes)
    metrics = classification_metrics(test_result["y_true"], test_result["y_pred"], num_classes)
    bootstrap = stratified_bootstrap_intervals(
        test_result["y_true"],
        test_result["y_pred"],
        num_classes,
        args.bootstrap_replicates or int(config["evaluation"]["bootstrap_replicates"]),
        float(config["evaluation"]["bootstrap_confidence"]),
        args.seed + 10000,
    )
    scalar_metrics = {
        "model": args.model,
        "fold": fold_name,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "selection_policy": selection_policy,
        "validation_macro_f1": best_score,
        "test_loss": test_result["loss"],
        "test_accuracy": metrics["accuracy"],
        "test_macro_precision": metrics["macro_precision"],
        "test_macro_recall": metrics["macro_recall"],
        "test_macro_f1": metrics["macro_f1"],
        "test_weighted_f1": metrics["weighted_f1"],
        "accuracy_ci_low": bootstrap["accuracy"][0],
        "accuracy_ci_high": bootstrap["accuracy"][1],
        "macro_f1_ci_low": bootstrap["macro_f1"][0],
        "macro_f1_ci_high": bootstrap["macro_f1"][1],
        "bootstrap_replicates": args.bootstrap_replicates or int(config["evaluation"]["bootstrap_replicates"]),
        "train_samples": len(rows_by_split["train"]),
        "test_samples": len(rows_by_split["test"]),
        "train_signers": train_signers,
        "test_signers": test_signers,
        **counts,
        "checkpoint": str(checkpoint_path.resolve()),
        "config_sha256": config_hash(config),
        "sample_manifest_sha256": sample_manifest_sha256,
        "class_weight_min": float(class_weights.min().detach().cpu()),
        "class_weight_max": float(class_weights.max().detach().cpu()),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(scalar_metrics, indent=2) + "\n", encoding="utf-8")

    metadata = {row["sample_id"]: row for row in rows_by_split["test"]}
    predictions = []
    for identifier, true_index, predicted_index, confidence in zip(
        test_result["sample_ids"],
        test_result["y_true"],
        test_result["y_pred"],
        test_result["confidence"],
        strict=True,
    ):
        row = metadata[identifier]
        predictions.append(
            {
                "fold": fold_name,
                "model": args.model,
                "seed": args.seed,
                "sample_id": identifier,
                "source_kind": row["source_kind"],
                "relative_path": row["relative_path"],
                "signer": row["signer"],
                "sign_type": row["sign_type"],
                "condition_group": row["condition_group"],
                "background_condition": row["background_condition"],
                "lighting": row["lighting"],
                "indoor_outdoor": row["indoor_outdoor"],
                "true_index": int(true_index),
                "true_label": labels[int(true_index)],
                "predicted_index": int(predicted_index),
                "predicted_label": labels[int(predicted_index)],
                "predicted_confidence": float(confidence),
                "correct": int(true_index == predicted_index),
            }
        )
    write_csv(args.output_dir / "test_predictions.csv", predictions)

    per_class = [
        {
            "class_index": index,
            "label": label,
            "precision": float(metrics["per_class_precision"][index]),
            "recall": float(metrics["per_class_recall"][index]),
            "f1": float(metrics["per_class_f1"][index]),
            "support": int(metrics["per_class_support"][index]),
        }
        for index, label in enumerate(labels)
    ]
    write_csv(args.output_dir / "per_class_metrics.csv", per_class)
    save_matrix(args.output_dir / "confusion_matrix_counts.csv", metrics["confusion_counts"], labels)
    save_matrix(args.output_dir / "confusion_matrix_normalized.csv", metrics["confusion_normalized"], labels)
    (args.output_dir / "environment.json").write_text(json.dumps(environment_snapshot(device), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scalar_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
