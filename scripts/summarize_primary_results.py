#!/usr/bin/env python3
"""Validate and aggregate the frozen NSL23 primary experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nsl23.metrics import classification_metrics, stratified_bootstrap_intervals


MODELS = ("single_frame_cnn", "cnn_bilstm", "cnn_transformer", "landmark_mlp", "landmark_bilstm")
FOLDS = ("heldout_s1", "heldout_s2")
METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")


def load_labels(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = [*payload["consonants"], *payload["vowels"]]
    if len(labels) != 49 or len(set(labels)) != 49:
        raise ValueError("The frozen class list must contain exactly 49 unique labels")
    return labels


def metric_row(frame: pd.DataFrame, num_classes: int) -> tuple[dict[str, float], dict[str, Any]]:
    y_true = frame["true_index"].to_numpy(dtype=np.int64)
    y_pred = frame["predicted_index"].to_numpy(dtype=np.int64)
    metrics = classification_metrics(y_true, y_pred, num_classes)
    scalars = {name: float(metrics[name]) for name in METRICS}
    return scalars, metrics


def write_matrix(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    frame.index.name = "true_label"
    frame.to_csv(path)


def summarize(frame: pd.DataFrame, groups: list[str], value_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(groups, sort=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(groups, key_values, strict=True))
        row["replicates"] = len(group)
        for column in value_columns:
            values = group[column].to_numpy(dtype=np.float64)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{column}_min"] = float(values.min())
            row[f"{column}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/runs/primary"))
    parser.add_argument("--classes", type=Path, default=Path("configs/nsl23_classes.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_local_m4.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results"))
    parser.add_argument("--allow-missing-checkpoints", action="store_true",
                        help="Audit exported predictions/history without weights, as in the compact bundle.")
    args = parser.parse_args()

    labels = load_labels(args.classes)
    label_to_index = {label: index for index, label in enumerate(labels)}
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = [int(value) for value in config["seed_list"]]
    bootstrap_replicates = int(config["evaluation"]["bootstrap_replicates"])
    confidence = float(config["evaluation"]["bootstrap_confidence"])
    expected = {(fold, model, seed) for fold in FOLDS for model in MODELS for seed in seeds}

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    seen: set[tuple[str, str, int]] = set()
    reference_ids: dict[str, tuple[str, ...]] = {}

    for fold, model, seed in sorted(expected):
        run_path = args.run_dir / fold / model / f"seed_{seed}"
        required = [run_path / name for name in ("metrics.json", "test_predictions.csv", "training_history.csv")]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete primary matrix; missing: {missing}")

        metrics = json.loads(required[0].read_text(encoding="utf-8"))
        key = (str(metrics["fold"]), str(metrics["model"]), int(metrics["seed"]))
        if key != (fold, model, seed) or key in seen:
            raise ValueError(f"Unexpected or duplicate run identity at {run_path}: {key}")
        seen.add(key)
        if metrics["selection_policy"] != "fixed_final_epoch" or int(metrics["best_epoch"]) != int(config["training"]["max_epochs"]):
            raise ValueError(f"Run does not use the frozen final-epoch policy: {run_path}")
        if int(metrics["bootstrap_replicates"]) != bootstrap_replicates:
            raise ValueError(f"Run does not use {bootstrap_replicates} bootstrap replicates: {run_path}")
        checkpoint = run_path / "best_checkpoint.pt"
        if not checkpoint.is_file() and not args.allow_missing_checkpoints:
            raise FileNotFoundError(checkpoint)
        export_metrics = dict(metrics)
        export_metrics["checkpoint"] = str(run_path / "best_checkpoint.pt")
        metric_rows.append(export_metrics)

        # ``NA`` is a valid NSL23 class label, not a missing-value sentinel.
        predictions = pd.read_csv(required[1], keep_default_na=False)
        if len(predictions) != 98:
            raise ValueError(f"{run_path} has {len(predictions)} predictions; expected 98")
        if predictions["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate sample IDs in {required[1]}")
        if set(predictions["fold"].astype(str)) != {fold} or set(predictions["model"].astype(str)) != {model}:
            raise ValueError(f"Prediction run identity mismatch in {required[1]}")
        if set(predictions["seed"].astype(int)) != {seed}:
            raise ValueError(f"Prediction seed mismatch in {required[1]}")
        expected_signer = "S1" if fold == "heldout_s1" else "S2"
        if set(predictions["signer"].astype(str)) != {expected_signer}:
            raise ValueError(f"Unexpected held-out signer in {required[1]}")
        if predictions["true_label"].nunique() != 49 or not (predictions.groupby("true_label").size() == 2).all():
            raise ValueError(f"Every held-out class must have exactly two clips in {required[1]}")
        lighting_counts = predictions["lighting"].value_counts().to_dict()
        if lighting_counts != {"bright": 49, "dark": 49}:
            raise ValueError(f"Unexpected held-out lighting counts in {required[1]}: {lighting_counts}")
        if not (predictions["true_label"].map(label_to_index) == predictions["true_index"]).all():
            raise ValueError(f"True-label indices disagree with the frozen class order in {required[1]}")
        if not (predictions["predicted_label"].map(label_to_index) == predictions["predicted_index"]).all():
            raise ValueError(f"Predicted-label indices disagree with the frozen class order in {required[1]}")
        if not (predictions["correct"].astype(int) == (predictions["true_index"] == predictions["predicted_index"]).astype(int)).all():
            raise ValueError(f"Correctness flags disagree with label indices in {required[1]}")
        ids = tuple(sorted(predictions["sample_id"].astype(str)))
        if fold in reference_ids and ids != reference_ids[fold]:
            raise ValueError(f"Test sample IDs vary between runs for {fold}")
        reference_ids.setdefault(fold, ids)
        prediction_frames.append(predictions)

        history = pd.read_csv(required[2])
        if len(history) != int(config["training"]["max_epochs"]):
            raise ValueError(f"{run_path} has {len(history)} epochs; expected {config['training']['max_epochs']}")
        if history["epoch"].astype(int).tolist() != list(range(1, int(config["training"]["max_epochs"]) + 1)):
            raise ValueError(f"Non-contiguous epoch history in {required[2]}")
        history.insert(0, "fold", fold)
        history.insert(1, "model", model)
        history.insert(2, "seed", seed)
        histories.append(history)

    if seen != expected:
        raise ValueError(f"Run identity mismatch: missing={sorted(expected - seen)}, extra={sorted(seen - expected)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_metrics = pd.DataFrame(metric_rows).sort_values(["model", "fold", "seed"])
    run_metrics.to_csv(args.output_dir / "run_metrics.csv", index=False)
    predictions_all = pd.concat(prediction_frames, ignore_index=True).sort_values(["model", "seed", "fold", "sample_id"])
    predictions_all.to_csv(args.output_dir / "predictions_all.csv", index=False)
    histories_all = pd.concat(histories, ignore_index=True).sort_values(["model", "fold", "seed", "epoch"])
    histories_all.to_csv(args.output_dir / "training_histories_all.csv", index=False)

    pooled_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    sign_type_rows: list[dict[str, Any]] = []
    seed_confusions: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {model: [] for model in MODELS}
    for (model, seed), group in predictions_all.groupby(["model", "seed"], sort=False):
        if len(group) != 196 or group["sample_id"].nunique() != 196:
            raise ValueError(f"Pooled out-of-fold set for {model}, seed {seed} is not 196 unique clips")
        scalars, metrics = metric_row(group, len(labels))
        intervals = stratified_bootstrap_intervals(
            group["true_index"].to_numpy(dtype=np.int64),
            group["predicted_index"].to_numpy(dtype=np.int64),
            len(labels),
            bootstrap_replicates,
            confidence,
            seed + 20000 + MODELS.index(model) * 1000,
        )
        pooled_rows.append(
            {
                "model": model,
                "seed": int(seed),
                "test_clips": len(group),
                **scalars,
                "accuracy_ci_low": intervals["accuracy"][0],
                "accuracy_ci_high": intervals["accuracy"][1],
                "macro_f1_ci_low": intervals["macro_f1"][0],
                "macro_f1_ci_high": intervals["macro_f1"][1],
            }
        )
        seed_confusions[model].append((metrics["confusion_counts"], metrics["confusion_normalized"]))
        for index, label in enumerate(labels):
            per_class_rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "class_index": index,
                    "label": label,
                    "sign_type": "consonant" if index < 36 else "vowel",
                    "precision": float(metrics["per_class_precision"][index]),
                    "recall": float(metrics["per_class_recall"][index]),
                    "f1": float(metrics["per_class_f1"][index]),
                    "support": int(metrics["per_class_support"][index]),
                }
            )
        for condition, condition_group in group.groupby("lighting", sort=True):
            condition_scalars, _ = metric_row(condition_group, len(labels))
            condition_rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "lighting": condition,
                    "test_clips": len(condition_group),
                    **condition_scalars,
                }
            )
        for sign_type, type_group in group.groupby("sign_type", sort=True):
            type_scalars, type_metrics = metric_row(type_group, len(labels))
            supported = np.asarray(type_metrics["per_class_support"]) > 0
            sign_type_rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "sign_type": sign_type,
                    "test_clips": len(type_group),
                    "classes": int(supported.sum()),
                    "accuracy": type_scalars["accuracy"],
                    "macro_precision_supported_classes": float(np.asarray(type_metrics["per_class_precision"])[supported].mean()),
                    "macro_recall_supported_classes": float(np.asarray(type_metrics["per_class_recall"])[supported].mean()),
                    "macro_f1_supported_classes": float(np.asarray(type_metrics["per_class_f1"])[supported].mean()),
                }
            )

    pooled = pd.DataFrame(pooled_rows).sort_values(["model", "seed"])
    pooled.to_csv(args.output_dir / "pooled_out_of_fold_by_seed.csv", index=False)
    model_summary = summarize(pooled, ["model"], list(METRICS))
    model_summary.to_csv(args.output_dir / "model_summary.csv", index=False)

    fold_values = [f"test_{name.removeprefix('test_')}" for name in METRICS]
    fold_summary = summarize(run_metrics, ["model", "fold"], fold_values)
    fold_summary.to_csv(args.output_dir / "fold_summary.csv", index=False)

    conditions = pd.DataFrame(condition_rows).sort_values(["model", "lighting", "seed"])
    conditions.to_csv(args.output_dir / "condition_by_seed.csv", index=False)
    condition_summary = summarize(conditions, ["model", "lighting"], list(METRICS))
    condition_summary.to_csv(args.output_dir / "condition_summary.csv", index=False)

    sign_types = pd.DataFrame(sign_type_rows).sort_values(["model", "sign_type", "seed"])
    sign_types.to_csv(args.output_dir / "sign_type_by_seed.csv", index=False)
    sign_type_summary = summarize(
        sign_types,
        ["model", "sign_type"],
        ["accuracy", "macro_precision_supported_classes", "macro_recall_supported_classes", "macro_f1_supported_classes"],
    )
    sign_type_summary.to_csv(args.output_dir / "sign_type_summary.csv", index=False)

    per_class = pd.DataFrame(per_class_rows).sort_values(["model", "class_index", "seed"])
    per_class.to_csv(args.output_dir / "per_class_by_seed.csv", index=False)
    per_class_summary = summarize(per_class, ["model", "class_index", "label", "sign_type"], ["precision", "recall", "f1"])
    per_class_summary.to_csv(args.output_dir / "per_class_summary.csv", index=False)

    comparisons = (
        ("cnn_transformer", "single_frame_cnn"),
        ("cnn_transformer", "cnn_bilstm"),
        ("cnn_bilstm", "single_frame_cnn"),
        ("landmark_bilstm", "single_frame_cnn"),
        ("landmark_bilstm", "cnn_transformer"),
        ("landmark_bilstm", "landmark_mlp"),
        ("landmark_mlp", "single_frame_cnn"),
    )
    paired_rows: list[dict[str, Any]] = []
    for left, right in comparisons:
        left_rows = pooled[pooled["model"] == left].set_index("seed")
        right_rows = pooled[pooled["model"] == right].set_index("seed")
        for seed in seeds:
            for metric in METRICS:
                paired_rows.append(
                    {
                        "comparison": f"{left}_minus_{right}",
                        "seed": seed,
                        "metric": metric,
                        "difference": float(left_rows.loc[seed, metric] - right_rows.loc[seed, metric]),
                    }
                )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.output_dir / "paired_differences_by_seed.csv", index=False)
    paired_summary = summarize(paired, ["comparison", "metric"], ["difference"])
    paired_summary.to_csv(args.output_dir / "paired_differences_summary.csv", index=False)

    for model, matrices in seed_confusions.items():
        counts = np.sum([item[0] for item in matrices], axis=0)
        normalized = np.mean([item[1] for item in matrices], axis=0)
        write_matrix(args.output_dir / f"confusion_{model}_sum_counts.csv", counts, labels)
        write_matrix(args.output_dir / f"confusion_{model}_mean_normalized.csv", normalized, labels)
        top_confusions = []
        per_class_total = int(counts.sum(axis=1)[0])
        if not np.all(counts.sum(axis=1) == per_class_total):
            raise ValueError(f"Unequal aggregated class support for {model}")
        for true_index, predicted_index in np.ndindex(counts.shape):
            if true_index == predicted_index or counts[true_index, predicted_index] == 0:
                continue
            top_confusions.append(
                {
                    "model": model,
                    "true_label": labels[true_index],
                    "predicted_label": labels[predicted_index],
                    "count_across_seeds_and_folds": int(counts[true_index, predicted_index]),
                    "proportion_of_repeated_true_class_predictions": float(counts[true_index, predicted_index] / per_class_total),
                    "repeated_true_class_predictions": per_class_total,
                }
            )
        pd.DataFrame(top_confusions).sort_values(
            ["count_across_seeds_and_folds", "true_label", "predicted_label"], ascending=[False, True, True]
        ).head(25).to_csv(args.output_dir / f"top_confusions_{model}.csv", index=False)

    runtime_rows = (
        histories_all.groupby(["model", "fold", "seed"], as_index=False)
        .agg(training_seconds=("epoch_seconds", "sum"), mean_epoch_seconds=("epoch_seconds", "mean"))
        .sort_values(["model", "fold", "seed"])
    )
    runtime_rows.to_csv(args.output_dir / "training_runtime.csv", index=False)

    summary_payload = {
        "status": "pass",
        "expected_runs": len(expected),
        "completed_runs": len(seen),
        "models": list(MODELS),
        "folds": list(FOLDS),
        "seeds": seeds,
        "unique_heldout_clips_per_seed": 196,
        "bootstrap_replicates_per_pooled_seed": bootstrap_replicates,
        "bootstrap_note": "Intervals are class-stratified, sample-level descriptive intervals; with two held-out signers they do not estimate population-level signer uncertainty.",
        "model_summary": model_summary.to_dict(orient="records"),
    }
    (args.output_dir / "primary_results_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
