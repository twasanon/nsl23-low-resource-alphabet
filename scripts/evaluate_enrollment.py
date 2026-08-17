#!/usr/bin/env python3
"""Nearest-prototype and 1-shot enrollment evaluation for landmark models."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from nsl23.dataset import load_labels, read_rows
from nsl23.geometry import transform_clip
from nsl23.metrics import classification_metrics
from nsl23.models import build_model


VOWELS = {
    "A", "AA", "I", "II", "U", "UU", "RI", "E", "AI", "O", "AU", "AM", "AH",
}


def load_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = payload["experiment_config"]
    labels = payload["labels"]
    model_name = payload["model_name"]
    if model_name == "proto_bilstm" and "proto_bilstm" not in config.get("models", {}):
        model_name = "landmark_bilstm"
    model = build_model(model_name, config, len(labels), int(config["num_frames"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, labels, config, payload


def feature_mode_from_payload(payload: dict, config: dict) -> str:
    if payload.get("feature_mode"):
        return str(payload["feature_mode"])
    if config.get("feature_mode"):
        return str(config["feature_mode"])
    return "xyz"


def embed_rows(model, rows: list[dict[str, str]], labels: list[str], config: dict, device: torch.device, feature_mode: str):
    mode = feature_mode
    label_to_index = {label: index for index, label in enumerate(labels)}
    embeddings = []
    logits = []
    y = []
    meta = []
    with torch.no_grad():
        for row in rows:
            array = np.load(Path("data/landmark_cache_primary") / f"{row['sample_id']}.npy")
            clip = torch.from_numpy(transform_clip(array, mode)).unsqueeze(0).to(device)
            features = model.forward_features(clip)
            logit = model(clip)
            embeddings.append(features.cpu().numpy()[0])
            logits.append(logit.cpu().numpy()[0])
            y.append(label_to_index[row["canonical_label"]])
            meta.append(row)
    return np.stack(embeddings), np.stack(logits), np.asarray(y), meta


def subset_scores(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], meta: list[dict[str, str]], num_classes: int) -> dict:
    metrics = classification_metrics(y_true, y_pred, num_classes)
    vowel_mask = np.array([row["sign_type"] == "vowel" or row["canonical_label"] in VOWELS for row in meta])
    cons_mask = ~vowel_mask
    out = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "n": int(len(y_true)),
    }
    if cons_mask.any():
        out["consonant_accuracy"] = float((y_true[cons_mask] == y_pred[cons_mask]).mean())
    if vowel_mask.any():
        out["vowel_accuracy"] = float((y_true[vowel_mask] == y_pred[vowel_mask]).mean())
    return out


def nearest_prototype(train_x, train_y, test_x, num_classes: int) -> np.ndarray:
    prototypes = []
    for class_index in range(num_classes):
        members = train_x[train_y == class_index]
        if len(members) == 0:
            prototypes.append(np.zeros(train_x.shape[1], dtype=np.float32))
        else:
            prototypes.append(members.mean(axis=0))
    prototypes = np.stack(prototypes)
    distances = ((test_x[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def topk_accuracy(logits: np.ndarray, y_true: np.ndarray, k: int) -> float:
    top = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([y_true[i] in top[i] for i in range(len(y_true))]))


def enrollment_predictions(train_x, train_y, test_x, test_y, test_meta, num_classes: int) -> dict[str, np.ndarray]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(test_meta):
        grouped[(row["canonical_label"], row["signer"])].append(index)
    one_shot = np.full(len(test_y), -1, dtype=np.int64)
    adapted = np.full(len(test_y), -1, dtype=np.int64)
    train_proto = nearest_prototype(train_x, train_y, test_x, num_classes)
    class_train_proto = []
    for class_index in range(num_classes):
        members = train_x[train_y == class_index]
        class_train_proto.append(members.mean(axis=0) if len(members) else np.zeros(train_x.shape[1]))
    class_train_proto = np.stack(class_train_proto)
    for _, indices in grouped.items():
        if len(indices) < 2:
            for index in indices:
                one_shot[index] = train_proto[index]
                adapted[index] = train_proto[index]
            continue
        for query in indices:
            support = [index for index in indices if index != query]
            support_mean = test_x[support].mean(axis=0)
            distances = ((test_x[query] - np.stack([test_x[s] for s in support])) ** 2).sum(axis=1)
            # 1-shot: the other same-class same-signer clip is the only support item;
            # classify by matching query to that clip's class (always that class) is trivial.
            # Instead: 1-shot means replace that class prototype with the enrolled clip, keep other classes as train prototypes.
            class_index = int(test_y[query])
            prototypes = class_train_proto.copy()
            enrolled = support_mean
            one_shot_prototypes = class_train_proto.copy()
            one_shot_prototypes[class_index] = enrolled
            # True 1-shot closed-set: prototypes are ONLY the enrolled clips of all classes.
            # Built globally below.
            adapted_prototypes = class_train_proto.copy()
            adapted_prototypes[class_index] = 0.5 * class_train_proto[class_index] + 0.5 * enrolled
            adapted[query] = int(((test_x[query] - adapted_prototypes) ** 2).sum(axis=1).argmin())
            one_shot[query] = int(((test_x[query] - one_shot_prototypes) ** 2).sum(axis=1).argmin())
    # Closed 1-shot: one support clip per class from the test signer (the other lighting).
    closed = np.full(len(test_y), -1, dtype=np.int64)
    by_lighting: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(test_meta):
        by_lighting[str(row.get("lighting", "")).lower()].append(index)
    for query, row in enumerate(test_meta):
        query_light = str(row.get("lighting", "")).lower()
        support_light = "dark" if query_light == "bright" else "bright"
        support_indices = by_lighting.get(support_light, [])
        if not support_indices:
            closed[query] = train_proto[query]
            continue
        prototypes = np.zeros((num_classes, test_x.shape[1]), dtype=np.float32)
        filled = np.zeros(num_classes, dtype=bool)
        for index in support_indices:
            prototypes[test_y[index]] = test_x[index]
            filled[test_y[index]] = True
        for class_index in range(num_classes):
            if not filled[class_index]:
                prototypes[class_index] = class_train_proto[class_index]
        closed[query] = int(((test_x[query] - prototypes) ** 2).sum(axis=1).argmin())
    return {
        "train_prototypes": train_proto,
        "enroll_replace_class": one_shot,
        "enroll_adapt_class": adapted,
        "enroll_closed_other_lighting": closed,
    }


def knn(train_x, train_y, test_x, k: int) -> np.ndarray:
    distances = ((test_x[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=2)
    neighbors = distances.argsort(axis=1)[:, :k]
    votes = train_y[neighbors]
    preds = []
    for row in votes:
        values, counts = np.unique(row, return_counts=True)
        preds.append(int(values[counts.argmax()]))
    return np.asarray(preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--classes", type=Path, default=Path("configs/nsl23_classes.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, labels, config, payload = load_model(args.checkpoint, device)
    mode = feature_mode_from_payload(payload, config)
    rows = read_rows(args.manifest)
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    train_x, _, train_y, _ = embed_rows(model, train_rows, labels, config, device, mode)
    test_x, test_logits, test_y, test_meta = embed_rows(model, test_rows, labels, config, device, mode)
    softmax_pred = test_logits.argmax(axis=1)
    methods = enrollment_predictions(train_x, train_y, test_x, test_y, test_meta, len(labels))
    trained_head = payload.get("model_name") != "proto_bilstm"
    if trained_head:
        methods["softmax"] = softmax_pred
    methods["train_5nn"] = knn(train_x, train_y, test_x, k=min(5, len(train_x)))
    report = {
        "checkpoint": str(args.checkpoint),
        "fold": payload.get("fold") or args.manifest.stem.replace("_cached", ""),
        "seed": payload.get("seed"),
        "model": payload.get("model_name"),
        "feature_mode": mode,
        "methods": {},
    }
    if trained_head:
        report["top1_softmax"] = subset_scores(test_y, softmax_pred, labels, test_meta, len(labels))
        report["top3_softmax"] = topk_accuracy(test_logits, test_y, 3)
    for name, pred in methods.items():
        report["methods"][name] = subset_scores(test_y, pred, labels, test_meta, len(labels))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
