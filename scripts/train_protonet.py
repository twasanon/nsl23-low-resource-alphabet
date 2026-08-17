#!/usr/bin/env python3
"""Train a prototypical-network encoder on landmark sequences."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nsl23.dataset import load_labels, read_rows
from nsl23.geometry import frame_dim, transform_clip
from nsl23.metrics import classification_metrics
from nsl23.models import LandmarkBiLSTM
from nsl23.train import seed_everything, select_device, write_csv


def load_split(rows: list[dict[str, str]], labels: list[str], mode: str) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, str]]]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    clips = []
    y = []
    kept = []
    for row in rows:
        array = np.load(Path("data/landmark_cache_primary") / f"{row['sample_id']}.npy")
        clips.append(transform_clip(array, mode))
        y.append(label_to_index[row["canonical_label"]])
        kept.append(row)
    return torch.from_numpy(np.stack(clips)), torch.tensor(y, dtype=torch.long), kept


def prototypical_loss(support: torch.Tensor, support_y: torch.Tensor, query: torch.Tensor, query_y: torch.Tensor, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    dim = support.shape[1]
    prototypes = []
    for class_index in range(num_classes):
        members = support[support_y == class_index]
        if len(members) == 0:
            prototypes.append(torch.zeros(dim, device=support.device))
        else:
            prototypes.append(members.mean(dim=0))
    prototypes = torch.stack(prototypes)
    distances = torch.cdist(query, prototypes)
    logits = -distances
    loss = nn.functional.cross_entropy(logits, query_y)
    return loss, logits.argmax(dim=1)


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
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = select_device(args.device)
    labels = load_labels(args.classes)
    rows = read_rows(args.samples)
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    train_x, train_y, _ = load_split(train_rows, labels, args.feature_mode)
    test_x, test_y, test_meta = load_split(test_rows, labels, args.feature_mode)
    encoder = LandmarkBiLSTM(num_classes=len(labels), input_dim=frame_dim(args.feature_mode), hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=3e-4, weight_decay=1e-4)
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
                chosen = rng.choice(pool, size=args.support + args.query, replace=len(pool) < args.support + args.query)
                support_idx.extend(chosen[: args.support].tolist())
                query_idx.extend(chosen[args.support :].tolist())
            support = encoder.forward_features(train_x[support_idx].to(device))
            query = encoder.forward_features(train_x[query_idx].to(device))
            loss, _ = prototypical_loss(support, train_y[support_idx].to(device), query, train_y[query_idx].to(device), len(labels))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        history.append({"epoch": epoch, "loss": epoch_loss / args.episodes})
        print(json.dumps(history[-1]), flush=True)

    encoder.eval()
    with torch.no_grad():
        train_embed = encoder.forward_features(train_x.to(device)).cpu()
        test_embed = encoder.forward_features(test_x.to(device)).cpu()
    prototypes = []
    for class_index in range(len(labels)):
        members = train_embed[train_y == class_index]
        prototypes.append(members.mean(dim=0) if len(members) else torch.zeros(train_embed.shape[1]))
    prototypes = torch.stack(prototypes)
    pred = torch.cdist(test_embed, prototypes).argmin(dim=1).numpy()
    y_true = test_y.numpy()
    metrics = classification_metrics(y_true, pred, len(labels))
    vowel = np.array([row["sign_type"] == "vowel" for row in test_meta])
    payload = {
        "model_name": "proto_bilstm",
        "feature_mode": args.feature_mode,
        "seed": args.seed,
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "consonant_accuracy": float((y_true[~vowel] == pred[~vowel]).mean()),
        "vowel_accuracy": float((y_true[vowel] == pred[vowel]).mean()),
        "labels": labels,
        "experiment_config": {"feature_mode": args.feature_mode, "num_frames": 16, "models": {"landmark_bilstm": {"hidden_dim": args.hidden_dim, "layers": 1, "dropout": 0.3}}},
        "state_dict": encoder.state_dict(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_dir / "best_checkpoint.pt")
    write_csv(args.output_dir / "training_history.csv", history)
    (args.output_dir / "metrics.json").write_text(json.dumps({k: payload[k] for k in payload if k not in {"state_dict", "labels", "experiment_config"}}, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in payload if k not in {"state_dict", "labels", "experiment_config"}}, indent=2))


if __name__ == "__main__":
    main()
