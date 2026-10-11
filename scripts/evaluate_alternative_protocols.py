#!/usr/bin/env python3
"""Score the prototypical xyz+angle encoder under common alternative exams.

The 49-way unseen top-1 number is unchanged. This script only records how the
same checkpoints score when graded the way few-shot and retrieval papers grade.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from nsl23.dataset import load_labels, read_rows
from nsl23.geometry import transform_clip
from nsl23.models import build_model


FOLDS = ("heldout_s1", "heldout_s2")
SEEDS = (17, 42, 73)
VOWELS = {"A", "AA", "I", "II", "U", "UU", "RI", "E", "AI", "O", "AU", "AM", "AH"}


def load_encoder(checkpoint: Path, labels: list[str], device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = payload["experiment_config"]
    model = build_model("landmark_bilstm", config, len(labels), int(config["num_frames"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    mode = str(payload.get("feature_mode") or config.get("feature_mode", "raw_angle"))
    return model, mode


def embed(model, rows: list[dict[str, str]], labels: list[str], mode: str, device: torch.device):
    xs, ys, lights = [], [], []
    with torch.no_grad():
        for row in rows:
            array = np.load(Path("data/landmark_cache_primary") / f"{row['sample_id']}.npy")
            clip = torch.from_numpy(transform_clip(array, mode)).unsqueeze(0).to(device)
            xs.append(model.forward_features(clip)[0].cpu().numpy())
            ys.append(labels.index(row["canonical_label"]))
            lights.append(str(row.get("lighting", "")).lower())
    return np.stack(xs), np.asarray(ys), np.asarray(lights)


def class_means(x: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    rows = []
    dim = x.shape[1]
    for class_index in range(num_classes):
        members = x[y == class_index]
        rows.append(members.mean(axis=0) if len(members) else np.zeros(dim, dtype=np.float32))
    return np.stack(rows)


def seed_pool(pairs: list[tuple[int, float]]) -> dict[str, float | list[float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for seed, value in pairs:
        grouped[seed].append(value)
    values = [float(np.mean(grouped[seed])) for seed in sorted(grouped) if len(grouped[seed]) == 2]
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)), "seeds": values}


def five_way_same_signer(x: np.ndarray, y: np.ndarray, *, episodes: int, rng: np.random.Generator) -> float:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(y.tolist()):
        by_class[int(label)].append(index)
    classes = [label for label, indices in by_class.items() if len(indices) >= 2]
    hits = []
    for _ in range(episodes):
        chosen = rng.choice(classes, size=5, replace=False)
        support, query, query_y = [], [], []
        for way, label in enumerate(chosen):
            pick = rng.choice(by_class[int(label)], size=2, replace=False)
            support.append(x[pick[0]])
            query.append(x[pick[1]])
            query_y.append(way)
        support_m = np.stack(support)
        query_m = np.stack(query)
        distances = ((query_m[:, None, :] - support_m[None, :, :]) ** 2).sum(axis=2)
        hits.append(float((distances.argmin(axis=1) == np.asarray(query_y)).mean()))
    return float(np.mean(hits))


def five_way_unseen(prototypes: np.ndarray, x: np.ndarray, y: np.ndarray, *, episodes: int, rng: np.random.Generator) -> float:
    """Restrict candidates to five full training-class means; this is not 1-shot."""
    classes = [c for c in range(prototypes.shape[0]) if (y == c).any()]
    hits = []
    for _ in range(episodes):
        chosen = rng.choice(classes, size=5, replace=False)
        support = np.stack([prototypes[c] for c in chosen])
        query, query_y = [], []
        for way, label in enumerate(chosen):
            index = int(rng.choice(np.where(y == label)[0]))
            query.append(x[index])
            query_y.append(way)
        query_m = np.stack(query)
        distances = ((query_m[:, None, :] - support[None, :, :]) ** 2).sum(axis=2)
        hits.append(float((distances.argmin(axis=1) == np.asarray(query_y)).mean()))
    return float(np.mean(hits))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/runs/improvements"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/improvements/alternative_protocols.json"))
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    labels = load_labels(Path("configs/nsl23_classes.json"))
    consonant_index = [i for i, name in enumerate(labels) if name not in VOWELS]
    device = torch.device(args.device)
    collected: dict[str, list[tuple[int, float]]] = defaultdict(list)
    runs = []

    for fold in FOLDS:
        rows = read_rows(Path(f"data/manifests/folds/{fold}_cached.csv"))
        train_rows = [row for row in rows if row["split"] == "train"]
        test_rows = [row for row in rows if row["split"] == "test"]
        for seed in SEEDS:
            checkpoint = args.run_dir / fold / "proto_bilstm" / f"seed_{seed}" / "best_checkpoint.pt"
            model, mode = load_encoder(checkpoint, labels, device)
            train_x, train_y, _ = embed(model, train_rows, labels, mode, device)
            test_x, test_y, _ = embed(model, test_rows, labels, mode, device)
            prototypes = class_means(train_x, train_y, len(labels))
            distances = ((test_x[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
            order = np.argsort(distances, axis=1)
            top1 = float((order[:, 0] == test_y).mean())
            top3 = float(np.mean([test_y[i] in order[i, :3] for i in range(len(test_y))]))
            top5 = float(np.mean([test_y[i] in order[i, :5] for i in range(len(test_y))]))
            cons_mask = np.isin(test_y, consonant_index)
            dist36 = distances[np.ix_(np.where(cons_mask)[0], consonant_index)]
            pred36 = np.asarray(consonant_index)[dist36.argmin(axis=1)]
            acc36 = float((pred36 == test_y[cons_mask]).mean())
            rng_ss = np.random.default_rng(10_000 + seed)
            rng_us = np.random.default_rng(20_000 + seed)
            five_ss = five_way_same_signer(test_x, test_y, episodes=args.episodes, rng=rng_ss)
            five_us = five_way_unseen(prototypes, test_x, test_y, episodes=args.episodes, rng=rng_us)
            payload = {
                "fold": fold,
                "seed": seed,
                "top1": top1,
                "top3": top3,
                "top5": top5,
                "consonant_36way": acc36,
                "five_way_same_signer": five_ss,
                "five_way_unseen": five_us,
            }
            runs.append(payload)
            for key in ("top1", "top3", "top5", "consonant_36way", "five_way_same_signer", "five_way_unseen"):
                collected[key].append((seed, payload[key]))
            print(json.dumps(payload), flush=True)

    summary = {key: seed_pool(pairs) for key, pairs in collected.items()}
    summary["chance"] = {
        "top1": 1.0 / 49.0,
        "top3": 3.0 / 49.0,
        "top5": 5.0 / 49.0,
        "consonant_36way": 1.0 / 36.0,
        "five_way_same_signer": 0.2,
        "five_way_unseen": 0.2,
        "closed_1shot_49way": 1.0 / 49.0,
    }
    summary["episodes"] = args.episodes
    summary["model"] = "proto_bilstm_raw_angle"
    summary["support_definitions"] = {
        "five_way_unseen": "All training clips per selected class form each mean; five candidate classes.",
        "five_way_same_signer": "One distinct other-lighting clip per class from the query signer.",
    }
    summary["runs"] = runs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
