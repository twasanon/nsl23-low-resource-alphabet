#!/usr/bin/env python3
"""Aggregate improvement-run metrics, enrollment JSON, and fusion JSON."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd


FOLDS = ("heldout_s1", "heldout_s2")
SEEDS = (17, 42, 73)


def pm(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
        "values": [float(v) for v in arr],
    }


def sign_type_from_predictions(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    out = {"accuracy": float(frame["correct"].mean())}
    for name, group in frame.groupby("sign_type"):
        out[f"{name}_accuracy"] = float(group["correct"].mean())
    return out


def pooled_macro_f1(paths: list[Path], classes: int = 49) -> float:
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    values = []
    for klass in range(classes):
        true = frame["true_index"].astype(int) == klass
        predicted = frame["predicted_index"].astype(int) == klass
        tp = int((true & predicted).sum())
        fp = int((~true & predicted).sum())
        fn = int((true & ~predicted).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(values))


def collect_runs(run_dir: Path) -> dict:
    models: dict[str, list[dict]] = defaultdict(list)
    for fold in FOLDS:
        for model_dir in sorted((run_dir / fold).glob("*")):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            for seed in SEEDS:
                metrics_path = model_dir / f"seed_{seed}" / "metrics.json"
                if not metrics_path.is_file():
                    continue
                metrics = json.loads(metrics_path.read_text())
                pred_path = model_dir / f"seed_{seed}" / "test_predictions.csv"
                row = {
                    "fold": fold,
                    "model": model,
                    "seed": seed,
                    "accuracy": float(metrics["test_accuracy"]),
                    "macro_f1": float(metrics.get("test_macro_f1") or metrics.get("macro_f1") or 0.0),
                    "consonant_accuracy": metrics.get("consonant_accuracy") or metrics.get("test_consonant_accuracy"),
                    "vowel_accuracy": metrics.get("vowel_accuracy") or metrics.get("test_vowel_accuracy"),
                    "prediction_path": str(pred_path) if pred_path.is_file() else "",
                }
                if pred_path.is_file():
                    extra = sign_type_from_predictions(pred_path)
                    row["accuracy"] = extra["accuracy"]
                    row["consonant_accuracy"] = extra.get("consonant_accuracy", row["consonant_accuracy"])
                    row["vowel_accuracy"] = extra.get("vowel_accuracy", row["vowel_accuracy"])
                models[model].append(row)
    summary = {}
    for model, rows in models.items():
        pooled = defaultdict(list)
        for seed in SEEDS:
            seed_rows = [row for row in rows if row["seed"] == seed]
            if len(seed_rows) != 2:
                continue
            pooled["accuracy"].append(mean(row["accuracy"] for row in seed_rows))
            prediction_paths = [Path(row["prediction_path"]) for row in seed_rows if row["prediction_path"]]
            pooled["macro_f1"].append(
                pooled_macro_f1(prediction_paths)
                if len(prediction_paths) == 2
                else mean(row["macro_f1"] for row in seed_rows)
            )
            if all(row.get("consonant_accuracy") is not None for row in seed_rows):
                pooled["consonant_accuracy"].append(mean(row["consonant_accuracy"] for row in seed_rows))
            if all(row.get("vowel_accuracy") is not None for row in seed_rows):
                pooled["vowel_accuracy"].append(mean(row["vowel_accuracy"] for row in seed_rows))
        summary[model] = {
            "runs": rows,
            "pooled": {key: pm(values) for key, values in pooled.items()},
            "by_fold": {
                fold: pm([row["accuracy"] for row in rows if row["fold"] == fold])
                for fold in FOLDS
            },
        }
    return summary


def collect_enrollment(result_dir: Path) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(result_dir.glob("enrollment_*.json")):
        payload = json.loads(path.read_text())
        stem = path.stem
        if stem.startswith("enrollment_rawangle_bilstm_"):
            family = "raw_angle_bilstm"
        elif stem.startswith("enrollment_proto_"):
            family = "proto_bilstm"
        elif stem.startswith("enrollment_heldout_"):
            family = "xyz_bilstm"
        else:
            family = stem
        groups[family].append(payload)
    out = {}
    for family, payloads in groups.items():
        methods: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        top3 = []
        for payload in payloads:
            if "top3_softmax" in payload:
                top3.append(float(payload["top3_softmax"]))
            for name, scores in payload["methods"].items():
                methods[name]["accuracy"].append(float(scores["accuracy"]))
                if "consonant_accuracy" in scores:
                    methods[name]["consonant_accuracy"].append(float(scores["consonant_accuracy"]))
                if "vowel_accuracy" in scores:
                    methods[name]["vowel_accuracy"].append(float(scores["vowel_accuracy"]))
        out[family] = {
            "n": len(payloads),
            "methods": {name: {metric: pm(values) for metric, values in metrics.items()} for name, metrics in methods.items()},
            "top3_softmax": pm(top3) if top3 else None,
        }
    return out


def collect_fusion(result_dir: Path) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(result_dir.glob("fusion_*.json")):
        payload = json.loads(path.read_text())
        family = path.stem.split("_seed")[0]
        groups[family].append(payload)
    out = {}
    for family, payloads in groups.items():
        out[family] = {
            "n": len(payloads),
            "landmark": pm([float(p["landmark"]) for p in payloads]),
            "crop": pm([float(p["crop"]) for p in payloads]),
            "fusion": pm([float(p["fusion"]) for p in payloads]),
            "fusion_consonant": pm([float(p["fusion_consonant_accuracy"]) for p in payloads if "fusion_consonant_accuracy" in p]),
            "fusion_vowel": pm([float(p["fusion_vowel_accuracy"]) for p in payloads if "fusion_vowel_accuracy" in p]),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/runs/improvements"))
    parser.add_argument("--result-dir", type=Path, default=Path("artifacts/results/improvements"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/improvements/summary.json"))
    args = parser.parse_args()

    payload = {
        "models": collect_runs(args.run_dir),
        "enrollment": collect_enrollment(args.result_dir),
        "fusion": collect_fusion(args.result_dir),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    rows = []
    for model, block in payload["models"].items():
        pooled = block["pooled"]
        rows.append(
            {
                "family": "zero_shot",
                "name": model,
                "accuracy_mean": pooled["accuracy"]["mean"],
                "accuracy_sd": pooled["accuracy"]["sd"],
                "macro_f1_mean": pooled.get("macro_f1", {}).get("mean"),
                "consonant_mean": pooled.get("consonant_accuracy", {}).get("mean"),
                "vowel_mean": pooled.get("vowel_accuracy", {}).get("mean"),
            }
        )
    for family, block in payload["enrollment"].items():
        closed = block["methods"].get("enroll_closed_other_lighting", {})
        if "accuracy" in closed:
            rows.append(
                {
                    "family": "closed_1shot",
                    "name": family,
                    "accuracy_mean": closed["accuracy"]["mean"],
                    "accuracy_sd": closed["accuracy"]["sd"],
                    "consonant_mean": closed.get("consonant_accuracy", {}).get("mean"),
                    "vowel_mean": closed.get("vowel_accuracy", {}).get("mean"),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output.with_suffix(".csv"), index=False)
    print(json.dumps({k: {kk: vv.get("pooled") if isinstance(vv, dict) and "pooled" in vv else vv for kk, vv in payload[k].items()} if k != "enrollment" else payload[k] for k in payload}, indent=2)[:8000])
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
