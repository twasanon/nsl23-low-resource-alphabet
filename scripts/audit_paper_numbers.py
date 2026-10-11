#!/usr/bin/env python3
"""Recompute the paper's main numbers from per-run primitives and fail on drift."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 42, 73)


def close(observed: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(observed, expected, rel_tol=0, abs_tol=tolerance):
        raise AssertionError(f"{observed} != {expected}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def macro_f1(rows: list[dict[str, str]], classes: int = 49) -> float:
    values = []
    for klass in range(classes):
        tp = sum(int(row["true_index"]) == klass and int(row["predicted_index"]) == klass for row in rows)
        fp = sum(int(row["true_index"]) != klass and int(row["predicted_index"]) == klass for row in rows)
        fn = sum(int(row["true_index"]) == klass and int(row["predicted_index"]) != klass for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return statistics.mean(values)


def validate_prediction_rows(
    rows: list[dict[str, str]], expected_rows: list[dict[str, str]], labels: list[str]
) -> None:
    """Check saved predictions against source identities and recompute correctness."""
    expected = {row["sample_id"]: row for row in expected_rows}
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise AssertionError("Prediction sample IDs differ from the evaluation manifest")
    for row in rows:
        source = expected[row["sample_id"]]
        truth, predicted = int(row["true_index"]), int(row["predicted_index"])
        if not (0 <= truth < len(labels) and 0 <= predicted < len(labels)):
            raise AssertionError("Prediction class index is outside the ontology")
        if labels[truth] != source["canonical_label"] or row["true_label"] != labels[truth]:
            raise AssertionError("Prediction truth label differs from the evaluation manifest")
        if row["predicted_label"] != labels[predicted]:
            raise AssertionError("Predicted label and index disagree")
        if int(row["correct"]) != int(truth == predicted):
            raise AssertionError("Saved correctness flag differs from class indices")
        for field in ("signer", "sign_type", "lighting"):
            if row.get(field, "") != source.get(field, ""):
                raise AssertionError(f"Prediction metadata differs for {field}")


def aggregate_run_metrics(run_dir: Path, model_key: str) -> dict[str, dict[str, object]]:
    ontology = json.loads((ROOT / "configs/nsl23_classes.json").read_text(encoding="utf-8"))
    labels = ontology["consonants"] + ontology["vowels"]
    grouped: dict[str, dict[int, list[tuple[dict, Path]]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(run_dir.glob("*/*/seed_*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = (
            path.parent.parent.name
            if run_dir.name == "improvements"
            else str(payload.get(model_key) or payload.get("model") or path.parent.parent.name)
        )
        seed = int(payload["seed"])
        grouped[model][seed].append((payload, path.parent))
    report: dict[str, dict[str, object]] = {}
    for model, seed_runs in sorted(grouped.items()):
        if set(seed_runs) != set(SEEDS) or any(len(runs) != 2 for runs in seed_runs.values()):
            raise AssertionError(f"{model}: expected exactly two folds for seeds {SEEDS}")
        accuracy = []
        pooled_f1 = []
        for seed in SEEDS:
            predictions: list[dict[str, str]] = []
            for payload, directory in seed_runs[seed]:
                fold = directory.parent.parent.name
                expected_rows = [
                    row for row in read_csv(ROOT / f"data/manifests/folds/{fold}_cached.csv")
                    if row["split"] == "test"
                ]
                rows = read_csv(directory / "test_predictions.csv")
                validate_prediction_rows(rows, expected_rows, labels)
                close(statistics.mean(int(row["correct"]) for row in rows), float(payload["test_accuracy"]))
                close(macro_f1(rows), float(payload["test_macro_f1"]))
                predictions.extend(rows)
            if len(predictions) != 196:
                raise AssertionError(f"{model}/seed_{seed}: expected 196 pooled predictions")
            if len({row["sample_id"] for row in predictions}) != 196:
                raise AssertionError(f"{model}/seed_{seed}: duplicate pooled prediction IDs")
            accuracy.append(statistics.mean(float(row["correct"]) for row in predictions))
            pooled_f1.append(macro_f1(predictions))
        report[model] = {
            "accuracy_by_seed": accuracy,
            "accuracy_mean": statistics.mean(accuracy),
            "accuracy_sd": statistics.stdev(accuracy),
            "macro_f1_by_seed": pooled_f1,
            "macro_f1_mean": statistics.mean(pooled_f1),
            "macro_f1_sd": statistics.stdev(pooled_f1),
        }
    return report


def verify_primary() -> dict[str, dict[str, object]]:
    computed = aggregate_run_metrics(ROOT / "artifacts/runs/primary", "model")
    expected = {row["model"]: row for row in read_csv(ROOT / "artifacts/results/model_summary.csv")}
    if set(computed) != set(expected):
        raise AssertionError("Primary model inventory differs from model_summary.csv")
    for model, result in computed.items():
        row = expected[model]
        for metric in ("accuracy_mean", "accuracy_sd", "macro_f1_mean", "macro_f1_sd"):
            close(float(result[metric]), float(row[metric]))

    # Every model/seed must predict the same 196 unique out-of-fold clips once.
    identities: dict[int, set[str]] = {}
    for model in computed:
        for seed in SEEDS:
            sample_ids: list[str] = []
            for fold in ("heldout_s1", "heldout_s2"):
                path = ROOT / f"artifacts/runs/primary/{fold}/{model}/seed_{seed}/test_predictions.csv"
                rows = read_csv(path)
                if len(rows) != 98:
                    raise AssertionError(f"{path}: expected 98 predictions")
                sample_ids.extend(row["sample_id"] for row in rows)
            if len(sample_ids) != 196 or len(set(sample_ids)) != 196:
                raise AssertionError(f"{model}/seed_{seed}: pooled prediction IDs are not unique")
            if seed in identities and identities[seed] != set(sample_ids):
                raise AssertionError(f"{model}/seed_{seed}: model evaluated a different clip set")
            identities[seed] = set(sample_ids)
    return computed


def verify_improvements() -> dict[str, dict[str, object]]:
    computed = aggregate_run_metrics(ROOT / "artifacts/runs/improvements", "model_name")
    expected = json.loads(
        (ROOT / "artifacts/results/improvements/summary.json").read_text(encoding="utf-8")
    )["models"]
    if set(computed) != set(expected):
        raise AssertionError("Improvement model inventory differs from summary.json")
    for model, result in computed.items():
        for metric in ("accuracy", "macro_f1"):
            close(float(result[f"{metric}_mean"]), float(expected[model][metric]["mean"]))
            close(float(result[f"{metric}_sd"]), float(expected[model][metric]["sd"]))
    return computed


def pooled_json_metric(pattern: str, field_path: tuple[str, ...]) -> dict[str, object]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for path in sorted((ROOT / "artifacts/results/improvements").glob(pattern)):
        match = re.search(r"seed(17|42|73)\.json$", path.name)
        if not match:
            raise AssertionError(path)
        payload: object = json.loads(path.read_text(encoding="utf-8"))
        for key in field_path:
            payload = payload[key]  # type: ignore[index]
        by_seed[int(match.group(1))].append(float(payload))
    if set(by_seed) != set(SEEDS) or any(len(values) != 2 for values in by_seed.values()):
        raise AssertionError(f"{pattern}: expected two folds for each seed")
    values = [statistics.mean(by_seed[seed]) for seed in SEEDS]
    return {"by_seed": values, "mean": statistics.mean(values), "sd": statistics.stdev(values)}


def verify_secondary() -> dict[str, object]:
    fusion_proto = pooled_json_metric("fusion_proto_*.json", ("fusion",))
    fusion_angle = pooled_json_metric("fusion_rawangle_*.json", ("fusion",))
    close(float(fusion_proto["mean"]), 0.5969387755102041)
    close(float(fusion_proto["sd"]), 0.03571428571428575)
    close(float(fusion_angle["mean"]), 0.4149659863945578)
    close(float(fusion_angle["sd"]), 0.007793496079856855)

    enrollment_xyz = pooled_json_metric(
        "enrollment_heldout_*.json", ("methods", "enroll_closed_other_lighting", "accuracy")
    )
    enrollment_angle = pooled_json_metric(
        "enrollment_rawangle_bilstm_*.json",
        ("methods", "enroll_closed_other_lighting", "accuracy"),
    )
    enrollment_proto = pooled_json_metric(
        "enrollment_proto_*.json", ("methods", "enroll_closed_other_lighting", "accuracy")
    )
    close(float(enrollment_xyz["mean"]), 0.7619047619047619)
    close(float(enrollment_angle["mean"]), 0.7823129251700681)
    close(float(enrollment_proto["mean"]), 0.7295918367346937)

    protocols = json.loads(
        (ROOT / "artifacts/results/improvements/alternative_protocols.json").read_text(encoding="utf-8")
    )
    close(float(protocols["chance"]["top1"]), 1 / 49)
    close(float(protocols["chance"]["top3"]), 3 / 49)
    close(float(protocols["chance"]["top5"]), 5 / 49)
    close(float(protocols["chance"]["consonant_36way"]), 1 / 36)
    close(float(protocols["chance"]["five_way_unseen"]), 1 / 5)
    close(float(protocols["chance"]["five_way_same_signer"]), 1 / 5)
    for key in ("top1", "top3", "top5", "consonant_36way", "five_way_unseen", "five_way_same_signer"):
        by_seed = defaultdict(list)
        for run in protocols["runs"]:
            by_seed[int(run["seed"])].append(float(run[key]))
        if set(by_seed) != set(SEEDS) or any(len(values) != 2 for values in by_seed.values()):
            raise AssertionError(f"Alternative protocol {key} has an incomplete run matrix")
        pooled = [statistics.mean(by_seed[seed]) for seed in SEEDS]
        close(statistics.mean(pooled), float(protocols[key]["mean"]))
        close(statistics.stdev(pooled), float(protocols[key]["sd"]))
        for computed, stored in zip(pooled, protocols[key]["seeds"], strict=True):
            close(computed, float(stored))
    return {
        "fusion_proto": fusion_proto,
        "fusion_rawangle": fusion_angle,
        "closed_1shot_xyz": enrollment_xyz,
        "closed_1shot_rawangle": enrollment_angle,
        "closed_1shot_proto": enrollment_proto,
        "alternative_protocols": {
            key: protocols[key] for key in ("top1", "top3", "top5", "consonant_36way", "five_way_unseen", "five_way_same_signer")
        },
    }


def verify_dataset_and_v2() -> dict[str, object]:
    manifest = read_csv(ROOT / "data/manifests/nsl23_manifest_all.csv")
    if len(manifest) != 630:
        raise AssertionError("Manifest must contain 630 videos")
    if any(Path(row["source_path"]).is_absolute() for row in manifest):
        raise AssertionError("Active manifest still contains absolute source paths")
    segments = read_csv(ROOT / "annotations/v2/segments.csv")
    tapes = read_csv(ROOT / "annotations/v2/tapes.csv")
    completed_pass1 = sum(bool(row.get("review_pass_1")) for row in segments)
    completed_pass2 = sum(bool(row.get("review_pass_2")) for row in segments)
    if len(segments) != 691 or len(tapes) != 21 or completed_pass1 or completed_pass2:
        raise AssertionError("V2 status no longer matches the pre-annotation gate")
    return {
        "manifest_videos": len(manifest),
        "v2_tapes": len(tapes),
        "v2_source_minutes": sum(float(row["duration_seconds"]) for row in tapes) / 60,
        "v2_proposals": len(segments),
        "pass1_complete": completed_pass1,
        "pass2_complete": completed_pass2,
    }


def verify_references() -> dict[str, object]:
    tex = (ROOT / "manuscript/nsl23_low_resource.tex").read_text(encoding="utf-8")
    bib = (ROOT / "manuscript/references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.MULTILINE))
    cited: set[str] = set()
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", tex):
        cited.update(key.strip() for key in group.split(","))
    if keys != cited:
        raise AssertionError(f"Bibliography mismatch: uncited={keys-cited}, missing={cited-keys}")
    return {"bib_entries": len(keys), "cited_entries": len(cited), "bidirectional_complete": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/audit/full_paper_audit.json")
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "scope": "Primitive-level numerical, dataset, V2 gate, and citation consistency audit; no retraining.",
        "primary": verify_primary(),
        "improvements": verify_improvements(),
        "secondary": verify_secondary(),
        "dataset_and_v2": verify_dataset_and_v2(),
        "references": verify_references(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
