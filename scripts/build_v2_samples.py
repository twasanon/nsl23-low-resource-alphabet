#!/usr/bin/env python3
"""Build fail-closed Version 2 manifests from frozen, hashed artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.samples import build_samples, condition_group, sample_id, summarize, truthy
from nsl23.v2_annotation import SHA256_RE, read_segments, sha256_file, write_csv
from nsl23.v2_protocol import load_and_validate_frozen_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/v2_protocol.json"
MANIFEST = ROOT / "data/manifests/nsl23_manifest_all.csv"
VERIFIED = ROOT / "annotations/v2/verified_segments_v2.0.csv"
COVERAGE = ROOT / "annotations/v2/coverage_v2.0.csv"
AGREEMENT = ROOT / "annotations/v2/agreement_pre_adjudication.json"
SOURCE_INVENTORY = ROOT / "data/manifests/v2_source_inventory_v2.0.csv"
CLASSES = ROOT / "configs/nsl23_classes.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def isolated_samples(manifest: Path) -> list[dict[str, object]]:
    return [row for row in build_samples(read_csv(manifest), [])]


def segment_sample(row: dict[str, str], manifest_by_path: dict[str, dict[str, str]]) -> dict[str, object]:
    meta = manifest_by_path.get(row["source_video"], {})
    realworld = truthy(meta.get("realworld_flag", ""))
    background = row.get("condition") or meta.get("background_condition", "")
    identifier = sample_id(
        "v2seg", row["segment_id"], row["class"], row["start_time_s"], row["end_time_s"]
    )
    return {
        "sample_id": identifier,
        "source_kind": "verified_segment",
        "source_path": row["source_path"],
        "relative_path": row["source_video"],
        "source_zip": meta.get("source_zip", ""),
        "signer": row["signer"],
        "sign_type": row["sign_type"],
        "canonical_label": row["class"],
        "start_seconds": row["start_time_s"],
        "end_seconds": row["end_time_s"],
        "background_condition": background,
        "scene_type": meta.get("scene_type", ""),
        "realworld_flag": realworld,
        "lighting": row.get("lighting") or meta.get("lighting", ""),
        "indoor_outdoor": meta.get("indoor_outdoor", ""),
        "condition_group": condition_group(background, realworld),
        "annotation_status": "v2_pass2_verified",
        "source_video_hash": row.get("source_video_hash", ""),
        "capture_device": row.get("capture_device", ""),
    }


def assert_class_coverage(
    samples: list[dict[str, object]], expected: list[str], stage: str
) -> None:
    expected_set = set(expected)
    train = [row for row in samples if row["split"] == "train"]
    test = [row for row in samples if row["split"] == "test"]
    if stage != "stage4_eval":
        missing = expected_set - {str(row["canonical_label"]) for row in train}
        if missing:
            raise SystemExit(f"Training coverage is incomplete: {sorted(missing)}")
    if stage in {"stage2", "stage3", "stage4_eval"}:
        if not test:
            raise SystemExit(f"{stage} requires an evaluation split")
        for signer in sorted({str(row["signer"]) for row in test}):
            labels = {str(row["canonical_label"]) for row in test if row["signer"] == signer}
            missing = expected_set - labels
            if missing:
                raise SystemExit(f"{stage} test signer {signer} is missing classes: {sorted(missing)}")


def assert_seal(
    samples: list[dict[str, object]],
    protected: set[str],
    heldout: str,
    stage: str,
) -> None:
    for row in samples:
        signer = str(row["signer"])
        split = str(row["split"])
        if split in {"train", "validation"} and signer in protected:
            raise SystemExit(f"Sealed-test leak: {signer} in {split} ({row['sample_id']})")
        if heldout and split == "train" and signer == heldout:
            raise SystemExit(f"Fold leak: {heldout} in train ({row['sample_id']})")
        if heldout and split == "test" and signer != heldout:
            raise SystemExit(f"{stage} test must contain only {heldout}, found {signer}")
    if stage == "final_retrain" and any(row["split"] != "train" for row in samples):
        raise SystemExit("Final retrain manifest must be train-only")
    if stage == "stage4_eval" and any(row["split"] != "test" for row in samples):
        raise SystemExit("Stage 4 manifest must be evaluation-only")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--verified", type=Path, default=VERIFIED)
    parser.add_argument("--coverage", type=Path, default=COVERAGE)
    parser.add_argument("--agreement", type=Path, default=AGREEMENT)
    parser.add_argument("--source-inventory", type=Path, default=SOURCE_INVENTORY)
    parser.add_argument("--classes", type=Path, default=CLASSES)
    parser.add_argument(
        "--stage", choices=["stage2", "stage3", "final_retrain", "stage4_eval"], required=True
    )
    parser.add_argument("--fold", choices=["heldout_s1", "heldout_s2"], default="heldout_s1")
    parser.add_argument("--heldout-signer", choices=["S1", "S2", "S3"], default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    try:
        protocol = load_and_validate_frozen_protocol(
            args.protocol,
            verified_path=args.verified,
            coverage_path=args.coverage,
            agreement_path=args.agreement,
            source_inventory_path=args.source_inventory,
            manifest_path=args.manifest,
            classes_path=args.classes,
            require_confirmatory=args.stage == "stage4_eval",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    roles = protocol.get("assigned_roles") or {}
    testers = set(roles.get("frozen_test") or [])
    protected = set(protocol.get("test_hierarchy") or [])
    development = set(roles.get("development") or [])
    validation = set(roles.get("validation") or [])
    augmentation = set(roles.get("development_augmentation") or [])

    manifest_rows = read_csv(args.manifest)
    manifest_by_path = {row["relative_path"]: row for row in manifest_rows}
    inventory_rows = read_csv(args.source_inventory)
    source_hash_by_path = {row["relative_path"]: row["source_sha256"] for row in inventory_rows}
    if len(source_hash_by_path) != len(inventory_rows):
        raise SystemExit("Frozen source inventory contains duplicate relative paths")
    invalid_hashes = [
        relative
        for relative, digest in source_hash_by_path.items()
        if not SHA256_RE.fullmatch(digest)
    ]
    if invalid_hashes:
        raise SystemExit(f"Frozen source inventory has an invalid SHA-256: {invalid_hashes[0]}")
    isolated = isolated_samples(args.manifest)
    segments = [
        segment_sample(row, manifest_by_path)
        for row in read_segments(args.verified)
        if row.get("include_in_v2") == "true" and row.get("class")
    ]
    all_rows = isolated + segments
    for row in all_rows:
        relative = str(row["relative_path"])
        source_hash = source_hash_by_path.get(relative)
        if not source_hash:
            raise SystemExit(f"Frozen source inventory is missing {relative}")
        recorded = str(row.get("source_video_hash") or "")
        if recorded and recorded != source_hash:
            raise SystemExit(f"Verified segment/source inventory hash mismatch: {relative}")
        row["source_video_hash"] = source_hash

    train_ok: set[str] = set()
    test_ok: set[str] = set()
    heldout = ""
    fold_name = ""
    if args.stage == "stage2":
        heldout = "S1" if args.fold == "heldout_s1" else "S2"
        fold_name = args.fold
        train_ok = (development | augmentation) - {heldout} - protected - validation
        test_ok = {heldout}
    elif args.stage == "stage3":
        heldout = args.heldout_signer or ""
        if not heldout:
            raise SystemExit("Stage 3 requires --heldout-signer S1, S2, or S3")
        if validation != {"S3"}:
            raise SystemExit("Stage 3 is allowed only when complete S3 is the frozen validation signer")
        fold_name = f"heldout_{heldout.lower()}"
        train_ok = (development | validation | augmentation) - {heldout} - protected
        test_ok = {heldout}
    elif args.stage == "final_retrain":
        fold_name = "final_retrain"
        train_ok = (development | validation | augmentation) - protected
    else:
        fold_name = "stage4"
        test_ok = testers

    samples: list[dict[str, object]] = []
    for original in all_rows:
        signer = str(original["signer"])
        if signer in test_ok:
            if args.stage == "stage2" and original["source_kind"] != "isolated":
                continue
            row = dict(original)
            row["split"] = "test"
        elif signer in train_ok:
            row = dict(original)
            row["split"] = "train"
        else:
            continue
        row["fold"] = fold_name
        row["v2_stage"] = args.stage
        row["v2_protocol_sha256"] = protocol["protocol_sha256"]
        row["v2_segments_sha256"] = protocol["segments_sha256"]
        row["v2_coverage_sha256"] = protocol["coverage_sha256"]
        row["v2_agreement_sha256"] = protocol["agreement_sha256"]
        row["v2_source_inventory_sha256"] = protocol["source_inventory_sha256"]
        row["v2_source_manifest_sha256"] = protocol["manifest_sha256"]
        row["v2_classes_sha256"] = protocol["classes_sha256"]
        samples.append(row)

    samples.sort(
        key=lambda row: (
            0 if row["split"] == "train" else 1,
            int(str(row["signer"])[1:]),
            str(row["canonical_label"]),
            str(row["sample_id"]),
        )
    )
    assert_seal(samples, protected, heldout, args.stage)
    labels = json.loads(args.classes.read_text(encoding="utf-8"))
    expected = labels["consonants"] + labels["vowels"]
    assert_class_coverage(samples, expected, args.stage)

    if not samples:
        raise SystemExit("No samples were generated")
    fieldnames: list[str] = []
    for row in samples:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    write_csv(args.output, fieldnames, samples)
    summary = summarize(samples, expected)
    summary.update(
        {
            "stage": args.stage,
            "fold": fold_name,
            "manifest_sha256": sha256_file(args.output),
            "protocol_sha256": protocol["protocol_sha256"],
            "testers": sorted(testers, key=lambda value: int(value[1:])),
            "split_counts": dict(Counter(str(row["split"]) for row in samples)),
        }
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
