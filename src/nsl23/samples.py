"""Combine raw isolated clips and reviewed aggregate segments into samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def sample_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def condition_group(background: str, realworld: bool) -> str:
    if background == "prepared":
        return "Prepared"
    if realworld:
        return "RealWorld Unprepared"
    if background == "unprepared":
        return "Indoor Unprepared"
    return "Unknown"


def build_samples(manifest_rows: list[dict[str, str]], segment_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for row in manifest_rows:
        if not truthy(row["usable_as_isolated_clip"]):
            continue
        identifier = sample_id("isolated", row["relative_path"])
        realworld = truthy(row["realworld_flag"])
        samples.append(
            {
                "sample_id": identifier,
                "source_kind": "isolated",
                "source_path": row["source_path"],
                "relative_path": row["relative_path"],
                "source_zip": row["source_zip"],
                "signer": row["signer"],
                "sign_type": row["sign_type"],
                "canonical_label": row["canonical_label"],
                "start_seconds": 0.0,
                "end_seconds": row["duration_seconds"],
                "background_condition": row["background_condition"],
                "scene_type": row["scene_type"],
                "realworld_flag": realworld,
                "lighting": row["lighting"],
                "indoor_outdoor": row["indoor_outdoor"],
                "condition_group": condition_group(row["background_condition"], realworld),
                "annotation_status": "source_isolated_clip",
            }
        )

    for row in segment_rows:
        identifier = sample_id(
            "segment",
            row["relative_path"],
            row["canonical_label"],
            row["start_seconds"],
            row["end_seconds"],
        )
        realworld = truthy(row["realworld_flag"])
        samples.append(
            {
                "sample_id": identifier,
                "source_kind": "verified_segment",
                "source_path": row["source_path"],
                "relative_path": row["relative_path"],
                "source_zip": row["source_zip"],
                "signer": row["signer"],
                "sign_type": row["sign_type"],
                "canonical_label": row["canonical_label"],
                "start_seconds": row["start_seconds"],
                "end_seconds": row["end_seconds"],
                "background_condition": row["background_condition"],
                "scene_type": row["scene_type"],
                "realworld_flag": realworld,
                "lighting": row["lighting"],
                "indoor_outdoor": row["indoor_outdoor"],
                "condition_group": condition_group(row["background_condition"], realworld),
                "annotation_status": row["review_status"],
            }
        )

    identifiers = [str(row["sample_id"]) for row in samples]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Duplicate sample IDs were generated")
    return sorted(samples, key=lambda row: (int(str(row["signer"])[1:]), str(row["canonical_label"]), str(row["sample_id"])))


def apply_split(samples: list[dict[str, object]], split_config: dict[str, object]) -> None:
    assignments: dict[str, str] = {}
    for split_key, output_name in (("train", "train"), ("validation", "validation"), ("test", "test")):
        for signer in split_config[split_key]:
            if signer in assignments:
                raise ValueError(f"Signer {signer} appears in more than one split")
            assignments[str(signer)] = output_name
    for row in samples:
        signer = str(row["signer"])
        if signer not in assignments:
            raise ValueError(f"No split assignment for signer {signer}")
        row["split"] = assignments[signer]


def summarize(samples: list[dict[str, object]], expected_labels: list[str]) -> dict[str, object]:
    summary: dict[str, object] = {"samples": len(samples)}
    per_split: dict[str, dict[str, object]] = {}
    for split in ("train", "validation", "test"):
        rows = [row for row in samples if row["split"] == split]
        label_counts = Counter(str(row["canonical_label"]) for row in rows)
        condition_counts = Counter(str(row["condition_group"]) for row in rows)
        per_split[split] = {
            "samples": len(rows),
            "signers": sorted({str(row["signer"]) for row in rows}, key=lambda value: int(value[1:])),
            "classes": len(label_counts),
            "missing_classes": [label for label in expected_labels if not label_counts[label]],
            "per_condition": dict(condition_counts),
            "per_sign_type": dict(Counter(str(row["sign_type"]) for row in rows)),
        }
    summary["per_split"] = per_split
    summary["per_signer"] = {
        signer: len([row for row in samples if row["signer"] == signer])
        for signer in sorted({str(row["signer"]) for row in samples}, key=lambda value: int(value[1:]))
    }
    summary["per_label"] = dict(sorted(Counter(str(row["canonical_label"]) for row in samples).items()))
    summary["per_source_kind"] = dict(Counter(str(row["source_kind"]) for row in samples))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, default=Path("configs/signer_split.json"))
    parser.add_argument("--classes", type=Path, default=Path("configs/nsl23_classes.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    if args.split_config.name == "signer_split.json":
        print(
            "WARNING: configs/signer_split.json is obsolete for Version 2 "
            "(S5 and S14 are in train). Use scripts/build_v2_samples.py after freeze.",
            file=sys.stderr,
        )

    segment_rows = read_csv(args.segments)
    if segment_rows and "include_in_v2" in segment_rows[0]:
        raise SystemExit("V2 segments.csv cannot be used with samples.py. Use scripts/build_v2_samples.py.")

    config = json.loads(args.classes.read_text(encoding="utf-8"))
    labels = config["consonants"] + config["vowels"]
    samples = build_samples(read_csv(args.manifest), segment_rows)
    apply_split(samples, json.loads(args.split_config.read_text(encoding="utf-8")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)

    summary = summarize(samples, labels)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    missing = {split: values["missing_classes"] for split, values in summary["per_split"].items()}
    print(json.dumps({"samples": summary["samples"], "missing_classes": missing}, indent=2))
    if any(missing.values()):
        raise SystemExit("Class coverage is incomplete; do not train this split")


if __name__ == "__main__":
    main()
