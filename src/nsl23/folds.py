"""Build the primary isolated-only, signer-held-out NSL23 fold manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def make_sample_id(relative_path: str) -> str:
    return hashlib.sha256(f"isolated|{relative_path}".encode("utf-8")).hexdigest()[:20]


def condition_group(row: dict[str, str]) -> str:
    if row["lighting"] == "dark":
        return "Indoor Unprepared - Dark"
    if row["lighting"] == "bright":
        return "Indoor Unprepared - Bright"
    if row["background_condition"] == "prepared":
        return "Indoor Prepared"
    if truthy(row["realworld_flag"]):
        return "Real-world Unprepared"
    if row["background_condition"] == "unprepared":
        return "Indoor Unprepared"
    return "Unspecified"


def primary_samples(manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in manifest_rows:
        if not truthy(source["usable_as_isolated_clip"]):
            continue
        rows.append(
            {
                "sample_id": make_sample_id(source["relative_path"]),
                "source_kind": "original_isolated",
                "source_path": source["source_path"],
                "relative_path": source["relative_path"],
                "source_zip": source["source_zip"],
                "signer": source["signer"],
                "sign_type": source["sign_type"],
                "canonical_label": source["canonical_label"],
                "start_seconds": 0.0,
                "end_seconds": source["duration_seconds"],
                "background_condition": source["background_condition"],
                "scene_type": source["scene_type"],
                "realworld_flag": source["realworld_flag"],
                "lighting": source["lighting"],
                "indoor_outdoor": source["indoor_outdoor"],
                "condition_group": condition_group(source),
                "annotation_status": "filename_label_original_isolated_clip",
            }
        )
    identifiers = [str(row["sample_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Duplicate isolated sample identifiers")
    return sorted(rows, key=lambda row: (int(str(row["signer"])[1:]), str(row["canonical_label"]), str(row["relative_path"])))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/isolated_twofold_protocol.json"))
    parser.add_argument("--classes", type=Path, default=Path("configs/nsl23_classes.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    class_config = json.loads(args.classes.read_text(encoding="utf-8"))
    labels = class_config["consonants"] + class_config["vowels"]
    samples = primary_samples(read_csv(args.manifest))
    available_signers = sorted({str(row["signer"]) for row in samples}, key=lambda value: int(value[1:]))

    summary: dict[str, Any] = {
        "protocol_name": protocol["protocol_name"],
        "primary_samples": len(samples),
        "available_signers": available_signers,
        "per_signer": dict(Counter(str(row["signer"]) for row in samples)),
        "per_sign_type": dict(Counter(str(row["sign_type"]) for row in samples)),
        "folds": {},
    }
    for fold in protocol["folds"]:
        fold_name = str(fold["name"])
        test_signer = str(fold["test_signer"])
        if test_signer not in available_signers:
            raise ValueError(f"Test signer is absent: {test_signer}")
        fold_rows: list[dict[str, Any]] = []
        for sample in samples:
            row = dict(sample)
            row["fold"] = fold_name
            row["split"] = "test" if row["signer"] == test_signer else "train"
            fold_rows.append(row)

        test_rows = [row for row in fold_rows if row["split"] == "test"]
        train_rows = [row for row in fold_rows if row["split"] == "train"]
        test_labels = {str(row["canonical_label"]) for row in test_rows}
        train_labels = {str(row["canonical_label"]) for row in train_rows}
        missing_test = [label for label in labels if label not in test_labels]
        missing_train = [label for label in labels if label not in train_labels]
        if missing_test or missing_train:
            raise RuntimeError({"fold": fold_name, "missing_train": missing_train, "missing_test": missing_test})

        write_csv(args.output_dir / f"{fold_name}.csv", fold_rows)
        summary["folds"][fold_name] = {
            "test_signer": test_signer,
            "train_samples": len(train_rows),
            "test_samples": len(test_rows),
            "train_signers": sorted({str(row["signer"]) for row in train_rows}, key=lambda value: int(value[1:])),
            "test_classes": len(test_labels),
            "train_classes": len(train_labels),
            "test_per_condition": dict(Counter(str(row["condition_group"]) for row in test_rows)),
        }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
