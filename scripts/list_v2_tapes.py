#!/usr/bin/env python3
"""Write the Version 2 original-aggregate tape inventory (excludes cropped copies)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def capture_device(relative_path: str) -> str:
    name = relative_path.lower()
    if "phone" in name or "pnone" in name:
        return "phone"
    if "camera" in name:
        return "camera"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/nsl23_manifest_all.csv"))
    parser.add_argument("--output", type=Path, default=Path("annotations/v2/tapes.csv"))
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if truthy(row["is_aggregate"])]

    original = [row for row in rows if not truthy(row["is_cropped"])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "signer",
        "sign_type",
        "duration_seconds",
        "fps",
        "expected_gesture_count",
        "condition",
        "lighting",
        "capture_device",
        "realworld_flag",
        "source_path",
        "v2_role_if_complete",
    ]
    role = {
        "S5": "preferred_test",
        "S14": "preferred_test",
        "S3": "preferred_validation",
        "S1": "development",
        "S2": "development",
        "S4": "held_as_fallback",
        "S6": "held_as_fallback",
        "S9": "development_consonant_only",
        "S13": "development_consonant_only",
    }
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(original, key=lambda item: (int(item["signer"][1:]), item["sign_type"], item["relative_path"])):
            expected = 13 if row["sign_type"] == "vowel" else 36
            writer.writerow(
                {
                    "video_id": row["relative_path"],
                    "signer": row["signer"],
                    "sign_type": row["sign_type"],
                    "duration_seconds": row["duration_seconds"],
                    "fps": row["fps"],
                    "expected_gesture_count": expected,
                    "condition": row["background_condition"],
                    "lighting": row["lighting"],
                    "capture_device": capture_device(row["relative_path"]),
                    "realworld_flag": row["realworld_flag"],
                    "source_path": row["source_path"],
                    "v2_role_if_complete": role.get(row["signer"], "unused"),
                }
            )
    print(f"Wrote {len(original)} original aggregate tapes ({len(rows) - len(original)} cropped copies excluded) to {args.output}")


if __name__ == "__main__":
    main()
