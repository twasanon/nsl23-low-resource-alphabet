#!/usr/bin/env python3
"""Export a complete audit trail for aggregate recordings excluded from primary modeling."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        aggregate = [row for row in csv.DictReader(handle) if truthy(row["is_aggregate"])]
    rows = []
    for row in aggregate:
        reasons = ["release_has_no_per_gesture_timestamps", "equal_duration_segmentation_failed_visual_audit"]
        if truthy(row["is_cropped"]):
            reasons.append("cropped_or_preprocessed_duplicate")
        if row["relative_path"].endswith("S6_NSL_Vowel_Prepared_all_Phone.mov"):
            reasons.append("visual_audit_suggests_incomplete_or_shifted_sequence")
        rows.append(
            {
                "relative_path": row["relative_path"],
                "source_zip": row["source_zip"],
                "signer": row["signer"],
                "sign_type": row["sign_type"],
                "duration_seconds": row["duration_seconds"],
                "fps": row["fps"],
                "is_cropped": row["is_cropped"],
                "decode_ok": row["decode_ok"],
                "primary_inclusion": False,
                "exclusion_reasons": ";".join(reasons),
            }
        )
    if len(rows) != 23:
        raise RuntimeError(f"Expected 23 aggregate sources, found {len(rows)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "aggregate_sources": len(rows),
        "decoded": sum(truthy(row["decode_ok"]) for row in rows),
        "cropped_or_preprocessed": sum(truthy(row["is_cropped"]) for row in rows),
        "primary_included": 0,
        "per_sign_type": dict(Counter(row["sign_type"] for row in rows)),
        "policy": "Aggregate sources are inventoried but excluded from primary modeling because the release has no gesture timestamps and equal-duration candidate bins did not reliably retain class identity under visual review.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
