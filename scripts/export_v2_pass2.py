#!/usr/bin/env python3
"""Export a pass-2 CSV that contains no pass-1 decisions, classes, or bounds."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.v2_annotation import REVIEW_DECISIONS, SEGMENT_FIELDS, read_segments, sha256_file, write_csv
from nsl23.v2_exchange import (
    pass1_projection_sha256,
    proposal_projection_sha256,
    sealed_pass2_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def validate_pass1(rows: list[dict[str, str]], labels: set[str]) -> None:
    failures: list[str] = []
    for row in rows:
        sid = row["segment_id"]
        decision = row.get("review_pass_1") or ""
        if decision not in REVIEW_DECISIONS:
            failures.append(f"{sid}: pass 1 has no completed decision")
            continue
        if not (row.get("pass1_reviewer") or "").strip():
            failures.append(f"{sid}: pass 1 has no named reviewer")
        if not (row.get("pass1_timestamp") or "").strip():
            failures.append(f"{sid}: pass 1 has no timestamp")
        if decision == "accepted":
            if (row.get("pass1_class") or "") not in labels:
                failures.append(f"{sid}: pass-1 class is outside the 49-class alphabet")
            try:
                if float(row["pass1_end_time_s"]) <= float(row["pass1_start_time_s"]):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                failures.append(f"{sid}: pass 1 has invalid accepted bounds")
    if failures:
        raise SystemExit(
            "Pass 2 cannot be exported until pass 1 is complete:\n" + "\n".join(failures[:50])
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, default=ROOT / "annotations/v2/segments.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/reviewer_packages/pass2",
    )
    args = parser.parse_args()
    rows = read_segments(args.segments)
    classes_path = ROOT / "configs/nsl23_classes.json"
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    labels = set(classes["consonants"] + classes["vowels"])
    validate_pass1(rows, labels)

    sealed = sealed_pass2_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "pass2_segments.csv"
    metadata_path = args.output_dir / "pass2_package.json"
    write_csv(csv_path, SEGMENT_FIELDS, sealed)
    protocol = json.loads((ROOT / "configs/v2_protocol.json").read_text(encoding="utf-8"))
    metadata = {
        "schema_version": 1,
        "purpose": "Independent NSL23 V2 pass-2 review package; pass-1 outputs are absent.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(sealed),
        "segment_ids": sorted(row["segment_id"] for row in sealed),
        "proposal_projection_sha256": proposal_projection_sha256(sealed),
        "pass1_projection_sha256": pass1_projection_sha256(rows),
        "classes_sha256": sha256_file(classes_path),
        "hierarchy_sha256": protocol["hierarchy_sha256"],
        "csv_sha256_at_export": sha256_file(csv_path),
        "instructions": "Run review_aggregates.py in pass2 mode against pass2_segments.csv. Return the modified CSV and this JSON file. Do not edit the JSON.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "csv": str(csv_path), "metadata": str(metadata_path), "rows": len(sealed)}, indent=2))


if __name__ == "__main__":
    main()
