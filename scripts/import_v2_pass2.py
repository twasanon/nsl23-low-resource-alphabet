#!/usr/bin/env python3
"""Validate and merge a sealed outside review, copying only pass-2 fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.v2_annotation import SEGMENT_FIELDS, load_class_sets, read_segments, sha256_file, write_csv
from nsl23.v2_exchange import (
    merge_pass2_rows,
    pass1_projection_sha256,
    proposal_projection_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--returned", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--reviewer", required=True, help="Expected pass-2 reviewer name")
    parser.add_argument("--segments", type=Path, default=ROOT / "annotations/v2/segments.csv")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    metadata = json.loads(args.package.read_text(encoding="utf-8"))
    master = read_segments(args.segments)
    returned = read_segments(args.returned)
    expected_ids = set(metadata["segment_ids"])
    current = [row for row in master if row["segment_id"] in expected_ids]
    received_existing = [row for row in returned if row["segment_id"] in expected_ids]
    if {row["segment_id"] for row in current} != expected_ids:
        raise SystemExit("Master proposal set changed since export")
    if {row["segment_id"] for row in received_existing} != expected_ids:
        raise SystemExit("Returned CSV is missing or duplicates an exported proposal")
    if proposal_projection_sha256(current) != metadata["proposal_projection_sha256"]:
        raise SystemExit("Master proposals changed since export")
    if proposal_projection_sha256(received_existing) != metadata["proposal_projection_sha256"]:
        raise SystemExit("Returned proposal identity or original bounds were modified")
    if pass1_projection_sha256(current) != metadata["pass1_projection_sha256"]:
        raise SystemExit("Pass-1 records changed after the pass-2 package was exported")
    classes_path = ROOT / "configs/nsl23_classes.json"
    if sha256_file(classes_path) != metadata["classes_sha256"]:
        raise SystemExit("Class configuration changed after export")
    protocol = json.loads((ROOT / "configs/v2_protocol.json").read_text(encoding="utf-8"))
    if protocol["hierarchy_sha256"] != metadata["hierarchy_sha256"]:
        raise SystemExit("V2 hierarchy lock changed after export")

    consonants, vowels = load_class_sets(classes_path)
    try:
        merged, report = merge_pass2_rows(
            master,
            returned,
            consonants=consonants,
            vowels=vowels,
            reviewer=args.reviewer,
            require_complete=not args.allow_partial,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    write_csv(args.segments, SEGMENT_FIELDS, merged, backup=True)
    report.update({"status": "pass", "master": str(args.segments)})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
