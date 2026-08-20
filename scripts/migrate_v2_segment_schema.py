#!/usr/bin/env python3
"""Rewrite annotations/v2/segments.csv to the locked schema.

Automatic class guesses from 'longest N holds' are moved to expected_class_hint
and cleared from class. include_in_v2 stays false.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.v2_annotation import SEGMENT_FIELDS, migrate_row, neutralize_unreviewed, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("annotations/v2/segments.csv"))
    parser.add_argument("--output", type=Path, default=Path("annotations/v2/segments.csv"))
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    hashes: dict[str, str] = {}
    rows = []
    for row in raw:
        migrated = neutralize_unreviewed(migrate_row(row, hashes))
        rows.append(migrated)
    write_csv(args.output, SEGMENT_FIELDS, rows)
    print(f"Wrote {len(rows)} rows with locked schema to {args.output}")


if __name__ == "__main__":
    main()
