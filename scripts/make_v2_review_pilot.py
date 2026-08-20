#!/usr/bin/env python3
"""Create a disposable S3 annotation-calibration file; never merge it automatically."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.v2_annotation import SEGMENT_FIELDS, read_segments, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, default=Path("annotations/v2/segments.csv"))
    parser.add_argument("--output", type=Path, default=Path("annotations/v2/pilot_segments.csv"))
    parser.add_argument("--count", type=int, default=24)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Pilot already exists: {args.output}")
    rows = [row for row in read_segments(args.segments) if row.get("signer") == "S3"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_video"]].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: float(row["proposal_start_time_s"] or 0))
    if len(rows) < args.count:
        raise SystemExit(f"Only {len(rows)} S3 proposals are available")
    tapes = sorted(grouped)
    base, remainder = divmod(args.count, len(tapes))
    pilot: list[dict[str, str]] = []
    for tape_index, tape in enumerate(tapes):
        group = grouped[tape]
        take = base + (1 if tape_index < remainder else 0)
        if take > len(group):
            raise SystemExit(f"Pilot quota {take} exceeds proposals on {tape}")
        indices = [min(len(group) - 1, int((index + 0.5) * len(group) / take)) for index in range(take)]
        pilot.extend(group[index] for index in indices)
    pilot.sort(key=lambda row: (row["source_video"], float(row["proposal_start_time_s"] or 0)))
    write_csv(args.output, SEGMENT_FIELDS, pilot)
    print(
        f"Wrote {len(pilot)} disposable S3 proposals across {len(tapes)} tapes to {args.output}. "
        "Do not merge pilot decisions into annotations/v2/segments.csv."
    )


if __name__ == "__main__":
    main()
