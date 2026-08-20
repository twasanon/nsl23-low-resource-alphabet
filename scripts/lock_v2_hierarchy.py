#!/usr/bin/env python3
"""Refresh the pre-annotation hierarchy hash; impossible after any review exists."""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.v2_annotation import hierarchy_sha256


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/v2_protocol.json"
SEGMENTS = ROOT / "annotations/v2/segments.csv"


def main() -> None:
    with SEGMENTS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    reviewed = [
        row
        for row in rows
        if row.get("review_pass_1")
        or row.get("review_pass_2")
        or row.get("adjudication_decision")
        or row.get("include_in_v2") == "true"
    ]
    if reviewed:
        raise SystemExit("Annotation outcomes exist. The hierarchy lock can no longer be changed.")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("frozen"):
        raise SystemExit("Protocol is already frozen")
    protocol["hierarchy_sha256"] = hierarchy_sha256(protocol)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{PROTOCOL.name}.", suffix=".tmp", dir=PROTOCOL.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(protocol, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, PROTOCOL)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(protocol["hierarchy_sha256"])


if __name__ == "__main__":
    main()
