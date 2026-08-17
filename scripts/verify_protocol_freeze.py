#!/usr/bin/env python3
"""Verify that every frozen primary input still matches its pre-run snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/audit/protocol_freeze.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    snapshot_path = args.snapshot if args.snapshot.is_absolute() else root / args.snapshot
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    mismatches = []
    for record in snapshot["inputs"]:
        path = root / record["path"]
        if not path.is_file():
            mismatches.append({"path": record["path"], "reason": "missing"})
            continue
        observed_size = path.stat().st_size
        observed_hash = sha256(path)
        if observed_size != record["bytes"] or observed_hash != record["sha256"]:
            mismatches.append(
                {
                    "path": record["path"],
                    "expected_bytes": record["bytes"],
                    "observed_bytes": observed_size,
                    "expected_sha256": record["sha256"],
                    "observed_sha256": observed_hash,
                }
            )
    report = {
        "status": "pass" if not mismatches else "fail",
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "verified_inputs": len(snapshot["inputs"]) - len(mismatches),
        "mismatches": mismatches,
    }
    print(json.dumps(report, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
