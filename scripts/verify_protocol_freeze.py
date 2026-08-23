#!/usr/bin/env python3
"""Verify the frozen Version 1 inputs in their recorded Git tree or the worktree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.freeze import load_snapshot, verify_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/audit/protocol_freeze.json"))
    parser.add_argument(
        "--mode",
        choices=("git", "worktree"),
        default="git",
        help="Verify the immutable historical Git tree (default) or diagnose live-tree drift.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Git revision for --mode git. Defaults to the bound revision in --source-map.",
    )
    parser.add_argument(
        "--source-map",
        type=Path,
        default=Path("artifacts/audit/protocol_freeze_source.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    snapshot_path = args.snapshot if args.snapshot.is_absolute() else root / args.snapshot
    snapshot = load_snapshot(snapshot_path)
    revision = args.revision
    if args.mode == "git" and revision is None:
        source_path = args.source_map if args.source_map.is_absolute() else root / args.source_map
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if source.get("snapshot_sha256") != snapshot["snapshot_sha256"]:
            raise SystemExit("Freeze source map is bound to a different snapshot")
        revision = source["git_revision"]
    report = verify_snapshot(root, snapshot, mode=args.mode, revision=revision)
    print(json.dumps(report, indent=2))
    if report["mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
