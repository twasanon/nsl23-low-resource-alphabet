#!/usr/bin/env python3
"""Verify or selectively extract only the 21 aggregate tapes needed for V2 review."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = {
    "NSL_Consonant_Part_1": "NSL_Consonant_Part_1.zip",
    "NSL_Consonant_Part_2": "NSL_Consonant_Part_2.zip",
    "NSL_Consonant_Part_3": "NSL_Consonant_Part_3.zip",
    "NSL_Vowel": "NSL_Vowel.zip",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapes", type=Path, default=ROOT / "annotations/v2/tapes.csv")
    parser.add_argument("--archive-dir", type=Path, default=ROOT / "data/raw_archives")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/extracted")
    parser.add_argument("--extract", action="store_true", help="Extract after validating all entries")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with args.tapes.open(newline="", encoding="utf-8") as handle:
        tapes = list(csv.DictReader(handle))
    if len(tapes) != 21:
        raise SystemExit(f"Expected the locked 21-tape review set, found {len(tapes)}")
    groups: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for tape in tapes:
        source = Path(tape["source_path"])
        try:
            member = source.relative_to("data/extracted").as_posix()
        except ValueError as exc:
            raise SystemExit(f"Unexpected tape path outside data/extracted: {source}") from exc
        top = Path(member).parts[0]
        if top not in ARCHIVES:
            raise SystemExit(f"No archive mapping for {member}")
        groups[ARCHIVES[top]].append((member, args.output_root / member))

    total_bytes = 0
    plans: list[tuple[Path, str, Path, int]] = []
    for archive_name, items in sorted(groups.items()):
        archive = args.archive_dir / archive_name
        if not archive.is_file():
            raise SystemExit(f"Missing archive: {archive}")
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
            for member, destination in items:
                if member not in names:
                    raise SystemExit(f"{archive_name} does not contain locked tape {member}")
                size = zf.getinfo(member).file_size
                total_bytes += size
                plans.append((archive, member, destination, size))

    if args.extract:
        for archive, member, destination, _ in plans:
            if destination.exists() and not args.force:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zf, zf.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
    print(
        json.dumps(
            {
                "status": "pass",
                "mode": "extract" if args.extract else "verify_only",
                "tapes": len(plans),
                "uncompressed_bytes": total_bytes,
                "uncompressed_gib": round(total_bytes / 1024**3, 3),
                "output_root": str(args.output_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
