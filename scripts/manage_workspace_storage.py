#!/usr/bin/env python3
"""Report large local assets and remove only checksum-verified duplicate archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remove-verified-archives",
        action="store_true",
        help="Delete the four official ZIPs only after hashes and all 630 extracted paths verify.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=ROOT / "artifacts/audit/storage_cleanup.json",
    )
    args = parser.parse_args()
    audit_path = ROOT / "artifacts/audit/archive_checksums.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    archive_records = [item for item in audit["files"] if item["name"].endswith(".zip")]
    archive_dir = ROOT / "data/raw_archives"
    manifest_path = ROOT / "data/manifests/nsl23_manifest_all.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    extracted = []
    for row in manifest:
        path = Path(row["source_path"])
        path = path if path.is_absolute() else ROOT / path
        extracted.append(path)
    extracted_ok = len(extracted) == 630 and len(set(extracted)) == 630 and all(
        path.is_file() for path in extracted
    )

    checked: list[dict[str, object]] = []
    for item in archive_records:
        path = archive_dir / item["name"]
        exists = path.is_file()
        size_ok = exists and path.stat().st_size == item["bytes"]
        hash_value = md5(path) if size_ok and args.remove_verified_archives else None
        checked.append(
            {
                "name": item["name"],
                "bytes": item["bytes"],
                "expected_md5": item["md5"],
                "exists_before": exists,
                "size_ok": size_ok,
                "observed_md5": hash_value,
                "hash_ok": hash_value == item["md5"] if hash_value else None,
            }
        )

    action = "report_only"
    removed_bytes = 0
    if args.remove_verified_archives:
        if not extracted_ok:
            raise SystemExit("Refusing cleanup: the manifest's 630 extracted source videos are not all present")
        if not all(item["exists_before"] and item["size_ok"] and item["hash_ok"] for item in checked):
            raise SystemExit("Refusing cleanup: an official archive is missing or fails its recorded checksum")
        for item in checked:
            path = archive_dir / str(item["name"])
            removed_bytes += path.stat().st_size
            path.unlink()
        action = "removed_checksum_verified_duplicate_archives"

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "source_doi": audit["source_doi"],
        "source_record": audit["source_record"],
        "license": audit["license"],
        "extracted_manifest_rows": len(extracted),
        "all_extracted_sources_present": extracted_ok,
        "archives": checked,
        "removed_bytes": removed_bytes,
        "removed_gib": round(removed_bytes / 1024**3, 3),
        "recoverability": "The four ZIPs can be downloaded again from the recorded Zenodo DOI; extracted media was retained.",
        "workspace_bytes_after": {
            "data_extracted": directory_bytes(ROOT / "data/extracted"),
            "data_raw_archives": directory_bytes(archive_dir),
            "frame_cache_primary": directory_bytes(ROOT / "data/frame_cache_primary"),
            "hand_crop_cache_primary": directory_bytes(ROOT / "data/hand_crop_cache_primary"),
        },
    }
    if args.remove_verified_archives:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
