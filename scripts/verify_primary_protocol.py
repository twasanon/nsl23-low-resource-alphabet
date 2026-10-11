#!/usr/bin/env python3
"""Fail closed if a primary NSL23 reproducibility invariant is violated."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.freeze import load_snapshot, verify_snapshot
from nsl23.paths import resolve_repo_path
from nsl23.manifest import probe_video


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/nsl23_manifest_all.csv"))
    parser.add_argument("--fold-dir", type=Path, default=Path("data/manifests/folds"))
    parser.add_argument("--classes", type=Path, default=Path("configs/nsl23_classes.json"))
    parser.add_argument("--archive-audit", type=Path, default=Path("artifacts/audit/archive_checksums.json"))
    parser.add_argument("--verify-archive-hashes", action="store_true")
    parser.add_argument("--verify-source-decode", action="store_true",
                        help="Freshly decode the first frame of every retained source video.")
    parser.add_argument(
        "--verify-v1-freeze",
        action="store_true",
        help="Also verify the 15 immutable inputs against their bound historical Git tree.",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    def rooted(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    class_config = json.loads(rooted(args.classes).read_text(encoding="utf-8"))
    labels = class_config["consonants"] + class_config["vowels"]
    assert len(labels) == 49 and len(set(labels)) == 49

    manifest = read_csv(rooted(args.manifest))
    assert len(manifest) == 630
    assert sum(truthy(row["decode_ok"]) for row in manifest) == 630
    assert sum(truthy(row["is_aggregate"]) for row in manifest) == 23
    assert sum(not truthy(row["is_aggregate"]) for row in manifest) == 607
    assert sum(truthy(row["is_cropped"]) for row in manifest) == 234
    assert sum(truthy(row["usable_as_isolated_clip"]) for row in manifest) == 375
    assert all(resolve_repo_path(row["source_path"], root).exists() for row in manifest)
    source_first_frame_decodes = None
    if args.verify_source_decode:
        for row in manifest:
            source = resolve_repo_path(row["source_path"], root)
            if not probe_video(source)["decode_ok"]:
                raise AssertionError(f"Source first-frame decode failed: {source}")
        source_first_frame_decodes = len(manifest)

    cache_paths: set[str] = set()
    fold_report = {}
    for fold_name, heldout in (("heldout_s1", "S1"), ("heldout_s2", "S2")):
        path = rooted(args.fold_dir) / f"{fold_name}_cached.csv"
        rows = read_csv(path)
        train = [row for row in rows if row["split"] == "train"]
        test = [row for row in rows if row["split"] == "test"]
        assert len(rows) == 375 and len(train) == 277 and len(test) == 98
        assert {row["signer"] for row in test} == {heldout}
        assert not ({row["signer"] for row in train} & {row["signer"] for row in test})
        assert {row["canonical_label"] for row in train} == set(labels)
        assert {row["canonical_label"] for row in test} == set(labels)
        assert sum(row["lighting"] == "bright" for row in test) == 49
        assert sum(row["lighting"] == "dark" for row in test) == 49
        for row in rows:
            cache = resolve_repo_path(row["frame_cache_path"], root)
            assert cache.exists()
            array = np.load(cache, mmap_mode="r", allow_pickle=False)
            assert array.shape == (16, 224, 224, 3) and array.dtype == np.uint8
            cache_paths.add(str(cache.resolve()))
        fold_report[fold_name] = {"train": len(train), "test": len(test), "test_signer": heldout}
    assert len(cache_paths) == 375

    archive_report = json.loads(rooted(args.archive_audit).read_text(encoding="utf-8"))
    if args.verify_archive_hashes:
        for item in archive_report["files"]:
            path = root / "Details Dataset.docx" if item["name"].endswith(".docx") else root / "data/raw_archives" / item["name"]
            assert path.stat().st_size == item["bytes"]
            assert md5(path) == item["md5"]

    freeze_report = None
    if args.verify_v1_freeze:
        snapshot = load_snapshot(root / "artifacts/audit/protocol_freeze.json")
        source = json.loads(
            (root / "artifacts/audit/protocol_freeze_source.json").read_text(encoding="utf-8")
        )
        if source.get("snapshot_sha256") != snapshot["snapshot_sha256"]:
            raise AssertionError("V1 freeze source map does not match the snapshot")
        freeze_report = verify_snapshot(
            root, snapshot, mode="git", revision=source["git_revision"]
        )
        assert freeze_report["status"] == "pass"

    report = {
        "status": "pass",
        "videos": 630,
        "manifest_decode_ok": 630,
        "source_first_frame_decodes": source_first_frame_decodes,
        "primary_original_isolated": 375,
        "aggregate_excluded": 23,
        "cropped_or_preprocessed": 234,
        "unique_frame_caches": len(cache_paths),
        "folds": fold_report,
        "archive_hashes_recomputed": args.verify_archive_hashes,
        "v1_freeze": freeze_report,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
