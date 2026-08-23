#!/usr/bin/env python3
"""Freeze Version 2 splits after pass-2 coverage. Do not run this before annotation.

Writes hashed verified_segments_v2.0.csv, coverage_v2.0.csv, and configs/v2_protocol.json.
After this succeeds, test signers and their annotations are immutable.
Stage 2 still evaluates S1/S2 only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.v2_annotation import (
    SEGMENT_FIELDS,
    TEST_FALLBACK,
    PREFERRED_TEST,
    agreement_projection_sha256,
    assign_testers,
    load_class_sets,
    overlaps_on_tape,
    needs_adjudication,
    read_segments,
    resolve_roles,
    sha256_file,
    validate_verified_row,
    validate_review_completion,
    verified_classes,
    write_csv,
)
from nsl23.v2_protocol import protocol_sha256, validate_hierarchy_lock
from nsl23.paths import resolve_repo_path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/v2_protocol.json"
SEGMENTS = ROOT / "annotations/v2/segments.csv"
VERIFIED = ROOT / "annotations/v2/verified_segments_v2.0.csv"
COVERAGE = ROOT / "annotations/v2/coverage_v2.0.csv"
AGREEMENT = ROOT / "annotations/v2/agreement_pre_adjudication.json"
VERSION2 = ROOT / "VERSION2.md"
MANIFEST = ROOT / "data/manifests/nsl23_manifest_all.csv"
CLASSES = ROOT / "configs/nsl23_classes.json"
SOURCE_INVENTORY = ROOT / "data/manifests/v2_source_inventory_v2.0.csv"
HASH_BLOCK = re.compile(
    r"<!-- V2_FREEZE_HASHES -->.*?<!-- /V2_FREEZE_HASHES -->",
    re.DOTALL,
)


def load_isolated(manifest: Path) -> dict[str, tuple[set[str], set[str]]]:
    def truthy(value: str) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes"}

    out: dict[str, tuple[set[str], set[str]]] = {}
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not truthy(row.get("usable_as_isolated_clip", "")):
                continue
            cons, vows = out.setdefault(row["signer"], (set(), set()))
            if row["sign_type"] == "vowel":
                vows.add(row["canonical_label"])
            else:
                cons.add(row["canonical_label"])
    return out


def build_source_inventory(
    manifest: Path,
    verified: list[dict[str, str]],
    source_hashes: dict[str, str],
) -> list[dict[str, str]]:
    """Hash every isolated source plus every retained aggregate source exactly once."""
    sources: dict[str, Path] = {}
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("usable_as_isolated_clip", "")).strip().lower() not in {
                "1",
                "true",
                "yes",
            }:
                continue
            relative = row["relative_path"]
            path = resolve_repo_path(row["source_path"], ROOT)
            if relative in sources and sources[relative] != path:
                raise SystemExit(f"Source inventory path collision: {relative}")
            sources[relative] = path
    for row in verified:
        relative = row["source_video"]
        path = resolve_repo_path(row["source_path"], ROOT)
        if relative in sources and sources[relative] != path:
            raise SystemExit(f"Source inventory path collision: {relative}")
        sources[relative] = path

    inventory: list[dict[str, str]] = []
    for relative, path in sorted(sources.items()):
        if not path.is_file():
            raise SystemExit(f"Missing source while building frozen inventory: {path}")
        key = str(path)
        if key not in source_hashes:
            source_hashes[key] = sha256_file(path)
        inventory.append(
            {
                "relative_path": relative,
                "source_sha256": source_hashes[key],
                "bytes": str(path.stat().st_size),
            }
        )
    return inventory


def build_coverage(
    segments: list[dict[str, str]],
    isolated: dict[str, tuple[set[str], set[str]]],
    n_cons: int,
    n_vow: int,
    frozen: bool,
) -> tuple[list[dict[str, str]], list[str]]:
    signers = sorted({*isolated, *(row["signer"] for row in segments)}, key=lambda item: int(item[1:]))
    complete: list[str] = []
    table = []
    for signer in signers:
        cons_i, vow_i = isolated.get(signer, (set(), set()))
        cons_s, vow_s = verified_classes(segments, signer)
        cons, vows = cons_i | cons_s, vow_i | vow_s
        eligible = len(cons | vows) == n_cons + n_vow
        if eligible:
            complete.append(signer)
        table.append(
            {
                "signer": signer,
                "consonants": str(len(cons)),
                "vowels": str(len(vows)),
                "total_classes": str(len(cons | vows)),
                "isolated_clips_classes": str(len(cons_i | vow_i)),
                "verified_aggregate_classes": str(len(cons_s | vow_s)),
                "complete_49": "true" if eligible else "false",
                "eligible_role": "",
            }
        )
    roles = resolve_roles(complete, frozen=frozen)
    for row in table:
        signer = row["signer"]
        if signer in roles:
            row["eligible_role"] = roles[signer]
        elif signer in (*PREFERRED_TEST, *TEST_FALLBACK):
            row["eligible_role"] = "incomplete_test_candidate" if frozen else "sealed_test_candidate"
        elif int(row["total_classes"]) > 0 and row["complete_49"] == "false":
            row["eligible_role"] = "development_augmentation"
        else:
            row["eligible_role"] = "incomplete"
    return table, complete


def update_version2_hashes(
    segments_hash: str,
    coverage_hash: str,
    agreement_hash: str,
    source_inventory_hash: str,
    manifest_hash: str,
    classes_hash: str,
    protocol_hash: str,
) -> None:
    text = VERSION2.read_text(encoding="utf-8")
    block = f"""<!-- V2_FREEZE_HASHES -->
## Freeze hashes

Frozen {date.today().isoformat()}. Test splits and verified segments are immutable.

| Artifact | SHA-256 |
|---|---|
| annotations/v2/verified_segments_v2.0.csv | `{segments_hash}` |
| annotations/v2/coverage_v2.0.csv | `{coverage_hash}` |
| annotations/v2/agreement_pre_adjudication.json | `{agreement_hash}` |
| data/manifests/v2_source_inventory_v2.0.csv | `{source_inventory_hash}` |
| data/manifests/nsl23_manifest_all.csv | `{manifest_hash}` |
| configs/nsl23_classes.json | `{classes_hash}` |
| configs/v2_protocol.json canonical payload (`protocol_sha256=null`) | `{protocol_hash}` |
<!-- /V2_FREEZE_HASHES -->"""
    if not HASH_BLOCK.search(text):
        raise SystemExit("VERSION2.md is missing the V2_FREEZE_HASHES markers")
    atomic_write_text(VERSION2, HASH_BLOCK.sub(block, text))


def atomic_write_text(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, default=SEGMENTS)
    parser.add_argument("--agreement", type=Path, default=AGREEMENT)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("frozen"):
        raise SystemExit(f"{PROTOCOL} is already frozen. Do not reassign testers.")
    try:
        validate_hierarchy_lock(protocol)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    labels = json.loads(CLASSES.read_text(encoding="utf-8"))
    consonants, vowels = load_class_sets(CLASSES)
    n_cons, n_vow = len(labels["consonants"]), len(labels["vowels"])
    segments = read_segments(args.segments)
    completion_errors = [
        f"{row.get('segment_id')}: {', '.join(errors)}"
        for row in segments
        if (errors := validate_review_completion(row))
    ]
    if completion_errors:
        raise SystemExit(
            "Every proposal must finish both review passes before freeze:\n"
            + "\n".join(completion_errors[:50])
        )
    unresolved = [
        row["segment_id"]
        for row in segments
        if needs_adjudication(row) and not row.get("adjudication_decision")
    ]
    if unresolved:
        raise SystemExit(
            "Unresolved pass disagreements remain; adjudicate before freeze:\n"
            + "\n".join(unresolved[:50])
        )
    if not args.agreement.is_file():
        raise SystemExit(
            f"Missing {args.agreement}. Save the pre-adjudication agreement report first."
        )
    agreement = json.loads(args.agreement.read_text(encoding="utf-8"))
    if agreement.get("before_adjudication") is not True:
        raise SystemExit("Agreement artifact is not marked pre-adjudication")
    if agreement.get("pass_projection_sha256") != agreement_projection_sha256(segments):
        raise SystemExit(
            "Agreement artifact does not match the completed pass records. "
            "Do not alter reviews after saving pre-adjudication agreement."
        )
    overall = agreement.get("overall") or {}
    if overall.get("proposals") != len(segments) or overall.get("both_passes_complete") != len(segments):
        raise SystemExit("Agreement artifact does not cover every proposal")
    tapes = {}
    tapes_path = ROOT / "annotations/v2/tapes.csv"
    if tapes_path.exists():
        with tapes_path.open(newline="", encoding="utf-8") as handle:
            tapes = {row["video_id"]: row for row in csv.DictReader(handle)}

    verified = []
    validation_errors: list[str] = []
    source_hashes: dict[str, str] = {}
    for row in segments:
        if row.get("include_in_v2") != "true":
            continue
        duration = float(tapes.get(row["source_video"], {}).get("duration_seconds") or row.get("duration_seconds") or 0)
        source_path = resolve_repo_path(row.get("source_path") or "", ROOT)
        source_key = str(source_path)
        if source_key not in source_hashes and source_path.is_file():
            source_hashes[source_key] = sha256_file(source_path)
        problems = validate_verified_row(
            row,
            consonants,
            vowels,
            duration,
            actual_source_hash=source_hashes.get(source_key),
        )
        if problems:
            validation_errors.append(f"{row['segment_id']}: {', '.join(problems)}")
            continue
        verified.append(row)
    validation_errors.extend(overlaps_on_tape(verified))
    if validation_errors:
        raise SystemExit("Freeze validators failed:\n" + "\n".join(validation_errors[:50]))
    if not verified:
        raise SystemExit("No rows passed both independent reviews. Finish annotation before freezing.")

    isolated = load_isolated(MANIFEST)
    table, complete = build_coverage(segments, isolated, n_cons, n_vow, frozen=True)
    testers = assign_testers(complete)
    confirmatory = len(testers) >= 2
    if not confirmatory:
        print(
            "Fewer than two complete 49-way testers. Freeze the annotation resource, "
            "but do not make a confirmatory 49-way claim. Fallback paper: V1 exploratory "
            "benchmark plus verified segments."
        )

    write_csv(VERIFIED, SEGMENT_FIELDS, verified)
    write_csv(COVERAGE, list(table[0].keys()), table)
    source_inventory = build_source_inventory(MANIFEST, verified, source_hashes)
    write_csv(
        SOURCE_INVENTORY,
        ["relative_path", "source_sha256", "bytes"],
        source_inventory,
    )

    def by_role(name: str) -> list[str]:
        return [row["signer"] for row in table if row["eligible_role"] == name]

    protocol["frozen"] = True
    protocol["frozen_date"] = date.today().isoformat()
    protocol["created_before_any_v2_training"] = True
    protocol["confirmatory_49way"] = confirmatory
    protocol["assigned_roles"] = {
        "frozen_test": testers,
        "validation": by_role("validation"),
        "development": by_role("development"),
        "held_as_fallback": [],
        "development_augmentation": by_role("development_augmentation"),
        "incomplete": by_role("incomplete") + by_role("incomplete_test_candidate"),
        "sealed_test_candidate": [],
    }
    protocol["segments_sha256"] = sha256_file(VERIFIED)
    protocol["coverage_sha256"] = sha256_file(COVERAGE)
    protocol["agreement_sha256"] = sha256_file(args.agreement)
    protocol["source_inventory_sha256"] = sha256_file(SOURCE_INVENTORY)
    protocol["manifest_sha256"] = sha256_file(MANIFEST)
    protocol["classes_sha256"] = sha256_file(CLASSES)
    protocol["protocol_sha256"] = None
    digest = protocol_sha256(protocol)
    protocol["protocol_sha256"] = digest
    atomic_write_text(PROTOCOL, json.dumps(protocol, indent=2) + "\n")
    update_version2_hashes(
        protocol["segments_sha256"],
        protocol["coverage_sha256"],
        protocol["agreement_sha256"],
        protocol["source_inventory_sha256"],
        protocol["manifest_sha256"],
        protocol["classes_sha256"],
        digest,
    )
    print(f"Froze {len(verified)} verified segments.")
    print("Frozen testers:", ", ".join(testers) or "(none)")
    print("Confirmatory 49-way:", confirmatory)
    print("Validation:", ", ".join(protocol["assigned_roles"]["validation"]) or "(none)")
    print("Development:", ", ".join(protocol["assigned_roles"]["development"]))
    print("Stage 2 still evaluates S1/S2 only. Do not score frozen testers until Stage 4.")


if __name__ == "__main__":
    main()
