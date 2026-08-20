#!/usr/bin/env python3
"""Freeze Version 2 splits after pass-2 coverage. Do not run this before annotation.

Writes hashed verified_segments_v2.0.csv, coverage_v2.0.csv, and configs/v2_protocol.json.
After this succeeds, test signers and their annotations are immutable.
Stage 2 still evaluates S1/S2 only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.v2_annotation import (
    SEGMENT_FIELDS,
    TEST_FALLBACK,
    PREFERRED_TEST,
    assign_testers,
    read_segments,
    resolve_roles,
    sha256_file,
    verified_classes,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/v2_protocol.json"
SEGMENTS = ROOT / "annotations/v2/segments.csv"
VERIFIED = ROOT / "annotations/v2/verified_segments_v2.0.csv"
COVERAGE = ROOT / "annotations/v2/coverage_v2.0.csv"
VERSION2 = ROOT / "VERSION2.md"
MANIFEST = ROOT / "data/manifests/nsl23_manifest_all.csv"
CLASSES = ROOT / "configs/nsl23_classes.json"
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
        elif signer in TEST_FALLBACK:
            row["eligible_role"] = "incomplete" if frozen else "held_as_fallback"
        elif signer in PREFERRED_TEST:
            row["eligible_role"] = "incomplete" if frozen else "sealed_test_candidate"
        elif int(row["consonants"]) > 0 and row["complete_49"] == "false":
            row["eligible_role"] = "development_consonant_only"
        else:
            row["eligible_role"] = "incomplete"
    return table, complete


def canonical_protocol(data: dict) -> str:
    payload = dict(data)
    payload["protocol_sha256"] = None
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def update_version2_hashes(segments_hash: str, coverage_hash: str, protocol_hash: str) -> None:
    text = VERSION2.read_text(encoding="utf-8")
    block = f"""<!-- V2_FREEZE_HASHES -->
## Freeze hashes

Frozen {date.today().isoformat()}. Test splits and verified segments are immutable.

| Artifact | SHA-256 |
|---|---|
| annotations/v2/verified_segments_v2.0.csv | `{segments_hash}` |
| annotations/v2/coverage_v2.0.csv | `{coverage_hash}` |
| configs/v2_protocol.json | `{protocol_hash}` |
<!-- /V2_FREEZE_HASHES -->"""
    if not HASH_BLOCK.search(text):
        raise SystemExit("VERSION2.md is missing the V2_FREEZE_HASHES markers")
    VERSION2.write_text(HASH_BLOCK.sub(block, text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, default=SEGMENTS)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("frozen"):
        raise SystemExit(f"{PROTOCOL} is already frozen. Do not reassign testers.")

    labels = json.loads(CLASSES.read_text(encoding="utf-8"))
    n_cons, n_vow = len(labels["consonants"]), len(labels["vowels"])
    segments = read_segments(args.segments)
    verified = [
        row
        for row in segments
        if row.get("include_in_v2") == "true" and row.get("review_pass_2") == "accepted" and row.get("class")
    ]
    if not verified:
        raise SystemExit("No pass-2 verified segments. Finish annotation before freezing.")

    isolated = load_isolated(MANIFEST)
    table, complete = build_coverage(segments, isolated, n_cons, n_vow, frozen=True)
    testers = assign_testers(complete)
    if len(testers) < 2:
        raise SystemExit(
            "Need two complete 49-way testers from S5/S14 then S4 then S6. "
            f"Complete signers: {complete or 'none'}. Testers so far: {testers or 'none'}."
        )

    write_csv(VERIFIED, SEGMENT_FIELDS, verified)
    write_csv(COVERAGE, list(table[0].keys()), table)

    def by_role(name: str) -> list[str]:
        return [row["signer"] for row in table if row["eligible_role"] == name]

    protocol["frozen"] = True
    protocol["frozen_date"] = date.today().isoformat()
    protocol["created_before_any_v2_training"] = True
    protocol["assigned_roles"] = {
        "frozen_test": testers,
        "validation": by_role("validation"),
        "development": by_role("development"),
        "held_as_fallback": [],
        "development_consonant_only": by_role("development_consonant_only"),
        "incomplete": by_role("incomplete"),
        "sealed_test_candidate": [],
    }
    protocol["stage2"]["never_evaluate"] = testers
    protocol["segments_sha256"] = sha256_file(VERIFIED)
    protocol["coverage_sha256"] = sha256_file(COVERAGE)
    protocol["protocol_sha256"] = None
    digest = hashlib.sha256(canonical_protocol(protocol).encode("utf-8")).hexdigest()
    protocol["protocol_sha256"] = digest
    PROTOCOL.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    update_version2_hashes(protocol["segments_sha256"], protocol["coverage_sha256"], digest)
    print(f"Froze {len(verified)} verified segments.")
    print("Frozen testers:", ", ".join(testers))
    print("Validation:", ", ".join(protocol["assigned_roles"]["validation"]) or "(none)")
    print("Development:", ", ".join(protocol["assigned_roles"]["development"]))
    print("Stage 2 still evaluates S1/S2 only. Do not score the frozen testers until Stage 4.")


if __name__ == "__main__":
    main()
