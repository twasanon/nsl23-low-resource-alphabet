#!/usr/bin/env python3
"""Signer/class coverage after pass-2. Stop here before Stage 2 training."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.v2_annotation import (
    TAPE_OUTCOME_FIELDS,
    TEST_FALLBACK,
    PREFERRED_TEST,
    assign_testers,
    read_segments,
    resolve_roles,
    verified_classes,
    write_csv,
)


def isolated_classes(manifest: Path) -> dict[str, tuple[set[str], set[str]]]:
    def truthy(value: str) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes"}

    out: dict[str, tuple[set[str], set[str]]] = {}
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not truthy(row.get("usable_as_isolated_clip", "")):
                continue
            signer = row["signer"]
            cons, vows = out.setdefault(signer, (set(), set()))
            if row["sign_type"] == "vowel":
                vows.add(row["canonical_label"])
            else:
                cons.add(row["canonical_label"])
    return out


def verified_on_tape(segments: list[dict[str, str]], source_video: str) -> set[str]:
    labels: set[str] = set()
    for row in segments:
        if row["source_video"] != source_video:
            continue
        if row.get("include_in_v2") != "true" or row.get("review_pass_2") != "accepted":
            continue
        if row.get("class"):
            labels.add(row["class"])
    return labels


def refresh_outcomes(path: Path, segments: list[dict[str, str]]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        verified = verified_on_tape(segments, row["source_video"])
        expected = int(row.get("expected_classes") or 0)
        row["verified_classes"] = str(len(verified))
        if len(verified) >= expected and expected > 0:
            row["status"] = "complete_verified"
        elif len(verified) > 0:
            row["status"] = "incomplete"
        elif row.get("status", "").startswith("count_mismatch"):
            row["status"] = "count_mismatch_pending_review"
        else:
            row["status"] = row.get("status") or "pending_review"
    write_csv(path, TAPE_OUTCOME_FIELDS, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, default=Path("annotations/v2/segments.csv"))
    parser.add_argument("--outcomes", type=Path, default=Path("annotations/v2/tape_outcomes.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/nsl23_manifest_all.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/nsl23_classes.json"))
    parser.add_argument("--output", type=Path, default=Path("annotations/v2/coverage.csv"))
    args = parser.parse_args()

    labels = json.loads(args.classes.read_text(encoding="utf-8"))
    n_cons, n_vow = len(labels["consonants"]), len(labels["vowels"])
    isolated = isolated_classes(args.manifest)
    segments = read_segments(args.segments) if args.segments.exists() else []
    outcomes = refresh_outcomes(args.outcomes, segments)

    signers = sorted({*isolated, *(row["signer"] for row in segments)}, key=lambda item: int(item[1:]))
    complete: list[str] = []
    table = []
    for signer in signers:
        cons_i, vow_i = isolated.get(signer, (set(), set()))
        cons_s, vow_s = verified_classes(segments, signer)
        cons, vows = cons_i | cons_s, vow_i | vow_s
        total = len(cons | vows)
        eligible = total == n_cons + n_vow
        if eligible:
            complete.append(signer)
        table.append(
            {
                "signer": signer,
                "consonants": str(len(cons)),
                "vowels": str(len(vows)),
                "total_classes": str(total),
                "isolated_clips_classes": str(len(cons_i | vow_i)),
                "verified_aggregate_classes": str(len(cons_s | vow_s)),
                "complete_49": "true" if eligible else "false",
                "eligible_role": "",
            }
        )
    roles = resolve_roles(complete, frozen=False)
    for row in table:
        signer = row["signer"]
        if signer in roles:
            row["eligible_role"] = roles[signer]
        elif signer in TEST_FALLBACK:
            row["eligible_role"] = "held_as_fallback"
        elif signer in PREFERRED_TEST:
            row["eligible_role"] = "sealed_test_candidate"
        elif int(row["consonants"]) > 0 and row["complete_49"] == "false":
            row["eligible_role"] = "development_consonant_only"
        else:
            row["eligible_role"] = "incomplete"
    write_csv(args.output, list(table[0].keys()) if table else ["signer"], table)
    testers = assign_testers(complete)
    print(f"Wrote {args.output}")
    print("\nSigner\tCons\tVow\tTotal\tComplete\tRole")
    for row in table:
        print(
            f"{row['signer']}\t{row['consonants']}\t{row['vowels']}\t{row['total_classes']}\t"
            f"{row['complete_49']}\t{row['eligible_role']}"
        )
    print("\nProvisional testers (completeness only, not frozen yet):", ", ".join(testers) or "(none — annotation incomplete)")
    print("S4 and S6 stay held_as_fallback until freeze. Do not train on them.")
    print("Stage 2, when it happens, evaluates S1/S2 only. Do not score S5/S14 until Stage 4.")
    if outcomes:
        print("\nTape\tSigner\tType\tExpected\tHolds\tVerified\tStatus")
        for row in outcomes:
            print(
                f"{row['source_video']}\t{row['signer']}\t{row['sign_type']}\t"
                f"{row['expected_classes']}\t{row['n_holds_detected']}\t"
                f"{row['verified_classes']}\t{row['status']}"
            )
    n_prop = len(segments)
    n_v2 = sum(row.get("include_in_v2") == "true" for row in segments)
    print(f"\nProposed holds: {n_prop}. Pass-2 include_in_v2: {n_v2}.")
    print("572 is an annotation ceiling, not recovered data.")
    print("Do not freeze or train until pass-2 annotation is finished.")


if __name__ == "__main__":
    main()
