"""Shared Version 2 aggregate-segment schema and coverage helpers."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

SEGMENT_FIELDS = [
    "segment_id",
    "source_video",
    "source_video_hash",
    "signer",
    "class",
    "start_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
    "fps",
    "condition",
    "lighting",
    "capture_device",
    "sign_type",
    "proposal_method",
    "hold_index",
    "n_holds_detected",
    "expected_class_hint",
    "review_pass_1",
    "review_pass_2",
    "include_in_v2",
    "exclusion_reason",
    "notes",
    "source_path",
]

TAPE_OUTCOME_FIELDS = [
    "source_video",
    "signer",
    "sign_type",
    "expected_classes",
    "n_holds_detected",
    "verified_classes",
    "status",
    "notes",
    "source_path",
    "source_video_hash",
]

PREFERRED_TEST = ("S5", "S14")
TEST_FALLBACK = ("S4", "S6")
TEST_HIERARCHY = PREFERRED_TEST + TEST_FALLBACK
PREFERRED_VAL = "S3"
LOCKED_DEVELOPMENT = ("S1", "S2")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def segment_id(source_video: str, hold_index: int) -> str:
    key = hashlib.sha256(source_video.encode("utf-8")).hexdigest()[:10]
    return f"seg_{key}_{int(hold_index):03d}"


def migrate_row(row: dict[str, str], hashes: dict[str, str]) -> dict[str, str]:
    source_video = row.get("source_video") or row.get("video_id") or ""
    source_path = row.get("source_path") or ""
    hold_index = int(row.get("hold_index") or 0)
    digest = row.get("source_video_hash") or hashes.get(source_video, "")
    if not digest and source_path:
        path = Path(source_path)
        if path.exists():
            digest = sha256_file(path)
            hashes[source_video] = digest
    status = row.get("verification_status", "")
    pass1 = row.get("review_pass_1", "")
    pass2 = row.get("review_pass_2", "")
    if not pass1 and status in {"pass1", "pass2"}:
        pass1 = "accepted"
    if not pass2 and status == "pass2":
        pass2 = "accepted"
    include = row.get("include_in_v2", "false").lower() in {"1", "true", "yes"}
    return {
        "segment_id": row.get("segment_id") or segment_id(source_video, hold_index),
        "source_video": source_video,
        "source_video_hash": digest,
        "signer": row.get("signer", ""),
        "class": row.get("class", ""),
        "start_frame": row.get("start_frame", ""),
        "end_frame": row.get("end_frame", ""),
        "start_time_s": row.get("start_time_s") or row.get("start_time") or "",
        "end_time_s": row.get("end_time_s") or row.get("end_time") or "",
        "fps": row.get("fps", ""),
        "condition": row.get("condition", ""),
        "lighting": row.get("lighting", ""),
        "capture_device": row.get("capture_device", ""),
        "sign_type": row.get("sign_type", ""),
        "proposal_method": row.get("proposal_method", "mediapipe_video_mode_hold_detection"),
        "hold_index": str(hold_index),
        "n_holds_detected": row.get("n_holds_detected", ""),
        "expected_class_hint": row.get("expected_class_hint", ""),
        "review_pass_1": pass1,
        "review_pass_2": pass2,
        "include_in_v2": "true" if include and pass2 == "accepted" else "false",
        "exclusion_reason": row.get("exclusion_reason", ""),
        "notes": row.get("notes", ""),
        "source_path": source_path,
    }


def neutralize_unreviewed(row: dict[str, str]) -> dict[str, str]:
    if row.get("review_pass_1") or row.get("review_pass_2"):
        return row
    if row.get("class") and not row.get("expected_class_hint"):
        row["expected_class_hint"] = row["class"]
    row["class"] = ""
    row["include_in_v2"] = "false"
    return row


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_segments(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    hashes: dict[str, str] = {}
    return [neutralize_unreviewed(migrate_row(row, hashes)) for row in raw]


def verified_classes(rows: list[dict[str, str]], signer: str) -> tuple[set[str], set[str]]:
    cons: set[str] = set()
    vows: set[str] = set()
    for row in rows:
        if row["signer"] != signer:
            continue
        if row.get("include_in_v2") != "true" or row.get("review_pass_2") != "accepted":
            continue
        label = row.get("class") or ""
        if not label:
            continue
        if row.get("sign_type") == "vowel":
            vows.add(label)
        else:
            cons.add(label)
    return cons, vows


def assign_testers(complete: list[str]) -> list[str]:
    """Completeness-only testers: S5/S14 if present, else fill with S4 then S6."""
    have = set(complete)
    testers: list[str] = []
    for signer in PREFERRED_TEST:
        if signer in have:
            testers.append(signer)
    for signer in TEST_FALLBACK:
        if len(testers) >= 2:
            break
        if signer in have and signer not in testers:
            testers.append(signer)
    return testers


def resolve_roles(complete: list[str], frozen: bool = False) -> dict[str, str]:
    """Assign roles from completeness. S1/S2 never become testers.

    Until freeze, S4 and S6 stay held_as_fallback unless they were selected as
    testers. After freeze, unused complete fallbacks are released to development.
    """
    testers = assign_testers(complete)
    tester_set = set(testers)
    val = ""
    if PREFERRED_VAL in complete and PREFERRED_VAL not in tester_set:
        val = PREFERRED_VAL
    roles: dict[str, str] = {}
    for signer in complete:
        if signer in tester_set:
            roles[signer] = "frozen_test"
        elif signer in LOCKED_DEVELOPMENT:
            roles[signer] = "development"
        elif signer == val:
            roles[signer] = "validation"
        elif signer in TEST_FALLBACK:
            roles[signer] = "development" if frozen else "held_as_fallback"
        else:
            roles[signer] = "development"
    return roles
