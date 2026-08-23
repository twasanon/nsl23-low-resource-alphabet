"""Shared Version 2 aggregate-segment schema, include rules, and coverage helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from .paths import resolve_repo_path

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
    "duration_seconds",
    "condition",
    "lighting",
    "capture_device",
    "sign_type",
    "proposal_method",
    "hold_index",
    "n_holds_detected",
    "expected_class_hint",
    "proposal_start_frame",
    "proposal_end_frame",
    "proposal_start_time_s",
    "proposal_end_time_s",
    "review_pass_1",
    "pass1_reviewer",
    "pass1_timestamp",
    "pass1_class",
    "pass1_start_frame",
    "pass1_end_frame",
    "pass1_start_time_s",
    "pass1_end_time_s",
    "review_pass_2",
    "pass2_reviewer",
    "pass2_timestamp",
    "pass2_class",
    "pass2_start_frame",
    "pass2_end_frame",
    "pass2_start_time_s",
    "pass2_end_time_s",
    "adjudication_reviewer",
    "adjudication_timestamp",
    "adjudication_decision",
    "adjudication_class",
    "adjudication_start_frame",
    "adjudication_end_frame",
    "adjudication_start_time_s",
    "adjudication_end_time_s",
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
BOUNDS_IOU_MIN = 0.5
REVIEW_DECISIONS = frozenset({"accepted", "rejected", "incomplete", "ambiguous"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
HIERARCHY_LOCK_KEYS = (
    "test_hierarchy",
    "test_assignment",
    "preferred_test",
    "test_fallback",
    "validation_hierarchy",
    "locked_development",
    "development_policy",
    "annotation_gate",
    "feature_extraction",
    "model_training",
    "estimand",
    "stage2",
    "stage3",
    "stage4_final_test",
    "isolated_task",
    "fallback_paper",
    "final_retrain",
)


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


def empty_segment() -> dict[str, str]:
    return {field: "" for field in SEGMENT_FIELDS}


def temporal_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    inter = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = (end_a - start_a) + (end_b - start_b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def load_class_sets(path: Path) -> tuple[set[str], set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload["consonants"]), set(payload["vowels"])


def hierarchy_payload(protocol: dict) -> dict:
    return {key: protocol[key] for key in HIERARCHY_LOCK_KEYS if key in protocol}


def hierarchy_sha256(protocol: dict) -> str:
    blob = json.dumps(hierarchy_payload(protocol), indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def agreement_projection_sha256(rows: list[dict[str, str]]) -> str:
    """Hash proposal and pass-only fields, which adjudication must not change."""
    fields = [
        "segment_id",
        "source_video",
        "source_video_hash",
        "signer",
        "sign_type",
        "proposal_method",
        "proposal_start_frame",
        "proposal_end_frame",
        "proposal_start_time_s",
        "proposal_end_time_s",
        "review_pass_1",
        "pass1_reviewer",
        "pass1_timestamp",
        "pass1_class",
        "pass1_start_frame",
        "pass1_end_frame",
        "pass1_start_time_s",
        "pass1_end_time_s",
        "review_pass_2",
        "pass2_reviewer",
        "pass2_timestamp",
        "pass2_class",
        "pass2_start_frame",
        "pass2_end_frame",
        "pass2_start_time_s",
        "pass2_end_time_s",
    ]
    projection = [
        {field: str(row.get(field) or "") for field in fields}
        for row in sorted(rows, key=lambda item: item.get("segment_id") or "")
    ]
    blob = json.dumps(projection, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _valid_utc_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def validate_review_completion(row: dict[str, str]) -> list[str]:
    """Validate that both independent review records are complete, accepted or not."""
    errors: list[str] = []
    for pass_n in (1, 2):
        decision = (row.get(f"review_pass_{pass_n}") or "").strip()
        if decision not in REVIEW_DECISIONS:
            errors.append(f"pass{pass_n}_incomplete")
            continue
        reviewer = (row.get(f"pass{pass_n}_reviewer") or "").strip()
        timestamp = (row.get(f"pass{pass_n}_timestamp") or "").strip()
        if not reviewer:
            errors.append(f"pass{pass_n}_requires_named_reviewer")
        if not timestamp:
            errors.append(f"pass{pass_n}_requires_timestamp")
        elif not _valid_utc_timestamp(timestamp):
            errors.append(f"invalid_pass{pass_n}_timestamp")
        if decision == "accepted":
            if not (row.get(f"pass{pass_n}_class") or "").strip():
                errors.append(f"pass{pass_n}_accepted_without_class")
            try:
                start = float(row[f"pass{pass_n}_start_time_s"])
                end = float(row[f"pass{pass_n}_end_time_s"])
                if start < 0 or end <= start:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                errors.append(f"pass{pass_n}_accepted_without_valid_bounds")
    reviewer_1 = (row.get("pass1_reviewer") or "").strip().casefold()
    reviewer_2 = (row.get("pass2_reviewer") or "").strip().casefold()
    if reviewer_1 and reviewer_2 and reviewer_1 == reviewer_2:
        errors.append("independent_passes_require_two_reviewers")
    return errors


def needs_adjudication(row: dict[str, str]) -> bool:
    """Return true for completed pass disagreements that need a final decision."""
    decision_1 = row.get("review_pass_1") or ""
    decision_2 = row.get("review_pass_2") or ""
    if decision_1 not in REVIEW_DECISIONS or decision_2 not in REVIEW_DECISIONS:
        return False
    if decision_1 != decision_2:
        return True
    if decision_1 != "accepted":
        return False
    class_1 = row.get("pass1_class") or ""
    class_2 = row.get("pass2_class") or ""
    if not class_1 or class_1 != class_2:
        return True
    try:
        overlap = temporal_iou(
            float(row["pass1_start_time_s"]),
            float(row["pass1_end_time_s"]),
            float(row["pass2_start_time_s"]),
            float(row["pass2_end_time_s"]),
        )
    except (KeyError, TypeError, ValueError):
        return True
    return overlap < BOUNDS_IOU_MIN


def migrate_row(row: dict[str, str], hashes: dict[str, str]) -> dict[str, str]:
    out = empty_segment()
    source_video = row.get("source_video") or row.get("video_id") or ""
    source_path = row.get("source_path") or ""
    hold_index = int(row.get("hold_index") or 0)
    digest = row.get("source_video_hash") or hashes.get(source_video, "")
    if not digest and source_path:
        path = resolve_repo_path(source_path)
        if path.exists():
            digest = sha256_file(path)
            hashes[source_video] = digest
    for field in SEGMENT_FIELDS:
        if row.get(field):
            out[field] = row[field]
    status = row.get("verification_status", "")
    if not out["review_pass_1"] and status in {"pass1", "pass2"}:
        out["review_pass_1"] = "accepted"
    if not out["review_pass_2"] and status == "pass2":
        out["review_pass_2"] = "accepted"
    out["segment_id"] = out["segment_id"] or segment_id(source_video, hold_index)
    out["source_video"] = source_video
    out["source_video_hash"] = digest
    out["source_path"] = source_path
    out["signer"] = row.get("signer", out["signer"])
    out["hold_index"] = str(hold_index)
    out["start_time_s"] = out["start_time_s"] or row.get("start_time") or ""
    out["end_time_s"] = out["end_time_s"] or row.get("end_time") or ""
    if not out["proposal_start_time_s"]:
        out["proposal_start_time_s"] = out["start_time_s"]
        out["proposal_end_time_s"] = out["end_time_s"]
        out["proposal_start_frame"] = out["start_frame"]
        out["proposal_end_frame"] = out["end_frame"]
    if not out["proposal_method"]:
        out["proposal_method"] = "mediapipe_video_mode_hold_detection"
    apply_include_rule(out)
    return out


def neutralize_unreviewed(row: dict[str, str]) -> dict[str, str]:
    if row.get("review_pass_1") or row.get("review_pass_2"):
        apply_include_rule(row)
        return row
    if row.get("class") and not row.get("expected_class_hint"):
        row["expected_class_hint"] = row["class"]
    row["class"] = ""
    row["include_in_v2"] = "false"
    return row


def apply_include_rule(row: dict[str, str]) -> None:
    ok, reason = verification_ok(row)
    row["include_in_v2"] = "true" if ok else "false"
    if ok:
        if row.get("adjudication_decision") == "accepted":
            row["class"] = row["adjudication_class"]
            row["start_frame"] = row["adjudication_start_frame"]
            row["end_frame"] = row["adjudication_end_frame"]
            row["start_time_s"] = row["adjudication_start_time_s"]
            row["end_time_s"] = row["adjudication_end_time_s"]
        else:
            row["class"] = row["pass2_class"] or row.get("class", "")
            row["start_frame"] = row["pass2_start_frame"] or row["start_frame"]
            row["end_frame"] = row["pass2_end_frame"] or row["end_frame"]
            row["start_time_s"] = row["pass2_start_time_s"] or row["start_time_s"]
            row["end_time_s"] = row["pass2_end_time_s"] or row["end_time_s"]
        row["exclusion_reason"] = ""
    elif reason and row.get("review_pass_1") not in {"rejected", "incomplete", "ambiguous"}:
        if not row.get("exclusion_reason") or row["exclusion_reason"] in {"", "accepted"}:
            row["exclusion_reason"] = reason


def verification_ok(row: dict[str, str]) -> tuple[bool, str]:
    for pass_n in (1, 2):
        decision = row.get(f"review_pass_{pass_n}") or ""
        if decision not in REVIEW_DECISIONS:
            return False, f"pass{pass_n}_incomplete"
        if not (row.get(f"pass{pass_n}_reviewer") or "").strip():
            return False, f"pass{pass_n}_requires_named_reviewer"
        if not (row.get(f"pass{pass_n}_timestamp") or "").strip():
            return False, f"pass{pass_n}_requires_timestamp"
    r1 = (row.get("pass1_reviewer") or "").strip().casefold()
    r2 = (row.get("pass2_reviewer") or "").strip().casefold()
    if r1 == r2:
        return False, "independent_passes_require_two_reviewers"
    adjudication = row.get("adjudication_decision") or ""
    if adjudication:
        if adjudication not in {"accepted", "rejected"}:
            return False, "invalid_adjudication_decision"
        if not (row.get("adjudication_reviewer") or "").strip():
            return False, "adjudication_requires_named_reviewer"
        if not (row.get("adjudication_timestamp") or "").strip():
            return False, "adjudication_requires_timestamp"
        if adjudication == "rejected":
            return False, "adjudicated_reject"
        if not (row.get("adjudication_class") or "").strip():
            return False, "adjudication_accept_requires_class"
        try:
            start = float(row["adjudication_start_time_s"])
            end = float(row["adjudication_end_time_s"])
            if start < 0 or end <= start:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            return False, "adjudication_accept_requires_valid_bounds"
        return True, ""
    if row.get("review_pass_1") != row.get("review_pass_2"):
        return False, "decision_disagreement"
    if row.get("review_pass_1") != "accepted":
        return False, "both_passes_do_not_accept"
    class_1 = row.get("pass1_class") or ""
    class_2 = row.get("pass2_class") or ""
    if not class_1 or not class_2:
        return False, "both_passes_need_class"
    if class_1 != class_2:
        return False, "class_disagreement"
    try:
        iou = temporal_iou(
            float(row["pass1_start_time_s"]),
            float(row["pass1_end_time_s"]),
            float(row["pass2_start_time_s"]),
            float(row["pass2_end_time_s"]),
        )
    except (TypeError, ValueError, KeyError):
        return False, "missing_pass_bounds"
    if iou < BOUNDS_IOU_MIN:
        return False, f"bounds_disagreement_iou_{iou:.2f}"
    return True, ""


def validate_verified_row(
    row: dict[str, str],
    consonants: set[str],
    vowels: set[str],
    duration: float,
    actual_source_hash: str | None = None,
) -> list[str]:
    errors: list[str] = []
    ok, reason = verification_ok(row)
    if not ok:
        errors.append(reason)
    label = row.get("class") or ""
    sign_type = row.get("sign_type") or ""
    if sign_type == "vowel" and label not in vowels:
        errors.append("class_not_in_vowels")
    if sign_type == "consonant" and label not in consonants:
        errors.append("class_not_in_consonants")
    if sign_type not in {"vowel", "consonant"}:
        errors.append("bad_sign_type")
    try:
        start_t = float(row["start_time_s"])
        end_t = float(row["end_time_s"])
        start_f = int(row["start_frame"])
        end_f = int(row["end_frame"])
    except (TypeError, ValueError, KeyError):
        errors.append("non_numeric_bounds")
        return errors
    if not (end_t > start_t and end_f > start_f and start_t >= 0 and start_f >= 0):
        errors.append("bounds_not_positive")
    if duration > 0 and (end_t > duration + 0.05 or start_t > duration):
        errors.append("bounds_outside_duration")
    recorded_hash = (row.get("source_video_hash") or "").strip().lower()
    if not recorded_hash:
        errors.append("missing_source_hash")
    elif not SHA256_RE.fullmatch(recorded_hash):
        errors.append("invalid_source_hash")
    source_path = resolve_repo_path(row.get("source_path") or "")
    if not source_path.is_file():
        errors.append("missing_source_video")
    elif recorded_hash:
        observed_hash = actual_source_hash or sha256_file(source_path)
        if recorded_hash != observed_hash:
            errors.append("source_hash_mismatch")
    if not row.get("fps"):
        errors.append("missing_fps")
    else:
        try:
            fps = float(row["fps"])
            tolerance = max(0.05, 1.5 / fps)
            if fps <= 0:
                raise ValueError
            if abs(start_f / fps - start_t) > tolerance or abs(end_f / fps - end_t) > tolerance:
                errors.append("frame_time_mismatch")
        except (TypeError, ValueError, ZeroDivisionError):
            errors.append("bad_fps")
    for field in ("pass1_timestamp", "pass2_timestamp"):
        value = (row.get(field) or "").strip()
        if value and not _valid_utc_timestamp(value):
            errors.append(f"invalid_{field}")
    if row.get("adjudication_decision") == "accepted":
        value = (row.get("adjudication_timestamp") or "").strip()
        if not value or not _valid_utc_timestamp(value):
            errors.append("invalid_adjudication_timestamp")
    return errors


def overlaps_on_tape(rows: list[dict[str, str]]) -> list[str]:
    problems: list[str] = []
    by_video: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_video.setdefault(row["source_video"], []).append(row)
    for video_id, group in by_video.items():
        ordered = sorted(group, key=lambda item: float(item["start_time_s"] or 0))
        for prev, cur in zip(ordered, ordered[1:]):
            if float(cur["start_time_s"]) < float(prev["end_time_s"]) - 1e-3:
                problems.append(f"{video_id}: {prev['segment_id']} overlaps {cur['segment_id']}")
    return problems


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    *,
    backup: bool = False,
) -> None:
    """Atomically replace a CSV, optionally preserving the immediately prior copy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        if backup and path.exists():
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


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
        if row.get("include_in_v2") != "true":
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
    """Freeze every complete signer in the predeclared test hierarchy."""
    have = set(complete)
    return [signer for signer in TEST_HIERARCHY if signer in have]


def resolve_roles(complete: list[str], frozen: bool = False) -> dict[str, str]:
    del frozen  # Test-hierarchy signers are never released into development.
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
        elif signer in TEST_HIERARCHY:
            roles[signer] = "held_as_fallback"
        else:
            roles[signer] = "development"
    return roles
