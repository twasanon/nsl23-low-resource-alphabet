"""Create and safely merge a pass-2 annotation file with pass-1 data removed."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .v2_annotation import (
    REVIEW_DECISIONS,
    SEGMENT_FIELDS,
    apply_include_rule,
    empty_segment,
    segment_id,
)

PASS2_FIELDS = (
    "review_pass_2",
    "pass2_reviewer",
    "pass2_timestamp",
    "pass2_class",
    "pass2_start_frame",
    "pass2_end_frame",
    "pass2_start_time_s",
    "pass2_end_time_s",
)

PASS1_FIELDS = (
    "review_pass_1",
    "pass1_reviewer",
    "pass1_timestamp",
    "pass1_class",
    "pass1_start_frame",
    "pass1_end_frame",
    "pass1_start_time_s",
    "pass1_end_time_s",
)

ADJUDICATION_FIELDS = tuple(field for field in SEGMENT_FIELDS if field.startswith("adjudication_"))

IMMUTABLE_PROPOSAL_FIELDS = (
    "segment_id",
    "source_video",
    "source_video_hash",
    "signer",
    "fps",
    "duration_seconds",
    "condition",
    "lighting",
    "capture_device",
    "sign_type",
    "proposal_method",
    "hold_index",
    "n_holds_detected",
    "proposal_start_frame",
    "proposal_end_frame",
    "proposal_start_time_s",
    "proposal_end_time_s",
)


def _text(row: dict[str, str], field: str) -> str:
    return str(row.get(field) or "")


def proposal_projection_sha256(rows: list[dict[str, str]]) -> str:
    projection = [
        {field: _text(row, field) for field in IMMUTABLE_PROPOSAL_FIELDS}
        for row in sorted(rows, key=lambda item: _text(item, "segment_id"))
    ]
    blob = json.dumps(projection, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def pass1_projection_sha256(rows: list[dict[str, str]]) -> str:
    fields = (*IMMUTABLE_PROPOSAL_FIELDS, *PASS1_FIELDS)
    projection = [
        {field: _text(row, field) for field in fields}
        for row in sorted(rows, key=lambda item: _text(item, "segment_id"))
    ]
    blob = json.dumps(projection, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sealed_pass2_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return rows usable by pass 2 without pass-1 labels, decisions, or bounds."""
    sealed: list[dict[str, str]] = []
    for source in rows:
        row = empty_segment()
        for field in IMMUTABLE_PROPOSAL_FIELDS:
            row[field] = _text(source, field)
        row["source_path"] = _text(source, "source_path")
        row["start_frame"] = row["proposal_start_frame"]
        row["end_frame"] = row["proposal_end_frame"]
        row["start_time_s"] = row["proposal_start_time_s"]
        row["end_time_s"] = row["proposal_end_time_s"]
        for field in PASS2_FIELDS:
            row[field] = _text(source, field)
        row["include_in_v2"] = "false"
        sealed.append(row)
    return sealed


def _valid_pass2(row: dict[str, str], labels: set[str], reviewer: str | None) -> list[str]:
    errors: list[str] = []
    decision = _text(row, "review_pass_2")
    if not decision:
        return errors
    if decision not in REVIEW_DECISIONS:
        errors.append("invalid_pass2_decision")
    name = _text(row, "pass2_reviewer").strip()
    if not name:
        errors.append("pass2_requires_named_reviewer")
    if reviewer and name.casefold() != reviewer.strip().casefold():
        errors.append("unexpected_pass2_reviewer")
    if not _text(row, "pass2_timestamp").strip():
        errors.append("pass2_requires_timestamp")
    else:
        try:
            parsed = datetime.fromisoformat(_text(row, "pass2_timestamp").replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError
        except ValueError:
            errors.append("invalid_pass2_timestamp")
    if decision == "accepted":
        if _text(row, "pass2_class") not in labels:
            errors.append("pass2_class_not_in_alphabet")
        try:
            start_t = float(row["pass2_start_time_s"])
            end_t = float(row["pass2_end_time_s"])
            start_f = int(row["pass2_start_frame"])
            end_f = int(row["pass2_end_frame"])
            duration = float(row["duration_seconds"] or 0)
            if start_t < 0 or end_t <= start_t or end_f <= start_f:
                raise ValueError
            if duration and end_t > duration + 0.05:
                errors.append("pass2_bounds_outside_tape")
        except (KeyError, TypeError, ValueError):
            errors.append("pass2_requires_valid_bounds")
    return errors


def merge_pass2_rows(
    master: list[dict[str, str]],
    returned: list[dict[str, str]],
    *,
    consonants: set[str],
    vowels: set[str],
    reviewer: str | None = None,
    require_complete: bool = True,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Copy only validated pass-2 fields into master; never trust pass-1/final fields."""
    master_by_id = {_text(row, "segment_id"): dict(row) for row in master}
    returned_by_id = {_text(row, "segment_id"): row for row in returned}
    if not all(returned_by_id) or len(returned_by_id) != len(returned):
        raise ValueError("Returned file has blank or duplicate segment IDs")
    missing = sorted(set(master_by_id) - set(returned_by_id))
    if missing:
        raise ValueError(f"Returned file omitted {len(missing)} exported proposals")

    completed = 0
    added = 0
    for sid, incoming in returned_by_id.items():
        for field in (*PASS1_FIELDS, *ADJUDICATION_FIELDS):
            if _text(incoming, field):
                raise ValueError(f"{sid}: sealed field {field} must be blank")
        if sid in master_by_id:
            target = master_by_id[sid]
            for field in IMMUTABLE_PROPOSAL_FIELDS:
                if _text(incoming, field) != _text(target, field):
                    raise ValueError(f"{sid}: immutable proposal field changed: {field}")
        else:
            if _text(incoming, "proposal_method") != "manual_insert":
                raise ValueError(f"{sid}: only reviewer-created manual_insert rows may be added")
            video = _text(incoming, "source_video")
            templates = [row for row in master if _text(row, "source_video") == video]
            if not templates:
                raise ValueError(f"{sid}: new row references an unknown source tape")
            try:
                hold_index = int(incoming["hold_index"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{sid}: invalid hold index") from exc
            if sid != segment_id(video, hold_index):
                raise ValueError(f"{sid}: manual insertion ID does not match tape and hold index")
            template = templates[0]
            for field in (
                "source_video_hash",
                "signer",
                "fps",
                "duration_seconds",
                "condition",
                "lighting",
                "capture_device",
                "sign_type",
            ):
                if _text(incoming, field) != _text(template, field):
                    raise ValueError(f"{sid}: new row changed source identity field {field}")
            target = empty_segment()
            for field in IMMUTABLE_PROPOSAL_FIELDS:
                target[field] = _text(incoming, field)
            target["source_path"] = _text(template, "source_path")
            target["start_frame"] = target["proposal_start_frame"]
            target["end_frame"] = target["proposal_end_frame"]
            target["start_time_s"] = target["proposal_start_time_s"]
            target["end_time_s"] = target["proposal_end_time_s"]
            target["proposal_method"] = "manual_insert"
            target["notes"] = "manually inserted during pass 2; pass 1 review still required"
            master_by_id[sid] = target
            added += 1

        labels = vowels if _text(target, "sign_type") == "vowel" else consonants
        errors = _valid_pass2(incoming, labels, reviewer)
        pass1_name = _text(target, "pass1_reviewer").strip().casefold()
        pass2_name = _text(incoming, "pass2_reviewer").strip().casefold()
        if pass1_name and pass2_name and pass1_name == pass2_name:
            errors.append("independent_passes_require_two_reviewers")
        if errors:
            raise ValueError(f"{sid}: {', '.join(errors)}")
        if _text(incoming, "review_pass_2"):
            completed += 1
        for field in PASS2_FIELDS:
            target[field] = _text(incoming, field)
        apply_include_rule(target)

    if require_complete and completed != len(returned):
        raise ValueError(
            f"Pass 2 is incomplete: {completed}/{len(returned)} proposals have a decision"
        )
    merged = sorted(
        master_by_id.values(),
        key=lambda row: (_text(row, "source_video"), float(_text(row, "proposal_start_time_s") or 0), int(_text(row, "hold_index") or 0)),
    )
    return merged, {"imported": completed, "added_manual_inserts": added, "total": len(merged)}
