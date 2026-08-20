#!/usr/bin/env python3
"""Report pre-adjudication decision, class, and temporal agreement for V2."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.v2_annotation import (
    BOUNDS_IOU_MIN,
    REVIEW_DECISIONS,
    TEST_HIERARCHY,
    agreement_projection_sha256,
    read_segments,
    sha256_file,
    temporal_iou,
    validate_review_completion,
)


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def cohens_kappa(pairs: list[tuple[str, str]], categories: list[str]) -> float | None:
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(left == right for left, right in pairs) / total
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum(
        (left_counts[category] / total) * (right_counts[category] / total)
        for category in categories
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else None
    return (observed - expected) / (1.0 - expected)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    both = [row for row in rows if row.get("review_pass_1") and row.get("review_pass_2")]
    decision_agree = sum(row["review_pass_1"] == row["review_pass_2"] for row in both)
    decision_pairs = [(row["review_pass_1"], row["review_pass_2"]) for row in both]
    both_accepted = [
        row
        for row in both
        if row["review_pass_1"] == "accepted" and row["review_pass_2"] == "accepted"
    ]
    class_agree = sum(row.get("pass1_class") == row.get("pass2_class") for row in both_accepted)
    class_pairs = [
        (row.get("pass1_class") or "", row.get("pass2_class") or "")
        for row in both_accepted
    ]
    ious = []
    for row in both_accepted:
        try:
            ious.append(
                temporal_iou(
                    float(row["pass1_start_time_s"]),
                    float(row["pass1_end_time_s"]),
                    float(row["pass2_start_time_s"]),
                    float(row["pass2_end_time_s"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    different_test_reviewers = [
        row
        for row in both
        if row.get("signer") in TEST_HIERARCHY
        and (row.get("pass1_reviewer") or "").strip().lower()
        != (row.get("pass2_reviewer") or "").strip().lower()
        and (row.get("pass1_reviewer") or "").strip()
        and (row.get("pass2_reviewer") or "").strip()
    ]
    different_reviewers = [
        row
        for row in both
        if (row.get("pass1_reviewer") or "").strip().casefold()
        != (row.get("pass2_reviewer") or "").strip().casefold()
        and (row.get("pass1_reviewer") or "").strip()
        and (row.get("pass2_reviewer") or "").strip()
    ]
    return {
        "proposals": len(rows),
        "both_passes_complete": len(both),
        "decision_exact_agreement": safe_rate(decision_agree, len(both)),
        "decision_cohens_kappa": cohens_kappa(decision_pairs, sorted(REVIEW_DECISIONS)),
        "decision_pair_counts": dict(
            sorted(Counter(f"{left}|{right}" for left, right in decision_pairs).items())
        ),
        "both_accepted": len(both_accepted),
        "accepted_class_exact_agreement": safe_rate(class_agree, len(both_accepted)),
        "accepted_class_cohens_kappa": cohens_kappa(
            class_pairs,
            sorted({label for pair in class_pairs for label in pair}),
        ),
        "temporal_iou_n": len(ious),
        "temporal_iou_mean": statistics.fmean(ious) if ious else None,
        "temporal_iou_median": statistics.median(ious) if ious else None,
        "temporal_iou_at_least_threshold": safe_rate(
            sum(value >= BOUNDS_IOU_MIN for value in ious), len(ious)
        ),
        "manual_insert_proposals": sum(row.get("proposal_method") == "manual_insert" for row in rows),
        "test_hierarchy_both_passes": sum(
            row.get("signer") in TEST_HIERARCHY for row in both
        ),
        "all_proposals_different_named_reviewers": len(different_reviewers),
        "test_hierarchy_different_named_reviewers": len(different_test_reviewers),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, default=Path("annotations/v2/segments.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("annotations/v2/agreement_pre_adjudication.json"),
    )
    args = parser.parse_args()

    rows = read_segments(args.segments)
    if any(row.get("adjudication_decision") for row in rows):
        raise SystemExit(
            "Adjudication already exists. This report must be saved before adjudication. "
            "Restore the pre-adjudication CSV backup if the report was missed."
        )
    completion_errors = [
        f"{row.get('segment_id')}: {', '.join(errors)}"
        for row in rows
        if (errors := validate_review_completion(row))
    ]
    if completion_errors:
        raise SystemExit(
            "Both reviews must be complete before the agreement report:\n"
            + "\n".join(completion_errors[:50])
        )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("signer") or "unknown"].append(row)
    payload = {
        "before_adjudication": True,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bounds_iou_threshold": BOUNDS_IOU_MIN,
        "source_segments_sha256": sha256_file(args.segments),
        "pass_projection_sha256": agreement_projection_sha256(rows),
        "overall": summarize(rows),
        "per_signer": {signer: summarize(group) for signer, group in sorted(grouped.items())},
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
