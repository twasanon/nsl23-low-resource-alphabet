#!/usr/bin/env python3
"""Extract the authoritative 630-video inventory from Details Dataset.docx.

The Word document distributed with NSL23 contains one row per source video.
This script turns that table into stable CSV/JSON audit artifacts without
pretending that aggregate recordings are isolated labelled clips.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from docx import Document


VIDEO_SUFFIX_RE = re.compile(r"\.(?:mov|mp4)$", re.IGNORECASE)
TAKE_SUFFIX_RE = re.compile(r"\s*\(\d+\)\s*$")

VOWEL_LABELS = {
    "A",
    "AA",
    "I",
    "II",
    "U",
    "UU",
    "RI",
    "E",
    "AI",
    "O",
    "AU",
    "AM",
    "AH",
}

AGGREGATE_TOKENS = (
    "ALL",
    "CONSONANT_CAMERA",
    "CONSONANT_PHONE",
    "CONSONANT_PNONE",
    "CONSONANT_UNPREPARED",
    "NSL_CONSONANT",
    "NSL_VOWEL",
)


def normalize_stem(filename: str) -> str:
    stem = VIDEO_SUFFIX_RE.sub("", filename.strip())
    stem = TAKE_SUFFIX_RE.sub("", stem)
    return re.sub(r"\s+", "_", stem).upper().strip("_")


def is_aggregate(filename: str, duration_seconds: float) -> bool:
    stem = normalize_stem(filename)
    return duration_seconds > 10.0 or any(token in stem for token in AGGREGATE_TOKENS)


def sign_type(filename: str, aggregate: bool) -> str:
    stem = normalize_stem(filename)
    if "VOWEL" in stem:
        return "vowel"
    if "CONSONANT" in stem:
        return "consonant"
    if not aggregate and stem in VOWEL_LABELS | {"RE"}:
        return "vowel"
    if not aggregate:
        return "consonant"
    return "unknown"


def extract_rows(docx_path: Path) -> list[dict[str, object]]:
    document = Document(docx_path)
    candidates = [
        table
        for table in document.tables
        if table.rows
        and "Volunteer" in table.rows[0].cells[0].text
        and "File Name" in table.rows[0].cells[1].text
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one video inventory table, found {len(candidates)}")

    table = candidates[0]
    rows: list[dict[str, object]] = []
    for source_row, row in enumerate(table.rows[1:], start=2):
        values = [" ".join(cell.text.split()) for cell in row.cells]
        if len(values) < 5 or not values[0]:
            continue
        signer, filename, resolution, fps_raw, duration_raw = values[:5]
        try:
            fps = float(fps_raw)
            duration = float(duration_raw)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric metadata in Word table row {source_row}: {values}") from exc

        width = height = None
        match = re.fullmatch(r"(\d+)\s*[xX×]\s*(\d+)", resolution)
        if match:
            width, height = (int(match.group(1)), int(match.group(2)))

        aggregate = is_aggregate(filename, duration)
        kind = sign_type(filename, aggregate)
        raw_label = "" if aggregate else normalize_stem(filename)
        rows.append(
            {
                "document_row": source_row,
                "signer": signer.upper(),
                "raw_filename": filename,
                "raw_label": raw_label,
                "sign_type": kind,
                "is_aggregate": aggregate,
                "expected_gesture_count": 36 if aggregate and kind == "consonant" else 13 if aggregate and kind == "vowel" else 1,
                "resolution": resolution,
                "width": width,
                "height": height,
                "fps": fps,
                "duration_seconds": duration,
            }
        )

    if len(rows) != 630:
        raise RuntimeError(f"Expected 630 source videos, extracted {len(rows)}")
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_signer: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        category = "aggregate" if row["is_aggregate"] else "isolated"
        by_signer[str(row["signer"])][category] += 1
        by_signer[str(row["signer"])]["videos"] += 1

    isolated = [row for row in rows if not row["is_aggregate"]]
    aggregate = [row for row in rows if row["is_aggregate"]]
    expected_gestures = sum(int(row["expected_gesture_count"]) for row in rows)
    return {
        "source_document_rows": len(rows),
        "isolated_video_rows": len(isolated),
        "aggregate_video_rows": len(aggregate),
        "expected_gestures_from_document": expected_gestures,
        "signers": sorted({str(row["signer"]) for row in rows}, key=lambda value: int(value[1:])),
        "per_signer": {
            signer: dict(by_signer[signer])
            for signer in sorted(by_signer, key=lambda value: int(value[1:]))
        },
        "isolated_raw_labels": sorted({str(row["raw_label"]) for row in isolated}),
        "aggregate_files": [
            {
                "signer": row["signer"],
                "raw_filename": row["raw_filename"],
                "sign_type": row["sign_type"],
                "duration_seconds": row["duration_seconds"],
                "expected_gesture_count": row["expected_gesture_count"],
            }
            for row in aggregate
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    rows = extract_rows(args.docx)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("source_document_rows", "isolated_video_rows", "aggregate_video_rows", "expected_gestures_from_document")}, indent=2))


if __name__ == "__main__":
    main()
