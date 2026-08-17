#!/usr/bin/env python3
"""Materialize explicit segment annotations from reviewed aggregate sources."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--classes", type=Path, default=Path("configs/nsl23_classes.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--central-fraction", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.25 <= args.central_fraction <= 0.9:
        raise ValueError("central-fraction must be between 0.25 and 0.9")

    manifest = {row["relative_path"]: row for row in read_csv(args.manifest)}
    reviews = read_csv(args.reviews)
    config = json.loads(args.classes.read_text(encoding="utf-8"))
    segments: list[dict[str, object]] = []

    for review in reviews:
        if not truthy(review["include_primary"]):
            continue
        relative_path = review["relative_path"]
        if relative_path not in manifest:
            raise KeyError(f"Reviewed source is missing from manifest: {relative_path}")
        source = manifest[relative_path]
        if source["is_aggregate"].lower() != "true":
            raise ValueError(f"Reviewed source is not marked aggregate: {relative_path}")
        if source["decode_ok"].lower() != "true":
            raise ValueError(f"Reviewed source does not decode: {relative_path}")

        sign_type = review["sign_type"]
        labels = config["vowels"] if sign_type == "vowel" else config["consonants"]
        expected_count = int(review["expected_gesture_count"])
        if expected_count != len(labels):
            raise ValueError(f"Expected count does not match class list for {relative_path}")

        duration = float(source["duration_seconds"])
        fps = float(source["fps"])
        bin_width = duration / len(labels)
        inset = (1.0 - args.central_fraction) / 2.0
        for class_index, label in enumerate(labels):
            bin_start = class_index * bin_width
            start_seconds = bin_start + inset * bin_width
            end_seconds = bin_start + (1.0 - inset) * bin_width
            start_frame = int(round(start_seconds * fps))
            end_frame = max(start_frame + 1, int(round(end_seconds * fps)))
            segments.append(
                {
                    "source_path": source["source_path"],
                    "relative_path": relative_path,
                    "source_zip": source["source_zip"],
                    "signer": source["signer"],
                    "sign_type": sign_type,
                    "canonical_label": label,
                    "class_index_within_type": class_index,
                    "start_seconds": round(start_seconds, 6),
                    "end_seconds": round(end_seconds, 6),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "fps": fps,
                    "background_condition": source["background_condition"],
                    "scene_type": source["scene_type"],
                    "realworld_flag": source["realworld_flag"],
                    "lighting": source["lighting"],
                    "indoor_outdoor": source["indoor_outdoor"],
                    "boundary_method": f"equal_order_bin_central_{int(args.central_fraction * 100)}pct_after_triptych_review",
                    "review_status": review["boundary_review"],
                    "reviewer": review["reviewer"],
                    "review_date": review["review_date"],
                }
            )

    if not segments:
        raise RuntimeError("No reviewed segments were selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(segments[0]))
        writer.writeheader()
        writer.writerows(segments)
    print(f"Wrote {len(segments)} verified segments from {len({row['relative_path'] for row in segments})} aggregate videos")


if __name__ == "__main__":
    main()
