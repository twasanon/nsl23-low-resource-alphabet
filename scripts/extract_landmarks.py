#!/usr/bin/env python3
"""Extract MediaPipe hand landmarks from cached NSL23 RGB clips."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from nsl23.landmarks import LANDMARK_DIM, cache_path, extract_clip_landmarks, make_detector


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/landmark_cache_primary"))
    parser.add_argument("--min-detection-confidence", type=float, default=0.35)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector = make_detector(static_image_mode=True, min_detection_confidence=args.min_detection_confidence)
    summary: dict[str, object] = {
        "clips": 0,
        "frames": 0,
        "detected_frames": 0,
        "clips_with_no_detection": 0,
    }
    try:
        for manifest in args.manifests:
            rows = read_rows(manifest)
            for row in rows:
                sample_id = row["sample_id"]
                destination = cache_path(args.output_dir, sample_id)
                if destination.exists() and not args.force:
                    array = np.load(destination, allow_pickle=False)
                    detected = int((array[:, -1] > 0.5).sum())
                else:
                    frames = np.load(row["frame_cache_path"], allow_pickle=False)
                    array, detected = extract_clip_landmarks(frames, detector)
                    if array.shape[-1] != LANDMARK_DIM:
                        raise RuntimeError(f"Bad landmark width {array.shape} for {sample_id}")
                    np.save(destination, array)
                summary["detected_frames"] = int(summary["detected_frames"]) + detected
                if detected == 0:
                    summary["clips_with_no_detection"] = int(summary["clips_with_no_detection"]) + 1
                row["landmark_cache_path"] = str(destination.resolve())
                summary["clips"] = int(summary["clips"]) + 1
                summary["frames"] = int(summary["frames"]) + int(array.shape[0])
            write_rows(manifest, rows)
            print(json.dumps({"manifest": str(manifest), "rows": len(rows)}), flush=True)
    finally:
        detector.close()
    if int(summary["frames"]):
        summary["detection_rate"] = int(summary["detected_frames"]) / int(summary["frames"])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
