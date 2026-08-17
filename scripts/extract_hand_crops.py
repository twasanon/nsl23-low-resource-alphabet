#!/usr/bin/env python3
"""Crop hands from cached NSL23 RGB frames using MediaPipe image-space landmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from nsl23.landmarks import make_detector


def crop_hand(frame: np.ndarray, detector, size: int, padding: float) -> np.ndarray:
    rgb = np.ascontiguousarray(frame)
    result = detector.process(rgb)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    if not result.multi_hand_landmarks:
        return canvas
    handedness = result.multi_handedness or []
    scored = []
    for index, landmarks in enumerate(result.multi_hand_landmarks):
        score = 1.0
        if index < len(handedness) and getattr(handedness[index], "classification", None):
            score = float(handedness[index].classification[0].score)
        scored.append((landmarks, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    points = np.array([[lm.x, lm.y] for lm in scored[0][0].landmark], dtype=np.float32)
    height, width = frame.shape[:2]
    xs = points[:, 0] * width
    ys = points[:, 1] * height
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    box_w = max(x1 - x0, 1.0)
    box_h = max(y1 - y0, 1.0)
    pad_x = padding * box_w
    pad_y = padding * box_h
    left = int(max(0, np.floor(x0 - pad_x)))
    top = int(max(0, np.floor(y0 - pad_y)))
    right = int(min(width, np.ceil(x1 + pad_x)))
    bottom = int(min(height, np.ceil(y1 + pad_y)))
    if right <= left or bottom <= top:
        return canvas
    crop = frame[top:bottom, left:right]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/hand_crop_cache_primary"))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--padding", type=float, default=0.35)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector = make_detector(static_image_mode=True, min_detection_confidence=0.35)
    seen: set[str] = set()
    written = 0
    missing_frames = 0
    try:
        for manifest in args.manifests:
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                sample_id = row["sample_id"]
                if sample_id in seen:
                    continue
                seen.add(sample_id)
                destination = args.output_dir / f"{sample_id}.npy"
                if destination.exists() and not args.force:
                    written += 1
                    continue
                frames = np.load(row["frame_cache_path"], allow_pickle=False)
                crops = np.stack([crop_hand(frame, detector, args.size, args.padding) for frame in frames], axis=0)
                if not crops.any():
                    missing_frames += 1
                np.save(destination, crops)
                written += 1
            print(json.dumps({"manifest": str(manifest), "rows": len(rows)}), flush=True)
    finally:
        detector.close()
    print(json.dumps({"clips": written, "unique": len(seen), "all_black_written_last_pass": missing_frames}, indent=2))


if __name__ == "__main__":
    main()
