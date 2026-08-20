#!/usr/bin/env python3
"""Propose hold segments from MediaPipe motion; never write pass-2 labels.

Equal-duration bins are not used. A later visual pass must confirm every row
before include_in_v2 can be true.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.landmarks import make_detector
from nsl23.v2_annotation import (
    SEGMENT_FIELDS,
    TAPE_OUTCOME_FIELDS,
    migrate_row,
    segment_id,
    sha256_file,
    write_csv,
)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def hold_runs(low: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(low.tolist()):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= min_len:
                runs.append((start, index))
            start = None
    if start is not None and len(low) - start >= min_len:
        runs.append((start, len(low)))
    return runs


def merge_close_holds(holds: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not holds:
        return []
    merged = [holds[0]]
    for start, end in holds[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def extract_track(path: Path, stride: int) -> tuple[np.ndarray, np.ndarray, float]:
    times: list[float] = []
    xy: list[np.ndarray] = []
    detected: list[float] = []
    detector = make_detector(static_image_mode=False, min_detection_confidence=0.35)
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fallback_rate = float(stream.average_rate) if stream.average_rate else 30.0
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index % stride:
                    continue
                timestamp = float(frame.time) if frame.time is not None else frame_index / fallback_rate
                image = np.asarray(frame.to_image().convert("RGB"))
                result = detector.process(image)
                times.append(timestamp)
                if not result.multi_hand_landmarks:
                    xy.append(np.zeros((21, 2), dtype=np.float32))
                    detected.append(0.0)
                    continue
                landmarks = result.multi_hand_landmarks[0]
                coords = np.array([[p.x, p.y] for p in landmarks.landmark], dtype=np.float32)
                xy.append(coords)
                detected.append(1.0)
    finally:
        detector.close()
    return np.asarray(times, dtype=np.float32), np.stack(xy), np.asarray(detected, dtype=np.float32)


def propose_holds(times: np.ndarray, xy: np.ndarray, detected: np.ndarray, fps: float) -> list[tuple[float, float, int, int]]:
    if len(times) < 8:
        return []
    dt = np.diff(times, prepend=times[0] + 1e-3)
    dt = np.clip(dt, 1e-3, 0.2)
    delta = np.linalg.norm(np.diff(xy, axis=0, prepend=xy[:1]), axis=2).mean(axis=1)
    speed = np.where(detected > 0.5, delta / dt, 0.0).astype(np.float32)
    smooth = moving_average(speed, max(3, int(round(0.12 * fps))))
    positive = smooth[detected > 0.5]
    if len(positive) == 0:
        return []
    threshold = float(np.median(positive) + 0.85 * (np.median(np.abs(positive - np.median(positive))) + 1e-6))
    low = (smooth < threshold) & (detected > 0.5)
    min_len = max(3, int(round(0.16 * fps)))
    max_gap = max(1, int(round(0.08 * fps)))
    holds = merge_close_holds(hold_runs(low, min_len), max_gap)
    out = []
    for start, end in holds:
        start_t = float(times[start])
        end_t = float(times[min(end, len(times) - 1)])
        if end_t - start_t < 0.14:
            continue
        start_frame = int(round(start_t * fps))
        end_frame = max(start_frame + 1, int(round(end_t * fps)))
        out.append((start_t, end_t, start_frame, end_frame))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapes", type=Path, default=Path("annotations/v2/tapes.csv"))
    parser.add_argument("--classes", type=Path, default=Path("configs/nsl23_classes.json"))
    parser.add_argument("--cache", type=Path, default=Path("data/aggregate_track_cache"))
    parser.add_argument("--output", type=Path, default=Path("annotations/v2/segments.csv"))
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth decoded frame.")
    parser.add_argument("--limit", type=int, default=0, help="Optional tape cap for smoke tests.")
    parser.add_argument("--sign-type", choices=["vowel", "consonant", "all"], default="all")
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Reuse data/aggregate_track_cache only; do not re-track video.",
    )
    args = parser.parse_args()

    labels = json.loads(args.classes.read_text(encoding="utf-8"))
    with args.tapes.open(newline="", encoding="utf-8") as handle:
        tapes = list(csv.DictReader(handle))
    if args.sign_type != "all":
        tapes = [row for row in tapes if row["sign_type"] == args.sign_type]
    if args.limit:
        tapes = tapes[: args.limit]

    fieldnames = SEGMENT_FIELDS
    rows: list[dict[str, str]] = []
    outcomes: list[dict[str, str]] = []
    hashes: dict[str, str] = {}
    args.cache.mkdir(parents=True, exist_ok=True)

    for tape in tapes:
        path = Path(tape["source_path"])
        fps = float(tape["fps"])
        expected = int(tape["expected_gesture_count"])
        class_list = labels["vowels"] if tape["sign_type"] == "vowel" else labels["consonants"]
        cache_path = args.cache / (Path(tape["video_id"]).stem + "_track.npz")
        if cache_path.exists():
            packed = np.load(cache_path)
            times, xy, detected = packed["times"], packed["xy"], packed["detected"]
        elif args.from_cache:
            raise SystemExit(f"Missing track cache {cache_path}")
        else:
            print(f"tracking {tape['video_id']}", flush=True)
            stride = args.stride
            if float(tape["duration_seconds"]) > 90:
                stride = max(stride, 2)
            times, xy, detected = extract_track(path, stride)
            np.savez_compressed(cache_path, times=times, xy=xy, detected=detected)
        holds = propose_holds(times, xy, detected, fps)
        n_holds = len(holds)
        print(f"{tape['video_id']}: {n_holds} holds (expected {expected})", flush=True)
        digest = hashes.get(tape["video_id"])
        if not digest:
            digest = sha256_file(path) if path.exists() else ""
            hashes[tape["video_id"]] = digest
        if n_holds == expected:
            note = "hold count matches expected class count; class hints are navigation only"
            status = "pending_review"
        else:
            note = (
                f"detected {n_holds} holds, expected {expected}; "
                "do not treat longest holds as labels, especially for dynamic vowels"
            )
            status = "count_mismatch_pending_review"
        for index, (start_t, end_t, start_f, end_f) in enumerate(holds):
            hint = class_list[index] if index < expected else ""
            rows.append(
                {
                    "segment_id": segment_id(tape["video_id"], index),
                    "source_video": tape["video_id"],
                    "source_video_hash": digest,
                    "signer": tape["signer"],
                    "class": "",
                    "start_frame": str(start_f),
                    "end_frame": str(end_f),
                    "start_time_s": f"{start_t:.4f}",
                    "end_time_s": f"{end_t:.4f}",
                    "fps": tape["fps"],
                    "condition": tape["condition"],
                    "lighting": tape["lighting"],
                    "capture_device": tape["capture_device"],
                    "sign_type": tape["sign_type"],
                    "proposal_method": "mediapipe_video_mode_hold_detection",
                    "hold_index": str(index),
                    "n_holds_detected": str(n_holds),
                    "expected_class_hint": hint,
                    "review_pass_1": "",
                    "review_pass_2": "",
                    "include_in_v2": "false",
                    "exclusion_reason": "",
                    "notes": note,
                    "source_path": tape["source_path"],
                }
            )
        outcomes.append(
            {
                "source_video": tape["video_id"],
                "signer": tape["signer"],
                "sign_type": tape["sign_type"],
                "expected_classes": str(expected),
                "n_holds_detected": str(n_holds),
                "verified_classes": "0",
                "status": status,
                "notes": note,
                "source_path": tape["source_path"],
                "source_video_hash": digest,
            }
        )

    if args.output.exists() and args.sign_type != "all":
        with args.output.open(newline="", encoding="utf-8") as handle:
            previous_raw = list(csv.DictReader(handle))
        hashes_prev: dict[str, str] = {}
        previous = []
        for row in previous_raw:
            migrated = migrate_row(row, hashes_prev)
            if migrated["sign_type"] == args.sign_type:
                continue
            previous.append(migrated)
        rows = previous + rows
    write_csv(args.output, SEGMENT_FIELDS, rows)
    outcome_path = args.output.with_name("tape_outcomes.csv")
    if outcome_path.exists() and args.sign_type != "all":
        with outcome_path.open(newline="", encoding="utf-8") as handle:
            prev_out = [row for row in csv.DictReader(handle) if row["sign_type"] != args.sign_type]
        outcomes = prev_out + outcomes
    write_csv(outcome_path, TAPE_OUTCOME_FIELDS, outcomes)
    print(
        f"Wrote {len(rows)} proposed holds from {len(tapes)} processed tapes to {args.output}; "
        f"outcomes {outcome_path}. No row is include_in_v2."
    )


if __name__ == "__main__":
    main()
