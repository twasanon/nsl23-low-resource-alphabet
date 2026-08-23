#!/usr/bin/env python3
"""Build frozen V2 image/world/palm landmark caches from sample manifests."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.frame_cache import decode_targets, target_times
from nsl23.landmarks import extract_clip_landmark_views, make_detector
from nsl23.v2_annotation import sha256_file
from nsl23.v2_protocol import (
    load_and_validate_frozen_protocol,
    validate_manifest_provenance,
)
from nsl23.paths import resolve_repo_path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def validate_existing_cache(
    path: Path,
    row: dict[str, str],
    protocol_sha256: str,
    frames_per_clip: int,
) -> tuple[int, int]:
    try:
        with np.load(path, allow_pickle=False) as packed:
            if str(packed["protocol_sha256"].item()) != protocol_sha256:
                raise ValueError("protocol hash")
            if str(packed["source_sha256"].item()) != row["source_video_hash"]:
                raise ValueError("source hash")
            if int(packed["num_frames"].item()) != frames_per_clip:
                raise ValueError("frame count")
            for key in ("image_xyz", "world_xyz", "palm_canonical_world_xyz"):
                if packed[key].shape != (frames_per_clip, 64):
                    raise ValueError(f"{key} shape {packed[key].shape}")
                if not np.isfinite(packed[key]).all():
                    raise ValueError(f"{key} contains NaN or infinity")
            image_detected = int((packed["image_xyz"][:, -1] > 0.5).sum())
            world_detected = int((packed["world_xyz"][:, -1] > 0.5).sum())
    except (KeyError, OSError, ValueError) as exc:
        raise SystemExit(
            f"Existing V2 cache is stale or invalid: {path} ({exc}). "
            "Remove that cache or rerun with --force before any evaluation."
        ) from exc
    return image_detected, world_detected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/v2_protocol.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/landmark_cache_v2"))
    parser.add_argument("--summary", type=Path, default=Path("artifacts/v2/cache_summary.json"))
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--min-detection-confidence", type=float, default=0.35)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        protocol = load_and_validate_frozen_protocol(args.protocol)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    extraction = protocol["feature_extraction"]
    locked = {
        "num_frames": int(extraction["frames_per_clip"]),
        "image_size": int(extraction["image_size"]),
        "min_detection_confidence": float(extraction["min_detection_confidence"]),
    }
    observed = {
        "num_frames": args.num_frames,
        "image_size": args.image_size,
        "min_detection_confidence": args.min_detection_confidence,
    }
    if observed != locked:
        raise SystemExit(f"Feature extraction arguments must match the frozen protocol: {locked}")
    installed_mediapipe = importlib.metadata.version("mediapipe")
    if installed_mediapipe != extraction["mediapipe_version"]:
        raise SystemExit(
            f"MediaPipe must be {extraction['mediapipe_version']}, found {installed_mediapipe}"
        )

    unique: dict[str, dict[str, str]] = {}
    manifest_hashes = {}
    for manifest in args.manifests:
        rows = read_rows(manifest)
        stages = {row.get("v2_stage") or "" for row in rows}
        if len(stages) != 1:
            raise SystemExit(f"Manifest must contain exactly one V2 stage: {manifest}")
        stage = next(iter(stages))
        try:
            validate_manifest_provenance(rows, protocol, stage)
        except ValueError as exc:
            raise SystemExit(f"{manifest}: {exc}") from exc
        manifest_hashes[str(manifest)] = sha256_file(manifest)
        for row in rows:
            identifier = row["sample_id"]
            identity = (row["source_path"], row["start_seconds"], row["end_seconds"])
            if identifier in unique:
                previous = unique[identifier]
                previous_identity = (
                    previous["source_path"],
                    previous["start_seconds"],
                    previous["end_seconds"],
                )
                if identity != previous_identity:
                    raise SystemExit(f"Sample ID collision with different bounds: {identifier}")
            else:
                unique[identifier] = row

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    image_detected_frames = 0
    world_detected_frames = 0
    for row in unique.values():
        destination = args.output_dir / f"{row['sample_id']}.npz"
        if destination.is_file() and not args.force:
            image_count, world_count = validate_existing_cache(
                destination,
                row,
                protocol["protocol_sha256"],
                args.num_frames,
            )
            image_detected_frames += image_count
            world_detected_frames += world_count
        else:
            grouped[row["source_path"]].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector = make_detector(
        static_image_mode=bool(extraction["static_image_mode"]),
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=float(extraction["min_tracking_confidence"]),
        max_num_hands=int(extraction["max_num_hands"]),
        model_complexity=int(extraction["model_complexity"]),
    )
    source_hashes: dict[str, str] = {}
    processed = 0
    try:
        for source_index, (source, rows) in enumerate(sorted(grouped.items()), start=1):
            source_path = resolve_repo_path(source, ROOT)
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            source_hash = sha256_file(source_path)
            source_hashes[source] = source_hash
            for row in rows:
                recorded = row.get("source_video_hash") or ""
                if recorded and recorded != source_hash:
                    raise SystemExit(f"Source hash mismatch for {row['sample_id']}: {source}")
            requests: list[tuple[float, str, int]] = []
            for row in rows:
                times = target_times(
                    float(row["start_seconds"]), float(row["end_seconds"]), args.num_frames
                )
                requests.extend(
                    (float(value), row["sample_id"], index)
                    for index, value in enumerate(times)
                )
            decoded = decode_targets(source_path, requests, args.image_size)
            print(f"[{source_index}/{len(grouped)}] {source}", flush=True)
            for row in rows:
                identifier = row["sample_id"]
                frames = np.stack([decoded[identifier][index] for index in range(args.num_frames)])
                views, detected = extract_clip_landmark_views(frames, detector)
                image_detected_frames += detected
                world_detected_frames += int((views["world_xyz"][:, -1] > 0.5).sum())
                processed += 1
                atomic_save_npz(
                    args.output_dir / f"{identifier}.npz",
                    {
                        **views,
                        "source_sha256": np.asarray(source_hash),
                        "protocol_sha256": np.asarray(protocol["protocol_sha256"]),
                        "num_frames": np.asarray(args.num_frames, dtype=np.int64),
                    },
                )
    finally:
        detector.close()

    expected_files = [args.output_dir / f"{identifier}.npz" for identifier in unique]
    missing = [str(path) for path in expected_files if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} V2 caches; first: {missing[0]}")
    payload = {
        "samples": len(unique),
        "newly_processed": processed,
        "frames_per_sample": args.num_frames,
        "image_detected_frame_rate": image_detected_frames / (len(unique) * args.num_frames),
        "world_detected_frame_rate": world_detected_frames / (len(unique) * args.num_frames),
        "protocol_sha256": protocol["protocol_sha256"],
        "feature_extraction": extraction,
        "mediapipe_version": installed_mediapipe,
        "manifest_sha256": manifest_hashes,
        "cache_dir": str(args.output_dir.resolve()),
        "newly_source_files_hashed": len(source_hashes),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
