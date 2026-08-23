"""Decode each source video once and cache uniformly sampled RGB frame tensors."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
from PIL import Image

from .paths import portable_path, resolve_repo_path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def letterbox(image: Image.Image, size: int) -> np.ndarray:
    image = image.convert("RGB")
    scale = min(size / image.width, size / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def target_times(start: float, end: float, count: int) -> np.ndarray:
    if end <= start:
        raise ValueError(f"Invalid sampling window {start}..{end}")
    # Temporal-bin centers avoid transition-heavy exact endpoints.
    return start + (np.arange(count, dtype=np.float64) + 0.5) * (end - start) / count


def decode_targets(
    source_path: Path,
    requests: list[tuple[float, str, int]],
    image_size: int,
) -> dict[str, dict[int, np.ndarray]]:
    ordered = sorted(requests, key=lambda item: item[0])
    outputs: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    request_index = 0
    previous_frame = None
    previous_time = None
    previous_array = None
    last_frame = None
    last_time = None
    last_array = None

    with av.open(str(source_path)) as container:
        stream = container.streams.video[0]
        fallback_rate = float(stream.average_rate) if stream.average_rate else 30.0
        for frame_index, frame in enumerate(container.decode(stream)):
            current_time = float(frame.time) if frame.time is not None else frame_index / fallback_rate
            current_array = None
            while request_index < len(ordered) and ordered[request_index][0] <= current_time:
                requested_time, identifier, sequence_index = ordered[request_index]
                choose_previous = (
                    previous_frame is not None
                    and previous_time is not None
                    and requested_time - previous_time <= current_time - requested_time
                )
                if choose_previous:
                    if previous_array is None:
                        previous_array = letterbox(previous_frame.to_image(), image_size)
                    selected = previous_array
                else:
                    if current_array is None:
                        current_array = letterbox(frame.to_image(), image_size)
                    selected = current_array
                outputs[identifier][sequence_index] = selected.copy()
                request_index += 1

            previous_frame = frame
            previous_time = current_time
            previous_array = current_array
            last_frame = frame
            last_time = current_time
            last_array = current_array

    if last_frame is None:
        raise RuntimeError(f"No video frames decoded from {source_path}")
    while request_index < len(ordered):
        _, identifier, sequence_index = ordered[request_index]
        if last_array is None:
            last_array = letterbox(last_frame.to_image(), image_size)
        outputs[identifier][sequence_index] = last_array.copy()
        request_index += 1
    return outputs


def cache_source_group(
    source_path: Path,
    rows: list[dict[str, str]],
    output_dir: Path,
    num_frames: int,
    image_size: int,
    overwrite: bool,
) -> None:
    pending = [row for row in rows if overwrite or not (output_dir / f"{row['sample_id']}.npy").exists()]
    if not pending:
        return
    requests: list[tuple[float, str, int]] = []
    for row in pending:
        times = target_times(float(row["start_seconds"]), float(row["end_seconds"]), num_frames)
        requests.extend((float(value), row["sample_id"], index) for index, value in enumerate(times))

    decoded = decode_targets(source_path, requests, image_size)
    for row in pending:
        identifier = row["sample_id"]
        if len(decoded[identifier]) != num_frames:
            raise RuntimeError(f"Decoded {len(decoded[identifier])}/{num_frames} frames for {identifier}")
        array = np.stack([decoded[identifier][index] for index in range(num_frames)])
        if array.shape != (num_frames, image_size, image_size, 3) or array.dtype != np.uint8:
            raise RuntimeError(f"Unexpected cache tensor for {identifier}: {array.shape} {array.dtype}")
        np.save(output_dir / f"{identifier}.npy", array, allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updated-manifest", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_csv(args.samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_path"]].append(row)

    for index, (source, group) in enumerate(sorted(grouped.items()), start=1):
        print(f"[{index}/{len(grouped)}] {source}", flush=True)
        cache_source_group(resolve_repo_path(source), group, args.output_dir, args.num_frames, args.image_size, args.overwrite)

    for row in rows:
        cache_path = (args.output_dir / f"{row['sample_id']}.npy").resolve()
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        row["frame_cache_path"] = portable_path(cache_path)
        row["num_frames"] = args.num_frames
        row["image_size"] = args.image_size

    args.updated_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.updated_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Cached {len(rows)} samples from {len(grouped)} source videos")


if __name__ == "__main__":
    main()
