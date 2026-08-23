#!/usr/bin/env python3
"""Contact sheets for motion-proposed holds. Not a label generator."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import av
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.paths import resolve_repo_path

ROOT = Path(__file__).resolve().parents[1]


def thumbnail(frame: av.VideoFrame) -> Image.Image:
    image = frame.to_image().convert("RGB")
    image.thumbnail((320, 200), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (320, 200), "black")
    canvas.paste(image, ((320 - image.width) // 2, (200 - image.height) // 2))
    return canvas


def sample_video(path: Path, targets: list[float]) -> list[Image.Image]:
    if targets != sorted(targets):
        raise ValueError("Target times must be sorted")
    samples: list[Image.Image] = []
    target_index = 0
    previous: av.VideoFrame | None = None
    previous_time = 0.0
    last: av.VideoFrame | None = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fallback_rate = float(stream.average_rate) if stream.average_rate else 30.0
        for frame_index, frame in enumerate(container.decode(stream)):
            current_time = float(frame.time) if frame.time is not None else frame_index / fallback_rate
            last = frame
            while target_index < len(targets) and current_time >= targets[target_index]:
                target = targets[target_index]
                use_previous = previous is not None and target - previous_time <= current_time - target
                samples.append(thumbnail(previous if use_previous else frame))
                target_index += 1
            if target_index == len(targets):
                break
            previous = frame
            previous_time = current_time
    if last is None:
        raise RuntimeError(f"No frames decoded from {path}")
    while target_index < len(targets):
        samples.append(thumbnail(last))
        target_index += 1
    return samples


def safe_stem(video_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(video_id).stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, default=Path("annotations/v2/segments.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v2_review"))
    args = parser.parse_args()

    with args.segments.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_pages = 0
    for video_id, items in grouped.items():
        items = sorted(items, key=lambda row: int(row["hold_index"]))
        path = resolve_repo_path(items[0]["source_path"], ROOT)
        targets = []
        for row in items:
            start = float(row["start_time"])
            end = float(row["end_time"])
            span = max(end - start, 0.05)
            targets.extend([start + 0.2 * span, start + 0.5 * span, start + 0.8 * span])
        samples = sample_video(path, targets)
        rows_per_page = 10
        for page_index, start in enumerate(range(0, len(items), rows_per_page), start=1):
            page_items = items[start : start + rows_per_page]
            header = 64
            row_h = 220
            label_w = 220
            canvas = Image.new("RGB", (label_w + 3 * 320, header + row_h * len(page_items)), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 10), f"{items[0]['signer']} | {path.name} | motion proposals | visual pass required", fill="black")
            draw.text((label_w + 120, 36), "20%", fill="black")
            draw.text((label_w + 440, 36), "50%", fill="black")
            draw.text((label_w + 760, 36), "80%", fill="black")
            for row_index, row in enumerate(page_items):
                y = header + row_index * row_h
                label = row["class"] or "(unlabeled)"
                draw.text((12, y + 8), f"{int(row['hold_index']) + 1:02d} {label}", fill="black")
                draw.text((12, y + 32), f"{row['start_time']}-{row['end_time']}s", fill="black")
                draw.text((12, y + 56), row["verification_status"], fill="#444444")
                abs_index = start + row_index
                for col in range(3):
                    canvas.paste(samples[abs_index * 3 + col], (label_w + col * 320, y))
            out = args.output_dir / f"{row['signer']}_{safe_stem(video_id)}_page{page_index:02d}.jpg"
            canvas.save(out, quality=90)
            n_pages += 1
    print(f"Wrote {n_pages} review sheets to {args.output_dir}")


if __name__ == "__main__":
    main()
