#!/usr/bin/env python3
"""Create visual review sheets for candidate equal-width aggregate segments.

The sheets are deliberately a review aid, not an annotation generator.  Each
row shows the 20%, 50%, and 80% frames inside a candidate segment so a reviewer
can see whether the expected gesture is contained cleanly.  A final segments
file must record explicit verification status before training can use it.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import av
from PIL import Image, ImageDraw


VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v"}


def signer_from_path(path: Path) -> str:
    match = re.search(r"(?:^|_)(S\d{1,2})(?:_|$)", "_".join(path.parts), re.IGNORECASE)
    return match.group(1).upper() if match else "UNK"


def video_duration(path: Path) -> float:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return float(container.duration / av.time_base)
    raise RuntimeError(f"Duration is unavailable for {path}")


def thumbnail(frame: av.VideoFrame) -> Image.Image:
    image = frame.to_image().convert("RGB")
    image.thumbnail((360, 220), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (360, 220), "black")
    canvas.paste(image, ((360 - image.width) // 2, (220 - image.height) // 2))
    return canvas


def sample_video(path: Path, targets: list[float]) -> list[Image.Image]:
    """Decode once and retain only the nearest thumbnail for each target time."""
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


def make_pages(path: Path, labels: list[str], output_dir: Path, rows_per_page: int = 12) -> list[Path]:
    duration = video_duration(path)
    step = duration / len(labels)
    signer = signer_from_path(path)
    outputs: list[Path] = []
    targets = [
        class_index * step + fraction * step
        for class_index in range(len(labels))
        for fraction in (0.2, 0.5, 0.8)
    ]
    samples = sample_video(path, targets)

    for page_index, start in enumerate(range(0, len(labels), rows_per_page), start=1):
        page_labels = labels[start : start + rows_per_page]
        header_height = 70
        row_height = 260
        label_width = 170
        canvas = Image.new("RGB", (label_width + 3 * 360, header_height + row_height * len(page_labels)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 12), f"{signer} | {path.name} | duration={duration:.2f}s | equal-width candidate review", fill="black")
        draw.text((label_width + 120, 42), "20%", fill="black")
        draw.text((label_width + 480, 42), "50%", fill="black")
        draw.text((label_width + 840, 42), "80%", fill="black")

        for row_index, label in enumerate(page_labels):
            absolute_index = start + row_index
            segment_start = absolute_index * step
            row_y = header_height + row_index * row_height
            draw.text((12, row_y + 10), f"{absolute_index + 1:02d} {label}", fill="black")
            draw.text((12, row_y + 34), f"{segment_start:.2f}-{segment_start + step:.2f}s", fill="black")
            for col_index in range(3):
                sample = samples[absolute_index * 3 + col_index]
                canvas.paste(sample, (label_width + col_index * 360, row_y))
            draw.line((0, row_y + row_height - 1, canvas.width, row_y + row_height - 1), fill="#cccccc", width=1)

        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)
        output = output_dir / f"{signer}_{safe_stem}_review_{page_index:02d}.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, quality=92, subsampling=0)
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--classes", type=Path, default=Path("configs/nsl23_classes.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-duration", type=float, default=10.0)
    args = parser.parse_args()

    config = json.loads(args.classes.read_text(encoding="utf-8"))
    outputs: list[Path] = []
    for path in sorted(args.root.rglob("*")):
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        try:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                duration = float(stream.duration * stream.time_base) if stream.duration is not None else 0.0
        except (av.AVError, IndexError):
            continue
        if duration <= args.minimum_duration:
            continue
        lowered = "_".join(part.lower() for part in path.parts)
        labels = config["vowels"] if "vowel" in lowered else config["consonants"]
        outputs.extend(make_pages(path, labels, args.output_dir))

    print(f"Generated {len(outputs)} review sheet(s) for {args.root}")


if __name__ == "__main__":
    main()
