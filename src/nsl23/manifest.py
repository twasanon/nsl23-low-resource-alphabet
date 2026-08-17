"""Build a lossless video manifest for the NSL23 Zenodo release."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import av


VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v"}
SIGNER_RE = re.compile(r"(?:^|_)(S\d{1,2})(?:_|$)", re.IGNORECASE)
TAKE_RE = re.compile(r"\s*\(\d+\)\s*$")
AGGREGATE_RE = re.compile(r"(?:^|_)(?:ALL|CONSONANT|VOWEL|CAMERA|PHONE|PNONECAMERA)(?:_|$)", re.IGNORECASE)


@dataclass
class VideoRecord:
    source_path: str
    relative_path: str
    source_zip: str
    signer: str
    sign_type: str
    raw_filename: str
    raw_label: str
    canonical_label: str
    is_aggregate: bool
    is_cropped: bool
    background_condition: str
    scene_type: str
    realworld_flag: bool
    lighting: str
    indoor_outdoor: str
    fps: float | None
    frame_count: int | None
    frame_count_source: str
    duration_seconds: float | None
    width: int | None
    height: int | None
    decode_ok: bool
    usable_as_isolated_clip: bool
    reason_excluded: str


def load_class_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    classes = config["consonants"] + config["vowels"]
    if len(classes) != 49 or len(set(classes)) != 49:
        raise ValueError("Class configuration must contain 49 unique labels")
    return config


def signer_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        match = SIGNER_RE.search(part)
        if match:
            signer = match.group(1).upper()
            number = int(signer[1:])
            return signer if 1 <= number <= 14 else "UNK"
    return "UNK"


def clean_raw_label(path: Path, signer: str) -> str:
    stem = TAKE_RE.sub("", path.stem.strip())
    stem = re.sub(rf"^{re.escape(signer)}[_\- ]*", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\s+", "_", stem.upper())
    return re.sub(r"_+", "_", stem).strip("_")


def is_aggregate_path(path: Path, raw_label: str) -> bool:
    lowered = "_".join(part.lower() for part in path.parts)
    return bool(
        AGGREGATE_RE.search(raw_label)
        or "_all_" in f"_{raw_label.lower()}_"
        or ("consonant" in raw_label.lower() and "nsl_consonant" in lowered)
        or ("vowel" in raw_label.lower() and "nsl_vowel" in lowered)
    )


def canonicalize_label(raw_label: str, signer: str, sign_type: str, config: dict[str, Any]) -> str:
    if signer in set(config["short_name_scheme_signers"]):
        label = config["short_name_scheme_aliases"].get(raw_label, raw_label)
    else:
        label = raw_label
    label = config["direct_aliases"].get(label, label)
    valid = set(config["vowels"] if sign_type == "vowel" else config["consonants"])
    return label if label in valid else ""


def parse_condition(path: Path, signer: str, sign_type: str) -> dict[str, Any]:
    lowered = "_".join(part.lower() for part in path.parts)
    prepared = "prepared" in lowered and "unprepared" not in lowered
    unprepared = "unprepared" in lowered or "realworld" in lowered or "real_world" in lowered
    # The release folders omit the word "Unprepared" for S1/S2 consonants,
    # but Table 1 of the NSL23 data paper explicitly describes their Bright
    # and Dark folders as indoor unprepared recordings.
    if sign_type == "consonant" and signer in {"S1", "S2"} and ("bright" in lowered or "dark" in lowered):
        unprepared = True
    realworld = "realworld" in lowered or "real_world" in lowered

    if prepared:
        background = "prepared"
    elif unprepared:
        background = "unprepared"
    else:
        background = "unknown"

    if realworld:
        if signer == "S3":
            scene = "realworld_outdoor"
            location = "outdoor"
        elif signer in {"S6", "S14"}:
            scene = "realworld_indoor"
            location = "indoor"
        else:
            scene = "realworld_unknown"
            location = "unknown"
    else:
        scene = "normal_indoor"
        location = "indoor"

    if "dark" in lowered:
        lighting = "dark"
    elif "bright" in lowered:
        lighting = "bright"
    elif realworld and location == "outdoor":
        lighting = "natural"
    else:
        lighting = "unknown"

    return {
        "background_condition": background,
        "scene_type": scene,
        "realworld_flag": realworld,
        "lighting": lighting,
        "indoor_outdoor": location,
    }


def infer_source_zip(path: Path) -> str:
    text = str(path).lower()
    if "nsl_vowel" in text:
        return "NSL_Vowel.zip"
    for part in (1, 2, 3):
        if f"part_{part}" in text or f"part{part}" in text:
            return f"NSL_Consonant_Part_{part}.zip"
    return "unknown"


def probe_video(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "fps": None,
        "frame_count": None,
        "frame_count_source": "unavailable",
        "duration_seconds": None,
        "width": None,
        "height": None,
        "decode_ok": False,
    }
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            result["width"] = int(stream.width) if stream.width else None
            result["height"] = int(stream.height) if stream.height else None
            result["fps"] = float(stream.average_rate) if stream.average_rate else None
            if stream.duration is not None:
                result["duration_seconds"] = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                result["duration_seconds"] = float(container.duration / av.time_base)

            if stream.frames:
                result["frame_count"] = int(stream.frames)
                result["frame_count_source"] = "container_metadata"
            elif result["duration_seconds"] and result["fps"]:
                result["frame_count"] = int(round(result["duration_seconds"] * result["fps"]))
                result["frame_count_source"] = "duration_times_fps"

            first = next(container.decode(stream), None)
            result["decode_ok"] = first is not None
    except Exception:
        pass
    return result


def build_record(path: Path, root: Path, config: dict[str, Any]) -> VideoRecord:
    relative = path.relative_to(root)
    signer = signer_from_path(path)
    sign_type = "vowel" if "vowel" in str(path).lower() else "consonant"
    raw_label = clean_raw_label(path, signer)
    aggregate = is_aggregate_path(path, raw_label)
    cropped = "cropped" in str(path).lower()
    canonical = "" if aggregate else canonicalize_label(raw_label, signer, sign_type, config)
    probe = probe_video(path)
    condition = parse_condition(path, signer, sign_type)

    reasons: list[str] = []
    if signer == "UNK":
        reasons.append("unknown_signer")
    if not probe["decode_ok"]:
        reasons.append("decode_failure")
    if aggregate:
        reasons.append("aggregate_requires_verified_segmentation")
    elif not canonical:
        reasons.append("unrecognized_label")
    if cropped:
        reasons.append("cropped_or_preprocessed_duplicate")

    usable = not reasons
    return VideoRecord(
        source_path=str(path.resolve()),
        relative_path=str(relative),
        source_zip=infer_source_zip(path),
        signer=signer,
        sign_type=sign_type,
        raw_filename=path.name,
        raw_label=raw_label,
        canonical_label=canonical,
        is_aggregate=aggregate,
        is_cropped=cropped,
        **condition,
        **probe,
        usable_as_isolated_clip=usable,
        reason_excluded=";".join(reasons),
    )


def summarize(records: list[VideoRecord], config: dict[str, Any]) -> dict[str, Any]:
    per_signer: dict[str, Counter[str]] = defaultdict(Counter)
    per_label = Counter(record.canonical_label for record in records if record.usable_as_isolated_clip)
    per_condition = Counter(
        (
            "Prepared"
            if record.background_condition == "prepared"
            else "RealWorld Unprepared"
            if record.realworld_flag
            else "Indoor Unprepared"
            if record.background_condition == "unprepared"
            else "Unknown"
        )
        for record in records
    )
    for record in records:
        per_signer[record.signer]["videos"] += 1
        per_signer[record.signer]["aggregate" if record.is_aggregate else "isolated"] += 1
        per_signer[record.signer]["cropped"] += int(record.is_cropped)
        per_signer[record.signer]["primary_isolated"] += int(record.usable_as_isolated_clip)

    expected = config["consonants"] + config["vowels"]
    usable_labels = sorted(label for label, count in per_label.items() if count)
    return {
        "videos": len(records),
        "decoded_videos": sum(record.decode_ok for record in records),
        "isolated_videos": sum(not record.is_aggregate for record in records),
        "aggregate_videos": sum(record.is_aggregate for record in records),
        "cropped_or_preprocessed_videos": sum(record.is_cropped for record in records),
        "primary_usable_isolated_clips": sum(record.usable_as_isolated_clip for record in records),
        "primary_usable_isolated_signers": sorted(
            {record.signer for record in records if record.usable_as_isolated_clip},
            key=lambda value: int(value[1:]),
        ),
        "primary_usable_isolated_classes": usable_labels,
        "missing_primary_isolated_classes": [label for label in expected if label not in usable_labels],
        "per_signer": {
            signer: dict(per_signer[signer])
            for signer in sorted(per_signer, key=lambda value: 99 if value == "UNK" else int(value[1:]))
        },
        "per_condition_all_videos": {" / ".join(key) if isinstance(key, tuple) else key: value for key, value in per_condition.items()},
        "per_label_primary_isolated": dict(sorted(per_label.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--classes", type=Path, default=Path("configs/nsl23_classes.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    config = load_class_config(args.classes)
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES)
    records = [build_record(path, root, config) for path in paths]
    if not records:
        raise RuntimeError(f"No videos found below {root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)

    summary = summarize(records, config)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("videos", "decoded_videos", "primary_usable_isolated_clips", "missing_primary_isolated_classes")}, indent=2))


if __name__ == "__main__":
    main()
