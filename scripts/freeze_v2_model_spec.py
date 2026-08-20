#!/usr/bin/env python3
"""Freeze the post-Stage-3 model choice before final retraining."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.v2_annotation import sha256_file
from nsl23.v2_protocol import (
    load_and_validate_frozen_protocol,
    model_spec_sha256,
)


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/v2_protocol.json"))
    parser.add_argument("--model-spec", type=Path, default=Path("configs/v2_model_spec.json"))
    parser.add_argument("--stage3-summary", type=Path, default=None)
    parser.add_argument("--skip-stage3", action="store_true")
    args = parser.parse_args()

    try:
        protocol = load_and_validate_frozen_protocol(args.protocol)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    spec = json.loads(args.model_spec.read_text(encoding="utf-8"))
    if spec.get("frozen"):
        raise SystemExit(f"Model specification is already frozen: {args.model_spec}")
    has_s3 = set(protocol.get("assigned_roles", {}).get("validation") or []) == {"S3"}
    control = protocol["stage3"]["trials"][0]["id"]
    if has_s3:
        if args.skip_stage3:
            raise SystemExit("S3 is complete; the preregistered three-trial Stage 3 cannot be skipped")
        if args.stage3_summary is None or not args.stage3_summary.is_file():
            raise SystemExit("Complete S3 requires --stage3-summary")
        summary = json.loads(args.stage3_summary.read_text(encoding="utf-8"))
        if summary.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise SystemExit("Stage 3 summary belongs to another protocol")
        selected = summary.get("selected_trial")
        allowed = {trial["id"] for trial in protocol["stage3"]["trials"]}
        if selected not in allowed:
            raise SystemExit(f"Invalid Stage 3 selection: {selected}")
        source = {
            "kind": "stage3_summary",
            "path": str(args.stage3_summary),
            "sha256": sha256_file(args.stage3_summary),
        }
    else:
        if not args.skip_stage3 or args.stage3_summary is not None:
            raise SystemExit("S3 is incomplete; invoke with --skip-stage3 and no summary")
        selected = control
        source = {"kind": "preregistered_skip_because_s3_incomplete"}

    training = protocol["model_training"]
    extraction = protocol["feature_extraction"]
    spec.update(
        {
            "frozen": True,
            "protocol_sha256": protocol["protocol_sha256"],
            "model_spec_sha256": None,
            "selected_trial": selected,
            "feature_mode": selected,
            "selection_source": source,
            "epochs": int(training["epochs"]),
            "episodes": int(training["episodes_per_epoch"]),
            "support": int(training["support_per_class"]),
            "query": int(training["query_per_class"]),
            "hidden_dim": int(training["hidden_dim"]),
            "layers": int(training["layers"]),
            "dropout": float(training["dropout"]),
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "num_frames": int(extraction["frames_per_clip"]),
            "seeds": protocol["final_retrain"]["seeds"],
        }
    )
    spec["model_spec_sha256"] = model_spec_sha256(spec)
    atomic_write_json(args.model_spec, spec)
    print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
