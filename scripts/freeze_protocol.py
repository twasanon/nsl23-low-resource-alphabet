#!/usr/bin/env python3
"""Record the immutable inputs to the preregistered primary experiment matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_INPUTS = (
    "PROTOCOL.md",
    "requirements.txt",
    "configs/nsl23_classes.json",
    "configs/isolated_twofold_protocol.json",
    "configs/experiment_local_m4.json",
    "data/manifests/nsl23_manifest_all.csv",
    "data/manifests/folds/heldout_s1_cached.csv",
    "data/manifests/folds/heldout_s2_cached.csv",
    "src/nsl23/dataset.py",
    "src/nsl23/frame_cache.py",
    "src/nsl23/metrics.py",
    "src/nsl23/models.py",
    "src/nsl23/train.py",
    "scripts/run_primary_experiments.py",
    "scripts/verify_primary_protocol.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("artifacts/audit/protocol_freeze.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output

    records: list[dict[str, object]] = []
    for relative in DEFAULT_INPUTS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    payload = {
        "schema_version": 1,
        "purpose": "Immutable input snapshot taken before the first full primary training run.",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "git_revision": git_revision(root),
        "inputs": records,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["snapshot_sha256"] = hashlib.sha256(canonical).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(output), "snapshot_sha256": payload["snapshot_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
