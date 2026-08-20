#!/usr/bin/env python3
"""Build every permitted V2 stage manifest after annotation freeze."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.v2_protocol import load_and_validate_frozen_protocol


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_v2_samples.py"


def build(protocol: Path, stage: str, name: str, extra: list[str]) -> Path:
    output = ROOT / "data/manifests/v2" / f"{name}.csv"
    summary = ROOT / "artifacts/v2/manifests" / f"{name}.json"
    command = [
        sys.executable,
        str(BUILDER),
        "--protocol",
        str(protocol),
        "--stage",
        stage,
        "--output",
        str(output),
        "--summary",
        str(summary),
        *extra,
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/v2_protocol.json")
    args = parser.parse_args()
    try:
        protocol = load_and_validate_frozen_protocol(args.protocol)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    outputs = [
        build(args.protocol, "stage2", "stage2_heldout_s1", ["--fold", "heldout_s1"]),
        build(args.protocol, "stage2", "stage2_heldout_s2", ["--fold", "heldout_s2"]),
    ]
    if set(protocol.get("assigned_roles", {}).get("validation") or []) == {"S3"}:
        for signer in ("S1", "S2", "S3"):
            outputs.append(
                build(
                    args.protocol,
                    "stage3",
                    f"stage3_heldout_{signer.lower()}",
                    ["--heldout-signer", signer],
                )
            )
    outputs.append(build(args.protocol, "final_retrain", "final_retrain", []))
    if protocol.get("confirmatory_49way") is True:
        outputs.append(build(args.protocol, "stage4_eval", "stage4_eval", []))
    print(json.dumps({"manifests": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
