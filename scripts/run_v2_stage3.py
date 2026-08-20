#!/usr/bin/env python3
"""Run exactly the three preregistered V2 geometry trials and summarize them."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.v2_protocol import load_and_validate_frozen_protocol


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/v2_protocol.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/landmark_cache_v2")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        protocol = load_and_validate_frozen_protocol(args.protocol)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if set(protocol.get("assigned_roles", {}).get("validation") or []) != {"S3"}:
        raise SystemExit("S3 is incomplete. Skip Stage 3 and freeze the control model spec.")

    training = protocol["model_training"]
    for trial in [item["id"] for item in protocol["stage3"]["trials"]]:
        for signer in protocol["stage3"]["folds"]:
            fold = f"heldout_{signer.lower()}"
            manifest = ROOT / "data/manifests/v2" / f"stage3_{fold}.csv"
            if not manifest.is_file():
                raise SystemExit(f"Missing {manifest}; run prepare_v2_manifests.py")
            for seed in protocol["estimand"]["seeds"]:
                output = ROOT / "artifacts/v2/stage3" / trial / fold / f"seed_{seed}"
                if (output / "metrics.json").exists():
                    raise SystemExit(f"Stage 3 result already exists: {output}")
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/train_protonet.py"),
                        "--samples",
                        str(manifest),
                        "--output-dir",
                        str(output),
                        "--cache-dir",
                        str(args.cache_dir),
                        "--seed",
                        str(seed),
                        "--feature-mode",
                        trial,
                        "--epochs",
                        str(training["epochs"]),
                        "--episodes",
                        str(training["episodes_per_epoch"]),
                        "--support",
                        str(training["support_per_class"]),
                        "--query",
                        str(training["query_per_class"]),
                        "--hidden-dim",
                        str(training["hidden_dim"]),
                        "--layers",
                        str(training["layers"]),
                        "--dropout",
                        str(training["dropout"]),
                        "--learning-rate",
                        str(training["learning_rate"]),
                        "--weight-decay",
                        str(training["weight_decay"]),
                        "--device",
                        args.device,
                        "--v2-protocol",
                        str(args.protocol),
                    ],
                    check=True,
                    cwd=ROOT,
                )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/summarize_v2_stage3.py"),
            "--protocol",
            str(args.protocol),
        ],
        check=True,
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
