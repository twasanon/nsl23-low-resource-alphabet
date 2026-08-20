#!/usr/bin/env python3
"""Train the three frozen final checkpoints without loading test rows."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.v2_protocol import (
    load_and_validate_frozen_protocol,
    load_and_validate_model_spec,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples", type=Path, default=ROOT / "data/manifests/v2/final_retrain.csv"
    )
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/v2_protocol.json")
    parser.add_argument("--model-spec", type=Path, default=ROOT / "configs/v2_model_spec.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/landmark_cache_v2")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/v2/final")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        protocol = load_and_validate_frozen_protocol(args.protocol, require_confirmatory=True)
        spec = load_and_validate_model_spec(args.model_spec, protocol)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not args.samples.is_file():
        raise SystemExit(f"Missing {args.samples}; run prepare_v2_manifests.py")

    for seed in spec["seeds"]:
        output = args.output_root / f"seed_{seed}"
        if (output / "best_checkpoint.pt").exists():
            raise SystemExit(f"Final checkpoint already exists: {output}")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/train_protonet.py"),
                "--samples",
                str(args.samples),
                "--output-dir",
                str(output),
                "--cache-dir",
                str(args.cache_dir),
                "--seed",
                str(seed),
                "--feature-mode",
                spec["feature_mode"],
                "--epochs",
                str(spec["epochs"]),
                "--episodes",
                str(spec["episodes"]),
                "--support",
                str(spec["support"]),
                "--query",
                str(spec["query"]),
                "--hidden-dim",
                str(spec["hidden_dim"]),
                "--layers",
                str(spec["layers"]),
                "--dropout",
                str(spec["dropout"]),
                "--learning-rate",
                str(spec["learning_rate"]),
                "--weight-decay",
                str(spec["weight_decay"]),
                "--device",
                args.device,
                "--v2-protocol",
                str(args.protocol),
                "--v2-model-spec",
                str(args.model_spec),
            ],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    main()
