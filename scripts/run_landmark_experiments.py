#!/usr/bin/env python3
"""Train landmark models on the frozen NSL23 signer-held-out folds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


MODELS = ("landmark_mlp", "landmark_bilstm")
FOLDS = ("heldout_s1", "heldout_s2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", type=Path, default=Path("data/manifests/folds"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_local_m4.json"))
    parser.add_argument("--classes", type=Path, default=Path("configs/nsl23_classes.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/primary"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--folds", nargs="+", choices=FOLDS, default=list(FOLDS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = args.seeds or [int(seed) for seed in config["seed_list"]]
    jobs = [(fold, model, seed) for fold in args.folds for model in args.models for seed in seeds]
    for job_index, (fold, model, seed) in enumerate(jobs, start=1):
        samples = args.fold_dir / f"{fold}_cached.csv"
        if not samples.exists():
            raise FileNotFoundError(samples)
        output = args.output_dir / fold / model / f"seed_{seed}"
        metrics = output / "metrics.json"
        if metrics.exists() and not args.force:
            print(f"[{job_index}/{len(jobs)}] skip completed {fold} {model} seed={seed}", flush=True)
            continue
        command = [
            sys.executable,
            "-m",
            "nsl23.train",
            "--samples",
            str(samples),
            "--experiment-config",
            str(args.config),
            "--classes",
            str(args.classes),
            "--model",
            model,
            "--seed",
            str(seed),
            "--output-dir",
            str(output),
            "--device",
            args.device,
        ]
        if args.max_epochs is not None:
            command.extend(["--max-epochs", str(args.max_epochs)])
        if args.bootstrap_replicates is not None:
            command.extend(["--bootstrap-replicates", str(args.bootstrap_replicates)])
        print(f"[{job_index}/{len(jobs)}] run {fold} {model} seed={seed}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
