#!/usr/bin/env python3
"""Train improvement models (raw-angle landmarks, cropped-hand CNN) on frozen folds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


FOLDS = ("heldout_s1", "heldout_s2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", type=Path, default=Path("data/manifests/folds"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_improvements.json"))
    parser.add_argument("--classes", type=Path, default=Path("configs/nsl23_classes.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/improvements"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--models", nargs="+", default=["landmark_bilstm", "landmark_mlp", "crop_cnn"])
    parser.add_argument("--folds", nargs="+", default=list(FOLDS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = args.seeds or [int(seed) for seed in config["seed_list"]]
    jobs = [(fold, model, seed) for fold in args.folds for model in args.models for seed in seeds]
    for job_index, (fold, model, seed) in enumerate(jobs, start=1):
        samples = args.fold_dir / f"{fold}_cached.csv"
        output = args.output_dir / fold / model / f"seed_{seed}"
        metrics = output / "metrics.json"
        if metrics.exists() and not args.force:
            print(f"[{job_index}/{len(jobs)}] skip {fold} {model} seed={seed}", flush=True)
            continue
        device = "cpu" if model.startswith("landmark_") else args.device
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
            device,
        ]
        print(f"[{job_index}/{len(jobs)}] run {fold} {model} seed={seed} device={device}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
