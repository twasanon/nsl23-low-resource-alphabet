#!/usr/bin/env python3
"""Run or resume the pre-specified batch-one latency benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


MODELS = ("single_frame_cnn", "cnn_bilstm", "cnn_transformer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/runs/primary"))
    parser.add_argument("--fold", default="heldout_s2", choices=("heldout_s1", "heldout_s2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=Path, default=Path("data/manifests/folds/heldout_s2_cached.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runtime"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.fold not in args.samples.name:
        raise ValueError(f"Benchmark fold {args.fold} disagrees with sample manifest {args.samples}")
    for index, model in enumerate(MODELS, start=1):
        checkpoint = args.run_dir / args.fold / model / f"seed_{args.seed}" / "best_checkpoint.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        destination = args.output_dir / model
        report = destination / "runtime.json"
        if report.is_file() and not args.force:
            print(f"[{index}/{len(MODELS)}] skip completed {model}", flush=True)
            continue
        command = [
            sys.executable,
            "-m",
            "nsl23.benchmark",
            "--checkpoint",
            str(checkpoint),
            "--samples",
            str(args.samples),
            "--output-dir",
            str(destination),
            "--device",
            args.device,
        ]
        print(f"[{index}/{len(MODELS)}] benchmark {model}", flush=True)
        subprocess.run(command, check=True)

    rows = []
    for model in MODELS:
        report_path = args.output_dir / model / "runtime.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": model,
                "fold": report["fold"],
                "seed": report["seed"],
                "parameters_total": report["parameters_total"],
                "parameters_trainable": report["parameters_trainable"],
                "model_only_median_ms": report["model_only"]["median_ms_per_clip"],
                "model_only_p95_ms": report["model_only"]["p95_ms_per_clip"],
                "model_only_clips_per_second": report["model_only"]["clips_per_second_from_median"],
                "end_to_end_median_ms": report["end_to_end"]["median_ms_per_clip"],
                "end_to_end_p95_ms": report["end_to_end"]["p95_ms_per_clip"],
                "end_to_end_clips_per_second": report["end_to_end"]["clips_per_second_from_median"],
                "device": report["environment"]["device"],
                "platform": report["environment"]["platform"],
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import csv

    with (args.output_dir / "runtime_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "pass", "benchmarks": len(rows), "summary": str(args.output_dir / "runtime_summary.csv")}, indent=2))


if __name__ == "__main__":
    main()
