#!/usr/bin/env python3
"""Validate all preregistered Stage 3 runs and select one geometry trial."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.v2_annotation import sha256_file
from nsl23.v2_protocol import load_and_validate_frozen_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/v2_protocol.json"))
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/v2/stage3"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v2/stage3_summary.json"))
    args = parser.parse_args()

    try:
        protocol = load_and_validate_frozen_protocol(args.protocol)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if set(protocol.get("assigned_roles", {}).get("validation") or []) != {"S3"}:
        raise SystemExit("S3 is incomplete; skip Stage 3 and freeze the control model spec")

    trials = [trial["id"] for trial in protocol["stage3"]["trials"]]
    folds = [f"heldout_{signer.lower()}" for signer in protocol["stage3"]["folds"]]
    seeds = protocol["estimand"]["seeds"]
    table = []
    source_hashes = {}
    for trial in trials:
        macro_values = []
        top1_values = []
        runs = []
        for fold in folds:
            for seed in seeds:
                path = args.runs_root / trial / fold / f"seed_{seed}" / "metrics.json"
                if not path.is_file():
                    raise SystemExit(f"Missing preregistered Stage 3 result: {path}")
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("protocol_sha256") != protocol["protocol_sha256"]:
                    raise SystemExit(f"Protocol hash mismatch: {path}")
                if data.get("feature_mode") != trial or int(data.get("seed", -1)) != seed:
                    raise SystemExit(f"Trial/seed mismatch: {path}")
                if data.get("fold") != fold:
                    raise SystemExit(f"Fold mismatch: {path}")
                macro_values.append(float(data["test_macro_recall"]))
                top1_values.append(float(data["test_accuracy"]))
                source_hashes[str(path)] = sha256_file(path)
                runs.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "macro_recall": macro_values[-1],
                        "top1": top1_values[-1],
                    }
                )
        table.append(
            {
                "trial": trial,
                "mean_macro_recall": statistics.fmean(macro_values),
                "sample_sd_macro_recall": statistics.stdev(macro_values),
                "mean_top1": statistics.fmean(top1_values),
                "sample_sd_top1": statistics.stdev(top1_values),
                "runs": runs,
            }
        )

    order = {trial: index for index, trial in enumerate(trials)}
    selected = sorted(
        table,
        key=lambda row: (
            -row["mean_macro_recall"],
            -row["mean_top1"],
            order[row["trial"]],
        ),
    )[0]
    payload = {
        "protocol_sha256": protocol["protocol_sha256"],
        "selection_metric": protocol["stage3"]["selection_metric"],
        "trials": table,
        "selected_trial": selected["trial"],
        "result_file_sha256": source_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
