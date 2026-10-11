#!/usr/bin/env python3
"""Hash the complete paper-analysis lineage without embedding host-local paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PATTERNS = (
    "README.md",
    "PAPER_DIRECTION.md",
    "VERSION2.md",
    "PROTOCOL.md",
    "CLAIM_LEDGER.md",
    "README_REPRODUCIBILITY.md",
    "requirements*.txt",
    "configs/*.json",
    "src/nsl23/*.py",
    "scripts/*.py",
    "scripts/compile_paper.sh",
    "tests/test_*.py",
    "manuscript/*.tex",
    "manuscript/README.md",
    "manuscript/references.bib",
    "manuscript/nsl23_low_resource.pdf",
    "manuscript/figures/*.pdf",
    "docs/*.md",
    "docs/*.docx",
    "annotations/*.csv",
    "annotations/v2/*.csv",
    "annotations/v2/*.md",
    "data/manifests/**/*.csv",
    "artifacts/audit/archive_checksums.json",
    "artifacts/audit/aggregate_exclusions*.json",
    "artifacts/audit/aggregate_exclusions.csv",
    "artifacts/audit/isolated_twofold_summary.json",
    "artifacts/audit/nsl23_manifest_summary.json",
    "artifacts/audit/primary_set_composition.json",
    "artifacts/audit/runtime_hardware.json",
    "artifacts/audit/test_access_timeline.json",
    "artifacts/audit/full_paper_audit.json",
    "artifacts/audit/checkpoint_replay.json",
    "artifacts/audit/protocol_freeze_source.json",
    "artifacts/audit/storage_cleanup.json",
    "artifacts/runs/primary/**/metrics.json",
    "artifacts/runs/primary/**/training_history.csv",
    "artifacts/runs/primary/**/test_predictions.csv",
    "artifacts/runs/primary/**/per_class_metrics.csv",
    "artifacts/runs/primary/**/confusion_matrix_counts.csv",
    "artifacts/runs/primary/**/confusion_matrix_normalized.csv",
    "artifacts/runs/primary/**/environment.json",
    "artifacts/runs/primary/**/best_checkpoint.pt",
    "artifacts/runs/improvements/**/metrics.json",
    "artifacts/runs/improvements/**/training_history.csv",
    "artifacts/runs/improvements/**/test_predictions.csv",
    "artifacts/runs/improvements/**/per_class_metrics.csv",
    "artifacts/runs/improvements/**/confusion_matrix_counts.csv",
    "artifacts/runs/improvements/**/confusion_matrix_normalized.csv",
    "artifacts/runs/improvements/**/environment.json",
    "artifacts/runs/improvements/**/best_checkpoint.pt",
    "artifacts/results/*",
    "artifacts/results/improvements/**/*",
    "artifacts/runtime/**/*",
    "artifacts/figures/*",
    "manuscript/blinded_manuscript.md",
    "manuscript/literature_matrix.md",
    "manuscript/nccsrj_style_tokens.json",
    "NSL23_CNN_Transformer_Reproducibility.ipynb",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("artifacts/audit/analysis_provenance.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    paths: dict[str, Path] = {}
    for pattern in PATTERNS:
        for path in root.glob(pattern):
            if path.is_file() and path.resolve() != output.resolve():
                paths[str(path.relative_to(root))] = path

    primary_metrics_count = sum(
        path.startswith("artifacts/runs/primary/") and path.endswith("/metrics.json")
        for path in paths
    )
    primary_checkpoints_count = sum(
        path.startswith("artifacts/runs/primary/") and path.endswith("/best_checkpoint.pt")
        for path in paths
    )
    improvement_metrics_count = sum(
        path.startswith("artifacts/runs/improvements/") and path.endswith("/metrics.json")
        for path in paths
    )
    improvement_checkpoints_count = sum(
        path.startswith("artifacts/runs/improvements/") and path.endswith("/best_checkpoint.pt")
        for path in paths
    )
    counts = {
        "primary metrics": primary_metrics_count,
        "primary checkpoints": primary_checkpoints_count,
        "follow-on metrics": improvement_metrics_count,
        "follow-on checkpoints": improvement_checkpoints_count,
    }
    if any(value != 30 for value in counts.values()):
        raise RuntimeError(f"Expected 30 files in every run inventory, found {counts}")
    required = (
        "artifacts/results/primary_results_summary.json",
        "artifacts/results/improvements/summary.json",
        "artifacts/audit/full_paper_audit.json",
        "artifacts/runtime/runtime_summary.csv",
        "manuscript/blinded_manuscript.md",
        "NSL23_CNN_Transformer_Reproducibility.ipynb",
    )
    missing = [value for value in required if value not in paths]
    if missing:
        raise FileNotFoundError(f"Final provenance inputs are missing: {missing}")

    records = [
        {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for relative, path in sorted(paths.items())
    ]
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "NSL23 version 2, DOI 10.5281/zenodo.10478554",
        "protocol_snapshot_sha256": json.loads((root / "artifacts/audit/protocol_freeze.json").read_text())["snapshot_sha256"],
        "primary_runs": primary_metrics_count,
        "follow_on_runs": improvement_metrics_count,
        "checkpoint_files": primary_checkpoints_count + improvement_checkpoints_count,
        "files": records,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["provenance_sha256"] = hashlib.sha256(canonical).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(output), "files": len(records), "provenance_sha256": payload["provenance_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
