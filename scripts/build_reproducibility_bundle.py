#!/usr/bin/env python3
"""Build an identity-scrubbed, deterministic supplementary reproduction ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Iterable


STATIC_FILES = (
    "PAPER_DIRECTION.md",
    "VERSION2.md",
    "PROTOCOL.md",
    "CLAIM_LEDGER.md",
    "README_REPRODUCIBILITY.md",
    "README.md",
    "requirements.txt",
    "requirements-colab.txt",
    "requirements-documents.txt",
    "NSL23_CNN_Transformer_Reproducibility.ipynb",
    "configs/experiment_local_m4.json",
    "configs/experiment_improvements.json",
    "configs/isolated_twofold_protocol.json",
    "configs/nsl23_classes.json",
    "src/nsl23/__init__.py",
    "src/nsl23/benchmark.py",
    "src/nsl23/dataset.py",
    "src/nsl23/folds.py",
    "src/nsl23/frame_cache.py",
    "src/nsl23/freeze.py",
    "src/nsl23/manifest.py",
    "src/nsl23/metrics.py",
    "src/nsl23/models.py",
    "src/nsl23/paths.py",
    "src/nsl23/samples.py",
    "src/nsl23/train.py",
    "src/nsl23/v2_annotation.py",
    "src/nsl23/v2_features.py",
    "src/nsl23/v2_protocol.py",
    "scripts/freeze_protocol.py",
    "scripts/audit_reproducibility_notebook.py",
    "scripts/audit_nccsrj_docx.py",
    "scripts/build_analysis_provenance.py",
    "scripts/audit_paper_numbers.py",
    "scripts/build_nccsrj_docx.py",
    "scripts/generate_publication_figures.py",
    "scripts/generate_improvement_figures.py",
    "scripts/generate_reproducibility_notebook.py",
    "scripts/report_aggregate_exclusions.py",
    "scripts/run_primary_experiments.py",
    "scripts/run_landmark_experiments.py",
    "scripts/run_improvement_experiments.py",
    "scripts/run_runtime_benchmarks.py",
    "scripts/summarize_primary_results.py",
    "scripts/summarize_improvements.py",
    "scripts/train_protonet.py",
    "scripts/evaluate_alternative_protocols.py",
    "scripts/evaluate_enrollment.py",
    "scripts/evaluate_fusion.py",
    "scripts/validate_primary_claims.py",
    "scripts/write_latex_tables.py",
    "scripts/write_results_summary.py",
    "scripts/verify_primary_protocol.py",
    "scripts/verify_protocol_freeze.py",
    "annotations/aggregate_source_review.csv",
    "annotations/v2/segments.csv",
    "annotations/v2/tapes.csv",
    "data/manifests/nsl23_manifest_all.csv",
    "data/manifests/folds/heldout_s1.csv",
    "data/manifests/folds/heldout_s1_cached.csv",
    "data/manifests/folds/heldout_s2.csv",
    "data/manifests/folds/heldout_s2_cached.csv",
    "manuscript/blinded_manuscript.md",
    "manuscript/literature_matrix.md",
    "manuscript/nccsrj_style_tokens.json",
    "manuscript/nsl23_low_resource.tex",
    "manuscript/references.bib",
    "manuscript/generated_tables.tex",
    "manuscript/generated_improvement_tables.tex",
    "artifacts/audit/archive_checksums.json",
    "artifacts/audit/protocol_freeze_source.json",
    "artifacts/audit/aggregate_exclusions.csv",
    "artifacts/audit/aggregate_exclusions_summary.json",
    "artifacts/audit/analysis_provenance.json",
    "artifacts/audit/isolated_twofold_summary.json",
    "artifacts/audit/nsl23_manifest_summary.json",
    "artifacts/audit/primary_set_composition.json",
    "artifacts/audit/runtime_hardware.json",
    "artifacts/audit/test_access_timeline.json",
    "artifacts/audit/full_paper_audit.json",
    "docs/FULL_AUDIT_2026-08-23.md",
    "docs/FINAL_SCAN_2026-10-11.md",
    "artifacts/audit/checkpoint_replay.json",
    "artifacts/qa/reproducibility_notebook_audit.json",
)

RESULT_PATTERNS = (
    "src/nsl23/*.py",
    "scripts/*.py",
    "scripts/compile_paper.sh",
    "configs/*.json",
    "tests/test_*.py",
    "annotations/v2/*.md",
    "manuscript/README.md",
    "manuscript/generated_protocol_tables.tex",
    "manuscript/figures/*.pdf",
    "artifacts/results/*.csv",
    "artifacts/results/*.json",
    "artifacts/results/*.md",
    "artifacts/results/improvements/*.csv",
    "artifacts/results/improvements/*.json",
    "artifacts/runs/primary/**/metrics.json",
    "artifacts/runs/primary/**/test_predictions.csv",
    "artifacts/runs/primary/**/training_history.csv",
    "artifacts/runs/improvements/**/metrics.json",
    "artifacts/runs/improvements/**/test_predictions.csv",
    "artifacts/runs/improvements/**/training_history.csv",
    "artifacts/runtime/runtime_summary.csv",
    "artifacts/figures/*.png",
    "artifacts/figures/*.pdf",
)

TEXT_SUFFIXES = {".md", ".txt", ".py", ".json", ".csv", ".ipynb", ".tex", ".bib"}
FORBIDDEN_TEXT = (
    "/Users/" + "sw" + "at",
    "Shas" + "wat",
    "Chap" + "again",
    "swat" + "legacy" + "@gmail.com",
    "Shas" + "wats-Mac-mini",
)

PUBLIC_REPLACEMENTS = (
    (FORBIDDEN_TEXT[0], "[local-home]"),
    (FORBIDDEN_TEXT[1], "Anonymous"),
    (FORBIDDEN_TEXT[2], "Author"),
    (FORBIDDEN_TEXT[3], "corresponding-author@example.org"),
    (FORBIDDEN_TEXT[4], "anonymized-host"),
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect(root: Path, require_results: bool) -> list[Path]:
    paths = [root / relative for relative in STATIC_FILES]
    for pattern in RESULT_PATTERNS:
        matches = sorted(root.glob(pattern))
        if require_results and not matches:
            raise FileNotFoundError(f"Required result pattern has no matches: {pattern}")
        paths.extend(matches)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Bundle inputs are missing: {missing}")
    unique = {path.resolve(): path for path in paths}
    return sorted(unique.values(), key=lambda path: str(path.relative_to(root)))


def public_protocol_freeze(root: Path) -> bytes:
    payload = json.loads((root / "artifacts/audit/protocol_freeze.json").read_text(encoding="utf-8"))
    payload.pop("host", None)
    payload.pop("platform", None)
    payload["privacy_note"] = "Host identity and local platform strings were removed from the blinded supplementary bundle."
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def reject_identity(path: str, data: bytes) -> None:
    if Path(path).suffix.lower() not in TEXT_SUFFIXES:
        return
    text = data.decode("utf-8")
    found = [value for value in FORBIDDEN_TEXT if value.lower() in text.lower()]
    if found:
        raise ValueError(f"Identity-sensitive string(s) {found} found in bundle member {path}")


def public_member_data(root: Path, path: Path, data: bytes) -> bytes:
    """Remove host-local prefixes and named-author strings from public text members."""
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return data
    text = data.decode("utf-8")
    text = text.replace(f"{root.as_posix()}/", "")
    if path.as_posix().endswith("manuscript/nsl23_low_resource.tex"):
        text = re.sub(
            r"\\author\{.*?\}\s*\n\s*\\date",
            lambda _: "\\author{Anonymous}\n\n\\date",
            text,
            flags=re.DOTALL,
        )
    for private, replacement in PUBLIC_REPLACEMENTS:
        text = re.sub(re.escape(private), replacement, text, flags=re.IGNORECASE)
    return text.encode("utf-8")


def write_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("deliverables/NSL23_reproducibility_bundle.zip"))
    parser.add_argument("--allow-missing-results", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    paths = collect(root, require_results=not args.allow_missing_results)

    members: list[tuple[str, bytes]] = []
    for path in paths:
        name = str(path.relative_to(root))
        data = public_member_data(root, path, path.read_bytes())
        reject_identity(name, data)
        members.append((name, data))
    freeze_name = "artifacts/audit/protocol_freeze.json"
    freeze_data = public_protocol_freeze(root)
    reject_identity(freeze_name, freeze_data)
    members.append((freeze_name, freeze_data))
    members.sort(key=lambda item: item[0])

    manifest = {
        "schema_version": 1,
        "source_dataset": "NSL23 version 2, DOI 10.5281/zenodo.10478554 (dataset not redistributed)",
        "members": [{"path": name, "bytes": len(data), "sha256": digest(data)} for name, data in members],
    }
    manifest_data = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    members.append(("BUNDLE_MANIFEST.json", manifest_data))

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(members):
            write_member(archive, name, data)

    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP integrity failure at {bad}")
        if set(archive.namelist()) != {name for name, _ in members}:
            raise RuntimeError("ZIP member inventory mismatch")
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(output),
                "bytes": output.stat().st_size,
                "sha256": digest(output.read_bytes()),
                "members": len(members),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
