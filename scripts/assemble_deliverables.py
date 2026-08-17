#!/usr/bin/env python3
"""Centralize final user-facing files and emit portable SHA-256 checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("deliverables"))
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    copies = {
        root / "NSL23_CNN_Transformer_Reproducibility.ipynb": output_dir / "NSL23_CNN_Transformer_Reproducibility.ipynb",
        root / "artifacts/results/RESULTS_SUMMARY.md": output_dir / "NSL23_RESULTS_SUMMARY.md",
        root / "artifacts/audit/analysis_provenance.json": output_dir / "NSL23_ANALYSIS_PROVENANCE.json",
    }
    for source, destination in copies.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)

    expected = (
        output_dir / "NSL23_NCCSRJ_Blinded_Manuscript.docx",
        output_dir / "NSL23_NCCSRJ_Cover_Page.docx",
        output_dir / "NSL23_CNN_Transformer_Reproducibility.ipynb",
        output_dir / "NSL23_reproducibility_bundle.zip",
        output_dir / "NSL23_RESULTS_SUMMARY.md",
        output_dir / "NSL23_ANALYSIS_PROVENANCE.json",
    )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Final deliverables are missing: {missing}")
    records = [
        {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in expected
    ]
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }
    manifest_path = output_dir / "DELIVERABLES.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    sums_path = output_dir / "SHA256SUMS.txt"
    sums_path.write_text("".join(f"{record['sha256']}  {record['file']}\n" for record in records), encoding="utf-8")
    print(json.dumps({"status": "pass", "output_dir": str(output_dir), "files": len(records)}, indent=2))


if __name__ == "__main__":
    main()
