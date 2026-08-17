#!/usr/bin/env python3
"""Statically validate the generated local/Colab reproduction notebook."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import nbformat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path, nargs="?", default=Path("NSL23_CNN_Transformer_Reproducibility.ipynb"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/qa/reproducibility_notebook_audit.json"))
    args = parser.parse_args()

    notebook = nbformat.read(args.notebook, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    markdown_cells = [cell.source for cell in notebook.cells if cell.cell_type == "markdown"]
    for index, source in enumerate(code_cells):
        ast.parse(source, filename=f"notebook_code_cell_{index}")
    content = "\n".join([*markdown_cells, *code_cells])

    required = (
        "https://zenodo.org/records/10478554",
        "scripts/verify_primary_protocol.py",
        "scripts/run_primary_experiments.py",
        "scripts/summarize_primary_results.py",
        "scripts/generate_publication_figures.py",
        "scripts/run_runtime_benchmarks.py",
        "RUN_FULL_MATRIX = False",
        "RUN_DOWNLOAD = False",
        "RUN_PREPARE = False",
        "98 test clips",
        "two held-out signers",
    )
    missing = [value for value in required if value not in content]
    if missing:
        raise AssertionError(f"Notebook is missing required content: {missing}")
    forbidden = ("<YOUR", "93.5%", "github.com/")
    present = [value for value in forbidden if value.lower() in content.lower()]
    if present:
        raise AssertionError(f"Notebook contains prohibited placeholder/claim content: {present}")
    if any(cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code"):
        raise AssertionError("Distributed notebook must not contain stale cell outputs")

    report = {
        "status": "pass",
        "notebook": str(args.notebook),
        "cells": len(notebook.cells),
        "code_cells": len(code_cells),
        "markdown_cells": len(markdown_cells),
        "all_code_cells_parse": True,
        "schema_valid": True,
        "stale_outputs": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
