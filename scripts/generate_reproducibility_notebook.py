#!/usr/bin/env python3
"""Generate the local/Colab NSL23 reproduction notebook from versioned cells."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("NSL23_CNN_Transformer_Reproducibility.ipynb"),
    )
    args = parser.parse_args()

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"name": args.output.name, "provenance": []},
    }
    notebook["cells"] = [
        markdown(
            """
# NSL23 exploratory RGB and landmark benchmark

This notebook reproduces the **frozen 30-run primary analysis** comparing a single-frame CNN, CNN–BiLSTM, CNN–Transformer, landmark MLP, and landmark BiLSTM on the public NSL23 Nepali Sign Language alphabet videos. The primary set contains only original, individually labeled clips. Aggregate recordings are excluded because the release provides no gesture timestamps, and cropped/preprocessed derivatives are excluded to avoid source duplication. The later 30-run follow-on matrix, including prototypical training, is a separate workflow documented in `README_REPRODUCIBILITY.md`; this notebook does not train it.

The evaluation uses two complete 49-class signer-held-out folds (S1 and S2), three fixed seeds, 30 fixed epochs, and no validation-driven model selection. Read `PROTOCOL.md` before running. The notebook works from the supplied project folder locally or from the accompanying `NSL23_reproducibility_bundle.zip` in Google Colab.

The study is exploratory: S1/S2 results informed the follow-on choices. Freezing these files preserves the reported analysis; it does not turn that development process into a confirmatory test.

**Compute/storage:** the official archives total about 9.6 GB and expand to about 9.7 GB. The 16-frame cache requires about 0.9 GB. A full 30-run matrix can take several hours depending on hardware.
"""
        ),
        markdown(
            """
## 1. Runtime mode and user controls

`RUN_DOWNLOAD`, `RUN_PREPARE`, `RUN_SMOKE_TEST`, and `RUN_FULL_MATRIX` are deliberately explicit and disabled by default. Full results are resumable: an existing `metrics.json` causes that run to be skipped.
"""
        ),
        code(
            """
from pathlib import Path
import os, sys, json, subprocess, platform

IN_COLAB = "google.colab" in sys.modules
PROJECT_ROOT = Path("/content/NSL23_project") if IN_COLAB else Path.cwd()
RUN_DOWNLOAD = False          # Set True only after confirming ~22 GB free disk space.
RUN_PREPARE = False           # Build manifest, signer folds, and 16-frame cache.
RUN_SMOKE_TEST = False
RUN_FULL_MATRIX = False       # 30 runs: 5 architectures × 2 folds × 3 seeds.
RUN_BENCHMARKS = False        # Requires completed checkpoints.
REPRODUCTION_RUN_DIR = Path("artifacts/runs/reproduction_primary")
RUN_DIR = REPRODUCTION_RUN_DIR if RUN_FULL_MATRIX else Path("artifacts/runs/primary")
RESULTS_DIR = Path("artifacts/results/reproduction_primary") if RUN_FULL_MATRIX else Path("artifacts/results")
FIGURES_DIR = Path("artifacts/figures/reproduction_primary") if RUN_FULL_MATRIX else Path("artifacts/figures")

print({"in_colab": IN_COLAB, "project_root": str(PROJECT_ROOT), "python": platform.python_version()})
"""
        ),
        markdown(
            """
## 2. Colab project setup

In Colab, upload `NSL23_reproducibility_bundle.zip` when prompted. This small bundle contains source code, configurations, the frozen protocol, and audit metadata—not the 10 GB dataset. Local users can skip this cell when the notebook is in the project root.
"""
        ),
        code(
            """
if IN_COLAB:
    from google.colab import files
    import zipfile
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    if not (PROJECT_ROOT / "src" / "nsl23" / "train.py").exists():
        uploaded = files.upload()
        bundle_name = next((name for name in uploaded if name.endswith(".zip")), None)
        if bundle_name is None:
            raise RuntimeError("Upload NSL23_reproducibility_bundle.zip")
        with zipfile.ZipFile(bundle_name) as archive:
            archive.extractall(PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)
else:
    os.chdir(PROJECT_ROOT)

assert Path("src/nsl23/train.py").is_file(), "Run from the project root or upload the bundle."
print("Project ready:", Path.cwd())
"""
        ),
        markdown(
            """
## 3. Install and record the environment

Colab keeps its CUDA-enabled PyTorch build; the remaining packages are pinned in `requirements-colab.txt`. The local reference environment is pinned in `requirements.txt`. Hardware and library versions are recorded inside every checkpoint and run directory.
"""
        ),
        code(
            """
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)

import torch, torchvision, numpy, pandas, sklearn, av
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
environment = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "scikit_learn": sklearn.__version__,
    "pyav": av.__version__,
    "device": device,
}
print(json.dumps(environment, indent=2))
"""
        ),
        markdown(
            """
## 4. Download and verify the official NSL23 release

The files come from Zenodo record [10478554](https://zenodo.org/records/10478554), DOI `10.5281/zenodo.10478554`, licensed CC BY 4.0. Downloads are resumed with HTTP range requests and are rejected unless both byte length and MD5 match the audited record.
"""
        ),
        code(
            """
from urllib.parse import quote
import hashlib, requests

RAW_DIR = Path("data/raw_archives")
EXTRACTED_DIR = Path("data/extracted")
RAW_DIR.mkdir(parents=True, exist_ok=True)
audit = json.loads(Path("artifacts/audit/archive_checksums.json").read_text())

def md5_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

def download_resumable(url, destination, expected_bytes):
    destination = Path(destination)
    existing = destination.stat().st_size if destination.exists() else 0
    if existing > expected_bytes:
        destination.unlink()
        existing = 0
    if existing == expected_bytes:
        return
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    mode = "ab" if existing else "wb"
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        if existing and response.status_code != 206:
            existing, mode = 0, "wb"
        with destination.open(mode) as handle:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if destination.stat().st_size != expected_bytes:
        raise IOError(f"Size mismatch for {destination.name}")

if RUN_DOWNLOAD:
    for item in audit["files"]:
        destination = Path("Details Dataset.docx") if item["name"].endswith(".docx") else RAW_DIR / item["name"]
        url = f"https://zenodo.org/records/10478554/files/{quote(item['name'])}?download=1"
        download_resumable(url, destination, int(item["bytes"]))
        observed = md5_file(destination)
        if observed != item["md5"]:
            raise IOError(f"MD5 mismatch for {destination.name}: {observed}")
        print("verified", destination.name)
else:
    print("Download disabled. Set RUN_DOWNLOAD=True when needed.")
"""
        ),
        markdown(
            """
## 5. Extract archives safely

The extraction check prevents path traversal. Re-running this cell is idempotent for already extracted files.
"""
        ),
        code(
            """
import zipfile

def safe_extract(zip_path, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        archive.extractall(destination)

if RUN_PREPARE:
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    for item in audit["files"]:
        if item["name"].endswith(".zip"):
            archive = RAW_DIR / item["name"]
            if archive.stat().st_size != int(item["bytes"]) or md5_file(archive) != item["md5"]:
                raise IOError(f"Archive is absent or unverified: {archive}")
            safe_extract(archive, EXTRACTED_DIR)
            print("extracted", archive.name)
else:
    print("Preparation disabled. Set RUN_PREPARE=True after verified downloads exist.")
"""
        ),
        markdown(
            """
## 6. Build the lossless manifest and frozen signer folds

This stage probes every video, canonicalizes the documented 49-class label order, excludes aggregate/cropped inputs from the primary set, and creates the S1- and S2-held-out folds. Each fold must contain 277 training clips and 98 test clips, with every class represented on both sides.
"""
        ),
        code(
            """
env = os.environ.copy()
env["PYTHONPATH"] = str(Path("src").resolve())

def run(command):
    print("$", " ".join(map(str, command)))
    subprocess.run(list(map(str, command)), check=True, env=env)

if RUN_PREPARE:
    run([sys.executable, "-m", "nsl23.manifest", EXTRACTED_DIR,
         "--output", "data/manifests/nsl23_manifest_all.csv",
         "--summary", "artifacts/audit/nsl23_manifest_summary.json"])
    run([sys.executable, "-m", "nsl23.folds",
         "--manifest", "data/manifests/nsl23_manifest_all.csv",
         "--output-dir", "data/manifests/folds",
         "--summary", "artifacts/audit/isolated_twofold_summary.json"])
"""
        ),
        markdown(
            """
## 7. Cache 16 frame-bin centers and hand landmarks per clip

Frames are decoded once with PyAV, sampled at temporal-bin centers, resized without distorting aspect ratio, letterboxed to 224×224, and stored as unsigned 8-bit NumPy arrays. Both fold manifests refer to the same 375 cache files. MediaPipe extraction adds the landmark-cache paths needed by the two landmark models.
"""
        ),
        code(
            """
if RUN_PREPARE:
    for fold in ["heldout_s1", "heldout_s2"]:
        run([sys.executable, "-m", "nsl23.frame_cache",
             f"data/manifests/folds/{fold}.csv",
             "--output-dir", "data/frame_cache_primary",
             "--updated-manifest", f"data/manifests/folds/{fold}_cached.csv",
             "--num-frames", "16", "--image-size", "224"])
    run([sys.executable, "scripts/extract_landmarks.py", "--manifests",
         "data/manifests/folds/heldout_s1_cached.csv",
         "data/manifests/folds/heldout_s2_cached.csv"])
    run([sys.executable, "scripts/verify_primary_protocol.py"])
"""
        ),
        markdown(
            """
## 8. One-epoch software smoke test

The smoke test checks the three RGB forward/backward paths and artifact exports, not the landmark models. Its scores are not scientific results. It uses only fold S1 and one seed, limits training to one epoch, and uses 20 bootstrap replicates to keep the check short.
"""
        ),
        code(
            """
if RUN_SMOKE_TEST:
    run([sys.executable, "scripts/run_primary_experiments.py",
         "--output-dir", "artifacts/smoke/notebook",
         "--folds", "heldout_s1",
         "--models", "single_frame_cnn", "cnn_bilstm", "cnn_transformer",
         "--seeds", "73", "--max-epochs", "1", "--bootstrap-replicates", "20"])
"""
        ),
        markdown(
            """
## 9. Full frozen experiment matrix

Do not modify the configuration after inspecting test results. The runners train each of the five primary architectures for exactly 30 epochs and evaluate the held-out signer after the final checkpoint is fixed. The class-weighted loss is computed from training counts only. New runs go into `REPRODUCTION_RUN_DIR`, separate from the reference results. Existing metrics there are skipped; choose another fresh directory for an independent repeat.
"""
        ),
        code(
            """
if RUN_FULL_MATRIX:
    run([sys.executable, "scripts/run_primary_experiments.py", "--output-dir", RUN_DIR])
    run([sys.executable, "scripts/run_landmark_experiments.py", "--output-dir", RUN_DIR])
else:
    print("Full matrix disabled. Set RUN_FULL_MATRIX=True to reproduce all 30 runs.")
"""
        ),
        markdown(
            """
## 10. Validate, aggregate, and plot results

Aggregation fails closed unless all 30 run identities, 30-epoch histories, and 98 held-out predictions per run are present. The compact bundle omits weights, so inspecting its reference results explicitly permits missing checkpoints; new training runs must have checkpoints. The two held-out folds are pooled within each seed (196 clips), then mean and sample SD are reported across the three seeds. Sample-level bootstrap intervals are descriptive and do not estimate signer-population uncertainty from only two test signers.
"""
        ),
        code(
            """
if RUN_DIR.exists() and len(list(RUN_DIR.glob("**/metrics.json"))) == 30:
    summary_command = [sys.executable, "scripts/summarize_primary_results.py", "--run-dir", RUN_DIR,
                       "--output-dir", RESULTS_DIR]
    if not RUN_FULL_MATRIX:
        summary_command.append("--allow-missing-checkpoints")
    run(summary_command)
    run([sys.executable, "scripts/generate_publication_figures.py", "--results-dir", RESULTS_DIR,
         "--output-dir", FIGURES_DIR])
    display(pandas.read_csv(RESULTS_DIR / "model_summary.csv"))
else:
    print("Complete the 30-run matrix before aggregation.")
"""
        ),
        code(
            """
from IPython.display import Image, display
for figure in sorted(FIGURES_DIR.glob("figure_[1-7]*.png")):
    print(figure.name)
    display(Image(filename=str(figure), width=900))
"""
        ),
        markdown(
            """
## 11. Batch-one runtime benchmark

The benchmark reports two explicitly synchronized quantities: model-only forward/softmax latency from a device-resident normalized tensor, and end-to-end latency from opening an original video through prediction. It uses one fixed completed checkpoint per architecture; latency is an implementation/hardware measurement, not a test-set selection criterion.
"""
        ),
        code(
            """
if RUN_BENCHMARKS:
    run([sys.executable, "scripts/run_runtime_benchmarks.py", "--run-dir", RUN_DIR,
         "--output-dir", "artifacts/runtime/reproduction_primary"])
"""
        ),
        markdown(
            """
## 12. Export a compact result archive

Checkpoints are intentionally omitted from this compact archive because they are large; the archived CSV/JSON/PDF/PNG files are sufficient to audit reported tables and figures. Keep the project folder to preserve checkpoints.
"""
        ),
        code(
            """
import shutil
if RESULTS_DIR.exists():
    archive = shutil.make_archive("NSL23_primary_results", "zip", root_dir=RESULTS_DIR)
    print("Created", archive)
"""
        ),
        markdown(
            """
## Interpretation boundary

This benchmark estimates performance on two held-out signers under the two indoor lighting conditions available as complete 49-class tests. It does **not** establish generalization to the wider signer population, prepared backgrounds, outdoor scenes, continuous signing, or real-time deployment. Aggregate recordings remain auditable source material but are outside the primary analysis until trustworthy gesture timestamps are available.
"""
        ),
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
