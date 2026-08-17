# NSL23 CNN Temporal Benchmark: Reproducibility Guide

This project implements the frozen protocol described in `PROTOCOL.md`. It compares a middle-frame MobileNetV2 classifier, a CNN–BiLSTM, and a CNN–Transformer using only original individually labelled NSL23 clips. The primary evaluation holds out S1 and S2 in separate complete-alphabet folds and repeats each architecture with seeds 17, 42, and 73.

## Scientific boundary

- Primary data: 375 original isolated clips, 49 classes, 7 available training/test signers.
- Excluded: 23 aggregate recordings without released gesture timestamps and 234 cropped/preprocessed files (two exclusion categories overlap).
- Test design: two unseen signers, 98 clips per signer, one bright and one dark clip per class.
- Selection: fixed 30th epoch; no validation set, early stopping, or test-driven tuning.
- Generalization: the results apply to these two held-out participants and recording conditions, not the signer population, continuous signing, or deployment.

`CLAIM_LEDGER.md` maps every permitted manuscript claim to its evidence and lists prohibited overclaims.

## Environment

The reference run used Python 3.11, PyTorch with the Apple Metal Performance Shaders backend, and the pinned packages in `requirements.txt`. PyTorch MPS was retained instead of porting the models to MLX so the same implementation, pretrained torchvision weights, checkpoints, and metrics code remain portable to Colab CUDA. A Colab-friendly dependency set that preserves Colab's CUDA-enabled PyTorch build is in `requirements-colab.txt`.

Local setup with `uv`:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

All commands below run from the project root:

```bash
export PYTHONPATH=src
```

## Official data and integrity

Download NSL23 version 2 from Zenodo record `10478554` (DOI `10.5281/zenodo.10478554`). Place the four ZIP archives in `data/raw_archives/` and `Details Dataset.docx` in the project root. The authoritative byte lengths and MD5 values are stored in `artifacts/audit/archive_checksums.json`.

After extraction below `data/extracted/`, verify the local reference data, cached tensors, fold membership, and archive hashes:

```bash
python scripts/verify_primary_protocol.py --verify-archive-hashes
```

The verification is intentionally strict: it asserts 630 decodable source videos, 375 eligible primary clips, 23 aggregate exclusions, 234 cropped/preprocessed flags, 375 distinct frame caches, exact fold sizes, full 49-class coverage, and archive hashes.

## Rebuild the analysis inputs

Build a lossless source manifest:

```bash
python -m nsl23.manifest data/extracted \
  --output data/manifests/nsl23_manifest_all.csv \
  --summary artifacts/audit/nsl23_manifest_summary.json
```

Build the two frozen signer folds:

```bash
python -m nsl23.folds \
  --manifest data/manifests/nsl23_manifest_all.csv \
  --output-dir data/manifests/folds \
  --summary artifacts/audit/isolated_twofold_summary.json
```

Cache 16 aspect-preserving, letterboxed frame-bin centers per clip and update each fold manifest:

```bash
python -m nsl23.frame_cache data/manifests/folds/heldout_s1.csv \
  --output-dir data/frame_cache_primary \
  --updated-manifest data/manifests/folds/heldout_s1_cached.csv \
  --num-frames 16 --image-size 224

python -m nsl23.frame_cache data/manifests/folds/heldout_s2.csv \
  --output-dir data/frame_cache_primary \
  --updated-manifest data/manifests/folds/heldout_s2_cached.csv \
  --num-frames 16 --image-size 224
```

## Smoke test

The following checks every architecture's training, checkpoint, prediction, metric, and confusion-matrix export. It is not a scientific experiment:

```bash
python scripts/run_primary_experiments.py \
  --output-dir artifacts/smoke/reproduction \
  --folds heldout_s1 \
  --models single_frame_cnn cnn_bilstm cnn_transformer \
  --seeds 73 --max-epochs 1 --bootstrap-replicates 20
```

## Full primary matrix

```bash
python scripts/run_primary_experiments.py --output-dir artifacts/runs/primary
```

The runner is resumable. It skips a run only when that run's `metrics.json` exists. Each completed directory also contains its checkpoint, 30-epoch training history, per-clip predictions, per-class metrics, both confusion matrices, and environment snapshot.

Validate and aggregate all 18 runs:

```bash
python scripts/summarize_primary_results.py
python scripts/generate_publication_figures.py
```

Aggregation fails unless the full matrix is present and every run has the expected identity, test size, sample IDs, label order, and epoch count.

## Runtime benchmark

After the matrix completes:

```bash
python scripts/run_runtime_benchmarks.py
```

This pre-specified benchmark uses the seed-42 checkpoint from the S2-held-out fold for each architecture. It reports synchronized batch-one model-only latency and end-to-end original-video latency separately. It does not measure a streaming pipeline.

## Notebook

`NSL23_CNN_Transformer_Reproducibility.ipynb` mirrors this workflow for local use and Google Colab. Its destructive/expensive stages are controlled by explicit Boolean flags. In Colab, upload the accompanying compact reproducibility bundle; the notebook can then resume and checksum the official Zenodo downloads.

Regenerate and validate the notebook source:

```bash
python scripts/generate_reproducibility_notebook.py
python -c "import nbformat; n=nbformat.read('NSL23_CNN_Transformer_Reproducibility.ipynb', as_version=4); nbformat.validate(n)"
```

## Result map

- `artifacts/results/model_summary.csv`: primary pooled-out-of-fold mean, SD, and range across seeds.
- `artifacts/results/fold_summary.csv`: S1- and S2-specific summaries.
- `artifacts/results/condition_summary.csv`: bright/dark descriptive breakdown.
- `artifacts/results/sign_type_summary.csv`: consonant/vowel descriptive breakdown.
- `artifacts/results/per_class_summary.csv`: fixed-order per-class metrics.
- `artifacts/results/paired_differences_summary.csv`: descriptive seed-matched architecture differences.
- `artifacts/results/top_confusions_*.csv`: most frequent off-diagonal errors across repeated predictions.
- `artifacts/runtime/runtime_summary.csv`: model-only and end-to-end latency.
- `artifacts/figures/`: PNG (350 dpi) and vector PDF versions of publication figures.
- `artifacts/audit/protocol_freeze.json`: exact primary-input hashes recorded before full training.

The manuscript is generated from these artifacts; no table or figure contains manually entered result values.

## Rebuild the Word files

The anonymous manuscript source is `manuscript/blinded_manuscript.md`; numeric placeholders are resolved directly from the audited CSV files. Install `requirements-documents.txt`, then provide cover-only author fields at build time:

```bash
python scripts/build_nccsrj_docx.py \
  --author "Author name" \
  --affiliation "Department, institution, city, country" \
  --email "corresponding-author@example.org"
```

The script produces a metadata-anonymous article body and a separate cover sheet. Run `scripts/audit_nccsrj_docx.py` after privacy-scrubbing the anonymous body, then render both files to inspect every page before submission.
