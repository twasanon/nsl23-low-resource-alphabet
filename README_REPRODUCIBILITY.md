# NSL23 CNN Temporal Benchmark: Reproducibility Guide

This guide reproduces the frozen Version 1 protocol described in `PROTOCOL.md`. The 30-run primary matrix compares three frozen-RGB models and two wrist-relative landmark controls using only original individually labelled NSL23 clips. A separate 30-run follow-on matrix evaluates hand crops, joint angles, and prototypical training. Each run holds out S1 or S2 and repeats each architecture with seeds 17, 42, and 73. Because the same S1/S2 folds informed the follow-on representation and loss choices, all resulting percentages are exploratory cross-signer development evidence, not an untouched confirmatory test. The prospective Version 2 route is in `VERSION2.md`.

## Scientific boundary

- Primary data: 375 original isolated clips, 49 classes, 7 available training/test signers.
- Excluded: 23 aggregate recordings without released gesture timestamps and 234 cropped/preprocessed files (two exclusion categories overlap).
- Evaluation design: two run-wise unseen signers, 98 clips per signer, one bright and one dark clip per class.
- Selection: fixed 30th epoch; no validation set, early stopping, or test-driven tuning.
- Generalization: the results apply to these two held-out participants and recording conditions, not the signer population, continuous signing, or deployment.
- Study-level boundary: S1/S2 informed follow-on analyses, so Version 1 must not be described as untouched confirmation.

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

The active manifests contain repository-relative paths, so the checks work after cloning or moving the project. After the four ZIPs have passed their hashes and all 630 extracted files are present, `python scripts/manage_workspace_storage.py --remove-verified-archives` can remove the duplicate ZIP copies and record a recovery receipt. They remain downloadable from the recorded DOI.

Verify the immutable V1 snapshot against its exact historical Git tree:

```bash
python scripts/verify_protocol_freeze.py
# optional diagnostic: expected to report later V2/current-tree drift
python scripts/verify_protocol_freeze.py --mode worktree
```

All 15 frozen inputs match commit `0eb1ff2` byte-for-byte. The dated snapshot is the chronology record; tag `v1-protocol-freeze` is a human-readable pointer to that tree.

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
python scripts/run_landmark_experiments.py --output-dir artifacts/runs/primary
```

The RGB runner covers three models; the landmark runner covers two. Both are resumable and skip a run when its `metrics.json` exists. The supplied `primary` directory holds the reference results. To actually retrain rather than inspect them, give both runners the same **fresh** output directory and pass that directory to `summarize_primary_results.py --run-dir`, with a fresh summary `--output-dir`. Each completed run also contains its checkpoint, 30-epoch training history, per-clip predictions, per-class metrics, both confusion matrices, and environment snapshot.

Validate and aggregate all 30 runs (five models × two folds × three seeds):

```bash
python scripts/summarize_primary_results.py
python scripts/generate_publication_figures.py
```

Aggregation fails unless the full matrix is present and every run has the expected identity, test size, sample IDs, label order, and epoch count.

The compact bundle deliberately excludes weights but includes all per-run predictions and training histories. To aggregate that exported evidence, explicitly add `--allow-missing-checkpoints`; this audits reported outcomes, not weight-level reproducibility. Without that flag, the aggregator requires a checkpoint in each local run directory and does not depend on historical absolute paths in the metrics receipts.

## Runtime benchmark

After the matrix completes:

```bash
python scripts/run_runtime_benchmarks.py
```

This pre-specified benchmark uses the seed-42 checkpoint from the S2-held-out fold for each architecture. It reports synchronized batch-one model-only latency and end-to-end original-video latency separately. It does not measure a streaming pipeline.

## Follow-on exploratory matrix

The follow-on experiments reuse the same two folds and three seeds. They are model-development evidence, not a second independent test. The cross-entropy angle/crop runner can be resumed with the commands below. Prototypical training refuses an output directory containing saved run artifacts; use a fresh directory for a new repeat rather than overwriting reference checkpoints.

```bash
python scripts/run_improvement_experiments.py
# Run train_protonet.py once per fold/seed for raw_angle and xyz, as listed in PROTOCOL.md.
python scripts/summarize_improvements.py
python scripts/evaluate_alternative_protocols.py
python scripts/generate_improvement_figures.py
python scripts/write_latex_tables.py
```

The summary code pools the 98 S1 and 98 S2 predictions within each seed before computing nonlinear metrics such as macro-F1. It then reports the mean and sample SD over seeds 17, 42, and 73.

## Notebook

`NSL23_CNN_Transformer_Reproducibility.ipynb` mirrors this workflow for local use and Google Colab. Its destructive/expensive stages are controlled by explicit Boolean flags. In Colab, upload the accompanying compact reproducibility bundle; the notebook can then resume and checksum the official Zenodo downloads.

The notebook's full matrix covers the **30 primary runs**, not the separate follow-on experiments, and writes fresh reproductions outside the reference run directory. `NSL23_Low_Resource_Colab.ipynb` is an older table-oriented notebook; `NSL23_CNN_Transformer_Colab_Benchmarks_NSLS23Robust.ipynb` is a legacy experiment. Neither is the current reproduction entry point. The current notebook is schema- and syntax-checked; this scan did not execute its downloads or training cells.

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
- `artifacts/audit/protocol_freeze_source.json`: independently reconstructed binding from that snapshot to the matching Git tree and tag.
- `artifacts/audit/full_paper_audit.json`: machine-readable recomputation of primary, follow-on, fusion, enrollment, alternative-protocol, dataset, V2-gate, and bibliography checks.
- `artifacts/results/improvements/`: follow-on and secondary-exam summaries; the compact ZIP also contains the per-run predictions needed to recompute pooled macro-F1.

The manuscript is generated from these artifacts; no table or figure contains manually entered result values.

## Final integrity suite and compact bundle

```bash
python scripts/verify_protocol_freeze.py
python scripts/verify_primary_protocol.py --verify-v1-freeze
python scripts/audit_paper_numbers.py
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/validate_primary_claims.py
python scripts/verify_saved_predictions.py --output artifacts/audit/checkpoint_replay.json
python scripts/build_analysis_provenance.py
python scripts/build_reproducibility_bundle.py
```

The provenance manifest hashes both 30-run matrices and all 60 checkpoints. The deterministic compact ZIP omits source video and checkpoints, but includes the manuscript inputs, aggregate tables, secondary-exam JSON, and all primary/follow-on per-run predictions and metrics needed for numerical verification without retraining.

## Current paper and historical Word template

The current paper is `manuscript/nsl23_low_resource.tex`; rebuild its PDF with `bash scripts/compile_paper.sh`. The human-readable companion is generated by `scripts/build_paper_explainer.py` and saved as `docs/NSL23_Paper_Explained.docx`.

`manuscript/blinded_manuscript.md` and `scripts/build_nccsrj_docx.py` are historical RGB-only templates, not current submission files. The builder refuses the old source unless `--allow-historical-template` is explicitly supplied. A future journal conversion must include the completed V2 analyses and receive numerical and page-layout checks; anonymity and formatting checks alone do not establish scientific accuracy.

The final scan and its verification limits are recorded in `docs/FINAL_SCAN_2026-10-11.md`. Checkpoint replay covers all landmark evaluation clips and a representative RGB subset, not complete RGB inference or a new training run. Original archives were removed after verification; the present scan checks the retained sources and the earlier checksum receipt rather than claiming fresh ZIP hashing.
