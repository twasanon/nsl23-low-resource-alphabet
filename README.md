# Low-resource NSL alphabet recognition on NSL23

Paper:

> Low-Resource Nepali Sign Language Alphabet Recognition on NSL23: An Exploratory Cross-Signer Study of RGB and Hand-Landmark Models

[Read the PDF](manuscript/nsl23_low_resource.pdf) · [Open the Colab notebook](https://colab.research.google.com/github/twasanon/nsl23-low-resource-alphabet/blob/main/NSL23_Low_Resource_Colab.ipynb)

This repository is a secondary analysis of the public **NSL23** alphabet videos (Sunuwar, Borah, and Kharga, 2024). The recordings are not ours. Download them from [Zenodo record 10478554](https://zenodo.org/records/10478554) (DOI [10.5281/zenodo.10478554](https://doi.org/10.5281/zenodo.10478554), CC BY 4.0). Git does not contain the zips, extracted `.MOV` files, or the RGB frame cache.

**Current direction:** [`PAPER_DIRECTION.md`](PAPER_DIRECTION.md). The PDF below is a Version 1 exploratory report; its 59.4% S1/S2 result is development evidence because those signers informed follow-on model choices. Do not train Version 2 until the annotation protocol and artifacts are frozen. If you are coordinating manual review, start with [`annotations/v2/ANNOTATION_GUIDE.md`](annotations/v2/ANNOTATION_GUIDE.md).

## What the paper claims

On 375 original isolated clips, with signers S1 and S2 held out in turn (196 pooled test clips per seed):

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Single-frame CNN | 5.8% ± 0.8% | 3.5% ± 0.2% |
| CNN–BiLSTM | 7.0% ± 2.1% | 4.2% ± 2.1% |
| CNN–Transformer | 6.3% ± 0.8% | 4.4% ± 1.3% |
| Landmark MLP (xyz) | 18.0% ± 2.1% | 11.1% ± 1.8% |
| Landmark BiLSTM (xyz, softmax) | 24.8% ± 4.0% | 16.8% ± 3.8% |
| Hand-crop CNN | 19.2% ± 1.3% | 15.9% ± 0.7% |
| xyz+angle BiLSTM (softmax) | 37.9% ± 2.4% | 30.1% ± 2.6% |
| xyz Proto-BiLSTM | 55.8% ± 1.8% | 50.6% ± 2.4% |
| **xyz+angle Proto-BiLSTM** | **59.4% ± 3.3%** | **54.7% ± 4.2%** |

Chance for 49 classes is 2.0%. The prototypical encoder’s consonant accuracy is 61.8% ± 5.7% (the xyz softmax BiLSTM was 10.0% ± 3.9% on consonants). Closed 1-shot enrollment on the test signer’s other lighting is 76.2%–78.2% for the softmax BiLSTMs; that is a tutor protocol, not the unseen-signer exam. On the same prototypical encoder, top-3 is 85.5% ± 6.0% and a 5-way 1-shot protocol is 90.6% ± 2.5% (chance 20%). Those are different exams.

A CPU webcam loop (`scripts/webcam_demo.py`) with the prototypical encoder processed a held-out NSL23 stream at **39.6 FPS** on an Apple M4; earlier xyz-softmax streams were **33–42 FPS**. That is a speed measurement, not a 49-class streaming accuracy.

Do not quote 93.5%, “state of the art,” 5-way ~91% as 49-way alphabet accuracy, or other papers’ percentages as if they were scores on these folds. Do not headline `enroll_replace_class`.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
export PYTHONPATH=src
python scripts/validate_primary_claims.py
```

`mediapipe==0.10.14` is required. Version 1.0.x crashed on the development Mac.

## Webcam demo

Needs a camera, or any video file:

```bash
export PYTHONPATH=src
python scripts/webcam_demo.py \
  --checkpoint artifacts/runs/improvements/heldout_s1/proto_bilstm/seed_42/best_checkpoint.pt \
  --camera 0
```

Held-out prototypical checkpoint for S1 is `heldout_s1/proto_bilstm/...`; for S2 use `heldout_s2/...`. The xyz softmax control remains under `artifacts/runs/primary/`. Press `q` to quit. `--video path.MOV --no-window` runs a stream without a GUI.

## Retrain (optional)

RGB frame caches are local-only (`data/frame_cache_primary/`, ~0.9 GB) and need the Zenodo archives. Landmark caches for the 375 primary clips **are** in git (`data/landmark_cache_primary/`).

```bash
export PYTHONPATH=src
python scripts/run_landmark_experiments.py          # skips completed runs
python scripts/run_improvement_experiments.py       # angles + crop CNN
python scripts/train_protonet.py --samples data/manifests/folds/heldout_s1_cached.csv --output-dir artifacts/runs/improvements/heldout_s1/proto_bilstm/seed_17 --seed 17
python scripts/evaluate_enrollment.py --checkpoint ... --manifest ... --output ...
python scripts/summarize_primary_results.py
python scripts/generate_publication_figures.py
python scripts/evaluate_alternative_protocols.py
python scripts/generate_improvement_figures.py
python scripts/write_results_summary.py
python scripts/write_latex_tables.py
python scripts/webcam_demo.py --checkpoint ... --video ... --no-window --max-frames 90
```

Compile the paper with [Tectonic](https://tectonic-typesetting.github.io/):

```bash
cd manuscript && tectonic -X compile nsl23_low_resource.tex
```

## Layout

- `manuscript/nsl23_low_resource.tex` — paper
- `PROTOCOL.md` — frozen sample, folds, and landmark addendum
- `CLAIM_LEDGER.md` — what the text is allowed to say
- `NSL23_Low_Resource_Colab.ipynb` — tables + landmark replay
- `src/nsl23/` — dataset, models, training
- `artifacts/results/` — 30-run summaries
- `artifacts/runs/primary/` — checkpoints and per-run predictions
- `VERSION2.md` — prospective annotation, split, model-selection, and final-evaluation protocol
- `annotations/v2/ANNOTATION_GUIDE.md` — simple human-review instructions and role separation

## Citation of the dataset

Sunuwar, J., Borah, S., & Kharga, A. (2024). NSL23 dataset for alphabets of Nepali Sign Language. *Data in Brief, 53*, 110080. https://doi.org/10.1016/j.dib.2024.110080
