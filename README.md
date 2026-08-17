# Low-resource NSL alphabet recognition on NSL23

Working title:

> Low-Resource Nepali Sign Language Alphabet Recognition on NSL23: Comparing RGB Video Models and Hand Landmarks for Webcam Use

This repository studies isolated Nepali Sign Language (NSL) alphabet recognition on the public **NSL23** video dataset (Sunuwar, Borah, and Kharga, 2024). The videos are not ours; download them from [Zenodo record 10478554](https://zenodo.org/records/10478554) (DOI [10.5281/zenodo.10478554](https://doi.org/10.5281/zenodo.10478554), CC BY 4.0).

## Current status

RGB video models (single-frame MobileNetV2, CNN–BiLSTM, CNN–Transformer) have been trained and evaluated under a frozen signer-held-out protocol. Hand-landmark models, a webcam demo, the LaTeX manuscript, and a public Colab notebook are in progress. See `PAPER_DIRECTION.md` and `PROTOCOL.md`.

Do not treat the RGB accuracies (~6–7% on unseen signers) as a finished webcam system. They are the full-image control.

## Setup

NSL23 archives belong in `data/raw_archives/`. Do not commit those files.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
export PYTHONPATH=src
python scripts/verify_primary_protocol.py --verify-archive-hashes
```

Full reproduction steps are in `README_REPRODUCIBILITY.md`.

## Citation of the dataset

Sunuwar, J., Borah, S., & Kharga, A. (2024). NSL23 dataset for alphabets of Nepali Sign Language. *Data in Brief, 53*, 110080. https://doi.org/10.1016/j.dib.2024.110080
