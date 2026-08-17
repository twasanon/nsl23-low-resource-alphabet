#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
python scripts/write_latex_tables.py
python scripts/generate_publication_figures.py
cp artifacts/figures/figure_1_study_flow.pdf manuscript/figures/
cp artifacts/figures/figure_2_architectures.pdf manuscript/figures/
cp artifacts/figures/figure_3_primary_performance.pdf manuscript/figures/
cp artifacts/figures/figure_4_training_curves.pdf manuscript/figures/
cp artifacts/figures/figure_5_lighting_performance.pdf manuscript/figures/
cp artifacts/figures/figure_6_landmark_bilstm_confusion.pdf manuscript/figures/
cp artifacts/figures/figure_7_landmark_bilstm_per_class_recall.pdf manuscript/figures/
(cd manuscript && tectonic -X compile nsl23_low_resource.tex)
