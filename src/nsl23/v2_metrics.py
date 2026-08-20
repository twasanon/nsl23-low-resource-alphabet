"""Locked hierarchical metrics for Version 2 development and Stage 4."""

from __future__ import annotations

import statistics

import numpy as np


def hierarchical_classification_metrics(
    rows: list[dict[str, str]], correct: np.ndarray, labels: list[str]
) -> dict[str, object]:
    if len(rows) != len(correct):
        raise ValueError("Rows and predictions differ in length")
    per_signer = {}
    for signer in sorted({row["signer"] for row in rows}, key=lambda value: int(value[1:])):
        class_recall = {}
        signer_mask = np.array([row["signer"] == signer for row in rows])
        for label in labels:
            mask = np.array(
                [row["signer"] == signer and row["canonical_label"] == label for row in rows]
            )
            if not mask.any():
                raise ValueError(f"Evaluation signer {signer} has no clip for {label}")
            class_recall[label] = float(correct[mask].mean())
        per_signer[signer] = {
            "macro_recall": statistics.fmean(class_recall.values()),
            "top1": float(correct[signer_mask].mean()),
            "clips": int(signer_mask.sum()),
            "per_class_recall": class_recall,
        }
    if not per_signer:
        raise ValueError("No evaluation signers")
    return {
        "primary_equal_signer_macro_recall": statistics.fmean(
            value["macro_recall"] for value in per_signer.values()
        ),
        "clip_pooled_top1": float(correct.mean()),
        "per_signer": per_signer,
    }


def mean_and_sample_sd(values: list[float]) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("At least two values are required for a sample SD")
    return {"mean": statistics.fmean(values), "sample_sd": statistics.stdev(values)}
