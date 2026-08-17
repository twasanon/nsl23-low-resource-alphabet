"""Fixed-label classification metrics and stratified bootstrap intervals."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, Any]:
    labels = np.arange(num_classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )[2]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=np.float64), where=row_totals != 0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "per_class_support": support,
        "confusion_counts": matrix,
        "confusion_normalized": normalized,
    }


def stratified_bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y_true == class_index) for class_index in range(num_classes)]
    if any(len(group) == 0 for group in groups):
        missing = [index for index, group in enumerate(groups) if len(group) == 0]
        raise ValueError(f"Stratified bootstrap requires support for every class; missing {missing}")
    accuracy_values = np.empty(replicates, dtype=np.float64)
    macro_f1_values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        metrics = classification_metrics(y_true[indices], y_pred[indices], num_classes)
        accuracy_values[replicate] = metrics["accuracy"]
        macro_f1_values[replicate] = metrics["macro_f1"]
    alpha = (1.0 - confidence) / 2.0
    return {
        "accuracy": [float(np.quantile(accuracy_values, alpha)), float(np.quantile(accuracy_values, 1.0 - alpha))],
        "macro_f1": [float(np.quantile(macro_f1_values, alpha)), float(np.quantile(macro_f1_values, 1.0 - alpha))],
    }
