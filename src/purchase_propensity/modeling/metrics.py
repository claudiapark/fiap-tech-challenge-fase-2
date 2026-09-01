"""Metrics for imbalanced purchase propensity classification."""

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CV_SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
}


def summarize_cross_validation(results: Mapping[str, NDArray[np.float64]]) -> dict[str, float]:
    """Summarize cross-validation scores as mean and standard deviation."""
    summary: dict[str, float] = {}
    for metric in CV_SCORING:
        scores = results[f"test_{metric}"]
        summary[f"cv_{metric}_mean"] = float(np.mean(scores))
        summary[f"cv_{metric}_std"] = float(np.std(scores))
    return summary


def classification_metrics(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    y_probability: NDArray[np.float64],
) -> dict[str, float]:
    """Calculate final holdout metrics for the positive purchase class."""
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_true, y_probability)),
        "test_pr_auc": float(average_precision_score(y_true, y_probability)),
    }
