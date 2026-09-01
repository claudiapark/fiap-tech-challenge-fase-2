"""Tests for model construction and evaluation."""

import numpy as np
import pandas as pd
import pytest

from purchase_propensity.modeling.candidates import build_candidates
from purchase_propensity.modeling.metrics import classification_metrics, summarize_cross_validation
from purchase_propensity.schema import FEATURE_COLUMNS, TARGET_COLUMN


def test_candidates_support_probability_predictions(sample_dataset: pd.DataFrame) -> None:
    features = sample_dataset[FEATURE_COLUMNS]
    target = sample_dataset[TARGET_COLUMN].astype(int)
    for candidate in build_candidates(random_state=42):
        candidate.estimator.fit(features, target)
        probabilities = candidate.estimator.predict_proba(features.head(3))[:, 1]
        assert probabilities.shape == (3,)
        assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_classification_metrics_are_complete() -> None:
    metrics = classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
        np.array([0.1, 0.6, 0.7, 0.9]),
    )
    assert set(metrics) == {
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
        "test_pr_auc",
    }
    assert metrics["test_recall"] == 1.0


def test_cross_validation_summary_contains_mean_and_std() -> None:
    scores = {
        f"test_{metric}": np.array([0.4, 0.6])
        for metric in ("pr_auc", "roc_auc", "recall", "precision", "f1")
    }
    summary = summarize_cross_validation(scores)
    assert summary["cv_pr_auc_mean"] == 0.5
    assert summary["cv_pr_auc_std"] == pytest.approx(0.1)
