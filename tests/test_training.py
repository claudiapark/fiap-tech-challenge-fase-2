"""Tests for training orchestration and holdout isolation."""

from argparse import Namespace
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from purchase_propensity.modeling.candidates import build_candidates
from purchase_propensity.schema import FEATURE_COLUMNS, TARGET_COLUMN
from purchase_propensity.training import train
from purchase_propensity.training.train import CandidateResult


def fitted_result(sample_dataset: pd.DataFrame, name: str = "candidate") -> CandidateResult:
    """Create a fitted candidate result for helper tests."""
    model = build_candidates(42)[0].estimator
    model.fit(
        sample_dataset[FEATURE_COLUMNS],
        sample_dataset[TARGET_COLUMN].astype(int),
    )

    metrics = {
        "cv_pr_auc_mean": 0.70,
        "cv_pr_auc_std": 0.02,
        "cv_precision_mean": 0.70,
        "cv_precision_std": 0.02,
        "cv_recall_mean": 0.70,
        "cv_recall_std": 0.02,
        "cv_f1_mean": 0.70,
        "cv_f1_std": 0.02,
        "cv_roc_auc_mean": 0.70,
        "cv_roc_auc_std": 0.02,
    }

    return CandidateResult(
        name,
        model,
        metrics,
        "run-id",
        "runs:/run/model",
    )


def test_load_select_evaluate_and_save(tmp_path: Path, sample_dataset: pd.DataFrame) -> None:
    source = tmp_path / "split.csv"
    sample_dataset.to_csv(source, index=False)
    features, target = train.load_split(source)
    weaker = fitted_result(sample_dataset, "weaker")
    stronger_metrics = {
        **weaker.metrics,
        "cv_pr_auc_mean": 0.8,
    }
    stronger = CandidateResult(
        "stronger", weaker.model, stronger_metrics, "run-2", "runs:/run-2/model"
    )
    champion = train.select_champion([weaker, stronger], "cv_pr_auc_mean")
    metrics = train.evaluate_holdout(champion, features, target)
    report = train.build_report([weaker, stronger], champion, metrics, target, target)
    comparison = train.build_comparison_table([weaker, stronger], champion.name)
    model_path = tmp_path / "models" / "model.joblib"
    report_path = tmp_path / "reports" / "metrics.json"
    comparison_path = tmp_path / "reports" / "model_comparison.csv"
    train.save_outputs(
        champion.model, report, comparison, model_path, report_path, comparison_path
    )
    assert champion.name == "stronger"
    assert joblib.load(model_path).predict(features.head(1)).shape == (1,)
    assert '"champion"' in report_path.read_text()
    assert comparison.iloc[0]["model"] == "stronger"
    assert comparison.iloc[0]["selected_champion"]
    assert "cv_pr_auc_mean" in comparison_path.read_text()


def test_unknown_primary_metric_raises(sample_dataset: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unknown primary metric"):
        train.select_champion([fitted_result(sample_dataset)], "missing")


def test_evaluate_holdout_rejects_invalid_threshold(sample_dataset: pd.DataFrame) -> None:
    features = sample_dataset[FEATURE_COLUMNS]
    target = sample_dataset[TARGET_COLUMN].astype(int)
    with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
        train.evaluate_holdout(fitted_result(sample_dataset), features, target, 1.1)


def test_evaluate_candidate_uses_cross_validation(sample_dataset, monkeypatch) -> None:
    monkeypatch.setattr(train, "log_candidate", lambda *_: ("run", "runs:/run/model"))
    candidate = build_candidates(42)[0]
    result = train.evaluate_candidate(
        candidate,
        sample_dataset[FEATURE_COLUMNS],
        sample_dataset[TARGET_COLUMN].astype(int),
        StratifiedKFold(2, shuffle=True, random_state=42),
    )
    assert result.run_id == "run"
    assert "cv_pr_auc_mean" in result.metrics


def test_configure_and_log_champion(monkeypatch, sample_dataset) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        train.mlflow, "set_tracking_uri", lambda value: calls.append(("uri", value))
    )
    monkeypatch.setattr(train.mlflow, "set_experiment", lambda value: calls.append(("exp", value)))

    class RunContext:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(train.mlflow, "start_run", lambda **_: RunContext())
    monkeypatch.setattr(train.mlflow, "log_metrics", lambda value: calls.append(("metrics", value)))
    monkeypatch.setattr(
        train.mlflow, "log_param", lambda key, value: calls.append(("param", key, value))
    )
    monkeypatch.setattr(train.mlflow, "set_tags", lambda value: calls.append(("tags", value)))
    train.configure_mlflow("sqlite:///test.db", "experiment")
    train.log_champion_metrics(
        fitted_result(sample_dataset),
        {"test_pr_auc": 0.7, "test_decision_threshold": 0.5},
    )
    assert ("uri", "sqlite:///test.db") in calls
    assert any(call[0] == "metrics" for call in calls)
    assert ("param", "decision_threshold", 0.5) in calls


def test_run_training_orchestrates_dependencies(tmp_path, sample_dataset, monkeypatch) -> None:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_dataset.to_csv(train_path, index=False)
    sample_dataset.to_csv(test_path, index=False)
    result = fitted_result(sample_dataset)
    monkeypatch.setattr(train, "configure_mlflow", lambda *_: None)
    monkeypatch.setattr(train, "evaluate_candidate", lambda *_: result)
    monkeypatch.setattr(
        train,
        "evaluate_holdout",
        lambda *_: {
            "test_accuracy": 0.8,
            "test_precision": 0.7,
            "test_recall": 0.7,
            "test_f1": 0.7,
            "test_roc_auc": 0.8,
            "test_pr_auc": 0.7,
            "test_decision_threshold": 0.5,
        },
    )
    monkeypatch.setattr(train, "log_champion_metrics", lambda *_: None)
    monkeypatch.setattr(train, "register_champion", lambda *_: "2")
    captured: list[object] = []
    monkeypatch.setattr(train, "save_outputs", lambda *values: captured.extend(values))
    args = Namespace(
        train=train_path,
        test=test_path,
        model_output=tmp_path / "model.joblib",
        metrics_output=tmp_path / "metrics.json",
        comparison_output=tmp_path / "model_comparison.csv",
        random_state=42,
        cv_folds=2,
        primary_metric="cv_pr_auc_mean",
        threshold=0.5,
        tracking_uri="sqlite:///test.db",
        experiment_name="test",
        registered_model_name="model",
        model_alias="champion",
        minimum_metric_score=0.50,
    )
    report = train.run_training(args)
    assert report["champion"]["name"] == "candidate"
    assert captured
