"""Train, compare, evaluate, and register purchase propensity models."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from purchase_propensity.modeling.candidates import ModelCandidate, build_candidates
from purchase_propensity.modeling.metrics import (
    CV_SCORING,
    classification_metrics,
    summarize_cross_validation,
)
from purchase_propensity.modeling.registry import register_champion
from purchase_propensity.schema import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """Fitted candidate and its auditable experiment metadata."""

    name: str
    model: Pipeline
    metrics: dict[str, float]
    run_id: str
    model_uri: str


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load one prepared split with a stable feature order."""
    dataset = pd.read_csv(path)
    return dataset.loc[:, FEATURE_COLUMNS], dataset[TARGET_COLUMN].astype(int)


def configure_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Configure a database-backed MLflow experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_candidate(
    candidate: ModelCandidate,
    model: Pipeline,
    metrics: dict[str, float],
    features: pd.DataFrame,
) -> tuple[str, str]:
    """Persist one comparable model experiment in MLflow."""
    predictions = model.predict(features.head(5))
    signature = infer_signature(features.head(5), predictions)
    with mlflow.start_run(run_name=candidate.name) as run:
        mlflow.log_params({"model": candidate.name, **candidate.parameters})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"stage": "candidate", "primary_metric": "cv_pr_auc_mean"})
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=features.head(5),
            serialization_format="cloudpickle",
        )
    return run.info.run_id, model_info.model_uri


def evaluate_candidate(
    candidate: ModelCandidate,
    features: pd.DataFrame,
    target: pd.Series,
    cross_validator: StratifiedKFold,
) -> CandidateResult:
    """Cross-validate, fit, and track one model candidate."""
    model = clone(candidate.estimator)
    scores = cross_validate(
        model,
        features,
        target,
        cv=cross_validator,
        scoring=CV_SCORING,
        n_jobs=-1,
    )
    metrics = summarize_cross_validation(scores)
    model.fit(features, target)
    run_id, model_uri = log_candidate(candidate, model, metrics, features)
    return CandidateResult(candidate.name, model, metrics, run_id, model_uri)


def select_champion(results: list[CandidateResult], primary_metric: str) -> CandidateResult:
    """Select the candidate with the highest validation metric."""
    if any(primary_metric not in result.metrics for result in results):
        raise ValueError(f"Unknown primary metric: {primary_metric}")
    return max(results, key=lambda result: result.metrics[primary_metric])


def evaluate_holdout(
    champion: CandidateResult, test_features: pd.DataFrame, test_target: pd.Series
) -> dict[str, float]:
    """Evaluate only the selected model on the untouched holdout split."""
    predictions = champion.model.predict(test_features)
    probabilities = champion.model.predict_proba(test_features)[:, 1]
    return classification_metrics(
        test_target.to_numpy(), predictions.astype(int), probabilities.astype(float)
    )


def log_champion_metrics(champion: CandidateResult, metrics: dict[str, float]) -> None:
    """Attach final holdout evidence to the selected candidate run."""
    with mlflow.start_run(run_id=champion.run_id):
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"selected": "true", "stage": "champion"})


def build_report(
    results: list[CandidateResult],
    champion: CandidateResult,
    test_metrics: dict[str, float],
    train_target: pd.Series,
    test_target: pd.Series,
) -> dict[str, Any]:
    """Create the machine-readable DVC metrics report."""
    return {
        "dataset": {
            "train_rows": int(len(train_target)),
            "test_rows": int(len(test_target)),
            "train_purchase_rate": float(train_target.mean()),
            "test_purchase_rate": float(test_target.mean()),
        },
        "candidates": {result.name: result.metrics for result in results},
        "champion": {
            "name": champion.name,
            **test_metrics,
        },
    }


def save_outputs(
    model: Pipeline, report: dict[str, Any], model_path: Path, report_path: Path
) -> None:
    """Persist the champion model and final metrics."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the controlled experiment and model registration workflow."""
    train_features, train_target = load_split(args.train)
    test_features, test_target = load_split(args.test)
    configure_mlflow(args.tracking_uri, args.experiment_name)
    cross_validator = StratifiedKFold(args.cv_folds, shuffle=True, random_state=args.random_state)
    results = [
        evaluate_candidate(candidate, train_features, train_target, cross_validator)
        for candidate in build_candidates(args.random_state)
    ]
    champion = select_champion(results, args.primary_metric)
    test_metrics = evaluate_holdout(champion, test_features, test_target)
    log_champion_metrics(champion, test_metrics)
    register_champion(champion.model_uri, args.registered_model_name, args.model_alias)
    report = build_report(results, champion, test_metrics, train_target, test_target)
    save_outputs(champion.model, report, args.model_output, args.metrics_output)
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--metrics-output", required=True, type=Path)
    parser.add_argument("--random-state", required=True, type=int)
    parser.add_argument("--cv-folds", required=True, type=int)
    parser.add_argument("--primary-metric", default="cv_pr_auc_mean")
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--registered-model-name", required=True)
    parser.add_argument("--model-alias", required=True)
    return parser.parse_args()


def main() -> None:
    """Run the training command and print the resulting report."""
    report = run_training(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
