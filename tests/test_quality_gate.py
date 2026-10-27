import pytest

from purchase_propensity.modeling.quality_gate import validate_report


@pytest.fixture
def passing_report() -> dict:
    return {
        "champion": {
            "name": "random_forest",
            "test_accuracy": 0.86,
            "test_precision": 0.54,
            "test_recall": 0.80,
            "test_f1": 0.64,
            "test_roc_auc": 0.91,
            "test_pr_auc": 0.70,
        },
        "candidates": {
            "random_forest": {
                "cv_pr_auc_mean": 0.72,
                "cv_precision_mean": 0.56,
                "cv_recall_mean": 0.83,
                "cv_f1_mean": 0.67,
                "cv_roc_auc_mean": 0.92,
            }
        },
    }


def test_passes_when_all_metrics_are_at_least_50_percent(
    passing_report: dict,
) -> None:
    assert validate_report(passing_report, 0.50) == []


def test_fails_when_a_metric_is_below_50_percent(
    passing_report: dict,
) -> None:
    passing_report["champion"]["test_f1"] = 0.49

    violations = validate_report(passing_report, 0.50)

    assert len(violations) == 1
    assert "test_f1" in violations[0]
    assert "below 50%" in violations[0]


def test_fails_when_a_required_metric_is_missing(
    passing_report: dict,
) -> None:
    del passing_report["champion"]["test_precision"]

    violations = validate_report(passing_report, 0.50)

    assert any("test_precision" in item for item in violations)


def test_rejects_invalid_threshold(passing_report: dict) -> None:
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        validate_report(passing_report, 1.5)


def test_fails_when_a_metric_is_not_finite(
    passing_report: dict,
) -> None:
    passing_report["champion"]["test_f1"] = float("nan")

    violations = validate_report(passing_report, 0.50)

    assert any("test_f1" in item for item in violations)
