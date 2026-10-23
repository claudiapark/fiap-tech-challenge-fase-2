import argparse
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_MINIMUM_SCORE = 0.50

# Métricas obrigatórias, expressas entre 0.0 e 1.0.
REQUIRED_METRICS = (
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc",
    "test_pr_auc",
    "cv_pr_auc_mean",
    "cv_precision_mean",
    "cv_recall_mean",
    "cv_f1_mean",
    "cv_roc_auc_mean",
)


def validate_report(
    report: dict[str, Any],
    minimum_score: float = DEFAULT_MINIMUM_SCORE,
) -> list[str]:
    """Return quality-gate violations; an empty list means the model passed."""
    if not math.isfinite(minimum_score) or not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be between 0.0 and 1.0")

    champion = report.get("champion", {})
    candidates = report.get("candidates", {})
    champion_name = champion.get("name")
    candidate_metrics = candidates.get(champion_name, {})
    metrics = {**candidate_metrics, **champion}

    violations: list[str] = []

    for metric in REQUIRED_METRICS:
        value = metrics.get(metric)

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            violations.append(f"{metric}: metric missing or invalid")
        elif not 0.0 <= value <= 1.0:
            violations.append(f"{metric}: value {value} is outside [0, 1]")
        elif value < minimum_score:
            violations.append(f"{metric}: {value:.3f} is below {minimum_score:.0%}")

    return violations


def check_report(
    report_path: Path,
    minimum_score: float = DEFAULT_MINIMUM_SCORE,
) -> None:
    """Read the report and fail if the model does not pass the quality gate."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    violations = validate_report(report, minimum_score)

    if violations:
        details = "\n - ".join(violations)
        raise SystemExit(
            f"MODEL QUALITY GATE FAILED: minimum allowed metric is "
            f"{minimum_score:.0%}.\n - {details}\n"
            "Model promotion has been blocked."
        )

    print(f"MODEL QUALITY GATE PASSED: all required metrics are at least {minimum_score:.0%}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--minimum-score",
        type=float,
        default=DEFAULT_MINIMUM_SCORE,
    )
    args = parser.parse_args()
    check_report(args.report, args.minimum_score)


if __name__ == "__main__":
    main()
