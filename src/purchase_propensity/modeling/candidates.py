"""Build comparable Scikit-Learn model candidates."""

from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from purchase_propensity.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    """Named estimator and auditable parameter set."""

    name: str
    estimator: Pipeline
    parameters: dict[str, Any]


def build_preprocessor() -> ColumnTransformer:
    """Create leakage-safe numeric and categorical transformations."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def build_pipeline(classifier: BaseEstimator) -> Pipeline:
    """Combine shared preprocessing and one classifier."""
    return Pipeline([("preprocess", build_preprocessor()), ("classifier", classifier)])


def build_candidates(random_state: int) -> list[ModelCandidate]:
    """Return Linear, ensemble and MLP candidates for fair comparison."""
    logistic_params = {"C": 1.0, "class_weight": "balanced", "max_iter": 1000}
    forest_params = {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 3,
        "class_weight": "balanced",
    }
    mlp_hidden_layers = (64, 32)
    mlp_params = {
        "hidden_layer_sizes": mlp_hidden_layers,
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "batch_size": 64,
        "learning_rate_init": 0.001,
        "max_iter": 300,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 15,
    }
    # Keep logged params JSON/MLflow-friendly (tuple -> string).
    mlp_logged_params = {**mlp_params, "hidden_layer_sizes": str(mlp_hidden_layers)}
    return [
        ModelCandidate(
            "logistic_regression",
            build_pipeline(LogisticRegression(random_state=random_state, **logistic_params)),
            logistic_params,
        ),
        ModelCandidate(
            "random_forest",
            build_pipeline(
                RandomForestClassifier(random_state=random_state, n_jobs=-1, **forest_params)
            ),
            forest_params,
        ),
        ModelCandidate(
            "mlp_classifier",
            build_pipeline(MLPClassifier(random_state=random_state, **mlp_params)),
            mlp_logged_params,
        ),
    ]
