"""Environment-backed application settings."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration for MLflow and model governance."""

    tracking_uri: str
    experiment_name: str
    registered_model_name: str
    model_alias: str

    @classmethod
    def from_env(cls, project_root: Path = PROJECT_ROOT) -> "Settings":
        """Load settings from a local .env file and environment variables."""
        load_dotenv(project_root / ".env")
        return cls(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "purchase-propensity"),
            registered_model_name=os.getenv(
                "MLFLOW_REGISTERED_MODEL", "purchase-propensity-classifier"
            ),
            model_alias=os.getenv("MLFLOW_MODEL_ALIAS", "champion"),
        )
