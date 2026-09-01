"""Tests for environment-backed settings."""

from pathlib import Path

from purchase_propensity.config import Settings


def test_settings_have_safe_local_defaults(tmp_path: Path, monkeypatch) -> None:
    for variable in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_REGISTERED_MODEL",
        "MLFLOW_MODEL_ALIAS",
    ):
        monkeypatch.delenv(variable, raising=False)
    settings = Settings.from_env(tmp_path)
    assert settings.tracking_uri == "sqlite:///mlflow.db"
    assert settings.model_alias == "champion"
