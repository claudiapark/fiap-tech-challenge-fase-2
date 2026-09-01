"""MLflow Model Registry integration."""

import mlflow
from mlflow import MlflowClient


def register_champion(model_uri: str, model_name: str, alias: str) -> str:
    """Register a model version and assign the governance alias."""
    version = mlflow.register_model(model_uri=model_uri, name=model_name)
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, version.version)
    client.set_registered_model_tag(model_name, "task", "purchase_propensity")
    client.set_model_version_tag(model_name, version.version, "status", "validated")
    return str(version.version)
