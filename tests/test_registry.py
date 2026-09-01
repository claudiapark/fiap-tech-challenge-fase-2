"""Tests for Model Registry governance."""

from dataclasses import dataclass

from purchase_propensity.modeling import registry


@dataclass
class FakeVersion:
    """Minimal MLflow model version response."""

    version: str = "3"


class FakeClient:
    """Capture registry operations without a backend."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def set_registered_model_alias(self, *args) -> None:
        self.calls.append(("alias", *args))

    def set_registered_model_tag(self, *args) -> None:
        self.calls.append(("model_tag", *args))

    def set_model_version_tag(self, *args) -> None:
        self.calls.append(("version_tag", *args))


def test_register_champion_sets_alias_and_tags(monkeypatch) -> None:
    client = FakeClient()
    monkeypatch.setattr(registry.mlflow, "register_model", lambda **_: FakeVersion())
    monkeypatch.setattr(registry, "MlflowClient", lambda: client)
    version = registry.register_champion("runs:/123/model", "purchase-model", "champion")
    assert version == "3"
    assert ("alias", "purchase-model", "champion", "3") in client.calls
    assert any(call[0] == "model_tag" for call in client.calls)
