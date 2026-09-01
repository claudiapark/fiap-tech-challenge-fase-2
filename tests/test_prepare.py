"""Tests for validation and deterministic splitting."""

import pandas as pd
import pytest

from purchase_propensity.data import prepare
from purchase_propensity.data.prepare import (
    normalize_target,
    prepare_dataset,
    split_dataset,
    validate_dataset,
)
from purchase_propensity.schema import TARGET_COLUMN


def test_split_is_stratified_and_reproducible(sample_dataset: pd.DataFrame) -> None:
    first_train, first_test = split_dataset(sample_dataset, test_size=0.2, random_state=42)
    second_train, second_test = split_dataset(sample_dataset, test_size=0.2, random_state=42)
    assert first_train.index.tolist() == second_train.index.tolist()
    assert first_test.index.tolist() == second_test.index.tolist()
    assert first_test[TARGET_COLUMN].mean() == sample_dataset[TARGET_COLUMN].mean()


def test_missing_required_column_raises(sample_dataset: pd.DataFrame) -> None:
    invalid = sample_dataset.drop(columns=["Month"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataset(invalid)


def test_empty_and_missing_datasets_raise(sample_dataset: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_dataset(sample_dataset.iloc[0:0])
    invalid = sample_dataset.copy()
    invalid.loc[0, "Month"] = None
    with pytest.raises(ValueError, match="missing values"):
        validate_dataset(invalid)


def test_invalid_target_raises() -> None:
    with pytest.raises(ValueError, match="unsupported labels"):
        normalize_target(pd.Series(["TRUE", "UNKNOWN"]))


def test_prepare_dataset_writes_both_splits(tmp_path, sample_dataset: pd.DataFrame) -> None:
    source = tmp_path / "source.csv"
    train = tmp_path / "processed" / "train.csv"
    test = tmp_path / "processed" / "test.csv"
    sample_dataset.to_csv(source, index=False)
    prepare_dataset(source, train, test, test_size=0.2, random_state=42)
    assert len(pd.read_csv(train)) == 64
    assert len(pd.read_csv(test)) == 16


def test_prepare_main_delegates_to_service(tmp_path, monkeypatch, capsys) -> None:
    args = type(
        "Args",
        (),
        {
            "input": tmp_path / "input.csv",
            "train_output": tmp_path / "train.csv",
            "test_output": tmp_path / "test.csv",
            "test_size": 0.2,
            "random_state": 42,
        },
    )()
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(prepare, "parse_args", lambda: args)
    monkeypatch.setattr(prepare, "prepare_dataset", lambda *values: calls.append(values))
    prepare.main()
    assert calls[0][-2:] == (0.2, 42)
    assert "Prepared datasets" in capsys.readouterr().out
