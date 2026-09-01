"""Validate and split the purchase propensity dataset."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from purchase_propensity.schema import EXPECTED_COLUMNS, TARGET_COLUMN


def validate_dataset(dataset: pd.DataFrame) -> None:
    """Validate required columns and basic dataset integrity."""
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(dataset.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if dataset.empty:
        raise ValueError("Dataset must not be empty")
    if dataset.loc[:, EXPECTED_COLUMNS].isna().any().any():
        raise ValueError("Dataset contains missing values")


def normalize_target(target: pd.Series) -> pd.Series:
    """Convert the Revenue target to integer labels."""
    normalized = target.astype(str).str.upper().map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0})
    if normalized.isna().any():
        raise ValueError("Revenue contains unsupported labels")
    return normalized.astype("int8")


def split_dataset(
    dataset: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic stratified train/test split."""
    prepared = dataset.loc[:, EXPECTED_COLUMNS].copy()
    prepared[TARGET_COLUMN] = normalize_target(prepared[TARGET_COLUMN])
    return train_test_split(
        prepared,
        test_size=test_size,
        random_state=random_state,
        stratify=prepared[TARGET_COLUMN],
    )


def write_dataset(dataset: pd.DataFrame, output_path: Path) -> None:
    """Write a dataset while ensuring its parent directory exists."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)


def prepare_dataset(
    input_path: Path,
    train_output: Path,
    test_output: Path,
    test_size: float,
    random_state: int,
) -> None:
    """Load, validate, split, and persist the modeling datasets."""
    dataset = pd.read_csv(input_path)
    validate_dataset(dataset)
    train_data, test_data = split_dataset(dataset, test_size, random_state)
    write_dataset(train_data, train_output)
    write_dataset(test_data, test_output)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--test-output", required=True, type=Path)
    parser.add_argument("--test-size", required=True, type=float)
    parser.add_argument("--random-state", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    """Run the preparation command."""
    args = parse_args()
    prepare_dataset(
        args.input,
        args.train_output,
        args.test_output,
        args.test_size,
        args.random_state,
    )
    print(f"Prepared datasets saved to {args.train_output} and {args.test_output}")


if __name__ == "__main__":
    main()
