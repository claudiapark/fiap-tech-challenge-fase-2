"""Shared test fixtures."""

import pandas as pd
import pytest

from purchase_propensity.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Build a deterministic dataset with the production schema."""
    rows = 80
    data: dict[str, list[object]] = {}
    for index, column in enumerate(NUMERIC_FEATURES, start=1):
        data[column] = [float((row * index) % 17) for row in range(rows)]
    data.update(
        {
            "Month": ["Feb" if row % 2 else "May" for row in range(rows)],
            "OperatingSystems": [(row % 3) + 1 for row in range(rows)],
            "Browser": [(row % 4) + 1 for row in range(rows)],
            "Region": [(row % 5) + 1 for row in range(rows)],
            "TrafficType": [(row % 6) + 1 for row in range(rows)],
            "VisitorType": [
                "Returning_Visitor" if row % 2 else "New_Visitor" for row in range(rows)
            ],
            "Weekend": [row % 2 == 0 for row in range(rows)],
            TARGET_COLUMN: [row % 4 == 0 for row in range(rows)],
        }
    )
    assert set(CATEGORICAL_FEATURES).issubset(data)
    return pd.DataFrame(data)
