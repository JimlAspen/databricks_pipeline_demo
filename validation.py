"""
Validation utilities for ensuring that the Bronze and Silver datasets meet
expected schema requirements before progressing through the pipeline.
"""

from typing import List

from pyspark.sql import DataFrame

from src.config.features import FEATURE_COLUMNS, TARGET_COLUMN, ID_COLUMN


def validate_bronze_schema(bronze_df: DataFrame) -> None:
    """
    Validate that the Bronze dataset contains all required columns.

    Args:
        bronze_df (DataFrame): The Bronze Spark DataFrame.

    Raises:
        ValueError: If any expected column is missing.
    """
    expected_columns: List[str] = FEATURE_COLUMNS + [TARGET_COLUMN, ID_COLUMN]
    bronze_columns = set(bronze_df.columns)

    missing = set(expected_columns) - bronze_columns
    if missing:
        raise ValueError(
            f"Bronze schema validation failed. Missing columns: {missing}"
        )
