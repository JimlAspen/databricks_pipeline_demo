"""Validation utilities for the Bronze, Silver, Gold, and ML layers.

Ensures each dataset meets expected schema requirements before
progressing through the pipeline.
"""

from typing import List

from pyspark.sql import DataFrame

from src.config.features import (
    FEATURE_COLUMNS,
    GOLD_FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
)


def _validate_columns_present(
    df: DataFrame, expected_columns: List[str], layer_name: str
) -> None:
    """Validate that a DataFrame contains all expected columns.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The DataFrame to validate.
    expected_columns : list[str]
        The columns that must be present.
    layer_name : str
        The name of the layer being validated, used in error messages.

    Raises
    ------
    ValueError
        If any expected column is missing.

    """
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{layer_name} schema validation failed. Missing columns: {missing}"
        )


def validate_bronze_schema(bronze_df: DataFrame) -> None:
    """Validate that the Bronze dataset contains all required columns.

    Parameters
    ----------
    bronze_df : pyspark.sql.DataFrame
        The Bronze Spark DataFrame.

    Raises
    ------
    ValueError
        If any expected column is missing.

    """
    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN, ID_COLUMN]
    _validate_columns_present(bronze_df, expected_columns, "Bronze")


def validate_silver_schema(silver_df: DataFrame) -> None:
    """Validate that the Silver dataset contains all required columns.

    Also confirms there are no nulls remaining.

    Parameters
    ----------
    silver_df : pyspark.sql.DataFrame
        The Silver Spark DataFrame.

    Raises
    ------
    ValueError
        If any expected column is missing, or if any nulls remain.

    """
    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN, ID_COLUMN]
    _validate_columns_present(silver_df, expected_columns, "Silver")

    null_counts = (
        silver_df.select(
            [silver_df[c].isNull().cast("int").alias(c) for c in expected_columns]
        )
        .groupBy()
        .sum()
        .collect()[0]
        .asDict()
    )

    columns_with_nulls = [
        col.replace("sum(", "").replace(")", "")
        for col, count in null_counts.items()
        if count and count > 0
    ]
    if columns_with_nulls:
        raise ValueError(
            f"Silver schema validation failed. Nulls found in: {columns_with_nulls}"
        )


def validate_gold_schema(gold_df: DataFrame) -> None:
    """Validate that the Gold dataset contains all required columns.

    Parameters
    ----------
    gold_df : pyspark.sql.DataFrame
        The Gold Spark DataFrame.

    Raises
    ------
    ValueError
        If any expected column is missing.

    """
    _validate_columns_present(gold_df, GOLD_FEATURE_COLUMNS, "Gold")


def validate_train_scoring_schema(df: DataFrame, layer_name: str) -> None:
    """Validate a train or scoring set before use in modeling.

    Confirms all Gold feature columns are present, the DataFrame is
    non-empty, and the continuous target has no nulls and a plausible
    (non-degenerate) range.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The train or scoring DataFrame to validate.
    layer_name : str
        The name of the layer being validated, used in error messages.

    Raises
    ------
    ValueError
        If any expected column is missing, the DataFrame is empty,
        the target has nulls, or the target has zero variance.

    """
    _validate_columns_present(df, GOLD_FEATURE_COLUMNS, layer_name)

    row_count = df.count()
    if row_count == 0:
        raise ValueError(f"{layer_name} validation failed. DataFrame is empty.")

    target_null_count = df.filter(df[TARGET_COLUMN].isNull()).count()
    if target_null_count > 0:
        raise ValueError(
            f"{layer_name} validation failed. Target column has "
            f"{target_null_count} null values."
        )

    target_stats = df.agg({TARGET_COLUMN: "stddev"}).collect()[0][0]
    if target_stats is None or target_stats == 0:
        raise ValueError(
            f"{layer_name} validation failed. Target column has zero "
            "variance — cannot train or evaluate a regression model."
        )
