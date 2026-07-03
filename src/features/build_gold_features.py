"""
Module for constructing the Gold feature layer from the Silver dataset.
This step applies deterministic, model-agnostic feature engineering and
ensures that the resulting table contains only canonical features.
"""

from typing import List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config.features import FEATURE_COLUMNS, TARGET_COLUMN, ID_COLUMN


def build_gold_features(silver_df: DataFrame) -> DataFrame:
    """
    Build the Gold feature layer from the Silver dataset.

    This function performs the following:
    - Selects canonical feature columns.
    - Ensures deterministic column ordering.
    - Preserves the target and ID columns.
    - Applies lightweight normalization (optional placeholder).

    Args:
        silver_df (DataFrame): The cleaned Silver Spark DataFrame.

    Returns:
        DataFrame: A Spark DataFrame containing the Gold feature layer.
    """
    selected_columns: List[str] = FEATURE_COLUMNS + [TARGET_COLUMN, ID_COLUMN]

    gold_df = silver_df.select(*selected_columns)

    # Placeholder for deterministic normalization (optional)
    # Example: normalize mean_radius
    # gold_df = gold_df.withColumn(
    #     "mean_radius_norm",
    #     F.col("mean_radius") / F.max("mean_radius").over(Window.partitionBy())
    # )

    return gold_df
