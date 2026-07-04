"""Module for constructing the Gold feature layer from the Silver dataset.

This step applies deterministic, model-agnostic feature selection and
ensures that the resulting table contains only canonical features.
"""
from typing import List

from pyspark.sql import DataFrame

from src.config.features import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN


def build_gold_features(silver_df: DataFrame) -> DataFrame:
    """Build the Gold feature layer from the Silver dataset.

    Selects canonical feature columns in a deterministic order and
    preserves the target and ID columns.

    Parameters
    ----------
    silver_df : pyspark.sql.DataFrame
        The cleaned Silver Spark DataFrame.

    Returns
    -------
    pyspark.sql.DataFrame
        A Spark DataFrame containing the Gold feature layer.
    """
    selected_columns: List[str] = FEATURE_COLUMNS + [TARGET_COLUMN, ID_COLUMN]
    return silver_df.select(*selected_columns)
