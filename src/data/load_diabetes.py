"""Load the sklearn diabetes progression dataset into a Spark DataFrame.

This module is deterministic and used as the first step in the
medallion architecture pipeline. It only loads and shapes data; the
Lakeflow pipeline engine is responsible for writing the result to the
Bronze Delta table.
"""
from typing import Any

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from sklearn.datasets import load_diabetes

from src.config.features import ID_COLUMN, TARGET_COLUMN


def load_diabetes_df() -> DataFrame:
    """Load the sklearn diabetes dataset as a Spark DataFrame.

    Loads the dataset with scaled=False, since sklearn's default
    pre-scaled version standardizes features using the full dataset,
    which leaks scoring-set information into the scaling parameters.
    Raw features are used instead, so scaling can be fit on the
    training split only, downstream in the modeling step.

    Returns
    -------
    pyspark.sql.DataFrame
        The diabetes dataset with raw (unscaled) feature columns, a
        continuous disease progression target column, and a
        synthetic patient ID column.
    """
    data: Any = load_diabetes(scaled=False)
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df[TARGET_COLUMN] = data.target
    df[ID_COLUMN] = range(1, len(df) + 1)

    spark = SparkSession.builder.getOrCreate()
    return spark.createDataFrame(df)
