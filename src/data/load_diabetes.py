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

    The function performs the following steps:

    - Loads the dataset from sklearn. Features are already
      mean-centered and unit-variance-scaled by sklearn itself.
    - Converts it to a pandas DataFrame.
    - Adds a synthetic patient ID column.
    - Converts the result to a Spark DataFrame.

    Returns
    -------
    pyspark.sql.DataFrame
        The diabetes dataset with feature columns, a continuous
        disease progression target column, and a synthetic patient
        ID column.

    """
    data: Any = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df[TARGET_COLUMN] = data.target
    df[ID_COLUMN] = range(1, len(df) + 1)

    spark = SparkSession.builder.getOrCreate()
    return spark.createDataFrame(df)
