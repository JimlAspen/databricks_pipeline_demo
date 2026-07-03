"""
Module for loading the sklearn breast cancer dataset and writing it to the
Bronze Delta layer. This module is deterministic and used as the first step
in the medallion architecture pipeline.
"""

from typing import Any

import pandas as pd
from sklearn.datasets import load_breast_cancer
from pyspark.sql import SparkSession

from src.config.features import ID_COLUMN, TARGET_COLUMN
from src.data.io import write_delta


def load_and_write_bronze(bronze_path: str) -> None:
    """
    Load the sklearn breast cancer dataset and write it to the Bronze Delta
    table.

    The function performs the following steps:
    - Loads the dataset from sklearn.
    - Converts it to a pandas DataFrame.
    - Adds a synthetic patient ID column.
    - Converts the DataFrame to a Spark DataFrame.
    - Writes the result to the Bronze Delta path.

    Args:
        bronze_path (str): The DBFS path where the Bronze Delta table
            should be written.

    Returns:
        None
    """
    data: Any = load_breast_cancer()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df[TARGET_COLUMN] = data.target
    df[ID_COLUMN] = range(1, len(df) + 1)

    spark = SparkSession.builder.getOrCreate()
    spark_df = spark.createDataFrame(df)

    write_delta(df=spark_df, path=bronze_path)
