"""
IO utilities for reading and writing Delta tables. These helpers centralize
Delta operations to ensure consistent behavior across the pipeline.
"""

from pyspark.sql import DataFrame, SparkSession


def read_delta(path: str) -> DataFrame:
    """
    Read a Delta table from the specified path.

    Args:
        path (str): The DBFS path of the Delta table.

    Returns:
        DataFrame: A Spark DataFrame containing the loaded Delta table.
    """
    spark = SparkSession.builder.getOrCreate()
    return spark.read.format("delta").load(path)


def write_delta(df: DataFrame, path: str, mode: str = "overwrite") -> None:
    """
    Write a Spark DataFrame to a Delta table.

    Args:
        df (DataFrame): The Spark DataFrame to write.
        path (str): The DBFS path where the Delta table should be stored.
        mode (str): The write mode (default: "overwrite").

    Returns:
        None
    """
    df.write.format("delta").mode(mode).save(path)
