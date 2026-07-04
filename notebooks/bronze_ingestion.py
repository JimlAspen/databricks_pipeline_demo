# Databricks notebook source
# COMMAND ----------
"""Bronze ingestion notebook.

Loads the sklearn diabetes progression dataset and declares it as the
Bronze Delta Live Table. The pipeline engine handles writing the
table into the configured Unity Catalog schema; this notebook only
defines the transformation, it never writes data directly.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import dlt
from src.data.load_diabetes import load_diabetes_df

# COMMAND ----------


@dlt.table(
    name="bronze_diabetes",
    comment="Raw sklearn diabetes progression dataset, ingested as-is.",
)
def bronze_diabetes():
    """Return the raw diabetes dataset as a Spark DataFrame.

    Returns
    -------
    pyspark.sql.DataFrame
        The unmodified diabetes progression dataset loaded from sklearn.
    """
    return load_diabetes_df()
