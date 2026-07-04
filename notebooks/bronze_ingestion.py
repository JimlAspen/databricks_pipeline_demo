# Databricks notebook source
# COMMAND ----------
"""Bronze ingestion notebook.

Loads the sklearn breast cancer dataset and declares it as the Bronze
Delta Live Table. The pipeline engine handles writing the table into
the configured Unity Catalog schema; this notebook only defines the
transformation, it never writes data directly.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import dlt

from src.data.load_breast_cancer import load_breast_cancer_df

# COMMAND ----------


@dlt.table(
    name="bronze_breast_cancer",
    comment="Raw sklearn breast cancer dataset, ingested as-is.",
)
def bronze_breast_cancer():
    """Return the raw breast cancer dataset as a Spark DataFrame.

    Returns
    -------
    pyspark.sql.DataFrame
        The unmodified breast cancer dataset loaded from sklearn.

    """
    return load_breast_cancer_df()
