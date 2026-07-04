# Databricks notebook source
# COMMAND ----------
"""Silver cleaning notebook.

Reads the Bronze table, validates its schema, applies cleaning
transformations, and declares the result as the Silver Delta Live
Table. The pipeline engine handles writing the table into the
configured Unity Catalog schema; this notebook only defines the
transformation, it never writes data directly.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import dlt
from src.scoring.validation import validate_bronze_schema, validate_silver_schema

# COMMAND ----------


@dlt.table(
    name="silver_diabetes",
    comment="Cleaned diabetes dataset, validated and null-dropped.",
)
def silver_diabetes():
    """Clean the Bronze diabetes table into the Silver layer.

    Reads the Bronze table, validates its schema, and drops rows
    containing null values.

    Returns
    -------
    pyspark.sql.DataFrame
        The cleaned diabetes dataset with nulls removed.
    """
    bronze_df = dlt.read("bronze_diabetes")
    validate_bronze_schema(bronze_df=bronze_df)
    silver_df = bronze_df.dropna()
    validate_silver_schema(silver_df=silver_df)
    return silver_df
