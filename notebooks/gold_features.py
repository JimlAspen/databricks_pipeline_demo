# Databricks notebook source
# COMMAND ----------
"""Gold feature engineering notebook.

Reads the Silver table, applies deterministic feature engineering,
and declares the result as the Gold Delta Live Table. The pipeline
engine handles writing the table into the configured Unity Catalog
schema; this notebook only defines the transformation, it never
writes data directly.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import dlt
from src.features.build_gold_features import build_gold_features

# COMMAND ----------


@dlt.table(
    name="gold_breast_cancer_features",
    comment="Feature-engineered breast cancer dataset, ready for training.",
)
def gold_breast_cancer_features():
    """Build the Gold feature table from the Silver breast cancer table.

    Reads the Silver table and applies deterministic feature
    engineering.

    Returns
    -------
    pyspark.sql.DataFrame
        The feature-engineered breast cancer dataset.
    """
    silver_df = dlt.read("silver_breast_cancer")
    gold_df = build_gold_features(silver_df=silver_df)
    validate_gold_schema(gold_df=gold_df)
    return gold_df
