# Databricks notebook source
# COMMAND ----------
"""
Silver cleaning notebook.

This notebook loads the Bronze dataset, validates its schema, applies
cleaning transformations, and writes the Silver Delta table.
"""

from src.config.paths import BRONZE_PATH, SILVER_PATH
from src.data.io import read_delta, write_delta
from src.scoring.validation import validate_bronze_schema

bronze_df = read_delta(path=BRONZE_PATH)

validate_bronze_schema(bronze_df=bronze_df)

silver_df = bronze_df.dropna()

write_delta(df=silver_df, path=SILVER_PATH)
