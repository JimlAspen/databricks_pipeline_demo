# Databricks notebook source
# COMMAND ----------
"""Split data notebook.

Reads the Gold feature table, splits it 50/50 into training and
scoring sets, and writes both as Unity Catalog managed tables. This
is a regular notebook task (not a Lakeflow pipeline stage) since the
split is a one-time, stateful operation tied to a specific pipeline
run, not a continuously materialized transformation.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from pyspark.sql import SparkSession

from src.data.split import split_train_scoring

# COMMAND ----------
from src.config.paths import GOLD_TABLE, TRAIN_TABLE, SCORING_TABLE

spark = SparkSession.builder.getOrCreate()
gold_df = spark.table(GOLD_TABLE)

train_df, scoring_df = split_train_scoring(
    df=gold_df, train_fraction=0.5, seed=42
)

train_df.write.mode("overwrite").saveAsTable(TRAIN_TABLE)
scoring_df.write.mode("overwrite").saveAsTable(SCORING_TABLE)

print(f"Train rows: {train_df.count()}, Scoring rows: {scoring_df.count()}")
