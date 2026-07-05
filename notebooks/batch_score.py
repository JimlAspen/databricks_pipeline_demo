# Databricks notebook source
# COMMAND ----------
%pip install optuna mlflow scikit-learn pandas ngboost --quiet

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
"""Batch scoring notebook.

Validates the scoring set, scores it with the champion, challenger,
and best-calibrated models, writes all predictions, and reports each
model's actual test-set RMSE and R^2 on genuinely held-out data.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import mlflow
import numpy as np
from pyspark.sql import SparkSession
from sklearn.metrics import mean_squared_error, r2_score

from src.config.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.config.paths import MODEL_NAME, SCORED_OUTPUT_TABLE, SCORING_TABLE
from src.scoring.validation import validate_train_scoring_schema

# COMMAND ----------

ALIASES = ["champion", "challenger", "best_calibrated"]

spark = SparkSession.builder.getOrCreate()
scoring_df = spark.table(SCORING_TABLE)
validate_train_scoring_schema(scoring_df, "Scoring set")

scored_df = scoring_df
for alias in ALIASES:
    model_uri = f"models:/{MODEL_NAME}@{alias}"
    predict = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type="double")
    scored_df = scored_df.withColumn(f"prediction_{alias}", predict(*FEATURE_COLUMNS))

scored_df.write.mode("overwrite").saveAsTable(SCORED_OUTPUT_TABLE)
print(f"Scored {scored_df.count()} rows, written to {SCORED_OUTPUT_TABLE}")

# COMMAND ----------

scored_pdf = scored_df.select(
    TARGET_COLUMN, *[f"prediction_{alias}" for alias in ALIASES]
).toPandas()

for alias in ALIASES:
    test_rmse = np.sqrt(
        mean_squared_error(scored_pdf[TARGET_COLUMN], scored_pdf[f"prediction_{alias}"])
    )
    test_r2 = r2_score(scored_pdf[TARGET_COLUMN], scored_pdf[f"prediction_{alias}"])
    print(f"{alias} test RMSE on held-out scoring set: {test_rmse:.2f}")
    print(f"{alias} test R^2 on held-out scoring set: {test_r2:.4f}")
