# Databricks notebook source
# COMMAND ----------
"""Train Model A: Logistic Regression.

Loads the training set, runs an Optuna hyperparameter search over
Logistic Regression, and logs all trials plus the best model to
MLflow.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from pyspark.sql import SparkSession

from src.config.paths import TRAIN_TABLE
from src.models.train import run_hyperparameter_search

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
train_pdf = spark.table(TRAIN_TABLE).toPandas()

study = run_hyperparameter_search(
    model_type="logistic_regression",
    train_pdf=train_pdf,
    n_trials=15,
)

print(f"Best val_auc: {study.best_value}")
print(f"Best params: {study.best_params}")
