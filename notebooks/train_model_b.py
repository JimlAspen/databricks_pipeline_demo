# Databricks notebook source
# COMMAND ----------
%pip install optuna mlflow scikit-learn pandas --quiet

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
"""Train Model B: Gradient Boosting.

Loads the training set, validates it, runs an Optuna hyperparameter
search over Gradient Boosting, and logs all trials plus the best
model to MLflow. Passes the resulting run ID to downstream tasks via
job task values.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from pyspark.sql import SparkSession

from src.config.paths import TRAIN_TABLE
from src.models.train import run_hyperparameter_search
from src.scoring.validation import validate_train_scoring_schema

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
train_df = spark.table(TRAIN_TABLE)
validate_train_scoring_schema(train_df, "Train set")
train_pdf = train_df.toPandas()

study, run_id = run_hyperparameter_search(
    model_type="gradient_boosting",
    train_pdf=train_pdf,
    n_trials=15,
)

print(f"Best val_auc: {study.best_value}")
print(f"Best params: {study.best_params}")
print(f"Run ID: {run_id}")

# COMMAND ----------

dbutils.jobs.taskValues.set(key="run_id", value=run_id)
dbutils.jobs.taskValues.set(key="model_type", value="gradient_boosting")
dbutils.jobs.taskValues.set(key="val_auc", value=study.best_value)
