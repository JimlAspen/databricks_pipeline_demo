# Databricks notebook source
# COMMAND ----------
%pip install mlflow --quiet

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
"""Select and register notebook.

Reads back the run IDs and validation RMSE from both training tasks,
ranks them, and registers both to Unity Catalog: the winner (lowest
RMSE) as 'champion', the runner-up as 'challenger'.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from src.config.paths import MODEL_NAME
from src.models.registry import register_model_version, select_best_run

# COMMAND ----------

candidates = [
    {
        "run_id": dbutils.jobs.taskValues.get(taskKey="train_model_a", key="run_id"),
        "model_type": dbutils.jobs.taskValues.get(
            taskKey="train_model_a", key="model_type"
        ),
        "val_rmse": dbutils.jobs.taskValues.get(
            taskKey="train_model_a", key="val_rmse"
        ),
    },
    {
        "run_id": dbutils.jobs.taskValues.get(taskKey="train_model_b", key="run_id"),
        "model_type": dbutils.jobs.taskValues.get(
            taskKey="train_model_b", key="model_type"
        ),
        "val_rmse": dbutils.jobs.taskValues.get(
            taskKey="train_model_b", key="val_rmse"
        ),
    },
]

winner, runner_up = select_best_run(candidates)

print(f"Champion: {winner['model_type']} (val_rmse={winner['val_rmse']})")
print(f"Challenger: {runner_up['model_type']} (val_rmse={runner_up['val_rmse']})")

register_model_version(run_id=winner["run_id"], model_name=MODEL_NAME, alias="champion")
register_model_version(
    run_id=runner_up["run_id"], model_name=MODEL_NAME, alias="challenger"
)
