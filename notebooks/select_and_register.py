# Databricks notebook source
# COMMAND ----------
%pip install mlflow --quiet

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
"""Select and register notebook.

Reads back the run IDs and validation metrics from both training
tasks, ranks them, and registers both to Unity Catalog: the winner
as 'champion', the runner-up as 'challenger'.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from src.models.registry import register_model_version, select_best_run
from src.config.paths import MODEL_NAME


# COMMAND ----------

candidates = [
    {
        "run_id": dbutils.jobs.taskValues.get(taskKey="train_model_a", key="run_id"),
        "model_type": dbutils.jobs.taskValues.get(
            taskKey="train_model_a", key="model_type"
        ),
        "val_auc": dbutils.jobs.taskValues.get(
            taskKey="train_model_a", key="val_auc"
        ),
    },
    {
        "run_id": dbutils.jobs.taskValues.get(taskKey="train_model_b", key="run_id"),
        "model_type": dbutils.jobs.taskValues.get(
            taskKey="train_model_b", key="model_type"
        ),
        "val_auc": dbutils.jobs.taskValues.get(
            taskKey="train_model_b", key="val_auc"
        ),
    },
]

winner, runner_up = select_best_run(candidates)

print(f"Champion: {winner['model_type']} (val_auc={winner['val_auc']})")
print(f"Challenger: {runner_up['model_type']} (val_auc={runner_up['val_auc']})")

register_model_version(run_id=winner["run_id"], model_name=MODEL_NAME, alias="champion")
register_model_version(
    run_id=runner_up["run_id"], model_name=MODEL_NAME, alias="challenger"
)
