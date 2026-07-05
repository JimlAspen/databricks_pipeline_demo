# Databricks notebook source
# COMMAND ----------
%pip install mlflow --quiet

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
"""Select and register notebook.

Reads back run IDs and validation metrics from all four training
tasks. Registers the overall RMSE-based winner as 'champion' and
runner-up as 'challenger'. Separately identifies the best-calibrated
distributional model (lowest NLL among NGBoost and Bayesian Ridge)
and registers it under a 'best_calibrated' alias.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from src.config.paths import MODEL_NAME
from src.models.registry import (
    register_model_version,
    select_best_calibrated,
    select_overall_champion,
)

# COMMAND ----------

TASK_KEYS = ["train_model_a", "train_model_b", "train_model_c", "train_model_d"]

candidates = []
for task_key in TASK_KEYS:
    candidate = {
        "run_id": dbutils.jobs.taskValues.get(taskKey=task_key, key="run_id"),
        "model_type": dbutils.jobs.taskValues.get(taskKey=task_key, key="model_type"),
        "val_rmse": dbutils.jobs.taskValues.get(taskKey=task_key, key="val_rmse"),
    }
    candidates.append(candidate)

# COMMAND ----------

winner, runner_up = select_overall_champion(candidates)
print(f"Overall champion: {winner['model_type']} (val_rmse={winner['val_rmse']})")
print(f"Overall challenger: {runner_up['model_type']} (val_rmse={runner_up['val_rmse']})")

register_model_version(run_id=winner["run_id"], model_name=MODEL_NAME, alias="champion")
register_model_version(
    run_id=runner_up["run_id"], model_name=MODEL_NAME, alias="challenger"
)

# COMMAND ----------

# NLL was logged per-model in MLflow but not passed via task values,
# since it only applies to distributional models. Fetch it directly
# from the run for the two distributional candidates.
import mlflow

distributional_candidates = [
    c for c in candidates if c["model_type"] in {"ngboost", "bayesian_ridge"}
]
for candidate in distributional_candidates:
    run = mlflow.get_run(candidate["run_id"])
    candidate["val_nll"] = run.data.metrics["best_val_nll"]

best_calibrated = select_best_calibrated(distributional_candidates)
print(
    f"Best calibrated: {best_calibrated['model_type']} "
    f"(val_nll={best_calibrated['val_nll']})"
)

register_model_version(
    run_id=best_calibrated["run_id"], model_name=MODEL_NAME, alias="best_calibrated"
)
