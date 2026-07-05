# Databricks notebook source
# COMMAND ----------
%pip install mlflow scikit-learn pandas ngboost matplotlib scipy --quiet

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
"""Calibration visualization notebook.

Loads the best-calibrated distributional model, generates its full
predictive distribution over the held-out scoring set, and overlays
the aggregate predicted distribution against the true outcome
histogram. This is a standalone analysis notebook, run manually to
produce a presentation artifact, not part of the automated job.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from pyspark.sql import SparkSession
from scipy.stats import norm

from src.config.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.config.paths import MODEL_NAME, SCORING_TABLE

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
scoring_df = spark.table(SCORING_TABLE)
scoring_pdf = scoring_df.select(*FEATURE_COLUMNS, TARGET_COLUMN).toPandas()

X_test = scoring_pdf[FEATURE_COLUMNS]
y_test = scoring_pdf[TARGET_COLUMN].to_numpy()

# COMMAND ----------

model_uri = f"models:/{MODEL_NAME}@best_calibrated"
loaded = mlflow.pyfunc.load_model(model_uri)
wrapped = loaded.unwrap_python_model()

X_test_scaled = wrapped.scaler.transform(X_test)

# Dispatch on the underlying model type to get mean + std per row,
# since NGBoost and Bayesian Ridge expose predictive uncertainty
# through different APIs.
underlying_model = wrapped.model
model_class_name = type(underlying_model).__name__

if model_class_name == "NGBRegressor":
    dist = underlying_model.pred_dist(X_test_scaled)
    means = dist.params["loc"]
    stds = dist.params["scale"]
else:
    means, stds = underlying_model.predict(X_test_scaled, return_std=True)

# COMMAND ----------

# Build the mixture density: the sum of every test point's predicted
# Normal distribution, evaluated over a shared grid, then averaged.
x_grid = np.linspace(y_test.min() - 30, y_test.max() + 30, 500)
mixture_density = np.mean(
    [norm.pdf(x_grid, loc=m, scale=s) for m, s in zip(means, stds)], axis=0
)

# COMMAND ----------

plt.figure(figsize=(8, 5))
plt.hist(
    y_test,
    bins=20,
    density=True,
    alpha=0.4,
    label="True outcomes",
    color="black",
)
plt.plot(
    x_grid,
    mixture_density,
    color="tab:blue",
    linewidth=2,
    label="Predicted (summed distributions)",
)
plt.fill_between(x_grid, mixture_density, alpha=0.2, color="tab:blue")
plt.xlabel("Disease progression score")
plt.ylabel("Density")
plt.title("Does the model's aggregate uncertainty match reality?")
plt.legend()
plt.tight_layout()
plt.savefig("/tmp/calibration_overlay.png", dpi=150)
plt.show()

# COMMAND ----------

# Second panel: individual predicted distributions for a handful of
# test points, with the true value marked, for an intuitive
# point-by-point illustration alongside the aggregate view above.
sample_indices = np.random.choice(len(y_test), size=4, replace=False)

fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for ax, idx in zip(axes, sample_indices):
    x_local = np.linspace(means[idx] - 4 * stds[idx], means[idx] + 4 * stds[idx], 200)
    ax.plot(x_local, norm.pdf(x_local, loc=means[idx], scale=stds[idx]))
    ax.axvline(y_test[idx], color="red", linestyle="--", label="Actual")
    ax.set_title(f"Test point {idx}")
    ax.legend()

plt.tight_layout()
plt.savefig("/tmp/individual_distributions.png", dpi=150)
plt.show()
