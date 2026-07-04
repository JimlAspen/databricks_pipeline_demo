"""Training utilities for the diabetes progression regression models.

Provides a shared Optuna + MLflow hyperparameter search interface
used by both candidate model notebooks (Linear Regression and
Gradient Boosting Regressor).
"""
from typing import Any

import mlflow
import optuna
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config.features import FEATURE_COLUMNS, TARGET_COLUMN


class IdentityWrapper(mlflow.pyfunc.PythonModel):
    """Wraps a scikit-learn regressor for consistent pyfunc logging.

    Unlike the classification case, regressors already output the
    quantity of interest directly via predict(), so this wrapper is a
    thin pass-through — kept for interface consistency with the
    scoring pipeline, which expects all registered models to be
    pyfunc models.
    """

    def __init__(self, sklearn_model):
        """Store the underlying scikit-learn regressor.

        Parameters
