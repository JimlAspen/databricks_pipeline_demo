"""Training utilities for the breast cancer classification models.

Provides a shared Optuna + MLflow hyperparameter search interface
used by both candidate model notebooks (Logistic Regression and
Gradient Boosting).
"""

from typing import Any

import mlflow
import optuna
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.features import FEATURE_COLUMNS, TARGET_COLUMN

class ProbaWrapper(mlflow.pyfunc.PythonModel):
    """Wraps a scikit-learn classifier to output probability, not label."""

    def __init__(self, sklearn_model):
        """Store the underlying scikit-learn classifier.

        Parameters
        ----------
        sklearn_model
            A fitted scikit-learn classifier exposing predict_proba.
        """
        self.sklearn_model = sklearn_model

    def predict(self, context, model_input, params=None):
        """Return the positive-class probability for each input row.

        Parameters
        ----------
        context
            MLflow pyfunc context, unused here.
        model_input
            Input features to score.
        params
            Optional inference parameters, unused here.

        Returns
        -------
        numpy.ndarray
            Predicted probability of the positive class per row.
        """
        return self.sklearn_model.predict_proba(model_input)[:, 1]


def prepare_train_val_split(
    train_pdf: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the training set into train/validation subsets.

    Parameters
    ----------
    train_pdf : pandas.DataFrame
        The training data, already loaded as a pandas DataFrame.
    val_fraction : float, optional
        Fraction of rows held out for validation, by default 0.2.
    seed : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series]
        X_train, X_val, y_train, y_val.

    """
    X = train_pdf[FEATURE_COLUMNS]
    y = train_pdf[TARGET_COLUMN]
    return train_test_split(X, y, test_size=val_fraction, random_state=seed, stratify=y)


def suggest_logistic_regression_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest a hyperparameter set for Logistic Regression.

    Parameters
    ----------
    trial : optuna.Trial
        The current Optuna trial.

    Returns
    -------
    dict[str, Any]
        Hyperparameters to build the model with.

    """
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 500),
    }


def suggest_gradient_boosting_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest a hyperparameter set for Gradient Boosting.

    Parameters
    ----------
    trial : optuna.Trial
        The current Optuna trial.

    Returns
    -------
    dict[str, Any]
        Hyperparameters to build the model with.

    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    }


MODEL_REGISTRY = {
    "logistic_regression": {
        "suggest_params": suggest_logistic_regression_params,
        "build_model": lambda params: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(**params)),
            ]
        ),
    },
    "gradient_boosting": {
        "suggest_params": suggest_gradient_boosting_params,
        "build_model": lambda params: GradientBoostingClassifier(**params),
    },
}


def make_objective(
    model_type: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
):
    """Build an Optuna objective function for the given model type.

    Parameters
    ----------
    model_type : str
        One of the keys in MODEL_REGISTRY.
    X_train, X_val, y_train, y_val
        Train/validation splits produced by prepare_train_val_split.

    Returns
    -------
    Callable[[optuna.Trial], float]
        An objective function suitable for optuna.Study.optimize.

    """
    model_spec = MODEL_REGISTRY[model_type]

    def objective(trial: optuna.Trial) -> float:
        params = model_spec["suggest_params"](trial)

        with mlflow.start_run(nested=True):
            model = model_spec["build_model"](params)
            model.fit(X_train, y_train)

            val_probs = model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_probs)

            mlflow.log_params(params)
            mlflow.log_metric("val_auc", val_auc)

        return val_auc

    return objective


def run_hyperparameter_search(
    model_type: str,
    train_pdf: pd.DataFrame,
    n_trials: int = 15,
    seed: int = 42,
) -> tuple[optuna.Study, str]:
    """Run an Optuna hyperparameter search for the given model type.

    Logs a parent MLflow run containing all trials as nested child
    runs, and the best model refit on the full training set as the
    parent run's own logged model.

    Parameters
    ----------
    model_type : str
        One of the keys in MODEL_REGISTRY.
    train_pdf : pandas.DataFrame
        The training data, already loaded as a pandas DataFrame.
    n_trials : int, optional
        Number of Optuna trials to run, by default 15.
    seed : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    tuple[optuna.Study, str]
        The completed Optuna study (with .best_params and .best_value
        available), and the MLflow run ID of the parent run.

    """
    X_train, X_val, y_train, y_val = prepare_train_val_split(
        train_pdf=train_pdf, seed=seed
    )

    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run(run_name=f"{model_type}_hyperparam_search") as parent_run:
        study = optuna.create_study(direction="maximize")
        objective = make_objective(model_type, X_train, X_val, y_train, y_val)
        study.optimize(objective, n_trials=n_trials)

        best_model_spec = MODEL_REGISTRY[model_type]
        best_model = best_model_spec["build_model"](study.best_params)
        best_model.fit(X_train, y_train)

        wrapped_model = ProbaWrapper(best_model)
        signature = infer_signature(X_train, wrapped_model.predict(None, X_train))

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_auc", study.best_value)
        mlflow.pyfunc.log_model(
            "model",
            python_model=wrapped_model,
            signature=signature,
            input_example=X_train.head(5),
        )

        run_id = parent_run.info.run_id

    return study, run_id
    