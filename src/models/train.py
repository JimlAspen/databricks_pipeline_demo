"""Training utilities for the diabetes progression regression models.

Provides a shared Optuna + MLflow hyperparameter search interface
used by four candidate model notebooks: Linear Regression and
Gradient Boosting Regressor (point estimate), plus NGBoost and
Bayesian Ridge (distributional).
"""

from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow.models import infer_signature
from ngboost import NGBRegressor
from ngboost.distns import Normal
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.config.features import FEATURE_COLUMNS, TARGET_COLUMN

DISTRIBUTIONAL_MODELS = {"ngboost", "bayesian_ridge"}


class ScaledModelWrapper(mlflow.pyfunc.PythonModel):
    """Wraps a fitted model together with its fitted scaler.

    Ensures inference-time inputs are scaled identically to how the
    model was trained, so batch scoring on raw features produces
    correct results without callers needing to scale manually.
    """

    def __init__(self, model, scaler: StandardScaler):
        """Store the underlying fitted model and its fitted scaler.

        Parameters
        ----------
        model
            A fitted regressor exposing predict.
        scaler : sklearn.preprocessing.StandardScaler
            The scaler fitted on the same training data as model.

        """
        self.model = model
        self.scaler = scaler

    def predict(self, context, model_input, params=None):
        """Scale the input, then return the model's point prediction.

        Parameters
        ----------
        context
            MLflow pyfunc context, unused here.
        model_input
            Raw (unscaled) input features to score.
        params
            Optional inference parameters, unused here.

        Returns
        -------
        numpy.ndarray
            Predicted disease progression score per row.

        """
        scaled_input = self.scaler.transform(model_input)
        return self.model.predict(scaled_input)


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
        X_train, X_val, y_train, y_val, all with raw (unscaled)
        feature values.

    """
    X = train_pdf[FEATURE_COLUMNS]
    y = train_pdf[TARGET_COLUMN]
    return train_test_split(X, y, test_size=val_fraction, random_state=seed)


def suggest_linear_regression_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest a hyperparameter set for Linear Regression.

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
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "positive": trial.suggest_categorical("positive", [True, False]),
    }


def suggest_gradient_boosting_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest a hyperparameter set for Gradient Boosting Regressor.

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


def suggest_ngboost_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest a hyperparameter set for NGBoost.

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
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "minibatch_frac": trial.suggest_float("minibatch_frac", 0.5, 1.0),
        "base_max_depth": trial.suggest_int("base_max_depth", 2, 4),
    }


def suggest_bayesian_ridge_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest a hyperparameter set for Bayesian Ridge.

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
        "alpha_1": trial.suggest_float("alpha_1", 1e-7, 1e-4, log=True),
        "alpha_2": trial.suggest_float("alpha_2", 1e-7, 1e-4, log=True),
        "lambda_1": trial.suggest_float("lambda_1", 1e-7, 1e-4, log=True),
        "lambda_2": trial.suggest_float("lambda_2", 1e-7, 1e-4, log=True),
    }


def build_ngboost_model(params: dict[str, Any]) -> NGBRegressor:
    """Build an NGBRegressor from suggested hyperparameters.

    Parameters
    ----------
    params : dict[str, Any]
        Hyperparameters from suggest_ngboost_params, including the
        base_max_depth key which configures the weak learner rather
        than being passed directly to NGBRegressor.

    Returns
    -------
    NGBRegressor
        An unfitted NGBoost regressor with a Normal output
        distribution.

    """
    base_max_depth = params.pop("base_max_depth")
    return NGBRegressor(
        Dist=Normal,
        Base=DecisionTreeRegressor(max_depth=base_max_depth),
        verbose=False,
        **params,
    )


MODEL_REGISTRY = {
    "linear_regression": {
        "suggest_params": suggest_linear_regression_params,
        "build_model": lambda params: LinearRegression(**params),
    },
    "gradient_boosting": {
        "suggest_params": suggest_gradient_boosting_params,
        "build_model": lambda params: GradientBoostingRegressor(**params),
    },
    "ngboost": {
        "suggest_params": suggest_ngboost_params,
        "build_model": build_ngboost_model,
    },
    "bayesian_ridge": {
        "suggest_params": suggest_bayesian_ridge_params,
        "build_model": lambda params: BayesianRidge(**params),
    },
}


def compute_rmse(y_true, y_pred) -> float:
    """Compute root mean squared error.

    Implemented via sqrt(mean_squared_error(...)) rather than relying
    on a squared= keyword argument, since that argument was removed
    in newer scikit-learn versions.

    Parameters
    ----------
    y_true
        Ground truth target values.
    y_pred
        Predicted target values.

    Returns
    -------
    float
        The root mean squared error.

    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_val_nll(model_type: str, model, X_val_scaled, y_val: pd.Series) -> float:
    """Compute validation negative log-likelihood for a distributional model.

    Parameters
    ----------
    model_type : str
        One of the keys in DISTRIBUTIONAL_MODELS.
    model
        The fitted distributional model.
    X_val_scaled
        Validation features, already scaled consistently with training.
    y_val : pandas.Series
        Validation targets.

    Returns
    -------
    float
        Mean negative log-likelihood of the true values under the
        model's predicted distribution, lower is better.

    """
    y_true = y_val.to_numpy()

    if model_type == "ngboost":
        val_dist = model.pred_dist(X_val_scaled)
        return float(-val_dist.logpdf(y_true).mean())

    if model_type == "bayesian_ridge":
        means, stds = model.predict(X_val_scaled, return_std=True)
        return float(-norm.logpdf(y_true, loc=means, scale=stds).mean())

    raise ValueError(f"No NLL computation defined for model_type={model_type!r}")


def make_objective(
    model_type: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
):
    """Build an Optuna objective function for the given model type.

    Fits a StandardScaler on X_train only (never on X_val), so the
    hyperparameter search never leaks validation-set statistics into
    feature scaling. All model types are compared on validation RMSE
    for a fair, common ranking metric. Distributional models
    additionally log validation NLL per trial.

    Parameters
    ----------
    model_type : str
        One of the keys in MODEL_REGISTRY.
    X_train, X_val, y_train, y_val
        Train/validation splits produced by prepare_train_val_split,
        with raw (unscaled) feature values.

    Returns
    -------
    Callable[[optuna.Trial], float]
        An objective function suitable for optuna.Study.optimize.
        Returns validation RMSE; Optuna direction should be minimize.

    """
    model_spec = MODEL_REGISTRY[model_type]

    def objective(trial: optuna.Trial) -> float:
        params = model_spec["suggest_params"](trial)

        with mlflow.start_run(nested=True):
            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model = model_spec["build_model"](dict(params))
            model.fit(X_train_scaled, y_train)

            val_preds = model.predict(X_val_scaled)
            val_rmse = compute_rmse(y_val, val_preds)
            val_r2 = r2_score(y_val, val_preds)

            mlflow.log_params(params)
            mlflow.log_metric("val_rmse", val_rmse)
            mlflow.log_metric("val_r2", val_r2)

            if model_type in DISTRIBUTIONAL_MODELS:
                val_nll = compute_val_nll(model_type, model, X_val_scaled, y_val)
                mlflow.log_metric("val_nll", val_nll)

        return val_rmse

    return objective


def run_hyperparameter_search(
    model_type: str,
    train_pdf: pd.DataFrame,
    n_trials: int = 15,
    seed: int = 42,
) -> tuple[optuna.Study, str]:
    """Run an Optuna hyperparameter search for the given model type.

    Logs a parent MLflow run containing all trials as nested child
    runs. After hyperparameter search, refits the winning
    configuration on the full training set (train + validation
    combined), since the validation split's only purpose was to
    guide hyperparameter selection; the final deployed model should
    use every available row. Validation-set metrics are still
    reported afterward for reference, using the same held-out slice
    that selected the hyperparameters.

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
        as validation RMSE), and the MLflow run ID of the parent run.

    """
    X_train, X_val, y_train, y_val = prepare_train_val_split(
        train_pdf=train_pdf, seed=seed
    )

    mlflow.sklearn.autolog(
        log_models=False, disable=(model_type in DISTRIBUTIONAL_MODELS)
    )

    with mlflow.start_run(run_name=f"{model_type}_hyperparam_search") as parent_run:
        study = optuna.create_study(direction="minimize")
        objective = make_objective(model_type, X_train, X_val, y_train, y_val)
        study.optimize(objective, n_trials=n_trials)

        best_model_spec = MODEL_REGISTRY[model_type]
        best_model = best_model_spec["build_model"](dict(study.best_params))

        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        best_model.fit(X_train_scaled, y_train)

        best_val_preds = best_model.predict(X_val_scaled)
        best_val_r2 = r2_score(y_val, best_val_preds)
        best_val_rmse = compute_rmse(y_val, best_val_preds)
        
        wrapped_model = ScaledModelWrapper(best_model, scaler)
        signature = infer_signature(
            X_train_scaled, wrapped_model.predict(None, X_train_scaled)
        )

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_rmse", best_val_rmse)
        mlflow.log_metric("best_val_r2", best_val_r2)

        if model_type in DISTRIBUTIONAL_MODELS:
            best_val_nll = compute_val_nll(
                model_type, best_model, X_val_scaled, y_val
            )
            mlflow.log_metric("best_val_nll", best_val_nll)

        # X_full = train_pdf[FEATURE_COLUMNS]
        # y_full = train_pdf[TARGET_COLUMN]

        # final_scaler = StandardScaler().fit(X_full)
        # X_full_scaled = final_scaler.transform(X_full)

        # final_model_spec = MODEL_REGISTRY[model_type]
        # final_model = best_model_spec["build_model"](dict(study.best_params))
        # final_model.fit(X_full_scaled, y_full)

        mlflow.pyfunc.log_model(
            "model",
            python_model=wrapped_model,
            signature=signature,
            input_example=X_train.head(5),
        )

        run_id = parent_run.info.run_id

    return study, run_id
