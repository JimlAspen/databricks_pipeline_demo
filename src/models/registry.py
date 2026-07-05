"""Model registry utilities.

Provides functions for comparing candidate models and promoting them
via Unity Catalog registered model aliases.
"""

import mlflow
from mlflow import MlflowClient

DISTRIBUTIONAL_MODELS = {"ngboost", "bayesian_ridge"}


def select_overall_champion(candidates: list[dict]) -> tuple[dict, dict]:
    """Rank all candidates by validation RMSE, lowest first.

    This is the fair, common metric across all candidates, since
    point-estimate models have no notion of NLL.

    Parameters
    ----------
    candidates : list[dict]
        Each dict must have keys "run_id", "model_type", "val_rmse".

    Returns
    -------
    tuple[dict, dict]
        The overall winner (lowest RMSE) and the runner-up, in that
        order.

    """
    ranked = sorted(candidates, key=lambda c: c["val_rmse"])
    return ranked[0], ranked[1]


def select_best_calibrated(candidates: list[dict]) -> dict:
    """Select the best-calibrated model among distributional candidates.

    Ranks only candidates whose model_type is in DISTRIBUTIONAL_MODELS,
    by lowest validation NLL.

    Parameters
    ----------
    candidates : list[dict]
        Each dict must have keys "run_id", "model_type", "val_nll".
        Non-distributional candidates should be excluded before
        calling this function.

    Returns
    -------
    dict
        The best-calibrated candidate's dict.

    Raises
    ------
    ValueError
        If no distributional candidates are present.

    """
    distributional = [c for c in candidates if c["model_type"] in DISTRIBUTIONAL_MODELS]
    if not distributional:
        raise ValueError("No distributional candidates provided.")
    return min(distributional, key=lambda c: c["val_nll"])


def register_model_version(
    run_id: str,
    model_name: str,
    alias: str,
) -> None:
    """Register a model version from a run and assign it an alias.

    Parameters
    ----------
    run_id : str
        The MLflow run ID containing the logged model artifact.
    model_name : str
        The full Unity Catalog model name, e.g. "catalog.schema.model".
    alias : str
        The registry alias to assign to this version (e.g. "champion",
        "challenger", "best_calibrated").

    """
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=registered_model.version,
    )
