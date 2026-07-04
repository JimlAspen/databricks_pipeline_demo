"""Model registry utilities.

Provides functions for comparing candidate models and promoting them
via Unity Catalog registered model aliases.
"""

import mlflow
from mlflow import MlflowClient


def select_best_run(candidates: list[dict]) -> tuple[dict, dict]:
    """Rank candidates by validation RMSE, lowest first.

    Parameters
    ----------
    candidates : list[dict]
        Each dict must have keys "run_id", "model_type", "val_rmse".

    Returns
    -------
    tuple[dict, dict]
        The winning candidate (lowest RMSE) and the runner-up, in
        that order.

    """
    ranked = sorted(candidates, key=lambda c: c["val_rmse"])
    return ranked[0], ranked[1]


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
        "challenger").

    """
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=registered_model.version,
    )
