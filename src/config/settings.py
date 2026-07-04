"""Global settings for the breast cancer batch pipeline.

Config-driven, deterministic, and environment-aware.
"""

# Reproducibility
RANDOM_SEED: int = 42

# Updated split: 50% training, 50% scoring
TRAIN_RATIO: float = 0.5

# Candidate model names (both classifiers)
MODEL_A_NAME: str = "breast_cancer_classifier_A"
MODEL_B_NAME: str = "breast_cancer_classifier_B"

# Which model is currently deployed to production
MODEL_DEPLOYED_NAME: str = MODEL_A_NAME

# MLflow experiment for tracking runs
MLFLOW_EXPERIMENT: str = "/Shared/breast_cancer_pipeline"

# Environment toggle
ENVIRONMENT: str = "dev"
