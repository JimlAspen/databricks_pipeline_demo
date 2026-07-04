"""Table name constants for the medallion and ML pipeline layers.

Centralizes all Unity Catalog table names so notebooks reference a
single source of truth instead of hardcoded strings.
"""

CATALOG = "main"
SCHEMA = "default"

BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_breast_cancer"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_breast_cancer"
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.gold_breast_cancer_features"

TRAIN_TABLE = f"{CATALOG}.{SCHEMA}.train_set"
SCORING_TABLE = f"{CATALOG}.{SCHEMA}.scoring_set"
SCORED_OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.scored_output"
