"""Table name constants for the medallion and ML pipeline layers.

Centralizes all Unity Catalog table names so notebooks reference a
single source of truth instead of hardcoded strings. Must be kept in
sync with the catalog/schema variables in databricks.yml.
"""

CATALOG = "databricks_pipeline_demo"
MEDALLION_SCHEMA = "medallion"
ML_SCHEMA = "ml"

BRONZE_TABLE = f"{CATALOG}.{MEDALLION_SCHEMA}.bronze_breast_cancer"
SILVER_TABLE = f"{CATALOG}.{MEDALLION_SCHEMA}.silver_breast_cancer"
GOLD_TABLE = f"{CATALOG}.{MEDALLION_SCHEMA}.gold_breast_cancer_features"

TRAIN_TABLE = f"{CATALOG}.{ML_SCHEMA}.train_set"
SCORING_TABLE = f"{CATALOG}.{ML_SCHEMA}.scoring_set"
SCORED_OUTPUT_TABLE = f"{CATALOG}.{ML_SCHEMA}.scored_output"
