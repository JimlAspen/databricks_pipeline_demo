"""
Centralized paths for medallion layers and scoring artifacts.
"""

BRONZE_PATH: str = "dbfs:/mnt/breast_cancer/bronze"
SILVER_PATH: str = "dbfs:/mnt/breast_cancer/silver"
GOLD_PATH: str = "dbfs:/mnt/breast_cancer/gold"

TRAIN_PATH: str = "dbfs:/mnt/breast_cancer/train"
SCORE_PATH: str = "dbfs:/mnt/breast_cancer/score"

SCORE_STORE_PATH: str = "dbfs:/mnt/breast_cancer/score_store"

RAW_SKLEARN_DUMP_PATH: str = "dbfs:/mnt/breast_cancer/raw_sklearn_dump"
