"""Canonical feature list for the diabetes progression dataset.

Gold is model-agnostic; all candidate models use the same features.
"""

FEATURE_COLUMNS = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
TARGET_COLUMN = "disease_progression"
ID_COLUMN = "patient_id"

GOLD_FEATURE_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN, ID_COLUMN]
