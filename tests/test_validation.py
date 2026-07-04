# tests/test_validation.py
"""Unit tests for schema validation utilities."""
import pytest
from pyspark.sql import SparkSession

from src.scoring.validation import (
    validate_bronze_schema,
    validate_gold_schema,
    validate_silver_schema,
)


@pytest.fixture(scope="module")
def spark():
    """Provide a local SparkSession for tests."""
    return SparkSession.builder.master("local[1]").getOrCreate()


def test_validate_bronze_schema_passes_with_all_columns(spark, monkeypatch):
    """Should not raise when all expected columns are present."""
    monkeypatch.setattr(
        "src.scoring.validation.FEATURE_COLUMNS", ["feature_1"]
    )
    df = spark.createDataFrame([(1.0, 0, 1)], ["feature_1", "target", "id"])
    monkeypatch.setattr("src.scoring.validation.TARGET_COLUMN", "target")
    monkeypatch.setattr("src.scoring.validation.ID_COLUMN", "id")
    validate_bronze_schema(bronze_df=df)  # should not raise


def test_validate_bronze_schema_raises_on_missing_column(spark, monkeypatch):
    """Should raise ValueError when a required column is missing."""
    monkeypatch.setattr(
        "src.scoring.validation.FEATURE_COLUMNS", ["feature_1", "feature_2"]
    )
    monkeypatch.setattr("src.scoring.validation.TARGET_COLUMN", "target")
    monkeypatch.setattr("src.scoring.validation.ID_COLUMN", "id")
    df = spark.createDataFrame([(1.0, 0, 1)], ["feature_1", "target", "id"])
    with pytest.raises(ValueError, match="Missing columns"):
        validate_bronze_schema(bronze_df=df)


def test_validate_silver_schema_raises_on_nulls(spark, monkeypatch):
    """Should raise ValueError when nulls are present in expected columns."""
    monkeypatch.setattr("src.scoring.validation.FEATURE_COLUMNS", ["feature_1"])
    monkeypatch.setattr("src.scoring.validation.TARGET_COLUMN", "target")
    monkeypatch.setattr("src.scoring.validation.ID_COLUMN", "id")
    df = spark.createDataFrame(
        [(1.0, 0, 1), (None, 1, 2)], ["feature_1", "target", "id"]
    )
    with pytest.raises(ValueError, match="Nulls found"):
        validate_silver_schema(silver_df=df)


def test_validate_gold_schema_passes_with_all_columns(spark, monkeypatch):
    """Should not raise when all Gold feature columns are present."""
    monkeypatch.setattr(
        "src.scoring.validation.GOLD_FEATURE_COLUMNS",
        ["feature_1", "target", "id"],
    )
    df = spark.createDataFrame([(1.0, 0, 1)], ["feature_1", "target", "id"])
    validate_gold_schema(gold_df=df)  # should not raise
