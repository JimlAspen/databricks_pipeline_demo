"""Unit tests for the Gold feature engineering function."""

import pytest
from pyspark.sql import SparkSession

from src.config.features import GOLD_FEATURE_COLUMNS
from src.features.build_gold_features import build_gold_features


@pytest.fixture(scope="module")
def spark():
    """Provide a local SparkSession for tests."""
    return SparkSession.builder.master("local[1]").getOrCreate()


@pytest.fixture
def silver_df(spark):
    """Build a minimal Silver-shaped DataFrame with all expected columns."""
    columns = GOLD_FEATURE_COLUMNS
    row = tuple(1.0 if col not in ("target", "id") else 0 for col in columns)
    return spark.createDataFrame([row], columns)


def test_build_gold_features_returns_expected_columns(silver_df):
    """Gold output should contain exactly the expected feature set."""
    gold_df = build_gold_features(silver_df=silver_df)
    assert set(gold_df.columns) == set(GOLD_FEATURE_COLUMNS)


def test_build_gold_features_preserves_row_count(silver_df):
    """Selecting columns should not change the number of rows."""
    gold_df = build_gold_features(silver_df=silver_df)
    assert gold_df.count() == silver_df.count()


def test_build_gold_features_column_order_is_deterministic(silver_df):
    """Column order should match GOLD_FEATURE_COLUMNS exactly."""
    gold_df = build_gold_features(silver_df=silver_df)
    assert gold_df.columns == GOLD_FEATURE_COLUMNS
