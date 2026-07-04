"""Unit tests for the Bronze layer's data loading function."""

import pytest
from pyspark.sql import SparkSession

from src.config.features import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN
from src.data.load_diabetes import load_diabetes_df


@pytest.fixture(scope="module")
def spark():
    """Provide a local SparkSession for tests."""
    return SparkSession.builder.master("local[1]").getOrCreate()


def test_load_diabetes_df_returns_expected_row_count(spark):
    """The sklearn diabetes dataset has exactly 442 rows."""
    df = load_diabetes_df()
    assert df.count() == 442


def test_load_diabetes_df_has_target_and_id_columns(spark):
    """Target and ID columns should be present."""
    df = load_diabetes_df()
    assert TARGET_COLUMN in df.columns
    assert ID_COLUMN in df.columns


def test_load_diabetes_df_has_all_feature_columns(spark):
    """All expected feature columns should be present."""
    df = load_diabetes_df()
    for col in FEATURE_COLUMNS:
        assert col in df.columns


def test_load_diabetes_df_id_column_is_sequential(spark):
    """The synthetic ID column should start at 1 and be unique per row."""
    df = load_diabetes_df()
    ids = sorted(row[ID_COLUMN] for row in df.select(ID_COLUMN).collect())
    assert ids == list(range(1, len(ids) + 1))


def test_load_diabetes_df_target_is_continuous(spark):
    """Confirm the target column is continuous, not a class label.

    Checks that there are more than a handful of distinct values.
    """
    df = load_diabetes_df()
    distinct_count = df.select(TARGET_COLUMN).distinct().count()
    assert distinct_count > 50
