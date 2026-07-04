"""Unit tests for the Bronze layer's data loading function."""

import pytest
from pyspark.sql import SparkSession

from src.config.features import ID_COLUMN, TARGET_COLUMN
from src.data.load_breast_cancer import load_breast_cancer_df


@pytest.fixture(scope="module")
def spark():
    """Provide a local SparkSession for tests."""
    return SparkSession.builder.master("local[1]").getOrCreate()


def test_load_breast_cancer_df_returns_expected_row_count(spark):
    """The sklearn breast cancer dataset has exactly 569 rows."""
    df = load_breast_cancer_df()
    assert df.count() == 569


def test_load_breast_cancer_df_has_target_and_id_columns(spark):
    """Target and ID columns should be present."""
    df = load_breast_cancer_df()
    assert TARGET_COLUMN in df.columns
    assert ID_COLUMN in df.columns


def test_load_breast_cancer_df_column_names_have_no_spaces(spark):
    """Delta Lake disallows spaces in column names.

    All column names should be sanitized to underscores.
    """
    df = load_breast_cancer_df()
    assert all(" " not in col for col in df.columns)


def test_load_breast_cancer_df_id_column_is_sequential(spark):
    """The synthetic ID column should start at 1 and be unique per row."""
    df = load_breast_cancer_df()
    ids = sorted(row[ID_COLUMN] for row in df.select(ID_COLUMN).collect())
    assert ids == list(range(1, len(ids) + 1))


def test_load_breast_cancer_df_target_is_binary(spark):
    """The target column should only contain 0 and 1."""
    df = load_breast_cancer_df()
    distinct_targets = {
        row[TARGET_COLUMN] for row in df.select(TARGET_COLUMN).distinct().collect()
    }
    assert distinct_targets == {0, 1}
