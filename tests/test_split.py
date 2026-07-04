"""Unit tests for the train/scoring split function."""

import pytest
from pyspark.sql import SparkSession

from src.data.split import split_train_scoring


@pytest.fixture(scope="module")
def spark():
    """Provide a local SparkSession for tests."""
    return SparkSession.builder.master("local[1]").getOrCreate()


def test_split_train_scoring_produces_disjoint_sets(spark):
    """Train and scoring sets should not overlap and cover all rows."""
    df = spark.createDataFrame([(i, i * 2) for i in range(100)], ["id", "val"])
    train_df, scoring_df = split_train_scoring(df, train_fraction=0.5, seed=42)

    train_ids = {row.id for row in train_df.collect()}
    scoring_ids = {row.id for row in scoring_df.collect()}

    assert train_ids.isdisjoint(scoring_ids)
    assert train_ids | scoring_ids == set(range(100))
