"""Split a Spark DataFrame into training and scoring sets.

Used to partition the Gold feature table into two disjoint sets: one
for model training, one held out for batch scoring.
"""

from pyspark.sql import DataFrame


def split_train_scoring(
    df: DataFrame,
    train_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[DataFrame, DataFrame]:
    """Split a DataFrame into training and scoring sets.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The input DataFrame to split.
    train_fraction : float, optional
        The fraction of rows to allocate to the training set, by
        default 0.5.
    seed : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame]
        A tuple of (train_df, scoring_df).

    """
    scoring_fraction = 1.0 - train_fraction
    train_df, scoring_df = df.randomSplit([train_fraction, scoring_fraction], seed=seed)
    return train_df, scoring_df
