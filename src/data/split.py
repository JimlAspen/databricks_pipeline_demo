"""Split a Spark DataFrame into training and scoring sets.

Used to partition the Gold feature table into two disjoint sets: one
for model training, one held out for batch scoring. Supports an
optional stratified split to preserve class balance across both sets.
"""
from functools import reduce

from pyspark.sql import DataFrame


def split_train_scoring(
    df: DataFrame,
    train_fraction: float = 0.5,
    seed: int = 42,
    stratify_col: str | None = None,
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
    stratify_col : str, optional
        If provided, performs a stratified split, preserving the
        proportion of each distinct value in this column across both
        the training and scoring sets. If None, performs a plain
        random split, by default None.

    Returns
    -------
    tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame]
        A tuple of (train_df, scoring_df).
    """
    scoring_fraction = 1.0 - train_fraction

    if stratify_col is None:
        train_df, scoring_df = df.randomSplit(
            [train_fraction, scoring_fraction], seed=seed
        )
        return train_df, scoring_df

    distinct_values = [
        row[stratify_col] for row in df.select(stratify_col).distinct().collect()
    ]

    train_parts = []
    scoring_parts = []
    for value in distinct_values:
        stratum_df = df.filter(df[stratify_col] == value)
        stratum_train, stratum_scoring = stratum_df.randomSplit(
            [train_fraction, scoring_fraction], seed=seed
        )
        train_parts.append(stratum_train)
        scoring_parts.append(stratum_scoring)

    train_df = reduce(DataFrame.unionByName, train_parts)
    scoring_df = reduce(DataFrame.unionByName, scoring_parts)
    return train_df, scoring_df
