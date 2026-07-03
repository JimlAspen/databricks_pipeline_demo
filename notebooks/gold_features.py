# Databricks notebook source
# COMMAND ----------
"""
Gold feature engineering notebook.

This notebook loads the Silver dataset, applies deterministic feature
engineering, and writes the Gold Delta table.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from src.config.paths import SILVER_PATH, GOLD_PATH
from src.data.io import read_delta, write_delta
from src.features.build_gold_features import build_gold_features

silver_df = read_delta(path=SILVER_PATH)

gold_df = build_gold_features(silver_df=silver_df)

write_delta(df=gold_df, path=GOLD_PATH)
