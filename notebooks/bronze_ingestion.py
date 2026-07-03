# Databricks notebook source
# COMMAND ----------
"""
Bronze ingestion notebook.

This notebook loads the sklearn breast cancer dataset and writes it to the
Bronze Delta layer. It orchestrates the ingestion step by calling into the
src/data module.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from src.config.paths import BRONZE_PATH
from src.data.load_breast_cancer import load_and_write_bronze

load_and_write_bronze(bronze_path=BRONZE_PATH)

