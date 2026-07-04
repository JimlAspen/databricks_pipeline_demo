# Databricks Medallion + ML Pipeline: Breast Cancer Classification

An end-to-end data engineering and MLOps pipeline built on Databricks Free Edition, orchestrated via Databricks Asset Bundles (DAB) and GitHub Actions. Ingests raw data, builds a governed medallion architecture, trains and compares two candidate models with hyperparameter tuning, and registers/scores the winning model through Unity Catalog.

## Architecture

```
                              ┌──────────────────┐
                              │  run_medallion    │  (Lakeflow Declarative Pipeline)
                              │  bronze → silver   │
                              │  → gold             │
                              └─────────┬─────────┘
                                        │
                              ┌─────────▼─────────┐
                              │    split_data      │  stratified 50/50 split
                              └─────────┬─────────┘
                                        │
                      ┌─────────────────┴─────────────────┐
                      ▼                                     ▼
            ┌──────────────────┐                 ┌──────────────────┐
            │  train_model_a    │                 │  train_model_b    │
            │  Logistic         │                 │  Gradient         │
            │  Regression       │                 │  Boosting         │
            │  (Optuna + MLflow)│                 │  (Optuna + MLflow)│
            └─────────┬─────────┘                 └─────────┬─────────┘
                      │                                     │
                      └─────────────────┬─────────────────┘
                                        ▼
                              ┌──────────────────┐
                              │ select_and_register│  compare val AUC,
                              │                     │  register both to UC,
                              │                     │  alias champion/challenger
                              └─────────┬─────────┘
                                        ▼
                              ┌──────────────────┐
                              │   batch_score      │  score both models on
                              │                     │  held-out scoring_set,
                              │                     │  report real test AUC
                              └───────────────────┘
```

One trigger — `databricks bundle run medallion_ml_pipeline` — runs the entire chain, from raw data ingestion to a scored output table.

## Data layers (Unity Catalog)

| Schema | Purpose | Tables |
|---|---|---|
| `databricks_pipeline_demo.medallion` | Data engineering (Lakeflow Declarative Pipeline) | `bronze_breast_cancer`, `silver_breast_cancer`, `gold_breast_cancer_features` |
| `databricks_pipeline_demo.ml` | ML workflow (Jobs) | `train_set`, `scoring_set`, `scored_output`, registered model `breast_cancer_model` |

Kept as separate schemas deliberately: medallion tables are the governed, reusable "source of truth"; ML tables are workflow-specific and tied to a given training run.

## Key design decisions

**Lakeflow Pipelines vs. Jobs.** Bronze/silver/gold are declarative `@dlt.table` transformations — pure functions of their upstream table, so Lakeflow owns materialization, lineage, and incremental refresh. Everything downstream of the split (training, registration, scoring) is stateful and side-effecting, so it runs as plain Job tasks instead.

**Validation at every layer boundary.** `validate_bronze_schema`, `validate_silver_schema`, `validate_gold_schema`, and `validate_train_scoring_schema` each check their own input independently, rather than trusting upstream stages stayed correct. This means a bad split or a corrupted table gets caught immediately, not discovered after burning compute on training.

**Stratified train/scoring split.** A plain random split doesn't guarantee equal class balance between the two sets. `split_train_scoring` splits each class independently and unions the results, preserving the true class ratio in both `train_set` and `scoring_set`.

**Feature scaling only where it matters.** Logistic Regression is wrapped in a scikit-learn `Pipeline` with a `StandardScaler`, since LR's regularization is sensitive to feature magnitude. Gradient Boosting is scale-invariant by construction and is left unscaled — scaling both uniformly would have been the easier but less correct choice.

**Parallel training, not sequential.** `train_model_a` and `train_model_b` share only `split_data` as a dependency, so Databricks runs them concurrently. Total pipeline time is `max(train_a, train_b)`, not the sum.

**Hyperparameter tuning via Optuna, tracked in MLflow.** Each model runs a 15-trial Optuna search, with every trial logged as a nested MLflow run under one parent run per model. The best trial's model is refit on the full training set and logged with an explicit signature (required by Unity Catalog's model registry) and a custom `pyfunc` wrapper so the deployed model outputs calibrated probabilities, not just hard class labels.

**Champion/challenger registration, not winner-take-all.** `select_and_register` ranks both models by validation AUC and registers *both* to Unity Catalog — the winner aliased `champion`, the runner-up `challenger` — rather than discarding the losing model. `batch_score` scores the held-out set with both, so the validation-time ranking can be checked against real, unseen data.

**Task-value handoff instead of MLflow run search.** Each training notebook writes its MLflow run ID via `dbutils.jobs.taskValues.set(...)`. `select_and_register` reads these back directly, rather than searching MLflow by name or timestamp — deterministic, not best-effort.

## Results

| Model | Validation AUC | Held-out test AUC |
|---|---|---|
| Logistic Regression (champion) | 0.9989 | **0.9940** |
| Gradient Boosting (challenger) | 0.9977 | 0.9618 |

The simpler, properly-scaled linear model outperformed the more complex ensemble on both validation and genuinely held-out data — a reminder that model complexity isn't a substitute for correct preprocessing, especially on a dataset this size and this close to linearly separable.

## CI/CD

GitHub Actions workflow, two-stage:

1. **`lint`** — `ruff check --fix .` + `ruff format .`, auto-commits any fixes back to the branch.
2. **`test`** — `ruff check .` (verify clean) + `pytest tests/`, runs on every push to `main`.
3. **`deploy`** — `databricks bundle validate` → `deploy` → `run medallion_ml_pipeline`, runs only on manual `workflow_dispatch`, gated behind `test` passing.

This means every push is tested automatically, but nothing deploys to the Databricks workspace without an explicit, deliberate trigger.

## Repo structure

```
.
├── databricks.yml              # DAB config: pipeline, job, targets (dev/prod)
├── pyproject.toml              # ruff config, project metadata
├── requirements.txt            # runtime dependencies
├── requirements-dev.txt        # + test/lint tooling
├── conftest.py                 # marks repo root for pytest imports
├── notebooks/
│   ├── bronze_ingestion.py     # @dlt.table
│   ├── silver_cleaning.py      # @dlt.table
│   ├── gold_features.py        # @dlt.table
│   ├── split_data.py           # stratified split, Job task
│   ├── train_model_a.py        # Logistic Regression + Optuna
│   ├── train_model_b.py        # Gradient Boosting + Optuna
│   ├── select_and_register.py  # champion/challenger promotion
│   └── batch_score.py          # dual-model scoring
├── src/
│   ├── config/
│   │   ├── features.py         # column name constants
│   │   ├── paths.py            # catalog/schema/table/model name constants
│   │   └── settings.py
│   ├── data/
│   │   ├── load_breast_cancer.py
│   │   └── split.py
│   ├── features/
│   │   └── build_gold_features.py
│   ├── models/
│   │   ├── train.py            # Optuna + MLflow + ProbaWrapper
│   │   └── registry.py         # champion/challenger promotion logic
│   └── scoring/
│       └── validation.py       # schema/null/class-balance validation
└── tests/
    ├── test_load_breast_cancer.py
    ├── test_build_gold_features.py
    ├── test_split.py
    └── test_validation.py
```

## Known limitations / what I'd do differently at scale

- **Single workspace.** Databricks Free Edition provides one workspace, so `dev`/`prod` targets in `databricks.yml` are a naming convention, not real infrastructure isolation. At a company, these would be separate workspaces.
- **PAT-based auth, not OIDC.** Free Edition doesn't expose account-console access, which workload identity federation (the recommended secretless GitHub Actions auth pattern) requires. A personal access token in GitHub Secrets is the practical fallback here.
- **No drift/performance monitoring in production.** The natural next step is Databricks Data Quality Monitoring's inference analysis mode, attached to `scored_output` — this requires a timestamp and model-ID column on the inference table (a design decision that needs to be made upfront, not retrofitted).
- **No regression gate.** `select_and_register` picks the better of two *new* candidates, but doesn't compare against the *currently deployed* champion before promoting. A production version would block promotion if the new model regresses against the incumbent.
- **Champion/challenger promotion is fully automatic.** There's no manual approval gate before a new model takes over `champion` — reasonable for a demo, not for a regulated or high-stakes deployment.

## Stack

Databricks (Free Edition) · Unity Catalog · Lakeflow Declarative Pipelines · Databricks Jobs · Databricks Asset Bundles · MLflow · Optuna · scikit-learn · PySpark · GitHub Actions · ruff · pytest
