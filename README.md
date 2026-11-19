
## Project Overview

This project implements an end-to-end **MLOps** workflow for predicting NYC yellow taxi trip durations (`yellow_tripdata_2025-01.parquet`) using **MLflow** for experiment tracking and model registry governance, and **FastAPI** to serve predictions over HTTP.

### Key Components

- **Training & Registration (`src/train.py`):** Trains models on trip data filtered to `trip_distance <= 50.0` miles (~80.47 km), logs validation metrics (`val_mae` and `val_rmse`), and registers versions in MLflow under aliases such as `@candidate` and `@champion`.
- **Governance & Promotion (`src/promote.py`):** Checks model registry status, evaluates candidate models against the current `@champion`, and promotes approved versions.
- **Inference API (`app/main.py` & `src/predict.py`):** Exposes `/health`, `/predict`, and interactive `/docs` endpoints, loads `models:/nyc_taxi_duration_model@champion`, and prepares input features for the active model.

### Model Evolution & Extrapolation Findings

- **Baseline (`01_rf_baseline_no_location`, `v1`):** `RandomForestRegressor` without location features (`val_mae`: `3.5120` min, `val_rmse`: `5.7246` min). When extrapolating beyond the `50.0`-mile training filter, predictions plateau at `63.96` min (e.g., `62.79` min at 30 mi, `63.38` min at 40 mi, `63.96` min at 50 mi, and `63.96` min at 100 mi / ~160.93 km).
- **Champion (`02_mlp_regressor_candidate`, `v2`):** `MLPRegressor` incorporating location and temporal features (`val_mae`: `3.0002` min, `val_rmse`: `4.8360` min). Promoted to `@champion` after improving validation metrics and verified through the API, showing continuous scaling across distances (`74.62` min at 30 mi, `96.72` min at 40 mi, `118.84` min at 50 mi, and `234.04` min at 100 mi / ~160.93 km).
