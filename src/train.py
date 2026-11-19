import argparse
from pathlib import Path
import mlflow
from mlflow.client import MlflowClient
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.features import BASE_FEATURES, FEATURE_COLUMNS, prepare_taxi_data

EXPERIMENT_NAME = "nyc_taxi_trip_duration"
REGISTERED_MODEL_NAME = "nyc_taxi_duration_model"

RF_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]
MLP_TRUSTED_TYPES = [
    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
]


def get_project_root() -> Path:
  return Path(__file__).resolve().parent.parent


def load_raw_data(data_path: Path) -> pd.DataFrame:
  suffix = data_path.suffix.lower()
  if suffix == ".parquet":
    return pd.read_parquet(data_path)
  if suffix == ".csv":
    return pd.read_csv(
        data_path,
        parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"],
    )
  raise ValueError(f"Unsupported dataset format: {data_path}")


def log_and_register(
    sk_model,
    run_name: str,
    model_family: str,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    extra_params: dict | None = None,
    trusted_types: list[str] | None = None,
) -> tuple[str, float, float]:
  preds = sk_model.predict(X_eval)
  rmse = float(np.sqrt(mean_squared_error(y_eval, preds)))
  mae = float(mean_absolute_error(y_eval, preds))

  sample_X = X_eval.head(5)
  signature = infer_signature(sample_X, sk_model.predict(sample_X))

  params = {
      "model_family": model_family,
      "features": ", ".join(list(X_eval.columns)),
      "num_features": X_eval.shape[1],
  }
  if extra_params:
    params.update(extra_params)

  log_kwargs = {
      "sk_model": sk_model,
      "name": "model",
      "signature": signature,
      "input_example": sample_X,
      "registered_model_name": REGISTERED_MODEL_NAME,
  }
  if trusted_types is not None:
    log_kwargs["skops_trusted_types"] = trusted_types

  with mlflow.start_run(run_name=run_name):
    mlflow.log_params(params)
    mlflow.log_metrics({"val_rmse": rmse, "val_mae": mae})
    model_info = mlflow.sklearn.log_model(**log_kwargs)

  version = str(model_info.registered_model_version)
  print(
      f"[{run_name}] Registered v{version} | RMSE: {rmse:.4f} min | MAE:"
      f" {mae:.4f} min"
  )
  return version, rmse, mae


def run_baseline(client: MlflowClient, df_clean: pd.DataFrame) -> str:
  X_train, X_test, y_train, y_test = train_test_split(
      df_clean[BASE_FEATURES],
      df_clean["duration"],
      test_size=0.2,
      random_state=42,
  )
  rf_model = RandomForestRegressor(
      n_estimators=100,
      max_depth=15,
      random_state=42,
      n_jobs=-1,
  )
  rf_model.fit(X_train, y_train)

  version, rmse, _ = log_and_register(
      sk_model=rf_model,
      run_name="01_rf_baseline_no_location",
      model_family="RandomForestRegressor",
      X_eval=X_test,
      y_eval=y_test,
      extra_params={"has_location_features": False, "step": "baseline"},
      trusted_types=RF_TRUSTED_TYPES,
  )

  client.set_registered_model_alias(
      name=REGISTERED_MODEL_NAME,
      alias="champion",
      version=version,
  )
  client.set_model_version_tag(
      REGISTERED_MODEL_NAME, version, "role", "baseline"
  )
  client.set_model_version_tag(
      REGISTERED_MODEL_NAME, version, "governance_status", "champion"
  )
  client.set_model_version_tag(
      REGISTERED_MODEL_NAME, version, "val_rmse", f"{rmse:.4f}"
  )
  print(f"Assigned @champion to baseline v{version} (RMSE={rmse:.4f}).")
  return version


def run_candidate(
    client: MlflowClient,
    df_clean: pd.DataFrame,
    min_improvement_rmse: float = 0.05,
    max_allowed_rmse: float = 10.0,
) -> str:
  X_train, X_test, y_train, y_test = train_test_split(
      df_clean[FEATURE_COLUMNS],
      df_clean["duration"],
      test_size=0.2,
      random_state=42,
  )

  mlp_num_cols = ["trip_distance"]
  mlp_cat_cols = [
      "pickup_hour",
      "pickup_dayofweek",
      "PULocationID",
      "DOLocationID",
  ]

  preprocessor = ColumnTransformer(
      transformers=[
          ("num", StandardScaler(), mlp_num_cols),
          ("cat", OneHotEncoder(handle_unknown="ignore"), mlp_cat_cols),
      ]
  )
  mlp_model = Pipeline(
      steps=[
          ("prep", preprocessor),
          (
              "mlp",
              MLPRegressor(
                  hidden_layer_sizes=(128, 64),
                  activation="relu",
                  batch_size=2048,
                  learning_rate_init=1e-3,
                  max_iter=50,
                  early_stopping=True,
                  validation_fraction=0.1,
                  n_iter_no_change=3,
                  random_state=42,
                  verbose=False,
              ),
          ),
      ]
  )
  mlp_model.fit(X_train, y_train)

  mlp_step = mlp_model.named_steps["mlp"]
  version, candidate_rmse, candidate_mae = log_and_register(
      sk_model=mlp_model,
      run_name="02_mlp_regressor_candidate",
      model_family="MLPRegressor",
      X_eval=X_test,
      y_eval=y_test,
      extra_params={
          "has_location_features": True,
          "step": "candidate",
          "hidden_layer_sizes": str(mlp_step.hidden_layer_sizes),
          "activation": mlp_step.activation,
          "max_iter": mlp_step.max_iter,
      },
      trusted_types=MLP_TRUSTED_TYPES,
  )

  # Marcar como @candidate (nunca promueve automáticamente a @champion aquí)
  client.set_registered_model_alias(
      name=REGISTERED_MODEL_NAME,
      alias="candidate",
      version=version,
  )
  client.set_model_version_tag(
      REGISTERED_MODEL_NAME, version, "role", "candidate"
  )
  client.set_model_version_tag(
      REGISTERED_MODEL_NAME, version, "candidate_rmse", f"{candidate_rmse:.4f}"
  )
  client.set_model_version_tag(
      REGISTERED_MODEL_NAME, version, "candidate_mae", f"{candidate_mae:.4f}"
  )

  # Evaluar reglas de gobernanza frente al @champion actual
  champion_version = None
  champion_rmse = None
  try:
    champ_mv = client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, "champion"
    )
    champion_version = str(champ_mv.version)
    champ_run = client.get_run(champ_mv.run_id)
    champion_rmse = champ_run.data.metrics.get("val_rmse")
  except Exception:
    pass

  if candidate_rmse > max_allowed_rmse:
    eligible = False
    governance_status = "rejected_max_rmse"
    reason = (
        f"Candidate v{version} RMSE ({candidate_rmse:.4f}) > max allowed"
        f" ({max_allowed_rmse:.4f})."
    )
  elif champion_rmse is None:
    eligible = True
    governance_status = "pending_approval"
    reason = (
        f"No @champion found; candidate v{version} (RMSE={candidate_rmse:.4f})"
        " is eligible and awaiting human approval."
    )
  else:
    target_rmse = float(champion_rmse) - min_improvement_rmse
    eligible = candidate_rmse <= target_rmse
    if eligible:
      governance_status = "pending_approval"
      reason = (
          f"Candidate v{version} improves @champion v{champion_version}"
          f" ({float(champion_rmse):.4f} -> {candidate_rmse:.4f}, threshold <="
          f" {target_rmse:.4f}). Eligible and awaiting human approval."
      )
    else:
      governance_status = "not_eligible"
      reason = (
          f"Candidate v{version} (RMSE={candidate_rmse:.4f}) did not beat"
          f" @champion v{champion_version} ({float(champion_rmse):.4f}) by"
          f" min_improvement={min_improvement_rmse:.4f} (needed <="
          f" {target_rmse:.4f})."
      )

  client.set_model_version_tag(
      REGISTERED_MODEL_NAME,
      version,
      "eligible_for_promotion",
      str(eligible).lower(),
  )
  client.set_model_version_tag(
      REGISTERED_MODEL_NAME, version, "governance_status", governance_status
  )
  if champion_version is not None:
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, version, "compared_champion_version", champion_version
    )
  if champion_rmse is not None:
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "champion_rmse",
        f"{float(champion_rmse):.4f}",
    )

  print(f"[Governance] v{version} -> alias @candidate | status={governance_status}")
  print(f"[Governance] {reason}")
  return version


def train_pipeline(
    data_path: Path,
    step: str = "all",
    sample_rows: int | None = None,
    min_improvement_rmse: float = 0.05,
    max_allowed_rmse: float = 10.0,
) -> None:
  project_root = get_project_root()
  db_path = (project_root / "mlflow.db").as_posix()
  mlflow.set_tracking_uri(f"sqlite:///{db_path}")
  mlflow.set_experiment(EXPERIMENT_NAME)
  client = MlflowClient()

  raw_df = load_raw_data(data_path)
  if sample_rows is not None and sample_rows < len(raw_df):
    raw_df = raw_df.sample(n=sample_rows, random_state=42)

  df_clean = prepare_taxi_data(raw_df)

  if step in ("baseline", "all"):
    run_baseline(client=client, df_clean=df_clean)

  if step in ("candidate", "all"):
    run_candidate(
        client=client,
        df_clean=df_clean,
        min_improvement_rmse=min_improvement_rmse,
        max_allowed_rmse=max_allowed_rmse,
    )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Train and register NYC taxi duration models with governance."
  )
  parser.add_argument(
      "--data-path",
      type=Path,
      required=True,
      help="Path to NYC Yellow Taxi parquet or CSV dataset.",
  )
  parser.add_argument(
      "--step",
      choices=["baseline", "candidate", "all"],
      default="all",
      help="Which pipeline step to run.",
  )
  parser.add_argument(
      "--sample-rows",
      type=int,
      default=None,
      help="Optional row sample for faster tests.",
  )
  parser.add_argument(
      "--min-improvement-rmse",
      type=float,
      default=0.05,
      help="Minimum RMSE improvement over @champion to be eligible.",
  )
  parser.add_argument(
      "--max-allowed-rmse",
      type=float,
      default=10.0,
      help="Maximum allowed RMSE for a candidate model.",
  )
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  train_pipeline(
      data_path=args.data_path,
      step=args.step,
      sample_rows=args.sample_rows,
      min_improvement_rmse=args.min_improvement_rmse,
      max_allowed_rmse=args.max_allowed_rmse,
  )