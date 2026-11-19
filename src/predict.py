from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd


MODEL_URI = "models:/nyc_taxi_duration_model@champion"


def get_tracking_uri() -> str:
  project_root = Path(__file__).resolve().parent.parent
  db_path = (project_root / "mlflow.db").as_posix()
  return f"sqlite:///{db_path}"


def load_champion_model():
  mlflow.set_tracking_uri(get_tracking_uri())
  return mlflow.sklearn.load_model(MODEL_URI)


def _resolve_expected_columns(loaded_model) -> list[str]:
  if hasattr(loaded_model, "feature_names_in_"):
    return list(loaded_model.feature_names_in_)
  if hasattr(loaded_model, "named_steps"):
    for step in loaded_model.named_steps.values():
      if hasattr(step, "feature_names_in_"):
        return list(step.feature_names_in_)
  return [
      "trip_distance",
      "pickup_hour",
      "pickup_dayofweek",
      "passenger_count",
      "PULocationID",
      "DOLocationID",
  ]


def predict_trip_duration(model, trip_payload: dict) -> float:
  expected_cols = _resolve_expected_columns(model)
  defaults = {
      "passenger_count": 1,
      "PULocationID": 1,
      "DOLocationID": 1,
  }
  row = {
      col: trip_payload[col] if col in trip_payload else defaults.get(col)
      for col in expected_cols
  }
  df_input = pd.DataFrame([row], columns=expected_cols)
  return float(model.predict(df_input)[0])