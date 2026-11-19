from pathlib import Path
from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel, Field

MODEL_URI = "models:/nyc_taxi_duration_model@champion"


def _resolve_tracking_uri() -> str:
  project_root = Path(__file__).resolve().parent.parent
  db_path = (project_root / "mlflow.db").as_posix()
  return f"sqlite:///{db_path}"


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


mlflow.set_tracking_uri(_resolve_tracking_uri())
model = mlflow.sklearn.load_model(MODEL_URI)
EXPECTED_COLUMNS = _resolve_expected_columns(model)

app = FastAPI(
    title="NYC Taxi Trip Duration API",
    version="1.0.0",
    description="Inference API serving the @champion model from MLflow.",
)


class TripFeatures(BaseModel):
  trip_distance: float = Field(
      ..., gt=0, le=100, description="Distance in miles"
  )
  pickup_hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
  pickup_dayofweek: int = Field(
      ..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)"
  )
  passenger_count: int = Field(
      default=1, ge=1, le=8, description="Number of passengers"
  )
  PULocationID: int = Field(
      default=1, ge=1, description="Pickup TLC taxi zone ID"
  )
  DOLocationID: int = Field(
      default=1, ge=1, description="Dropoff TLC taxi zone ID"
  )


class PredictionResponse(BaseModel):
  predicted_duration_minutes: float
  model_uri: str


@app.get("/health")
def health_check() -> dict[str, str]:
  return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict", response_model=PredictionResponse)
def predict_duration(trip: TripFeatures) -> PredictionResponse:
  try:
    payload = trip.model_dump()
    row = {col: payload.get(col) for col in EXPECTED_COLUMNS}
    df_input = pd.DataFrame([row], columns=EXPECTED_COLUMNS)
    pred = float(model.predict(df_input)[0])
    return PredictionResponse(
        predicted_duration_minutes=round(pred, 2),
        model_uri=MODEL_URI,
    )
  except Exception as exc:
    raise HTTPException(status_code=500, detail=str(exc)) from exc