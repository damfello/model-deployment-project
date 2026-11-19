import pandas as pd

FEATURE_COLUMNS = [
    "trip_distance",
    "pickup_hour",
    "pickup_dayofweek",
    "PULocationID",
    "DOLocationID",
]

BASE_FEATURES = [
    "trip_distance",
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
]


def prepare_taxi_data(df: pd.DataFrame) -> pd.DataFrame:
  """Cleans raw NYC taxi trips and builds duration and time features."""
  data = df.copy()
  data["duration"] = (
      data["tpep_dropoff_datetime"] - data["tpep_pickup_datetime"]
  ).dt.total_seconds() / 60.0
  data["avg_speed_mph"] = data["trip_distance"] / (data["duration"] / 60.0)

  clean_mask = (
      (data["duration"] >= 1.0)
      & (data["duration"] <= 130.0)
      & (data["trip_distance"] > 0.0)
      & (data["trip_distance"] <= 50.0)
      & (data["passenger_count"] > 0)
      & (data["passenger_count"] <= 6)
      & (data["avg_speed_mph"] <= 70.0)
  )
  df_clean = data.loc[clean_mask].copy()
  df_clean["pickup_hour"] = df_clean["tpep_pickup_datetime"].dt.hour
  df_clean["pickup_dayofweek"] = df_clean["tpep_pickup_datetime"].dt.day_of_week
  return df_clean