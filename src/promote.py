import argparse
from datetime import datetime, timezone
from pathlib import Path
import mlflow
from mlflow.client import MlflowClient

REGISTERED_MODEL_NAME = "nyc_taxi_duration_model"


def get_tracking_uri() -> str:
  project_root = Path(__file__).resolve().parent.parent
  db_path = (project_root / "mlflow.db").as_posix()
  return f"sqlite:///{db_path}"


def resolve_target_version(client: MlflowClient, version: str | None) -> str:
  if version is not None:
    return str(version)
  mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "candidate")
  return str(mv.version)


def inspect_status(client: MlflowClient, version: str | None) -> None:
  target_ver = resolve_target_version(client, version)
  mv = client.get_model_version(REGISTERED_MODEL_NAME, target_ver)
  run = client.get_run(mv.run_id)

  champ_info = "none"
  try:
    champ_mv = client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, "champion"
    )
    champ_run = client.get_run(champ_mv.run_id)
    champ_rmse = champ_run.data.metrics.get("val_rmse")
    champ_info = (
        f"v{champ_mv.version} (RMSE={champ_rmse:.4f})"
        if champ_rmse is not None
        else f"v{champ_mv.version}"
    )
  except Exception:
    pass

  cand_rmse = run.data.metrics.get("val_rmse")
  cand_mae = run.data.metrics.get("val_mae")
  eligible = mv.tags.get("eligible_for_promotion", "unknown")
  status = mv.tags.get("governance_status", "unknown")

  print(f"Current @champion : {champ_info}")
  print(f"Target version    : v{target_ver} (run_name={run.info.run_name})")
  print(
      f"Metrics           : val_rmse={cand_rmse:.4f} |"
      f" val_mae={cand_mae:.4f}"
      if cand_rmse is not None and cand_mae is not None
      else f"Metrics           : {run.data.metrics}"
  )
  print(f"Eligible          : {eligible}")
  print(f"Governance status : {status}")


def approve_and_promote(
    client: MlflowClient,
    version: str | None,
    force: bool = False,
) -> None:
  target_ver = resolve_target_version(client, version)
  mv = client.get_model_version(REGISTERED_MODEL_NAME, target_ver)
  eligible_tag = mv.tags.get("eligible_for_promotion", "false").lower()

  if eligible_tag != "true" and not force:
    raise RuntimeError(
        f"Version v{target_ver} has eligible_for_promotion='{eligible_tag}'. "
        "Use --force if you want to override governance checks."
    )

  previous_champion = None
  try:
    champ_mv = client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, "champion"
    )
    previous_champion = str(champ_mv.version)
  except Exception:
    pass

  client.set_registered_model_alias(
      name=REGISTERED_MODEL_NAME,
      alias="champion",
      version=target_ver,
  )
  promoted_at = datetime.now(timezone.utc).isoformat()
  client.set_model_version_tag(
      REGISTERED_MODEL_NAME, target_ver, "governance_status", "approved_promoted"
  )
  client.set_model_version_tag(
      REGISTERED_MODEL_NAME, target_ver, "approved_by_human", "true"
  )
  client.set_model_version_tag(
      REGISTERED_MODEL_NAME, target_ver, "promoted_at_utc", promoted_at
  )
  if previous_champion is not None:
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        target_ver,
        "previous_champion",
        previous_champion,
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        previous_champion,
        "governance_status",
        "archived_champion",
    )

  print(f"Approved and promoted v{target_ver} to @champion.")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Inspect and approve @candidate promotion to @champion."
  )
  parser.add_argument(
      "--version",
      type=str,
      default=None,
      help="Specific model version to inspect/promote (defaults to @candidate).",
  )
  parser.add_argument(
      "--status",
      action="store_true",
      help="Print governance status and metrics without promoting.",
  )
  parser.add_argument(
      "--approve",
      action="store_true",
      help="Explicit human approval to promote target version to @champion.",
  )
  parser.add_argument(
      "--force",
      action="store_true",
      help="Override eligible_for_promotion check during --approve.",
  )
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  mlflow.set_tracking_uri(get_tracking_uri())
  client = MlflowClient()

  if args.approve:
    approve_and_promote(client=client, version=args.version, force=args.force)
  else:
    inspect_status(client=client, version=args.version)