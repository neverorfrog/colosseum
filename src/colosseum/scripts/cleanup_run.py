"""Clean up a training run from local disk and W&B.

Usage:
  pixi run cleanup-run <run_name>                          # Delete local + W&B (auto-derived)
  pixi run cleanup-run <run_name> --wandb-run-id abc-123   # Local with custom W&B run ID
  pixi run cleanup-run <run_name> --local                  # Delete local only
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import wandb
import yaml
from loguru import logger


def _read_wandb_config(run_dir: Path) -> tuple[str | None, str | None]:
  """Read project and entity from the run's config.yaml."""
  config_path = run_dir / "config.yaml"
  if not config_path.exists():
    return None, None
  try:
    cfg = yaml.safe_load(config_path.read_text())
    if cfg and "logger" in cfg:
      return cfg["logger"].get("project"), cfg["logger"].get("entity")
  except Exception:
    pass
  return None, None


def cleanup(
  run_name: str,
  log_dir: str | Path = "./logs",
  delete_wandb: bool = True,
  wandb_run_id: str | None = None,
  project: str | None = None,
  entity: str | None = None,
) -> None:
  """Delete a run from local disk and optionally from W&B.

  Args:
    run_name: Run name (directory name under log_dir)
    log_dir: Base log directory
    delete_wandb: If True, also delete from W&B server
    wandb_run_id: Override W&B run ID (default: derived from run_name)
    project: Override W&B project (default: read from config.yaml)
    entity: Override W&B entity (default: read from config.yaml)
  """
  run_dir = Path(log_dir) / run_name

  # Read project/entity from config if not explicitly provided
  config_project, config_entity = _read_wandb_config(run_dir)
  if project is None:
    project = config_project
  if entity is None:
    entity = config_entity

  # Delete W&B run
  if delete_wandb:
    if project is None and wandb_run_id is None:
      logger.warning(
        "Could not determine W&B project. "
        "Pass --project and --wandb-run-id to delete a W&B run, or use --local."
      )
    elif project is None and wandb_run_id is not None:
      logger.warning(
        "W&B run ID provided but no project. Pass --project or use --local."
      )
    else:
      if wandb_run_id is None:
        id_file = run_dir / "wandb_id.txt"
        if id_file.exists():
          wandb_run_id = id_file.read_text().strip()
        else:
          wandb_run_id = run_name.replace("_", "-")

      path = f"{entity}/{project}/{wandb_run_id}" if entity else f"{project}/{wandb_run_id}"
      try:
        api = wandb.Api()
        run = api.run(path)
        run.delete()
        logger.info(f"Deleted W&B run: {path}")
      except wandb.errors.CommError as e:
        logger.warning(f"Could not find W&B run at {path}: {e}")
      except Exception as e:
        logger.error(f"Failed to delete W&B run: {e}")

  # Delete local directory
  if run_dir.exists():
    shutil.rmtree(run_dir)
    logger.info(f"Deleted local dir: {run_dir}")
  else:
    logger.warning(f"Local dir not found: {run_dir}")


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Clean up a training run from local disk and W&B."
  )
  parser.add_argument(
    "run_name",
    help="Run name (directory name under ./logs/, e.g. 't1-vel_ppo_20260101_120000')",
  )
  parser.add_argument(
    "--log-dir",
    default="./logs",
    help="Base log directory (default: ./logs)",
  )
  parser.add_argument(
    "--wandb-run-id",
    default=None,
    help="Explicit W&B run ID (default: derived from run_name by replacing _ with -)",
  )
  parser.add_argument(
    "--project",
    default=None,
    help="Override W&B project (default: read from local config.yaml)",
  )
  parser.add_argument(
    "--entity",
    default=None,
    help="Override W&B entity (default: read from local config.yaml)",
  )
  parser.add_argument(
    "--local",
    action="store_true",
    help="Only delete local directory, skip W&B",
  )
  args = parser.parse_args()

  cleanup(
    args.run_name,
    log_dir=args.log_dir,
    delete_wandb=not args.local,
    wandb_run_id=args.wandb_run_id,
    project=args.project,
    entity=args.entity,
  )


if __name__ == "__main__":
  main()
