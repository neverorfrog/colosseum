#!/usr/bin/env python3
"""Import the most recent checkpoint and run metadata from a remote machine via SCP.

Usage:
  pixi run import-run --remote gin --root /home/phd_student/Maiorana/colosseum --run t1-vel_ppo_20260101_120000
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import tyro
from loguru import logger
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ImportRunConfig:
  remote: str = ""
  """Remote host (SSH alias or user@host)."""

  root: str = ""
  """Absolute path to colosseum root on the remote machine."""

  run: str = ""
  """Run name (directory under <root>/logs/)."""

  log_dir: str = "./logs"
  """Local log directory (default: ./logs)."""


RUN_FILES = ["config.yaml", "train.log", "wandb_id.txt"]


def main() -> None:
  config = tyro.cli(ImportRunConfig)

  remote_root = Path(config.root)
  remote_run = remote_root / "logs" / config.run
  remote_checkpoints = remote_run / "checkpoints"

  local_run = Path(config.log_dir) / config.run
  local_ckpt_dir = local_run / "checkpoints"
  local_ckpt_dir.mkdir(parents=True, exist_ok=True)

  # Find the latest checkpoint on the remote
  logger.info(f"Listing checkpoints on {config.remote}:{remote_checkpoints}")
  result = subprocess.run(
    ["ssh", config.remote, f"ls -t {remote_checkpoints}/*.pt 2>/dev/null"],
    capture_output=True,
    text=True,
  )
  if result.returncode != 0 or not result.stdout.strip():
    logger.error(f"No checkpoints found at {config.remote}:{remote_checkpoints}")
    return

  latest = result.stdout.strip().split("\n")[0]
  ckpt_name = Path(latest).name
  logger.info(f"Latest checkpoint: {ckpt_name}")

  # Import checkpoint
  subprocess.run(
    ["scp", f"{config.remote}:{latest}", str(local_ckpt_dir / ckpt_name)],
    check=True,
  )

  # Import run metadata files (best-effort)
  for fname in RUN_FILES:
    result = subprocess.run(
      ["scp", f"{config.remote}:{remote_run / fname}", str(local_run / fname)],
      capture_output=True,
      text=True,
    )
    if result.returncode == 0:
      logger.info(f"Imported {fname}")
    else:
      logger.warning(f"Skipped {fname} (not found on remote)")

  # Import wandb run data
  wandb_id_path = local_run / "wandb_id.txt"
  if wandb_id_path.exists():
    wandb_id = wandb_id_path.read_text().strip()
    logger.info(f"W&B run ID: {wandb_id}")

    remote_wandb_dir = remote_root / "logs" / "wandb"
    result = subprocess.run(
      ["ssh", config.remote, f"ls -d {remote_wandb_dir}/run-*-{wandb_id} 2>/dev/null"],
      capture_output=True,
      text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
      remote_dir = result.stdout.strip().split("\n")[0]
      dir_name = Path(remote_dir).name
      local_wandb_dir = Path(config.log_dir) / "wandb"
      local_wandb_dir.mkdir(parents=True, exist_ok=True)
      logger.info(f"Importing wandb run from {config.remote}:{remote_dir}")
      subprocess.run(
        ["scp", "-r", f"{config.remote}:{remote_dir}", str(local_wandb_dir / dir_name)],
        check=True,
      )
    else:
      logger.warning(f"No wandb run directory found for ID '{wandb_id}' on remote")
  else:
    logger.warning("No wandb_id.txt found, skipping wandb data import")

  logger.success(f"Imported '{config.run}' from {config.remote}")


if __name__ == "__main__":
  main()
