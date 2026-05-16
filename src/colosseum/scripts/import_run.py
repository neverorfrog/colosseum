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

  logger.success(f"Imported '{config.run}' from {config.remote}")


if __name__ == "__main__":
  main()
