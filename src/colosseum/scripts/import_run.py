#!/usr/bin/env python3
"""Import the most recent checkpoint and run metadata from a remote machine via SCP.

Usage:
  pixi run import-run --remote gin --root /home/phd_student/Maiorana/colosseum --run t1-vel_ppo_20260101_120000

With ONNX export:
  pixi run import-run --remote gin --root /home/phd_student/Maiorana/colosseum --run t1-vel_ppo_20260101_120000 --task t1-velocity --name jun14_2 --destination-root ~/code/spqr/arena
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import tyro
from loguru import logger
from pydantic.dataclasses import dataclass

from colosseum.config.types.experiment import BaseExperimentConfig
from colosseum.config.types.task import get_task
from colosseum.scripts.export_gains import export_gains
from colosseum.utils.export import export_policy_to_onnx
from colosseum.utils.model_registry import ModelRegistry


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

  no_wandb: bool = False
  """Skip importing wandb run data."""

  task: str = ""
  """Task name for ONNX export (e.g. 't1-velocity')."""

  name: str = ""
  """Registration name for ONNX export (e.g. 'jun14_2')."""

  destination_root: str | None = None
  """Absolute path to an arena checkout. Copies model and updates registry."""


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
  if not config.no_wandb:
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
  else:
    logger.info("Skipping wandb import (--no-wandb)")

  logger.success(f"Imported '{config.run}' from {config.remote}")

  # ONNX export
  if config.task and config.name:
    logger.info("Starting ONNX export...")
    _export_onnx(config, local_run, local_ckpt_dir / ckpt_name)
    logger.success(f"Exported '{config.name}' for task '{config.task}'")
  elif config.task or config.name:
    logger.warning("Both --task and --name are required for ONNX export, skipping")


def _export_onnx(config: ImportRunConfig, run_dir: Path, ckpt_path: Path) -> None:
  import colosseum.tasks  # noqa: F401  # populate task registry

  task_cfg = get_task(config.task)
  algo_cfg = task_cfg.algo_cfg
  assert algo_cfg is not None, (
    f"Task '{config.task}' has no algo_cfg. "
    "Implement the algo_cfg property in the task's __init__.py."
  )
  algo = algo_cfg.name.lower()
  task_name = config.task

  ckpt_path = ckpt_path.resolve()
  stem = ckpt_path.stem
  step_str = stem.split("_")[-1] if "_" in stem else stem
  try:
    step = int(step_str)
  except ValueError:
    step = -1

  filename = f"{task_name}_{algo}_{config.name}.onnx"

  task_dir = Path("models") / task_name
  out_dir = task_dir / config.name
  out_dir.mkdir(parents=True, exist_ok=True)
  output_path = out_dir / filename

  exp_cfg = BaseExperimentConfig(task=task_cfg)
  result = export_policy_to_onnx(exp_cfg, ckpt_path, output_path)

  # Copy .pt checkpoint into the subfolder
  pt_name = result.stem + ".pt"
  shutil.copy2(ckpt_path, out_dir / pt_name)
  logger.info(f"Copied checkpoint: {out_dir / pt_name}")

  # Copy config.yaml from the run directory
  config_path = run_dir / "config.yaml"
  if config_path.exists():
    shutil.copy2(config_path, out_dir / "config.yaml")
    logger.info(f"Copied config: {out_dir / 'config.yaml'}")
    # Regenerate the deploy gains.yaml from the fresh config (never leave it stale).
    export_gains(out_dir)

  # Register in models/registry.yaml
  rel_file = f"{config.name}/{result.name}"
  ModelRegistry.register(
    task=task_name,
    name=config.name,
    file=rel_file,
    run=config.run,
    step=step if step >= 0 else 0,
  )
  _set_default_symlink(task_dir, rel_file)

  if config.destination_root:
    script = Path(__file__).resolve().parent / "export_model.sh"
    subprocess.run(
      [
        str(script),
        "--policy", task_name,
        "--version", config.name,
        "--destination-root", str(Path(config.destination_root).expanduser()),
      ],
      check=True,
    )
    logger.info(f"Exported to destination: {config.destination_root}")


def _set_default_symlink(task_dir: Path, rel_file: str) -> None:
  link = task_dir / "default.onnx"
  if link.is_symlink() or link.exists():
    link.unlink()
  link.symlink_to(rel_file)


if __name__ == "__main__":
  main()
