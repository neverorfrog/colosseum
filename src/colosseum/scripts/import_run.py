#!/usr/bin/env python3
"""Import the latest checkpoint from a remote machine, export to ONNX, and push to DVC.

Usage:
  pixi run import-run --remote gin --root /home/phd_student/Maiorana/colosseum --run t1-vel_ppo_20260101_120000 --task t1-velocity

  # Named version (defaults to "latest"):
  pixi run import-run --remote gin --root ... --run ... --task t1-velocity --name v2

  # Skip DVC:
  pixi run import-run ... --task t1-velocity --no-dvc

  # Keep raw checkpoint in logs/:
  pixi run import-run ... --task t1-velocity --keep-checkpoint
"""

from __future__ import annotations

import shutil
import subprocess
import sys
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

  task: str = ""
  """Task name for ONNX export (e.g. 't1-velocity'). Triggers export + DVC push."""

  name: str = "latest"
  """Registration name for the exported model (default: 'latest')."""

  log_dir: str = "./logs"
  """Local log directory (default: ./logs)."""

  keep_checkpoint: bool = False
  """Keep the raw .pt checkpoint in logs/ after export."""

  no_dvc: bool = False
  """Skip DVC add and push."""

  wandb: bool = False
  """Import wandb run data from the remote."""

  destination_root: str | None = None
  """Absolute path to an arena checkout. Copies model and updates registry."""


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

  ckpt_path = local_ckpt_dir / ckpt_name

  # Import checkpoint
  subprocess.run(
    ["scp", f"{config.remote}:{latest}", str(ckpt_path)],
    check=True,
  )

  # Import config.yaml (needed for ONNX export gains)
  subprocess.run(
    ["scp", f"{config.remote}:{remote_run / 'config.yaml'}", str(local_run / "config.yaml")],
    capture_output=True,
    text=True,
  )

  # Wandb import
  if config.wandb:
    _import_wandb(config, remote_root, local_run)

  logger.success(f"Imported '{config.run}' from {config.remote}")

  # ONNX export + DVC
  if config.task:
    if not config.no_dvc:
      _check_dvc_remote()

    _export_onnx(config, local_run, ckpt_path)

    if not config.no_dvc:
      _dvc_push(config)

    logger.success(f"Exported '{config.name}' for task '{config.task}'")

    if not config.keep_checkpoint:
      shutil.rmtree(local_run)
      logger.info(f"Cleaned up {local_run}")


def _import_wandb(config: ImportRunConfig, remote_root: Path, local_run: Path) -> None:
  wandb_id_path = local_run / "wandb_id.txt"
  subprocess.run(
    ["scp", f"{config.remote}:{remote_root / 'logs' / config.run / 'wandb_id.txt'}", str(wandb_id_path)],
    capture_output=True,
    text=True,
  )
  if not wandb_id_path.exists():
    logger.warning("No wandb_id.txt found on remote, skipping wandb import")
    return

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


def _check_dvc_remote() -> None:
  result = subprocess.run(["dvc", "remote", "list"], capture_output=True, text=True)
  if not result.stdout.strip():
    logger.error("No DVC remote configured. Run: dvc remote add -d myremote <url>")
    sys.exit(1)


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

  pt_name = result.stem + ".pt"
  shutil.copy2(ckpt_path, out_dir / pt_name)
  logger.info(f"Copied checkpoint: {out_dir / pt_name}")

  config_path = run_dir / "config.yaml"
  if config_path.exists():
    shutil.copy2(config_path, out_dir / "config.yaml")
    logger.info(f"Copied config: {out_dir / 'config.yaml'}")
    export_gains(out_dir)

  rel_file = f"{config.name}/{result.name}"
  ModelRegistry.register(
    task=task_name,
    name=config.name,
    file=rel_file,
    run=config.run,
    step=step if step >= 0 else 0,
  )
  _set_default_symlink(task_dir, rel_file)

  remote_out = config.root and Path(config.root) / "models" / task_name / config.name
  if config.remote and remote_out:
    subprocess.run(
      ["ssh", config.remote, f"mkdir -p {remote_out}"],
      check=True,
    )
    subprocess.run(
      ["scp", "-r", f"{out_dir}/", f"{config.remote}:{remote_out.parent}/"],
      check=True,
    )
    logger.info(f"Uploaded to remote: {config.remote}:{remote_out}")

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


def _dvc_push(config: ImportRunConfig) -> None:
  logger.info("Running dvc add models/ ...")
  subprocess.run(["dvc", "add", "models/"], check=True)
  logger.info("Running dvc push ...")
  subprocess.run(["dvc", "push"], check=True)
  logger.info(
    "Done. Now commit the metadata and push to git:\n"
    "  git add models.dvc && git commit -m 'update {} {}' && git push",
    config.task,
    config.name,
  )


def _set_default_symlink(task_dir: Path, rel_file: str) -> None:
  link = task_dir / "default.onnx"
  if link.is_symlink() or link.exists():
    link.unlink()
  link.symlink_to(rel_file)


if __name__ == "__main__":
  main()
