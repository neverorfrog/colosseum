#!/usr/bin/env python3
"""Export a colosseum checkpoint to ONNX. Every export is registered.

  pixi run export-onnx task:t1-velocity --name v1 --run-name t1-vel_ppo_20260101_120000
  pixi run export-onnx task:t1-velocity --name v1 --checkpoint path/to/model_1000.pt
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import tyro
from loguru import logger
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from colosseum.config.types.experiment import BaseExperimentConfig
from colosseum.utils.checkpoint import resolve_checkpoint
from colosseum.utils.export import export_policy_to_onnx
from colosseum.utils.model_registry import ModelRegistry


def _set_default_symlink(task_dir: Path, rel_file: str) -> None:
  """Point <task_dir>/default.onnx -> rel_file."""
  link = task_dir / "default.onnx"
  if link.is_symlink() or link.exists():
    link.unlink()
  link.symlink_to(rel_file)


def _run_name_from_ckpt(ckpt: Path, log_dir: str = "./logs") -> str | None:
  """Extract run directory name from checkpoint path."""
  ckpt = ckpt.resolve()
  try:
    ckpt.relative_to(Path(log_dir).resolve())
  except ValueError:
    return None
  for parent in ckpt.parents:
    if parent.name == "checkpoints":
      return parent.parent.name
  return None


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class ExportConfig(BaseExperimentConfig):
  """Export configuration."""

  checkpoint: str = "latest"
  """Checkpoint path or 'latest'."""

  run_name: str | None = None
  """Run name to export from (looks in ./logs/<run_name>/checkpoints/)."""

  name: str = ""
  """Registration name (e.g. 'v1'). Required."""

  out_dir: str | None = None
  """Output directory (default: models/<task_name>/)."""

  destination_root: str | None = None
  """Absolute path to an arena checkout. After export, calls export_model.sh to copy the
  model and update the destination's model registry."""

  def __post_init__(self) -> None:
    if self.destination_root is not None and not Path(self.destination_root).is_absolute():
      raise ValueError("--destination-root must be an absolute path")


def main() -> None:
  """Export a trained policy checkpoint to ONNX."""
  config = tyro.cli(ExportConfig, config=(tyro.conf.CascadeSubcommandArgs,))

  ckpt = resolve_checkpoint(config.checkpoint, config.run_name)
  if ckpt is None or not ckpt.exists():
    display = f"run_name={config.run_name}" if config.run_name else f"checkpoint={config.checkpoint}"
    logger.error(f"No checkpoint found for {display}")
    sys.exit(1)

  task_name = config.task.name

  algo_cfg = config.task.algo_cfg
  assert algo_cfg is not None, (
    f"Task '{task_name}' has no algo_cfg. "
    "Implement the algo_cfg property in the task's __init__.py."
  )
  algo = algo_cfg.name.lower()
  ckpt = ckpt.resolve()

  stem = ckpt.stem
  step_str = stem.split("_")[-1] if "_" in stem else stem
  try:
    step = int(step_str)
  except ValueError:
    step = -1

  if not config.name:
    logger.error("--name is required (e.g. --name v1)")
    sys.exit(1)

  filename = f"{task_name}_{algo}_{config.name}.onnx"

  task_dir = Path(config.out_dir) if config.out_dir else Path("models") / task_name
  out_dir = task_dir / config.name
  out_dir.mkdir(parents=True, exist_ok=True)
  output_path = out_dir / filename

  result = export_policy_to_onnx(config, ckpt, output_path)

  # Copy .pt checkpoint into the subfolder
  pt_name = result.stem + ".pt"
  shutil.copy2(ckpt, out_dir / pt_name)
  logger.info(f"Copied checkpoint: {out_dir / pt_name}")

  # Copy config.yaml from the run directory
  run_dir = _run_name_from_ckpt(ckpt)
  if run_dir:
    config_path = Path("./logs") / run_dir / "config.yaml"
    if config_path.exists():
      shutil.copy2(config_path, out_dir / "config.yaml")
      logger.info(f"Copied config: {out_dir / 'config.yaml'}")

  # Register in models/registry.yaml
  rel_file = f"{config.name}/{result.name}"
  run_name = config.run_name or _run_name_from_ckpt(ckpt) or ""
  ModelRegistry.register(
    task=task_name,
    name=config.name,
    file=rel_file,
    run=run_name,
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
        "--destination-root", config.destination_root,
      ],
      check=True,
    )
    logger.info(f"Exported to destination: {config.destination_root}")

  logger.success(f"Exported + registered '{config.name}': {result}")


if __name__ == "__main__":
  main()
