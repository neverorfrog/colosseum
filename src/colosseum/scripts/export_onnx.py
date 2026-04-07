#!/usr/bin/env python3
"""Export a colosseum PPO checkpoint to ONNX.

Output is placed next to the checkpoint as <task_name>_<algo_name>.onnx.

Usage:
    pixi run -e train export-onnx task:t1-velocity-flat
    pixi run -e train export-onnx task:t1-velocity-rough --checkpoint ./wandb/latest-run/files/model_16000.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import tyro
from loguru import logger
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from colosseum.algorithm.base_algorithm import get_latest_checkpoint
from colosseum.config.types.experiment import BaseExperimentConfig
from colosseum.utils.train.export import export_policy_to_onnx


def _resolve_checkpoint(checkpoint: str | None) -> Path | None:
  """Resolve checkpoint path, supporting 'latest', directories, and direct paths."""
  if not checkpoint or checkpoint.lower() == "latest":
    ckpt_dir = Path("./logs/wandb/latest-run/checkpoints")
    if ckpt_dir.exists():
      return get_latest_checkpoint(ckpt_dir.resolve())
    return None

  p = Path(checkpoint)
  if p.is_dir():
    return get_latest_checkpoint(p.resolve())
  return p.resolve() if p.exists() else None


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class ExportConfig(BaseExperimentConfig):
  """Export configuration."""

  checkpoint: str = "latest"


def main() -> None:
  """Export a trained policy checkpoint to ONNX."""
  config = tyro.cli(ExportConfig, config=(tyro.conf.CascadeSubcommandArgs,))

  ckpt = _resolve_checkpoint(config.checkpoint)
  if ckpt is None or not ckpt.exists():
    logger.error(f"No checkpoint found for: {config.checkpoint}")
    sys.exit(1)

  # Export to project-level models/ directory
  models_dir = Path("models")
  models_dir.mkdir(exist_ok=True)

  # Resolve symlinks so we get the real checkpoint name (e.g. latest.pt → model_0099483648.pt)
  ckpt = ckpt.resolve()
  stem = ckpt.stem
  step = stem.split("_")[-1] if "_" in stem else stem
  algo_cfg = config.task.algo_cfg
  assert algo_cfg is not None, (
    f"Task '{config.task.name}' has no algo_cfg. "
    "Implement the algo_cfg property in the task's __init__.py."
  )
  base_name = f"{config.task.name}_{algo_cfg.name.lower()}"
  filename = f"{base_name}_{step}.onnx"
  output_path = models_dir / filename

  result = export_policy_to_onnx(config, ckpt, output_path)

  # Create a stable latest symlink for deploy configs to reference
  latest_link = models_dir / f"{base_name}_latest.onnx"
  try:
    if latest_link.exists() or latest_link.is_symlink():
      latest_link.unlink()
    latest_link.symlink_to(result.name)
    logger.info(f"Symlink: {latest_link.name} -> {result.name}")
  except OSError:
    pass

  logger.success(f"Exported: {result}")


if __name__ == "__main__":
  main()
