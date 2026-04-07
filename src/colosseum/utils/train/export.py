"""Training-side policy export utilities (ONNX conversion).

Loads a colosseum PPO checkpoint and exports the actor to ONNX
using BaseAlgorithm.export_onnx().
"""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path

from loguru import logger

# Import tasks to populate registry
import colosseum.tasks  # noqa: F401
from colosseum.config.types.experiment import BaseExperimentConfig
from colosseum.utils.torch import get_device
from colosseum.utils.train.env import make_env


def export_policy_to_onnx(
  config: BaseExperimentConfig,
  checkpoint_path: str | Path,
  output_path: str | Path,
) -> Path:
  """Load a colosseum checkpoint and export the actor to ONNX.

  Args:
      config: Experiment config (task + algo) matching the checkpoint.
      checkpoint_path: Path to the .pt checkpoint.
      output_path: Where to write the .onnx file.

  Returns:
      Resolved path to the written ONNX file.
  """
  checkpoint_path = Path(checkpoint_path)
  output_path = Path(output_path)

  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

  device = get_device(cuda=False, device_id=0)
  env_cfg = config.task.env
  # Single env is enough to get dimensions
  env_cfg = replace(env_cfg, scene=replace(env_cfg.scene, num_envs=1))
  env = make_env(env_cfg, str(device))

  algo_cfg = config.task.algo_cfg
  assert algo_cfg is not None, (
    f"Task '{config.task.name}' has no algo_cfg. "
    "Implement the algo_cfg property in the task's __init__.py."
  )

  module_path, class_name = algo_cfg.target.rsplit(":", 1)
  module = importlib.import_module(module_path)
  algo_class = getattr(module, class_name)

  algo = algo_class(
    config=algo_cfg,
    env=env,
    device=device,
    log_fn=lambda _m, _s: None,
    log_interval=-1,
  )

  state = algo.load(checkpoint_path)
  step = state.get("global_step", "?")
  logger.info(f"Loaded checkpoint from step {step}")

  result = algo.export_onnx(output_path)
  env.close()
  return result


__all__ = ["export_policy_to_onnx"]
