#!/usr/bin/env python3
"""Phase 2 RMA adaptation encoder training.

Loads a Phase 1 RmaPPO checkpoint, freezes the actor/critic/privileged encoders,
and trains the adaptation encoders (e.g. depth CNN+LSTM) via MSE regression
against the frozen privileged encoder targets.

Usage:
    pixi run -e train train-phase2 \\
        --checkpoint ./logs/run/checkpoints/latest.pt \\
        task:t1-dribbling

    pixi run -e train train-phase2 \\
        --checkpoint ./logs/run/checkpoints/latest.pt \\
        --adaptation-lr 3e-4 \\
        task:t1-dribbling logger:wandb
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import tyro
import wandb
from loguru import logger
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.utils.torch import configure_torch_backends
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

# Import tasks to populate registry
import colosseum.tasks  # noqa: F401

from colosseum.config.types.experiment import TrainConfig
from colosseum.utils.logger import (
  generate_run_name,
  save_experiment_config,
  setup_loguru,
  setup_wandb,
  teardown_wandb,
)
from colosseum.utils.torch import get_device, set_seed
from colosseum.utils.train.env import make_env


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Phase2Config(TrainConfig):
  """Phase 2 RMA adaptation training — same CLI as train, plus adaptation_lr.

  ``--checkpoint`` (inherited from TrainConfig) is required here; the script
  exits early if it is not provided.
  """

  adaptation_lr: float = 1e-3
  """Learning rate for the adaptation encoder optimizer."""

  save_interval: int = 10_000
  """Checkpoint save interval in environment steps. Overrides logger.save_interval.
  Smaller default than Phase 1 since the adaptation encoder converges quickly."""


def main() -> None:
  """Phase 2 RMA adaptation encoder training entry point."""
  config: Phase2Config = tyro.cli(
    Phase2Config,
    config=(tyro.conf.CascadeSubcommandArgs,),
  )

  if config.checkpoint is None:
    logger.error("--checkpoint is required for Phase 2 training.")
    sys.exit(1)

  checkpoint_path = Path(config.checkpoint)
  if not checkpoint_path.exists():
    logger.error(f"Phase 1 checkpoint not found: {checkpoint_path}")
    sys.exit(1)

  algo_cfg = config.task.algo_cfg
  assert algo_cfg is not None, (
    f"Task '{config.task.name}' has no algo_cfg. "
    "Implement the algo_cfg property in the task's __init__.py."
  )

  env_cfg: ManagerBasedRlEnvCfg = config.task.train_env_cfg

  run_name = generate_run_name(
    task_name=config.task.name,
    algo_name=f"{algo_cfg.name}_phase2",
    seed=config.seed,
  )

  logger_cfg = config.logger
  if logger_cfg.group is None:
    from dataclasses import replace
    logger_cfg = replace(logger_cfg, group=config.task.name)

  try:
    wandb_run, run_dir = setup_wandb(
      logger_cfg, logger_cfg.log_dir, run_name, is_main_process=True
    )

    if run_dir is None:
      run_dir = Path(logger_cfg.log_dir) / run_name
      run_dir.mkdir(parents=True, exist_ok=True)

    setup_loguru(run_dir, is_main_process=True, console_level=logger_cfg.console_level)

    logger.info("=" * 80)
    logger.info("RMA Phase 2 — Adaptation Encoder Training")
    logger.info(f"Task:            {config.task.name}")
    logger.info(f"Algorithm:       {algo_cfg.name}")
    logger.info(f"Phase 1 ckpt:    {checkpoint_path}")
    logger.info(f"Adaptation LR:   {config.adaptation_lr}")
    logger.info(f"Seed:            {config.seed}")
    logger.info(f"Num envs:        {env_cfg.scene.num_envs}")
    logger.info(f"Run directory:   {run_dir}")
    logger.info("=" * 80)

    save_experiment_config(config, run_dir, wandb_run)

    def log_fn(metrics: dict[str, float], step: int) -> None:
      if wandb_run is not None:
        wandb.log(metrics, step=step)

  except Exception as e:
    print(f"Failed to set up logging: {e}")
    sys.exit(1)

  set_seed(config.seed)
  configure_torch_backends()
  if config.cuda < 0:
    raise ValueError(f"--cuda must be >= 0, got {config.cuda}")
  device_id = config.cuda
  if config.use_cuda:
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_id)
  device = get_device(cuda=config.use_cuda, device_id=device_id)
  logger.info(f"Using device: {device}")

  env = make_env(env_cfg=env_cfg, device=str(device))

  import importlib
  module_path, class_name = algo_cfg.target.rsplit(":", 1)
  module = importlib.import_module(module_path)
  algo_class = getattr(module, class_name)

  from colosseum.algorithm.rma_ppo import RmaPPO
  if not issubclass(algo_class, RmaPPO):
    logger.error(
      f"Phase 2 training requires an RmaPPO algorithm, got {algo_class.__name__}. "
      "Check the task's algo_cfg."
    )
    sys.exit(1)

  algo: RmaPPO = algo_class(
    config=algo_cfg,
    env=env,
    device=device,
    log_fn=log_fn,
    log_interval=logger_cfg.log_interval,
  )

  algo.attach_metadata(
    experiment_config=config.to_serializable_dict(),
    wandb_run_id=wandb_run.id if wandb_run is not None else None,
    timestamp=datetime.now().isoformat(),
    seed=config.seed,
    device=str(device),
    phase=2,
    phase1_checkpoint=str(checkpoint_path),
  )

  if run_dir is not None and config.save_interval > 0:
    ckpt_dir = run_dir / "checkpoints"
    algo.configure_checkpointing(ckpt_dir, config.save_interval)

  logger.info(f"Loading Phase 1 checkpoint: {checkpoint_path}")
  state = algo.load(checkpoint_path)
  logger.info(f"Phase 1 checkpoint loaded (step {state.get('global_step', 0)})")

  algo.build_adaptation_optimizer(lr=config.adaptation_lr)
  logger.info(
    f"Adaptation optimizer built (lr={config.adaptation_lr}). "
    "Actor, critic, and privileged encoders are frozen."
  )

  algo._phase = 2
  logger.info("Starting Phase 2 adaptation encoder training...")

  try:
    algo.train()
  except KeyboardInterrupt:
    logger.warning("Training interrupted by user (Ctrl+C)")
    assert run_dir is not None
    interrupt_path = run_dir / "checkpoints" / "interrupted.pt"
    interrupt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
      algo.save(interrupt_path, global_step=algo.global_step)
      logger.success(f"Saved interrupted checkpoint: {interrupt_path}")
    except Exception as e:
      logger.error(f"Failed to save checkpoint: {e}")

  teardown_wandb()
  logger.success("Phase 2 training complete!")


if __name__ == "__main__":
  main()
