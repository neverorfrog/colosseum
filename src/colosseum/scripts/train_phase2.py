#!/usr/bin/env python3
"""Phase 2 RMA adaptation encoder training.

Loads a Phase 1 RmaPPO checkpoint, freezes actor/critic/privileged encoders,
and trains the adaptation encoder(s) via MSE regression against frozen
privileged encoder targets.

Usage:
    pixi run train-phase2 \
        --checkpoint ./logs/run/checkpoints/latest.pt \
        task:t1-velocity-rma

    pixi run train-phase2 \
        --checkpoint ./logs/run/checkpoints/latest.pt \
        --adaptation-lr 3e-4 \
        task:t1-velocity-rma logger:wandb
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
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


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Phase2Config(TrainConfig):
  """Phase 2 RMA adaptation training config.

  ``--checkpoint`` (inherited from TrainConfig) is required; the script
  exits early if it is not provided.
  """

  adaptation_lr: float = 1e-3
  """Learning rate for the adaptation encoder optimizer."""

  save_interval: int = 10_000
  """Checkpoint save interval in environment steps."""

  adaptation_epochs: int | None = None
  """Override algo_cfg.num_learning_epochs for Phase 2 regression (default: use algo_cfg value).
  Phase 1 PPO typically uses 8 epochs; for Phase 2 MSE regression 1-2 is usually enough."""


def _parse_cuda_devices(cuda_arg: str) -> list[int]:
  parts = [p.strip() for p in str(cuda_arg).split(",") if p.strip()]
  if not parts:
    raise ValueError("--cuda must contain at least one GPU id, e.g. --cuda 0")
  devices: list[int] = []
  for part in parts:
    try:
      dev = int(part)
    except ValueError as exc:
      raise ValueError(f"Invalid --cuda value '{cuda_arg}'.") from exc
    if dev < 0:
      raise ValueError(f"--cuda ids must be >= 0, got {dev}")
    devices.append(dev)
  if len(set(devices)) != len(devices):
    raise ValueError(f"Duplicate GPU ids in --cuda: {cuda_arg}")
  return devices


def _init_distributed(use_cuda: bool) -> tuple[bool, int, int, int]:
  world_size = int(os.environ.get("WORLD_SIZE", "1"))
  rank = int(os.environ.get("RANK", "0"))
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  is_distributed = world_size > 1
  if not is_distributed:
    return False, 1, 0, 0
  if use_cuda and not torch.cuda.is_available():
    raise RuntimeError("Distributed training requested but CUDA is not available.")
  if use_cuda:
    torch.cuda.set_device(local_rank)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
  if not dist.is_initialized():
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
  return True, world_size, rank, local_rank


def _teardown_distributed() -> None:
  if dist.is_available() and dist.is_initialized():
    dist.destroy_process_group()


def main() -> None:
  config: Phase2Config = tyro.cli(
    Phase2Config,
    config=(tyro.conf.CascadeSubcommandArgs,),
  )
  cuda_devices = _parse_cuda_devices(config.cuda)

  pre_world_size = int(os.environ.get("WORLD_SIZE", "1"))
  relaunched = os.environ.get("COLOSSEUM_TRAIN_RELAUNCHED") == "1"
  if config.use_cuda and pre_world_size <= 1:
    if len(cuda_devices) == 1:
      os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices[0])
    elif not relaunched:
      env = os.environ.copy()
      env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_devices))
      env["COLOSSEUM_TRAIN_RELAUNCHED"] = "1"
      cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={len(cuda_devices)}",
        "-m", "colosseum.scripts.train_phase2",
        *sys.argv[1:],
      ]
      print(f"[INFO] Relaunching phase2 with torchrun on GPUs {env['CUDA_VISIBLE_DEVICES']}", flush=True)
      proc = subprocess.Popen(cmd, env=env, start_new_session=True)
      try:
        raise SystemExit(proc.wait())
      except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received, stopping torchrun workers...", flush=True)
        try:
          os.killpg(proc.pid, signal.SIGTERM)
          proc.wait(timeout=10)
        except Exception:
          try:
            os.killpg(proc.pid, signal.SIGKILL)
          except Exception:
            pass
        raise SystemExit(130)

  is_distributed = False
  world_size = 1
  rank = 0
  local_rank = 0

  try:
    is_distributed, world_size, rank, local_rank = _init_distributed(config.use_cuda)
    is_main_process = rank == 0

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

    from dataclasses import replace as dc_replace
    if config.learning_steps is not None:
      algo_cfg = dc_replace(algo_cfg, learning_steps=config.learning_steps)
    if config.adaptation_epochs is not None:
      algo_cfg = dc_replace(algo_cfg, num_learning_epochs=config.adaptation_epochs)

    env_cfg: ManagerBasedRlEnvCfg = config.task.train_env_cfg

    run_name = config.logger.name or generate_run_name(
      task_name=config.task.name,
      algo_name=f"{algo_cfg.name}_phase2",
      seed=config.seed,
    )

    logger_cfg = replace(
      config.logger,
      project=f"colosseum-{config.task.name}",
      group=f"{algo_cfg.name}_phase2",
      job_type=f"{algo_cfg.name}_phase2_{config.task.name}",
    )

    try:
      wandb_run, run_dir = setup_wandb(
        logger_cfg, logger_cfg.log_dir, run_name, is_main_process=is_main_process
      )
      if run_dir is None:
        run_dir = Path(logger_cfg.log_dir) / run_name
      if is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
      else:
        run_dir = run_dir / f"rank_{rank}"
        run_dir.mkdir(parents=True, exist_ok=True)

      setup_loguru(run_dir, is_main_process=is_main_process, console_level=logger_cfg.console_level)

      logger.info("=" * 80)
      logger.info("RMA Phase 2 — Adaptation Encoder Training")
      logger.info(f"Task:          {config.task.name}")
      logger.info(f"Algorithm:     {algo_cfg.name}")
      logger.info(f"Phase 1 ckpt:  {checkpoint_path}")
      logger.info(f"Adapt LR:      {config.adaptation_lr}")
      logger.info(f"Seed:          {config.seed}")
      logger.info(f"Num envs:      {env_cfg.scene.num_envs}")
      logger.info(f"Distributed:   {is_distributed} (world={world_size}, rank={rank})")
      logger.info(f"Run dir:       {run_dir}")
      logger.info("=" * 80)

      if is_main_process:
        save_experiment_config(config, run_dir, wandb_run)

      def log_fn(metrics: dict[str, float], step: int) -> None:
        if is_main_process and wandb_run is not None:
          wandb.log(metrics, step=step)

    except Exception as e:
      print(f"Failed to set up logging: {e}")
      sys.exit(1)

    rank_seed = config.seed + rank
    set_seed(rank_seed)
    configure_torch_backends()

    device_id = local_rank if is_distributed else 0
    if config.use_cuda and not is_distributed:
      os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_id)
    device = get_device(cuda=config.use_cuda, device_id=device_id)
    logger.info(f"Using device: {device}")

    env = env_cfg.class_type(cfg=env_cfg, device=str(device))

    import importlib
    module_path, class_name = algo_cfg.target.rsplit(":", 1)
    module = importlib.import_module(module_path)
    algo_class = getattr(module, class_name)

    from colosseum.algorithm.rma_ppo import RmaPPO
    if not issubclass(algo_class, RmaPPO):
      logger.error(
        f"Phase 2 requires an RmaPPO algorithm, got {algo_class.__name__}."
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
      seed=rank_seed,
      base_seed=config.seed,
      device=str(device),
      rank=rank,
      world_size=world_size,
      phase=2,
      phase1_checkpoint=str(checkpoint_path),
    )

    if is_main_process and run_dir is not None and config.save_interval > 0:
      ckpt_dir = run_dir / "checkpoints"
      algo.configure_checkpointing(ckpt_dir, config.save_interval)

    logger.info(f"Loading Phase 1 checkpoint: {checkpoint_path}")
    algo.load(checkpoint_path)
    algo.global_step = 0
    logger.info("Phase 1 checkpoint loaded; global_step reset to 0 for Phase 2")

    if is_distributed and dist.is_initialized():
      dist.barrier()

    algo.build_adaptation_optimizer(lr=config.adaptation_lr)
    logger.info(
      f"Adaptation optimizer built (lr={config.adaptation_lr}). "
      "Actor, critic, and privileged encoders frozen."
    )

    logger.info("Starting Phase 2 adaptation encoder training...")
    try:
      algo.train()
    except KeyboardInterrupt:
      logger.warning("Training interrupted (Ctrl+C)")
      if is_main_process:
        assert run_dir is not None
        interrupt_path = run_dir / "checkpoints" / "interrupted.pt"
        interrupt_path.parent.mkdir(parents=True, exist_ok=True)
        try:
          algo.save(interrupt_path, global_step=algo.global_step)
          logger.success(f"Saved interrupted checkpoint: {interrupt_path}")
        except Exception as e:
          logger.error(f"Failed to save checkpoint: {e}")

    if is_main_process:
      teardown_wandb()
    logger.success("Phase 2 training complete!")
  finally:
    _teardown_distributed()


if __name__ == "__main__":
  main()
