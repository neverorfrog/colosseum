#!/usr/bin/env python3
"""Integrated RMA training pipeline: Phase 1 → Phase 2 → Phase 3.

Each phase runs with its own env+algo instance created from a checkpoint.
Phase transitions are automatic; checkpoints are saved at the end of each phase.

Usage:
    pixi run -e train rma-train task:t1-velocity
    pixi run -e train rma-train task:t1-velocity --phase2-lr 3e-4
    pixi run -e train rma-train task:t1-velocity --phase2-num-envs 1024
    pixi run -e train rma-train task:t1-velocity --phase2-loss-threshold 0.01
    pixi run -e train rma-train task:t1-velocity --start-phase 2 --checkpoint ./logs/.../phase1_final.pt
    pixi run -e train rma-train task:t1-velocity --start-phase 3 --checkpoint ./logs/.../phase2_final.pt
"""

from __future__ import annotations

import gc
import importlib
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
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.utils.torch import configure_torch_backends
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

import colosseum.tasks  # noqa: F401
from colosseum.algorithm.rma_ppo import RmaPPO
from colosseum.config.types.experiment import TrainConfig
from colosseum.utils.logger import (
  generate_run_name,
  save_experiment_config,
  setup_loguru,
  setup_wandb,
  start_live_display,
  stop_live_display,
  teardown_wandb,
)
from colosseum.utils.torch import get_device, set_seed


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class RmaTrainConfig(TrainConfig):
  start_phase: int = 1
  """Which phase to begin from (1, 2, or 3).  Use with --checkpoint to resume."""

  phase2_steps: int = 50_000_000
  """Env steps for Phase 2 (adaptation encoder regression)."""

  phase2_lr: float = 1e-3
  """Learning rate for the Phase 2 adaptation encoder optimizer."""

  phase2_loss_threshold: float | None = None
  """Early-stop Phase 2 when the smoothed latent_mse falls below this value."""

  phase3_steps: int = 250_000_000
  """Env steps for Phase 3 (policy fine-tuning with frozen encoders)."""

  phase2_num_envs: int | None = None
  """Number of parallel environments for Phase 2. Defaults to the same as Phase 1/3.
  Reduce this if Phase 2 (depth-buffer regression) causes GPU OOM."""


# ---------------------------------------------------------------------------
# Distributed helpers (same as train.py / train_phase2.py)
# ---------------------------------------------------------------------------


def _parse_cuda_devices(cuda_arg: str) -> list[int]:
  parts = [p.strip() for p in str(cuda_arg).split(",") if p.strip()]
  if not parts:
    raise ValueError("--cuda must contain at least one GPU id")
  devices: list[int] = []
  for part in parts:
    try:
      dev = int(part)
    except ValueError as exc:
      raise ValueError(f"Invalid --cuda value '{cuda_arg}'") from exc
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
  config: RmaTrainConfig = tyro.cli(
    RmaTrainConfig,
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
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(cuda_devices)}",
        "-m",
        "colosseum.scripts.rma_train",
        *sys.argv[1:],
      ]
      print(
        f"[INFO] Relaunching rma-train with torchrun on GPUs {env['CUDA_VISIBLE_DEVICES']}",
        flush=True,
      )
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

    def _barrier() -> None:
      if not (is_distributed and dist.is_initialized()):
        return
      # NCCL needs the device id to avoid serializing on the wrong GPU.
      if config.use_cuda:
        dist.barrier(device_ids=[local_rank])
      else:
        dist.barrier()

    algo_cfg = config.task.algo_cfg
    assert algo_cfg is not None, f"Task '{config.task.name}' has no algo_cfg."

    phase1_steps = config.learning_steps or algo_cfg.learning_steps
    env_cfg = config.task.train_env_cfg

    # Phase 2 may run with a different number of environments
    if config.phase2_num_envs is not None:
      phase2_scene = replace(env_cfg.scene, num_envs=config.phase2_num_envs)
      phase2_env_cfg: ManagerBasedRlEnvCfg = replace(env_cfg, scene=phase2_scene)
    else:
      phase2_env_cfg = env_cfg

    run_name = config.logger.name or generate_run_name(
      task_name=config.task.name,
      algo_name=f"{algo_cfg.name}",
      seed=config.seed,
    )

    logger_cfg = replace(
      config.logger,
      project=f"colosseum-{config.task.name}",
      group=f"{algo_cfg.name}",
      job_type=f"{algo_cfg.name}_rma_{config.task.name}",
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

      setup_loguru(
        run_dir, is_main_process=is_main_process, console_level=logger_cfg.console_level
      )

      logger.info("=" * 80)
      logger.info("RMA Integrated Training Pipeline")
      logger.info(f"Task:          {config.task.name}")
      logger.info(f"Algorithm:     {algo_cfg.name}")
      logger.info(f"Start phase:   {config.start_phase}")
      logger.info(f"Phase 1 steps: {phase1_steps}  num_envs={env_cfg.scene.num_envs}")
      logger.info(
        f"Phase 2 steps: {config.phase2_steps}  lr={config.phase2_lr}"
        f"  threshold={config.phase2_loss_threshold}"
        f"  num_envs={phase2_env_cfg.scene.num_envs}"
      )
      logger.info(
        f"Phase 3 steps: {config.phase3_steps}  num_envs={phase2_env_cfg.scene.num_envs}"
      )
      logger.info(f"Seed:          {config.seed}")
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

    module_path, class_name = algo_cfg.target.rsplit(":", 1)
    module = importlib.import_module(module_path)
    algo_class = getattr(module, class_name)
    if not issubclass(algo_class, RmaPPO):
      logger.error(
        f"rma-train requires an RmaPPO algorithm, got {algo_class.__name__}."
      )
      sys.exit(1)

    ckpt_dir = run_dir / "checkpoints" if run_dir is not None else None

    def _make_algo(
      cfg: ManagerBasedRlEnvCfg, env: ManagerBasedRlEnv | None = None
    ) -> RmaPPO:
      """Create an algo for one phase, reusing ``env`` when provided.

      Reusing the env across phases avoids a full MuJoCo-Warp recompile, which
      is the dominant cost of a phase transition."""
      phase_env = (
        env if env is not None else cfg.class_type(cfg=cfg, device=str(device))
      )
      _barrier()
      algo: RmaPPO = algo_class(
        config=algo_cfg,
        env=phase_env,
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
      )
      if is_main_process and ckpt_dir is not None and config.logger.save_interval > 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        algo.configure_checkpointing(ckpt_dir, config.logger.save_interval)
      return algo

    def _save_phase(algo: RmaPPO, label: str) -> Path | None:
      if ckpt_dir is None:
        return None
      ckpt_dir.mkdir(parents=True, exist_ok=True)
      path = ckpt_dir / f"{label}_final.pt"
      algo.save(path, global_step=algo.global_step)
      logger.success(f"Saved {label} checkpoint: {path}")
      return path

    def _handle_interrupt(algo: RmaPPO, label: str) -> None:
      if is_main_process and ckpt_dir is not None:
        path = ckpt_dir / f"{label}_interrupted.pt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        try:
          algo.save(path, global_step=algo.global_step)
          logger.success(f"Saved interrupted checkpoint: {path}")
        except Exception as exc:
          logger.error(f"Failed to save checkpoint: {exc}")

    phase1_ckpt: Path | None = None
    phase2_ckpt: Path | None = None
    phase1_end_step: int = 0

    # Env carried over from Phase 2 to Phase 3 (identical config) to skip a rebuild.
    def _reclaim_memory(skip_gc: bool = False) -> None:
      """Reclaim freed GPU memory after the caller has dropped its algo ref.

      The caller must ``del`` its own ``algo`` binding first (a helper can't free
      it). Run before building the next phase so the old phase's allocations —
      most importantly the RMA rollout buffer with depth-frame storage — don't
      sit alongside the new env's allocations and cause fragmentation/OOM."""
      if config.use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
      if not skip_gc:
        gc.collect()

    reused_env: ManagerBasedRlEnv | None = None

    if is_main_process:
      start_live_display()

    # ------------------------------------------------------------------
    # Phase 1 — PPO with privileged encoder
    # ------------------------------------------------------------------
    if config.start_phase <= 1:
      logger.info("--- Phase 1: PPO with privileged encoder ---")
      algo = _make_algo(env_cfg)
      if config.checkpoint is not None:
        checkpoint_path = Path(config.checkpoint)
        if not checkpoint_path.exists():
          logger.error(f"Checkpoint not found: {checkpoint_path}")
          sys.exit(1)
        algo.load(checkpoint_path)
        algo.global_step = 0
      _barrier()
      try:
        algo._ppo_loop(title="RMA Phase 1", total_steps=phase1_steps)
      except KeyboardInterrupt:
        _handle_interrupt(algo, "phase1")
        raise
      phase1_end_step = algo.global_step
      phase1_ckpt = _save_phase(algo, "phase1")
      algo.env.close()
      del algo
      _reclaim_memory()

    # ------------------------------------------------------------------
    # Phase 2 — adaptation encoder regression
    # ------------------------------------------------------------------
    if config.start_phase <= 2:
      logger.info("--- Phase 2: adaptation encoder regression ---")
      _barrier()
      p2_source = Path(config.checkpoint) if config.start_phase == 2 else phase1_ckpt
      if p2_source is None or not p2_source.exists():
        logger.error(
          "Phase 2 requires a Phase 1 checkpoint (--checkpoint or from Phase 1)."
        )
        sys.exit(1)
      algo = _make_algo(phase2_env_cfg)
      algo.load(p2_source)
      if phase1_end_step == 0:
        phase1_end_step = algo.global_step
      algo.build_adaptation_optimizer(lr=config.phase2_lr)
      try:
        algo._train_phase2(
          loss_threshold=config.phase2_loss_threshold,
          total_steps=config.phase2_steps,
        )
      except KeyboardInterrupt:
        _handle_interrupt(algo, "phase2")
        raise
      algo._metadata["phase1_end_step"] = phase1_end_step
      phase2_ckpt = _save_phase(algo, "phase2")
      # Phase 3 uses the same env config: keep the env, free everything else.
      reused_env = algo.env
      del algo
      _reclaim_memory(skip_gc=True)

    # ------------------------------------------------------------------
    # Phase 3 — policy fine-tuning with frozen adaptation encoder
    # ------------------------------------------------------------------
    logger.info("--- Phase 3: policy fine-tuning with frozen encoders ---")
    _barrier()
    p3_source = Path(config.checkpoint) if config.start_phase == 3 else phase2_ckpt
    if p3_source is None or not p3_source.exists():
      logger.error(
        "Phase 3 requires a Phase 2 checkpoint (--checkpoint or from Phase 2)."
      )
      sys.exit(1)
    algo = _make_algo(phase2_env_cfg, env=reused_env)
    loaded = algo.load(p3_source)
    if phase1_end_step == 0:
      phase1_end_step = loaded.get("metadata", {}).get("phase1_end_step", 0)
    algo.global_step = phase1_end_step
    if phase1_end_step > 0:
      algo._restore_env_step_counter()
    algo.build_phase3_optimizer()
    try:
      algo._ppo_loop(title="RMA Phase 3", total_steps=config.phase3_steps)
    except KeyboardInterrupt:
      _handle_interrupt(algo, "phase3")
      raise
    _save_phase(algo, "phase3")
    algo.env.close()

    if is_main_process:
      stop_live_display()
      teardown_wandb()
    logger.success("RMA pipeline complete!")

  finally:
    _teardown_distributed()


if __name__ == "__main__":
  main()
