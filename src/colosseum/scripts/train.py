#!/usr/bin/env python3
"""Custom PPO training script for Colosseum.

Uses the PPO implementation from colosseum.algorithm.ppo (iros26-derived).
This is independent of mjlab's RSL-RL runner.

Usage:
    pixi run -e train train
    pixi run -e train train task:t1-velocity logger:wandb
    pixi run -e train train task:t1-velocity --task.env.scene.num-envs 2048
    pixi run -e train train task:t1-velocity --task.algo-cfg.learning-steps 10000000
    pixi run -e train train --seed 0 --checkpoint ./logs/run/checkpoints/latest.pt
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import tyro
import wandb
import torch
import torch.distributed as dist
from loguru import logger
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.utils.torch import configure_torch_backends

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


def _make_env(env_cfg: ManagerBasedRlEnvCfg, device: str) -> ManagerBasedRlEnv:
    return make_env(env_cfg, device)


def _init_distributed(use_cuda: bool) -> tuple[bool, int, int, int]:
    """Initialize torch.distributed from torchrun environment variables.

    Returns:
        (is_distributed, world_size, rank, local_rank)
    """
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
        # Keep MuJoCo EGL device aligned with CUDA rank.
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)

    if not dist.is_initialized():
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    return True, world_size, rank, local_rank


def _teardown_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    """Main training entry point."""
    config = tyro.cli(
        TrainConfig,
        config=(tyro.conf.CascadeSubcommandArgs,),
    )

    if config.cuda < 0:
        raise ValueError(f"--cuda must be >= 0, got {config.cuda}")

    # In single-process mode, hard-pin visibility to the requested physical GPU.
    # This avoids auxiliary CUDA contexts on other devices (e.g. cuda:0) from
    # third-party libraries while still letting users choose --cuda N.
    pre_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if config.use_cuda and pre_world_size <= 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda)

    is_distributed = False
    world_size = 1
    rank = 0
    local_rank = 0

    try:
        is_distributed, world_size, rank, local_rank = _init_distributed(config.use_cuda)
        is_main_process = rank == 0

        algo_cfg = config.task.algo_cfg
        assert algo_cfg is not None, (
            f"Task '{config.task.name}' has no algo_cfg. "
            "Implement the algo_cfg property in the task's __init__.py."
        )

        env_cfg = config.task.train_env_cfg

        run_name = generate_run_name(
            task_name=config.task.name,
            algo_name=algo_cfg.name,
            seed=config.seed,
        )

        if config.logger.group is None:
            from dataclasses import replace
            config = replace(config, logger=replace(config.logger, group=config.task.name))

        try:
            wandb_run, run_dir = setup_wandb(
                config.logger,
                config.logger.log_dir,
                run_name,
                is_main_process=is_main_process,
            )

            if run_dir is None:
                run_dir = Path(config.logger.log_dir) / run_name

            if is_main_process:
                run_dir.mkdir(parents=True, exist_ok=True)
            else:
                # Use rank-local directory to avoid multi-process file collisions.
                run_dir = run_dir / f"rank_{rank}"
                run_dir.mkdir(parents=True, exist_ok=True)

            setup_loguru(
                run_dir,
                is_main_process=is_main_process,
                console_level=config.logger.console_level,
            )

            logger.info("=" * 80)
            logger.info(f"Task: {config.task.name}")
            logger.info(f"Algorithm: {algo_cfg.name}")
            logger.info(f"Seed: {config.seed}")
            logger.info(f"Num envs: {env_cfg.scene.num_envs}")
            logger.info(f"Distributed: {is_distributed} (world_size={world_size}, rank={rank}, local_rank={local_rank})")
            logger.info(f"Run directory: {run_dir}")
            logger.info("=" * 80)

            if is_main_process:
                save_experiment_config(config, run_dir, wandb_run)

            def log_fn(metrics: dict[str, float], step: int) -> None:
                if is_main_process and wandb_run is not None:
                    wandb.log(metrics, step=step)

        except Exception as e:
            print(f"Failed to set up logging: {e}")
            sys.exit(1)

        # Per-rank seed provides diverse rollout data across GPUs.
        rank_seed = config.seed + rank
        set_seed(rank_seed)
        configure_torch_backends()

        if is_distributed and is_main_process:
            logger.warning(
                "Ignoring --cuda in distributed mode; torchrun LOCAL_RANK determines device."
            )

        # Non-distributed mode has a single visible GPU after pinning above,
        # so always use logical cuda:0.
        device_id = local_rank if is_distributed else 0
        if config.use_cuda and not is_distributed:
            # Keep MuJoCo EGL device aligned with selected CUDA device.
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_id)

        device = get_device(cuda=config.use_cuda, device_id=device_id)
        logger.info(f"Using device: {device}")

        assert env_cfg is not None, "Training env config must be provided in the task config."
        env = _make_env(env_cfg=env_cfg, device=str(device))

        import importlib
        module_path, class_name = algo_cfg.target.rsplit(":", 1)
        module = importlib.import_module(module_path)
        algo_class = getattr(module, class_name)

        algo = algo_class(
            config=algo_cfg,
            env=env,
            device=device,
            log_fn=log_fn,
            log_interval=config.logger.log_interval,
        )

        algo.attach_metadata(
            experiment_config=config.to_serializable_dict(),
            wandb_run_id=wandb_run.id if wandb_run is not None else None,
            timestamp=datetime.now().isoformat(),
            seed=rank_seed,
            base_seed=config.seed,
            device=str(device),
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
        )

        if is_main_process and run_dir is not None and config.logger.save_interval > 0:
            ckpt_dir = run_dir / "checkpoints"
            algo.configure_checkpointing(ckpt_dir, config.logger.save_interval)

        if config.checkpoint is not None:
            checkpoint_path = Path(config.checkpoint)
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint file does not exist: {checkpoint_path}")
                sys.exit(1)
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            state = algo.load(checkpoint_path)
            logger.info(f"Resumed from step {state.get('global_step', 0)}")

        if is_distributed and dist.is_initialized():
            dist.barrier()

        logger.info("Starting training...")

        try:
            algo.train()
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user (Ctrl+C)")
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
        logger.success("Training complete!")
    finally:
        _teardown_distributed()


if __name__ == "__main__":
    main()
