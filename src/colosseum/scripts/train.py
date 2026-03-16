#!/usr/bin/env python3
"""Custom PPO training script for Colosseum.

Uses the PPO implementation from colosseum.algorithm.ppo (iros26-derived).
This is independent of mjlab's RSL-RL runner.

Usage:
    pixi run -e train train
    pixi run -e train train task:t1-velocity-flat algo:ppo logger:wandb
    pixi run -e train train task:t1-velocity-flat --task.env.scene.num-envs 2048
    pixi run -e train train --seed 0 --algo.learning-steps 10000000
    pixi run -e train train --checkpoint ./logs/run/checkpoints/latest.pt
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import tyro
import wandb
from loguru import logger
from mjlab.envs import ManagerBasedRlEnvCfg
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
from colosseum.utils.train.env import ViewerCompatibleEnv


def _make_env(env_cfg: ManagerBasedRlEnvCfg, device: str) -> ViewerCompatibleEnv:
    return ViewerCompatibleEnv(cfg=env_cfg, device=device, render_mode=None)


def main() -> None:
    """Main PPO training entry point."""
    config = tyro.cli(
        TrainConfig,
        config=(tyro.conf.CascadeSubcommandArgs,),
    )
    env_cfg = config.task.env

    run_name = generate_run_name(
        task_name=config.task.name,
        algo_name=config.algo.name,
        seed=config.seed,
    )

    if config.logger.group is None:
        from dataclasses import replace
        config = replace(config, logger=replace(config.logger, group=config.task.name))

    try:
        wandb_run, run_dir = setup_wandb(
            config.logger, config.logger.log_dir, run_name, is_main_process=True
        )

        if run_dir is None:
            run_dir = Path(config.logger.log_dir) / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

        setup_loguru(run_dir, is_main_process=True, console_level=config.logger.console_level)

        logger.info("=" * 80)
        logger.info(f"Task: {config.task.name}")
        logger.info(f"Algorithm: {config.algo.name}")
        logger.info(f"Seed: {config.seed}")
        logger.info(f"Num envs: {env_cfg.scene.num_envs}")
        logger.info(f"Run directory: {run_dir}")
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
    device = get_device(cuda=config.use_cuda, device_id=0)
    logger.info(f"Using device: {device}")

    assert env_cfg is not None, "Training env config must be provided in the task config."
    env = _make_env(env_cfg=env_cfg, device=str(device))

    import importlib
    module_path, class_name = config.algo.target.rsplit(":", 1)
    module = importlib.import_module(module_path)
    algo_class = getattr(module, class_name)

    algo = algo_class(
        config=config.algo,
        env=env,
        device=device,
        log_fn=log_fn,
        log_interval=config.logger.log_interval,
    )

    algo.attach_metadata(
        experiment_config=config.to_serializable_dict(),
        wandb_run_id=wandb_run.id if wandb_run is not None else None,
        timestamp=datetime.now().isoformat(),
        seed=config.seed,
        device=str(device),
    )

    if run_dir is not None and config.logger.save_interval > 0:
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

    logger.info("Starting training...")

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
    logger.success("Training complete!")


if __name__ == "__main__":
    main()
