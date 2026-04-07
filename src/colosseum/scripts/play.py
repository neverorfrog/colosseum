#!/usr/bin/env python3
"""Play script - evaluate a trained checkpoint in simulation.

This is for simulation validation ONLY. For deploying to real robot, use deploy.py.

Usage:
    pixi run -e train python -m colosseum.scripts.play task:t1-velocity-rough --checkpoint ./logs/run/checkpoints/latest.pt
    pixi run -e train python -m colosseum.scripts.play task:t1-velocity-rough --agent zero
    pixi run -e train python -m colosseum.scripts.play task:t1-velocity-rough --agent random
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch
import tyro
from loguru import logger
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer

# Import tasks to populate registry
import colosseum.tasks  # noqa: F401

from colosseum.algorithm.base_algorithm import get_latest_checkpoint
from colosseum.config.types.experiment import BaseExperimentConfig
from colosseum.utils.torch import get_device
from colosseum.utils.train.env import make_env


@dataclass(frozen=True)
class PlayConfig(BaseExperimentConfig):
    """Play configuration."""
    agent: Literal["trained", "zero", "random"] = "trained"
    num_envs: int = 1
    viewer: Literal["native", "auto"] = "auto"
    video: bool = False
    video_length: int = 500


def _resolve_checkpoint(checkpoint: str | None) -> Path | None:
    if checkpoint and str(checkpoint).lower() != "latest":
        p = Path(checkpoint)
        if p.is_dir():
            return get_latest_checkpoint(p)
        return p if p.exists() else None

    latest_run = Path("./logs") / "wandb" / "latest-run"
    ckpt_dir = latest_run / "checkpoints"
    if ckpt_dir.exists():
        latest_link = ckpt_dir / "latest.pt"
        if latest_link.exists():
            return latest_link
        return get_latest_checkpoint(ckpt_dir)

    return None


def _make_env(env_cfg: ManagerBasedRlEnvCfg, device: str, render_mode: str | None) -> ManagerBasedRlEnv:
    return make_env(env_cfg, device, render_mode)


def create_agent(config: PlayConfig, env: ManagerBasedRlEnv, device: torch.device):
    if config.agent == "zero":
        logger.info("Using zero-action agent")
        def zero_agent(obs_dict):
            obs = obs_dict.get("policy", obs_dict) if isinstance(obs_dict, dict) else obs_dict
            return torch.zeros((obs.shape[0], env.action_manager.total_action_dim), device=device)
        return zero_agent

    elif config.agent == "random":
        logger.info("Using random-action agent")
        def random_agent(obs_dict):
            obs = obs_dict.get("policy", obs_dict) if isinstance(obs_dict, dict) else obs_dict
            return torch.randn((obs.shape[0], env.action_manager.total_action_dim), device=device)
        return random_agent

    elif config.agent == "trained":
        checkpoint_path = _resolve_checkpoint(config.checkpoint)
        if checkpoint_path is None or not checkpoint_path.exists():
            logger.error("No checkpoint found. Provide --checkpoint <path>.")
            sys.exit(1)
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        algo_cfg = config.task.algo_cfg
        assert algo_cfg is not None, (
            f"Task '{config.task.name}' has no algo_cfg. "
            "Implement the algo_cfg property in the task's __init__.py."
        )

        import importlib
        module_path, class_name = algo_cfg.target.rsplit(":", 1)
        module = importlib.import_module(module_path)
        algo_class = getattr(module, class_name)

        # Create minimal env just to get observation/action dimensions
        env_cfg = config.task.env
        dim_env_cfg = replace(env_cfg, scene=replace(env_cfg.scene, num_envs=1))
        dim_env = _make_env(dim_env_cfg, str(device), render_mode=None)

        algo = algo_class(
            config=algo_cfg,
            env=dim_env,
            device=device,
            log_fn=lambda _m, _s: None,
            log_interval=-1,
        )
        state = algo.load(checkpoint_path)
        logger.info(f"Loaded from step {state.get('global_step', 0)}")
        dim_env.close()

        algo.actor.eval()
        algo.actor_obs_normalizer.eval()

        def trained_agent(obs_dict):
            with torch.no_grad():
                actor_obs = algo.get_actor_obs(obs_dict)
                normalized_obs = algo.actor_obs_normalizer(actor_obs)
                return algo._eval_get_action(normalized_obs)

        return trained_agent

    raise ValueError(f"Unknown agent: {config.agent}")


def main() -> None:
    config = tyro.cli(PlayConfig, config=(tyro.conf.CascadeSubcommandArgs,))

    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="DEBUG")

    configure_torch_backends()
    device = get_device(cuda=config.use_cuda, device_id=0)
    logger.info(f"Device: {device}")

    env_cfg = config.task.play_env_cfg or config.task.env
    if config.num_envs != 1:
        env_cfg = replace(env_cfg, scene=replace(env_cfg.scene, num_envs=config.num_envs))

    render_mode = "rgb_array" if config.video else None
    env = _make_env(env_cfg=env_cfg, device=str(device), render_mode=render_mode)
    env.reset()

    agent = create_agent(config, env, device)

    if config.video:
        _record_video(config, env, agent)
    else:
        viewer = NativeMujocoViewer(env, agent)
        viewer.run()
        env.close()


def _record_video(config: PlayConfig, env, agent) -> None:
    import imageio
    import numpy as np

    video_dir = Path("./videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    algo_cfg = config.task.algo_cfg
    algo_name = algo_cfg.name if algo_cfg is not None else "unknown"
    video_path = video_dir / f"{config.task.name}-{algo_name}.mp4"
    logger.info(f"Recording {config.video_length} steps to {video_path}")

    obs, _ = env.reset()
    frames = []
    for _ in range(config.video_length):
        actions = agent(obs)
        obs, _, _, _, _ = env.step(actions)
        frame = env.render()
        if frame is not None:
            if isinstance(frame, np.ndarray) and frame.ndim == 4:
                frame = frame[0]
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            frames.append(frame)

    if frames:
        fps = env.metadata.get("render_fps", 30)
        imageio.mimwrite(str(video_path), frames, fps=fps)
        logger.success(f"Video saved: {video_path}")
    env.close()


if __name__ == "__main__":
    main()
