#!/usr/bin/env python3
"""Play script for RSL-RL trained policies.

Evaluates checkpoints saved by train_rsl_rl.py (RSL-RL OnPolicyRunner format).

Usage:
    pixi run play-rsl-rl task:t1-velocity --checkpoint logs/rsl_rl/t1_velocity/.../model_150.pt
    pixi run play-rsl-rl task:t1-velocity --checkpoint logs/rsl_rl/t1_velocity/.../model_150.pt --viewer viser --debug-vel
    pixi run play-rsl-rl task:t1-velocity --agent zero --viewer native
    pixi run play-rsl-rl task:t1-velocity --agent random --viewer native
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch
import tyro
from loguru import logger
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from tensordict import TensorDict

# Import tasks to populate registry
import colosseum.tasks  # noqa: F401

from colosseum.config.types.experiment import BaseExperimentConfig
from colosseum.utils.checkpoint import resolve_checkpoint
from colosseum.utils.torch import get_device, set_seed


@dataclass(frozen=True)
class PlayRslRlConfig(BaseExperimentConfig):
    agent: Literal["trained", "zero", "random"] = "trained"
    num_envs: int = 1
    viewer: Literal["native", "viser", "auto"] = "auto"
    video: bool = False
    video_length: int = 500
    video_height: int = 720
    video_width: int = 1280
    video_path: str | None = None
    run_name: str | None = None
    debug_vel: bool = False
    debug_vel_every: int = 20
    seed: int = 42


def _make_env(env_cfg: ManagerBasedRlEnvCfg, device: str, render_mode: str | None) -> ManagerBasedRlEnv:
    return env_cfg.class_type(cfg=env_cfg, device=device, render_mode=render_mode)


def _parse_single_cuda_device(cuda_arg: str) -> int:
    parts = [p.strip() for p in str(cuda_arg).split(",") if p.strip()]
    if len(parts) != 1:
        raise ValueError(f"play expects a single GPU id for --cuda, got '{cuda_arg}'.")
    device_id = int(parts[0])
    if device_id < 0:
        raise ValueError(f"--cuda must be >= 0, got {device_id}")
    return device_id


def _migrate_legacy_checkpoint(loaded_dict: dict) -> dict:
    """Migrate legacy checkpoint format (model_state_dict) to current format."""
    if "actor_state_dict" in loaded_dict:
        return loaded_dict

    if "model_state_dict" not in loaded_dict:
        return loaded_dict

    logger.info("Detected legacy checkpoint. Migrating to new format...")
    model_state_dict = loaded_dict.pop("model_state_dict")
    actor_state_dict = {}
    critic_state_dict = {}

    for key, value in model_state_dict.items():
        if key.startswith("actor."):
            actor_state_dict[key.replace("actor.", "mlp.")] = value
        elif key.startswith("actor_obs_normalizer."):
            actor_state_dict[key.replace("actor_obs_normalizer.", "obs_normalizer.")] = value
        elif key in ("std", "log_std"):
            actor_state_dict[key] = value

        if key.startswith("critic."):
            critic_state_dict[key.replace("critic.", "mlp.")] = value
        elif key.startswith("critic_obs_normalizer."):
            critic_state_dict[key.replace("critic_obs_normalizer.", "obs_normalizer.")] = value

    loaded_dict["actor_state_dict"] = actor_state_dict
    loaded_dict["critic_state_dict"] = critic_state_dict
    return loaded_dict


def _migrate_distribution_keys(actor_state_dict: dict) -> dict:
    """Migrate rsl-rl 4.x actor keys (std/log_std) to 5.x (distribution.std_param/log_std_param)."""
    if "std" in actor_state_dict:
        actor_state_dict["distribution.std_param"] = actor_state_dict.pop("std")
    if "log_std" in actor_state_dict:
        actor_state_dict["distribution.log_std_param"] = actor_state_dict.pop("log_std")
    return actor_state_dict


def create_rsl_rl_agent(config: PlayRslRlConfig, env: ManagerBasedRlEnv, device: torch.device):
    if config.agent == "zero":
        logger.info("Using zero-action agent")
        def zero_agent(obs_dict):
            obs = obs_dict.get("actor", next(iter(obs_dict.values())))
            return torch.zeros((obs.shape[0], env.action_manager.total_action_dim), device=device)
        return zero_agent

    elif config.agent == "random":
        logger.info("Using random-action agent")
        def random_agent(obs_dict):
            obs = obs_dict.get("actor", next(iter(obs_dict.values())))
            return torch.randn((obs.shape[0], env.action_manager.total_action_dim), device=device)
        return random_agent

    elif config.agent == "trained":
        checkpoint_path = resolve_checkpoint(config.checkpoint, config.run_name)
        if checkpoint_path is None or not checkpoint_path.exists():
            logger.error("No checkpoint found. Provide --checkpoint <path>.")
            sys.exit(1)
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        rl_cfg = config.task.rl_cfg
        actor_cfg = rl_cfg.actor

        # Get observation structure from the environment
        raw_obs = env.observation_manager.compute()
        obs_td = TensorDict(raw_obs, batch_size=[env.num_envs])

        # Build distribution config (copy to avoid mutation)
        dist_cfg = None
        if actor_cfg.distribution_cfg:
            dist_cfg = dict(actor_cfg.distribution_cfg)

        # Create actor model
        from rsl_rl.models import MLPModel

        actor = MLPModel(
            obs=obs_td,
            obs_groups=rl_cfg.obs_groups,
            obs_set="actor",
            output_dim=env.action_manager.total_action_dim,
            hidden_dims=actor_cfg.hidden_dims,
            activation=actor_cfg.activation,
            obs_normalization=actor_cfg.obs_normalization,
            distribution_cfg=dist_cfg,
        ).to(device)

        # Load checkpoint
        loaded_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        loaded_dict = _migrate_legacy_checkpoint(loaded_dict)
        actor_sd = loaded_dict["actor_state_dict"]
        actor_sd = _migrate_distribution_keys(actor_sd)
        actor.load_state_dict(actor_sd)
        logger.success(f"Checkpoint loaded: {checkpoint_path}")

        actor.eval()

        def rsl_rl_agent(obs_dict):
            obs_td = TensorDict(obs_dict, batch_size=[env.num_envs])
            with torch.no_grad():
                return actor(obs_td)

        return rsl_rl_agent

    raise ValueError(f"Unknown agent: {config.agent}")


def _wrap_agent_with_vel_debug(agent, env: ManagerBasedRlEnv, every: int = 20):
    step_counter = [0]

    def wrapped(obs_dict):
        actions = agent(obs_dict)

        step_counter[0] += 1
        if step_counter[0] % every != 0:
            return actions

        robot = env.scene["robot"]
        cmd = env.command_manager.get_command("twist")[0]
        act_lin = robot.data.root_link_lin_vel_b[0]
        act_ang = robot.data.root_link_ang_vel_b[0]
        heading = robot.data.heading_w[0].item()

        cmd_vx, cmd_vy, cmd_wz = cmd[0].item(), cmd[1].item(), cmd[2].item()
        act_vx, act_vy = act_lin[0].item(), act_lin[1].item()
        act_wz = act_ang[2].item()

        import math
        print(
            f"[vel] step={step_counter[0]:>5d} | "
            f"cmd: vx={cmd_vx:+.2f}  vy={cmd_vy:+.2f}  wz={cmd_wz:+.2f} | "
            f"act: vx={act_vx:+.2f}  vy={act_vy:+.2f}  wz={act_wz:+.2f} | "
            f"err: vx={cmd_vx-act_vx:+.2f}  vy={cmd_vy-act_vy:+.2f}  wz={cmd_wz-act_wz:+.2f} | "
            f"heading={math.degrees(heading):+.1f}°"
        )
        return actions

    return wrapped


def main() -> None:
    config = tyro.cli(PlayRslRlConfig, config=(tyro.conf.CascadeSubcommandArgs,))
    device_id = _parse_single_cuda_device(config.cuda)

    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="DEBUG")

    configure_torch_backends()
    if config.use_cuda:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_id)
    device = get_device(cuda=config.use_cuda, device_id=device_id)
    logger.info(f"Device: {device}")

    set_seed(config.seed)
    logger.info(f"Random seed: {config.seed}")

    env_cfg = config.task.play_env_cfg or config.task.train_env_cfg
    env_cfg = replace(env_cfg, scene=replace(env_cfg.scene, num_envs=config.num_envs))
    if config.video:
        env_cfg = replace(env_cfg, viewer=replace(env_cfg.viewer, height=config.video_height, width=config.video_width))

    render_mode = "rgb_array" if config.video else None
    env = _make_env(env_cfg=env_cfg, device=str(device), render_mode=render_mode)
    env.reset()

    agent = create_rsl_rl_agent(config, env, device)
    if config.debug_vel:
        agent = _wrap_agent_with_vel_debug(agent, env, every=config.debug_vel_every)

    if config.video:
        _record_video(config, env, agent)
    elif config.viewer == "viser":
        viewer = ViserPlayViewer(env, agent)
        viewer.run()
        env.close()
    else:
        viewer = NativeMujocoViewer(env, agent)
        viewer.run()
        env.close()


def _record_video(config: PlayRslRlConfig, env, agent) -> None:
    import imageio
    import numpy as np

    video_dir = Path("./videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    if config.video_path:
        video_path = Path(config.video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        algo_name = config.task.rl_cfg.algorithm.class_name if config.task.rl_cfg else "unknown"
        video_path = video_dir / f"{config.task.name}-{algo_name}.mp4"
    logger.info(f"Recording episode to {video_path}")

    obs, _ = env.reset()
    frames = []
    for _ in range(config.video_length):
        actions = agent(obs)
        obs, _, terminated, _, _ = env.step(actions)
        frame = env.render()
        if frame is not None:
            if isinstance(frame, np.ndarray) and frame.ndim == 4:
                frame = frame[0]
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            frames.append(frame)
        if terminated[0]:
            break

    if frames:
        fps = env.metadata.get("render_fps", 30)
        imageio.mimwrite(str(video_path), frames, fps=fps)
        logger.success(f"Video saved: {video_path}")
    env.close()


if __name__ == "__main__":
    main()
