#!/usr/bin/env python3
"""Play script - evaluate a trained policy in simulation.

This is for simulation validation ONLY. For deploying to real robot, use deploy.py.

Usage:
    pixi run play task:t1-velocity-rough --checkpoint ./logs/run/checkpoints/latest.pt
    pixi run play task:t1-velocity-rough --run-name t1-vel_ppo_20260506_145046
    pixi run play task:t1-velocity-rough --agent zero
    pixi run play task:t1-velocity-rough --agent random
    pixi run play task:t1-velocity-rough --agent onnx --onnx v1
    pixi run play task:t1-velocity-rough --agent onnx --onnx default
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

# Import tasks to populate registry
import colosseum.tasks  # noqa: F401
from colosseum.config.types.experiment import BaseExperimentConfig
from colosseum.utils.checkpoint import resolve_checkpoint
from colosseum.utils.model_registry import ModelRegistry
from colosseum.utils.torch import get_device, set_seed


@dataclass(frozen=True)
class PlayConfig(BaseExperimentConfig):
  """Play configuration."""

  agent: Literal["trained", "zero", "random", "onnx"] = "trained"
  num_envs: int = 1
  viewer: Literal["native", "viser", "auto"] = "auto"
  video: bool = False
  video_length: int = 500
  video_height: int = 720
  video_width: int = 1280
  video_path: str | None = None
  """Output video path. Defaults to videos/<task>-<algo>.mp4."""
  run_name: str | None = None
  """Run name to load checkpoint from (looks in ./logs/<run_name>/checkpoints/)."""
  onnx: str | None = None
  """ONNX model name, path, or 'default' (looks up in models/registry.yaml)."""
  debug_vel: bool = False
  """Print component-wise velocity tracking errors every debug_vel_every steps."""
  debug_vel_every: int = 20
  """How often (in steps) to print velocity debug info when --debug-vel is set."""
  seed: int = 42
  """Random seed for reproducibility of obstacle placement and other RNG."""


def _resolve_onnx(config: PlayConfig) -> Path | None:
  """Resolve ONNX model path from --onnx arg.

  Priority:
    1. Direct filesystem path
    2. 'default' → registry default for task
    3. Name ('v1', 'stable') → registry lookup
    4. Default path: models/<task>/<name>.onnx
  """
  onnx = config.onnx
  if onnx is None:
    return None

  # 1. Direct path
  p = Path(onnx)
  if p.exists():
    return p.resolve()

  # 2/3. Registry lookup
  if onnx == "default":
    entry = ModelRegistry.get_default(config.task.name)
    if entry:
      task_dir = Path("models") / config.task.name
      return (task_dir / entry["file"]).resolve()

  entry = ModelRegistry.get(config.task.name, onnx)
  if entry:
    task_dir = Path("models") / config.task.name
    return (task_dir / entry["file"]).resolve()

  # 4. Default path: models/<task>/<name>.onnx
  default_p = Path("models") / config.task.name / f"{onnx}.onnx"
  if default_p.exists():
    return default_p.resolve()

  return None


def _make_env(
  env_cfg: ManagerBasedRlEnvCfg, device: str, render_mode: str | None
) -> ManagerBasedRlEnv:
  return env_cfg.class_type(cfg=env_cfg, device=device, render_mode=render_mode)


def _parse_single_cuda_device(cuda_arg: str) -> int:
  parts = [p.strip() for p in str(cuda_arg).split(",") if p.strip()]
  if len(parts) != 1:
    raise ValueError(
      f"play expects a single GPU id for --cuda, got '{cuda_arg}'. "
      "Use one value, e.g. --cuda 0"
    )
  try:
    device_id = int(parts[0])
  except ValueError as exc:
    raise ValueError(
      f"Invalid --cuda value '{cuda_arg}'. Use an integer like 0"
    ) from exc
  if device_id < 0:
    raise ValueError(f"--cuda must be >= 0, got {device_id}")
  return device_id


def create_agent(config: PlayConfig, env: ManagerBasedRlEnv, device: torch.device):
  # --onnx implies agent=onnx
  if config.onnx and config.agent == "trained":
    config = replace(config, agent="onnx")

  if config.agent == "zero":
    logger.info("Using zero-action agent")

    def zero_agent(obs_dict):
      if isinstance(obs_dict, dict):
        obs = next(iter(obs_dict.values()))
      else:
        obs = obs_dict
      return torch.zeros(
        (obs.shape[0], env.action_manager.total_action_dim), device=device
      )

    return zero_agent

  elif config.agent == "random":
    logger.info("Using random-action agent")

    def random_agent(obs_dict):
      if isinstance(obs_dict, dict):
        obs = next(iter(obs_dict.values()))
      else:
        obs = obs_dict
      return torch.randn(
        (obs.shape[0], env.action_manager.total_action_dim), device=device
      )

    return random_agent

  elif config.agent == "trained":
    checkpoint_path = resolve_checkpoint(config.checkpoint, config.run_name)
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

    algo = algo_class(
      config=algo_cfg,
      env=env,
      device=device,
      log_fn=lambda _m, _s: None,
      log_interval=-1,
    )
    state = algo.load(checkpoint_path)
    logger.info(f"Loaded from step {state.get('global_step', 0)}")

    # Optional RMA latent-path override (diagnostics): force oracle vs adaptation.
    phase_override = getattr(config, "rma_phase", "auto")
    if phase_override != "auto" and hasattr(algo, "phase"):
      from colosseum.algorithm.rma_ppo import RmaPhase

      algo.phase = RmaPhase(phase_override)
      logger.info(f"RMA phase forced -> {algo.phase.value}")

    algo.actor.eval()
    algo.actor_obs_normalizer.eval()
    if hasattr(algo, "rma_manager"):
      algo.rma_manager.eval()

    def trained_agent(obs_dict):
      with torch.no_grad():
        actor_obs = algo.get_actor_obs(obs_dict)
        normalized_obs = algo.actor_obs_normalizer(actor_obs)
        privileged_obs = (
          algo.get_privileged_obs(obs_dict)
          if hasattr(algo, "get_privileged_obs")
          else {}
        )
        composed_obs = algo._compose_actor_input(normalized_obs, privileged_obs)
        return algo._eval_get_action(composed_obs)

    return trained_agent

  elif config.agent == "onnx":
    onnx_path = _resolve_onnx(config)
    if onnx_path is None or not onnx_path.exists():
      logger.error(f"ONNX model not found: {config.onnx}")
      sys.exit(1)
    logger.info(f"Loading ONNX: {onnx_path}")

    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))

    algo_cfg = config.task.algo_cfg
    assert algo_cfg is not None, (
      f"Task '{config.task.name}' has no algo_cfg. "
      "Implement the algo_cfg property in the task's __init__.py."
    )
    import importlib

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

    # Generic stateful pattern: extra inputs (beyond "obs") are recurrent
    # state; the model outputs updated state after "actions", positionally paired.
    state_input_names = [inp.name for inp in session.get_inputs() if inp.name != "obs"]
    state_bufs: list[np.ndarray] = []
    for inp in session.get_inputs():
      if inp.name != "obs":
        shape = tuple(d if isinstance(d, int) else 1 for d in inp.shape)
        state_bufs.append(np.zeros(shape, dtype=np.float32))

    def onnx_agent(obs_dict):
      with torch.no_grad():
        actor_obs_np = algo.get_actor_obs(obs_dict).cpu().numpy()
        feed: dict[str, np.ndarray] = {"obs": actor_obs_np}
        for name, buf in zip(state_input_names, state_bufs):
          feed[name] = buf
        outputs = session.run(None, feed)
        for i in range(len(state_bufs)):
          state_bufs[i] = outputs[i + 1]
        return torch.from_numpy(outputs[0]).to(device)

    return onnx_agent

  raise ValueError(f"Unknown agent: {config.agent}")


def _wrap_agent_with_vel_debug(
  agent,
  env: ManagerBasedRlEnv,
  every: int = 20,
):
  """Wrap an agent to periodically print the same debug blocks as the arena.

  Output format mirrors arena/src/tasks/T1Velocity.cpp exactly ([vel] line +
  [OBS] block with leg pos_rel / vel / last_action in LHP..RAR order) so the
  two can be diffed directly. Reports for env 0 only.
  """
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  # Leg joints in the arena's hardware/sim order (LHP..RAR).
  leg_joint_names = (
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
  )
  leg_labels = ("LHP", "LHR", "LHY", "LKN", "LAP", "LAR",
                "RHP", "RHR", "RHY", "RKN", "RAP", "RAR")
  leg_cfg = SceneEntityCfg("robot", joint_names=leg_joint_names, preserve_order=True)
  leg_cfg.resolve(env.scene)
  leg_ids = leg_cfg.joint_ids
  # Leg slice of the 21-dim raw action (arms 8 + waist 1 precede legs); matches
  # the arena's observation[52 + 9 + k].
  leg_action_idx = list(range(9, 21))

  step_counter = [0]

  def _fmt(labels, values):
    return " ".join(f"{lbl}={v:g}" for lbl, v in zip(labels, values))

  def wrapped(obs_dict):
    actions = agent(obs_dict)
    step_counter[0] += 1
    if step_counter[0] % every != 0:
      return actions

    robot = env.scene["robot"]
    cmd = env.command_manager.get_command("twist")[0]  # (3,)
    act_lin = robot.data.root_link_lin_vel_b[0]  # (3,)
    act_ang = robot.data.root_link_ang_vel_b[0]  # (3,) ~ gyro (imu_ang_vel)
    pg = robot.data.projected_gravity_b[0]  # (3,)

    cmd_vx, cmd_vy, cmd_wz = cmd[0].item(), cmd[1].item(), cmd[2].item()
    act_vx, act_vy, act_wz = act_lin[0].item(), act_lin[1].item(), act_ang[2].item()

    pos_rel = (robot.data.joint_pos[0, leg_ids]
               - robot.data.default_joint_pos[0, leg_ids]).tolist()
    vel = robot.data.joint_vel[0, leg_ids].tolist()
    raw = actions[0]
    act_leg = (raw[leg_action_idx].tolist()
               if raw.numel() >= 21 else [float("nan")] * 12)

    s = step_counter[0]
    print(
      f"\n[vel] step={s:>5d} | "
      f"cmd: vx={cmd_vx:+g} vy={cmd_vy:+g} wz={cmd_wz:+g} | "
      f"act: vx={act_vx:+g} vy={act_vy:+g} wz={act_wz:+g} | "
      f"err: vx={cmd_vx - act_vx:+g} vy={cmd_vy - act_vy:+g} wz={cmd_wz - act_wz:+g}"
    )
    print(
      f"[OBS] step={s} cmd=[{cmd_vx:g},{cmd_vy:g},{cmd_wz:g}] "
      f"gyro=[{act_ang[0]:g},{act_ang[1]:g},{act_ang[2]:g}] "
      f"pg=[{pg[0]:g},{pg[1]:g},{pg[2]:g}]"
    )
    print("  leg pos_rel: " + _fmt(leg_labels, pos_rel))
    print("  leg vel:     " + _fmt(leg_labels, vel))
    print("  last_action: " + _fmt(leg_labels, act_leg), flush=True)
    return actions

  return wrapped


def main() -> None:
  config = tyro.cli(PlayConfig, config=(tyro.conf.CascadeSubcommandArgs,))
  device_id = _parse_single_cuda_device(config.cuda)

  logger.remove()
  logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG",
  )

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
    env_cfg = replace(
      env_cfg,
      viewer=replace(
        env_cfg.viewer, height=config.video_height, width=config.video_width
      ),
    )

  render_mode = "rgb_array" if config.video else None
  env = _make_env(env_cfg=env_cfg, device=str(device), render_mode=render_mode)
  env.reset()

  agent = create_agent(config, env, device)
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


def _record_video(config: PlayConfig, env, agent) -> None:
  import imageio
  import numpy as np

  video_dir = Path("./videos")
  video_dir.mkdir(parents=True, exist_ok=True)
  if config.video_path:
    video_path = Path(config.video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
  else:
    algo_cfg = config.task.algo_cfg
    algo_name = algo_cfg.name if algo_cfg is not None else "unknown"
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
