#!/usr/bin/env python3
"""Run the dribbling evaluation protocol.

Example:
    pixi run -e train eval-dribbling \
        task:t1-dribbling \
        --checkpoint ./logs/run/checkpoints/latest.pt \
        --episodes-per-condition 1000 \
        --num-envs 128 \
        --seeds 0 1 2
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import tyro
from loguru import logger
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from tqdm.auto import tqdm

import colosseum.tasks  # noqa: F401
from colosseum.config.types.experiment import BaseExperimentConfig
from colosseum.tasks.dribbling.mdp.observations import (
  ball_position,
  ball_velocity_xy,
  obstacle_position_b,
  obstacle_velocity_b,
)
from colosseum.utils.torch import get_device, set_seed
from colosseum.utils.train.env import make_env

from colosseum.scripts.play import (  # reuse checkpoint/action path from play.py
  _parse_single_cuda_device,
  _resolve_checkpoint,
  create_agent,
)


@dataclass(frozen=True)
class EvalCondition:
  name: str
  label: str
  stage_index: int
  static_three: bool = False
  include_in_main: bool = True


MAIN_CONDITIONS = (
  EvalCondition("no_obstacles", "No obstacles", 0),
  EvalCondition("static_3", "3 static obstacles", 1, static_three=True),
  EvalCondition("moving_3", "3 moving obstacles", 4),
)

VELOCITY_CONDITIONS = (
  EvalCondition("velocity_no_obstacles", "No obstacles", 0, include_in_main=False),
  EvalCondition("velocity_single_obstacle", "Single obstacle", 3, include_in_main=False),
)

CONDITIONS_BY_NAME = {
  condition.name: condition for condition in (*MAIN_CONDITIONS, *VELOCITY_CONDITIONS)
}


@dataclass(frozen=True)
class DribblingEvalConfig(BaseExperimentConfig):
  """Configuration for the dribbling evaluation protocol."""

  agent: Literal["trained", "zero", "random"] = "trained"
  viewer: Literal["native", "viser"] = "native"
  viewer_condition: Literal[
    "none",
    "no_obstacles",
    "static_3",
    "moving_3",
    "velocity_no_obstacles",
    "velocity_single_obstacle",
  ] = "none"
  view_during_eval: bool = False
  """Open a realtime viewer while still running the full evaluation loop."""

  eval_viewer_frame_rate: float = 60.0
  """Render frame rate used by --view-during-eval."""

  eval_viewer_realtime: bool = True
  """Throttle single-env viewed evaluation to realtime."""

  num_envs: int = 64
  """Parallel simulator environments used for each condition."""

  episodes_per_condition: int = 300
  """Target-reaching trials collected for each main condition and seed."""

  velocity_episodes_per_condition: int = 64
  """Target-reaching trials collected for each velocity diagnostic condition."""

  seeds: tuple[int, ...] = (0,)
  """Evaluation seeds; results are aggregated across all listed seeds."""

  output_dir: str = "logs/dribbling_eval"
  """Parent directory for timestamped reports and plots."""

  report_name: str = "evaluation.md"
  """Markdown report filename written inside the timestamped output directory."""

  max_trial_s: float = 30.0
  """Maximum duration of one target-reaching trial before counting failure."""

  run_main: bool = True
  """Run the main no-obstacle, 3-static, and 3-moving task evaluation."""

  run_velocity_diagnostic: bool = True
  """Run the separate velocity tracking diagnostic conditions."""

  save_plots: bool = True
  """Save tracking-error and XY trajectory plots for diagnostic conditions."""

  progress_bar: bool = True
  """Show a progress bar for completed trials in each condition and seed."""

  plot_condition_env_index: int = 0
  """Environment index whose trajectory is recorded for diagnostic plots."""

  blocked_detection_range: float = 2.0
  """Max forward ball-to-obstacle distance for a timestep to count as blocked."""

  blocked_tube_radius: float = 0.75
  """Max lateral distance from the ball-target corridor to count as blocked."""

  robot_obstacle_collision_distance: float = 0.5
  """Robot-obstacle distance threshold used to count a collision/safety event."""

  ball_obstacle_collision_distance: float = 0.26
  """Ball-obstacle distance threshold used to count a collision/safety event."""

  fall_limit_deg: float = 70.0
  """Base tilt threshold, in degrees, used to count a fall event."""

  ball_lost_distance: float = 2.0
  """Robot-ball distance threshold used to end a trial as ball-lost."""

  perception_valid_only: bool = True
  """Compute depth-head perception errors only when the visual mask is valid."""

  randomize_target_and_obstacle: bool = False
  """Sample target and obstacle parameters from the eval_*_range fields."""

  eval_target_distance: float = 5.0
  """Directly generated target distance from the current ball position."""

  eval_target_distance_range: tuple[float, float] = (2.5, 8.0)
  """Uniform per-trial target-distance range used when randomization is enabled."""

  eval_target_heading_offset: float = 0.0
  """Target heading offset in radians relative to robot yaw."""

  eval_target_heading_offset_range: tuple[float, float] = (-0.45, 0.45)
  """Uniform per-trial heading-offset range used when randomization is enabled."""

  eval_obstacle_forward_fractions: tuple[float, float, float] = (0.3, 0.5, 0.7)
  """Obstacle locations as fractions along the ball-to-target segment."""

  eval_obstacle_forward_fraction_ranges: (
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
  ) = ((0.2, 0.5), (0.4, 0.7), (0.6, 0.9))
  """Per-obstacle forward-fraction ranges used when randomization is enabled."""

  eval_obstacle_lateral_offsets: tuple[float, float, float] = (-0.35, 0.0, 0.35)
  """Obstacle lateral offsets from the ball-to-target segment, in metres."""

  eval_obstacle_lateral_offset_ranges: (
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
  ) = ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))
  """Per-obstacle lateral-offset ranges used when randomization is enabled."""

  eval_obstacle_speed: float = 0.15
  """Scripted speed for moving evaluation obstacles, in m/s."""

  eval_obstacle_speed_range: tuple[float, float] = (0.1, 0.3)
  """Uniform per-env obstacle-speed range used when randomization is enabled."""

  eval_lateral_limit: float = 0.45
  """Lateral blocker travel limit before reversing direction, in metres."""

  eval_lateral_limit_range: tuple[float, float] = (0.3, 0.5)
  """Uniform per-env lateral-travel range used when randomization is enabled."""


class ScalarAccumulator:
  def __init__(self) -> None:
    self.values: list[float] = []

  def add_tensor(self, value: torch.Tensor, mask: torch.Tensor | None = None) -> None:
    if mask is not None:
      value = value[mask]
    if value.numel() == 0:
      return
    self.values.extend(value.detach().float().cpu().reshape(-1).tolist())

  def add(self, value: float) -> None:
    if math.isfinite(value):
      self.values.append(float(value))

  def mean(self) -> float:
    return float(np.mean(self.values)) if self.values else float("nan")

  def std(self) -> float:
    return float(np.std(self.values)) if self.values else float("nan")

  def var(self) -> float:
    return float(np.var(self.values)) if self.values else float("nan")

  def median(self) -> float:
    return float(np.median(self.values)) if self.values else float("nan")

  def percentile(self, q: float) -> float:
    return float(np.percentile(self.values, q)) if self.values else float("nan")


class ConditionStats:
  def __init__(self, condition: EvalCondition) -> None:
    self.condition = condition
    self.successes = 0
    self.failures = 0
    self.success_times = ScalarAccumulator()
    self.censored_times = ScalarAccumulator()
    self.falls = 0
    self.ball_losts = 0
    self.robot_collisions = 0
    self.ball_collisions = 0
    self.robot_contact_counts = ScalarAccumulator()
    self.ball_contact_counts = ScalarAccumulator()
    self.min_ball_clearance = ScalarAccumulator()

    self.velocity_all = ScalarAccumulator()
    self.velocity_blocked = ScalarAccumulator()
    self.velocity_unblocked = ScalarAccumulator()
    self.speed_all = ScalarAccumulator()
    self.speed_blocked = ScalarAccumulator()
    self.speed_unblocked = ScalarAccumulator()
    self.angle_all = ScalarAccumulator()
    self.angle_blocked = ScalarAccumulator()
    self.angle_unblocked = ScalarAccumulator()

    self.ball_pos_error = ScalarAccumulator()
    self.ball_vel_error = ScalarAccumulator()
    self.obstacle_pos_error = ScalarAccumulator()
    self.obstacle_vel_error = ScalarAccumulator()
    self.fov_coverage = ScalarAccumulator()
    self.valid_depth_coverage = ScalarAccumulator()

    self.plot_history: dict[str, list[Any]] = {
      "time": [],
      "progress": [],
      "position_error": [],
      "velocity_error": [],
      "speed_error": [],
      "angle_error": [],
      "ball_speed": [],
      "cmd_speed": [],
      "ball_heading": [],
      "cmd_heading": [],
      "blocked": [],
      "robot_xy": [],
      "robot_yaw": [],
      "ball_xy": [],
      "target_xy": [],
      "obstacles_xy": [],
    }
    self.plot_trial_done = False
    self.target_reached_threshold: float | None = None

  @property
  def episodes(self) -> int:
    return self.successes + self.failures

  @property
  def success_rate(self) -> float:
    return self.successes / self.episodes if self.episodes else float("nan")

  @property
  def fall_rate(self) -> float:
    return self.falls / self.episodes if self.episodes else float("nan")

  @property
  def ball_lost_rate(self) -> float:
    return self.ball_losts / self.episodes if self.episodes else float("nan")

  @property
  def robot_collision_rate(self) -> float:
    return self.robot_collisions / self.episodes if self.episodes else float("nan")

  @property
  def ball_collision_rate(self) -> float:
    return self.ball_collisions / self.episodes if self.episodes else float("nan")


def _make_env(env_cfg: ManagerBasedRlEnvCfg, device: str) -> ManagerBasedRlEnv:
  _cleanup_runtime_memory()
  return make_env(env_cfg, device, render_mode=None)


def _make_live_viewer(
  config: DribblingEvalConfig,
  env: ManagerBasedRlEnv,
  agent: Any,
) -> NativeMujocoViewer | ViserPlayViewer:
  if config.viewer == "viser":
    return ViserPlayViewer(env, agent)
  return NativeMujocoViewer(env, agent, frame_rate=config.eval_viewer_frame_rate)


def _cleanup_runtime_memory() -> None:
  """Release cyclic env references before creating the next render context."""
  for _ in range(3):
    gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
  try:
    import warp as wp

    wp.synchronize_device()
    # Warp keeps a per-device pool of staged allocations. Free it so the next
    # condition's render context does not compound with the previous one.
    for name in ("free_all", "free_all_temporary"):
      fn = getattr(wp, name, None)
      if callable(fn):
        try:
          fn()
        except Exception:
          pass
  except Exception:
    pass


def _stage_task(config: DribblingEvalConfig, stage_index: int):
  if not hasattr(config.task, "obstacle_stage_index"):
    raise ValueError("This evaluator expects a dribbling task with obstacle_stage_index.")
  return replace(config.task, obstacle_stage_index=stage_index)


def _build_env_cfg(
  config: DribblingEvalConfig,
  condition: EvalCondition,
) -> ManagerBasedRlEnvCfg:
  # Evaluation target/obstacle placement is generated below by
  # EvalSceneController. Build from stage 0 so no curriculum/stage placement
  # leaks into the protocol.
  task = _stage_task(config, 0)
  env_cfg = task.play_env_cfg or task.train_env_cfg
  env_cfg = replace(
    env_cfg,
    scene=replace(env_cfg.scene, num_envs=config.num_envs),
    # The evaluator owns trial termination. Keep the underlying env from
    # auto-resetting before metrics/final frames are recorded.
    episode_length_s=1e9,
    terminations={},
    constraints={},
  )
  return env_cfg


class EvalSceneController:
  """Evaluation-only target and obstacle generator.

  The task still exposes the normal `ball_vel` and `adversary` command terms,
  but their reset/update hooks are patched to use this minimal deterministic
  protocol instead of curriculum stage sampling.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    condition: EvalCondition,
    config: DribblingEvalConfig,
  ) -> None:
    self.env = env
    self.condition = condition
    self.config = config
    self.ball_term = env.command_manager.get_term("ball_vel")
    self.obstacle_term = env.command_manager.get_term("adversary")
    n = env.num_envs
    k = self.obstacle_term.cfg.num_obstacles
    self.target_distance = torch.full(
      (n,), config.eval_target_distance, device=env.device
    )
    self.target_heading_offset = torch.full(
      (n,), config.eval_target_heading_offset, device=env.device
    )
    self.obstacle_forward_fractions = torch.zeros((n, k), device=env.device)
    self.obstacle_lateral_offsets = torch.zeros((n, k), device=env.device)
    self.obstacle_speed = torch.full((n,), config.eval_obstacle_speed, device=env.device)
    self.lateral_limit = torch.full((n,), config.eval_lateral_limit, device=env.device)
    self._original_ball_resample_command = self.ball_term._resample_command
    self._original_ball_resample_obstacles = self.ball_term._resample_obstacles
    self._original_ball_update_command = self.ball_term._update_command
    self._original_ball_resample = self.ball_term._resample
    self._original_obstacle_resample_command = self.obstacle_term._resample_command
    self._original_obstacle_update_command = self.obstacle_term._update_command
    self._original_obstacle_resample = self.obstacle_term._resample
    self._installed = False

  @property
  def num_active(self) -> int:
    if self.condition.name in {"no_obstacles", "velocity_no_obstacles"}:
      return 0
    if self.condition.name == "velocity_single_obstacle":
      return 1
    return 3

  def install(self) -> None:
    if self._installed:
      return
    self.obstacle_term.cfg.num_active = self.num_active
    self.obstacle_term.cfg.replay_fraction = 0.0
    self.obstacle_term.cfg.behavior = "eval_direct"

    self.ball_term._resample_command = self.resample_targets
    self.ball_term._resample_obstacles = lambda env_ids: None
    self.ball_term._update_command = self.update_target_command
    # The base CommandTerm.compute() decrements time_left every step and
    # triggers _resample when it expires. During evaluation the evaluator
    # owns resampling, so make the timer-driven path a no-op; otherwise the
    # target would jump mid-trial.
    self.ball_term._resample = lambda env_ids: None
    self.obstacle_term._resample_command = self.resample_obstacles
    self.obstacle_term._update_command = self.update_obstacles
    self.obstacle_term._resample = lambda env_ids: None
    self._installed = True

  def uninstall(self) -> None:
    """Restore patched command hooks to avoid reference cycles between envs."""
    if not self._installed:
      return
    self.ball_term._resample_command = self._original_ball_resample_command
    self.ball_term._resample_obstacles = self._original_ball_resample_obstacles
    self.ball_term._update_command = self._original_ball_update_command
    self.ball_term._resample = self._original_ball_resample
    self.obstacle_term._resample_command = self._original_obstacle_resample_command
    self.obstacle_term._update_command = self._original_obstacle_update_command
    self.obstacle_term._resample = self._original_obstacle_resample
    self._installed = False

  def _as_env_ids(self, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
      return torch.arange(self.env.num_envs, device=self.env.device)
    return env_ids.to(device=self.env.device, dtype=torch.long)

  def _sample_uniform(
    self,
    range_value: tuple[float, float],
    fallback: float,
    shape: tuple[int, ...],
  ) -> torch.Tensor:
    if not self.config.randomize_target_and_obstacle:
      return torch.full(shape, fallback, device=self.env.device)
    lo, hi = range_value
    if hi <= lo:
      return torch.full(shape, lo, device=self.env.device)
    return torch.rand(shape, device=self.env.device) * (hi - lo) + lo

  def _tuple_at(self, values: tuple[float, ...], index: int) -> float:
    return values[min(index, len(values) - 1)]

  def _range_tuple_at(
    self,
    values: tuple[tuple[float, float], ...],
    index: int,
  ) -> tuple[float, float]:
    return values[min(index, len(values) - 1)]

  def _sample_scene_params(self, env_ids: torch.Tensor) -> None:
    n = env_ids.numel()
    self.target_distance[env_ids] = self._sample_uniform(
      self.config.eval_target_distance_range,
      self.config.eval_target_distance,
      (n,),
    )
    self.target_heading_offset[env_ids] = self._sample_uniform(
      self.config.eval_target_heading_offset_range,
      self.config.eval_target_heading_offset,
      (n,),
    )
    self.obstacle_speed[env_ids] = self._sample_uniform(
      self.config.eval_obstacle_speed_range,
      self.config.eval_obstacle_speed,
      (n,),
    )
    self.lateral_limit[env_ids] = self._sample_uniform(
      self.config.eval_lateral_limit_range,
      self.config.eval_lateral_limit,
      (n,),
    )

    for obstacle_idx in range(self.obstacle_term.cfg.num_obstacles):
      self.obstacle_forward_fractions[env_ids, obstacle_idx] = self._sample_uniform(
        self._range_tuple_at(
          self.config.eval_obstacle_forward_fraction_ranges, obstacle_idx
        ),
        self._tuple_at(self.config.eval_obstacle_forward_fractions, obstacle_idx),
        (n,),
      )
      self.obstacle_lateral_offsets[env_ids, obstacle_idx] = self._sample_uniform(
        self._range_tuple_at(
          self.config.eval_obstacle_lateral_offset_ranges, obstacle_idx
        ),
        self._tuple_at(self.config.eval_obstacle_lateral_offsets, obstacle_idx),
        (n,),
      )

  def resample_all(self) -> None:
    env_ids = torch.arange(self.env.num_envs, device=self.env.device)
    self.resample_targets(env_ids)
    self.resample_obstacles(env_ids)

  def resample_targets(self, env_ids: torch.Tensor | slice | None) -> None:
    env_ids = self._as_env_ids(env_ids)
    if env_ids.numel() == 0:
      return
    self._sample_scene_params(env_ids)

    robot_quat = self.env.scene["robot"].data.root_link_quat_w[env_ids]
    robot_yaw = _quat_yaw(robot_quat)
    heading = robot_yaw + self.target_heading_offset[env_ids]
    distance = self.target_distance[env_ids]

    ball_xy = self.env.scene["ball"].data.root_link_pos_w[env_ids, :2]
    target_xy = torch.stack(
      [
        ball_xy[:, 0] + torch.cos(heading) * distance,
        ball_xy[:, 1] + torch.sin(heading) * distance,
      ],
      dim=-1,
    )

    self.ball_term.target_position[env_ids] = target_xy
    self.ball_term.sampled_target_distance[env_ids] = distance
    self.ball_term._recompute_velocity_command(env_ids)
    self.resample_obstacles(env_ids)

  def resample_obstacles(self, env_ids: torch.Tensor | slice | None) -> None:
    env_ids = self._as_env_ids(env_ids)
    if env_ids.numel() == 0:
      return

    self.obstacle_term.cfg.num_active = self.num_active
    for obstacle_idx in range(self.obstacle_term.cfg.num_obstacles):
      if obstacle_idx >= self.num_active:
        self.obstacle_term._park_obstacle(obstacle_idx, env_ids)
        continue
      self._place_obstacle(obstacle_idx, env_ids)

  def update_target_command(self) -> None:
    """Update desired ball velocity without resampling a reached target.

    The training command immediately samples a new target when the old target is
    reached. During evaluation a target reach is terminal for the current trial,
    so the target and obstacle layout must stay fixed until the evaluator starts
    the next trial.
    """
    all_env_ids = torch.arange(self.env.num_envs, device=self.env.device)
    self.ball_term.target_reached_mask[:] = False
    self.ball_term._recompute_velocity_command(all_env_ids)

    ball_xy = self.env.scene["ball"].data.root_link_pos_w[:, :2]
    target_distance = (self.ball_term.target_position - ball_xy).norm(dim=-1)
    reached_env_ids = torch.where(
      target_distance <= self.ball_term.cfg.target_reached_threshold
    )[0]
    if len(reached_env_ids) > 0:
      self.ball_term.target_reached_mask[reached_env_ids] = True

  def _path_frame(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ball_xy = self.env.scene["ball"].data.root_link_pos_w[env_ids, :2]
    target_xy = self.ball_term.target_position[env_ids, :2]
    segment = target_xy - ball_xy
    segment_dir = segment / segment.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    side_dir = torch.stack([-segment_dir[:, 1], segment_dir[:, 0]], dim=-1)
    return ball_xy, segment, side_dir

  def _place_obstacle(self, obstacle_idx: int, env_ids: torch.Tensor) -> None:
    ball_xy, segment, side_dir = self._path_frame(env_ids)
    forward_fraction = self.obstacle_forward_fractions[
      env_ids, obstacle_idx
    ].unsqueeze(-1)
    lateral_offset = self.obstacle_lateral_offsets[
      env_ids, obstacle_idx
    ].unsqueeze(-1)
    position = (
      ball_xy
      + forward_fraction * segment
      + lateral_offset * side_dir
    )
    self.obstacle_term._positions_w[env_ids, obstacle_idx] = position
    self.obstacle_term._velocities_w[env_ids, obstacle_idx] = 0.0
    self.obstacle_term._lateral_signs[env_ids, obstacle_idx] = (
      1.0 if obstacle_idx % 2 else -1.0
    )
    self.obstacle_term._random_dirs[env_ids, obstacle_idx] = side_dir
    self.obstacle_term._write_obstacle_to_sim(obstacle_idx, env_ids, z=0.0)

  def update_obstacles(self) -> None:
    env_ids = torch.arange(self.env.num_envs, device=self.env.device)
    if self.num_active == 0:
      for obstacle_idx in range(self.obstacle_term.cfg.num_obstacles):
        self.obstacle_term._park_obstacle(obstacle_idx, env_ids)
      return

    dt = self.env.step_dt
    ball_xy, segment, side_dir = self._path_frame(env_ids)
    segment_dir = segment / segment.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    for obstacle_idx in range(self.obstacle_term.cfg.num_obstacles):
      if obstacle_idx >= self.num_active:
        self.obstacle_term._park_obstacle(obstacle_idx, env_ids)
        continue

      role = self._role(obstacle_idx)
      obs_xy = self.obstacle_term._positions_w[:, obstacle_idx]
      if role == "static":
        velocity = torch.zeros_like(obs_xy)
      elif role == "attacker":
        to_ball = ball_xy - obs_xy
        velocity = (
          to_ball / to_ball.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        ) * self.obstacle_speed.unsqueeze(-1)
      elif role == "lateral":
        lateral_offset = ((obs_xy - ball_xy) * side_dir).sum(dim=-1)
        sign = self.obstacle_term._lateral_signs[:, obstacle_idx]
        sign = torch.where(
          lateral_offset.abs() > self.lateral_limit,
          -torch.sign(lateral_offset).clamp(min=-1.0, max=1.0),
          sign,
        )
        sign = torch.where(sign == 0.0, torch.ones_like(sign), sign)
        self.obstacle_term._lateral_signs[:, obstacle_idx] = sign
        velocity = (
          sign.unsqueeze(-1) * side_dir * self.obstacle_speed.unsqueeze(-1)
        )
      else:
        velocity = (
          self.obstacle_term._random_dirs[:, obstacle_idx]
          * self.obstacle_speed.unsqueeze(-1)
        )
        radial = obs_xy - ball_xy
        too_far = radial.norm(dim=-1) > (self.target_distance + 1.0)
        if too_far.any():
          self.obstacle_term._random_dirs[too_far, obstacle_idx] = -segment_dir[
            too_far
          ]
          velocity = (
            self.obstacle_term._random_dirs[:, obstacle_idx]
            * self.obstacle_speed.unsqueeze(-1)
          )

      self.obstacle_term._velocities_w[:, obstacle_idx] = velocity
      self.obstacle_term._positions_w[:, obstacle_idx] = obs_xy + velocity * dt
      self.obstacle_term._write_obstacle_to_sim(obstacle_idx, env_ids, z=0.0)

  def _role(self, obstacle_idx: int) -> str:
    if self.condition.name == "static_3":
      return "static"
    if self.condition.name == "velocity_single_obstacle":
      return "attacker"
    if self.condition.name == "moving_3":
      if obstacle_idx == 0:
        return "attacker"
      if obstacle_idx == 1:
        return "lateral"
      return "distractor"
    return "static"


def _quat_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat_wxyz.unbind(dim=-1)
  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  return torch.atan2(siny_cosp, cosy_cosp)


def _blocked_mask(
  env: ManagerBasedRlEnv,
  config: DribblingEvalConfig,
) -> torch.Tensor:
  term = env.command_manager.get_term("adversary")
  if term.cfg.num_active == 0:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2]
  target_xy = env.command_manager.get_term("ball_vel").target_position[:, :2]
  target_vec = target_xy - ball_xy
  target_dist = target_vec.norm(dim=-1)
  target_dir = target_vec / target_dist.unsqueeze(-1).clamp(min=1e-6)

  obs_xy = term.obstacle_positions_w[:, : term.cfg.num_active]
  ball_to_obs = obs_xy - ball_xy.unsqueeze(1)
  obs_forward = (ball_to_obs * target_dir.unsqueeze(1)).sum(dim=-1)
  obs_lateral = (
    ball_to_obs - obs_forward.unsqueeze(-1) * target_dir.unsqueeze(1)
  ).norm(dim=-1)

  relevant = (
    (target_dist.unsqueeze(1) > 1e-6)
    & (obs_forward > 0.0)
    & (obs_forward < target_dist.unsqueeze(1))
    & (obs_forward <= config.blocked_detection_range)
    & (obs_lateral <= config.blocked_tube_radius)
  )
  return relevant.any(dim=1)


def _safe_angle_error(ball_vel: torch.Tensor, cmd_vel: torch.Tensor) -> torch.Tensor:
  ball_speed = ball_vel.norm(dim=-1)
  cmd_speed = cmd_vel.norm(dim=-1)
  psi_ball = torch.atan2(ball_vel[:, 1], ball_vel[:, 0])
  psi_cmd = torch.atan2(cmd_vel[:, 1], cmd_vel[:, 0])
  err = (psi_ball - psi_cmd + math.pi) % (2 * math.pi) - math.pi
  valid = (ball_speed > 1e-4) & (cmd_speed > 1e-4)
  return torch.where(valid, err.abs(), torch.full_like(err, float("nan")))


def _min_obstacle_distances(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  term = env.command_manager.get_term("adversary")
  inf = torch.full((env.num_envs,), float("inf"), device=env.device)
  if term.cfg.num_active == 0:
    return inf, inf
  obs_xy = term.obstacle_positions_w[:, : term.cfg.num_active]
  robot_xy = env.scene["robot"].data.root_link_pos_w[:, :2]
  ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2]
  robot_dist = (obs_xy - robot_xy.unsqueeze(1)).norm(dim=-1).min(dim=-1).values
  ball_dist = (obs_xy - ball_xy.unsqueeze(1)).norm(dim=-1).min(dim=-1).values
  return robot_dist, ball_dist


def _per_obstacle_distances(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor] | None:
  """Return (robot_dist, ball_dist) of shape (N, num_active). None if no obstacles."""
  term = env.command_manager.get_term("adversary")
  if term.cfg.num_active == 0:
    return None
  obs_xy = term.obstacle_positions_w[:, : term.cfg.num_active]
  robot_xy = env.scene["robot"].data.root_link_pos_w[:, :2]
  ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2]
  robot_dist = (obs_xy - robot_xy.unsqueeze(1)).norm(dim=-1)
  ball_dist = (obs_xy - ball_xy.unsqueeze(1)).norm(dim=-1)
  return robot_dist, ball_dist


def _target_distance(env: ManagerBasedRlEnv) -> torch.Tensor:
  ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2]
  target_xy = env.command_manager.get_term("ball_vel").target_position[:, :2]
  return (target_xy - ball_xy).norm(dim=-1)


def _fall_mask(env: ManagerBasedRlEnv, limit_deg: float) -> torch.Tensor:
  robot = env.scene["robot"]
  gravity = getattr(robot.data, "projected_gravity_b", None)
  if gravity is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return gravity[:, :2].norm(dim=-1) > math.sin(math.radians(limit_deg))


def _ball_lost_mask(env: ManagerBasedRlEnv, max_robot_ball_distance: float) -> torch.Tensor:
  robot_xy = env.scene["robot"].data.root_link_pos_w[:, :2]
  ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2]
  return (ball_xy - robot_xy).norm(dim=-1) > max_robot_ball_distance


def _prediction_errors(
  env: ManagerBasedRlEnv,
  active_obstacles: bool,
  valid_only: bool,
) -> dict[str, torch.Tensor | None]:
  rma = getattr(getattr(env, "unwrapped", env), "rma_manager", None)
  term = None if rma is None else rma._terms.get("dribbling")
  if term is None:
    return {}

  valid_mask = term.get_adaptation_mask()
  if valid_mask is None:
    valid_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  fov_mask = valid_mask.clone()
  if not valid_only:
    valid_mask = torch.ones_like(valid_mask, dtype=torch.bool)

  pred_ball = term.predict_ball_state()
  result: dict[str, torch.Tensor | None] = {
    "fov_mask": fov_mask,
    "valid_mask": valid_mask,
  }
  if pred_ball is not None:
    gt_ball = torch.cat([ball_position(env), ball_velocity_xy(env)], dim=-1)
    result["ball_pos"] = (pred_ball[:, :2] - gt_ball[:, :2]).norm(dim=-1)
    result["ball_vel"] = (pred_ball[:, 2:] - gt_ball[:, 2:]).norm(dim=-1)

  z_adapt = getattr(term, "_last_z_adapt", None)
  obs_head = getattr(term, "_obstacle_head", None)
  if active_obstacles and z_adapt is not None and obs_head is not None:
    obs_pred_norm = obs_head(z_adapt)
    obs_pred = torch.cat(
      [
        obs_pred_norm[:, :2] * term.cfg.obs_pos_scale,
        obs_pred_norm[:, 2:] * term.cfg.obs_vel_scale,
      ],
      dim=-1,
    )
    gt_obs = torch.cat([obstacle_position_b(env), obstacle_velocity_b(env)], dim=-1)
    result["obstacle_pos"] = (obs_pred[:, :2] - gt_obs[:, :2]).norm(dim=-1)
    result["obstacle_vel"] = (obs_pred[:, 2:] - gt_obs[:, 2:]).norm(dim=-1)

  return result


def _record_step_metrics(
  env: ManagerBasedRlEnv,
  config: DribblingEvalConfig,
  stats: ConditionStats,
  trial_start_distance: torch.Tensor,
  fov_steps: torch.Tensor,
  valid_steps: torch.Tensor,
  step_counts: torch.Tensor,
  active_mask: torch.Tensor,
) -> None:
  ball_vel = env.scene["ball"].data.root_link_lin_vel_w[:, :2]
  cmd_vel = env.command_manager.get_command("ball_vel")[:, :2]
  vector_error = (ball_vel - cmd_vel).norm(dim=-1)
  speed_error = (ball_vel.norm(dim=-1) - cmd_vel.norm(dim=-1)).abs()
  angle_error = _safe_angle_error(ball_vel, cmd_vel)
  blocked = _blocked_mask(env, config)

  stats.velocity_all.add_tensor(vector_error, active_mask)
  stats.velocity_blocked.add_tensor(vector_error, active_mask & blocked)
  stats.velocity_unblocked.add_tensor(vector_error, active_mask & ~blocked)
  stats.speed_all.add_tensor(speed_error, active_mask)
  stats.speed_blocked.add_tensor(speed_error, active_mask & blocked)
  stats.speed_unblocked.add_tensor(speed_error, active_mask & ~blocked)
  stats.angle_all.add_tensor(angle_error, active_mask & torch.isfinite(angle_error))
  stats.angle_blocked.add_tensor(angle_error, active_mask & blocked & torch.isfinite(angle_error))
  stats.angle_unblocked.add_tensor(angle_error, active_mask & ~blocked & torch.isfinite(angle_error))

  robot_dist, ball_dist = _min_obstacle_distances(env)
  if torch.isfinite(ball_dist).any():
    stats.min_ball_clearance.add_tensor(ball_dist, active_mask & torch.isfinite(ball_dist))

  pred = _prediction_errors(
    env,
    active_obstacles=env.command_manager.get_term("adversary").cfg.num_active > 0,
    valid_only=config.perception_valid_only,
  )
  valid_mask = pred.get("valid_mask")
  fov_mask = pred.get("fov_mask")
  if isinstance(valid_mask, torch.Tensor):
    coverage_mask = fov_mask if isinstance(fov_mask, torch.Tensor) else valid_mask
    fov_steps += coverage_mask.float() * active_mask.float()
    valid_steps += coverage_mask.float() * active_mask.float()
    if "ball_pos" in pred:
      stats.ball_pos_error.add_tensor(pred["ball_pos"], active_mask & valid_mask)  # type: ignore[arg-type]
      stats.ball_vel_error.add_tensor(pred["ball_vel"], active_mask & valid_mask)  # type: ignore[arg-type]
    if "obstacle_pos" in pred:
      stats.obstacle_pos_error.add_tensor(pred["obstacle_pos"], active_mask & valid_mask)  # type: ignore[arg-type]
      stats.obstacle_vel_error.add_tensor(pred["obstacle_vel"], active_mask & valid_mask)  # type: ignore[arg-type]

  step_counts += active_mask.float()

  plot_idx = min(config.plot_condition_env_index, env.num_envs - 1)
  if config.save_plots and not stats.plot_trial_done and bool(active_mask[plot_idx].item()):
    ball_xy = env.scene["ball"].data.root_link_pos_w[plot_idx, :2]
    robot_xy = env.scene["robot"].data.root_link_pos_w[plot_idx, :2]
    robot_yaw = _quat_yaw(env.scene["robot"].data.root_link_quat_w[plot_idx : plot_idx + 1])[0]
    target_xy = env.command_manager.get_term("ball_vel").target_position[plot_idx, :2]
    current_dist = (target_xy - ball_xy).norm()
    start_dist = trial_start_distance[plot_idx].clamp(min=1e-6)
    progress = (1.0 - current_dist / start_dist).clamp(0.0, 1.0)
    obstacle_term = env.command_manager.get_term("adversary")
    obstacles = obstacle_term.obstacle_positions_w[
      plot_idx, : obstacle_term.cfg.num_active
    ].detach().cpu().numpy()
    ball_v = ball_vel[plot_idx]
    cmd_v = cmd_vel[plot_idx]
    ball_sp = float(ball_v.norm().item())
    cmd_sp = float(cmd_v.norm().item())
    ball_hd = float(torch.atan2(ball_v[1], ball_v[0]).item()) if ball_sp > 1e-4 else float("nan")
    cmd_hd = float(torch.atan2(cmd_v[1], cmd_v[0]).item()) if cmd_sp > 1e-4 else float("nan")
    ang_err = float(angle_error[plot_idx].item())
    stats.plot_history["time"].append(len(stats.plot_history["time"]) * env.step_dt)
    stats.plot_history["progress"].append(float(progress.item()))
    stats.plot_history["position_error"].append(float(current_dist.item()))
    stats.plot_history["velocity_error"].append(float(vector_error[plot_idx].item()))
    stats.plot_history["speed_error"].append(float(speed_error[plot_idx].item()))
    stats.plot_history["angle_error"].append(ang_err)
    stats.plot_history["ball_speed"].append(ball_sp)
    stats.plot_history["cmd_speed"].append(cmd_sp)
    stats.plot_history["ball_heading"].append(ball_hd)
    stats.plot_history["cmd_heading"].append(cmd_hd)
    stats.plot_history["blocked"].append(bool(blocked[plot_idx].item()))
    stats.plot_history["robot_xy"].append(robot_xy.detach().cpu().numpy())
    stats.plot_history["robot_yaw"].append(float(robot_yaw.item()))
    stats.plot_history["ball_xy"].append(ball_xy.detach().cpu().numpy())
    stats.plot_history["target_xy"].append(target_xy.detach().cpu().numpy())
    stats.plot_history["obstacles_xy"].append(obstacles)


def _run_condition(
  config: DribblingEvalConfig,
  condition: EvalCondition,
  seed: int,
  episodes: int,
  env: ManagerBasedRlEnv,
  agent: Any,
) -> ConditionStats:
  set_seed(seed)
  scene_controller = EvalSceneController(env, condition, config)
  scene_controller.install()
  obs, _ = env.reset(seed=seed)
  scene_controller.resample_all()
  live_viewer = None
  next_step_time = time.perf_counter()
  next_render_time = next_step_time
  if config.view_during_eval:
    if env.num_envs != 1:
      raise ValueError("--view-during-eval requires --num-envs 1.")
    live_viewer = _make_live_viewer(config, env, agent)
    live_viewer.setup()
    live_viewer.sync_env_to_viewer()

  stats = ConditionStats(condition)
  stats.target_reached_threshold = float(
    env.command_manager.get_term("ball_vel").cfg.target_reached_threshold
  )
  elapsed = torch.zeros(env.num_envs, device=env.device)
  trial_start_distance = _target_distance(env).clone()
  min_ball_clearance = torch.full((env.num_envs,), float("inf"), device=env.device)
  robot_collision_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  ball_collision_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  fall_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  ball_lost_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  num_obstacle_slots = env.command_manager.get_term("adversary").cfg.num_obstacles
  robot_contact_prev = torch.zeros(
    (env.num_envs, num_obstacle_slots), dtype=torch.bool, device=env.device
  )
  ball_contact_prev = torch.zeros(
    (env.num_envs, num_obstacle_slots), dtype=torch.bool, device=env.device
  )
  robot_contact_count = torch.zeros(env.num_envs, device=env.device)
  ball_contact_count = torch.zeros(env.num_envs, device=env.device)
  fov_steps = torch.zeros(env.num_envs, device=env.device)
  valid_steps = torch.zeros(env.num_envs, device=env.device)
  step_counts = torch.zeros(env.num_envs, device=env.device)
  active_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  dt = float(env.step_dt)

  progress = tqdm(
    total=episodes,
    desc=f"{condition.label} seed={seed}",
    unit="trial",
    disable=not config.progress_bar,
    leave=False,
  )
  try:
    while stats.episodes < episodes:
      if live_viewer is not None and live_viewer.is_running():
        live_viewer.sync_viewer_to_env()

      _record_step_metrics(
        env,
        config,
        stats,
        trial_start_distance,
        fov_steps,
        valid_steps,
        step_counts,
        active_mask,
      )

      robot_dist, ball_dist = _min_obstacle_distances(env)
      min_ball_clearance = torch.minimum(min_ball_clearance, ball_dist)
      robot_collision_seen |= robot_dist <= config.robot_obstacle_collision_distance
      ball_collision_seen |= ball_dist <= config.ball_obstacle_collision_distance
      fall_seen |= _fall_mask(env, config.fall_limit_deg)

      with torch.no_grad():
        actions = agent(obs)
      obs, _, _, _, _ = env.step(actions)
      elapsed += dt

      robot_dist, ball_dist = _min_obstacle_distances(env)
      min_ball_clearance = torch.minimum(min_ball_clearance, ball_dist)
      robot_collision_seen |= robot_dist <= config.robot_obstacle_collision_distance
      ball_collision_seen |= ball_dist <= config.ball_obstacle_collision_distance
      # Count per-obstacle rising-edge contacts (outside threshold -> inside).
      # A single prolonged proximity counts as one contact, and touching two
      # different obstacles in the same trial counts as two.
      per_obs = _per_obstacle_distances(env)
      if per_obs is not None:
        robot_per, ball_per = per_obs
        num_active = robot_per.shape[1]
        robot_now = robot_per <= config.robot_obstacle_collision_distance
        ball_now = ball_per <= config.ball_obstacle_collision_distance
        robot_rising = robot_now & ~robot_contact_prev[:, :num_active]
        ball_rising = ball_now & ~ball_contact_prev[:, :num_active]
        robot_contact_count += robot_rising.sum(dim=1).float() * active_mask.float()
        ball_contact_count += ball_rising.sum(dim=1).float() * active_mask.float()
        robot_contact_prev[:, :num_active] = robot_now
        ball_contact_prev[:, :num_active] = ball_now
      fell_over = _fall_mask(env, config.fall_limit_deg)
      ball_lost = _ball_lost_mask(env, config.ball_lost_distance)
      fall_seen |= fell_over
      ball_lost_seen |= ball_lost

      if live_viewer is not None and live_viewer.is_running():
        now = time.perf_counter()
        if now >= next_render_time:
          live_viewer.sync_env_to_viewer()
          next_render_time = now + 1.0 / config.eval_viewer_frame_rate
        if config.eval_viewer_realtime:
          next_step_time += dt
          sleep_s = next_step_time - time.perf_counter()
          if sleep_s > 0.0:
            time.sleep(sleep_s)
          elif sleep_s < -dt:
            next_step_time = time.perf_counter()

      reached = env.command_manager.get_term("ball_vel").target_reached_mask.clone()
      timed_out = elapsed >= config.max_trial_s
      done = timed_out | fell_over | ball_lost
      finished = reached | done
      if not finished.any():
        continue

      previous_episodes = stats.episodes
      done_indices = finished.nonzero(as_tuple=False).flatten()
      next_trial_indices: list[torch.Tensor] = []
      for idx in done_indices:
        if stats.episodes >= episodes:
          active_mask[idx] = False
          continue
        success = bool(reached[idx].item())
        if success:
          stats.successes += 1
          stats.success_times.add(float(elapsed[idx].item()))
        else:
          stats.failures += 1
        stats.censored_times.add(
          float(elapsed[idx].item()) if success else float(config.max_trial_s)
        )
        if bool(fall_seen[idx].item()):
          stats.falls += 1
        if not success and bool(ball_lost_seen[idx].item()):
          stats.ball_losts += 1
        if bool(robot_collision_seen[idx].item()):
          stats.robot_collisions += 1
        if bool(ball_collision_seen[idx].item()):
          stats.ball_collisions += 1
        stats.robot_contact_counts.add(float(robot_contact_count[idx].item()))
        stats.ball_contact_counts.add(float(ball_contact_count[idx].item()))
        if torch.isfinite(min_ball_clearance[idx]):
          stats.min_ball_clearance.add(float(min_ball_clearance[idx].item()))
        if step_counts[idx] > 0:
          stats.fov_coverage.add(float((fov_steps[idx] / step_counts[idx]).item()))
          stats.valid_depth_coverage.add(
            float((valid_steps[idx] / step_counts[idx]).item())
          )
        if idx.item() == min(config.plot_condition_env_index, env.num_envs - 1):
          stats.plot_trial_done = True
        if stats.episodes < episodes:
          next_trial_indices.append(idx)
        else:
          active_mask[idx] = False

      completed_now = stats.episodes - previous_episodes
      if completed_now > 0:
        progress.update(completed_now)
        progress.set_postfix(success=f"{stats.success_rate:.1%}", refresh=False)

      if next_trial_indices:
        next_env_ids = torch.stack(next_trial_indices)
        elapsed[next_env_ids] = 0.0
        min_ball_clearance[next_env_ids] = float("inf")
        robot_collision_seen[next_env_ids] = False
        ball_collision_seen[next_env_ids] = False
        robot_contact_count[next_env_ids] = 0.0
        ball_contact_count[next_env_ids] = 0.0
        robot_contact_prev[next_env_ids] = False
        ball_contact_prev[next_env_ids] = False
        fall_seen[next_env_ids] = False
        ball_lost_seen[next_env_ids] = False
        fov_steps[next_env_ids] = 0.0
        valid_steps[next_env_ids] = 0.0
        step_counts[next_env_ids] = 0.0
        scene_controller.resample_targets(next_env_ids)
        trial_start_distance[next_env_ids] = _target_distance(env)[next_env_ids]
  finally:
    progress.close()
    if live_viewer is not None:
      live_viewer.close()
    scene_controller.uninstall()
  return stats


def _merge_stats(stats_list: list[ConditionStats]) -> ConditionStats:
  merged = ConditionStats(stats_list[0].condition)
  for stats in stats_list:
    merged.successes += stats.successes
    merged.failures += stats.failures
    merged.falls += stats.falls
    merged.ball_losts += stats.ball_losts
    merged.robot_collisions += stats.robot_collisions
    merged.ball_collisions += stats.ball_collisions
    for name, value in stats.__dict__.items():
      if isinstance(value, ScalarAccumulator):
        getattr(merged, name).values.extend(value.values)
    if not merged.plot_history["progress"] and stats.plot_history["progress"]:
      merged.plot_history = stats.plot_history
      merged.plot_trial_done = stats.plot_trial_done
    if merged.target_reached_threshold is None:
      merged.target_reached_threshold = stats.target_reached_threshold
  return merged


def _run_viewer_condition(
  config: DribblingEvalConfig,
  condition_name: str,
  seed: int,
  device: torch.device,
) -> None:
  condition = CONDITIONS_BY_NAME[condition_name]
  set_seed(seed)
  env_cfg = _build_env_cfg(config, condition)
  env = _make_env(env_cfg, str(device))
  scene_controller = EvalSceneController(env, condition, config)
  scene_controller.install()
  env.reset(seed=seed)
  scene_controller.resample_all()
  agent = create_agent(config, env, device)
  logger.info(
    f"Viewer condition: {condition.label} "
    f"(name={condition.name}, num_envs={config.num_envs}, seed={seed})"
  )
  if config.viewer == "viser":
    viewer = ViserPlayViewer(env, agent)
  else:
    viewer = NativeMujocoViewer(env, agent)
  try:
    viewer.run()
  finally:
    scene_controller.uninstall()
    env.close()
    _cleanup_runtime_memory()


def _fmt(value: float, digits: int = 3, percent: bool = False) -> str:
  if not math.isfinite(value):
    return "n/a"
  if percent:
    return f"{100.0 * value:.1f}%"
  return f"{value:.{digits}f}"


def _main_task_table(stats: list[ConditionStats]) -> str:
  lines = [
    "Columns marked **[T]** end a trial; unmarked columns are informational safety metrics.",
    "A trial ends on target reach (success), timeout (failure), fall (failure), or ball-lost (failure).",
    "",
    "| Environment | Episodes | Success rate **[T]** | Time to target **[T]** | Censored time **[T]** | Fall rate **[T]** | Ball-lost rate **[T]** | Robot collision | Robot contacts/trial | Ball collision | Ball contacts/trial | Min clearance |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for s in stats:
    obstacle_free = s.condition.stage_index == 0 and not s.condition.static_three
    lines.append(
      "| "
      + " | ".join(
        [
          s.condition.label,
          str(s.episodes),
          _fmt(s.success_rate, percent=True),
          _fmt(s.success_times.mean()),
          _fmt(s.censored_times.mean()),
          _fmt(s.fall_rate, percent=True),
          _fmt(s.ball_lost_rate, percent=True),
          "n/a" if obstacle_free else _fmt(s.robot_collision_rate, percent=True),
          "n/a" if obstacle_free else _fmt(s.robot_contact_counts.mean()),
          "n/a" if obstacle_free else _fmt(s.ball_collision_rate, percent=True),
          "n/a" if obstacle_free else _fmt(s.ball_contact_counts.mean()),
          "n/a" if obstacle_free else _fmt(s.min_ball_clearance.mean()),
        ]
      )
      + " |"
    )
  return "\n".join(lines)


def _velocity_table(stats: list[ConditionStats]) -> str:
  lines = [
    "| Setup | Segment | Vector error mean | Vector error var | Speed error mean | Angular error mean |",
    "|---|---|---:|---:|---:|---:|",
  ]
  rows = [
    ("all timesteps", "velocity_all", "speed_all", "angle_all"),
    ("unblocked", "velocity_unblocked", "speed_unblocked", "angle_unblocked"),
    ("blocked", "velocity_blocked", "speed_blocked", "angle_blocked"),
  ]
  for s in stats:
    for label, v_name, sp_name, a_name in rows:
      v = getattr(s, v_name)
      sp = getattr(s, sp_name)
      a = getattr(s, a_name)
      lines.append(
        f"| {s.condition.label} | {label} | {_fmt(v.mean())} | {_fmt(v.var())} | "
        f"{_fmt(sp.mean())} | {_fmt(a.mean())} |"
      )
  return "\n".join(lines)


def _perception_table(stats: list[ConditionStats]) -> str:
  lines = [
    "| Environment | Ball pos error | Ball vel error | Obstacle pos error | Obstacle vel error | FOV coverage | Valid depth coverage |",
    "|---|---:|---:|---:|---:|---:|---:|",
  ]
  for s in stats:
    obstacle_free = s.condition.stage_index == 0 and not s.condition.static_three
    lines.append(
      "| "
      + " | ".join(
        [
          s.condition.label,
          _fmt(s.ball_pos_error.mean()),
          _fmt(s.ball_vel_error.mean()),
          "n/a" if obstacle_free else _fmt(s.obstacle_pos_error.mean()),
          "n/a" if obstacle_free else _fmt(s.obstacle_vel_error.mean()),
          _fmt(s.fov_coverage.mean(), percent=True),
          _fmt(s.valid_depth_coverage.mean(), percent=True),
        ]
      )
      + " |"
    )
  return "\n".join(lines)


def _plot_segmented_line(
  ax: Any,
  x: np.ndarray,
  y: np.ndarray,
  blocked: np.ndarray,
  *,
  unblocked_color: str = "#1f77b4",
  blocked_color: str = "#d62728",
  linewidth: float = 0.9,
) -> None:
  """Plot connected line segments colored by whether the obstacle is relevant."""
  if len(x) == 0:
    return
  if len(x) == 1:
    color = blocked_color if blocked[0] else unblocked_color
    ax.plot(x, y, color=color, linewidth=linewidth)
    return

  labeled = {"unblocked": False, "blocked": False}
  for i in range(len(x) - 1):
    is_blocked = bool(blocked[i] or blocked[i + 1])
    label = "blocked" if is_blocked else "unblocked"
    ax.plot(
      x[i : i + 2],
      y[i : i + 2],
      color=blocked_color if is_blocked else unblocked_color,
      linewidth=linewidth,
      label=None if labeled[label] else label,
    )
    labeled[label] = True


def _save_plots(stats: list[ConditionStats], output_dir: Path) -> list[Path]:
  try:
    import matplotlib.pyplot as plt
  except Exception as exc:
    logger.warning(f"Could not import matplotlib, skipping plots: {exc}")
    return []

  compact_rc = {
    "font.size": 5,
    "axes.titlesize": 5,
    "axes.labelsize": 4,
    "xtick.labelsize": 4,
    "ytick.labelsize": 4,
    "legend.fontsize": 4,
    "legend.handlelength": 1.8,
    "legend.handletextpad": 0.6,
    "legend.borderpad": 0.5,
    "legend.borderaxespad": 0.5,
    "legend.labelspacing": 0.4,
    "axes.labelpad": 2.5,
    "axes.titlepad": 3.0,
    "xtick.major.pad": 2.5,
    "ytick.major.pad": 2.5,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.06,
    "figure.constrained_layout.w_pad": 0.06,
    "figure.constrained_layout.hspace": 0.04,
    "figure.constrained_layout.wspace": 0.04,
  }

  plot_paths: list[Path] = []
  with plt.rc_context(compact_rc):
    for s in stats:
      hist = s.plot_history
      if not hist["progress"]:
        continue

      time = np.asarray(hist["time"])
      pos_error = np.asarray(hist["position_error"])
      vel_error = np.asarray(hist["velocity_error"])
      angle_error_deg = np.degrees(np.asarray(hist["angle_error"]))
      ball_speed = np.asarray(hist["ball_speed"])
      cmd_speed = np.asarray(hist["cmd_speed"])
      ball_heading_deg = np.degrees(np.asarray(hist["ball_heading"]))
      cmd_heading_deg = np.degrees(np.asarray(hist["cmd_heading"]))
      blocked = np.asarray(hist["blocked"], dtype=bool)

      cmd_color = "#2ca02c"
      fig, axes = plt.subplots(5, 1, figsize=(3.6, 6.0), sharex=True)
      _plot_segmented_line(axes[0], time, pos_error, blocked)
      axes[0].set_ylabel("Pos err [m]")
      axes[0].set_title(f"{s.condition.label}: tracking errors vs time")
      axes[0].grid(True, alpha=0.3)
      if s.target_reached_threshold is not None:
        axes[0].axhline(
          s.target_reached_threshold,
          color="#555555",
          linestyle="--",
          linewidth=0.7,
          label=f"target reached ({s.target_reached_threshold:g} m)",
        )
      axes[0].legend()
      axes[0].tick_params(length=2)

      _plot_segmented_line(axes[1], time, vel_error, blocked)
      axes[1].set_ylabel("Vel err [m/s]")
      axes[1].grid(True, alpha=0.3)
      axes[1].legend()
      axes[1].tick_params(length=2)

      axes[2].plot(time, cmd_speed, color=cmd_color, linewidth=0.9, label="cmd")
      _plot_segmented_line(axes[2], time, ball_speed, blocked)
      axes[2].set_ylabel("Speed [m/s]")
      axes[2].grid(True, alpha=0.3)
      axes[2].legend()
      axes[2].tick_params(length=2)

      axes[3].plot(time, cmd_heading_deg, color=cmd_color, linewidth=0.9, label="cmd")
      _plot_segmented_line(axes[3], time, ball_heading_deg, blocked)
      axes[3].set_ylabel("Heading [deg]")
      axes[3].grid(True, alpha=0.3)
      axes[3].legend()
      axes[3].tick_params(length=2)

      _plot_segmented_line(axes[4], time, angle_error_deg, blocked)
      axes[4].set_xlabel("Time [s]")
      axes[4].set_ylabel("Angle err [deg]")
      axes[4].grid(True, alpha=0.3)
      axes[4].legend()
      axes[4].tick_params(length=2)

      path = output_dir / f"{s.condition.name}_tracking_errors.png"
      fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.12)
      plt.close(fig)
      plot_paths.append(path)

      robot_xy = np.asarray(hist["robot_xy"])
      ball_xy = np.asarray(hist["ball_xy"])
      target_xy = np.asarray(hist["target_xy"])
      robot_yaw = np.asarray(hist["robot_yaw"])
      fig, ax = plt.subplots(figsize=(3.2, 3.2))
      ax.plot(robot_xy[:, 0], robot_xy[:, 1], label="robot", c="#2ca02c", linewidth=0.8)
      ax.plot(ball_xy[:, 0], ball_xy[:, 1], label="ball", c="#ff7f0e", linewidth=0.8)
      ax.scatter(target_xy[-1, 0], target_xy[-1, 1], marker="*", s=60, label="target", c="#1f77b4")
      stride = max(1, len(robot_xy) // 20)
      ax.quiver(
        robot_xy[::stride, 0],
        robot_xy[::stride, 1],
        np.cos(robot_yaw[::stride]),
        np.sin(robot_yaw[::stride]),
        angles="xy",
        scale_units="xy",
        scale=10.0,
        width=0.003,
        color="#2ca02c",
        alpha=0.7,
      )
      obstacle_samples = [o for o in hist["obstacles_xy"] if len(o) > 0]
      if obstacle_samples:
        obs = np.concatenate(obstacle_samples, axis=0)
        ax.scatter(obs[:, 0], obs[:, 1], s=8, label="obstacles", c="#d62728", alpha=0.35)
      ax.set_xlabel("X [m]")
      ax.set_ylabel("Y [m]")
      ax.set_title(f"{s.condition.label}: trajectory")
      ax.axis("equal")
      ax.grid(True, alpha=0.3)
      ax.legend()
      ax.tick_params(length=2)
      path = output_dir / f"{s.condition.name}_trajectory.png"
      fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.12)
      plt.close(fig)
      plot_paths.append(path)
  return plot_paths


def _build_report(
  config: DribblingEvalConfig,
  checkpoint_path: Path | None,
  main_stats: list[ConditionStats],
  velocity_stats: list[ConditionStats],
  plot_paths: list[Path],
) -> str:
  lines = [
    "# Dribbling Evaluation Results",
    "",
    f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
    f"Checkpoint: `{checkpoint_path}`",
    "",
    "## Parameters",
    "",
    f"- Task: `{config.task.name}`",
    f"- Agent: `{config.agent}`",
    f"- View during eval: `{config.view_during_eval}`",
    f"- Num envs: `{config.num_envs}`",
    f"- Main episodes per condition per seed: `{config.episodes_per_condition}`",
    f"- Velocity episodes per condition per seed: `{config.velocity_episodes_per_condition}`",
    f"- Seeds: `{', '.join(str(s) for s in config.seeds)}`",
    f"- Max trial duration: `{config.max_trial_s} s`",
    f"- Progress bar: `{config.progress_bar}`",
    f"- Blocked detection range: `{config.blocked_detection_range} m`",
    f"- Blocked tube radius: `{config.blocked_tube_radius} m`",
    f"- Robot-obstacle collision distance: `{config.robot_obstacle_collision_distance} m`",
    f"- Ball-obstacle collision distance: `{config.ball_obstacle_collision_distance} m`",
    f"- Ball-lost distance: `{config.ball_lost_distance} m`",
    f"- Perception valid-only: `{config.perception_valid_only}`",
    f"- Randomize target and obstacle: `{config.randomize_target_and_obstacle}`",
    f"- Direct eval target distance: `{config.eval_target_distance} m`",
    f"- Direct eval target distance range: `{config.eval_target_distance_range}`",
    f"- Direct eval target heading offset: `{config.eval_target_heading_offset} rad`",
    f"- Direct eval target heading offset range: `{config.eval_target_heading_offset_range}`",
    f"- Direct eval obstacle forward fractions: `{config.eval_obstacle_forward_fractions}`",
    f"- Direct eval obstacle forward fraction ranges: `{config.eval_obstacle_forward_fraction_ranges}`",
    f"- Direct eval obstacle lateral offsets: `{config.eval_obstacle_lateral_offsets} m`",
    f"- Direct eval obstacle lateral offset ranges: `{config.eval_obstacle_lateral_offset_ranges}`",
    f"- Direct eval obstacle speed: `{config.eval_obstacle_speed} m/s`",
    f"- Direct eval obstacle speed range: `{config.eval_obstacle_speed_range}`",
    f"- Direct eval lateral limit: `{config.eval_lateral_limit} m`",
    f"- Direct eval lateral limit range: `{config.eval_lateral_limit_range}`",
    "",
  ]
  if main_stats:
    lines.extend(["## Main Task Metrics", "", _main_task_table(main_stats), ""])
  if velocity_stats:
    lines.extend(["## Ball Velocity Tracking Diagnostic", "", _velocity_table(velocity_stats), ""])
  if main_stats:
    lines.extend(["## Perception And Depth Encoder Metrics", "", _perception_table(main_stats), ""])
  if plot_paths:
    lines.extend(["## Plots", ""])
    for path in plot_paths:
      lines.append(f"- `{path}`")
    lines.append("")
  return "\n".join(lines)


def main() -> None:
  config = tyro.cli(DribblingEvalConfig, config=(tyro.conf.CascadeSubcommandArgs,))
  if hasattr(config.task, "obstacle_stage_index"):
    config = replace(
      config,
      task=replace(config.task, obstacle_stage_index=config.obstacle_stage_index),
    )

  logger.remove()
  logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
  )

  checkpoint_path = _resolve_checkpoint(config.checkpoint)
  if config.agent == "trained" and checkpoint_path is None:
    raise FileNotFoundError("No checkpoint found. Provide --checkpoint <path>.")

  device_id = _parse_single_cuda_device(config.cuda)
  configure_torch_backends()
  if config.use_cuda:
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_id)
  device = get_device(cuda=config.use_cuda, device_id=device_id)

  output_dir = Path(config.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
  output_dir.mkdir(parents=True, exist_ok=True)

  logger.info(f"Device: {device}")
  logger.info(f"Output directory: {output_dir}")
  logger.info(f"Checkpoint: {checkpoint_path}")

  if config.viewer_condition != "none":
    _run_viewer_condition(
      config,
      condition_name=config.viewer_condition,
      seed=config.seeds[0],
      device=device,
    )
    return

  main_stats: list[ConditionStats] = []
  velocity_stats: list[ConditionStats] = []

  # Build the env and agent ONCE and reuse them across every condition and
  # seed. Rebuilding per condition leaks warp render-context buffers (~2 GB)
  # because nothing in the mjlab env class explicitly releases them on close;
  # matches training, which is memory-stable for the same reason.
  env_cfg = _build_env_cfg(config, MAIN_CONDITIONS[0])
  env = _make_env(env_cfg, str(device))
  agent = create_agent(config, env, device)
  try:
    if config.run_main:
      for condition in MAIN_CONDITIONS:
        logger.info(f"Running main condition: {condition.label}")
        per_seed = [
          _run_condition(
            config,
            condition,
            seed,
            config.episodes_per_condition,
            env,
            agent,
          )
          for seed in config.seeds
        ]
        main_stats.append(_merge_stats(per_seed))

    if config.run_velocity_diagnostic:
      for condition in VELOCITY_CONDITIONS:
        logger.info(f"Running velocity diagnostic condition: {condition.label}")
        per_seed = [
          _run_condition(
            config,
            condition,
            seed,
            config.velocity_episodes_per_condition,
            env,
            agent,
          )
          for seed in config.seeds
        ]
        velocity_stats.append(_merge_stats(per_seed))
  finally:
    env.close()

  plot_paths = _save_plots(velocity_stats, output_dir) if config.save_plots else []
  report = _build_report(config, checkpoint_path, main_stats, velocity_stats, plot_paths)
  report_path = output_dir / config.report_name
  report_path.write_text(report, encoding="utf-8")
  print(report)
  logger.success(f"Saved report: {report_path}")


if __name__ == "__main__":
  main()
