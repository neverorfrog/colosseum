"""Command term: target-driven ball velocity in world frame.

At resampling time the term samples a persistent world-frame target from the
current robot position.
At every step it recomputes the desired ball velocity from the current ball
position toward that target:

    dir = normalize(target - ball_pos)
    speed = clip(speed_gain * distance_to_target, min_speed, max_speed)
    cmd = speed * dir

The command interface remains a world-frame velocity vector [vx, vy, 0.0], so
existing rewards and body-frame observations stay consistent. The difference is
that the command now encodes progress toward a persistent goal instead of a
locally sampled free velocity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class BallVelocityCommand(CommandTerm):
  """World-frame ball velocity command induced by a persistent target."""

  cfg: BallVelocityCommandCfg

  def __init__(self, cfg: BallVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.velocity_command = torch.zeros((env.num_envs, 3), device=env.device)
    self.target_position = torch.zeros((env.num_envs, 2), device=env.device)
    self.metrics["ball_distance"] = torch.zeros(env.num_envs, device=env.device)
    self.metrics["cmd_ball_vel_error"] = torch.zeros(env.num_envs, device=env.device)
    self.metrics["target_distance"] = torch.zeros(env.num_envs, device=env.device)

    # Commands must be valid from the very first reset/step. Otherwise some
    # envs would keep the zero target at the global origin, which is disastrous
    # in a tiled multi-env world because target_distance becomes tens of meters.
    all_env_ids = torch.arange(self.num_envs, device=self.device)
    self._resample(all_env_ids)

  # ------------------------------------------------------------------
  # CommandTerm interface
  # ------------------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    return self.velocity_command

  @property
  def world_vel_cmd(self) -> torch.Tensor:
    """World-frame XY ball velocity target. Shape (N, 2)."""
    return self.velocity_command[:, :2]

  def _recompute_velocity_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return

    ball_pos = self._env.scene[self.cfg.ball_entity].data.root_link_pos_w[env_ids, :2]
    target_delta = self.target_position[env_ids] - ball_pos
    distance = target_delta.norm(dim=-1, keepdim=True)
    direction = torch.where(
      distance > 1e-6,
      target_delta / distance.clamp(min=1e-6),
      torch.zeros_like(target_delta),
    )

    speed = (self.cfg.speed_gain * distance).clamp(
      min=self.cfg.speed_range[0],
      max=self.cfg.speed_range[1],
    )
    speed = torch.where(
      distance > self.cfg.target_reached_threshold,
      speed,
      torch.zeros_like(speed),
    )

    self.velocity_command[env_ids, 0:2] = direction * speed
    self.velocity_command[env_ids, 2] = 0.0

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    device = self._env.device

    lo, hi = -self.cfg.heading_range, self.cfg.heading_range
    heading_offsets = torch.rand(n, device=device) * (hi - lo) + lo

    lo, hi = self.cfg.target_distance_range
    target_distances = torch.rand(n, device=device) * (hi - lo) + lo

    robot_quat = self._env.scene[self.cfg.robot_entity].data.root_link_quat_w[env_ids]
    robot_yaw = torch.atan2(
      2.0 * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2]),
      1.0 - 2.0 * (robot_quat[:, 2] ** 2 + robot_quat[:, 3] ** 2),
    )
    target_heading = robot_yaw + heading_offsets

    robot_pos = self._env.scene[self.cfg.robot_entity].data.root_link_pos_w[env_ids, :2]
    self.target_position[env_ids, 0] = (
      robot_pos[:, 0] + torch.cos(target_heading) * target_distances
    )
    self.target_position[env_ids, 1] = (
      robot_pos[:, 1] + torch.sin(target_heading) * target_distances
    )
    self._recompute_velocity_command(env_ids)
    self._resample_obstacles(env_ids)

  def _resample_obstacles(self, env_ids: torch.Tensor) -> None:
    """Force the adversary command to resample obstacles for the given envs
    so obstacle placement stays consistent with the freshly sampled target."""
    if not self.cfg.resample_obstacles_on_target_reset:
      return
    try:
      term = self._env.command_manager.get_term(self.cfg.obstacle_command_name)
    except Exception:
      return
    resample_fn = getattr(term, "resample_for_env_ids", None)
    if resample_fn is None:
      return
    resample_fn(env_ids)

  def _update_command(self) -> None:
    all_env_ids = torch.arange(self.num_envs, device=self.device)
    self._recompute_velocity_command(all_env_ids)

    ball_pos = self._env.scene[self.cfg.ball_entity].data.root_link_pos_w[:, :2]
    target_distance = (self.target_position - ball_pos).norm(dim=-1)
    invalid_env_ids = torch.where(
      (~torch.isfinite(target_distance))
      | (target_distance > self.cfg.target_distance_range[1] * 3.0)
    )[0]
    if len(invalid_env_ids) > 0:
      self._resample(invalid_env_ids)
      target_distance = (self.target_position - ball_pos).norm(dim=-1)

    reached_env_ids = torch.where(target_distance <= self.cfg.target_reached_threshold)[0]
    if len(reached_env_ids) > 0:
      self._resample(reached_env_ids)
      target_distance = (self.target_position - ball_pos).norm(dim=-1)

    robot_pos = self._env.scene[self.cfg.robot_entity].data.root_link_pos_w[:, :2]
    robot_target_distance = (self.target_position - robot_pos).norm(dim=-1)
    robot_reached_env_ids = torch.where(
      robot_target_distance <= self.cfg.robot_target_reached_threshold
    )[0]
    if len(robot_reached_env_ids) > 0:
      self._resample(robot_reached_env_ids)

  def _update_metrics(self) -> None:
    robot_pos = self._env.scene[self.cfg.robot_entity].data.root_link_pos_w[:, :2]
    ball_pos = self._env.scene[self.cfg.ball_entity].data.root_link_pos_w[:, :2]
    ball_vel = self._env.scene[self.cfg.ball_entity].data.root_link_lin_vel_w[:, :2]

    self.metrics["ball_distance"] = (ball_pos - robot_pos).norm(dim=-1)
    self.metrics["cmd_ball_vel_error"] = (self.velocity_command[:, :2] - ball_vel).norm(
      dim=-1
    )
    self.metrics["target_distance"] = (self.target_position - ball_pos).norm(dim=-1)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    ball_pos = self._env.scene[self.cfg.ball_entity].data.root_link_pos_w[batch]
    vel_2d = self.velocity_command[batch, :2]
    vel_3d = torch.cat([vel_2d, torch.zeros(1, device=vel_2d.device)])

    visualizer.add_arrow(
      start=ball_pos.cpu().numpy(),
      end=(ball_pos + vel_3d * 2.0).cpu().numpy(),
      color=(0.2, 0.8, 0.2, 0.9),
      label=f"ball_cmd |v|={vel_2d.norm():.2f}",
    )

    real_vel_2d = self._env.scene[self.cfg.ball_entity].data.root_link_lin_vel_w[batch, :2]
    real_vel_3d = torch.cat([real_vel_2d, torch.zeros(1, device=real_vel_2d.device)])
    visualizer.add_arrow(
      start=ball_pos.cpu().numpy(),
      end=(ball_pos + real_vel_3d * 2.0).cpu().numpy(),
      color=(0.8, 0.2, 0.2, 0.9),
      label=f"ball_vel |v|={real_vel_2d.norm():.2f}",
    )

    target_pos = self.target_position[batch]
    target_pos_3d = torch.cat(
      [target_pos, torch.tensor([0.05], device=target_pos.device)]
    )
    visualizer.add_sphere(
      center=target_pos_3d.cpu().numpy(),
      radius=0.05,
      color=(0.2, 0.4, 1.0, 0.65),
      label=f"ball_target d={(target_pos - ball_pos[:2]).norm():.2f}",
    )
    visualizer.add_arrow(
      start=ball_pos.cpu().numpy(),
      end=target_pos_3d.cpu().numpy(),
      color=(0.2, 0.4, 1.0, 0.55),
      label="target_dir",
    )


@dataclass(kw_only=True)
class BallVelocityCommandCfg(CommandTermCfg):
  """Configuration for BallVelocityCommand."""

  class_type: type[CommandTerm] = BallVelocityCommand

  # Resample command every 5–10 seconds (like UniformVelocityCommand).
  resampling_time_range: tuple[float, float] = (5.0, 10.0)

  robot_entity: str = "robot"
  ball_entity: str = "ball"

  # Command speed is recomputed every step and clipped to this range.
  speed_range: tuple[float, float] = (0.1, 0.1)

  # Target distance sampled uniformly from the robot position at each
  # reset/resample.
  target_distance_range: tuple[float, float] = (1.5, 4.0)

  # Gain mapping target distance -> desired speed before clipping.
  speed_gain: float = 1.0

  # Resample immediately when the ball is this close to the target.
  target_reached_threshold: float = 0.25

  # Also resample when the robot base enters this radius around the target —
  # keeps the robot from parking next to a target it cannot commit to.
  robot_target_reached_threshold: float = 0.5

  # Half-width of the heading range around the robot forward direction.
  heading_range: float = math.pi / 8

  # When the target resamples, also force the adversary command term to
  # resample obstacles so obstacles stay on the new ball→target corridor.
  resample_obstacles_on_target_reset: bool = True
  obstacle_command_name: str = "adversary"

  def build(self, env: ManagerBasedRlEnv) -> BallVelocityCommand:
    return BallVelocityCommand(self, env)
