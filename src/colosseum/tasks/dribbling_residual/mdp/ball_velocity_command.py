"""Command term: target-driven ball velocity in world frame (trimmed).

At episode start the term samples a world-frame target from the ball position
(radius from ``target_distance_range``, heading relative to the robot yaw) and
resamples a new target whenever the ball reaches it. Each step it recomputes the
desired ball velocity toward the current target:

    dir   = normalize(target - ball_pos)
    speed = clip(speed_gain * distance, speed_range), zeroed within reach
    cmd   = speed * dir   (world-frame [vx, vy, 0])

This is a trimmed variant of the dribbling task's BallVelocityCommand: it keeps
the persistent target and speed ramp but drops timer/drift resampling and the
obstacle/adversary coupling. Losing the ball is handled by the ball-lost
termination, not by resampling the target.
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
  """World-frame ball velocity command induced by a persistent target, resampled on reach."""

  cfg: BallVelocityCommandCfg

  def __init__(self, cfg: BallVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.velocity_command = torch.zeros((env.num_envs, 3), device=env.device)
    self.target_position = torch.zeros((env.num_envs, 2), device=env.device)
    self.target_reached_mask = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    self.metrics["target_distance"] = torch.zeros(env.num_envs, device=env.device)
    self.metrics["cmd_ball_vel_error"] = torch.zeros(env.num_envs, device=env.device)
    # Per-episode count of foot->ball contact onsets ("hits"). Auto-zeroed and
    # logged as Metrics/ball_vel/foot_ball_hits at reset by the command manager.
    self.metrics["foot_ball_hits"] = torch.zeros(env.num_envs, device=env.device)
    self._prev_foot_ball_contact = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    self.just_resampled = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    # Commands must be valid from the first step: a stale zero target at the
    # global origin is tens of metres away in a tiled multi-env world.
    self._resample(torch.arange(self.num_envs, device=self.device))

  @property
  def command(self) -> torch.Tensor:
    return self.velocity_command

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

    self.just_resampled[env_ids] = True

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

    ball_pos = self._env.scene[self.cfg.ball_entity].data.root_link_pos_w[env_ids, :2]
    self.target_position[env_ids, 0] = (
      ball_pos[:, 0] + torch.cos(target_heading) * target_distances
    )
    self.target_position[env_ids, 1] = (
      ball_pos[:, 1] + torch.sin(target_heading) * target_distances
    )
    self._recompute_velocity_command(env_ids)

  def _update_command(self) -> None:
    all_env_ids = torch.arange(self.num_envs, device=self.device)
    self.target_reached_mask[:] = False
    self._recompute_velocity_command(all_env_ids)

    ball_pos = self._env.scene[self.cfg.ball_entity].data.root_link_pos_w[:, :2]
    target_distance = (self.target_position - ball_pos).norm(dim=-1)
    reached_env_ids = torch.where(target_distance <= self.cfg.target_reached_threshold)[
      0
    ]
    if len(reached_env_ids) > 0:
      self.target_reached_mask[reached_env_ids] = True
      self._resample(reached_env_ids)

  def _foot_ball_contact(self) -> torch.Tensor:
    """Per-env bool: any foot geom in contact with the ball this step."""
    try:
      sensor = self._env.scene[self.cfg.foot_ball_contact_sensor]
    except KeyError:
      return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    found = sensor.data.found
    if found is None:
      return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    return (found.flatten(start_dim=1) > 0).any(dim=-1)

  def _update_metrics(self) -> None:
    self.just_resampled[:] = False
    ball_pos = self._env.scene[self.cfg.ball_entity].data.root_link_pos_w[:, :2]
    ball_vel = self._env.scene[self.cfg.ball_entity].data.root_link_lin_vel_w[:, :2]
    self.metrics["target_distance"] = (self.target_position - ball_pos).norm(dim=-1)
    self.metrics["cmd_ball_vel_error"] = (self.velocity_command[:, :2] - ball_vel).norm(
      dim=-1
    )

    # Count rising edges (no-contact -> contact) so the metric reads as the
    # number of distinct foot-ball touches per episode, not the dwell time.
    contact = self._foot_ball_contact()
    self.metrics["foot_ball_hits"] += (contact & ~self._prev_foot_ball_contact).float()
    self._prev_foot_ball_contact = contact

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    ball = self._env.scene[self.cfg.ball_entity]
    ball_pos = ball.data.root_link_pos_w[batch]

    # Desired ball velocity (green).
    vel_2d = self.velocity_command[batch, :2]
    vel_3d = torch.cat([vel_2d, torch.zeros(1, device=vel_2d.device)])
    visualizer.add_arrow(
      start=ball_pos.cpu().numpy(),
      end=(ball_pos + vel_3d * 2.0).cpu().numpy(),
      color=(0.2, 0.8, 0.2, 0.9),
      label=f"ball_cmd |v|={vel_2d.norm():.2f}",
    )

    # Actual ball velocity (orange) — overlaid so tracking quality is visible.
    actual_2d = ball.data.root_link_lin_vel_w[batch, :2]
    actual_3d = torch.cat([actual_2d, torch.zeros(1, device=actual_2d.device)])
    visualizer.add_arrow(
      start=ball_pos.cpu().numpy(),
      end=(ball_pos + actual_3d * 2.0).cpu().numpy(),
      color=(0.95, 0.6, 0.1, 0.9),
      label=f"ball_vel |v|={actual_2d.norm():.2f}",
    )

    target_pos = self.target_position[batch]
    target_pos_3d = torch.cat(
      [target_pos, torch.tensor([0.05], device=target_pos.device)]
    )
    visualizer.add_sphere(
      center=target_pos_3d.cpu().numpy(),
      radius=0.5,
      color=(0.2, 0.4, 1.0, 0.65),
      label="ball_target",
    )


@dataclass(kw_only=True)
class BallVelocityCommandCfg(CommandTermCfg):
  """Configuration for BallVelocityCommand."""

  class_type: type[CommandTerm] = BallVelocityCommand

  debug_vis: bool = True

  # Never time-resample: the target is fixed for the whole episode (set the
  # interval past any episode length, like BallTwistCommand).
  resampling_time_range: tuple[float, float] = (1e9, 1e9)

  robot_entity: str = "robot"
  ball_entity: str = "ball"
  # Contact sensor (registered in the scene) used to count foot->ball touches.
  foot_ball_contact_sensor: str = "foot_ball_contact"

  # Desired ball speed, recomputed each step and clipped to this range.
  speed_range: tuple[float, float] = (0.1, 1.0)

  # Sampled target radius (m) drawn from the ball position at episode start.
  # Far targets keep the ball from arriving, so the command stays a steady
  # dribble-direction setpoint for the whole episode.
  target_distance_range: tuple[float, float] = (2.0, 6.0)

  # Gain mapping target distance -> desired speed before clipping.
  speed_gain: float = 1.0

  # Resample immediately when the ball is this close to the target.
  target_reached_threshold: float = 0.5

  # Half-width of the heading range around the robot forward direction.
  heading_range: float = math.pi / 8

  def build(self, env: ManagerBasedRlEnv) -> BallVelocityCommand:
    return BallVelocityCommand(self, env)
